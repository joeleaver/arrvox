//! Skeleton hierarchy and bone definitions.
//!
//! A skeleton is an ordered list of bones with parent-child relationships.
//! Each bone has a bind-pose transform and its precomputed inverse, used
//! during animation evaluation to produce final bone matrices.

use glam::{Mat4, Quat, Vec3};
use arvx_core::scene_node::Transform;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::clip::AnimationClip;

/// Errors that can occur when constructing a skeleton.
#[derive(Debug, Error)]
pub enum SkeletonError {
    /// A bone references a parent index that is out of range.
    #[error("bone {bone_index} references invalid parent index {parent_index} (skeleton has {bone_count} bones)")]
    InvalidParentIndex {
        /// The bone with the bad parent reference.
        bone_index: usize,
        /// The invalid parent index.
        parent_index: i32,
        /// Total number of bones in the skeleton.
        bone_count: usize,
    },

    /// A bone references itself as its parent.
    #[error("bone {0} references itself as parent")]
    SelfReference(usize),

    /// The hierarchy contains a cycle.
    #[error("cycle detected in hierarchy involving bone {0}")]
    CycleDetected(usize),

    /// The bone and hierarchy vectors have different lengths.
    #[error("bones length ({bones}) does not match hierarchy length ({hierarchy})")]
    LengthMismatch {
        /// Number of bones.
        bones: usize,
        /// Number of hierarchy entries.
        hierarchy: usize,
    },
}

/// A single bone in the skeleton hierarchy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Bone {
    /// Bone name (e.g. "UpperArm_L").
    pub name: String,
    /// Bind-pose transform (local space relative to parent).
    pub bind_transform: Mat4,
    /// Inverse bind-pose transform (world to bone local).
    pub inverse_bind: Mat4,
}

/// Skeleton: bone hierarchy with ordered bone list.
///
/// Bones are stored in a flat array. Parent-child relationships are encoded
/// in the `hierarchy` vector where `hierarchy[i]` gives the parent index
/// of bone `i`, or -1 if bone `i` is a root.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Skeleton {
    /// Bones ordered by index.
    pub bones: Vec<Bone>,
    /// Parent index per bone. -1 means root (no parent).
    pub hierarchy: Vec<i32>,
}

impl Skeleton {
    /// Create a new skeleton, validating the hierarchy.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - `bones` and `hierarchy` have different lengths
    /// - Any parent index is out of range
    /// - Any bone references itself as parent
    /// - The hierarchy contains cycles
    pub fn new(bones: Vec<Bone>, hierarchy: Vec<i32>) -> Result<Self, SkeletonError> {
        if bones.len() != hierarchy.len() {
            return Err(SkeletonError::LengthMismatch {
                bones: bones.len(),
                hierarchy: hierarchy.len(),
            });
        }

        let bone_count = bones.len();

        for (i, &parent) in hierarchy.iter().enumerate() {
            if parent == -1 {
                continue;
            }
            if parent < 0 || parent as usize >= bone_count {
                return Err(SkeletonError::InvalidParentIndex {
                    bone_index: i,
                    parent_index: parent,
                    bone_count,
                });
            }
            if parent as usize == i {
                return Err(SkeletonError::SelfReference(i));
            }
        }

        // Cycle detection: walk parent chain from each bone, should terminate
        // within bone_count steps.
        for i in 0..bone_count {
            let mut current = hierarchy[i];
            let mut steps = 0;
            while current != -1 {
                steps += 1;
                if steps > bone_count {
                    return Err(SkeletonError::CycleDetected(i));
                }
                current = hierarchy[current as usize];
            }
        }

        Ok(Self { bones, hierarchy })
    }

    /// Returns the number of bones in the skeleton.
    pub fn bone_count(&self) -> usize {
        self.bones.len()
    }

    /// Returns indices of all root bones (bones with parent == -1).
    pub fn root_bones(&self) -> Vec<usize> {
        self.hierarchy
            .iter()
            .enumerate()
            .filter(|(_, parent)| **parent == -1)
            .map(|(i, _)| i)
            .collect()
    }

    /// Evaluate bone matrices for the given clip at the specified time.
    ///
    /// Returns world-space bone matrices suitable for skinning:
    /// `bone_matrix[i] = world[i] * inverse_bind[i]`
    ///
    /// For bones not present in the clip, the bind-pose transform is used
    /// as the local transform.
    pub fn evaluate(&self, clip: &AnimationClip, time: f32) -> Vec<Mat4> {
        let bone_count = self.bones.len();

        // Build a lookup: bone_index -> channel index in clip.
        // Small skeletons make a linear scan fast enough.
        let mut local_transforms = Vec::with_capacity(bone_count);

        for (i, bone) in self.bones.iter().enumerate() {
            let local = if let Some(channel) = clip.channel_for_bone(i as u32) {
                let (pos, rot, scl) = channel.sample(time);
                Mat4::from_scale_rotation_translation(scl, rot, pos)
            } else {
                bone.bind_transform
            };
            local_transforms.push(local);
        }

        // Compute world-space transforms by walking hierarchy.
        let mut world_transforms = vec![Mat4::IDENTITY; bone_count];

        for i in 0..bone_count {
            let parent = self.hierarchy[i];
            if parent == -1 {
                world_transforms[i] = local_transforms[i];
            } else {
                world_transforms[i] = world_transforms[parent as usize] * local_transforms[i];
            }
        }

        // Final skinning matrices: world * inverse_bind.
        let mut bone_matrices = Vec::with_capacity(bone_count);
        for i in 0..bone_count {
            bone_matrices.push(world_transforms[i] * self.bones[i].inverse_bind);
        }

        bone_matrices
    }

    /// Evaluate per-bone local transforms for the given clip at the specified time.
    ///
    /// Returns a `Transform` per bone — these are the local-space transforms
    /// that should be written directly to SceneNode `local_transform` fields.
    ///
    /// For bones not present in the clip, the bind-pose is decomposed into
    /// position/rotation/scale and returned as-is.
    pub fn evaluate_local(&self, clip: &AnimationClip, time: f32) -> Vec<Transform> {
        let bone_count = self.bones.len();
        let mut transforms = Vec::with_capacity(bone_count);

        for (i, bone) in self.bones.iter().enumerate() {
            let t = if let Some(channel) = clip.channel_for_bone(i as u32) {
                let (pos, rot, scl) = channel.sample(time);
                Transform::new(pos, rot, scl)
            } else {
                // Decompose bind-pose matrix to Transform.
                let (scale, rotation, translation) =
                    bone.bind_transform.to_scale_rotation_translation();
                Transform::new(translation, rotation, scale)
            };
            transforms.push(t);
        }

        transforms
    }

    /// Decompose a Mat4 into (scale, rotation, translation) components.
    ///
    /// Utility for extracting SRT from a bind-pose matrix. Assumes uniform
    /// scale (as required by the SDF engine).
    pub fn decompose_transform(transform: &Mat4) -> (Vec3, Quat, Vec3) {
        let (scale, rotation, translation) = transform.to_scale_rotation_translation();
        (scale, rotation, translation)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::FRAC_PI_2;

    /// Helper: create a simple bone with translation-only bind pose.
    fn bone(name: &str, translation: Vec3) -> Bone {
        let bind = Mat4::from_translation(translation);
        Bone {
            name: name.to_string(),
            bind_transform: bind,
            inverse_bind: bind.inverse(),
        }
    }

    /// Helper: create a bone with identity bind pose.
    fn identity_bone(name: &str) -> Bone {
        Bone {
            name: name.to_string(),
            bind_transform: Mat4::IDENTITY,
            inverse_bind: Mat4::IDENTITY,
        }
    }

    #[test]
    fn test_new_valid_hierarchy() {
        let bones = vec![
            bone("root", Vec3::ZERO),
            bone("child", Vec3::new(0.0, 1.0, 0.0)),
            bone("grandchild", Vec3::new(0.0, 0.5, 0.0)),
        ];
        let hierarchy = vec![-1, 0, 1];
        let skel = Skeleton::new(bones, hierarchy).unwrap();
        assert_eq!(skel.bone_count(), 3);
    }

    #[test]
    fn test_new_rejects_length_mismatch() {
        let bones = vec![identity_bone("a"), identity_bone("b")];
        let hierarchy = vec![-1];
        let result = Skeleton::new(bones, hierarchy);
        assert!(matches!(result, Err(SkeletonError::LengthMismatch { .. })));
    }

    #[test]
    fn test_new_rejects_invalid_parent_index() {
        let bones = vec![identity_bone("a"), identity_bone("b")];
        let hierarchy = vec![-1, 5]; // 5 is out of range
        let result = Skeleton::new(bones, hierarchy);
        assert!(matches!(
            result,
            Err(SkeletonError::InvalidParentIndex { .. })
        ));
    }

    #[test]
    fn test_new_rejects_self_reference() {
        let bones = vec![identity_bone("a"), identity_bone("b")];
        let hierarchy = vec![-1, 1]; // bone 1 references itself
        let result = Skeleton::new(bones, hierarchy);
        assert!(matches!(result, Err(SkeletonError::SelfReference(1))));
    }

    #[test]
    fn test_new_rejects_cycle() {
        // bone 0: parent=-1, bone 1: parent=2, bone 2: parent=1
        let bones = vec![
            identity_bone("a"),
            identity_bone("b"),
            identity_bone("c"),
        ];
        let hierarchy = vec![-1, 2, 1]; // 1->2->1 cycle
        let result = Skeleton::new(bones, hierarchy);
        assert!(matches!(result, Err(SkeletonError::CycleDetected(_))));
    }

    #[test]
    fn test_bone_count() {
        let skel = Skeleton::new(
            vec![identity_bone("a"), identity_bone("b")],
            vec![-1, 0],
        )
        .unwrap();
        assert_eq!(skel.bone_count(), 2);
    }

    #[test]
    fn test_root_bones_single_root() {
        let skel = Skeleton::new(
            vec![
                identity_bone("root"),
                identity_bone("child"),
                identity_bone("grandchild"),
            ],
            vec![-1, 0, 1],
        )
        .unwrap();
        assert_eq!(skel.root_bones(), vec![0]);
    }

    #[test]
    fn test_root_bones_multiple_roots() {
        let skel = Skeleton::new(
            vec![
                identity_bone("root_a"),
                identity_bone("root_b"),
                identity_bone("child_of_a"),
            ],
            vec![-1, -1, 0],
        )
        .unwrap();
        assert_eq!(skel.root_bones(), vec![0, 1]);
    }

    #[test]
    fn test_evaluate_bind_pose_produces_identity() {
        // When the clip provides transforms matching bind pose,
        // bone_matrix = world * inverse_bind should be ~identity.
        use crate::clip::{AnimationClip, BoneChannel, Keyframe};

        let bones = vec![
            bone("root", Vec3::ZERO),
            bone("child", Vec3::new(0.0, 1.0, 0.0)),
        ];
        let hierarchy = vec![-1, 0];
        let skel = Skeleton::new(bones.clone(), hierarchy).unwrap();

        // Clip with keyframes matching bind pose exactly.
        let clip = AnimationClip::new(
            "bind".to_string(),
            1.0,
            vec![
                BoneChannel {
                    bone_index: 0,
                    keyframes: vec![Keyframe {
                        time: 0.0,
                        position: Vec3::ZERO,
                        rotation: Quat::IDENTITY,
                        scale: Vec3::ONE,
                    }],
                },
                BoneChannel {
                    bone_index: 1,
                    keyframes: vec![Keyframe {
                        time: 0.0,
                        position: Vec3::new(0.0, 1.0, 0.0),
                        rotation: Quat::IDENTITY,
                        scale: Vec3::ONE,
                    }],
                },
            ],
        );

        let matrices = skel.evaluate(&clip, 0.0);
        for (i, m) in matrices.iter().enumerate() {
            let diff = (*m - Mat4::IDENTITY).abs().to_cols_array();
            let max_diff = diff.iter().cloned().fold(0.0f32, f32::max);
            assert!(
                max_diff < 1e-5,
                "bone {i} matrix differs from identity by {max_diff}"
            );
        }
    }

    #[test]
    fn test_evaluate_child_accumulates_parent_transform() {
        use crate::clip::{AnimationClip, BoneChannel, Keyframe};

        // Root at origin, child offset by (0, 1, 0) in bind pose.
        // Animate root to rotate 90deg around Z. Child should move accordingly.
        let bones = vec![
            bone("root", Vec3::ZERO),
            bone("child", Vec3::new(0.0, 1.0, 0.0)),
        ];
        let hierarchy = vec![-1, 0];
        let skel = Skeleton::new(bones, hierarchy).unwrap();

        let rotation = Quat::from_rotation_z(FRAC_PI_2);
        let clip = AnimationClip::new(
            "rotate".to_string(),
            1.0,
            vec![BoneChannel {
                bone_index: 0,
                keyframes: vec![Keyframe {
                    time: 0.0,
                    position: Vec3::ZERO,
                    rotation,
                    scale: Vec3::ONE,
                }],
            }],
            // bone 1 not in clip -> uses bind_transform
        );

        let matrices = skel.evaluate(&clip, 0.0);

        // Root bone matrix: world = rotation, bone_mat = rotation * inv_bind(identity) = rotation
        let root_mat = matrices[0];
        // The root had bind at origin, so inverse_bind = identity.
        // world = rotation_matrix, bone_mat = rotation_matrix * IDENTITY = rotation_matrix
        let expected_root = Mat4::from_quat(rotation);
        let diff = (root_mat - expected_root).abs().to_cols_array();
        let max_diff = diff.iter().cloned().fold(0.0f32, f32::max);
        assert!(max_diff < 1e-5, "root bone matrix error: {max_diff}");

        // Child uses bind_transform as local (not in clip).
        // world[1] = world[0] * bind_transform[1] = rotation * translation(0,1,0)
        // bone_mat[1] = world[1] * inverse_bind[1]
        // Since bind_transform[1] = translation(0,1,0) and inverse_bind[1] = translation(0,-1,0):
        // world[1] = rotation * translation(0,1,0)
        // bone_mat[1] = rotation * translation(0,1,0) * translation(0,-1,0) = rotation
        let child_mat = matrices[1];
        let diff = (child_mat - expected_root).abs().to_cols_array();
        let max_diff = diff.iter().cloned().fold(0.0f32, f32::max);
        assert!(max_diff < 1e-5, "child bone matrix error: {max_diff}");
    }

    #[test]
    fn test_evaluate_bone_not_in_clip_uses_bind() {
        use crate::clip::AnimationClip;

        let bones = vec![
            bone("root", Vec3::ZERO),
            bone("child", Vec3::new(2.0, 0.0, 0.0)),
        ];
        let hierarchy = vec![-1, 0];
        let skel = Skeleton::new(bones, hierarchy).unwrap();

        // Empty clip: no channels at all.
        let clip = AnimationClip::new("empty".to_string(), 1.0, vec![]);

        let matrices = skel.evaluate(&clip, 0.0);

        // Both bones use bind_transform as local.
        // Root: world = bind(identity at origin), bone_mat = identity * identity = identity
        let diff = (matrices[0] - Mat4::IDENTITY).abs().to_cols_array();
        assert!(diff.iter().cloned().fold(0.0f32, f32::max) < 1e-5);

        // Child: world = identity * translation(2,0,0), bone_mat = translation(2,0,0) * translation(-2,0,0) = identity
        let diff = (matrices[1] - Mat4::IDENTITY).abs().to_cols_array();
        assert!(diff.iter().cloned().fold(0.0f32, f32::max) < 1e-5);
    }

    #[test]
    fn test_evaluate_local_returns_correct_transforms() {
        use crate::clip::{AnimationClip, BoneChannel, Keyframe};

        let bones = vec![
            bone("root", Vec3::ZERO),
            bone("child", Vec3::new(0.0, 1.0, 0.0)),
        ];
        let hierarchy = vec![-1, 0];
        let skel = Skeleton::new(bones, hierarchy).unwrap();

        let clip = AnimationClip::new(
            "test".to_string(),
            1.0,
            vec![
                BoneChannel {
                    bone_index: 0,
                    keyframes: vec![Keyframe {
                        time: 0.0,
                        position: Vec3::new(1.0, 2.0, 3.0),
                        rotation: Quat::IDENTITY,
                        scale: Vec3::splat(2.0),
                    }],
                },
            ],
        );

        let locals = skel.evaluate_local(&clip, 0.0);
        assert_eq!(locals.len(), 2);

        // Bone 0: from clip keyframe.
        assert!((locals[0].position - Vec3::new(1.0, 2.0, 3.0)).length() < 1e-5);
        assert_eq!(locals[0].rotation, Quat::IDENTITY);
        assert!((locals[0].scale - Vec3::splat(2.0)).length() < 1e-5);

        // Bone 1: from bind pose (translation 0,1,0, identity rotation, scale 1).
        assert!((locals[1].position - Vec3::new(0.0, 1.0, 0.0)).length() < 1e-5);
        assert!((locals[1].scale - Vec3::ONE).length() < 1e-5);
    }

    #[test]
    fn test_evaluate_local_empty_clip_uses_bind() {
        use crate::clip::AnimationClip;

        let bones = vec![
            bone("root", Vec3::ZERO),
            bone("child", Vec3::new(3.0, 0.0, 0.0)),
        ];
        let hierarchy = vec![-1, 0];
        let skel = Skeleton::new(bones, hierarchy).unwrap();

        let clip = AnimationClip::new("empty".to_string(), 1.0, vec![]);
        let locals = skel.evaluate_local(&clip, 0.0);

        // Root: identity bind → position=0, scale=1.
        assert!((locals[0].position).length() < 1e-5);
        assert!((locals[0].scale - Vec3::ONE).length() < 1e-5);

        // Child: bind_transform = translation(3,0,0) → position=(3,0,0).
        assert!((locals[1].position - Vec3::new(3.0, 0.0, 0.0)).length() < 1e-5);
    }

    // ── serde roundtrip ────────────────────────────────────────────────────

    #[test]
    fn test_bone_serde_roundtrip() {
        let b = bone("UpperArm_L", Vec3::new(1.0, 2.0, 3.0));
        let ron_str = ron::to_string(&b).expect("serialize Bone");
        let b2: Bone = ron::from_str(&ron_str).expect("deserialize Bone");
        assert_eq!(b.name, b2.name);
        let diff = (b.bind_transform - b2.bind_transform).abs().to_cols_array();
        assert!(diff.iter().all(|&d| d < 1e-6));
        let diff = (b.inverse_bind - b2.inverse_bind).abs().to_cols_array();
        assert!(diff.iter().all(|&d| d < 1e-6));
    }

    #[test]
    fn test_skeleton_serde_roundtrip() {
        let bones = vec![
            bone("root", Vec3::ZERO),
            bone("child", Vec3::new(0.0, 1.0, 0.0)),
            bone("grandchild", Vec3::new(0.0, 0.5, 0.0)),
        ];
        let hierarchy = vec![-1, 0, 1];
        let skel = Skeleton::new(bones, hierarchy).unwrap();

        let ron_str = ron::to_string(&skel).expect("serialize Skeleton");
        let skel2: Skeleton = ron::from_str(&ron_str).expect("deserialize Skeleton");

        assert_eq!(skel.bone_count(), skel2.bone_count());
        assert_eq!(skel.hierarchy, skel2.hierarchy);
        for (a, b) in skel.bones.iter().zip(skel2.bones.iter()) {
            assert_eq!(a.name, b.name);
        }
    }

    #[test]
    fn test_decompose_transform() {
        let translation = Vec3::new(1.0, 2.0, 3.0);
        let rotation = Quat::from_rotation_y(1.0);
        let scale = Vec3::splat(2.0);
        let mat = Mat4::from_scale_rotation_translation(scale, rotation, translation);

        let (s, r, t) = Skeleton::decompose_transform(&mat);
        assert!((t - translation).length() < 1e-5);
        assert!((r - rotation).length() < 1e-5 || (r + rotation).length() < 1e-5); // quat sign ambiguity
        assert!((s - scale).length() < 1e-5);
    }
}
