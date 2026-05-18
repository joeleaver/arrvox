//! Animated character assembly — v2 SceneNode tree model.
//!
//! An [`AnimatedCharacter`] is a character whose bones are SceneNode children.
//! Animation = updating bone local transforms from keyframe interpolation.
//! Joint blending happens naturally via SmoothUnion in the ray marcher.

use glam::{Mat4, Quat, Vec3};
use arvx_core::aabb::Aabb;
use arvx_core::scene_node::{BlendMode, SceneNode, SdfPrimitive, SdfSource, Transform};

use crate::clip::AnimationClip;
use crate::player::AnimationPlayer;
use crate::skeleton::Skeleton;

/// Per-bone visual definition — what SDF shape represents each bone.
#[derive(Debug, Clone)]
pub struct BoneVisual {
    /// Name of the bone.
    pub bone_name: String,
    /// SDF source for this bone's geometry.
    pub sdf_source: SdfSource,
    /// Blend mode for combining with sibling bones.
    pub blend_mode: BlendMode,
}

/// An animated character built from a skeleton and bone visuals.
///
/// In v2, a character is a SceneNode tree where each bone is a child node.
/// Animation playback writes per-bone local transforms to the SceneNode
/// hierarchy. Joint blending is automatic via SmoothUnion in the ray marcher.
#[derive(Debug, Clone)]
pub struct AnimatedCharacter {
    /// The skeleton (bone hierarchy and bind poses).
    pub skeleton: Skeleton,
    /// Visual definition per bone.
    pub bone_visuals: Vec<BoneVisual>,
    /// Animation player.
    pub player: AnimationPlayer,
    /// Smooth-min blend radius for joint blending.
    pub joint_blend_radius: f32,
}

impl AnimatedCharacter {
    /// Create a new animated character.
    ///
    /// `bone_visuals` should have one entry per bone that has geometry.
    /// Bones without a matching visual become pure transform nodes (SdfSource::None).
    pub fn new(
        skeleton: Skeleton,
        bone_visuals: Vec<BoneVisual>,
        clip: AnimationClip,
        joint_blend_radius: f32,
    ) -> Self {
        Self {
            skeleton,
            bone_visuals,
            player: AnimationPlayer::new(clip),
            joint_blend_radius,
        }
    }

    /// Build a SceneNode tree matching the skeleton hierarchy.
    ///
    /// Each bone becomes a SceneNode. The node's local_transform is set from
    /// the bind pose. Bones with matching entries in `bone_visuals` get their
    /// SDF source; others are pure transform nodes.
    pub fn build_scene_node(&self) -> SceneNode {
        let bone_count = self.skeleton.bone_count();

        // Create a flat list of nodes, one per bone.
        let mut nodes: Vec<SceneNode> = Vec::with_capacity(bone_count);
        for (i, bone) in self.skeleton.bones.iter().enumerate() {
            let (scale, rotation, translation) =
                bone.bind_transform.to_scale_rotation_translation();

            let mut node = SceneNode::new(&bone.name);
            node.local_transform = Transform::new(translation, rotation, scale);

            // Assign SDF source if bone has a visual.
            if let Some(visual) = self.bone_visuals.iter().find(|v| v.bone_name == bone.name) {
                node.sdf_source = visual.sdf_source.clone();
                node.blend_mode = visual.blend_mode;
            } else {
                // Pure transform node.
                node.blend_mode = BlendMode::SmoothUnion(self.joint_blend_radius);
            }

            // For bones with geometry but no explicit blend mode, use joint blend radius.
            if let Some(visual) = self.bone_visuals.iter().find(|v| v.bone_name == bone.name) {
                if visual.blend_mode == BlendMode::default() {
                    node.blend_mode = BlendMode::SmoothUnion(self.joint_blend_radius);
                }
            }

            nodes.push(node);
            let _ = i; // used implicitly via enumerate
        }

        // Build tree from flat list using hierarchy.
        // Process in reverse order so children are attached before parents.
        // We need to work with indices since we're moving nodes around.
        let hierarchy = &self.skeleton.hierarchy;
        let roots = self.skeleton.root_bones();

        // Build tree bottom-up: collect children for each bone.
        let mut children_of: Vec<Vec<usize>> = vec![Vec::new(); bone_count];
        for (i, &parent) in hierarchy.iter().enumerate() {
            if parent >= 0 {
                children_of[parent as usize].push(i);
            }
        }

        // Recursive tree builder.
        fn build_subtree(
            bone_idx: usize,
            nodes: &[SceneNode],
            children_of: &[Vec<usize>],
        ) -> SceneNode {
            let mut node = nodes[bone_idx].clone();
            for &child_idx in &children_of[bone_idx] {
                let child = build_subtree(child_idx, nodes, children_of);
                node.children.push(child);
            }
            node
        }

        if roots.len() == 1 {
            build_subtree(roots[0], &nodes, &children_of)
        } else {
            // Multiple roots — wrap in a container node.
            let mut container = SceneNode::new("character_root");
            for &root_idx in &roots {
                let subtree = build_subtree(root_idx, &nodes, &children_of);
                container.children.push(subtree);
            }
            container
        }
    }

    /// Update the character's pose by evaluating the animation at the current time
    /// and writing local transforms to the SceneNode tree.
    ///
    /// The `root_node` must be the SceneNode tree returned by `build_scene_node()`.
    pub fn update_pose(&self, root_node: &mut SceneNode) {
        let local_transforms = self.player.evaluate_local(&self.skeleton);
        apply_transforms_to_tree(root_node, &self.skeleton, &local_transforms);
    }

    /// Advance animation by dt seconds and update the SceneNode tree pose.
    pub fn advance_and_update(&mut self, dt: f32, root_node: &mut SceneNode) {
        self.player.advance(dt);
        let local_transforms = self.player.evaluate_local(&self.skeleton);
        apply_transforms_to_tree(root_node, &self.skeleton, &local_transforms);
    }
}

/// Apply per-bone local transforms to a SceneNode tree by matching bone names.
///
/// Walks the skeleton's bone list and finds the corresponding node by name
/// in the SceneNode tree, then writes the local transform.
fn apply_transforms_to_tree(
    root: &mut SceneNode,
    skeleton: &Skeleton,
    transforms: &[Transform],
) {
    for (i, bone) in skeleton.bones.iter().enumerate() {
        if let Some(node) = root.find_by_name_mut(&bone.name) {
            node.local_transform = transforms[i];
        }
    }
}

/// Transform an AABB by a matrix, producing a new axis-aligned bounding box.
///
/// This expands the AABB to contain all 8 transformed corners.
pub fn transform_aabb(aabb: &Aabb, mat: &Mat4) -> Aabb {
    let corners = [
        Vec3::new(aabb.min.x, aabb.min.y, aabb.min.z),
        Vec3::new(aabb.max.x, aabb.min.y, aabb.min.z),
        Vec3::new(aabb.min.x, aabb.max.y, aabb.min.z),
        Vec3::new(aabb.max.x, aabb.max.y, aabb.min.z),
        Vec3::new(aabb.min.x, aabb.min.y, aabb.max.z),
        Vec3::new(aabb.max.x, aabb.min.y, aabb.max.z),
        Vec3::new(aabb.min.x, aabb.max.y, aabb.max.z),
        Vec3::new(aabb.max.x, aabb.max.y, aabb.max.z),
    ];

    let mut new_min = Vec3::splat(f32::MAX);
    let mut new_max = Vec3::splat(f32::MIN);
    for c in &corners {
        let t = mat.transform_point3(*c);
        new_min = new_min.min(t);
        new_max = new_max.max(t);
    }
    Aabb::new(new_min, new_max)
}

/// Build a standard 14-bone humanoid skeleton for testing.
///
/// Bone hierarchy:
/// ```text
/// 0: Hips (root)
///   1: Spine
///     2: Chest
///       3: Head
///       4: UpperArm_L
///         5: LowerArm_L
///           6: Hand_L
///       7: UpperArm_R
///         8: LowerArm_R
///           9: Hand_R
///   10: UpperLeg_L
///     11: LowerLeg_L
///   12: UpperLeg_R
///     13: LowerLeg_R
/// ```
pub fn build_humanoid_skeleton() -> Skeleton {
    use std::f32::consts::FRAC_PI_2;

    // Arm bones need rotation so capsules (Y-axis aligned) point along the arm
    // direction (X-axis). Children inherit the parent's rotation, so only the
    // upper arm bones carry an explicit rotation; child translations are expressed
    // in the parent's rotated local frame (local +Y = arm extension direction).
    let left_arm_rot = Quat::from_rotation_z(FRAC_PI_2);   // local Y → parent -X
    let right_arm_rot = Quat::from_rotation_z(-FRAC_PI_2);  // local Y → parent +X

    let bones = vec![
        make_bone("Hips", Vec3::new(0.0, 0.9, 0.0)),
        make_bone("Spine", Vec3::new(0.0, 0.15, 0.0)),
        make_bone("Chest", Vec3::new(0.0, 0.15, 0.0)),
        make_bone("Head", Vec3::new(0.0, 0.2, 0.0)),
        make_bone_rotated("UpperArm_L", Vec3::new(-0.18, 0.0, 0.0), left_arm_rot),
        make_bone("LowerArm_L", Vec3::new(0.0, 0.22, 0.0)),  // local +Y = further along arm
        make_bone("Hand_L", Vec3::new(0.0, 0.18, 0.0)),
        make_bone_rotated("UpperArm_R", Vec3::new(0.18, 0.0, 0.0), right_arm_rot),
        make_bone("LowerArm_R", Vec3::new(0.0, 0.22, 0.0)),
        make_bone("Hand_R", Vec3::new(0.0, 0.18, 0.0)),
        make_bone("UpperLeg_L", Vec3::new(-0.1, -0.05, 0.0)),
        make_bone("LowerLeg_L", Vec3::new(0.0, -0.35, 0.0)),
        make_bone("UpperLeg_R", Vec3::new(0.1, -0.05, 0.0)),
        make_bone("LowerLeg_R", Vec3::new(0.0, -0.35, 0.0)),
    ];

    let hierarchy = vec![
        -1, // 0: Hips (root)
        0,  // 1: Spine → Hips
        1,  // 2: Chest → Spine
        2,  // 3: Head → Chest
        2,  // 4: UpperArm_L → Chest
        4,  // 5: LowerArm_L → UpperArm_L
        5,  // 6: Hand_L → LowerArm_L
        2,  // 7: UpperArm_R → Chest
        7,  // 8: LowerArm_R → UpperArm_R
        8,  // 9: Hand_R → LowerArm_R
        0,  // 10: UpperLeg_L → Hips
        10, // 11: LowerLeg_L → UpperLeg_L
        0,  // 12: UpperLeg_R → Hips
        12, // 13: LowerLeg_R → UpperLeg_R
    ];

    Skeleton::new(bones, hierarchy).expect("valid humanoid skeleton")
}

/// Build default bone visuals for a humanoid — each bone is an analytical capsule.
pub fn build_humanoid_visuals(material_id: u16) -> Vec<BoneVisual> {
    vec![
        bone_visual("Hips", capsule(0.12, 0.06), material_id),
        bone_visual("Spine", capsule(0.1, 0.06), material_id),
        bone_visual("Chest", capsule(0.12, 0.06), material_id),
        bone_visual("Head", sphere(0.1), material_id),
        bone_visual("UpperArm_L", capsule(0.04, 0.1), material_id),
        bone_visual("LowerArm_L", capsule(0.035, 0.09), material_id),
        bone_visual("Hand_L", sphere(0.04), material_id),
        bone_visual("UpperArm_R", capsule(0.04, 0.1), material_id),
        bone_visual("LowerArm_R", capsule(0.035, 0.09), material_id),
        bone_visual("Hand_R", sphere(0.04), material_id),
        bone_visual("UpperLeg_L", capsule(0.06, 0.15), material_id),
        bone_visual("LowerLeg_L", capsule(0.05, 0.15), material_id),
        bone_visual("UpperLeg_R", capsule(0.06, 0.15), material_id),
        bone_visual("LowerLeg_R", capsule(0.05, 0.15), material_id),
    ]
}

/// Build a simple walk animation for the humanoid skeleton.
pub fn build_walk_clip() -> AnimationClip {
    use crate::clip::{BoneChannel, Keyframe};
    use std::f32::consts::{FRAC_PI_2, PI};

    let dur = 1.0; // 1 second per cycle

    // Arm bind rotations must be composed into animation keyframes because
    // evaluate_local() completely replaces the node's local_transform.
    let left_arm_bind = Quat::from_rotation_z(FRAC_PI_2);
    let right_arm_bind = Quat::from_rotation_z(-FRAC_PI_2);

    AnimationClip::new(
        "walk".to_string(),
        dur,
        vec![
            // Hips: slight vertical bob.
            BoneChannel {
                bone_index: 0,
                keyframes: vec![
                    Keyframe { time: 0.0, position: Vec3::new(0.0, 0.9, 0.0), rotation: Quat::IDENTITY, scale: Vec3::ONE },
                    Keyframe { time: dur * 0.25, position: Vec3::new(0.0, 0.92, 0.0), rotation: Quat::IDENTITY, scale: Vec3::ONE },
                    Keyframe { time: dur * 0.5, position: Vec3::new(0.0, 0.9, 0.0), rotation: Quat::IDENTITY, scale: Vec3::ONE },
                    Keyframe { time: dur * 0.75, position: Vec3::new(0.0, 0.92, 0.0), rotation: Quat::IDENTITY, scale: Vec3::ONE },
                    Keyframe { time: dur, position: Vec3::new(0.0, 0.9, 0.0), rotation: Quat::IDENTITY, scale: Vec3::ONE },
                ],
            },
            // UpperArm_L: swing forward/back around X, composed with bind rotation.
            BoneChannel {
                bone_index: 4,
                keyframes: vec![
                    Keyframe { time: 0.0, position: Vec3::new(-0.18, 0.0, 0.0), rotation: left_arm_bind * Quat::from_rotation_x(PI / 8.0), scale: Vec3::ONE },
                    Keyframe { time: dur * 0.5, position: Vec3::new(-0.18, 0.0, 0.0), rotation: left_arm_bind * Quat::from_rotation_x(-PI / 8.0), scale: Vec3::ONE },
                    Keyframe { time: dur, position: Vec3::new(-0.18, 0.0, 0.0), rotation: left_arm_bind * Quat::from_rotation_x(PI / 8.0), scale: Vec3::ONE },
                ],
            },
            // UpperArm_R: opposite phase to L, composed with bind rotation.
            BoneChannel {
                bone_index: 7,
                keyframes: vec![
                    Keyframe { time: 0.0, position: Vec3::new(0.18, 0.0, 0.0), rotation: right_arm_bind * Quat::from_rotation_x(-PI / 8.0), scale: Vec3::ONE },
                    Keyframe { time: dur * 0.5, position: Vec3::new(0.18, 0.0, 0.0), rotation: right_arm_bind * Quat::from_rotation_x(PI / 8.0), scale: Vec3::ONE },
                    Keyframe { time: dur, position: Vec3::new(0.18, 0.0, 0.0), rotation: right_arm_bind * Quat::from_rotation_x(-PI / 8.0), scale: Vec3::ONE },
                ],
            },
            // UpperLeg_L: swing forward/back.
            BoneChannel {
                bone_index: 10,
                keyframes: vec![
                    Keyframe { time: 0.0, position: Vec3::new(-0.1, -0.05, 0.0), rotation: Quat::from_rotation_x(-PI / 6.0), scale: Vec3::ONE },
                    Keyframe { time: dur * 0.5, position: Vec3::new(-0.1, -0.05, 0.0), rotation: Quat::from_rotation_x(PI / 6.0), scale: Vec3::ONE },
                    Keyframe { time: dur, position: Vec3::new(-0.1, -0.05, 0.0), rotation: Quat::from_rotation_x(-PI / 6.0), scale: Vec3::ONE },
                ],
            },
            // LowerLeg_L: bend at mid-stride.
            BoneChannel {
                bone_index: 11,
                keyframes: vec![
                    Keyframe { time: 0.0, position: Vec3::new(0.0, -0.35, 0.0), rotation: Quat::IDENTITY, scale: Vec3::ONE },
                    Keyframe { time: dur * 0.25, position: Vec3::new(0.0, -0.35, 0.0), rotation: Quat::from_rotation_x(-PI / 5.0), scale: Vec3::ONE },
                    Keyframe { time: dur * 0.5, position: Vec3::new(0.0, -0.35, 0.0), rotation: Quat::IDENTITY, scale: Vec3::ONE },
                    Keyframe { time: dur, position: Vec3::new(0.0, -0.35, 0.0), rotation: Quat::IDENTITY, scale: Vec3::ONE },
                ],
            },
            // UpperLeg_R: opposite phase.
            BoneChannel {
                bone_index: 12,
                keyframes: vec![
                    Keyframe { time: 0.0, position: Vec3::new(0.1, -0.05, 0.0), rotation: Quat::from_rotation_x(PI / 6.0), scale: Vec3::ONE },
                    Keyframe { time: dur * 0.5, position: Vec3::new(0.1, -0.05, 0.0), rotation: Quat::from_rotation_x(-PI / 6.0), scale: Vec3::ONE },
                    Keyframe { time: dur, position: Vec3::new(0.1, -0.05, 0.0), rotation: Quat::from_rotation_x(PI / 6.0), scale: Vec3::ONE },
                ],
            },
            // LowerLeg_R: bend at mid-stride (offset phase).
            BoneChannel {
                bone_index: 13,
                keyframes: vec![
                    Keyframe { time: 0.0, position: Vec3::new(0.0, -0.35, 0.0), rotation: Quat::IDENTITY, scale: Vec3::ONE },
                    Keyframe { time: dur * 0.5, position: Vec3::new(0.0, -0.35, 0.0), rotation: Quat::IDENTITY, scale: Vec3::ONE },
                    Keyframe { time: dur * 0.75, position: Vec3::new(0.0, -0.35, 0.0), rotation: Quat::from_rotation_x(-PI / 5.0), scale: Vec3::ONE },
                    Keyframe { time: dur, position: Vec3::new(0.0, -0.35, 0.0), rotation: Quat::IDENTITY, scale: Vec3::ONE },
                ],
            },
        ],
    )
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

use crate::skeleton::Bone;

fn make_bone(name: &str, translation: Vec3) -> Bone {
    let bind = glam::Mat4::from_translation(translation);
    Bone {
        name: name.to_string(),
        bind_transform: bind,
        inverse_bind: bind.inverse(),
    }
}

fn make_bone_rotated(name: &str, translation: Vec3, rotation: Quat) -> Bone {
    let bind = Mat4::from_rotation_translation(rotation, translation);
    Bone {
        name: name.to_string(),
        bind_transform: bind,
        inverse_bind: bind.inverse(),
    }
}

fn bone_visual(name: &str, primitive: SdfPrimitive, material_id: u16) -> BoneVisual {
    BoneVisual {
        bone_name: name.to_string(),
        sdf_source: SdfSource::Analytical {
            primitive,
            material_id,
        },
        blend_mode: BlendMode::default(),
    }
}

fn capsule(radius: f32, half_height: f32) -> SdfPrimitive {
    SdfPrimitive::Capsule {
        radius,
        half_height,
    }
}

fn sphere(radius: f32) -> SdfPrimitive {
    SdfPrimitive::Sphere { radius }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::FRAC_PI_2;

    #[test]
    fn transform_aabb_identity_returns_same() {
        let aabb = Aabb::new(Vec3::new(1.0, 2.0, 3.0), Vec3::new(4.0, 5.0, 6.0));
        let result = transform_aabb(&aabb, &Mat4::IDENTITY);
        assert!((result.min - aabb.min).length() < 1e-5);
        assert!((result.max - aabb.max).length() < 1e-5);
    }

    #[test]
    fn transform_aabb_translation_shifts() {
        let aabb = Aabb::new(Vec3::ZERO, Vec3::ONE);
        let mat = Mat4::from_translation(Vec3::new(10.0, 20.0, 30.0));
        let result = transform_aabb(&aabb, &mat);
        assert!((result.min - Vec3::new(10.0, 20.0, 30.0)).length() < 1e-5);
        assert!((result.max - Vec3::new(11.0, 21.0, 31.0)).length() < 1e-5);
    }

    #[test]
    fn transform_aabb_90deg_rotation_swaps_axes() {
        let aabb = Aabb::new(Vec3::ZERO, Vec3::ONE);
        let mat = Mat4::from_rotation_z(FRAC_PI_2);
        let result = transform_aabb(&aabb, &mat);
        assert!((result.min.x - (-1.0)).abs() < 1e-4);
        assert!(result.max.x.abs() < 1e-4);
        assert!(result.min.y.abs() < 1e-4);
        assert!((result.max.y - 1.0).abs() < 1e-4);
    }

    // ── Humanoid skeleton ─────────────────────────────────────────────────────

    #[test]
    fn humanoid_skeleton_has_14_bones() {
        let skel = build_humanoid_skeleton();
        assert_eq!(skel.bone_count(), 14);
    }

    #[test]
    fn humanoid_skeleton_single_root() {
        let skel = build_humanoid_skeleton();
        assert_eq!(skel.root_bones(), vec![0], "Hips should be the only root");
    }

    #[test]
    fn humanoid_visuals_has_14_entries() {
        let visuals = build_humanoid_visuals(1);
        assert_eq!(visuals.len(), 14);
    }

    // ── AnimatedCharacter ─────────────────────────────────────────────────────

    #[test]
    fn character_build_scene_node_creates_tree() {
        let skel = build_humanoid_skeleton();
        let visuals = build_humanoid_visuals(1);
        let clip = build_walk_clip();

        let character = AnimatedCharacter::new(skel, visuals, clip, 0.05);
        let root = character.build_scene_node();

        // Root should be "Hips" (single skeleton root).
        assert_eq!(root.name, "Hips");
        // Total nodes = 14 bones.
        assert_eq!(root.node_count(), 14);
    }

    #[test]
    fn character_tree_has_correct_hierarchy() {
        let skel = build_humanoid_skeleton();
        let visuals = build_humanoid_visuals(1);
        let clip = build_walk_clip();

        let character = AnimatedCharacter::new(skel, visuals, clip, 0.05);
        let root = character.build_scene_node();

        // Hips has 3 children: Spine, UpperLeg_L, UpperLeg_R
        assert_eq!(root.children.len(), 3);
        assert_eq!(root.children[0].name, "Spine");
        assert_eq!(root.children[1].name, "UpperLeg_L");
        assert_eq!(root.children[2].name, "UpperLeg_R");

        // Spine → Chest
        let spine = &root.children[0];
        assert_eq!(spine.children.len(), 1);
        assert_eq!(spine.children[0].name, "Chest");

        // Chest → Head, UpperArm_L, UpperArm_R
        let chest = &spine.children[0];
        assert_eq!(chest.children.len(), 3);
        assert_eq!(chest.children[0].name, "Head");
        assert_eq!(chest.children[1].name, "UpperArm_L");
        assert_eq!(chest.children[2].name, "UpperArm_R");
    }

    #[test]
    fn character_bones_have_sdf_sources() {
        let skel = build_humanoid_skeleton();
        let visuals = build_humanoid_visuals(5);
        let clip = build_walk_clip();

        let character = AnimatedCharacter::new(skel, visuals, clip, 0.05);
        let root = character.build_scene_node();

        // Head should be a sphere.
        let head = root.find_by_name("Head").unwrap();
        match &head.sdf_source {
            SdfSource::Analytical { primitive, material_id } => {
                assert!(matches!(primitive, SdfPrimitive::Sphere { .. }));
                assert_eq!(*material_id, 5);
            }
            _ => panic!("Head should be Analytical Sphere"),
        }

        // UpperArm_L should be a capsule.
        let arm = root.find_by_name("UpperArm_L").unwrap();
        match &arm.sdf_source {
            SdfSource::Analytical { primitive, .. } => {
                assert!(matches!(primitive, SdfPrimitive::Capsule { .. }));
            }
            _ => panic!("UpperArm_L should be Analytical Capsule"),
        }
    }

    #[test]
    fn character_update_pose_modifies_transforms() {
        let skel = build_humanoid_skeleton();
        let visuals = build_humanoid_visuals(1);
        let clip = build_walk_clip();

        let mut character = AnimatedCharacter::new(skel, visuals, clip, 0.05);
        let mut root = character.build_scene_node();

        // Record initial hip position.
        let initial_hip_y = root.local_transform.position.y;

        // Advance to t=0.25 where hips bob up.
        character.player.seek(0.25);
        character.update_pose(&mut root);

        let updated_hip_y = root.local_transform.position.y;
        assert!(
            (updated_hip_y - 0.92).abs() < 1e-4,
            "hips should bob to 0.92 at t=0.25, got {updated_hip_y}"
        );
        assert!(updated_hip_y > initial_hip_y, "hips should move up");
    }

    #[test]
    fn character_advance_and_update_progresses_animation() {
        let skel = build_humanoid_skeleton();
        let visuals = build_humanoid_visuals(1);
        let clip = build_walk_clip();

        let mut character = AnimatedCharacter::new(skel, visuals, clip, 0.05);
        let mut root = character.build_scene_node();

        let t0 = character.player.current_time();
        character.advance_and_update(0.1, &mut root);
        let t1 = character.player.current_time();

        assert!(t1 > t0, "time should advance");
    }

    #[test]
    fn walk_clip_has_correct_duration() {
        let clip = build_walk_clip();
        assert_eq!(clip.duration, 1.0);
        assert_eq!(clip.channels.len(), 7); // 7 animated bones
    }

    #[test]
    fn walk_clip_bone_channels_target_valid_bones() {
        let skel = build_humanoid_skeleton();
        let clip = build_walk_clip();

        for ch in &clip.channels {
            assert!(
                (ch.bone_index as usize) < skel.bone_count(),
                "channel targets bone {} which exceeds skeleton size {}",
                ch.bone_index,
                skel.bone_count()
            );
        }
    }
}
