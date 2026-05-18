//! Shared skeleton asset definition.
//!
//! A [`SkeletonAsset`] is an immutable skeleton definition that can be shared
//! across multiple animated entities. It bundles the bone hierarchy with
//! animation clips and provides I/O for the `.arvxskel` file format.

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::clip::AnimationClip;
use crate::skeleton::{Skeleton, SkeletonError};

// ─── Errors ─────────────────────────────────────────────────────────────────

/// Errors from skeleton asset operations.
#[derive(Debug, Error)]
pub enum SkeletonAssetError {
    /// The skeleton hierarchy is invalid.
    #[error("skeleton validation failed: {0}")]
    Validation(#[from] SkeletonError),

    /// I/O error reading or writing a .arvxskel file.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// RON serialization/deserialization error.
    #[error("RON error: {0}")]
    Ron(#[from] ron::error::SpannedError),

    /// RON serialization error (write path).
    #[error("RON write error: {0}")]
    RonWrite(#[from] ron::Error),
}

// ─── SkeletonAsset ──────────────────────────────────────────────────────────

/// Shared, immutable skeleton definition.
///
/// Contains the bone hierarchy and animation clips. Multiple entities can
/// reference the same `SkeletonAsset` via `Arc<SkeletonAsset>`.
///
/// Inverse bind matrices are stored per-bone in [`Skeleton::bones`] as
/// `Bone::inverse_bind` — no separate field needed.
///
/// Per-voxel bone weights are NOT stored here — they are per-object data
/// stored alongside geometry in `.rkf` v4 files.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkeletonAsset {
    /// Bone hierarchy (flat array + parent indices).
    pub skeleton: Skeleton,

    /// Animation clips associated with this skeleton.
    pub clips: Vec<AnimationClip>,

    /// Mesh normalization center applied during voxelization.
    /// Bone transforms are in original glTF space; this center + scale
    /// maps them to the normalized voxel grid space.
    /// `normalized_pos = (original_pos - mesh_center) * mesh_scale`
    #[serde(default)]
    pub mesh_center: [f32; 3],

    /// Mesh normalization scale factor (1/longest_axis).
    #[serde(default = "default_mesh_scale")]
    pub mesh_scale: f32,

    /// Rotation offset (degrees, XYZ Euler) applied to the mesh before normalization.
    /// Bone transforms must be rotated by the same amount to match the voxelized mesh.
    #[serde(default)]
    pub rotation_offset: [f32; 3],

    /// Pre-rotation mesh center (the pivot point for rotation_offset).
    #[serde(default)]
    pub rotation_center: [f32; 3],
}

fn default_mesh_scale() -> f32 {
    1.0
}

impl SkeletonAsset {
    /// Create a new skeleton asset.
    pub fn new(skeleton: Skeleton, clips: Vec<AnimationClip>) -> Self {
        Self {
            skeleton, clips,
            mesh_center: [0.0; 3], mesh_scale: 1.0,
            rotation_offset: [0.0; 3], rotation_center: [0.0; 3],
        }
    }

    /// Create with explicit normalization parameters from the converter.
    pub fn with_normalization(
        skeleton: Skeleton,
        clips: Vec<AnimationClip>,
        mesh_center: [f32; 3],
        mesh_scale: f32,
        rotation_offset: [f32; 3],
        rotation_center: [f32; 3],
    ) -> Self {
        Self { skeleton, clips, mesh_center, mesh_scale, rotation_offset, rotation_center }
    }

    /// Find an animation clip by name.
    pub fn find_clip(&self, name: &str) -> Option<&AnimationClip> {
        self.clips.iter().find(|c| c.name == name)
    }

    /// Validate the skeleton hierarchy.
    ///
    /// Call this after deserialization to ensure the data is well-formed.
    pub fn validate(&self) -> Result<(), SkeletonError> {
        // Re-run the Skeleton::new validation logic.
        Skeleton::new(
            self.skeleton.bones.clone(),
            self.skeleton.hierarchy.clone(),
        )?;
        Ok(())
    }
}

// ─── .arvxskel I/O ────────────────────────────────────────────────────────────

/// Write a [`SkeletonAsset`] to a writer in RON format.
pub fn write_rkskel(
    asset: &SkeletonAsset,
    writer: &mut impl std::io::Write,
) -> Result<(), SkeletonAssetError> {
    let config = ron::ser::PrettyConfig::default();
    let ron_str = ron::ser::to_string_pretty(asset, config)?;
    writer.write_all(ron_str.as_bytes())?;
    Ok(())
}

/// Read a [`SkeletonAsset`] from a reader in RON format.
///
/// Validates the skeleton hierarchy after deserialization.
pub fn read_rkskel(
    reader: &mut impl std::io::Read,
) -> Result<SkeletonAsset, SkeletonAssetError> {
    let mut buf = String::new();
    reader.read_to_string(&mut buf)?;
    let asset: SkeletonAsset = ron::from_str(&buf)?;
    asset.validate()?;
    Ok(asset)
}

/// Save a [`SkeletonAsset`] to a file path.
pub fn save_rkskel(
    asset: &SkeletonAsset,
    path: &std::path::Path,
) -> Result<(), SkeletonAssetError> {
    let mut file = std::fs::File::create(path)?;
    write_rkskel(asset, &mut file)
}

/// Load a [`SkeletonAsset`] from a file path.
pub fn load_rkskel(
    path: &std::path::Path,
) -> Result<SkeletonAsset, SkeletonAssetError> {
    let mut file = std::fs::File::open(path)?;
    read_rkskel(&mut file)
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clip::{BoneChannel, Keyframe};
    use crate::skeleton::Bone;
    use glam::{Mat4, Quat, Vec3};
    use std::io::Cursor;

    fn make_skeleton() -> Skeleton {
        let bones = vec![
            Bone {
                name: "root".into(),
                bind_transform: Mat4::IDENTITY,
                inverse_bind: Mat4::IDENTITY,
            },
            Bone {
                name: "child".into(),
                bind_transform: Mat4::from_translation(Vec3::new(0.0, 1.0, 0.0)),
                inverse_bind: Mat4::from_translation(Vec3::new(0.0, -1.0, 0.0)),
            },
        ];
        Skeleton::new(bones, vec![-1, 0]).unwrap()
    }

    fn make_clip() -> AnimationClip {
        AnimationClip::new(
            "walk".into(),
            1.0,
            vec![BoneChannel {
                bone_index: 0,
                keyframes: vec![Keyframe {
                    time: 0.0,
                    position: Vec3::ZERO,
                    rotation: Quat::IDENTITY,
                    scale: Vec3::ONE,
                }],
            }],
        )
    }

    // ── Construction ────────────────────────────────────────────────────────

    #[test]
    fn test_skeleton_asset_new() {
        let skel = make_skeleton();
        let clip = make_clip();
        let asset = SkeletonAsset::new(skel, vec![clip]);
        assert_eq!(asset.skeleton.bone_count(), 2);
        assert_eq!(asset.clips.len(), 1);
    }

    #[test]
    fn test_find_clip_hit() {
        let asset = SkeletonAsset::new(make_skeleton(), vec![make_clip()]);
        let found = asset.find_clip("walk");
        assert!(found.is_some());
        assert_eq!(found.unwrap().name, "walk");
    }

    #[test]
    fn test_find_clip_miss() {
        let asset = SkeletonAsset::new(make_skeleton(), vec![make_clip()]);
        assert!(asset.find_clip("idle").is_none());
    }

    // ── Serde roundtrip ─────────────────────────────────────────────────────

    #[test]
    fn test_skeleton_asset_serde_roundtrip() {
        let asset = SkeletonAsset::new(make_skeleton(), vec![make_clip()]);
        let ron_str = ron::to_string(&asset).expect("serialize");
        let asset2: SkeletonAsset = ron::from_str(&ron_str).expect("deserialize");
        assert_eq!(asset.skeleton.bone_count(), asset2.skeleton.bone_count());
        assert_eq!(asset.clips.len(), asset2.clips.len());
        assert_eq!(asset.clips[0].name, asset2.clips[0].name);
    }

    // ── Validation ──────────────────────────────────────────────────────────

    #[test]
    fn test_validate_good_skeleton() {
        let asset = SkeletonAsset::new(make_skeleton(), vec![]);
        assert!(asset.validate().is_ok());
    }

    #[test]
    fn test_validate_rejects_bad_hierarchy() {
        // Manually construct invalid data (bypassing Skeleton::new).
        let bad = SkeletonAsset {
            skeleton: Skeleton {
                bones: vec![Bone {
                    name: "a".into(),
                    bind_transform: Mat4::IDENTITY,
                    inverse_bind: Mat4::IDENTITY,
                }],
                hierarchy: vec![0], // self-reference
            },
            clips: vec![],
            mesh_center: [0.0; 3],
            mesh_scale: 1.0,
            rotation_offset: [0.0; 3],
            rotation_center: [0.0; 3],
        };
        assert!(bad.validate().is_err());
    }

    // ── .arvxskel I/O ─────────────────────────────────────────────────────────

    #[test]
    fn test_rkskel_write_read_roundtrip() {
        let asset = SkeletonAsset::new(make_skeleton(), vec![make_clip()]);

        let mut buf = Vec::new();
        write_rkskel(&asset, &mut buf).expect("write");

        let mut cursor = Cursor::new(buf);
        let asset2 = read_rkskel(&mut cursor).expect("read");

        assert_eq!(asset.skeleton.bone_count(), asset2.skeleton.bone_count());
        assert_eq!(asset.skeleton.hierarchy, asset2.skeleton.hierarchy);
        assert_eq!(asset.clips.len(), asset2.clips.len());
        assert_eq!(asset.clips[0].name, asset2.clips[0].name);
    }

    #[test]
    fn test_rkskel_read_validates_hierarchy() {
        // Serialize invalid data, then try to load it.
        let bad = SkeletonAsset {
            skeleton: Skeleton {
                bones: vec![Bone {
                    name: "a".into(),
                    bind_transform: Mat4::IDENTITY,
                    inverse_bind: Mat4::IDENTITY,
                }],
                hierarchy: vec![0], // self-reference
            },
            clips: vec![],
            mesh_center: [0.0; 3],
            mesh_scale: 1.0,
            rotation_offset: [0.0; 3],
            rotation_center: [0.0; 3],
        };

        let mut buf = Vec::new();
        // write_rkskel doesn't validate — it just serializes.
        write_rkskel(&bad, &mut buf).expect("write bad data");

        let mut cursor = Cursor::new(buf);
        let result = read_rkskel(&mut cursor);
        assert!(result.is_err(), "should reject invalid skeleton on load");
    }
}
