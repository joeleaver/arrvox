//! Import pipeline configuration + result types.
//!
//! Fresh opacity-octree-specific versions. Dropped from the original
//! `rkf-import::ImportConfig`: `lod_levels` (the octree is the LOD
//! hierarchy), `pool_size` (brick pool in rkp-core is sized
//! independently at load time), `verbose` (replaced by
//! [`crate::event::ProgressReporter`]).

use std::path::PathBuf;

use rkp_core::Aabb;

/// Configuration for a mesh-to-.rkp import.
///
/// Constructed programmatically or from an editor-level import profile
/// (`rkp-engine::import_profile`). All fields have sensible defaults
/// via [`Default::default`] — callers typically set just `voxel_size`
/// and leave the rest alone.
#[derive(Debug, Clone)]
pub struct ImportConfig {
    /// Finest voxel size in metres. `None` picks an auto-detected tier
    /// based on mesh extent (see `voxelize::auto_voxel_size`).
    pub voxel_size: Option<f32>,
    /// Target size for the longest axis after normalization. Ignored
    /// if [`no_normalize`](Self::no_normalize) is set.
    pub target_size: f32,
    /// Skip normalization — keep original mesh coordinates.
    pub no_normalize: bool,
    /// Force a single material ID for every voxel (otherwise, material
    /// is taken from the nearest triangle's material slot).
    pub material_id_override: Option<u16>,
    /// Sample per-voxel colour from the mesh's albedo textures.
    pub import_colors: bool,
    /// Euler rotation offset `[X, Y, Z]` in degrees, applied in original
    /// mesh space before normalization.
    pub rotation_offset: [f32; 3],
    /// Additional uniform scale applied after normalization.
    pub scale_override: Option<f32>,
}

impl Default for ImportConfig {
    fn default() -> Self {
        Self {
            voxel_size: None,
            target_size: 1.0,
            no_normalize: false,
            material_id_override: None,
            import_colors: true,
            rotation_offset: [0.0, 0.0, 0.0],
            scale_override: None,
        }
    }
}

impl ImportConfig {
    /// Validate precondition invariants. Called at the top of
    /// [`crate::voxelize::import_mesh_to_opacity_rkp_with`] so a
    /// malformed config produces a clean error before any expensive
    /// mesh loading or BVH work happens.
    ///
    /// Checks: finite sizes, positive sizes where required, no NaN
    /// rotations. Returns a human-readable message on the first
    /// failure (importer then converts it into an
    /// [`crate::ImportEvent::Error`]).
    pub fn validate(&self) -> Result<(), String> {
        if let Some(vs) = self.voxel_size {
            if !vs.is_finite() || vs <= 0.0 {
                return Err(format!(
                    "voxel_size must be positive and finite, got {vs}"
                ));
            }
        }
        if !self.target_size.is_finite() {
            return Err(format!(
                "target_size must be finite, got {}",
                self.target_size
            ));
        }
        if !self.no_normalize && self.target_size <= 0.0 {
            return Err(format!(
                "target_size must be positive when normalizing, got {}",
                self.target_size
            ));
        }
        for (axis, v) in ['X', 'Y', 'Z'].iter().zip(self.rotation_offset.iter()) {
            if !v.is_finite() {
                return Err(format!(
                    "rotation_offset.{axis} must be finite, got {v}"
                ));
            }
        }
        if let Some(s) = self.scale_override {
            if !s.is_finite() || s <= 0.0 {
                return Err(format!(
                    "scale_override must be positive and finite, got {s}"
                ));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_validates() {
        assert!(ImportConfig::default().validate().is_ok());
    }

    #[test]
    fn negative_voxel_size_rejected() {
        let c = ImportConfig { voxel_size: Some(-1.0), ..Default::default() };
        assert!(c.validate().is_err());
    }

    #[test]
    fn zero_voxel_size_rejected() {
        let c = ImportConfig { voxel_size: Some(0.0), ..Default::default() };
        assert!(c.validate().is_err());
    }

    #[test]
    fn nan_voxel_size_rejected() {
        let c = ImportConfig { voxel_size: Some(f32::NAN), ..Default::default() };
        assert!(c.validate().is_err());
    }

    #[test]
    fn zero_target_size_rejected_when_normalizing() {
        let c = ImportConfig { target_size: 0.0, ..Default::default() };
        assert!(c.validate().is_err());
    }

    #[test]
    fn zero_target_size_ok_when_not_normalizing() {
        let c = ImportConfig { target_size: 0.0, no_normalize: true, ..Default::default() };
        assert!(c.validate().is_ok());
    }

    #[test]
    fn nan_rotation_rejected() {
        let c = ImportConfig { rotation_offset: [0.0, f32::NAN, 0.0], ..Default::default() };
        assert!(c.validate().is_err());
    }

    #[test]
    fn negative_scale_rejected() {
        let c = ImportConfig { scale_override: Some(-1.0), ..Default::default() };
        assert!(c.validate().is_err());
    }

    #[test]
    fn positive_scale_accepted() {
        let c = ImportConfig { scale_override: Some(2.0), ..Default::default() };
        assert!(c.validate().is_ok());
    }
}

/// Result of a successful mesh-to-.rkp import.
#[derive(Debug, Clone)]
pub struct ImportResult {
    /// Tight AABB of the imported model (after normalization/transforms).
    pub aabb: Aabb,
    /// Number of shell voxels emitted (the 1-voxel-thick outer surface).
    pub shell_voxels: u32,
    /// Finest voxel size used (post auto-detect).
    pub finest_voxel_size: f32,
    /// Output `.rkp` file size in bytes.
    pub file_size: u64,
    /// Path to the generated `.rkskel` skeleton asset, if the source had bones.
    pub skeleton_path: Option<PathBuf>,
}
