//! Skeleton extraction and animation import.
//!
//! Extracts a bone hierarchy, per-vertex skin weights, and animation
//! clips from source mesh files. Returns `Ok(None)` for static
//! meshes — a missing skeleton is not an error.
//!
//! Per-format extractors live in the sibling modules ([`gltf`],
//! [`fbx`]); dispatched by extension via [`extract_skeleton`].

use std::path::Path;

use rkp_animation::clip::AnimationClip;
use rkp_animation::skeleton::Skeleton;

pub mod fbx;
pub mod gltf;

/// Per-vertex bone influences: up to 4 `(bone_index, weight)` pairs
/// per vertex. `bone_index = -1` means "no influence" for that slot.
/// The vector lengths match the source mesh's vertex count.
#[derive(Clone, Debug, Default)]
pub struct VertexSkinning {
    /// `[bone_index; 4]` per vertex. `-1` is a sentinel for unused slots.
    pub joints: Vec<[i32; 4]>,
    /// `[weight; 4]` per vertex. Each row sums to ~1.0 after
    /// normalization; unused slots carry 0.0.
    pub weights: Vec<[f32; 4]>,
}

/// Full output of [`extract_skeleton`].
#[derive(Clone, Debug)]
pub struct SkeletonExtraction {
    /// Bone hierarchy + bind poses.
    pub skeleton: Skeleton,
    /// Per-vertex bone weights (mesh-order).
    pub skinning: VertexSkinning,
    /// Animation clips bundled with the source.
    pub clips: Vec<AnimationClip>,
}

/// Dispatch to the per-format skeleton extractor. Returns `Ok(None)`
/// when the source file has no skeleton (a static mesh is not an
/// error).
pub fn extract_skeleton(path: &str) -> Result<Option<SkeletonExtraction>, String> {
    let ext = Path::new(path)
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();

    match ext.as_str() {
        "gltf" | "glb" => gltf::extract(path),
        "fbx" => fbx::extract(path),
        other => Err(format!(
            "Unsupported format for skeleton extraction: .{other}. Supported: .gltf, .glb, .fbx"
        )),
    }
}
