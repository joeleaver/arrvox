//! Result of baking one terrain tile.
//!
//! Self-contained — produced off the engine thread by `bake_tile`,
//! integrated back into the scene's shared pools by the main thread
//! at materialise-time (Phase 2). Carries everything the renderer
//! needs: the octree, the per-leaf attribute pool, the brick payloads,
//! the face-adjacency table, the surface mesh, and the cluster DAG.

use crate::tile_key::TileKey;
use arvx_core::asset_file::MeshSectionsBlob;
use arvx_core::voxelize_octree::BakeArtifact;

/// One baked terrain tile, ready to integrate into the scene.
///
/// The `artifact` carries worker-local IDs (leaf attrs + brick ids in
/// dense `0..n` ranges); the main thread's integrate-time path is
/// responsible for relocating them into scene-global pools — same
/// pattern as `arvx-import`'s async bake.
pub struct BakedTile {
    /// Which tile this is.
    pub key: TileKey,
    /// Voxel size used for the bake, in metres.
    pub voxel_size_m: f32,
    /// Voxelisation result (octree + leaf attrs + brick cells + faces).
    pub artifact: BakeArtifact,
    /// Surface mesh + cluster DAG, ready for the v6 `.arvx` mesh sections
    /// (or for direct GPU upload at integrate-time).
    pub mesh: MeshSectionsBlob,
    /// Wall time of the bake in milliseconds (voxelize + mesh + DAG).
    pub bake_time_ms: f32,
}

impl BakedTile {
    /// Convenience: number of vertices in the surface mesh's LOD-0 plus
    /// every higher-LOD level. Useful for stats and tests.
    pub fn vertex_count(&self) -> usize {
        // 32 B per MeshVertex (per CLAUDE.md's "Key Data Types").
        self.mesh.vertices.len() / 32
    }

    /// Convenience: total index count across all LOD levels.
    pub fn index_count(&self) -> usize {
        self.mesh.indices.len() / 4
    }

    /// Convenience: cluster count across all LOD levels (64 B per
    /// `MeshletCluster`).
    pub fn cluster_count(&self) -> usize {
        self.mesh.clusters.len() / 64
    }
}
