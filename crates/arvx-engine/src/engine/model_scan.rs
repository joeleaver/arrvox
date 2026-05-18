//! Project asset discovery.
//!
//! Walks `assets/models/` for `.arvx` files, reads voxel-count footers,
//! maps each `.arvx` back to its source mesh, and publishes the list in
//! the next `StateUpdate`. Also hosts low-level octree helpers
//! (`spatial_from_handle`, `collect_leaf_slots`) used by loads, bakes,
//! and scene IO — they sit here because the asset-scan path was their
//! original home.

use super::state::EngineState;
use crate::components::SpatialData;

impl EngineState {
    pub(crate) fn scan_models(&mut self) {
        self.available_models.clear();
        if let Some(ref project_dir) = self.project_dir {
            let assets_dir = project_dir.join("assets");
            if assets_dir.exists() {
                Self::scan_rkp_recursive(&assets_dir, &mut self.available_models);
            }
            self.available_models.sort_by(|a, b| a.name.cmp(&b.name));
            self.models_dirty = true;
            eprintln!("[ArvxEngine] scanned {} models", self.available_models.len());
        }
        // Same lifecycle — scan presets alongside models so a project
        // open / model-watcher refresh picks up new `.arvxgen` files too.
        self.scan_generator_presets();
    }
}

/// Walk the octree subtree rooted at `node_idx` and collect every
/// leaf_attr_pool slot that's reachable from this entity.
///
/// Handles all three terminal node kinds:
/// - LEAF nodes: the node encodes a leaf_attr_slot directly.
/// - BRICK nodes: the node encodes a brick_id; the brick's 64 cells
///   each hold a leaf_attr_slot (or `BRICK_EMPTY`).
/// - EMPTY / INTERIOR: contribute nothing.
///
/// Modern arrvox terminates the octree at BRICK nodes for most
/// geometry, so callers that only walked LEAFs saw zero voxels. This
/// function descends into bricks via the passed-in `brick_pool`.
pub(crate) fn collect_leaf_slots(
    all_nodes: &[u32],
    brick_pool: &arvx_core::brick_pool::BrickPool,
    node_idx: usize,
    out: &mut Vec<u32>,
) {
    if node_idx >= all_nodes.len() {
        return;
    }
    let node = all_nodes[node_idx];
    if node == arvx_core::sparse_octree::EMPTY_NODE || node == arvx_core::sparse_octree::INTERIOR_NODE {
        return;
    }
    if arvx_core::sparse_octree::is_leaf(node) {
        out.push(arvx_core::sparse_octree::leaf_slot(node));
        return;
    }
    if arvx_core::sparse_octree::is_brick(node) {
        let bid = arvx_core::sparse_octree::brick_id(node);
        for &cell in brick_pool.brick_cells(bid) {
            if cell != arvx_core::brick_pool::BRICK_EMPTY {
                out.push(cell);
            }
        }
        return;
    }
    // Branch — value is absolute offset to 8 contiguous children.
    let children_offset = node as usize;
    for octant in 0..8 {
        collect_leaf_slots(all_nodes, brick_pool, children_offset + octant, out);
    }
}

/// Walk the subtree and collect every prefilter `internal_attr` slot at
/// branch nodes. These are the aggregated leaf_attr ids the march falls
/// back to when the LOD cutoff kicks in at distance — they must be
/// mutated alongside the per-voxel slots whenever material state on
/// real voxels changes, otherwise distant pixels shade through a stale
/// prefilter and appear visually different from the close-up.
pub(crate) fn collect_internal_attr_slots(
    all_nodes: &[u32],
    internal_attrs: &[u32],
    node_idx: usize,
    out: &mut Vec<u32>,
) {
    if node_idx >= all_nodes.len() {
        return;
    }
    let node = all_nodes[node_idx];
    if node == arvx_core::sparse_octree::EMPTY_NODE
        || node == arvx_core::sparse_octree::INTERIOR_NODE
        || arvx_core::sparse_octree::is_leaf(node)
        || arvx_core::sparse_octree::is_brick(node)
    {
        return;
    }
    // Branch. Emit its prefilter slot (if any) and recurse.
    if node_idx < internal_attrs.len() {
        let attr = internal_attrs[node_idx];
        if attr != arvx_core::sparse_octree::INTERNAL_ATTR_NONE {
            out.push(attr);
        }
    }
    let children_offset = node as usize;
    for octant in 0..8 {
        collect_internal_attr_slots(all_nodes, internal_attrs, children_offset + octant, out);
    }
}

/// Convert a SpatialHandle from arvx_render into our SpatialData component.
pub(crate) fn spatial_from_handle(
    handle: &arvx_core::scene_node::SpatialHandle,
    voxel_size: f32,
    aabb: &arvx_core::Aabb,
    grid_origin: glam::Vec3,
    voxel_slot_start: u32,
    voxel_slot_count: u32,
    brick_ids: Vec<u32>,
) -> SpatialData {
    if let arvx_core::scene_node::SpatialHandle::Octree {
        root_offset, len, depth, base_voxel_size,
    } = handle
    {
        SpatialData {
            root_offset: *root_offset,
            len: *len,
            depth: *depth,
            base_voxel_size: *base_voxel_size,
            aabb: *aabb,
            voxel_size,
            grid_origin,
            voxel_slot_start,
            voxel_slot_count,
            brick_ids,
        }
    } else {
        SpatialData {
            root_offset: 0, len: 0, depth: 0, base_voxel_size: voxel_size,
            aabb: *aabb, voxel_size,
            grid_origin,
            voxel_slot_start, voxel_slot_count,
            brick_ids,
        }
    }
}

/// Read just the voxel count from a `.arvx` header. Opens the file,
/// parses the header (cheap — header carries `voxel_count` directly
/// near the start), then drops the reader. None on any I/O or format
/// error; callers fall back to 0 (unknown).
pub(crate) fn read_rkp_voxel_count(path: &std::path::Path) -> Option<u32> {
    let file = std::fs::File::open(path).ok()?;
    let mut reader = std::io::BufReader::new(file);
    let header = arvx_core::asset_file::read_rkp_header(&mut reader).ok()?;
    Some(header.voxel_count)
}

impl EngineState {
    pub(crate) fn scan_rkp_recursive(dir: &std::path::Path, out: &mut Vec<crate::snapshot::ModelInfo>) {
        let Ok(entries) = std::fs::read_dir(dir) else { return };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                Self::scan_rkp_recursive(&path, out);
            } else if path.extension().map(|e| e == "arvx").unwrap_or(false) {
                let size = entry.metadata().map(|m| m.len()).unwrap_or(0);
                let arvx_path = path.to_string_lossy().into_owned();

                // Try to find the source mesh file (the .arvx was generated from it).
                // Convention: source.glb → source.arvx, so source = rkp with mesh extension.
                let source_path = Self::find_source_for_rkp(&path);
                let profile = source_path.as_ref().map(|sp| {
                    crate::import_profile::ImportProfile::load_or_default(sp)
                });

                // Display name: profile override → filename stem.
                let name = profile.as_ref()
                    .and_then(|p| p.display_name.clone())
                    .unwrap_or_else(|| {
                        path.file_stem()
                            .map(|s| s.to_string_lossy().into_owned())
                            .unwrap_or_default()
                    });

                // Read just the header to surface the voxel count in
                // the Asset Properties panel. Header is the first
                // bytes of the file — one small seek per asset during
                // the scan, negligible vs the full .arvx load.
                let voxel_count = read_rkp_voxel_count(&path).unwrap_or(0);

                out.push(crate::snapshot::ModelInfo {
                    name,
                    path: arvx_path,
                    source_path: source_path
                        .map(|p| p.to_string_lossy().into_owned())
                        .unwrap_or_default(),
                    size,
                    voxel_count,
                    import_profile: profile,
                });
            }
        }
    }

    /// Find the source mesh file for a .arvx output.
    /// Convention: bunny.arvx was generated from bunny.glb (or .gltf, .obj, .fbx).
    pub(crate) fn find_source_for_rkp(arvx_path: &std::path::Path) -> Option<std::path::PathBuf> {
        let stem = arvx_path.with_extension("");
        for ext in &["glb", "gltf", "obj", "fbx"] {
            let candidate = stem.with_extension(ext);
            if candidate.exists() {
                return Some(candidate);
            }
        }
        None
    }
}
