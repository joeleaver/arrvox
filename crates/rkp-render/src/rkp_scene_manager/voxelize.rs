//! Voxelize + integrate + deallocate_geometry methods.
//!
//! Sibling impl block on `RkpSceneManager`. Procedural primitives and
//! mesh-import results land here through `voxelize_primitive` /
//! `voxelize_sdf_fn` / `integrate_artifact`.

use super::manager::RkpSceneManager;
use super::types::{emit_faces, VoxelizeResult};

impl RkpSceneManager {
    pub fn voxelize_primitive(
        &mut self,
        primitive: &rkp_core::scene_node::SdfPrimitive,
        material_id: u16,
        voxel_size: f32,
        bake_scale: glam::Vec3,
        object_id: u32,
    ) -> Option<VoxelizeResult> {
        self.bump_geometry_epoch();
        use rkp_core::scene_node::SdfPrimitive;

        fn primitive_half_extents(prim: &SdfPrimitive) -> glam::Vec3 {
            match *prim {
                SdfPrimitive::Sphere { radius } => glam::Vec3::splat(radius),
                SdfPrimitive::Box { half_extents } => half_extents,
                SdfPrimitive::Capsule { radius, half_height } => {
                    glam::Vec3::new(radius, half_height + radius, radius)
                }
                SdfPrimitive::Torus { major_radius, minor_radius } => {
                    let r = major_radius + minor_radius;
                    glam::Vec3::new(r, minor_radius, r)
                }
                SdfPrimitive::Cylinder { radius, half_height } => {
                    glam::Vec3::new(radius, half_height, radius)
                }
                SdfPrimitive::Plane { .. } => glam::Vec3::splat(1.0),
            }
        }

        let half_extents = primitive_half_extents(primitive) * bake_scale;
        let margin = voxel_size * 8.0 * 1.8 + voxel_size;
        let aabb = rkp_core::Aabb::new(
            -half_extents - glam::Vec3::splat(margin),
            half_extents + glam::Vec3::splat(margin),
        );

        // SDF closure passed directly to the voxelizer. Negative = inside.
        let sdf_fn: Box<dyn Fn(glam::Vec3) -> f32> = match primitive {
            SdfPrimitive::Box { half_extents: he } => {
                let scaled = SdfPrimitive::Box { half_extents: *he * bake_scale };
                Box::new(move |pos| rkp_core::evaluate_primitive(&scaled, pos))
            }
            _ => {
                let prim = primitive.clone();
                let min_scale = bake_scale.x.min(bake_scale.y).min(bake_scale.z).max(1e-6);
                let inv_scale = glam::Vec3::new(
                    1.0 / bake_scale.x.max(1e-6),
                    1.0 / bake_scale.y.max(1e-6),
                    1.0 / bake_scale.z.max(1e-6),
                );
                Box::new(move |pos| rkp_core::evaluate_primitive(&prim, pos * inv_scale) * min_scale)
            }
        };

        // Batched callback: primitive SDF is CPU-only, so just loop.
        // `voxelize_octree`'s BFS hands us one call per level plus one
        // per terminal-geometry phase — the extra Vec allocations are
        // negligible next to the primitive evaluation cost.
        let sdf_batch = |positions: &[glam::Vec3]| -> Vec<(f32, u16, u16, u8, u32)> {
            // Single-material import path — secondary/blend left at 0,
            // so the shader's dual-material guard short-circuits.
            // Color = 0 = "no override, use material base color".
            positions
                .iter()
                .map(|p| (sdf_fn(*p), material_id, 0u16, 0u8, 0u32))
                .collect()
        };

        let r = rkp_core::voxelize_octree::voxelize_octree(
            sdf_batch, &aabb, voxel_size, &mut self.leaf_attr_pool, &mut self.brick_pool,
        )?;

        emit_faces(&r.octree, object_id, &mut self.pending_faces);
        self.faces_dirty = true;

        self.merge_face_links(&r.brick_face_links);
        let handle = self.octree.allocate(&r.octree);
        let spatial = rkp_core::scene_node::SpatialHandle::Octree {
            root_offset: handle.root_offset,
            len: handle.len,
            depth: handle.depth,
            base_voxel_size: handle.base_voxel_size,
        };

        let geometry_aabb = rkp_core::Aabb::new(-half_extents, half_extents);
        Some(VoxelizeResult {
            spatial,
            voxel_size,
            aabb: geometry_aabb,
            grid_origin: r.grid_origin,
            voxel_count: r.voxel_count,
            leaf_attr_slot_start: r.leaf_attr_slot_start,
            leaf_attr_slot_count: r.leaf_attr_unique_count,
            brick_ids: r.brick_ids,
        })
    }

    /// Voxelize an arbitrary SDF function into the octree.
    ///
    /// The closure takes a batch of positions and returns a parallel
    /// vec of `(signed_distance, primary_material, secondary_material,
    /// blend_weight_u4)` — one entry per input. Negative distance =
    /// inside. Pass `(secondary = 0, blend = 0)` for single-material
    /// voxelization; the shader's dual-material lerp is guarded behind
    /// `blend_weight > 0` so zero-blend voxels render identically to
    /// the old single-material path. The batched shape lets GPU-
    /// backed evaluators dispatch one compute pass per octree level.
    pub fn voxelize_sdf_fn<F>(
        &mut self,
        sdf_fn: F,
        aabb: &rkp_core::Aabb,
        voxel_size: f32,
        object_id: u32,
    ) -> Option<VoxelizeResult>
    where
        F: FnMut(&[glam::Vec3]) -> Vec<(f32, u16, u16, u8, u32)>,
    {
        self.bump_geometry_epoch();
        let r = rkp_core::voxelize_octree::voxelize_octree(
            sdf_fn, aabb, voxel_size, &mut self.leaf_attr_pool, &mut self.brick_pool,
        )?;

        emit_faces(&r.octree, object_id, &mut self.pending_faces);
        self.faces_dirty = true;

        self.merge_face_links(&r.brick_face_links);
        let handle = self.octree.allocate(&r.octree);
        let spatial = rkp_core::scene_node::SpatialHandle::Octree {
            root_offset: handle.root_offset,
            len: handle.len,
            depth: handle.depth,
            base_voxel_size: handle.base_voxel_size,
        };

        Some(VoxelizeResult {
            spatial,
            voxel_size,
            aabb: *aabb,
            grid_origin: r.grid_origin,
            voxel_count: r.voxel_count,
            leaf_attr_slot_start: r.leaf_attr_slot_start,
            leaf_attr_slot_count: r.leaf_attr_unique_count,
            brick_ids: r.brick_ids,
        })
    }

    /// Integrate a self-contained [`rkp_core::BakeArtifact`] (produced
    /// by `voxelize_to_artifact` on a worker thread against fresh
    /// private pools) into the shared scene pools. Remaps all
    /// worker-local leaf_attr IDs and brick IDs into the scene's global
    /// IDs, then runs the same tail that `voxelize_sdf_fn` does:
    /// `emit_faces`, `merge_face_links`, `octree.allocate`.
    pub fn integrate_artifact(
        &mut self,
        mut artifact: rkp_core::BakeArtifact,
        aabb: &rkp_core::Aabb,
        voxel_size: f32,
    ) -> Option<VoxelizeResult> {
        self.bump_geometry_epoch();
        use rkp_core::brick_face_links::{FACE_EMPTY, FACE_INTERIOR};
        use rkp_core::brick_pool::{BRICK_EMPTY, BRICK_INTERIOR};
        use rkp_core::sparse_octree::{
            brick_id as node_brick_id, is_brick, is_leaf, leaf_slot as node_leaf_slot,
            make_brick, make_leaf, INTERNAL_ATTR_NONE,
        };
        let t_start = std::time::Instant::now();

        // ── Leaf-attr pool: allocate a contiguous range, copy ──
        let n_attrs = artifact.leaf_attrs.len() as u32;
        let leaf_attr_slot_start = self
            .leaf_attr_pool
            .allocate_contiguous_bump(n_attrs)?;
        let t_attr_alloc = t_start.elapsed();
        for (i, attr) in artifact.leaf_attrs.iter().enumerate() {
            let scene_id = leaf_attr_slot_start + i as u32;
            *self.leaf_attr_pool.get_mut(scene_id) = *attr;
            let color = artifact.leaf_attr_colors[i];
            if color != 0 {
                self.leaf_attr_pool.set_color(scene_id, color);
            }
        }

        let t_attr_copy = t_start.elapsed();
        // ── Brick pool: allocate scene IDs, copy cells with leaf remap ──
        let n_bricks = artifact.brick_cells.len();
        let mut worker_to_scene_brick: Vec<u32> = Vec::with_capacity(n_bricks);
        let mut brick_ids_scene: Vec<u32> = Vec::with_capacity(n_bricks);
        let mut max_scene_brick: u32 = 0;
        for cells in &artifact.brick_cells {
            let scene_id = self.brick_pool.allocate()?;
            worker_to_scene_brick.push(scene_id);
            brick_ids_scene.push(scene_id);
            if scene_id > max_scene_brick {
                max_scene_brick = scene_id;
            }
            // Bulk-copy the cell slice, adding `leaf_attr_slot_start`
            // to every non-sentinel entry. A flat slice walk beats
            // 64 `set_cell` calls per brick — at millions of bricks
            // the overhead per cell dominates.
            let dst = self.brick_pool.brick_cells_mut(scene_id);
            debug_assert_eq!(dst.len(), cells.len());
            for (d, &c) in dst.iter_mut().zip(cells.iter()) {
                *d = if c == BRICK_EMPTY || c == BRICK_INTERIOR {
                    c
                } else {
                    leaf_attr_slot_start + c
                };
            }
        }

        let t_brick_copy = t_start.elapsed();
        // ── Octree node slice: remap leaf slots + brick IDs ──
        {
            let nodes = artifact.octree.as_slice_mut();
            for node in nodes.iter_mut() {
                let v = *node;
                if is_leaf(v) {
                    let worker_slot = node_leaf_slot(v);
                    *node = make_leaf(leaf_attr_slot_start + worker_slot);
                } else if is_brick(v) {
                    let worker_id = node_brick_id(v);
                    *node = make_brick(worker_to_scene_brick[worker_id as usize]);
                }
                // EMPTY_NODE / INTERIOR_NODE / branch pointers pass through.
            }
        }

        // ── Prefiltered internal attrs: remap parallel to nodes ──
        {
            let old = artifact.octree.internal_attr_slice().to_vec();
            let new: Vec<u32> = old
                .into_iter()
                .map(|v| if v == INTERNAL_ATTR_NONE { v } else { leaf_attr_slot_start + v })
                .collect();
            artifact.octree.set_internal_attr_index(new);
        }

        let t_octree_remap = t_start.elapsed();
        // ── Face links: remap indices + values into scene brick space ──
        // The scene-wide table is indexed by scene brick_id, so we
        // place each worker row at its remapped slot and pad the rest.
        if n_bricks > 0 {
            let mut scene_rows: Vec<[u32; 6]> =
                vec![[FACE_EMPTY; 6]; (max_scene_brick + 1) as usize];
            for (worker_id, row) in artifact.brick_face_links.iter().enumerate() {
                if worker_id >= n_bricks {
                    // `brick_face_links` is sized to max_worker_brick + 1
                    // which equals n_bricks since worker IDs are a dense
                    // 0..n range. Defensive against future changes.
                    break;
                }
                let scene_id = worker_to_scene_brick[worker_id];
                let mut remapped = [FACE_EMPTY; 6];
                for (i, &neighbor) in row.iter().enumerate() {
                    remapped[i] = if neighbor == FACE_EMPTY || neighbor == FACE_INTERIOR {
                        neighbor
                    } else {
                        worker_to_scene_brick[neighbor as usize]
                    };
                }
                scene_rows[scene_id as usize] = remapped;
            }
            self.merge_face_links(&scene_rows);
        }

        let t_face_links = t_start.elapsed();
        // NOTE: previously called `emit_faces` here to populate
        // `pending_faces`, but that Vec has no consumer anywhere in
        // the engine today (splat raster pipeline is retired). At
        // 5-10 M voxels the per-leaf 6-neighbor-lookup pass is
        // multi-second on the main thread for zero benefit. If a
        // face rasterizer comes back, resurrect this + re-wire the
        // consumer rather than routing unused work through every
        // bake.
        let handle = self.octree.allocate(&artifact.octree);
        let t_octree_alloc = t_start.elapsed();
        let ms = |d: std::time::Duration| d.as_secs_f32() * 1000.0;
        eprintln!(
            "[integrate_artifact] voxels={} bricks={} attrs={}  \
             attr_alloc={:.2}ms attr_copy={:.2}ms brick_copy={:.2}ms \
             octree_remap={:.2}ms face_links={:.2}ms \
             octree_alloc={:.2}ms total={:.2}ms",
            artifact.voxel_count,
            n_bricks,
            n_attrs,
            ms(t_attr_alloc),
            ms(t_attr_copy - t_attr_alloc),
            ms(t_brick_copy - t_attr_copy),
            ms(t_octree_remap - t_brick_copy),
            ms(t_face_links - t_octree_remap),
            ms(t_octree_alloc - t_face_links),
            ms(t_octree_alloc),
        );
        let spatial = rkp_core::scene_node::SpatialHandle::Octree {
            root_offset: handle.root_offset,
            len: handle.len,
            depth: handle.depth,
            base_voxel_size: handle.base_voxel_size,
        };

        Some(VoxelizeResult {
            spatial,
            voxel_size,
            aabb: *aabb,
            grid_origin: artifact.grid_origin,
            voxel_count: artifact.voxel_count,
            leaf_attr_slot_start,
            leaf_attr_slot_count: n_attrs,
            brick_ids: brick_ids_scene,
        })
    }

    /// Deallocate geometry previously produced by voxelize_*. Frees the
    /// octree, the leaf_attr range, and every brick that voxelization
    /// allocated. Bricks go through the bulk-batch path — the per-
    /// brick `BrickPool::deallocate` has a tail-coalesce loop that is
    /// O(n²) when the batch is a contiguous range (the common case
    /// when re-baking a procedural). Async bakes of 10M+ voxels had
    /// apply times of 5 s+ sitting in that loop; the batch path
    /// collapses it to milliseconds.
    pub fn deallocate_geometry(
        &mut self,
        spatial: &rkp_core::OctreeHandle,
        leaf_attr_slot_start: u32,
        leaf_attr_slot_count: u32,
        brick_ids: &[u32],
    ) {
        self.bump_geometry_epoch();
        self.octree.deallocate(*spatial);
        self.leaf_attr_pool.deallocate_range(leaf_attr_slot_start, leaf_attr_slot_count);
        self.brick_pool.deallocate_batch(brick_ids);
    }
}
