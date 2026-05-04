//! Per-frame user-shader pipeline tick.
//!
//! Two top-level functions:
//! - [`tick_instance_pipeline`] — Phase B-redux: bake user-shader
//!   prototypes into the host pool tail and register one
//!   `RkpGpuAsset` per registered instance shader so the band-cell
//!   descent path can resolve the prototype by `shader_id`.
//! - [`run_user_shader_geom`] — Phase C: run the BFS bake for every
//!   live `ShaderRegionRequest`, then return transient assets +
//!   instances for the march to find.
//!
//! Plus the two cache-key hashing helpers ([`topology_hash_for`] and
//! [`fill_hash_for`]) the latter relies on for `lookup_or_allocate`'s
//! dirty-bit accounting.

use crate::render_frame::RenderFrame;

use super::state::RenderState;

/// Phase B-redux per-frame tick — bake user-shader prototypes into the
/// host pool tail and register one [`rkp_render::rkp_gpu_object::RkpGpuAsset`]
/// per registered instance shader so the band-cell descent path can resolve
/// the prototype by `shader_id`. Phase 5 retired Option B's per-pixel
/// emit/cull/scatter pipeline; this is the surviving half.
///
/// Sequence (runs BEFORE `run_user_shader_geom` so the proto-pool
/// buffer reservation stacks cleanly with the user-shader-cache
/// reservation):
///   1. Reload the bake pipeline (idempotent on source-hash match).
///   2. `begin_frame` on the proto cache.
///   3. Dedup the band regions' shaders that need a baked prototype.
///   4. Snapshot `cpu_*_bytes` from scene_mgr.
///   5. Reserve `cpu + proto_max (+ phase_c_max)` on the scene
///      buffers. Re-upload geometry on realloc.
///   6. Configure proto pool bases.
///   7. Walk needed shaders, look up cache, queue dirty bakes,
///      register one [`rkp_render::rkp_gpu_object::RkpGpuAsset`] per shader.
///   8. Encode bake dispatches into a local encoder + submit.
///   9. `evict_untouched` to drop unreferenced cache entries.
pub(super) fn tick_instance_pipeline(
    state: &mut RenderState,
    frame: &RenderFrame,
) -> Vec<rkp_render::rkp_gpu_object::RkpGpuAsset> {
    use rkp_render::user_shader_proto_pass::{
        build_internal_levels, PrototypeUniform, MAX_PROTO_MAX_DEPTH,
        PROTO_TAIL_OCTREE_BYTES, PROTO_TAIL_BRICK_BYTES, PROTO_TAIL_LEAF_ATTR_BYTES,
    };
    use rkp_render::user_shader_pass::{
        BRICK_CELLS, MAX_GLOBAL_BRICKS, MAX_GLOBAL_LEAF_ATTRS, MAX_GLOBAL_OCTREE_NODES,
    };
    use rkp_render::rkp_gpu_object::{geom_type, RkpGpuAsset};
    use rkp_core::brick_pool::BRICK_DIM;

    // 1. Pipeline reload — cheap when source hash unchanged.
    state.instance_proto_pass.reload_user_shaders(
        &state.device,
        &frame.user_shader_proto_chunk,
        frame.user_shader_source_hash,
    );

    // 2. Mark cache untouched.
    state.instance_proto_cache.begin_frame();

    // 3. Dedup the band-region shaders that need a baked prototype.
    //    Phase B-redux: prototypes feed `descend_proto_octree` at march
    //    time on band-cell hits. One asset per registered instance shader.
    let mut needed: Vec<(u32, u32)> = Vec::new();
    let mut seen: std::collections::HashSet<u32> = std::collections::HashSet::new();
    for req in &frame.user_shader_regions {
        if !req.is_band_region { continue; }
        let Some(entry) = frame
            .user_shader_entries
            .iter()
            .find(|e| e.name == req.shader_name)
        else { continue; };
        if !seen.insert(entry.id) { continue; }
        let info = frame
            .user_shader_infos
            .iter()
            .find(|i| i.name == req.shader_name);
        let max_depth = info
            .and_then(|i| i.max_depth)
            .unwrap_or(2)
            .min(MAX_PROTO_MAX_DEPTH);
        needed.push((entry.id, max_depth));
    }
    if needed.is_empty() {
        state.instance_proto_cache.evict_untouched();
        return Vec::new();
    }

    // 4. Snapshot cpu_*_bytes from scene_mgr.
    let (cpu_octree_bytes, cpu_brick_bytes, cpu_leaf_attr_bytes, cpu_face_links_bytes) = {
        let sm = state.scene_mgr.lock().expect("scene_mgr poisoned");
        let g = sm.geometry_upload();
        (
            g.octree_nodes.len() as u64 * 8,
            g.brick_pool.len() as u64,
            g.leaf_attr_pool.len() as u64,
            g.brick_face_links.len() as u64,
        )
    };

    // 5. Reserve proto tail (and Phase C extras when active). Without
    //    the gate, sizing for Phase C's extras (~1 GB on the brick
    //    buffer at MAX_GLOBAL_BRICKS = 3M) breaches `max_buffer_size`
    //    on devices that don't support a full 2 GB binding.
    let phase_c_active = !frame.user_shader_regions.is_empty();
    let extra_octree: u64 = if phase_c_active {
        MAX_GLOBAL_OCTREE_NODES as u64 * 8
    } else { 0 };
    let extra_brick: u64 = if phase_c_active {
        MAX_GLOBAL_BRICKS as u64 * BRICK_CELLS as u64 * 4
    } else { 0 };
    let extra_leaf: u64 = if phase_c_active {
        MAX_GLOBAL_LEAF_ATTRS as u64 * 8
    } else { 0 };
    let extra_face_links: u64 = if phase_c_active {
        MAX_GLOBAL_BRICKS as u64 * 6 * 4
    } else { 0 };
    let proto_brick_count =
        (PROTO_TAIL_BRICK_BYTES / 4 / BRICK_CELLS as u64) as u32;
    let proto_face_links_bytes = (proto_brick_count as u64) * 6 * 4;
    let realloc = state.renderer.scene.ensure_pool_layout(
        &state.device,
        cpu_octree_bytes, PROTO_TAIL_OCTREE_BYTES, extra_octree,
        cpu_brick_bytes, PROTO_TAIL_BRICK_BYTES, extra_brick,
        cpu_leaf_attr_bytes, PROTO_TAIL_LEAF_ATTR_BYTES, extra_leaf,
        cpu_face_links_bytes, proto_face_links_bytes, extra_face_links,
    );
    if realloc {
        let sm = state.scene_mgr.lock().expect("scene_mgr poisoned");
        let g = sm.geometry_upload();
        state.renderer.upload_geometry(&state.queue, &g);
        state.last_uploaded_geometry_epoch = sm.geometry_epoch();
        drop(sm);
        const FACE_EMPTY: u32 = 0xFFFFFFFFu32;
        const FACE_INIT_CHUNK: usize = 4 * 1024 * 1024;
        let chunk_data: Vec<u32> = vec![FACE_EMPTY; FACE_INIT_CHUNK];
        let init_total = proto_face_links_bytes + extra_face_links;
        let mut written: u64 = 0;
        while written < init_total {
            let remaining = (init_total - written) as usize;
            let this_chunk_bytes = (FACE_INIT_CHUNK * 4).min(remaining);
            state.queue.write_buffer(
                &state.renderer.scene.brick_face_links_buffer,
                cpu_face_links_bytes + written,
                bytemuck::cast_slice(&chunk_data[..this_chunk_bytes / 4]),
            );
            written += this_chunk_bytes as u64;
        }
    }

    // 6. Configure proto pool bases — element units (octree slot,
    //    brick id, leaf-attr slot).
    let proto_octree_base_elems = (cpu_octree_bytes / 8) as u32;
    let proto_brick_base_bricks =
        (cpu_brick_bytes / 4 / BRICK_CELLS as u64) as u32;
    let proto_leaf_attr_base_elems = (cpu_leaf_attr_bytes / 8) as u32;
    let bases_changed = state.instance_proto_cache.set_pool_bases(
        proto_octree_base_elems,
        proto_brick_base_bricks,
        proto_leaf_attr_base_elems,
    );
    if bases_changed || realloc {
        state.instance_proto_pass.reset_cursors(
            &state.queue,
            proto_brick_base_bricks,
            proto_leaf_attr_base_elems,
        );
    }

    // 7. Walk needed shaders, queue dirty bakes, register one
    //    `RkpGpuAsset` per shader so the host march can resolve the
    //    prototype by linear scan on `shader_id`.
    struct DirtyBake {
        uniform: PrototypeUniform,
        max_depth: u32,
        octree_extent_offset: u32,
    }
    let mut dirty_bakes: Vec<DirtyBake> = Vec::new();
    let mut user_shader_assets: Vec<RkpGpuAsset> = Vec::new();
    for (shader_id, max_depth) in needed {
        let (proto_entry, proto_dirty) = match state.instance_proto_cache.lookup_or_allocate(
            shader_id,
            frame.user_shader_source_hash,
            max_depth,
        ) {
            Some(p) => p,
            None => {
                eprintln!(
                    "[inst] proto cache exhausted for shader_id {shader_id} \
                     — bake skipped this frame"
                );
                continue;
            }
        };
        if proto_dirty {
            dirty_bakes.push(DirtyBake {
                uniform: PrototypeUniform::from_entry(&proto_entry, &state.instance_proto_cache),
                max_depth: proto_entry.max_depth,
                octree_extent_offset: proto_entry.octree_extent.0,
            });
        }
        let extent = 1.0_f32; // prototype's local-space cube
        let voxel_size_local =
            extent / ((1u32 << proto_entry.max_depth) as f32 * BRICK_DIM as f32);
        user_shader_assets.push(RkpGpuAsset {
            aabb_min: [0.0, 0.0, 0.0],
            octree_root: proto_entry
                .octree_root(state.instance_proto_cache.pool_octree_base()),
            aabb_max: [extent, extent, extent],
            octree_depth: proto_entry.max_depth,
            octree_extent_bits: extent.to_bits(),
            voxel_size: voxel_size_local,
            geom_type: geom_type::VOXELIZED,
            bone_count: 0,
            grid_origin: [0.0, 0.0, 0.0],
            rest_octree_root: 0,
            rest_octree_depth: 0,
            rest_octree_extent_bits: 0,
            // shader_id != 0 — host march `march_object` will route
            // descent through the user shader's hooks.
            shader_id,
            _pad: 0,
        });
    }

    // 8. Encode bake dispatches.
    if !dirty_bakes.is_empty() {
        let mut encoder = state.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("inst bake"),
        });
        let bake_q = state.renderer.profiler.begin_query("inst_bake", &mut encoder);
        for bake in &dirty_bakes {
            // Pre-fill internal octree levels at the proto's reserved
            // offset within the host octree pool.
            let internal = build_internal_levels(
                proto_octree_base_elems,
                bake.octree_extent_offset,
                bake.max_depth,
            );
            let mut bytes: Vec<u8> = Vec::with_capacity(internal.len() * 8);
            for [v0, v1] in internal {
                bytes.extend_from_slice(&v0.to_le_bytes());
                bytes.extend_from_slice(&v1.to_le_bytes());
            }
            let octree_byte_offset =
                (proto_octree_base_elems as u64 + bake.octree_extent_offset as u64) * 8;
            state.queue.write_buffer(
                &state.renderer.scene.octree_nodes_buffer,
                octree_byte_offset,
                &bytes,
            );

            // Reset overflow only — brick + leaf-attr cursors are
            // PERSISTENT across bakes.
            state.queue.write_buffer(&state.instance_proto_pass.overflow_buffer, 0, &[0u8; 12 * 4]);

            state.queue.write_buffer(
                &state.instance_proto_pass.proto_uniform_buffer,
                0,
                bytemuck::bytes_of(&bake.uniform),
            );

            let bake_g0 = state.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("inst bake g0"),
                layout: &state.instance_proto_pass.group0_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: state.renderer.scene.octree_nodes_buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: state.renderer.scene.brick_pool_buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: state.renderer.scene.leaf_attr_pool_buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: state.instance_proto_pass.cursors_buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: state.instance_proto_pass.overflow_buffer.as_entire_binding() },
                ],
            });
            let bake_g1 = state.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("inst bake g1"),
                layout: &state.instance_proto_pass.group1_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: state.instance_proto_pass.proto_uniform_buffer.as_entire_binding(),
                }],
            });

            let bricks_per_axis = 1u32 << bake.max_depth;
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("inst bake"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&state.instance_proto_pass.bake_pipeline);
            cpass.set_bind_group(0, &bake_g0, &[]);
            cpass.set_bind_group(1, &bake_g1, &[]);
            cpass.dispatch_workgroups(bricks_per_axis, bricks_per_axis, bricks_per_axis);
        }
        state.renderer.profiler.end_query(&mut encoder, bake_q);
        state.queue.submit(Some(encoder.finish()));
    }

    // 9. Drop cache entries not referenced this frame.
    state.instance_proto_cache.evict_untouched();

    user_shader_assets
}

pub(super) fn run_user_shader_geom(
    state: &mut RenderState,
    frame: &RenderFrame,
    asset_id_base: u32,
) -> (
    Vec<rkp_render::rkp_gpu_object::RkpGpuAsset>,
    Vec<rkp_render::rkp_gpu_object::RkpGpuInstance>,
) {
    use rkp_render::user_shader_pass::{
        build_region_uniform, estimate_region_pool, resolve_shader_id, CachedSlot,
        RegionUniform, BRICK_CELLS, MAX_GLOBAL_BRICKS, MAX_GLOBAL_LEAF_ATTRS,
        MAX_GLOBAL_OCTREE_NODES, MAX_REGIONS,
    };
    use rkp_render::user_shader_proto_pass::{
        PROTO_TAIL_OCTREE_BYTES, PROTO_TAIL_BRICK_BYTES, PROTO_TAIL_LEAF_ATTR_BYTES,
    };

    const FACE_EMPTY: u32 = 0xFFFFFFFFu32;

    // 1. Pipeline reload — track the shade-side hash; the geom and
    //    shade chunks share the same `source_hash`.
    state.user_shader_pass.reload_user_shaders(
        &state.device,
        &frame.user_shader_generate_chunk,
        frame.user_shader_source_hash,
    );

    // 2. Mark cache entries untouched; we'll touch the ones we hit.
    state.user_shader_cache.begin_frame();

    if frame.user_shader_regions.is_empty() {
        // Nothing to dispatch — drop any entries left over from prior
        // frames so they release their pool extents.
        state.user_shader_cache.evict_untouched();
        return state.user_shader_cache.build_transient_assets_and_instances(asset_id_base);
    }

    // 3. Buffer reservation. Stable across frames once geometry is
    //    loaded. The user-shader transient tail is sized at the
    //    global caps; the cache sub-allocates within.
    let extra_octree: u64 = MAX_GLOBAL_OCTREE_NODES as u64 * 8;
    let extra_brick: u64 = MAX_GLOBAL_BRICKS as u64 * BRICK_CELLS as u64 * 4;
    let extra_leaf: u64 = MAX_GLOBAL_LEAF_ATTRS as u64 * 8;
    let extra_face_links: u64 = MAX_GLOBAL_BRICKS as u64 * 6 * 4;

    let need_regions = (frame.user_shader_regions.len() as u32).min(MAX_REGIONS);

    let (cpu_octree_bytes, cpu_brick_bytes, cpu_leaf_attr_bytes, cpu_face_links_bytes) = {
        let sm = state.scene_mgr.lock().expect("scene_mgr poisoned");
        let g = sm.geometry_upload();
        (
            g.octree_nodes.len() as u64 * 8,
            g.brick_pool.len() as u64,
            g.leaf_attr_pool.len() as u64,
            g.brick_face_links.len() as u64,
        )
    };
    // Phase 4 — proto tail sits between CPU and Phase C transient.
    // The proto reservation matches `tick_instance_pipeline`'s; both
    // calls grow the buffer to the union, so order doesn't matter.
    // Phase C's brick range needs face_links covering its absolute
    // brick_ids; proto's bricks need face_links too, init'd to
    // FACE_EMPTY so the march cleanly exits proto bricks at boundaries.
    let proto_brick_count =
        (PROTO_TAIL_BRICK_BYTES / 4 / BRICK_CELLS as u64) as u32;
    let proto_face_links_bytes = (proto_brick_count as u64) * 6 * 4;
    let realloc = state.renderer.scene.ensure_pool_layout(
        &state.device,
        cpu_octree_bytes, PROTO_TAIL_OCTREE_BYTES, extra_octree,
        cpu_brick_bytes, PROTO_TAIL_BRICK_BYTES, extra_brick,
        cpu_leaf_attr_bytes, PROTO_TAIL_LEAF_ATTR_BYTES, extra_leaf,
        cpu_face_links_bytes, proto_face_links_bytes, extra_face_links,
    );
    if realloc {
        let sm = state.scene_mgr.lock().expect("scene_mgr poisoned");
        let g = sm.geometry_upload();
        state.renderer.upload_geometry(&state.queue, &g);
        state.last_uploaded_geometry_epoch = sm.geometry_epoch();
        drop(sm);
        // One-time face-links init: the user-shader BFS never writes
        // into this buffer but the march reads it for any
        // user-shader-allocated brick. Uninitialised values would jump
        // the DDA chain into stale brick_id=0. Also covers the proto
        // range — proto bake doesn't write face_links either, so leaving
        // them at FACE_EMPTY makes the host march cleanly exit
        // user-shader instance bricks at boundaries (cross-brick
        // navigation within a single instance is unsupported until a
        // follow-up fix populates them).
        let init_total_bytes = proto_face_links_bytes + extra_face_links;
        const FACE_INIT_CHUNK: usize = 4 * 1024 * 1024;
        let chunk_data: Vec<u32> = vec![FACE_EMPTY; FACE_INIT_CHUNK];
        let mut written: u64 = 0;
        while written < init_total_bytes {
            let remaining = (init_total_bytes - written) as usize;
            let this_chunk_bytes = (FACE_INIT_CHUNK * 4).min(remaining);
            state.queue.write_buffer(
                &state.renderer.scene.brick_face_links_buffer,
                cpu_face_links_bytes + written,
                bytemuck::cast_slice(&chunk_data[..this_chunk_bytes / 4]),
            );
            written += this_chunk_bytes as u64;
        }
    }

    // 4. Configure pool bases — flushes the cache if they changed.
    //    Also reconcile against the host's geometry epoch (any host
    //    geometry change invalidates every region's topology_hash).
    //    Phase C's transient range starts past the proto reservation
    //    so user-shader proto bricks and Phase C bricks have disjoint
    //    brick_ids (and disjoint face_links slots).
    let proto_octree_elems = (PROTO_TAIL_OCTREE_BYTES / 8) as u32;
    let proto_leaf_attr_elems = (PROTO_TAIL_LEAF_ATTR_BYTES / 8) as u32;
    let octree_base_elems = (cpu_octree_bytes / 8) as u32 + proto_octree_elems;
    let brick_base_bricks =
        (cpu_brick_bytes / 4 / BRICK_CELLS as u64) as u32 + proto_brick_count;
    let leaf_base_elems = (cpu_leaf_attr_bytes / 8) as u32 + proto_leaf_attr_elems;
    state.user_shader_cache.set_pool_bases(
        octree_base_elems, brick_base_bricks, leaf_base_elems,
    );
    state.user_shader_cache.reconcile_epoch(frame.geometry_epoch);

    // 5. Walk regions, look up cache, gather dirty ones into two
    //    contiguous groups: topology-dirty first, then fill-only.
    let mut topology_dirty_uniforms: Vec<RegionUniform> = Vec::new();
    let mut fill_only_uniforms: Vec<RegionUniform> = Vec::new();
    let mut topology_dirty_slots: Vec<CachedSlot> = Vec::new();
    let mut fill_only_slots: Vec<CachedSlot> = Vec::new();
    let mut max_max_depth: u32 = 0;
    let time_seconds = frame.shade_params_base.time;
    for req in frame.user_shader_regions.iter().take(need_regions as usize) {
        let shader_id = resolve_shader_id(&frame.user_shader_infos, &req.shader_name);
        if shader_id == 0 {
            continue;
        }
        let topology_hash = topology_hash_for(req, frame.geometry_epoch, frame.paint_epoch);
        let fill_hash = fill_hash_for(
            req,
            topology_hash,
            frame.user_shader_source_hash,
            time_seconds,
        );
        let estimate = estimate_region_pool(req);
        let mut slot = match state.user_shader_cache.lookup_or_allocate(
            req, topology_hash, fill_hash, &estimate,
        ) {
            Some(s) => s,
            None => continue,
        };
        if !slot.topology_dirty && !slot.fill_dirty {
            continue; // Skip entirely; cached GPU contents still valid.
        }
        max_max_depth = max_max_depth.max(slot.max_depth);
        if slot.topology_dirty {
            slot.region_index = topology_dirty_slots.len() as u32;
            // index will get bumped by fill_only's count below; we
            // patch the uniform's region_index after gathering.
            topology_dirty_slots.push(slot);
            topology_dirty_uniforms.push(
                build_region_uniform(req, &slot, shader_id, time_seconds),
            );
        } else {
            fill_only_slots.push(slot);
            fill_only_uniforms.push(
                build_region_uniform(req, &slot, shader_id, time_seconds),
            );
        }
    }

    let topology_dirty_count = topology_dirty_uniforms.len() as u32;
    // Fill-only regions live at indices [topology_dirty_count, total).
    // Their region_index in the dispatch uniform must reflect that.
    for (i, _slot) in fill_only_slots.iter().enumerate() {
        // No-op: region_index is implicit in array order; the WGSL
        // reads `regions[wid.y]`, where wid.y = topology_dirty_count + i.
        let _ = i;
    }

    let mut uniforms = topology_dirty_uniforms;
    uniforms.extend(fill_only_uniforms);

    if !uniforms.is_empty() {
        let mut encoder = state
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("user_shader_geom_encoder"),
            });
        state.user_shader_pass.dispatch_regions(
            &state.device,
            &state.queue,
            &mut encoder,
            &uniforms,
            topology_dirty_count,
            max_max_depth,
            &state.renderer.scene.octree_nodes_buffer,
            &state.renderer.scene.brick_pool_buffer,
            &state.renderer.scene.leaf_attr_pool_buffer,
            &state.renderer.scene.instance_overlay_buffer,
            state.renderer.scene.buffers_epoch(),
        );
        state.queue.submit(Some(encoder.finish()));
        state.user_shader_pass.submit_overflow_readback();
    }

    // 6. Drop entries not touched this frame; their extents go back
    //    to the bucket allocators' free lists.
    state.user_shader_cache.evict_untouched();

    state.user_shader_cache.build_transient_assets_and_instances(asset_id_base)
}

/// Hash inputs that affect classify (BFS topology). Unchanged
/// topology hash → skip classify dispatch for this region.
pub(super) fn topology_hash_for(
    req: &rkp_render::user_shader_pass::ShaderRegionRequest,
    geometry_epoch: u64,
    paint_epoch: u64,
) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    let prime = 0x100000001b3u64;
    let mix = |h: &mut u64, b: u8| {
        *h ^= b as u64;
        *h = h.wrapping_mul(prime);
    };
    for &b in &geometry_epoch.to_le_bytes() { mix(&mut h, b); }
    // Paint changes the host's leaf-slot → material mapping (via the
    // overlay), which is what the BFS host probe consults. Fold the
    // paint epoch in so any paint forces a re-bake; without this, a
    // paint that lands in an already-allocated overlay slot leaves
    // overlay_offset/count unchanged and the BFS uses last frame's
    // bake.
    for &b in &paint_epoch.to_le_bytes() { mix(&mut h, b); }
    for &b in &req.host_octree_root.to_le_bytes() { mix(&mut h, b); }
    for &b in &req.host_octree_depth.to_le_bytes() { mix(&mut h, b); }
    for &b in &req.host_octree_extent.to_le_bytes() { mix(&mut h, b); }
    for v in req.host_grid_origin.iter() {
        for &b in &v.to_le_bytes() { mix(&mut h, b); }
    }
    for row in req.host_inverse_world.iter() {
        for v in row.iter() {
            for &b in &v.to_le_bytes() { mix(&mut h, b); }
        }
    }
    for &b in &req.region_thickness.to_le_bytes() { mix(&mut h, b); }
    for &b in &req.max_depth.to_le_bytes() { mix(&mut h, b); }
    for v in req.aabb_min.iter().chain(req.aabb_max.iter()) {
        for &b in &v.to_le_bytes() { mix(&mut h, b); }
    }
    for &b in &req.cell_size.to_le_bytes() { mix(&mut h, b); }
    // Band-cell anchor projection y — invalidate cache when the
    // painted surface moves vertically.
    for &b in &req.host_surface_y.to_le_bytes() { mix(&mut h, b); }
    // Per-instance paint overlay slice — invalidate cache when paint
    // changes on the host instance (the overlay slice migrates).
    for &b in &req.host_overlay_offset.to_le_bytes() { mix(&mut h, b); }
    for &b in &req.host_overlay_count.to_le_bytes() { mix(&mut h, b); }
    h
}

/// Hash inputs that affect fill (per-cell shader output). Unchanged
/// fill hash AND unchanged topology hash → skip fill dispatch.
pub(super) fn fill_hash_for(
    req: &rkp_render::user_shader_pass::ShaderRegionRequest,
    topology_hash: u64,
    shader_source_hash: u64,
    time_seconds: f32,
) -> u64 {
    let mut h = topology_hash;
    let prime = 0x100000001b3u64;
    let mix = |h: &mut u64, b: u8| {
        *h ^= b as u64;
        *h = h.wrapping_mul(prime);
    };
    for &b in &shader_source_hash.to_le_bytes() { mix(&mut h, b); }
    for &b in &req.input_hash.to_le_bytes() { mix(&mut h, b); }
    for &b in &req.material_id.to_le_bytes() { mix(&mut h, b); }
    for &p in &req.params {
        for &b in &p.to_le_bytes() { mix(&mut h, b); }
    }
    if req.animated {
        for &b in &time_seconds.to_le_bytes() { mix(&mut h, b); }
    }
    h
}

#[cfg(test)]
mod tests {
    use super::*;
    use rkp_render::user_shader_pass::{ShaderRegionRequest, NO_TILE};

    fn base_request() -> ShaderRegionRequest {
        ShaderRegionRequest {
            host_object_id: 1,
            material_id: 7,
            shader_name: "grass".to_string(),
            params: vec![],
            aabb_min: [0.0; 3],
            aabb_max: [1.0; 3],
            cell_size: 0.04,
            input_hash: 0,
            animated: false,
            region_thickness: 0.5,
            max_depth: 5,
            painted_leaf_count: 64,
            host_octree_root: 0,
            host_octree_depth: 8,
            host_octree_extent: 8.0,
            host_grid_origin: [0.0; 3],
            host_inverse_world: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            tile_index: NO_TILE,
            is_band_region: true,
            host_surface_y: 0.5,
            host_overlay_offset: 0,
            host_overlay_count: 0,
        }
    }

    /// V1.1 regression — topology hash MUST differ on paint_epoch
    /// changes. Without this, paint that lands in an already-allocated
    /// overlay slot (offset/count unchanged, content updated) leaves
    /// the cached BFS bake serving stale paint state and the most
    /// recent paint goes invisible.
    #[test]
    fn topology_hash_invalidates_on_paint_epoch() {
        let req = base_request();
        let h1 = topology_hash_for(&req, 0, 0);
        let h2 = topology_hash_for(&req, 0, 1);
        assert_ne!(
            h1, h2,
            "paint_epoch must affect the topology hash; otherwise paint into existing \
             overlay slots leaves stale BFS bake."
        );
    }

    /// V1.1 regression — topology hash MUST differ when the host's
    /// overlay slice moves (new entity painted, slice reshuffled).
    /// The BFS host-material probe consumes overlay_offset/count to
    /// find painted material; a stale slice mis-points the probe.
    #[test]
    fn topology_hash_invalidates_on_overlay_slice_move() {
        let mut req = base_request();
        let baseline = topology_hash_for(&req, 0, 0);
        req.host_overlay_offset = 64;
        let moved_offset = topology_hash_for(&req, 0, 0);
        assert_ne!(baseline, moved_offset, "overlay_offset must affect hash");
        req.host_overlay_offset = 0;
        req.host_overlay_count = 32;
        let moved_count = topology_hash_for(&req, 0, 0);
        assert_ne!(baseline, moved_count, "overlay_count must affect hash");
    }

    /// Geometry epoch already had to invalidate (host octree topology
    /// changing means the BFS reads different cells); pin the
    /// behavior to keep the contract clear alongside the new fields.
    #[test]
    fn topology_hash_invalidates_on_geometry_epoch() {
        let req = base_request();
        let h1 = topology_hash_for(&req, 0, 0);
        let h2 = topology_hash_for(&req, 1, 0);
        assert_ne!(h1, h2, "geometry_epoch must affect the topology hash");
    }
}
