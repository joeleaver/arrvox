//! Per-frame proto bake tick.
//!
//! Bakes each registered user-shader's prototype octree (the
//! canonical [0,1]³ shape returned by `proto_sample_at(uvw)`) into
//! the shared host pool tail and registers one `RkpGpuAsset` per
//! shader so the host march can descend a baked proto when an
//! emitted blade instance points at it.
//!
//! The previous BFS classify+fill pass (`run_user_shader_geom`) was
//! deleted along with the band-cell descent path; emitted blades are
//! now real `RkpInstance`s sharing one of these proto assets, marched
//! through the standard `march_object` flow.
//!
//! `topology_hash_for` / `fill_hash_for` are kept (and re-exported)
//! for use by the new emit pass when it lands — they're the right
//! shape for the per-region cache the new pass will own.

use crate::render_frame::RenderFrame;

use super::state::RenderState;

/// Per-frame proto bake. Returns one `RkpGpuAsset` per registered
/// instance shader; emitted blades reference these by `asset_id`.
///
/// Sequence:
///   1. Reload the bake pipeline (idempotent on source-hash match).
///   2. `begin_frame` on the proto cache.
///   3. Dedup the shaders that need a baked proto (have an
///      `instance_at` hook in their parsed body).
///   4. Snapshot `cpu_*_bytes` from scene_mgr.
///   5. Reserve the proto tail on the shared host pool. Re-upload
///      geometry on realloc.
///   6. Configure proto pool bases.
///   7. Walk needed shaders, look up cache, queue dirty bakes,
///      register one `RkpGpuAsset` per shader.
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
    use rkp_render::rkp_gpu_object::{geom_type, RkpGpuAsset};
    use rkp_core::brick_pool::{BRICK_CELLS, BRICK_DIM};

    // 1. Pipeline reload — cheap when source hash unchanged.
    state.instance_proto_pass.reload_user_shaders(
        &state.device,
        &frame.user_shader_proto_chunk,
        frame.user_shader_source_hash,
    );

    // 2. Mark cache untouched.
    state.instance_proto_cache.begin_frame();

    // 3. Dedup the shaders that need a baked proto. A shader needs a
    //    proto exactly when it has an `instance_at` hook (the new
    //    emit pass will dispatch its hooks per painted leaf and write
    //    `RkpInstance`s pointing at the proto). Shaders with only
    //    `shade` or no hooks at all skip the bake.
    let mut needed: Vec<(u32, u32)> = Vec::new();
    let mut seen: std::collections::HashSet<u32> = std::collections::HashSet::new();
    for entry in &frame.user_shader_entries {
        if entry.instance_at_text.is_none() {
            continue;
        }
        if !seen.insert(entry.id) {
            continue;
        }
        let info = frame
            .user_shader_infos
            .iter()
            .find(|i| i.name == entry.name);
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

    // 5. Reserve the proto tail past the CPU-managed head. The 768 MB
    //    Phase-C transient extras are gone — the new path scatters
    //    blades into a separate `user_shader_instance_buffer`, not
    //    into a transient octree.
    let proto_brick_count =
        (PROTO_TAIL_BRICK_BYTES / 4 / BRICK_CELLS as u64) as u32;
    let proto_face_links_bytes = (proto_brick_count as u64) * 6 * 4;
    let realloc = state.renderer.scene.ensure_pool_layout(
        &state.device,
        cpu_octree_bytes, PROTO_TAIL_OCTREE_BYTES,
        cpu_brick_bytes, PROTO_TAIL_BRICK_BYTES,
        cpu_leaf_attr_bytes, PROTO_TAIL_LEAF_ATTR_BYTES,
        cpu_face_links_bytes, proto_face_links_bytes,
    );
    if realloc {
        let sm = state.scene_mgr.lock().expect("scene_mgr poisoned");
        let g = sm.geometry_upload();
        state.renderer.upload_geometry(&state.queue, &g);
        state.last_uploaded_geometry_epoch = sm.geometry_epoch();
        drop(sm);
        // The proto bake doesn't write face_links itself, but the host
        // march reads them for any brick the ray enters. Initialise the
        // proto range to FACE_EMPTY so cross-brick navigation in proto
        // bricks cleanly exits at boundaries.
        const FACE_EMPTY: u32 = 0xFFFFFFFFu32;
        const FACE_INIT_CHUNK: usize = 4 * 1024 * 1024;
        let chunk_data: Vec<u32> = vec![FACE_EMPTY; FACE_INIT_CHUNK];
        let mut written: u64 = 0;
        while written < proto_face_links_bytes {
            let remaining = (proto_face_links_bytes - written) as usize;
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
    //    `RkpGpuAsset` per shader.
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

/// Per-frame user-shader emit dispatch. Reads `frame.painted_leaves`,
/// builds the per-material → (shader_id, proto_asset_id) lookup, and
/// dispatches the emit pass into its own command encoder.
///
/// Caller passes:
///   - `proto_assets` — the result of `tick_instance_pipeline` (one
///     `RkpGpuAsset` per registered instance shader; `shader_id` field
///     identifies which shader each entry corresponds to).
///   - `proto_asset_id_base` — absolute asset index where `proto_assets`
///     start in the combined `assets[]` buffer (= `frame.gpu_assets.len()`).
///
/// Reset of `user_shader_instance_count_buffer` happens here too —
/// the emit pass atomically bumps it as it allocates slots.
pub(super) fn tick_emit_pass(
    state: &mut RenderState,
    frame: &RenderFrame,
    proto_assets: &[rkp_render::rkp_gpu_object::RkpGpuAsset],
    proto_asset_id_base: u32,
) {
    use rkp_render::user_shader_emit_pass::{EmitParams, MatToProto};

    // Reload pipeline if shader source changed (the composed `emit`
    // chunk needs to be spliced in).
    state.user_shader_emit_pass.reload_user_shaders(
        &state.device,
        &frame.user_shader_emit_chunk,
        frame.user_shader_source_hash,
    );

    // Always reset the count buffer — even when there are no leaves,
    // downstream readers (Task #10) need a clean 0.
    state
        .user_shader_emit_pass
        .reset_instance_count(&state.queue, &state.renderer.scene.user_shader_instance_count_buffer);

    if frame.painted_leaves.is_empty() {
        return;
    }

    // Build mat_to_proto. Indexed by material_id; entries default to
    // `(0, 0)` (= "no shader, no asset", emit-pass thread early-returns).
    // Sized to `materials.len()` so leaf.material_id can index directly.
    let mut mat_to_proto = vec![
        MatToProto { shader_id: 0, proto_asset_id: 0 };
        frame.materials.len()
    ];
    for (mat_id, mat) in frame.materials.iter().enumerate() {
        if mat.instance_shader_id == 0 {
            continue;
        }
        // Find the proto asset for this shader_id.
        let Some(idx) = proto_assets.iter().position(|a| a.shader_id == mat.instance_shader_id)
        else {
            continue;
        };
        mat_to_proto[mat_id] = MatToProto {
            shader_id: mat.instance_shader_id,
            proto_asset_id: proto_asset_id_base + idx as u32,
        };
    }

    state.user_shader_emit_pass.upload_mat_to_proto(
        &state.device,
        &state.queue,
        &mat_to_proto,
    );
    state.user_shader_emit_pass.upload_leaves(
        &state.device,
        &state.queue,
        &frame.painted_leaves,
    );

    let leaf_count = frame.painted_leaves.len() as u32;
    let instance_capacity = rkp_render::rkp_scene::USER_SHADER_INSTANCE_CAPACITY;
    state.user_shader_emit_pass.update_params(
        &state.queue,
        &EmitParams {
            leaf_count,
            instance_capacity,
            time: frame.shade_params_base.time,
            _pad0: 0,
        },
    );

    state.user_shader_emit_pass.ensure_bind_group(
        &state.device,
        &state.renderer.scene.user_shader_instance_buffer,
        &state.renderer.scene.user_shader_instance_count_buffer,
        &state.renderer.scene.user_shader_instance_aabbs_buffer,
        state
            .viewport_renderers
            .values()
            .next()
            .map(|vr| vr.shade.shader_params_buffer())
            .expect("at least one viewport"),
        state.renderer.scene.buffers_epoch(),
    );

    let mut encoder = state
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("user_shader_emit_encoder"),
        });
    let q = state.renderer.profiler.begin_query("user_shader_emit", &mut encoder);
    state.user_shader_emit_pass.dispatch(&mut encoder, leaf_count);
    state.renderer.profiler.end_query(&mut encoder, q);
    // Stage the count readback so we can verify the dispatch is
    // producing instances. Skip-if-busy keeps successive frames from
    // double-mapping the same staging buffer.
    let count_copied = state.user_shader_emit_pass.copy_count_for_readback(
        &mut encoder,
        &state.renderer.scene.user_shader_instance_count_buffer,
    );
    state.queue.submit(Some(encoder.finish()));
    if count_copied {
        state.user_shader_emit_pass.submit_count_readback();
    }
}
