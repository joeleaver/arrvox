//! Frame lifecycle — render-frame submission, result drain, and the
//! sim-thread tick loop.
//!
//! `tick_loop` owns the outer pacing + command-drain cycle that
//! `RkpEngine::spawn` launches on the engine thread.
//! `submit_render_frame` builds the per-frame `RenderFrame` snapshot
//! and ships it to the render worker; `drain_render_results` consumes
//! the corresponding `RenderResult`s on the return channel.




use super::picking_ops::collect_ghost_primitives;
use super::state::EngineState;

impl EngineState {
    /// Build a [`RenderFrame`] snapshot from current ECS / environment
    /// state and submit it to the render thread.
    ///
    /// Sim does no GPU work directly anymore — every per-frame thing the
    /// renderer used to read off `EngineState` is now packaged into a
    /// snapshot and shipped over `render_worker.inbox`. The render
    /// thread consumes, encodes, submits, and returns a
    /// [`RenderResult`] back via `render_worker.outbox` (which we drain
    /// in [`Self::drain_render_results`] called from the tick loop).
    ///
    /// Returns the CPU phases for this submission (setup vs. snapshot
    /// build vs. submit-handoff). The post-submit bucket reflects the
    /// time spent waiting for render-thread results, which is also a
    /// proxy for GPU backpressure.
    ///
    /// Originally a 700-line method that owned both the build *and* the
    /// GPU work. The latter migrated to [`crate::render_worker`]; what
    /// remains here is purely sim-side data assembly.
    pub(crate) fn submit_render_frame(&mut self) {
        use crate::viewport::ViewportId;
        let frame_start = std::time::Instant::now();

        // 0. Drain RenderResults that landed since last submit. The
        //    render thread runs on its own pace; the latest result it
        //    finished publishing carries the freshest pick decoding,
        //    cloud-sun atten, and GPU pass timings for us to fold back
        //    into sim state before we build the next snapshot.
        self.drain_render_results();

        // 0a. Material palette — built every tick and shipped in the
        //     snapshot. Render uploads every frame. Cheap (small Vec)
        //     and robust to snapshot drops; the old "ship only when
        //     dirty" pattern could lose the upload if its carrying
        //     snapshot was dropped by the newest-wins inbox before
        //     render saw it.
        let (materials, shader_params_slots) = {
            let registry = &self.user_shader_registry;
            // Only resolve shader_ids for shaders that actually have a
            // `shade` hook. The shade pass takes the PBR path when
            // `material.shader_id == 0`; if we resolved every name
            // unconditionally, a geom-only shader (no `shade` hook)
            // would dispatch into the default `shade_result_passthrough`
            // arm and write raw albedo (~1 nit) to the HDR output, which
            // tone-maps to black against ~90 000-nit sun lighting. The
            // geom pass resolves shader_ids separately by name, so this
            // restriction doesn't affect it.
            let palette = self.material_lib.build_palette(&|name| {
                registry
                    .entries()
                    .iter()
                    .find(|e| e.name == name && e.shade_text.is_some())
                    .map(|e| e.id)
            });
            let params = self.material_lib.build_shader_params(registry);
            (palette, params)
        };
        // Compose the shade-pass chunk once per tick. Cheap (small
        // string) and the render thread compares the hash to skip
        // pipeline rebuilds when nothing changed.
        let composed = rkp_render::shader_composer::compose(&self.user_shader_registry);
        let user_shader_shade_chunk = composed.shade;
        let user_shader_generate_chunk = composed.generate;
        let user_shader_proto_chunk = composed.proto;
        let user_shader_emit_chunk = composed.emit;
        let user_shader_inst_to_local_chunk = composed.inst_to_local;
        let user_shader_inst_aabb_chunk = composed.inst_aabb;
        let user_shader_instance_at_chunk = composed.instance_at;
        let user_shader_source_hash = self.user_shader_registry.source_hash();
        let user_shader_infos = self.user_shader_registry.shader_infos();
        // Full registry entries for the Option B prototype lookup. The
        // render thread runs `flatten_prototype_lookup` against these.
        // Cheap to clone — registry rarely has more than a handful of
        // entries.
        let user_shader_entries =
            self.user_shader_registry.entries().to_vec();
        // Phase C V6 — auto ECS scan with per-leaf material support.
        // Scans each entity's leaf_attr range to find which generate-
        // hook-equipped materials are present (entity-level fallback
        // material, painted leaves, or both). Emits one region per
        // (entity, painted-material). Cached on (paint_epoch,
        // geometry_epoch) so the scan only runs when leaf data
        // changed.
        let mut user_shader_regions: Vec<rkp_render::user_shader_pass::ShaderRegionRequest> =
            Vec::with_capacity(self.user_shader_regions.len());
        // Manual registrations come first (engine-command path; empty
        // unless something explicitly populates it).
        user_shader_regions.extend(self.user_shader_regions.iter().cloned());
        // Stage 6d — Option B instance shader region requests. Same
        // (entity × painted-material) walk as `user_shader_regions`,
        // partitioned by shader kind in the emit loop below.
        let mut instance_region_requests: Vec<
            rkp_render::user_shader_emit_pass::InstanceRegionRequest,
        > = Vec::new();
        let infos = self.user_shader_registry.shader_infos();

        // Build the set of "shader-bearing material ids" — materials
        // whose shader has either a `generate` hook (Phase C per-cell
        // pipeline) or an `is_instance_pipeline` flag (Option B voxel
        // sprite instancing). Same painted-AABB scan feeds both
        // request types; the per-tile emit loop below partitions on
        // shader kind.
        let mut shader_materials: std::collections::HashMap<
            u16,
            rkp_render::shader_composer::UserShaderInfo,
        > = std::collections::HashMap::new();
        if infos.iter().any(|i| i.has_generate || i.is_instance_pipeline) {
            for slot_id in 0..self.material_lib.slot_count() as u16 {
                let Some(def) = self.material_lib.get_def(slot_id) else { continue; };
                let Some(shader_name) = def.shader.as_deref() else { continue; };
                let Some(info) = infos.iter().find(|i| i.name == shader_name) else { continue; };
                if info.has_generate || info.is_instance_pipeline {
                    shader_materials.insert(slot_id, info.clone());
                }
            }
        }

        if !shader_materials.is_empty() {
            // Reconcile the per-entity painted-material cache against
            // current paint + geometry epochs. Both bump on any
            // leaf-attr write (paint stroke, voxelize, asset load),
            // so a single equality check covers all invalidation.
            let (cur_paint, cur_geom) = {
                let sm = self.scene_mgr.lock().expect("scene_mgr poisoned");
                (sm.paint_epoch(), sm.geometry_epoch())
            };
            if cur_paint != self.painted_materials_paint_epoch
                || cur_geom != self.painted_materials_geometry_epoch
            {
                use crate::components::{Renderable, Transform};
                self.painted_materials.clear();
                let sm = self.scene_mgr.lock().expect("scene_mgr poisoned");
                let octree_data = sm.octree.data();
                let brick_pool_data = sm.brick_pool.as_slice();
                for (entity, _) in self
                    .world
                    .query::<(&Renderable, &Transform)>()
                    .iter()
                {
                    let Ok(r) = self.world.get::<&Renderable>(entity) else { continue; };
                    let Some(spatial) = &r.spatial else { continue; };
                    let Some(&gpu_idx) = self.entity_to_gpu.get(&entity) else { continue; };
                    let object_id = gpu_idx as u32;
                    // Walk the entity's octree to build per-material
                    // bounding boxes for leaves whose material has a
                    // generate-hook shader. The AABB lets the
                    // region request size itself tightly, so painting
                    // grass on one ear doesn't grass-ify the whole
                    // elephant.
                    let mut mat_tiles: std::collections::HashMap<
                        u16,
                        std::collections::HashMap<[i32; 3], super::state::PaintedTileEntry>,
                    > = std::collections::HashMap::new();
                    // World matrix for transforming leaf-local positions
                    // and normals into the world frame the emit pass
                    // hands to user shaders. Entries without a matching
                    // GPU object skip the leaf collection (mat_tiles
                    // still gets the AABB so Phase C still works).
                    let entity_world: Option<glam::Mat4> = self
                        .gpu_instances
                        .iter()
                        .find(|i| i.object_id == object_id)
                        .map(|i| glam::Mat4::from_cols_array_2d(&i.world));
                    scan_painted_aabbs(
                        octree_data,
                        brick_pool_data,
                        &sm.leaf_attr_pool,
                        self.paint_overlays.get(&entity),
                        spatial.root_offset as usize,
                        spatial.depth,
                        spatial.grid_origin,
                        spatial.base_voxel_size,
                        &shader_materials,
                        entity_world,
                        &mut mat_tiles,
                    );
                    if !mat_tiles.is_empty() {
                        self.painted_materials.insert(object_id, mat_tiles);
                    }
                }
                drop(sm);
                self.painted_materials_paint_epoch = cur_paint;
                self.painted_materials_geometry_epoch = cur_geom;
            }
        }

        // Emit one region per (object, painted-material) pair. For
        // objects whose primary material itself has a generate hook,
        // that material lands in `painted_materials` from the scan
        // above (every leaf carries the entity-level fallback unless
        // explicitly painted), so we don't need a separate primary-
        // material code path.
        if !shader_materials.is_empty() {
            const HARD_DEPTH_CEIL: u32 = rkp_render::user_shader_pass::MAX_DEPTH;
            const DEFAULT_MAX_DEPTH: u32 = 8;
            for inst in self.gpu_instances.iter() {
                let Some(mat_tiles) = self.painted_materials.get(&inst.object_id) else { continue; };
                let asset = match self.gpu_assets.get(inst.asset_id as usize) {
                    Some(a) => a,
                    None => continue,
                };
                let world: glam::Mat4 = glam::Mat4::from_cols_array_2d(&inst.world);
                // Phase 2: inverse_world dropped from RkpGpuInstance.
                // Compute on demand for the user-shader region uniforms,
                // which carry their own host_inverse_world copy.
                let inverse_world = world.inverse().to_cols_array_2d();
                let host_voxelized = asset.geom_type
                    == rkp_render::rkp_gpu_object::geom_type::VOXELIZED;
                let host_octree_root = if host_voxelized {
                    asset.octree_root
                } else {
                    0xFFFFFFFFu32
                };
                let host_octree_extent = f32::from_bits(asset.octree_extent_bits);
                let host_voxel_size = if asset.octree_depth > 0 {
                    host_octree_extent / (4.0 * (1u32 << asset.octree_depth) as f32)
                } else {
                    1.0
                };
                for (&mat_id, tiles) in mat_tiles.iter() {
                    let Some(def) = self.material_lib.get_def(mat_id) else { continue; };
                    let Some(info) = shader_materials.get(&mat_id) else { continue; };
                    // V11 — adaptive region_thickness. If the shader
                    // declares a `blade_height` parameter (or a
                    // `region_thickness` parameter), use the
                    // material's value as the effective band — so
                    // the BFS classifier prunes cells beyond actual
                    // blade reach instead of the worst-case
                    // directive value. Falls back to the directive
                    // when neither param exists.
                    let effective_band = {
                        let from_param = info.params.iter()
                            .find(|p| p.name == "blade_height" || p.name == "region_thickness")
                            .and_then(|p| {
                                def.shader_params
                                    .get(&p.name)
                                    .and_then(|v| v.as_f64())
                                    .map(|v| v as f32)
                                    .or(Some(p.default))
                            });
                        // Add a small margin for blade tip + bend reach.
                        let with_margin = from_param.map(|h| h * 1.25 + 0.01);
                        // Never exceed the shader's declared @region_thickness
                        // — that's the buffer-reserved upper bound.
                        with_margin
                            .map(|b| b.min(info.region_thickness))
                            .unwrap_or(info.region_thickness)
                            .max(1e-3)
                    };
                    // Pack params in the shader's declared order.
                    let mut params: Vec<f32> = Vec::with_capacity(info.params.len());
                    for p in &info.params {
                        let v = def
                            .shader_params
                            .get(&p.name)
                            .and_then(|v| v.as_f64())
                            .map(|v| v as f32)
                            .unwrap_or(p.default);
                        params.push(v);
                    }
                    let max_depth_cap = info
                        .max_depth
                        .unwrap_or(DEFAULT_MAX_DEPTH)
                        .min(HARD_DEPTH_CEIL);
                    let shader_name = def
                        .shader
                        .as_deref()
                        .map(|s| s.to_string())
                        .unwrap_or_default();
                    let shader_name_for_request = shader_name.clone();

                    // Each tile in this (object, material) pair becomes
                    // one ShaderRegionRequest. For non-tiled shaders
                    // there's a single entry under NO_TILE_COORD.
                    for (&tile_coord, tile_entry) in tiles.iter() {
                        let local_aabb = &tile_entry.aabb;
                        let painted_leaf_count = tile_entry.leaf_count;

                        // Determine the host-local AABB this region
                        // covers. For tiled shaders, it's the tile
                        // cube `[i*s, (i+1)*s]^3` (independent of
                        // paint extent — that's the whole point).
                        // For non-tiled, it's the painted-leaf bounds
                        // (V9 behaviour).
                        let (lmin, lmax, region_extent_local, tile_index) = if tile_coord == NO_TILE_COORD {
                            let pad = effective_band;
                            let lmin = [
                                local_aabb.min.x - pad,
                                local_aabb.min.y - pad,
                                local_aabb.min.z - pad,
                            ];
                            let lmax = [
                                local_aabb.max.x + pad,
                                local_aabb.max.y + pad,
                                local_aabb.max.z + pad,
                            ];
                            let extent = (lmax[0] - lmin[0])
                                .max(lmax[1] - lmin[1])
                                .max(lmax[2] - lmin[2]);
                            (lmin, lmax, extent, rkp_render::user_shader_pass::NO_TILE)
                        } else {
                            let s = info.tile_size.unwrap_or(1.0).max(1e-4);
                            let pad = effective_band;
                            let lmin = [
                                tile_coord[0] as f32 * s - pad,
                                tile_coord[1] as f32 * s - pad,
                                tile_coord[2] as f32 * s - pad,
                            ];
                            let lmax = [
                                (tile_coord[0] + 1) as f32 * s + pad,
                                (tile_coord[1] + 1) as f32 * s + pad,
                                (tile_coord[2] + 1) as f32 * s + pad,
                            ];
                            (lmin, lmax, s + pad * 2.0, tile_coord)
                        };

                        // Transform local AABB to world AABB by
                        // projecting all 8 corners.
                        let local_corners = [
                            [lmin[0], lmin[1], lmin[2]],
                            [lmax[0], lmin[1], lmin[2]],
                            [lmin[0], lmax[1], lmin[2]],
                            [lmax[0], lmax[1], lmin[2]],
                            [lmin[0], lmin[1], lmax[2]],
                            [lmax[0], lmin[1], lmax[2]],
                            [lmin[0], lmax[1], lmax[2]],
                            [lmax[0], lmax[1], lmax[2]],
                        ];
                        let mut world_min = [f32::INFINITY; 3];
                        let mut world_max = [f32::NEG_INFINITY; 3];
                        for c in &local_corners {
                            let p = world.project_point3(glam::Vec3::new(c[0], c[1], c[2]));
                            for i in 0..3 {
                                world_min[i] = world_min[i].min(p[i]);
                                world_max[i] = world_max[i].max(p[i]);
                            }
                        }

                        // Depth derivation. With tiling the extent is
                        // FIXED (tile_size + thickness padding), so
                        // cell_size doesn't grow with paint area —
                        // the whole point. Without tiling we keep
                        // V9 behaviour where depth is derived from
                        // the painted-leaf extent.
                        let extent = (world_max[0] - world_min[0])
                            .max(world_max[1] - world_min[1])
                            .max(world_max[2] - world_min[2])
                            .max(region_extent_local);
                        let target_cell_size = info.cell_size.unwrap_or(host_voxel_size).max(1e-4);
                        let cells_per_axis_target = (extent / target_cell_size).max(1.0);
                        let bricks_per_axis_target = (cells_per_axis_target / 4.0).max(1.0);
                        let derived_depth = (bricks_per_axis_target.log2().ceil() as u32).max(0);
                        let max_depth = derived_depth.min(max_depth_cap);
                        let cell_size = (extent / (4.0 * (1u32 << max_depth) as f32)).max(1e-4);

                        // Input hash folds object/material/tile/transform.
                        let mut input_hash: u64 = 0xcbf29ce484222325;
                        let mix = |h: &mut u64, b: u8| {
                            *h ^= b as u64;
                            *h = h.wrapping_mul(0x100000001b3);
                        };
                        for &b in &inst.object_id.to_le_bytes() { mix(&mut input_hash, b); }
                        for &b in &(mat_id as u32).to_le_bytes() { mix(&mut input_hash, b); }
                        for &c in &tile_index { for &b in &c.to_le_bytes() { mix(&mut input_hash, b); } }
                        for col in &inst.world {
                            for &v in col { for &b in &v.to_le_bytes() { mix(&mut input_hash, b); } }
                        }

                        // Centered cube around the world AABB so the
                        // brick grid extent is uniform across axes.
                        let world_center = [
                            0.5 * (world_min[0] + world_max[0]),
                            0.5 * (world_min[1] + world_max[1]),
                            0.5 * (world_min[2] + world_max[2]),
                        ];
                        let half_extent = extent * 0.5;
                        let centered_min = [
                            world_center[0] - half_extent,
                            world_center[1] - half_extent,
                            world_center[2] - half_extent,
                        ];
                        let centered_max = [
                            world_center[0] + half_extent,
                            world_center[1] + half_extent,
                            world_center[2] + half_extent,
                        ];

                        // Partition by shader kind. A single material's
                        // shader is one or the other (a generate hook
                        // and an instance pipeline are mutually
                        // exclusive at registration time), so each
                        // tile produces exactly one request.
                        if info.has_generate {
                            user_shader_regions.push(rkp_render::user_shader_pass::ShaderRegionRequest {
                                host_object_id: inst.object_id,
                                material_id: mat_id as u32,
                                shader_name: shader_name_for_request.clone(),
                                params: params.clone(),
                                aabb_min: centered_min,
                                aabb_max: centered_max,
                                cell_size,
                                input_hash,
                                animated: info.animated,
                                region_thickness: effective_band,
                                max_depth,
                                painted_leaf_count,
                                host_octree_root,
                                host_octree_depth: asset.octree_depth,
                                host_octree_extent,
                                host_grid_origin: asset.grid_origin,
                                host_inverse_world: inverse_world,
                                tile_index,
                            });
                        } else if info.is_instance_pipeline {
                            let stride_u32 = info
                                .instance_struct_size
                                .map(|s| s.div_ceil(4))
                                .unwrap_or(8);
                            let leaves = tile_entry.leaves.clone();
                            // Phase 7b — deterministic slot allocation. Each
                            // painted leaf reserves `max_emits_per_thread`
                            // consecutive slots in `instance_pool` so the
                            // CPU TLAS builder can map slot K back to its
                            // owning painted leaf without atomicAdd
                            // permutation. Reservation = leaves × cap;
                            // unused per-thread slots stay zero (see emit
                            // pre-clear) and produce degenerate AABBs that
                            // the tile cull / shadow trace skip naturally.
                            let max_emits_per_thread =
                                info.max_emits_per_thread.unwrap_or(1).max(1);
                            let max_instances = (leaves.len() as u32)
                                .saturating_mul(max_emits_per_thread)
                                .max(1);
                            instance_region_requests.push(
                                rkp_render::user_shader_emit_pass::InstanceRegionRequest {
                                    host_object_id: inst.object_id,
                                    material_id: mat_id as u32,
                                    shader_name: shader_name_for_request.clone(),
                                    params: params.clone(),
                                    aabb_min: centered_min,
                                    aabb_max: centered_max,
                                    cell_size,
                                    input_hash,
                                    animated: info.animated,
                                    region_thickness: effective_band,
                                    tile_index,
                                    stride_u32,
                                    max_instances,
                                    max_emits_per_thread,
                                    host_octree_root,
                                    host_octree_depth: asset.octree_depth,
                                    host_octree_extent,
                                    host_grid_origin: asset.grid_origin,
                                    host_inverse_world: inverse_world,
                                    leaves,
                                },
                            );
                        }
                    }
                }
            }
        }
        // Clear the dirty flag so any other consumers (UI, etc.)
        // know the palette they observed has been published. We
        // ship every tick regardless, so the flag is purely for
        // outside-of-render bookkeeping now.
        self.material_lib.clear_dirty();

        // MAIN camera first: atmosphere LUTs + sun-light tinting both
        // depend on its altitude (scene-wide values shared across VRs).
        let main_cam = self.build_camera_uniforms(ViewportId::MAIN);
        let cam_y = main_cam.position[1];

        // Cloud-sun atten: smooth toward the latest render-thread
        // readback (fed in via `last_cloud_sun_atten_raw` by
        // `drain_render_results`). NaN sentinel = render hasn't
        // published one yet (first frame, MAIN hidden), so we hold the
        // last EMA target.
        let target_atten = if self.environment.attenuate_sun_by_clouds
            && self.environment.clouds_enabled
        {
            if self.last_cloud_sun_atten_raw.is_nan() {
                self.cloud_sun_atten
            } else {
                self.last_cloud_sun_atten_raw
            }
        } else {
            1.0
        };
        self.cloud_sun_atten += (target_atten - self.cloud_sun_atten) * 0.04;

        // Sun + entity-driven point/spot lights, all in the order the
        // shade shader expects (entry 0 = sun).
        let mut sun_light = self.environment.to_gpu_light(cam_y);
        sun_light.color[0] *= self.cloud_sun_atten;
        sun_light.color[1] *= self.cloud_sun_atten;
        sun_light.color[2] *= self.cloud_sun_atten;
        let mut gpu_lights = vec![sun_light];
        for (_entity, (transform, pl)) in self
            .world
            .query::<(&crate::components::Transform, &crate::components::PointLight)>()
            .iter()
        {
            gpu_lights.push(rkp_render::rkp_shade::GpuLight {
                position: [transform.position.x, transform.position.y, transform.position.z, 1.0],
                color: [pl.color[0], pl.color[1], pl.color[2], pl.intensity],
                direction: [0.0, 0.0, 0.0, 0.0],
                params: [pl.range, 0.0, 0.0, if pl.cast_shadow { 1.0 } else { 0.0 }],
            });
        }
        for (_entity, (transform, sl)) in self
            .world
            .query::<(&crate::components::Transform, &crate::components::SpotLight)>()
            .iter()
        {
            gpu_lights.push(rkp_render::rkp_shade::GpuLight {
                position: [transform.position.x, transform.position.y, transform.position.z, 2.0],
                color: [sl.color[0], sl.color[1], sl.color[2], sl.intensity],
                direction: [
                    sl.direction.x,
                    sl.direction.y,
                    sl.direction.z,
                    sl.outer_angle.to_radians(),
                ],
                params: [
                    sl.range,
                    sl.inner_angle.to_radians(),
                    0.0,
                    if sl.cast_shadow { 1.0 } else { 0.0 },
                ],
            });
        }

        let mut shade_params = self.environment.to_shade_params(cam_y);
        shade_params.num_lights = gpu_lights.len() as u32;
        // Engine clock for user shaders that need a time input (hologram
        // scroll, fresnel pulse). Frame-index based at 60 Hz — same
        // convention used elsewhere (cloud_params). Wraps at ~414 days.
        shade_params.time = self.frame_index as f32 / 60.0;
        self.shade_params_base = shade_params;
        self.num_lights_cache = shade_params.num_lights;

        // Env update — shipped every tick (cheap; render writes a few
        // u32-sized queue.write_buffers). Same drop-safety rationale
        // as `materials`.
        let env_update = crate::render_frame::EnvUpdate {
            exposure: self.environment.exposure,
            bloom_threshold: self.environment.bloom_threshold,
            bloom_knee: self.environment.bloom_knee,
            bloom_intensity: self.environment.bloom_intensity,
        };
        // Clear the legacy flag for other consumers; render no longer
        // gates on it.
        self.environment_dirty = false;

        // 0c. Rebuild GPU objects from ECS world only when
        //     transforms/objects/membership changed.
        let gpu_objects_dirty_this_frame = self.gpu_objects_dirty;
        if self.gpu_objects_dirty {
            self.update_scene_gpu();
            self.gpu_objects_dirty = false;
        }

        let t_cpu_setup = frame_start.elapsed();

        // 1. Geometry epoch — read lock-free via the shared atomic
        //    handle. Render compares against its own last-uploaded
        //    epoch and re-uploads when behind. Robust to dropped
        //    snapshots: the next snapshot still carries the latest
        //    epoch, so render always catches up.
        //
        //    The lock-free read is what keeps sim at 60 Hz while
        //    bake_worker is busy — taking `scene_mgr.lock()` here
        //    would block sim for the full duration of any bake
        //    integrate (50 ms+).
        //
        //    The legacy `self.geometry_dirty` flag is kept for collider
        //    rebuild scheduling (independent of GPU upload). It's set
        //    by every code path that mutates scene geometry.
        let geometry_epoch = self
            .geometry_epoch_handle
            .load(std::sync::atomic::Ordering::Acquire);
        if self.geometry_dirty {
            self.collider_caches_dirty = true;
            self.geometry_dirty = false;
        }
        if self.collider_caches_dirty {
            self.rebuild_collider_caches();
            self.collider_caches_dirty = false;
        }

        // 2. Bone matrix bytes for shading (LBS + DQ paths).
        let bone_matrix_lbs = self.bone_matrix_allocator.bytes().to_vec();
        let bone_matrix_dqs = self.bone_matrix_allocator.bytes_dq().to_vec();

        // 2b. Skin scatter — fold per-entity dispatches into one
        //     batched compute dispatch sim-side; render fires the
        //     batch on its thread. `skin_reuse` short-circuits when
        //     every skinned pose was byte-identical to the previous
        //     frame (paused animation), in which case the bone_field
        //     buffer from last frame is still valid and the scatter
        //     can skip entirely.
        let skin = if self.skinning_enabled
            && !self.skin_dispatches.is_empty()
            && !self.skin_reuse
        {
            self.skin_batch.clear();
            for plan in &self.skin_dispatches {
                let d = rkp_render::SkinDispatch {
                    uniforms: plan.uniforms,
                    bricks: &plan.bricks,
                };
                self.skin_batch.push(&d);
            }
            Some(crate::render_frame::RenderSkin {
                bone_field_bytes: self.skin_bone_field_bytes,
                bone_field_occ_bytes: self.skin_bone_field_occ_bytes,
                batch: self.skin_batch.clone(),
            })
        } else {
            if self.skinning_enabled && self.frame_index % 60 == 0 {
                // Once a second, log why scatter isn't running when
                // the user has the toggle on — most common reason is
                // a stale `.rkp` without the new skin-meta section.
                let skinned_entities = self
                    .world
                    .query::<&crate::components::Skeleton>()
                    .iter()
                    .count();
                if skinned_entities > 0 {
                    eprintln!(
                        "[RkpEngine] skinning enabled, {} skinned entities, but 0 scatter dispatches this frame. \
                         Likely cause: stale .rkp without skin-meta section — re-import the asset.",
                        skinned_entities,
                    );
                }
            }
            None
        };

        // 3. Per-viewport snapshot build — derive every per-VR
        //    parameter the render thread needs from current sim state
        //    and stash it in `viewports` for the snapshot. No GPU
        //    calls; the render thread does all the actual encoding
        //    and submission against this data.
        let visible_ids: Vec<ViewportId> = self
            .viewports
            .iter()
            .filter(|(_, v)| v.visible)
            .map(|(id, _)| *id)
            .collect();

        // Gizmo overlay is drawn on MAIN only — selection state is global.
        let gizmo_verts_main = self.build_gizmo_wireframe();
        let mut vp_list: Vec<crate::render_frame::RenderViewport> =
            Vec::with_capacity(visible_ids.len());

        for &viewport_id in &visible_ids {
            let cam_uniforms = self.build_camera_uniforms(viewport_id);
            let (vp_w, vp_h) = self
                .viewports
                .get(viewport_id)
                .map(|v| (v.width, v.height))
                .expect("viewport must exist");

            // Per-viewport screen-AABBs (camera-dependent) for tile cull.
            let vp_matrix = glam::Mat4::from_cols_array_2d(&cam_uniforms.view_proj);
            let screen_aabbs = crate::scene_sync::compute_screen_aabbs(
                &self.gpu_instances,
                &self.gpu_assets,
                &vp_matrix,
                vp_w as f32,
                vp_h as f32,
            );
            let screen_aabbs_bytes: Vec<u8> = bytemuck::cast_slice(&screen_aabbs).to_vec();
            // Per-tile object lists — replaces the 32-object bitmask so
            // the march shader handles arbitrary scene object counts.
            let tile_lists = crate::scene_sync::build_tile_lists(
                &screen_aabbs, vp_w, vp_h,
            );
            let tile_offsets_bytes: Vec<u8> =
                bytemuck::cast_slice(&tile_lists.offsets).to_vec();
            let tile_object_ids_bytes: Vec<u8> =
                bytemuck::cast_slice(&tile_lists.object_ids).to_vec();
            let tile_count_x = tile_lists.tile_count_x;

            // Per-VR vol/cloud/atmo/god-ray params — derived from
            // environment + this VR's camera. Render writes them into
            // the corresponding per-VR uniform buffers right before
            // submit (one submit per VR keeps the writes correctly
            // paired with their dispatches).
            let vol_params = self.environment.to_volumetric_params(
                &cam_uniforms,
                vp_w,
                vp_h,
                self.frame_index as u32,
            );
            let cloud_params =
                self.environment.to_cloud_params(self.frame_index as f32 / 60.0);

            let sun_d = self.environment.sun_direction();
            let cam_y_vp = cam_uniforms.position[1];
            let atmo_frame = rkp_render::rkp_atmosphere::AtmosphereFrameParams {
                sun_dir: [-sun_d[0], -sun_d[1], -sun_d[2]],
                sun_intensity: self.environment.sun_intensity,
                camera_altitude: self.environment.effective_altitude(cam_y_vp),
                ground_albedo: self.environment.ground_albedo,
                cam_pos: [
                    cam_uniforms.position[0],
                    cam_uniforms.position[1],
                    cam_uniforms.position[2],
                ],
                _pad1b: 0.0,
                cam_forward: [
                    cam_uniforms.forward[0],
                    cam_uniforms.forward[1],
                    cam_uniforms.forward[2],
                ],
                _pad2: 0.0,
                cam_right: [
                    cam_uniforms.right[0],
                    cam_uniforms.right[1],
                    cam_uniforms.right[2],
                ],
                _pad3: 0.0,
                cam_up: [
                    cam_uniforms.up[0],
                    cam_uniforms.up[1],
                    cam_uniforms.up[2],
                ],
                _pad4: 0.0,
            };

            let god_ray_params = {
                let sun_toward = [-sun_d[0], -sun_d[1], -sun_d[2]];
                let sun_world = glam::Vec3::new(
                    cam_uniforms.position[0] + sun_toward[0] * 1000.0,
                    cam_uniforms.position[1] + sun_toward[1] * 1000.0,
                    cam_uniforms.position[2] + sun_toward[2] * 1000.0,
                );
                let clip = vp_matrix * glam::Vec4::new(sun_world.x, sun_world.y, sun_world.z, 1.0);
                let sun_on_screen = if clip.w > 0.0 { 1.0 } else { 0.0 };
                let ndc = if clip.w > 0.0 {
                    glam::Vec2::new(clip.x / clip.w, clip.y / clip.w)
                } else {
                    glam::Vec2::ZERO
                };
                let sun_uv = [ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5];
                rkp_render::rkp_god_rays::GodRayParams {
                    sun_screen_pos: sun_uv,
                    sun_on_screen,
                    density: self.environment.god_ray_density,
                    weight: self.environment.god_ray_weight,
                    decay: self.environment.god_ray_decay,
                    exposure: self.environment.god_ray_exposure,
                    num_samples: 64,
                    sun_color: self.environment.sun_tint(cam_y_vp),
                    _pad: 0.0,
                }
            };

            let (vp_mode, vp_preview_mode) = self
                .viewports
                .get(viewport_id)
                .map(|v| (v.mode, v.preview_mode))
                .unwrap_or((
                    rkp_render::RenderMode::InSitu,
                    rkp_render::BuildPreviewMode::Voxel,
                ));

            // The procedural being previewed in raymarch mode is
            // always the currently-selected entity — keeps the
            // preview following selection automatically.
            let vp_preview_entity = self.selected_entity.and_then(|entity| {
                if self
                    .world
                    .get::<&crate::components::ProceduralGeometry>(entity)
                    .is_ok()
                {
                    self.entity_uuids.get(&entity).copied()
                } else {
                    None
                }
            });

            // Per-VR shade params: same scene-wide values plus the
            // per-VR `isolation` flag and a clamp on the light count
            // when isolated (so the BUILD preview doesn't pick up
            // the main scene's point lights).
            let isolation = matches!(vp_mode, rkp_render::RenderMode::Isolation);
            let mut shade_params_vr = self.shade_params_base;
            shade_params_vr.isolation = isolation as u32;
            if isolation {
                shade_params_vr.num_lights = shade_params_vr.num_lights.min(1);
            }
            // Paint cursor overlay — MAIN only. The brush sphere is a
            // viewport-camera-centric tool; a BUILD preview of the
            // same scene shouldn't show it. When the engine isn't in
            // paint mode `brush_active` stays 0 and the shader skips
            // the overlay entirely.
            if viewport_id == ViewportId::MAIN
                && self.paint_mode_active
            {
                if let Some(center) = self.paint_cursor_world {
                    shade_params_vr.brush_active = 1;
                    shade_params_vr.brush_radius = self.paint_mode_radius;
                    shade_params_vr.brush_falloff = 0.5; // editor slider in Phase 5
                    shade_params_vr.brush_center = [center.x, center.y, center.z, 0.0];
                    // Color: warm yellow rim — distinct from the light-
                    // gizmo yellow the sphere placeholder used. Alpha
                    // channel reserved; the shader does its own alpha.
                    shade_params_vr.brush_color = [1.0, 0.85, 0.2, 1.0];
                }
            }
            let bloom_composite_intensity = if isolation {
                0.0
            } else {
                self.environment.bloom_intensity
            };

            // Procedural raymarch state — only when this VR is in
            // raymarch preview mode AND a procedural entity is
            // selected. Sim flattens the tree, builds the AABB, and
            // pre-filters ghost primitives; render uploads + binds.
            let proc_raymarch =
                if matches!(vp_preview_mode, rkp_render::BuildPreviewMode::Raymarch) {
                    let entity = vp_preview_entity.and_then(|uuid| {
                        self.entity_uuids
                            .iter()
                            .find_map(|(e, u)| (*u == uuid).then_some(*e))
                    });

                    let (instructions, aabb_min, aabb_max) = entity
                        .and_then(|e| {
                            self.world
                                .get::<&crate::components::ProceduralGeometry>(e)
                                .ok()
                                .map(|pg| {
                                    let ins = rkp_procedural::flatten_tree(&pg.tree);
                                    let bounds = rkp_procedural::compute_bounds(&pg.tree);
                                    (ins, bounds.min, bounds.max)
                                })
                        })
                        // Empty-AABB sentinel: -1..+1 degenerate box
                        // any sane ray-AABB slab test fails. Covers
                        // "raymarch enabled but no procedural entity
                        // selected" so we don't get a bogus hit.
                        .unwrap_or_else(|| {
                            (Vec::new(), glam::Vec3::splat(1.0), glam::Vec3::splat(-1.0))
                        });

                    // Any stable per-entity u32 works — the shader
                    // packs it into the material G channel for the
                    // (now-unused) old 8-bit pick byte; retained here
                    // only as a non-breaking placeholder until
                    // `ProcRaymarchParams.object_id` gets cleaned up.
                    let object_id = entity.map(|e| e.to_bits().get() as u32).unwrap_or(0);

                    let entity_world = entity
                        .and_then(|e| {
                            self.world
                                .get::<&crate::components::Transform>(e)
                                .ok()
                                .map(|xf| {
                                    glam::Affine3A::from_scale_rotation_translation(
                                        xf.scale,
                                        glam::Quat::from_euler(
                                            glam::EulerRot::XYZ,
                                            xf.rotation.x.to_radians(),
                                            xf.rotation.y.to_radians(),
                                            xf.rotation.z.to_radians(),
                                        ),
                                        xf.position,
                                    )
                                })
                        })
                        .unwrap_or(glam::Affine3A::IDENTITY);

                    // Ghost overlay: every cutter-role primitive,
                    // regardless of selection. Filter the flattened
                    // instruction stream so ghost renders use the
                    // same composed transforms the main raymarch does.
                    let ghost_ids = entity
                        .and_then(|e| {
                            self.world
                                .get::<&crate::components::ProceduralGeometry>(e)
                                .ok()
                                .map(|pg| collect_ghost_primitives(&pg.tree))
                        })
                        .unwrap_or_default();
                    let ghost_set: std::collections::HashSet<u32> =
                        ghost_ids.into_iter().collect();
                    let ghost_instructions: Vec<rkp_procedural::ProcInstruction> = instructions
                        .iter()
                        .filter(|ins| ghost_set.contains(&ins.node_id))
                        .copied()
                        .collect();

                    Some(crate::render_frame::RenderProcRaymarch {
                        instructions,
                        ghost_instructions,
                        object_id,
                        entity_world,
                        aabb_min,
                        aabb_max,
                        selected_node: self.selected_procedural_node,
                    })
                } else {
                    None
                };

            // Wireframe verts: gizmo on MAIN, procedural-node gizmo
            // on BUILD when in raymarch preview. The procedural-node
            // gizmo is only meaningful in raymarch mode — in voxel
            // mode the user sees the baked result and any drag would
            // silently edit the tree without visual feedback.
            let wireframe_verts = if viewport_id == ViewportId::MAIN {
                gizmo_verts_main.clone()
            } else if viewport_id == ViewportId::BUILD
                && matches!(vp_preview_mode, rkp_render::BuildPreviewMode::Raymarch)
            {
                let cam_pos = glam::Vec3::new(
                    cam_uniforms.position[0],
                    cam_uniforms.position[1],
                    cam_uniforms.position[2],
                );
                self.build_procedural_gizmo_wireframe(cam_pos)
            } else {
                Vec::new()
            };

            // Editor-overlay gate. MAIN gates on the EDITOR_ONLY
            // layer bit (off in play mode); BUILD always shows its
            // proc-gizmo when one's present.
            let show_editor_overlays = if viewport_id == ViewportId::MAIN {
                self.viewports
                    .get(ViewportId::MAIN)
                    .map(|v| v.filter.base_layers & crate::viewport::layer::EDITOR_ONLY != 0)
                    .unwrap_or(false)
            } else {
                true
            };

            // BUILD: pin the studio-floor grid under the previewed
            // entity instead of world origin. Without this, moving
            // the entity in world-Y leaves the grid at y=0 while the
            // camera orbits around the entity, so the object floats
            // relative to the grid.
            let grid_override = if viewport_id == ViewportId::BUILD {
                let p = proc_raymarch
                    .as_ref()
                    .map(|p| p.entity_world.translation)
                    .unwrap_or(glam::Vec3A::ZERO);
                Some(rkp_render::rkp_grid::GridParams {
                    plane_origin: [p.x, p.y, p.z, 0.0],
                    ..Default::default()
                })
            } else {
                None
            };

            vp_list.push(crate::render_frame::RenderViewport {
                id: viewport_id,
                width: vp_w,
                height: vp_h,
                mode: vp_mode,
                preview_mode: vp_preview_mode,
                camera: cam_uniforms,
                screen_aabbs_bytes,
                tile_offsets_bytes,
                tile_object_ids_bytes,
                tile_count_x,
                vp_matrix,
                vol_params,
                cloud_params,
                atmo_frame,
                god_ray_params,
                shade_params: shade_params_vr,
                bloom_composite_intensity,
                grid_override,
                wireframe_verts,
                show_editor_overlays,
                proc_raymarch,
            });

            // Update sim-side `prev_view_proj` so next frame's
            // CameraUniforms carry the right reprojection matrix for
            // cloud TAA / temporal upscale.
            if let Some(v) = self.viewports.get_mut(viewport_id) {
                v.prev_view_proj = cam_uniforms.view_proj;
            }
        }

        // 4. Pending pick — convert sim's `PendingPick` (which carries
        //    a CPU-resolved ghost hint) to the render-side struct.
        //    Ghost hint stays sim-side; we'll re-apply it when the
        //    matching `PickResult` comes back.
        //
        //    Re-ship every snapshot until [`process_pick_result`]
        //    clears `self.pending_pick`. Picks used to be cleared
        //    eagerly with `take()`, but the GPU-backpressure backoff
        //    in `render_worker` now causes the inbox (newest-wins) to
        //    drop a sizeable fraction of snapshots before render sees
        //    them — eager-clearing meant the click was lost forever
        //    whenever its carrier snapshot got dropped. Re-shipping
        //    is safe because render's `pick_in_flight` gate dedupes
        //    duplicates: at most one map_async is ever in flight per
        //    pick request.
        let pending_pick = if let Some(pp) = self.pending_pick.as_ref() {
            // Map viewport+preview-mode → kind. BUILD raymarch decodes
            // the gbuf_pick texture for procedural NodeIds; everything
            // else (MAIN voxel, BUILD voxel) decodes gbuf_material for
            // the entity scene_id.
            let kind = if pp.viewport == ViewportId::BUILD
                && self
                    .viewports
                    .get(ViewportId::BUILD)
                    .map(|v| matches!(v.preview_mode, rkp_render::BuildPreviewMode::Raymarch))
                    .unwrap_or(false)
            {
                crate::render_frame::PickKind::ProceduralNode
            } else {
                crate::render_frame::PickKind::Material
            };
            self.in_flight_pick_ghost = pp.ghost_pick_node_id;
            Some(crate::render_frame::PendingPick {
                viewport: pp.viewport,
                x: pp.x,
                y: pp.y,
                kind,
            })
        } else {
            None
        };

        // 5. Build + submit the snapshot. `submit` is non-blocking;
        //    if render hadn't consumed the previous snapshot yet,
        //    that one is dropped (newest-wins). Sim never stalls on
        //    render's GPU rate.
        let brush_overlay_epoch = self
            .brush_overlay_epoch_handle
            .load(std::sync::atomic::Ordering::Acquire);
        let paint_epoch = self
            .paint_epoch_handle
            .load(std::sync::atomic::Ordering::Acquire);

        let frame = crate::render_frame::RenderFrame {
            frame_index: self.frame_index,
            gpu_assets: self.gpu_assets.clone(),
            gpu_instances: self.gpu_instances.clone(),
            gpu_instance_overlays: self.gpu_instance_overlays.clone(),
            gpu_objects_dirty: gpu_objects_dirty_this_frame,
            geometry_epoch,
            brush_overlay_epoch,
            paint_epoch,
            materials,
            shader_params_slots,
            user_shader_shade_chunk,
            user_shader_source_hash,
            user_shader_generate_chunk,
            user_shader_proto_chunk,
            user_shader_emit_chunk,
            user_shader_inst_to_local_chunk,
            user_shader_inst_aabb_chunk,
            user_shader_instance_at_chunk,
            user_shader_infos,
            user_shader_entries,
            user_shader_regions,
            // Stage 6d — Option B instance pipeline. Built above by
            // the same painted-AABB walk as `user_shader_regions`,
            // partitioned on shader kind.
            instance_region_requests,
            lights: gpu_lights,
            shade_params_base: self.shade_params_base,
            env_update,
            viewports: vp_list,
            skin,
            bone_matrix_lbs,
            bone_matrix_dqs,
            pending_pick,
            cloud_sun_atten: self.cloud_sun_atten,
            lod_enabled: self.lod_enabled,
            surfacenet_enabled: self.surfacenet_enabled,
            shadow_steps: self.environment.shadow_steps,
        };

        let t_encode = frame_start.elapsed();
        self.render_worker.inbox.submit(frame);
        let t_frame_end = frame_start.elapsed();

        // 6. Push CPU-side timings into profiling history. GPU pass
        //    timings get stitched into the most-recent sample by
        //    `drain_render_results` once the render thread publishes
        //    them (typically 1-2 frames behind sim).
        let ms = |d: std::time::Duration| d.as_secs_f32() * 1000.0;
        let cpu = crate::profiling::CpuPhaseTimings {
            setup_ms: ms(t_cpu_setup),
            snapshot_ms: ms(t_encode - t_cpu_setup),
            submit_ms: ms(t_frame_end - t_encode),
            total_ms: ms(t_frame_end),
        };
        self.profiling.push(crate::profiling::FrameSample {
            frame_idx: self.frame_index,
            cpu,
            // Both filled in by `drain_render_results` once the render
            // thread publishes the matching frame's `RenderResult`.
            // Lag is typically 1-2 frames, fine for display.
            gpu_passes: Vec::new(),
            render_dt_ms: 0.0,
            gpu_object_count: self.gpu_instances.len() as u32,
        });

        self.frame_index += 1;
    }

}

/// Walk an entity's octree (rooted at `root_offset` inside the
/// global packed `octree_data` buffer) and accumulate, per
/// shader-bearing material, the object-local AABB of leaves with
/// that material. Used by the per-leaf-material auto-scan to size
/// the geom-pipeline region tightly.
///
/// `octree_data` is the absolute-rebased packed buffer; branches
/// store offsets directly into this slice. Bricks are flattened
/// further — for each brick we walk its 64 cells in `brick_pool` and
/// look up cell leaf-attrs. Leaves at higher levels (shallow trees
/// without bricks) cover a 2^(depth-leaf_level) cube of voxel cells.
/// Resolve the effective `LeafAttr` for `slot` on a specific instance —
/// overlay if present (Phase 3), else the asset's shared pool. Mirrors
/// `fetch_leaf_attr_for` in WGSL.
#[inline]
fn resolve_leaf_attr(
    overlay: Option<&rkp_core::LeafAttrOverlay>,
    leaf_attrs: &rkp_core::LeafAttrPool,
    slot: u32,
) -> rkp_core::LeafAttr {
    if let Some(o) = overlay {
        if let Some(e) = o.get(slot) {
            return e.attr();
        }
    }
    *leaf_attrs.get(slot)
}

fn scan_painted_aabbs(
    octree_data: &[u32],
    brick_pool: &[u32],
    leaf_attrs: &rkp_core::LeafAttrPool,
    overlay: Option<&rkp_core::LeafAttrOverlay>,
    root_offset: usize,
    depth: u8,
    grid_origin: glam::Vec3,
    base_voxel_size: f32,
    shader_materials: &std::collections::HashMap<u16, rkp_render::shader_composer::UserShaderInfo>,
    entity_world: Option<glam::Mat4>,
    out: &mut std::collections::HashMap<
        u16,
        std::collections::HashMap<[i32; 3], super::state::PaintedTileEntry>,
    >,
) {
    use rkp_core::sparse_octree::{
        is_brick, is_leaf, leaf_slot, EMPTY_NODE, INTERIOR_NODE,
    };
    use rkp_core::brick_pool::{BRICK_DIM, BRICK_CELLS, BRICK_INTERIOR};
    const BRICK_CELL_EMPTY: u32 = 0xFFFF_FFFFu32;

    #[allow(clippy::too_many_arguments)]
    fn walk(
        octree_data: &[u32],
        brick_pool: &[u32],
        leaf_attrs: &rkp_core::LeafAttrPool,
        overlay: Option<&rkp_core::LeafAttrOverlay>,
        offset: usize,
        level: u8,
        max_depth: u8,
        coord_voxels: glam::UVec3,
        grid_origin: glam::Vec3,
        base_vs: f32,
        shader_materials: &std::collections::HashMap<u16, rkp_render::shader_composer::UserShaderInfo>,
        entity_world: Option<glam::Mat4>,
        out: &mut std::collections::HashMap<
            u16,
            std::collections::HashMap<[i32; 3], super::state::PaintedTileEntry>,
        >,
    ) {
        use rkp_core::sparse_octree::{
            is_brick, is_leaf, leaf_slot, brick_id, EMPTY_NODE, INTERIOR_NODE,
        };
        use rkp_core::brick_pool::{BRICK_DIM, BRICK_CELLS, BRICK_INTERIOR};
        const BRICK_CELL_EMPTY: u32 = 0xFFFF_FFFFu32;

        if offset >= octree_data.len() { return; }
        let node = octree_data[offset];
        if node == EMPTY_NODE || node == INTERIOR_NODE { return; }
        if is_brick(node) {
            let brick_id = brick_id(node);
            let base_idx = (brick_id * BRICK_CELLS) as usize;
            for cz in 0..BRICK_DIM {
                for cy in 0..BRICK_DIM {
                    for cx in 0..BRICK_DIM {
                        let cell_idx = (cz * BRICK_DIM * BRICK_DIM + cy * BRICK_DIM + cx) as usize;
                        let pool_idx = base_idx + cell_idx;
                        if pool_idx >= brick_pool.len() { continue; }
                        let cell = brick_pool[pool_idx];
                        if cell == BRICK_CELL_EMPTY || cell == BRICK_INTERIOR { continue; }
                        let attr = resolve_leaf_attr(overlay, leaf_attrs, cell);
                        let primary = attr.material_primary;
                        let secondary: u16 = attr.material_secondary_blend & 0x0FFF;
                        let blend: u16 = (attr.material_secondary_blend >> 12) & 0xF;
                        let painted_mat = if shader_materials.contains_key(&primary) {
                            Some(primary)
                        } else if blend > 0 && shader_materials.contains_key(&secondary) {
                            Some(secondary)
                        } else {
                            None
                        };
                        if let Some(mat) = painted_mat {
                            let cell_voxel = glam::UVec3::new(
                                coord_voxels.x + cx,
                                coord_voxels.y + cy,
                                coord_voxels.z + cz,
                            );
                            let cell_local = grid_origin
                                + glam::Vec3::new(
                                    cell_voxel.x as f32,
                                    cell_voxel.y as f32,
                                    cell_voxel.z as f32,
                                ) * base_vs;
                            let cell_max = cell_local + glam::Vec3::splat(base_vs);
                            let info = shader_materials.get(&mat);
                            let tile_size = info.and_then(|i| i.tile_size);
                            // Build a PaintedLeaf record only for instance
                            // shaders; Phase C generate shaders ignore
                            // leaves and we save the allocation.
                            let leaf = if info.map(|i| i.is_instance_pipeline).unwrap_or(false) {
                                let cell_center = cell_local + glam::Vec3::splat(base_vs * 0.5);
                                let local_normal = rkp_core::leaf_attr::unpack_oct(attr.normal_oct);
                                let (world_pos, world_normal) = transform_leaf(
                                    cell_center, local_normal, entity_world,
                                );
                                let material_packed = (primary as u32)
                                    | ((secondary as u32) << 16)
                                    | ((blend as u32) << 28);
                                Some(rkp_render::user_shader_emit_pass::PaintedLeaf {
                                    world_pos: world_pos.to_array(),
                                    material_packed,
                                    world_normal: world_normal.to_array(),
                                    _pad: 0.0,
                                })
                            } else {
                                None
                            };
                            super::lifecycle::expand_aabb(
                                out, mat, cell_local, cell_max, tile_size, leaf,
                            );
                        }
                    }
                }
            }
            return;
        }
        if is_leaf(node) {
            let slot = leaf_slot(node);
            let attr = resolve_leaf_attr(overlay, leaf_attrs, slot);
            let primary = attr.material_primary;
            let secondary: u16 = attr.material_secondary_blend & 0x0FFF;
            let blend: u16 = (attr.material_secondary_blend >> 12) & 0xF;
            let painted_mat = if shader_materials.contains_key(&primary) {
                Some(primary)
            } else if blend > 0 && shader_materials.contains_key(&secondary) {
                Some(secondary)
            } else {
                None
            };
            if let Some(mat) = painted_mat {
                let voxels_per_side = 1u32 << (max_depth - level);
                let leaf_size = voxels_per_side as f32 * base_vs;
                let leaf_min = grid_origin
                    + glam::Vec3::new(
                        coord_voxels.x as f32,
                        coord_voxels.y as f32,
                        coord_voxels.z as f32,
                    ) * base_vs;
                let leaf_max = leaf_min + glam::Vec3::splat(leaf_size);
                let info = shader_materials.get(&mat);
                let tile_size = info.and_then(|i| i.tile_size);
                let leaf = if info.map(|i| i.is_instance_pipeline).unwrap_or(false) {
                    let cell_center = leaf_min + glam::Vec3::splat(leaf_size * 0.5);
                    let local_normal = rkp_core::leaf_attr::unpack_oct(attr.normal_oct);
                    let (world_pos, world_normal) = transform_leaf(
                        cell_center, local_normal, entity_world,
                    );
                    let material_packed = (primary as u32)
                        | ((secondary as u32) << 16)
                        | ((blend as u32) << 28);
                    Some(rkp_render::user_shader_emit_pass::PaintedLeaf {
                        world_pos: world_pos.to_array(),
                        material_packed,
                        world_normal: world_normal.to_array(),
                        _pad: 0.0,
                    })
                } else {
                    None
                };
                super::lifecycle::expand_aabb(
                    out, mat, leaf_min, leaf_max, tile_size, leaf,
                );
            }
            return;
        }
        // Branch — descend into 8 children. `node` is the absolute
        // offset of the first child (rebased at allocation time).
        let _ = leaf_slot(0);
        let _ = INTERIOR_NODE;
        if level >= max_depth { return; }
        let child_voxels = 1u32 << (max_depth - level - 1);
        for octant in 0u32..8 {
            let dx = octant & 1;
            let dy = (octant >> 1) & 1;
            let dz = (octant >> 2) & 1;
            let child_coord = glam::UVec3::new(
                coord_voxels.x + dx * child_voxels,
                coord_voxels.y + dy * child_voxels,
                coord_voxels.z + dz * child_voxels,
            );
            let child_offset = node as usize + octant as usize;
            walk(
                octree_data,
                brick_pool,
                leaf_attrs,
                overlay,
                child_offset,
                level + 1,
                max_depth,
                child_coord,
                grid_origin,
                base_vs,
                shader_materials,
                entity_world,
                out,
            );
        }
    }

    walk(
        octree_data,
        brick_pool,
        leaf_attrs,
        overlay,
        root_offset,
        0,
        depth,
        glam::UVec3::ZERO,
        grid_origin,
        base_voxel_size,
        shader_materials,
        entity_world,
        out,
    );
    let _ = (is_brick, is_leaf, leaf_slot, EMPTY_NODE, INTERIOR_NODE, BRICK_DIM, BRICK_CELLS, BRICK_INTERIOR, BRICK_CELL_EMPTY);
}

/// Transform a leaf's local-space center + normal to world space using
/// the entity's world matrix. When no matrix is available, the leaf is
/// treated as already-world (identity transform).
fn transform_leaf(
    local_pos: glam::Vec3,
    local_normal: glam::Vec3,
    entity_world: Option<glam::Mat4>,
) -> (glam::Vec3, glam::Vec3) {
    match entity_world {
        Some(w) => {
            let world_pos = w.transform_point3(local_pos);
            // Normal goes through the inverse-transpose. For a uniform-
            // scale + rotation transform this is equivalent to the
            // upper-3x3 of `w` re-normalized; for non-uniform scales
            // we'd need the inverse-transpose explicitly. V1: use
            // upper-3x3 directly and renormalize — accurate enough
            // for grass orientation gating.
            let world_normal = w.transform_vector3(local_normal).normalize_or_zero();
            (world_pos, world_normal)
        }
        None => (local_pos, local_normal),
    }
}

/// Diagnostic — count painted leaves on each side of `mid_x` (in
/// object-local space). Mirrors the structure of `scan_painted_aabbs`
/// but only tallies; used once per scan to disambiguate "scan misses
/// half" from "paint really only on one half".
fn count_painted_halves(
    octree_data: &[u32],
    brick_pool: &[u32],
    leaf_attrs: &rkp_core::LeafAttrPool,
    root_offset: usize,
    depth: u8,
    grid_origin: glam::Vec3,
    base_voxel_size: f32,
    mid_x: f32,
    shader_materials: &std::collections::HashMap<u16, rkp_render::shader_composer::UserShaderInfo>,
    left: &mut u32,
    right: &mut u32,
) {
    use rkp_core::sparse_octree::{
        is_brick, is_leaf, leaf_slot, brick_id, EMPTY_NODE, INTERIOR_NODE,
    };
    use rkp_core::brick_pool::{BRICK_DIM, BRICK_CELLS, BRICK_INTERIOR};
    const BRICK_CELL_EMPTY: u32 = 0xFFFF_FFFFu32;

    fn check(material_packed: u32,
             shader_materials: &std::collections::HashMap<u16, rkp_render::shader_composer::UserShaderInfo>) -> bool {
        let primary = (material_packed & 0xFFFF) as u16;
        let sec_blend = ((material_packed >> 16) & 0xFFFF) as u16;
        let secondary = sec_blend & 0x0FFF;
        let blend = (sec_blend >> 12) & 0xF;
        if shader_materials.contains_key(&primary) {
            return true;
        }
        if blend > 0 && shader_materials.contains_key(&secondary) {
            return true;
        }
        false
    }

    fn walk(
        octree_data: &[u32],
        brick_pool: &[u32],
        leaf_attrs: &rkp_core::LeafAttrPool,
        offset: usize,
        level: u8,
        max_depth: u8,
        coord_voxels: glam::UVec3,
        grid_origin: glam::Vec3,
        base_vs: f32,
        mid_x: f32,
        shader_materials: &std::collections::HashMap<u16, rkp_render::shader_composer::UserShaderInfo>,
        left: &mut u32,
        right: &mut u32,
    ) {
        use rkp_core::sparse_octree::{
            is_brick, is_leaf, brick_id, EMPTY_NODE, INTERIOR_NODE,
        };
        use rkp_core::brick_pool::{BRICK_DIM, BRICK_CELLS, BRICK_INTERIOR};
        const BRICK_CELL_EMPTY: u32 = 0xFFFF_FFFFu32;
        if offset >= octree_data.len() { return; }
        let node = octree_data[offset];
        if node == EMPTY_NODE || node == INTERIOR_NODE { return; }
        if is_brick(node) {
            let bid = brick_id(node);
            let base_idx = (bid * BRICK_CELLS) as usize;
            for cz in 0..BRICK_DIM {
                for cy in 0..BRICK_DIM {
                    for cx in 0..BRICK_DIM {
                        let cell_idx = (cz * BRICK_DIM * BRICK_DIM + cy * BRICK_DIM + cx) as usize;
                        let pool_idx = base_idx + cell_idx;
                        if pool_idx >= brick_pool.len() { continue; }
                        let cell = brick_pool[pool_idx];
                        if cell == BRICK_CELL_EMPTY || cell == BRICK_INTERIOR { continue; }
                        let attr = leaf_attrs.get(cell);
                        let packed = (attr.material_primary as u32)
                            | ((attr.material_secondary_blend as u32) << 16);
                        if super::lifecycle::count_painted_halves_check(packed, shader_materials) {
                            let cell_x = grid_origin.x + (coord_voxels.x + cx) as f32 * base_vs;
                            if cell_x < mid_x { *left += 1; } else { *right += 1; }
                        }
                    }
                }
            }
            return;
        }
        if is_leaf(node) {
            return;
        }
        if level >= max_depth { return; }
        let child_voxels = 1u32 << (max_depth - level - 1);
        for octant in 0u32..8 {
            let dx = octant & 1;
            let dy = (octant >> 1) & 1;
            let dz = (octant >> 2) & 1;
            let child_coord = glam::UVec3::new(
                coord_voxels.x + dx * child_voxels,
                coord_voxels.y + dy * child_voxels,
                coord_voxels.z + dz * child_voxels,
            );
            walk(
                octree_data,
                brick_pool,
                leaf_attrs,
                node as usize + octant as usize,
                level + 1,
                max_depth,
                child_coord,
                grid_origin,
                base_vs,
                mid_x,
                shader_materials,
                left,
                right,
            );
        }
    }

    walk(
        octree_data, brick_pool, leaf_attrs, root_offset, 0, depth,
        glam::UVec3::ZERO, grid_origin, base_voxel_size, mid_x,
        shader_materials, left, right,
    );
    let _ = (is_brick, is_leaf, leaf_slot, EMPTY_NODE, INTERIOR_NODE, BRICK_DIM, BRICK_CELLS, BRICK_INTERIOR, BRICK_CELL_EMPTY);
}

pub(super) fn count_painted_halves_check(
    material_packed: u32,
    shader_materials: &std::collections::HashMap<u16, rkp_render::shader_composer::UserShaderInfo>,
) -> bool {
    let primary = (material_packed & 0xFFFF) as u16;
    let sec_blend = ((material_packed >> 16) & 0xFFFF) as u16;
    let secondary = sec_blend & 0x0FFF;
    let blend = (sec_blend >> 12) & 0xF;
    if shader_materials.contains_key(&primary) {
        return true;
    }
    if blend > 0 && shader_materials.contains_key(&secondary) {
        return true;
    }
    false
}

/// Sentinel tile coord for non-tiled shaders (matching
/// `rkp_render::user_shader_pass::NO_TILE`). Single inner-map entry
/// for the whole painted area — V9 single-region behaviour.
const NO_TILE_COORD: [i32; 3] = [i32::MIN, i32::MIN, i32::MIN];

/// Register a painted leaf occupying `[mn, mx]` (host-local) for
/// material `mat`. For shaders with a `@tile_size`, the leaf is
/// bucketed into every tile coord it overlaps; for non-tiled
/// shaders, the leaf is registered under `NO_TILE_COORD` and its
/// AABB merged into the running painted-leaf bounds.
///
/// Tiling assumes the grid is anchored at host-local origin
/// (tile_coord = floor(pos / tile_size)). For typical paint where a
/// leaf's size ≪ tile_size, each leaf lives in a single tile — the
/// boundary case (leaf straddling tiles) registers it in all
/// overlapping tiles, which slightly inflates per-tile counts but
/// is correct for the band gate.
fn expand_aabb(
    out: &mut std::collections::HashMap<
        u16,
        std::collections::HashMap<[i32; 3], super::state::PaintedTileEntry>,
    >,
    mat: u16,
    mn: glam::Vec3,
    mx: glam::Vec3,
    tile_size: Option<f32>,
    leaf: Option<rkp_render::user_shader_emit_pass::PaintedLeaf>,
) {
    let mat_map = out.entry(mat).or_default();

    fn merge(
        mat_map: &mut std::collections::HashMap<[i32; 3], super::state::PaintedTileEntry>,
        key: [i32; 3],
        mn: glam::Vec3,
        mx: glam::Vec3,
        leaf: Option<rkp_render::user_shader_emit_pass::PaintedLeaf>,
    ) {
        let entry = mat_map.entry(key).or_insert_with(super::state::PaintedTileEntry::empty);
        entry.aabb.min = entry.aabb.min.min(mn);
        entry.aabb.max = entry.aabb.max.max(mx);
        entry.leaf_count = entry.leaf_count.saturating_add(1);
        if let Some(l) = leaf {
            entry.leaves.push(l);
        }
    }

    match tile_size {
        None => merge(mat_map, NO_TILE_COORD, mn, mx, leaf),
        Some(s) if s > 0.0 => {
            // Compute tile coord range the leaf overlaps. Use a tiny
            // epsilon on the upper bound so a leaf whose max sits
            // exactly on a tile boundary doesn't count for the next
            // tile too.
            let inv = 1.0 / s;
            let lo = (mn * inv).floor();
            let hi = ((mx - glam::Vec3::splat(1e-6)) * inv).floor();
            for ix in (lo.x as i32)..=(hi.x as i32) {
                for iy in (lo.y as i32)..=(hi.y as i32) {
                    for iz in (lo.z as i32)..=(hi.z as i32) {
                        // Same leaf is appended to every tile it
                        // overlaps. Tiled instance shaders aren't
                        // exercised yet but the API stays consistent.
                        merge(mat_map, [ix, iy, iz], mn, mx, leaf);
                    }
                }
            }
        }
        // tile_size 0 or negative → treat as non-tiled.
        Some(_) => merge(mat_map, NO_TILE_COORD, mn, mx, leaf),
    }
}

