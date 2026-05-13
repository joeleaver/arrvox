//! `EngineState::new` — wires up render/bake workers, generators, ECS,
//! input system, scene manager, and seeds defaults for every flag/cache.
//! Big ceremony function isolated from the struct definition.

use rkp_render::rkp_scene_manager::RkpSceneManager;

use crate::camera::CameraControlState;
use super::super::{CameraState, EngineConfig, FrameCallback};
use super::EngineState;

impl EngineState {
    pub(crate) fn new(config: &EngineConfig, frame_callback: FrameCallback) -> Self {
        let ctx = rkp_render::RenderContext::new_headless();
        let device = ctx.device;
        let queue = ctx.queue;

        let width = config.width;
        let height = config.height;

        let scene_mgr = std::sync::Arc::new(std::sync::Mutex::new(
            RkpSceneManager::new(1_000_000),
        ));
        // Clone the lock-free epoch handle ONCE at startup; sim
        // reads it every tick to detect geometry changes without
        // having to take the scene_mgr Mutex.
        let (geometry_epoch_handle, paint_epoch_handle) = {
            let sm = scene_mgr.lock().expect("scene_mgr poisoned");
            (sm.epoch_handle(), sm.paint_epoch_handle())
        };

        // Input system with default action map.
        let mut input_system = rkp_runtime::input::InputSystem::new();
        input_system.add_map(crate::camera::default_action_map());
        input_system.set_active_map("editor");
        let camera_control = CameraControlState::default();

        // Bake worker: shares device/queue clones (cheap — wgpu wraps
        // the underlying objects in Arcs internally) and the same
        // scene_mgr we'll hand to the render thread below.
        let bake_worker = crate::bake_worker::BakeWorker::spawn(
            device.clone(),
            queue.clone(),
            scene_mgr.clone(),
        );
        let generator_system = crate::generator::GeneratorSystem::new(
            crate::generator::GeneratorRegistry::new(),
            bake_worker.tx_generator.clone(),
            bake_worker.rx_generator.clone(),
        );

        // Render worker — takes ownership of `device` + `queue` and
        // builds the renderer + per-VR pass chains on the render
        // thread. Sim never touches wgpu after this point; everything
        // GPU goes through the snapshot/result/command channels on
        // `render_worker`.
        let render_init = crate::render_frame::RenderInit {
            device,
            queue,
            initial_width: width,
            initial_height: height,
            scene_mgr: scene_mgr.clone(),
            render_pacing: config.render_pacing,
        };
        let render_worker = crate::render_worker::RenderWorker::spawn(
            render_init,
            frame_callback,
        );

        Self {
            bake_worker,
            generator_system,
            user_shader_registry: rkp_render::shader_composer::UserShaderRegistry::empty(),
            painted_materials: std::collections::HashMap::new(),
            painted_anchors: std::sync::Arc::new(std::collections::HashMap::new()),
            debug_last_anchor_seeds: None,
            painted_per_entity: std::collections::HashMap::new(),
            painted_dirty_entities: std::collections::HashSet::new(),
            entities_known_empty: std::collections::HashSet::new(),
            mutation_log: super::super::mutation_log::MutationLog::new(),
            painted_materials_paint_epoch: 0,
            painted_materials_geometry_epoch: 0,
            paint_overlays: std::collections::HashMap::new(),
            sculpt_overlays: std::collections::HashMap::new(),
            material_is_glass: Vec::new(),
            material_glass_lib_epoch: 0,
            asset_has_glass_cache: std::collections::HashMap::new(),
            render_worker,
            scene_mgr,
            geometry_epoch_handle,
            paint_epoch_handle,
            skinning_data_cache: std::collections::HashMap::new(),
            // 0 → cache rebuilds the first time epoch > 0 (any
            // geometry mutation triggers it). Until then the cache
            // is empty, which matches a freshly-spawned scene.
            skinning_data_cache_epoch: 0,
            input_system,
            camera_control,
            camera: CameraState::default(),
            viewports: {
                let mut v = crate::viewport::Viewports::new();
                v.insert(crate::viewport::Viewport::new_main(width, height));
                // BUILD starts hidden (new_build sets visible: false) and
                // the editor flips it via SetViewportVisible when the user
                // opens the procedural preview.
                v.insert(crate::viewport::Viewport::new_build(800, 600));
                v
            },
            world: hecs::World::new(),
            registry: {
                let mut r = crate::component_registry::ComponentRegistry::new();
                crate::component_registry::register_builtins(&mut r);
                r
            },
            entity_uuids: std::collections::HashMap::new(),
            uuid_to_entity: std::collections::HashMap::new(),
            next_entity_uuid: 1,
            entity_tree_order: std::collections::HashMap::new(),
            next_tree_order: 0.0,
            selected_entity: None,
            selected_procedural_node: None,
            gpu_assets: std::sync::Arc::new(Vec::new()),
            gpu_instances: std::sync::Arc::new(Vec::new()),
            gpu_instance_overlays: std::sync::Arc::new(Vec::new()),
            gpu_instance_sculpts: std::sync::Arc::new(Vec::new()),
            splat_draws: std::sync::Arc::new(Vec::new()),
            proxy_draws: std::sync::Arc::new(Vec::new()),
            gpu_to_entity: Vec::new(),
            entity_to_gpu: std::collections::HashMap::new(),
            project_loaded: false,
            project_name: String::new(),
            project_dir: None,
            project_path: None,
            scene_path: None,
            project_dirty: true, // push initial state
            available_models: Vec::new(),
            models_dirty: false,
            generators_dirty: true,
            available_generator_presets: Vec::new(),
            generator_presets_dirty: true,
            pending_generator_slot_keys: std::collections::HashMap::new(),
            importing_sources: std::collections::HashSet::new(),
            importing_progress: std::collections::HashMap::new(),
            importing_dirty: false,
            editor_layout_json: None,
            editor_layout_pending: false,
            animation_cache: crate::animation::AnimationAssetCache::new(),
            bone_matrix_allocator: crate::scene_sync::BoneMatrixAllocator::new(),
            skin_dispatches: Vec::new(),
            skin_batch: rkp_render::SkinBatchScratch::default(),
            skin_bone_field_bytes: 0,
            skin_bone_field_occ_bytes: 0,
            last_skin_poses: std::collections::HashMap::new(),
            skin_reuse: false,
            skinning_enabled: true,
            dqs_enabled: false,
            last_cloud_sun_atten_raw: f32::NAN,
            in_flight_pick_ghost: None,
            material_lib: crate::material_library::MaterialLibrary::new(),
            selected_material: None,
            selected_model: None,
            environment: crate::environment::EnvironmentSettings::default(),
            environment_dirty: true, // upload on first frame
            environment_ui_dirty: true,
            console: crate::console::ConsoleLog::new(),
            gameplay_loader: crate::gameplay_loader::GameplayLoader::new(),
            behavior_executor: None,
            behavior_commands: crate::behavior::CommandQueue::new(),
            game_store: crate::behavior::GameStore::new(),
            gameplay_systems: Vec::new(),
            play_total_time: 0.0,
            play_frame_count: 0,
            play_state: None,
            show_colliders: false,
            collider_caches_dirty: true,
            tick_hz_ema: 60.0,
            physics_hz_ema: 0.0,
            render_hz_ema: 0.0,
            delivered_hz_ema: 0.0,
            prev_inspector: None,
            cached_material_usage: None,
            prev_procedural: None,
            prev_environment: None,
            prev_user_shader_hash: 0,
            user_shader_first_send: true,
            file_watcher: None,
            import_worker: crate::import_worker::ImportWorker::new(),
            geometry_dirty: false,
            scene_dirty: false,
            gpu_objects_dirty: true,
            frame_index: 0,
            profiling: crate::profiling::ProfilingHistory::default(),
            behavior_fixed_accumulator: 0.0,
            cloud_sun_atten: 1.0,
            width,
            height,
            gizmo: crate::gizmo::GizmoState::new(),
            proc_gizmo: crate::gizmo::GizmoState::new(),
            build_mouse_pos: glam::Vec2::ZERO,
            build_mouse_left: false,
            proc_gizmo_parent_world: glam::Affine3A::IDENTITY,
            proc_gizmo_initial_local: (glam::Vec3::ZERO, glam::Quat::IDENTITY, glam::Vec3::ONE),
            mouse_pos: glam::Vec2::ZERO,
            pending_pick: None,
            pending_drop: None,
            drag_preview: None,
            paint_pick_settings: None,
            last_paint_stamp_at: None,
            paint_mode_active: false,
            paint_mode_radius: 0.5,
            sculpt_pick_settings: None,
            sculpt_pending_at: None,
            sculpt_mode_active: false,
            sculpt_mode_radius: 0.5,
            num_lights_cache: 1,
            shade_params_base: rkp_render::rkp_shade::ShadeParams::default(),
            lod_enabled: true,
            // Bake-time Laplacian smoothing of stored normals (see
            // `load_asset` → `smooth_shell_normals`) makes the shader-
            // time centroid reconstruction redundant. Default OFF so
            // the shader uses the smoothed baked normal via its
            // existing 1-fetch path.
            surfacenet_enabled: false,
        }
    }
}
