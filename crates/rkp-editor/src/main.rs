//! RKIPatch Editor — gaussian splat scene editor.
//!
//! Thin wrapper over rkf-editor's shared infrastructure. Uses `SplatMarchPass`
//! (opacity field surface finding) instead of `RayMarchPass` (SDF sphere tracing),
//! and a direct mesh-to-opacity import pipeline instead of SDF voxelization.

fn main() -> anyhow::Result<()> {
    env_logger::init();

    let march_factory: rkf_editor::engine_loop::MarchFactory = Box::new(
        |device, queue, _scene, gbuffer, _tile_cull, _material_buffer, _shader_params, _opacity_code| -> Box<dyn rkf_render::MarchPass> {
            // Use standalone pipeline: own scene buffers, shadow/AO, shading.
            // The engine skips its shadow_ao + shading because handles_full_pipeline() returns true.
            let (w, h) = (gbuffer.width, gbuffer.height);
            Box::new(rkp_render::SplatRasterPass::new_standalone(
                device, queue, w, h,
            ))
        },
    );

    let import_fn: rkf_editor::import_worker::ImportFn = Box::new(
        |input_path, output_path, config| {
            // Use octree-native .rkp import instead of flat .rkf.
            let rkp_path = output_path.with_extension("rkp");
            rkp_render::import_mesh_to_opacity_rkp(input_path, &rkp_path, config)
        },
    );

    rkf_editor::run_editor("RKIPatch Editor", Some(march_factory), Some(import_fn))
}
