//! RKIPatch Editor — gaussian splat scene editor.
//!
//! Thin wrapper over rkf-editor's shared infrastructure. Uses `SplatMarchPass`
//! (opacity field surface finding) instead of `RayMarchPass` (SDF sphere tracing),
//! and a direct mesh-to-opacity import pipeline instead of SDF voxelization.

fn main() -> anyhow::Result<()> {
    env_logger::init();

    let march_factory: rkf_editor::engine_loop::MarchFactory = Box::new(
        |device, scene, gbuffer, tile_cull, material_buffer, shader_params, opacity_code| -> Box<dyn rkf_render::MarchPass> {
            Box::new(rkp_render::SplatRasterPass::new(
                device, scene, gbuffer, tile_cull,
                material_buffer, shader_params, opacity_code,
            ))
        },
    );

    let import_fn: rkf_editor::import_worker::ImportFn = Box::new(
        |input_path, output_path, config| {
            rkp_render::import_mesh_to_opacity_rkf(input_path, output_path, config)
        },
    );

    rkf_editor::run_editor("RKIPatch Editor", Some(march_factory), Some(import_fn))
}
