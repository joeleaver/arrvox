//! RKIPatch Editor — gaussian splat scene editor.
//!
//! Thin wrapper over rkf-editor's shared infrastructure. Uses `SplatMarchPass`
//! (opacity field surface finding) instead of `RayMarchPass` (SDF sphere tracing).

fn main() -> anyhow::Result<()> {
    env_logger::init();

    let march_factory: rkf_editor::engine_loop::MarchFactory = Box::new(
        |device, scene, gbuffer, tile_cull| -> Box<dyn rkf_render::MarchPass> {
            Box::new(rkp_render::SplatMarchPass::new(device, scene, gbuffer, tile_cull))
        },
    );

    rkf_editor::run_editor("RKIPatch Editor", Some(march_factory))
}
