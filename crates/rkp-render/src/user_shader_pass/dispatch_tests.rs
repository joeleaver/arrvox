use super::*;
use crate::shader_composer::UserShaderInfo;
use super::super::setup::compose_geom_source;

#[test]
fn geom_shader_validates_with_empty_chunk() {
    let src = compose_geom_source("");
    let module = naga::front::wgsl::parse_str(&src).unwrap_or_else(|e| {
        panic!("parse error:\n{}", e.emit_to_string(&src))
    });
    let mut v = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    );
    v.validate(&module).unwrap_or_else(|e| panic!("validation error: {e:?}"));
}

#[test]
fn compose_splices_user_chunk() {
    let chunk = "fn dispatch_user_generate(shader_id: u32, cell_world_pos: vec3<f32>, host: HostSample, ctx: UserCtx) -> VoxelEmit { return voxel_emit_skip(); }";
    let src = compose_geom_source(chunk);
    assert!(src.contains("dispatch_user_generate"));
    assert!(!src.contains("Default identity stub"));
}

#[test]
fn level_uniform_size_is_16() {
    assert_eq!(std::mem::size_of::<LevelUniform>(), 16);
}

#[test]
fn active_cell_size_is_32() {
    assert_eq!(std::mem::size_of::<ActiveCell>(), 32);
}

#[test]
fn resolve_shader_id_alphabetical_one_based() {
    let infos = vec![
        UserShaderInfo { name: "zeta".into(), ..Default::default() },
        UserShaderInfo { name: "alpha".into(), ..Default::default() },
        UserShaderInfo { name: "mu".into(), ..Default::default() },
    ];
    assert_eq!(resolve_shader_id(&infos, "alpha"), 1);
    assert_eq!(resolve_shader_id(&infos, "mu"), 2);
    assert_eq!(resolve_shader_id(&infos, "zeta"), 3);
    assert_eq!(resolve_shader_id(&infos, ""), 0);
    assert_eq!(resolve_shader_id(&infos, "missing"), 0);
}
