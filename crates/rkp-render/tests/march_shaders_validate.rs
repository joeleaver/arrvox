//! Phase 1a guard — naga-validate the post-split octree_march and
//! rkp_shadow_trace WGSL. The runtime `validate_wgsl` helper logs to
//! stderr but doesn't fail the pipeline build, so a syntax bug only
//! surfaces as a black screen. This test fails the build instead.

fn naga_validates(source: &str, label: &str) {
    let module = match naga::front::wgsl::parse_str(source) {
        Ok(m) => m,
        Err(e) => panic!("{label} parse error:\n{}", e.emit_to_string(source)),
    };
    let mut v = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    );
    if let Err(e) = v.validate(&module) {
        panic!("{label} validation error: {e}");
    }
}

#[test]
fn octree_march_validates() {
    naga_validates(
        wesl::include_wesl!("octree_march"),
        "octree_march",
    );
}

#[test]
fn rkp_shadow_trace_validates() {
    naga_validates(
        wesl::include_wesl!("rkp_shadow_trace"),
        "rkp_shadow_trace",
    );
}

#[test]
fn splat_validates() {
    naga_validates(
        wesl::include_wesl!("splat"),
        "splat",
    );
}

#[test]
fn splat_resolve_validates() {
    naga_validates(
        wesl::include_wesl!("splat_resolve"),
        "splat_resolve",
    );
}
