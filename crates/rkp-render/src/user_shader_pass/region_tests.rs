use super::*;

#[test]
fn region_uniform_size_is_240() {
    // Mirrors the compile-time const-assert above. A runtime test
    // gives a clearer failure mode if the layout drifts (the
    // const-assert reports as "evaluation of constant value failed").
    assert_eq!(std::mem::size_of::<RegionUniform>(), 240);
}
