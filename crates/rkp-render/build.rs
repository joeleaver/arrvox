use wesl::Wesl;

fn main() {
    println!("cargo:rerun-if-changed=src/shaders");

    // Stripping must stay off: the user-shader composer splices new
    // function bodies into the template at runtime via marker pairs
    // (see `shader_composer::splice_inst_chunks`), and those splices
    // reference helpers (`unpack_oct_normal`, `descend_proto_octree`)
    // that are only transitively reachable AFTER the splice. WESL's
    // default lazy-compile would strip them as dead code.
    Wesl::new("src/shaders")
        .use_stripping(false)
        .build_artifact(
            &"package::skin_deform".parse().unwrap(),
            "skin_deform",
        );
}
