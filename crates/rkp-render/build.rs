use std::path::Path;
use wesl::Wesl;

fn main() {
    let shaders_dir = Path::new("src/shaders");
    println!("cargo:rerun-if-changed={}", shaders_dir.display());

    let out_dir = std::env::var("OUT_DIR").expect("OUT_DIR set by cargo");
    let out_dir = Path::new(&out_dir);

    // Stripping must stay off: the user-shader composer splices new
    // function bodies into shader templates at runtime via the
    // const-decl anchors `USER_<NAME>_DISPATCH_BEGIN/_END` (see
    // `shader_composer::splice_const_marker`). Some helpers are only
    // transitively reachable AFTER that splice runs, and WESL's
    // default lazy-compile would strip them as dead code.
    let mut resolver = Wesl::new(shaders_dir);
    resolver.use_stripping(false);

    // Skiplist: compose-by-concat fragments. Currently `include_str!`'d
    // individually and string-concatenated by `proc_sample.rs` /
    // `proc_raymarch.rs` to form a complete shader. Phase 2 Wave E
    // folds them onto WESL `import` and removes this skiplist.
    const SKIP: &[&str] = &[
        "proc_eval",
        "proc_eval_types",
        "proc_sample",
        "proc_raymarch",
    ];

    // Enumerate every top-level `.wesl` file and emit a flat WGSL
    // artifact for it under its file stem. Files under `lib/` are
    // imports-only — they get pulled into emitting artifacts via
    // `import package::lib::<module>::<symbol>;` and don't need
    // their own artifact.
    for entry in std::fs::read_dir(shaders_dir).expect("read src/shaders") {
        let entry = entry.expect("read shaders dir entry");
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        if path.extension().and_then(|e| e.to_str()) != Some("wesl") {
            continue;
        }
        let stem = path
            .file_stem()
            .and_then(|s| s.to_str())
            .expect("shader stem utf8");
        if SKIP.contains(&stem) {
            continue;
        }
        let module_path = format!("package::{stem}")
            .parse()
            .unwrap_or_else(|e| panic!("parse module path package::{stem}: {e}"));
        resolver.build_artifact(&module_path, stem);

        // Strict naga validation. WESL emits flat WGSL without
        // checking that every identifier resolves (it leaves unknown
        // names in place expecting them to be locally declared);
        // unresolved identifiers slip through to wgpu pipeline
        // creation as runtime errors. Catch them here at build time.
        let emitted = out_dir.join(format!("{stem}.wgsl"));
        let src = std::fs::read_to_string(&emitted)
            .unwrap_or_else(|e| panic!("read emitted artifact {stem}.wgsl: {e}"));
        match naga::front::wgsl::parse_str(&src) {
            Ok(module) => {
                let mut v = naga::valid::Validator::new(
                    naga::valid::ValidationFlags::all(),
                    naga::valid::Capabilities::all(),
                );
                if let Err(e) = v.validate(&module) {
                    panic!("naga validation failed for `{stem}`:\n{e:?}");
                }
            }
            Err(e) => {
                panic!(
                    "naga parse failed for `{stem}`:\n{}",
                    e.emit_to_string(&src),
                );
            }
        }
    }
}
