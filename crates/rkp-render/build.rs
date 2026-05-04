use std::path::Path;
use wesl::Wesl;

fn main() {
    let shaders_dir = Path::new("src/shaders");
    println!("cargo:rerun-if-changed={}", shaders_dir.display());

    // Stripping must stay off: the user-shader composer splices new
    // function bodies into shader templates at runtime via the
    // const-decl anchors `USER_<NAME>_DISPATCH_BEGIN/_END` (see
    // `shader_composer::splice_const_marker`). Some helpers are only
    // transitively reachable AFTER that splice runs, and WESL's
    // default lazy-compile would strip them as dead code.
    let mut resolver = Wesl::new(shaders_dir);
    resolver.use_stripping(false);

    // Skiplist: shaders that don't compile standalone in Phase 1.
    // Two kinds:
    //   - Orphans (no Rust include site, dead leftovers from the
    //     pre-octree opacity-pipeline removal): opacity_grass,
    //     opacity_shade_pbr.
    //   - Compose-by-concat fragments (loaded together with sibling
    //     files at runtime to form a complete shader): proc_eval,
    //     proc_eval_types, proc_sample, proc_raymarch — these get
    //     `include_str!`'d separately and string-concatenated by
    //     `proc_sample.rs` / `proc_raymarch.rs`. Phase 2 will fold
    //     them onto WESL `import` and remove this skiplist entry.
    const SKIP: &[&str] = &[
        // Orphans — leftovers from pre-octree opacity-pipeline removal.
        // No Rust include site, references unbound symbols.
        "opacity_grass",
        "opacity_shade_pbr",
        "shade_grass",
        // Compose-by-concat fragments — currently `include_str!`'d
        // separately and string-concatenated by their `proc_*.rs`
        // pipeline constructors. Phase 2 Wave E folds them onto WESL
        // `import` and removes these entries.
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
    }
}
