use std::path::Path;
use wesl::Wesl;

fn main() {
    let shaders_dir = Path::new("src/shaders");
    println!("cargo:rerun-if-changed={}", shaders_dir.display());

    let out_dir = std::env::var("OUT_DIR").expect("OUT_DIR set by cargo");
    let out_dir = Path::new(&out_dir);

    // `use_stripping(false)` is a deliberate trade-off vs. the WESL
    // idiom (`use_stripping(true)` + `keep_declarations(...)`).
    //
    // Why we keep stripping off in wesl-rs 0.3.2:
    //
    //   1. The user-shader composer text-splices new bodies into
    //      templates at runtime (see `splice_const_marker`). Helpers
    //      called from those splices (`intersect_aabb`,
    //      `descend_proto_octree`) and the const-decl anchors that
    //      bracket the splice region are NOT reachable from any
    //      pre-splice entry point and would be stripped.
    //
    //   2. Setting `keep_declarations` OVERRIDES the auto-keep-
    //      entrypoints default — the keep list becomes exclusive,
    //      so it must also enumerate every `@compute`/`@vertex`/
    //      `@fragment` entry across every shader.
    //
    //   3. CRITICAL — wesl-rs 0.3.2's stripping doesn't trace
    //      reachability through imports for binding declarations.
    //      `proc_eval::eval_tree` (imported by `proc_sample`)
    //      references the `instructions` binding declared in
    //      `proc_sample` at root scope, but stripping doesn't see
    //      `main → eval_tree (imported) → instructions` and erases
    //      the binding. To work around it the keep list would have
    //      to enumerate every root-level binding/struct/const/type
    //      across every shader, which eats most of the size win.
    //
    // The cost: artifacts include all imported lib code even when
    // unused by a particular consumer. Acceptable — engine ships
    // ~40 shaders, none over 2 MB compiled, and runtime perf is
    // unaffected (drivers run their own dead-code elimination).
    //
    // Revisit when wesl-rs implements `@publicName` (currently
    // proposed but not implemented per the spec) or improves
    // import-aware stripping.
    let mut resolver = Wesl::new(shaders_dir);
    resolver.use_stripping(false);

    // Skiplist: imports-only modules with no entry point. They get
    // pulled into emitting artifacts via `import package::<stem>`
    // and don't need their own artifact.
    //
    // `proc_sample` and `proc_raymarch` import from these and emit
    // standalone artifacts (Wave E folded the old compose-by-concat
    // model onto WESL imports).
    const SKIP: &[&str] = &[
        "proc_eval",
        "proc_eval_types",
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
