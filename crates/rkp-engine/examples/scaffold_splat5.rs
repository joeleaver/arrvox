//! One-shot M1 validation: scaffold the splat5 project and report discovered
//! generator script files. Run with:
//!
//!     cargo run -p rkp-engine --example scaffold_splat5
//!
//! Then verify by building the resulting gameplay crate:
//!
//!     cargo build --release --manifest-path \
//!       /home/joe/dev/rkifield_game/splat5/.rkpatch-cache/gameplay/Cargo.toml

use std::path::PathBuf;

fn main() {
    let project: PathBuf = "/home/joe/dev/rkifield_game/splat5".into();
    let crate_dir = rkp_engine::scaffold::generate_gameplay_crate(&project)
        .expect("scaffold");
    println!("scaffolded -> {}", crate_dir.display());

    let lib_rs = std::fs::read_to_string(crate_dir.join("src/lib.rs")).unwrap();
    println!("--- lib.rs ---\n{lib_rs}");

    let gen_mod = crate_dir.join("src/generators/mod.rs");
    if gen_mod.exists() {
        println!("--- generators/mod.rs ---\n{}", std::fs::read_to_string(gen_mod).unwrap());
    } else {
        println!("(no generators/mod.rs — no generator scripts found)");
    }
}
