//! Game crate scaffold — auto-generates a compilable gameplay crate from scripts.
//!
//! The user writes `.rs` files in `project_dir/assets/scripts/{components,systems}/`.
//! This module generates a complete Rust cdylib crate in `.rkpatch-cache/gameplay/`
//! that the engine builds, loads, and hot-reloads. The user never touches the
//! generated crate — they only write script files.

use std::fs;
use std::path::{Path, PathBuf};

/// Cache directory name placed in the project root.
pub const CACHE_DIR: &str = ".rkpatch-cache";
/// Name of the generated gameplay crate.
pub const GAMEPLAY_CRATE: &str = "gameplay";

/// Errors during scaffold generation.
#[derive(Debug)]
pub enum ScaffoldError {
    Io(String),
}

impl std::fmt::Display for ScaffoldError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(msg) => write!(f, "scaffold I/O: {msg}"),
        }
    }
}

impl From<std::io::Error> for ScaffoldError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e.to_string())
    }
}

/// Path to the engine workspace root (parent of crates/).
/// Derived at compile time from this crate's manifest directory.
pub fn engine_root() -> PathBuf {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    manifest.parent().and_then(|p| p.parent()).unwrap_or(manifest).to_path_buf()
}

/// Path to the generated gameplay crate for a project.
pub fn gameplay_crate_dir(project_dir: &Path) -> PathBuf {
    project_dir.join(CACHE_DIR).join(GAMEPLAY_CRATE)
}

/// Path to the built dylib for a project.
pub fn gameplay_dylib_path(project_dir: &Path) -> PathBuf {
    let crate_dir = gameplay_crate_dir(project_dir);
    let target_dir = crate_dir.join("target/release");
    if cfg!(target_os = "linux") {
        target_dir.join("libgameplay.so")
    } else if cfg!(target_os = "macos") {
        target_dir.join("libgameplay.dylib")
    } else {
        target_dir.join("gameplay.dll")
    }
}

/// Generate the gameplay crate from `project_dir/assets/scripts/`.
///
/// Creates `.rkpatch-cache/gameplay/` with Cargo.toml, src/lib.rs, and
/// copies of the user's component and system source files.
///
/// Returns the path to the generated crate directory.
pub fn generate_gameplay_crate(project_dir: &Path) -> Result<PathBuf, ScaffoldError> {
    let scripts_dir = project_dir.join("assets/scripts");
    let crate_dir = gameplay_crate_dir(project_dir);

    let components = discover_rs_files(&scripts_dir.join("components"));
    let systems = discover_rs_files(&scripts_dir.join("systems"));

    // Create directory structure.
    let src_dir = crate_dir.join("src");
    let comp_dir = src_dir.join("components");
    let sys_dir = src_dir.join("systems");
    fs::create_dir_all(&comp_dir)?;
    fs::create_dir_all(&sys_dir)?;

    // Generate Cargo.toml.
    let engine_path = engine_root().join("crates/rkp-engine");
    let macros_path = engine_root().join("crates/rkp-macros");
    write_if_changed(&crate_dir.join("Cargo.toml"), &gen_cargo_toml(
        &engine_path.display().to_string(),
        &macros_path.display().to_string(),
    ))?;

    // Remove stale source files.
    remove_stale(&comp_dir, &components)?;
    remove_stale(&sys_dir, &systems)?;

    // Copy user source files.
    for name in &components {
        copy_if_changed(
            &scripts_dir.join("components").join(format!("{name}.rs")),
            &comp_dir.join(format!("{name}.rs")),
        )?;
    }
    for name in &systems {
        copy_if_changed(
            &scripts_dir.join("systems").join(format!("{name}.rs")),
            &sys_dir.join(format!("{name}.rs")),
        )?;
    }

    // Generate mod.rs files.
    write_if_changed(&comp_dir.join("mod.rs"), &gen_mod_rs(&components, true))?;
    write_if_changed(&sys_dir.join("mod.rs"), &gen_mod_rs(&systems, false))?;

    // Generate lib.rs.
    write_if_changed(&src_dir.join("lib.rs"), &gen_lib_rs(&components, &systems))?;

    Ok(crate_dir)
}

// ── File discovery ──────────────────────────────────────────────────────

fn discover_rs_files(dir: &Path) -> Vec<String> {
    let mut names = Vec::new();
    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_file() && path.extension().and_then(|e| e.to_str()) == Some("rs") {
                if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                    if stem != "mod" {
                        names.push(stem.to_string());
                    }
                }
            }
        }
    }
    names.sort();
    names
}

// ── File helpers ────────────────────────────────────────────────────────

fn write_if_changed(path: &Path, content: &str) -> Result<bool, ScaffoldError> {
    if let Ok(existing) = fs::read_to_string(path) {
        if existing == content {
            return Ok(false);
        }
    }
    fs::write(path, content)?;
    Ok(true)
}

fn copy_if_changed(src: &Path, dst: &Path) -> Result<bool, ScaffoldError> {
    if dst.exists() {
        let src_data = fs::read(src)?;
        if let Ok(dst_data) = fs::read(dst) {
            if src_data == dst_data {
                return Ok(false);
            }
        }
    }
    fs::copy(src, dst)?;
    Ok(true)
}

fn remove_stale(dir: &Path, expected: &[String]) -> Result<(), ScaffoldError> {
    if !dir.is_dir() {
        return Ok(());
    }
    for entry in fs::read_dir(dir)?.flatten() {
        let path = entry.path();
        if !path.is_file() || path.extension().and_then(|e| e.to_str()) != Some("rs") {
            continue;
        }
        let stem = match path.file_stem().and_then(|s| s.to_str()) {
            Some(s) => s,
            None => continue,
        };
        if stem == "mod" {
            continue;
        }
        if !expected.iter().any(|n| n == stem) {
            let _ = fs::remove_file(&path);
        }
    }
    Ok(())
}

// ── Template generators ─────────────────────────────────────────────────

fn gen_cargo_toml(engine_path: &str, macros_path: &str) -> String {
    format!(
        r#"[package]
name = "gameplay"
version = "0.1.0"
edition = "2021"

[lib]
crate-type = ["cdylib"]

[dependencies]
rkp-engine = {{ path = "{engine_path}" }}
rkp-macros = {{ path = "{macros_path}" }}
inventory = "0.3"
glam = {{ version = "0.29", features = ["serde"] }}
hecs = "0.10"
serde = {{ version = "1", features = ["derive"] }}
serde_json = "1"

# Must match editor's release profile — hecs layout changes between debug/release.
[profile.release]
opt-level = 1
"#
    )
}

fn gen_lib_rs(components: &[String], systems: &[String]) -> String {
    let mut s = String::new();
    s.push_str("//! Auto-generated gameplay crate — do not edit.\n");
    s.push_str("//! Write your components in assets/scripts/components/\n");
    s.push_str("//! Write your systems in assets/scripts/systems/\n\n");

    if !components.is_empty() {
        s.push_str("pub mod components;\n");
    }
    if !systems.is_empty() {
        s.push_str("pub mod systems;\n");
    }
    s.push('\n');

    s.push_str("use rkp_engine::component_registry::ComponentEntry;\n");
    s.push_str("use rkp_engine::behavior::SystemEntry;\n\n");

    // Component FFI export.
    s.push_str("#[unsafe(no_mangle)]\n");
    s.push_str("pub extern \"C\" fn rkp_gameplay_entries() -> rkp_engine::gameplay_loader::GameplayEntries {\n");
    s.push_str("    let entries: Vec<&'static ComponentEntry> =\n");
    s.push_str("        inventory::iter::<ComponentEntry>.into_iter().collect();\n");
    s.push_str("    rkp_engine::gameplay_loader::GameplayEntries::from_iter(entries)\n");
    s.push_str("}\n\n");

    // System FFI export.
    s.push_str("#[unsafe(no_mangle)]\n");
    s.push_str("pub extern \"C\" fn rkp_gameplay_systems() -> rkp_engine::gameplay_loader::GameplaySystems {\n");
    s.push_str("    let entries: Vec<&'static SystemEntry> =\n");
    s.push_str("        inventory::iter::<SystemEntry>.into_iter().collect();\n");
    s.push_str("    rkp_engine::gameplay_loader::GameplaySystems::from_iter(entries)\n");
    s.push_str("}\n");

    s
}

fn gen_mod_rs(names: &[String], glob_reexport: bool) -> String {
    let mut s = String::new();
    for name in names {
        s.push_str(&format!("pub mod {name};\n"));
    }
    if glob_reexport && !names.is_empty() {
        s.push('\n');
        for name in names {
            s.push_str(&format!("pub use {name}::*;\n"));
        }
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scaffold_from_scripts() {
        let tmp = tempfile::tempdir().unwrap();
        let project = tmp.path().join("test_project");
        let comp_dir = project.join("assets/scripts/components");
        let sys_dir = project.join("assets/scripts/systems");
        fs::create_dir_all(&comp_dir).unwrap();
        fs::create_dir_all(&sys_dir).unwrap();

        fs::write(comp_dir.join("spin.rs"), "// spin component").unwrap();
        fs::write(comp_dir.join("health.rs"), "// health component").unwrap();
        fs::write(sys_dir.join("spin_system.rs"), "// spin system").unwrap();

        let crate_dir = generate_gameplay_crate(&project).unwrap();

        assert!(crate_dir.join("Cargo.toml").is_file());
        assert!(crate_dir.join("src/lib.rs").is_file());
        assert!(crate_dir.join("src/components/spin.rs").exists());
        assert!(crate_dir.join("src/components/health.rs").exists());
        assert!(crate_dir.join("src/systems/spin_system.rs").exists());

        let lib_rs = fs::read_to_string(crate_dir.join("src/lib.rs")).unwrap();
        assert!(lib_rs.contains("pub mod components;"));
        assert!(lib_rs.contains("pub mod systems;"));
        assert!(lib_rs.contains("rkp_gameplay_entries"));
        assert!(lib_rs.contains("rkp_gameplay_systems"));

        let comp_mod = fs::read_to_string(crate_dir.join("src/components/mod.rs")).unwrap();
        assert!(comp_mod.contains("pub mod health;"));
        assert!(comp_mod.contains("pub mod spin;"));
        assert!(comp_mod.contains("pub use health::*;"));
    }

    #[test]
    fn scaffold_empty_project() {
        let tmp = tempfile::tempdir().unwrap();
        let project = tmp.path().join("empty");
        fs::create_dir_all(project.join("assets/scripts/components")).unwrap();
        fs::create_dir_all(project.join("assets/scripts/systems")).unwrap();

        let crate_dir = generate_gameplay_crate(&project).unwrap();
        let lib_rs = fs::read_to_string(crate_dir.join("src/lib.rs")).unwrap();
        assert!(lib_rs.contains("rkp_gameplay_entries"));
        assert!(!lib_rs.contains("pub mod components;"));
    }

    #[test]
    fn stale_files_removed() {
        let tmp = tempfile::tempdir().unwrap();
        let project = tmp.path().join("stale_test");
        let comp_dir = project.join("assets/scripts/components");
        fs::create_dir_all(&comp_dir).unwrap();
        fs::create_dir_all(project.join("assets/scripts/systems")).unwrap();

        // First scaffold with spin.rs
        fs::write(comp_dir.join("spin.rs"), "// spin").unwrap();
        generate_gameplay_crate(&project).unwrap();
        let crate_dir = gameplay_crate_dir(&project);
        assert!(crate_dir.join("src/components/spin.rs").exists());

        // Delete spin.rs, re-scaffold
        fs::remove_file(comp_dir.join("spin.rs")).unwrap();
        generate_gameplay_crate(&project).unwrap();
        assert!(!crate_dir.join("src/components/spin.rs").exists());
    }
}
