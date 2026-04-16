//! `rkp-convert` — command-line front-end for `rkp-import`.
//!
//! Takes a mesh file (`.glb`, `.gltf`, `.obj`, `.fbx`), runs the
//! opacity-octree bake pipeline, and writes an `.rkp` file (plus a
//! `.rkskel` sidecar if the source contained a skeleton). Progress
//! streams to stderr; the result summary prints to stdout on success.

use std::path::PathBuf;
use std::process::ExitCode;

use clap::Parser;

use rkp_import::{ImportConfig, StderrReporter, import_mesh_to_opacity_rkp_with};

#[derive(Parser, Debug)]
#[command(name = "rkp-convert")]
#[command(about = "Convert a mesh file to a .rkp opacity-octree asset")]
struct Args {
    /// Source mesh file (.glb / .gltf / .obj / .fbx).
    input: PathBuf,

    /// Output .rkp path. Defaults to `<input>.rkp` next to the source.
    #[arg(short, long)]
    output: Option<PathBuf>,

    /// Finest voxel size in metres. Omit to auto-detect from mesh extent.
    #[arg(long)]
    voxel_size: Option<f32>,

    /// Normalize longest axis to this size in metres. Default 1.0.
    #[arg(long, default_value_t = 1.0)]
    target_size: f32,

    /// Skip normalization — keep original mesh coordinates.
    #[arg(long)]
    no_normalize: bool,

    /// Force a single material ID for every voxel.
    #[arg(long)]
    material_id: Option<u16>,

    /// Skip per-voxel colour sampling from albedo textures.
    #[arg(long)]
    no_colors: bool,

    /// Euler rotation offset in degrees: `X,Y,Z`.
    #[arg(long, value_parser = parse_rotation, default_value = "0,0,0")]
    rotation: [f32; 3],

    /// Uniform scale multiplier applied after normalization.
    #[arg(long)]
    scale: Option<f32>,
}

fn parse_rotation(s: &str) -> Result<[f32; 3], String> {
    let parts: Vec<&str> = s.split(',').collect();
    if parts.len() != 3 {
        return Err(format!("rotation must be X,Y,Z; got {parts:?}"));
    }
    let mut out = [0.0f32; 3];
    for (i, p) in parts.iter().enumerate() {
        out[i] = p.trim().parse::<f32>().map_err(|e| e.to_string())?;
    }
    Ok(out)
}

fn main() -> ExitCode {
    let args = Args::parse();

    let output = args
        .output
        .clone()
        .unwrap_or_else(|| args.input.with_extension("rkp"));

    let config = ImportConfig {
        voxel_size: args.voxel_size,
        target_size: args.target_size,
        no_normalize: args.no_normalize,
        material_id_override: args.material_id,
        import_colors: !args.no_colors,
        rotation_offset: args.rotation,
        scale_override: args.scale,
    };

    match import_mesh_to_opacity_rkp_with(&args.input, &output, &config, &StderrReporter) {
        Ok(result) => {
            println!(
                "Wrote {} ({} shell voxels, {:.1} KiB, voxel_size={:.4} m)",
                output.display(),
                result.shell_voxels,
                result.file_size as f64 / 1024.0,
                result.finest_voxel_size,
            );
            if let Some(skel) = result.skeleton_path {
                println!("Wrote skeleton: {}", skel.display());
            }
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("rkp-convert: {e}");
            ExitCode::FAILURE
        }
    }
}
