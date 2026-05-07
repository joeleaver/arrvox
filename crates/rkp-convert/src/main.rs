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

    /// Rebuild the surface mesh + cluster DAG of an existing .rkp in
    /// place, preserving its octree and voxel data. Works on both v4
    /// (which had no mesh sections) and v5 files (replaces the
    /// existing mesh sections). Use when the source mesh is
    /// unavailable (procedural bakes, generator outputs in
    /// `assets/converted/`) or after a DAG-builder change to re-bake
    /// without re-voxelizing — much faster than a full re-import.
    #[arg(long, alias = "upgrade-v4")]
    rebuild_mesh: bool,
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

    if args.rebuild_mesh {
        return match rebuild_mesh_sections(&args.input) {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => {
                eprintln!("rkp-convert --rebuild-mesh: {e}");
                ExitCode::FAILURE
            }
        };
    }

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

/// Rebuild the mesh + cluster DAG sections of a .rkp in place,
/// preserving everything else. Reads every existing section,
/// synthesises the LeafAttr Vec the mesh extractor needs, re-runs
/// `extract_surface_mesh` + `build_cluster_dag` over the file-local
/// pools, and writes the result back with fresh mesh sections.
/// Skips voxelization — purely a re-encoding pass. Atomic: writes to
/// `<path>.inprogress`, renames on success. Works on v4 files (which
/// had no mesh sections — the readers no-op when size is 0) and on
/// v5 files (replaces the existing mesh sections).
fn rebuild_mesh_sections(path: &std::path::Path) -> Result<(), String> {
    use rkp_core::asset_file::{
        build_mesh_sections_blob, read_rkp_bricks, read_rkp_color, read_rkp_header,
        read_rkp_mesh_indices, read_rkp_mesh_vertices, read_rkp_meshlet_clusters,
        read_rkp_normals, read_rkp_octree, read_rkp_skin_meta, read_rkp_voxels,
        write_rkp_with_progress, MeshSectionsIn, RkpHeader, SkinMetaIn,
    };
    use rkp_core::leaf_attr::LeafAttr;

    let t0 = std::time::Instant::now();

    let mut file = std::fs::File::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
    let mut reader = std::io::BufReader::new(&mut file);
    let header: RkpHeader = read_rkp_header(&mut reader)
        .map_err(|e| format!("read header: {e}"))?;
    let octree_nodes = read_rkp_octree(&mut reader, &header)
        .map_err(|e| format!("read octree: {e}"))?;
    let voxel_data = read_rkp_voxels(&mut reader, &header)
        .map_err(|e| format!("read voxels: {e}"))?;
    let normals_data = read_rkp_normals(&mut reader, &header)
        .map_err(|e| format!("read normals: {e}"))?;
    let bricks_data = read_rkp_bricks(&mut reader, &header)
        .map_err(|e| format!("read bricks: {e}"))?;
    let color_data = read_rkp_color(&mut reader, &header)
        .map_err(|e| format!("read color: {e}"))?;
    let skin_meta_decoded = read_rkp_skin_meta(&mut reader, &header)
        .map_err(|e| format!("read skin: {e}"))?;
    // Drain the existing mesh sections so the read cursor advances
    // past them — needed on v5 files; v4 readers no-op when size=0.
    let _ = read_rkp_mesh_vertices(&mut reader, &header);
    let _ = read_rkp_mesh_indices(&mut reader, &header);
    let _ = read_rkp_meshlet_clusters(&mut reader, &header);
    drop(reader);

    // Synthesise the LeafAttr Vec the extractor reads. It only
    // touches `normal_oct`, so material fields can be zero.
    let normals_u32: &[u32] = if normals_data.is_empty() {
        &[]
    } else {
        bytemuck::cast_slice::<u8, u32>(&normals_data)
    };
    let leaf_attrs: Vec<LeafAttr> = (0..header.voxel_count as usize)
        .map(|i| LeafAttr {
            normal_oct: normals_u32.get(i).copied().unwrap_or(0),
            material_primary: 0,
            material_secondary_blend: 0,
        })
        .collect();
    let bricks_u32: &[u32] = if bricks_data.is_empty() {
        &[]
    } else {
        bytemuck::cast_slice::<u8, u32>(&bricks_data)
    };

    let asset_extent = (1u32 << header.octree_depth) as f32 * header.base_voxel_size;
    let aabb_center = (glam::Vec3::from(header.aabb_min) + glam::Vec3::from(header.aabb_max))
        * 0.5;
    let grid_origin = aabb_center - glam::Vec3::splat(asset_extent * 0.5);

    // Decode the file's BoneVoxel quads from the skin-meta payload —
    // the upgrade re-bakes the surface mesh, so it has to feed the
    // extractor the same per-cell bone weights the original import had,
    // otherwise skinned assets that go through `--upgrade-v4` would
    // ship rest-pose vertices in v5.
    let bone_voxels: &[rkp_core::companion::BoneVoxel] =
        if skin_meta_decoded.bone_voxels.len() >= std::mem::size_of::<rkp_core::companion::BoneVoxel>() {
            bytemuck::cast_slice(&skin_meta_decoded.bone_voxels)
        } else {
            &[]
        };

    let (mesh_v_bytes, mesh_i_bytes, meshlet_bytes, lod0_index_count) =
        build_mesh_sections_blob(
            &octree_nodes,
            header.octree_depth as u8,
            header.base_voxel_size,
            grid_origin,
            bricks_u32,
            &leaf_attrs,
            bone_voxels,
        );
    if mesh_v_bytes.is_empty() {
        eprintln!(
            "[rebuild-mesh] {}: no surface mesh extracted (degenerate octree), wrote v5 without mesh sections",
            path.display()
        );
    }
    let mesh_sections = if !mesh_v_bytes.is_empty() {
        Some(MeshSectionsIn {
            vertices: &mesh_v_bytes,
            indices: &mesh_i_bytes,
            clusters: &meshlet_bytes,
            lod0_index_count,
        })
    } else {
        None
    };

    let skin_meta_in = if header.flags & rkp_core::asset_file::FLAG_HAS_BONES != 0
        && !skin_meta_decoded.bone_voxels.is_empty()
    {
        Some(SkinMetaIn {
            bone_voxels: &skin_meta_decoded.bone_voxels,
            brick_origins: &skin_meta_decoded.brick_origins,
            rest_bone_aabbs: &skin_meta_decoded.rest_bone_aabbs,
        })
    } else {
        None
    };

    let normals_opt = if normals_data.is_empty() { None } else { Some(normals_data.as_slice()) };
    let bricks_opt = if bricks_data.is_empty() { None } else { Some(bricks_data.as_slice()) };
    let color_opt = if color_data.is_empty() { None } else { Some(color_data.as_slice()) };

    let mat_ids: Vec<u16> = header
        .material_ids
        .iter()
        .copied()
        .take_while(|&id| id != 0 || header.material_ids[0] == 0)
        .collect();

    let mut tmp = path.as_os_str().to_owned();
    tmp.push(".inprogress");
    let tmp = std::path::PathBuf::from(tmp);
    let _ = std::fs::remove_file(&tmp);

    {
        let f = std::fs::File::create(&tmp).map_err(|e| format!("create tmp: {e}"))?;
        let mut writer = std::io::BufWriter::new(f);
        write_rkp_with_progress(
            &mut writer,
            &octree_nodes,
            header.octree_depth as u8,
            header.base_voxel_size,
            header.voxel_count,
            header.aabb_min,
            header.aabb_max,
            &mat_ids,
            &voxel_data,
            normals_opt,
            bricks_opt,
            color_opt,
            skin_meta_in,
            mesh_sections,
            None,
        )
        .map_err(|e| format!("write v5: {e}"))?;
    }
    std::fs::rename(&tmp, path).map_err(|e| {
        let _ = std::fs::remove_file(&tmp);
        format!("rename: {e}")
    })?;

    let cluster_count = meshlet_bytes.len() / 64;
    eprintln!(
        "[rebuild-mesh] {}: in {:.2}s ({} clusters)",
        path.display(),
        t0.elapsed().as_secs_f32(),
        cluster_count,
    );
    Ok(())
}
