//! `arvx-convert` — command-line front-end for `arvx-import`.
//!
//! Takes a mesh file (`.glb`, `.gltf`, `.obj`, `.fbx`), runs the
//! opacity-octree bake pipeline, and writes an `.arvx` file (plus a
//! `.arvxskel` sidecar if the source contained a skeleton). Progress
//! streams to stderr; the result summary prints to stdout on success.

use std::path::PathBuf;
use std::process::ExitCode;

use clap::Parser;

use arvx_import::{ImportConfig, StderrReporter, import_mesh_to_opacity_rkp_with};

#[derive(Parser, Debug)]
#[command(name = "arvx-convert")]
#[command(about = "Convert a mesh file to a .arvx opacity-octree asset")]
struct Args {
    /// Source mesh file (.glb / .gltf / .obj / .fbx).
    input: PathBuf,

    /// Output .arvx path. Defaults to `<input>.arvx` next to the source.
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

    /// Rebuild the surface mesh + cluster DAG of an existing .arvx in
    /// place, preserving its octree and voxel data. Works on both v4
    /// (which had no mesh sections) and v5 files (replaces the
    /// existing mesh sections). Use when the source mesh is
    /// unavailable (procedural bakes, generator outputs in
    /// `assets/converted/`) or after a DAG-builder change to re-bake
    /// without re-voxelizing — much faster than a full re-import.
    #[arg(long, alias = "upgrade-v4")]
    rebuild_mesh: bool,

    /// Downgrade a v6 .arvx in place to v5: drop the three trailing
    /// DAG-topology sections and rewrite the header with version=5
    /// and `dag_*_compressed_size=0`. All other sections (octree,
    /// voxels, normals, bricks, color, bone, mesh_*) pass through
    /// verbatim — no re-bake. Used by the perf bisect against the
    /// pre-v6 editor. Fast (~ms per file).
    #[arg(long)]
    downgrade_to_v5: bool,
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
                eprintln!("arvx-convert --rebuild-mesh: {e}");
                ExitCode::FAILURE
            }
        };
    }

    if args.downgrade_to_v5 {
        return match downgrade_v6_to_v5(&args.input) {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => {
                eprintln!("arvx-convert --downgrade-to-v5: {e}");
                ExitCode::FAILURE
            }
        };
    }

    let output = args
        .output
        .clone()
        .unwrap_or_else(|| args.input.with_extension("arvx"));

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
            eprintln!("arvx-convert: {e}");
            ExitCode::FAILURE
        }
    }
}

/// Rebuild the mesh + cluster DAG sections of a .arvx in place,
/// preserving everything else. Reads every existing section,
/// synthesises the LeafAttr Vec the mesh extractor needs, re-runs
/// `extract_surface_mesh` + `build_cluster_dag` over the file-local
/// pools, and writes the result back with fresh mesh sections.
/// Skips voxelization — purely a re-encoding pass. Atomic: writes to
/// `<path>.inprogress`, renames on success. Works on v4 files (which
/// had no mesh sections — the readers no-op when size is 0) and on
/// v5 files (replaces the existing mesh sections).
fn rebuild_mesh_sections(path: &std::path::Path) -> Result<(), String> {
    use arvx_core::asset_file::{
        build_mesh_sections_blob, read_rkp_bricks, read_rkp_color, read_rkp_header,
        read_rkp_mesh_indices, read_rkp_mesh_vertices, read_rkp_meshlet_clusters,
        read_rkp_normals, read_rkp_octree, read_rkp_skin_meta, read_rkp_voxels,
        write_rkp_with_progress, ArvxHeader, SkinMetaIn,
    };
    use arvx_core::leaf_attr::LeafAttr;

    let t0 = std::time::Instant::now();

    let mut file = std::fs::File::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
    let mut reader = std::io::BufReader::new(&mut file);
    let header: ArvxHeader = read_rkp_header(&mut reader)
        .map_err(|e| format!("read header: {e}"))?;
    // `--rebuild-mesh` rewrites the mesh sections (and re-bakes as the
    // v5-style blob, dropping DAG); it does NOT carry the v7 per-leaf
    // distance section through. Before distances were ever persisted
    // that was harmless, but a re-baked distance-carrying asset (terrain
    // `.arvxtile`, or any DC bake) loses its stored field and silently
    // reverts to blur re-extract / sculpt on the next load. Warn loudly
    // rather than drop it silently — re-bake from source to keep
    // Manifold-DC. (Preserving/regenerating distances through this tool
    // is tracked with the broader arvx-convert / arvx-import distance
    // support, which is entangled with this path's v5 downgrade.)
    if header.distance_compressed_size > 0 {
        eprintln!(
            "[rebuild-mesh] {}: WARNING — input carries a v7 per-leaf distance \
             section ({} compressed bytes) that --rebuild-mesh does NOT preserve; \
             the rebuilt asset will fall back to blur re-extract/sculpt. Re-bake \
             from source to retain Manifold-DC.",
            path.display(),
            header.distance_compressed_size,
        );
    }
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
    let bone_voxels: &[arvx_core::companion::BoneVoxel] =
        if skin_meta_decoded.bone_voxels.len() >= std::mem::size_of::<arvx_core::companion::BoneVoxel>() {
            bytemuck::cast_slice(&skin_meta_decoded.bone_voxels)
        } else {
            &[]
        };

    let mesh_blob = build_mesh_sections_blob(
        &octree_nodes,
        header.octree_depth as u8,
        header.base_voxel_size,
        grid_origin,
        bricks_u32,
        &leaf_attrs,
        bone_voxels,
    );
    if mesh_blob.vertices.is_empty() {
        eprintln!(
            "[rebuild-mesh] {}: no surface mesh extracted (degenerate octree), wrote without mesh sections",
            path.display()
        );
    }
    let mesh_sections = if !mesh_blob.vertices.is_empty() {
        Some(mesh_blob.as_in())
    } else {
        None
    };

    let skin_meta_in = if header.flags & arvx_core::asset_file::FLAG_HAS_BONES != 0
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
            None, // distance_data
            None, // progress
        )
        .map_err(|e| format!("write v5: {e}"))?;
    }
    std::fs::rename(&tmp, path).map_err(|e| {
        let _ = std::fs::remove_file(&tmp);
        format!("rename: {e}")
    })?;

    let cluster_count = mesh_blob.clusters.len() / 64;
    eprintln!(
        "[rebuild-mesh] {}: in {:.2}s ({} clusters)",
        path.display(),
        t0.elapsed().as_secs_f32(),
        cluster_count,
    );
    Ok(())
}

/// Downgrade a v6 `.arvx` in place to v5 by reading the existing
/// compressed sections verbatim and re-emitting the file with the
/// three trailing DAG sections dropped + header `version=5` +
/// `dag_*_compressed_size=0`. All non-DAG sections (octree, voxels,
/// normals, bricks, color, bone, mesh_*) pass through unchanged —
/// no re-bake. The file rename is atomic via `.inprogress` tmp.
fn downgrade_v6_to_v5(path: &std::path::Path) -> Result<(), String> {
    use std::io::{Read, Seek, SeekFrom, Write};

    let t0 = std::time::Instant::now();

    let mut file = std::fs::File::open(path)
        .map_err(|e| format!("open {}: {e}", path.display()))?;

    // Read first 8 bytes to peek version, then rewind.
    let mut prefix = [0u8; 8];
    file.read_exact(&mut prefix)
        .map_err(|e| format!("read prefix: {e}"))?;
    let magic: [u8; 4] = prefix[0..4].try_into().unwrap();
    if magic != arvx_core::asset_file::ARVX_MAGIC {
        return Err("not a .arvx file (bad magic)".into());
    }
    let version = u32::from_le_bytes(prefix[4..8].try_into().unwrap());
    if version == 5 {
        eprintln!(
            "[downgrade-to-v5] {}: already v5, skipping",
            path.display(),
        );
        return Ok(());
    }
    if version != 6 {
        return Err(format!("unsupported version {version} (only 6 → 5 is supported)"));
    }

    // Read full v6 header (156 B).
    file.seek(SeekFrom::Start(0))
        .map_err(|e| format!("seek 0: {e}"))?;
    const V6_HEADER_SIZE: usize = std::mem::size_of::<arvx_core::asset_file::ArvxHeader>();
    let mut v6_header_bytes = vec![0u8; V6_HEADER_SIZE];
    file.read_exact(&mut v6_header_bytes)
        .map_err(|e| format!("read v6 header: {e}"))?;
    let v6_header: arvx_core::asset_file::ArvxHeader =
        *bytemuck::from_bytes(&v6_header_bytes);

    // Compressed-section sizes from header, in file order.
    // Order matches the writer: octree, voxels, normals, bricks,
    // color, bone, mesh_vertices, mesh_indices, meshlet_clusters,
    // dag_groups, dag_consumed, dag_produced.
    let octree_sz = v6_header.octree_compressed_size as u64;
    let voxel_sz = v6_header.voxel_compressed_size as u64;
    let normals_sz = v6_header.normals_compressed_size as u64;
    let bricks_sz = v6_header.bricks_compressed_size as u64;
    let color_sz = v6_header.color_compressed_size as u64;
    let bone_sz = v6_header.bone_compressed_size as u64;
    let mesh_v_sz = v6_header.mesh_vertices_compressed_size as u64;
    let mesh_i_sz = v6_header.mesh_indices_compressed_size as u64;
    let mesh_c_sz = v6_header.meshlet_clusters_compressed_size as u64;
    // DAG sections — dropped on downgrade. Sizes ignored.

    // Read all 9 non-DAG sections as opaque bytes.
    let mut read_section = |n: u64, label: &str| -> Result<Vec<u8>, String> {
        if n == 0 {
            return Ok(Vec::new());
        }
        let mut buf = vec![0u8; n as usize];
        file.read_exact(&mut buf)
            .map_err(|e| format!("read {label}: {e}"))?;
        Ok(buf)
    };
    let octree_bytes = read_section(octree_sz, "octree")?;
    let voxel_bytes = read_section(voxel_sz, "voxels")?;
    let normals_bytes = read_section(normals_sz, "normals")?;
    let bricks_bytes = read_section(bricks_sz, "bricks")?;
    let color_bytes = read_section(color_sz, "color")?;
    let bone_bytes = read_section(bone_sz, "bone")?;
    let mesh_v_bytes = read_section(mesh_v_sz, "mesh vertices")?;
    let mesh_i_bytes = read_section(mesh_i_sz, "mesh indices")?;
    let mesh_c_bytes = read_section(mesh_c_sz, "meshlet clusters")?;
    drop(file);

    // Build v5 header by mutating a copy of the v6 header: version=5
    // and zero out the three DAG-section sizes. The v5 reader only
    // reads the first 144 bytes; the trailing 12 bytes of v6 header
    // become *omitted* output bytes.
    let mut v5_header = v6_header;
    v5_header.version = 5;
    v5_header.dag_groups_compressed_size = 0;
    v5_header.dag_consumed_compressed_size = 0;
    v5_header.dag_produced_compressed_size = 0;
    let v5_header_bytes = bytemuck::bytes_of(&v5_header);
    const V5_HEADER_SIZE: usize = 144;
    let v5_header_prefix = &v5_header_bytes[..V5_HEADER_SIZE];

    // Write atomically via .inprogress.
    let tmp = {
        let mut s = path.as_os_str().to_owned();
        s.push(".inprogress");
        std::path::PathBuf::from(s)
    };
    let _ = std::fs::remove_file(&tmp);
    {
        let f = std::fs::File::create(&tmp)
            .map_err(|e| format!("create {}: {e}", tmp.display()))?;
        let mut w = std::io::BufWriter::new(f);
        w.write_all(v5_header_prefix)
            .map_err(|e| format!("write v5 header: {e}"))?;
        for (chunk, label) in [
            (&octree_bytes, "octree"),
            (&voxel_bytes, "voxels"),
            (&normals_bytes, "normals"),
            (&bricks_bytes, "bricks"),
            (&color_bytes, "color"),
            (&bone_bytes, "bone"),
            (&mesh_v_bytes, "mesh vertices"),
            (&mesh_i_bytes, "mesh indices"),
            (&mesh_c_bytes, "meshlet clusters"),
        ] {
            if !chunk.is_empty() {
                w.write_all(chunk)
                    .map_err(|e| format!("write {label}: {e}"))?;
            }
        }
        w.flush().map_err(|e| format!("flush: {e}"))?;
    }
    std::fs::rename(&tmp, path).map_err(|e| {
        let _ = std::fs::remove_file(&tmp);
        format!("rename: {e}")
    })?;

    eprintln!(
        "[downgrade-to-v5] {}: in {:.3}s",
        path.display(),
        t0.elapsed().as_secs_f32(),
    );
    Ok(())
}
