//! Bake a single terrain tile and print stats.
//!
//! Phase 1 deliverable smoke test — proves the bake pipeline works
//! end-to-end with the real `FbmTerrainFn` source. Optionally dumps
//! the LOD-0 surface mesh as a `.obj` file for offline viewing.
//!
//! Usage:
//!   cargo run --example bake_one_tile -p arvx-terrain
//!   cargo run --example bake_one_tile -p arvx-terrain -- --obj /tmp/tile.obj
//!   cargo run --example bake_one_tile -p arvx-terrain -- --seed 7 --tile-x 1

use std::env;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;

use arvx_core::mesh_extract::MeshVertex;
use arvx_terrain::{bake_tile, FbmTerrainFn, Terrain, TileKey};

struct Args {
    tile_x: i32,
    tile_y: i32,
    tile_z: i32,
    seed: u32,
    obj_out: Option<PathBuf>,
}

impl Args {
    fn parse() -> Result<Self, String> {
        let mut tile_x = 0i32;
        let mut tile_y = 0i32;
        let mut tile_z = 0i32;
        let mut seed = 42u32;
        let mut obj_out: Option<PathBuf> = None;
        let mut iter = env::args().skip(1);
        while let Some(a) = iter.next() {
            let mut v = || iter.next().ok_or_else(|| format!("{a}: missing value"));
            match a.as_str() {
                "--tile-x" => tile_x = v()?.parse().map_err(|e| format!("--tile-x: {e}"))?,
                "--tile-y" => tile_y = v()?.parse().map_err(|e| format!("--tile-y: {e}"))?,
                "--tile-z" => tile_z = v()?.parse().map_err(|e| format!("--tile-z: {e}"))?,
                "--seed" => seed = v()?.parse().map_err(|e| format!("--seed: {e}"))?,
                "--obj" => obj_out = Some(PathBuf::from(v()?)),
                "-h" | "--help" => {
                    print_help();
                    std::process::exit(0);
                }
                other => return Err(format!("unknown argument: {other}")),
            }
        }
        Ok(Self { tile_x, tile_y, tile_z, seed, obj_out })
    }
}

fn print_help() {
    eprintln!(
        "bake_one_tile — Phase 1 smoke test for arvx-terrain\n\
         \n\
         Options:\n  \
         --tile-x N        Tile X coord (default 0)\n  \
         --tile-y N        Tile Y coord (default 0)\n  \
         --tile-z N        Tile Z coord (default 0)\n  \
         --seed N          FBM seed (default 42)\n  \
         --obj <path>      Dump LOD-0 mesh as .obj for offline viewing\n  \
         -h / --help       This message"
    );
}

fn main() {
    let args = match Args::parse() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("error: {e}");
            print_help();
            std::process::exit(2);
        }
    };

    let terrain = Terrain::default();
    let voxel_size = terrain.voxel_size_for_level(0);
    let key = TileKey::level0(args.tile_x, args.tile_y, args.tile_z);
    let fbm = FbmTerrainFn {
        seed: args.seed,
        ..Default::default()
    }
    .resolve(&arvx_core::NullMaterialLookup);

    println!(
        "Baking tile ({}, {}, {}) at level 0 — voxel_size = {} m, extent = {} m",
        key.x,
        key.y,
        key.z,
        voxel_size,
        key.extent_m(),
    );

    let baked = match bake_tile(
        key,
        voxel_size,
        &fbm,
        &[],
        &arvx_terrain::TerrainRegionSnapshot::new(),
    ) {
        Some(b) => b,
        None => {
            eprintln!("bake failed");
            std::process::exit(1);
        }
    };

    println!("\n=== bake stats ===");
    println!("  bake time:       {:.1} ms", baked.bake_time_ms);
    println!("  octree depth:    {}", baked.artifact.octree.depth());
    println!("  octree nodes:    {}", baked.artifact.octree.node_count());
    println!("  voxel count:     {}", baked.artifact.voxel_count);
    println!("  leaf attrs:      {}", baked.artifact.leaf_attrs.len());
    println!("  brick cells:     {}", baked.artifact.brick_cells.len());
    println!("  mesh vertices:   {}", baked.vertex_count());
    println!("  mesh indices:    {}", baked.index_count());
    println!("  mesh triangles:  {}", baked.index_count() / 3);
    println!("  cluster count:   {}", baked.cluster_count());
    println!("  LOD-0 indices:   {}", baked.mesh.lod0_index_count);

    if let Some(path) = args.obj_out {
        match write_obj(&path, &baked.mesh.vertices, &baked.mesh.indices, baked.mesh.lod0_index_count) {
            Ok(_) => println!("\nWrote LOD-0 mesh to {}", path.display()),
            Err(e) => {
                eprintln!("\nfailed to write .obj: {e}");
                std::process::exit(1);
            }
        }
    }
}

/// Dump LOD-0 vertices + indices as a plain `.obj`. Only positions are
/// emitted; normals/UVs/materials are skipped — purpose is offline
/// geometry inspection in Blender/MeshLab.
fn write_obj(path: &PathBuf, vertex_bytes: &[u8], index_bytes: &[u8], lod0_index_count: u32) -> Result<(), String> {
    let vertices: &[MeshVertex] =
        bytemuck::cast_slice(vertex_bytes);
    let indices: &[u32] = bytemuck::cast_slice(index_bytes);
    let lod0_end = (lod0_index_count as usize).min(indices.len());
    let lod0_indices = &indices[..lod0_end];

    let f = File::create(path).map_err(|e| format!("create: {e}"))?;
    let mut w = BufWriter::new(f);
    writeln!(w, "# arvx-terrain bake_one_tile LOD-0 dump").map_err(|e| e.to_string())?;
    for v in vertices {
        writeln!(w, "v {} {} {}", v.local_pos[0], v.local_pos[1], v.local_pos[2])
            .map_err(|e| e.to_string())?;
    }
    for tri in lod0_indices.chunks_exact(3) {
        // OBJ is 1-indexed.
        writeln!(w, "f {} {} {}", tri[0] + 1, tri[1] + 1, tri[2] + 1)
            .map_err(|e| e.to_string())?;
    }
    w.flush().map_err(|e| e.to_string())?;
    Ok(())
}
