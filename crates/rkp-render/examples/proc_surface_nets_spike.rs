//! Spike runner for the GPU surface-nets-from-SDF pass.
//!
//! Builds a few representative procedurals, runs the extractor at
//! 64³ / 128³ / 256³, prints timings + counts. Set
//! `RKP_SN_DUMP_OBJ=path` to dump the 128³ mesh of each case as a
//! Wavefront .obj for visual sanity checking.
//!
//! Usage:
//!   cargo run -p rkp-render --release --example proc_surface_nets_spike

use glam::{Affine3A, Vec3};
use rkp_procedural::{
    flatten_tree, MaterialCombine, NodeKind, ProceduralObject,
};
use rkp_render::context::RenderContext;
use rkp_render::proc_surface_nets::{GpuSurfaceNets, SurfaceMesh};

fn main() {
    let ctx = RenderContext::new_headless();
    let device = &ctx.device;
    let queue = &ctx.queue;

    let mut sn = GpuSurfaceNets::new(device);
    let dump_dir = std::env::var("RKP_SN_DUMP_OBJ").ok();

    // ── Case A: single sphere (radius 0.5, default origin) ──────────
    let sphere = build_sphere();
    run_case(&mut sn, device, queue, "sphere(r=0.5)", &sphere, dump_dir.as_deref());

    // ── Case B: Tower-ish CSG — base box + 4 corner pillars + cap ──
    let tower = build_tower();
    run_case(&mut sn, device, queue, "tower(7 prims)", &tower, dump_dir.as_deref());

    // ── Case C: NoiseDisplace over the same tower (warps path) ──────
    let noisy = build_noisy_tower();
    run_case(
        &mut sn,
        device,
        queue,
        "tower+noise(displace)",
        &noisy,
        dump_dir.as_deref(),
    );
}

fn run_case(
    sn: &mut GpuSurfaceNets,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    label: &str,
    obj: &ProceduralObject,
    dump_dir: Option<&str>,
) {
    let instructions = flatten_tree(obj);
    let bounds = rkp_procedural::compute_bounds(obj);
    // Pad the AABB by ~5% so the surface doesn't touch the boundary
    // (avoids artefacts from the empty out-of-bounds shell).
    let extent = bounds.max - bounds.min;
    let pad = extent.max_element() * 0.05;
    let aabb_min = bounds.min - Vec3::splat(pad);
    let aabb_max = bounds.max + Vec3::splat(pad);

    println!(
        "\n=== {label} — {} ops, AABB extent ({:.2}, {:.2}, {:.2}) ===",
        instructions.len(),
        (aabb_max - aabb_min).x,
        (aabb_max - aabb_min).y,
        (aabb_max - aabb_min).z,
    );
    println!(
        "{:>5}  {:>9}  {:>9}  {:>9}  {:>9}  {:>9}  {:>9}  {:>9}",
        "N", "verts", "indices", "classify", "vertex", "index", "readback", "total",
    );

    for &grid_n in &[64u32, 128u32, 256u32] {
        // Cap sizing: surface area scales as O(N²). 16× cells² is a
        // safe ceiling for Tower-shaped procedurals; 32× covers more
        // cluttered surfaces like noise-displaced.
        let surface_estimate = (grid_n as u64).pow(2);
        let vertex_cap = (surface_estimate * 16) as u32;
        let index_cap = (surface_estimate * 96) as u32;

        let want_dump = dump_dir.is_some() && grid_n == 128;
        let (mesh, stats) = sn.extract(
            device,
            queue,
            &instructions,
            aabb_min,
            aabb_max,
            grid_n,
            vertex_cap,
            index_cap,
            want_dump,
        );

        println!(
            "{:>5}  {:>9}  {:>9}  {:>7.2}ms  {:>7.2}ms  {:>7.2}ms  {:>7.2}ms  {:>7.2}ms",
            grid_n,
            stats.vertex_count,
            stats.index_count,
            ms(stats.classify),
            ms(stats.vertex_emit),
            ms(stats.index_emit),
            ms(stats.readback),
            ms(stats.total),
        );

        if stats.vertex_count >= vertex_cap {
            println!(
                "  ⚠ vertex_cap ({}) hit — counts truncated. Re-run with bigger cap.",
                vertex_cap
            );
        }
        if stats.index_count >= index_cap {
            println!(
                "  ⚠ index_cap ({}) hit — counts truncated. Re-run with bigger cap.",
                index_cap
            );
        }

        if let (Some(dir), Some(mesh)) = (dump_dir, mesh) {
            // Sanity-check the proxy cluster the renderer would consume.
            let cluster = mesh.single_cluster();
            assert_eq!(cluster.index_count as usize, mesh.indices.len());
            assert_eq!(cluster.aabb_min, aabb_min.to_array());
            assert!(cluster.parent_group_error.is_infinite());

            let safe_label = label.replace([' ', '(', ')', '+', '=', ','], "_");
            let path = format!("{}/{}_n{}.obj", dir.trim_end_matches('/'), safe_label, grid_n);
            match dump_obj(&path, &mesh) {
                Ok(_) => println!("  → dumped {} (cluster: idx_count={}, lod=0)", path, cluster.index_count),
                Err(e) => eprintln!("  ✗ dump failed ({}): {}", path, e),
            }
        }
    }
}

fn ms(d: std::time::Duration) -> f32 {
    d.as_secs_f32() * 1000.0
}

// ── Procedural builders ─────────────────────────────────────────────

fn build_sphere() -> ProceduralObject {
    let mut obj = ProceduralObject::new(NodeKind::Root);
    let _s = obj.add_child(
        obj.root(),
        NodeKind::Sphere(rkp_procedural::node_kind::SphereParams {
            radius: 0.5,
            material_id: 0,
            color: Vec3::ZERO,
        }),
    );
    obj
}

fn build_tower() -> ProceduralObject {
    use rkp_procedural::node_kind::{BoxParams, CylinderParams};
    let mut obj = ProceduralObject::new(NodeKind::Root);
    let union_id = obj.add_child(
        obj.root(),
        NodeKind::Union {
            material_combine: MaterialCombine::Winner,
        },
    );
    // Base
    let base = obj.add_child(
        union_id,
        NodeKind::Box(BoxParams {
            half_extents: Vec3::new(1.0, 0.2, 1.0),
            rounding: 0.05,
            material_id: 0,
            color: Vec3::ZERO,
        }),
    );
    obj.set_transform(base, Affine3A::from_translation(Vec3::new(0.0, 0.0, 0.0)));
    // 4 corner pillars
    for (i, (sx, sz)) in [(-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0), (1.0, 1.0)]
        .iter()
        .enumerate()
    {
        let p = obj.add_child(
            union_id,
            NodeKind::Cylinder(CylinderParams {
                radius: 0.15,
                half_height: 1.0,
                material_id: 0,
                color: Vec3::ZERO,
            }),
        );
        let _ = i;
        obj.set_transform(
            p,
            Affine3A::from_translation(Vec3::new(*sx * 0.85, 1.0, *sz * 0.85)),
        );
    }
    // Cap
    let cap = obj.add_child(
        union_id,
        NodeKind::Box(BoxParams {
            half_extents: Vec3::new(1.1, 0.15, 1.1),
            rounding: 0.05,
            material_id: 0,
            color: Vec3::ZERO,
        }),
    );
    obj.set_transform(cap, Affine3A::from_translation(Vec3::new(0.0, 2.0, 0.0)));
    obj
}

fn build_noisy_tower() -> ProceduralObject {
    use rkp_procedural::node_kind::NoiseDisplaceParams;
    // For the spike: NoiseDisplace at the top wrapping a Tower-like
    // CSG child. Built fresh rather than mutating `build_tower()` —
    // re-parenting via the arena API is more code than the spike needs.
    let mut obj = ProceduralObject::new(NodeKind::Root);
    let nd = obj.add_child(
        obj.root(),
        NodeKind::NoiseDisplace(NoiseDisplaceParams {
            amplitude: 0.05,
            frequency: 4.0,
            seed: 1234,
            octaves: 2,
        }),
    );
    let union_id = obj.add_child(
        nd,
        NodeKind::Union {
            material_combine: MaterialCombine::Winner,
        },
    );
    use rkp_procedural::node_kind::{BoxParams, CylinderParams};
    let base = obj.add_child(
        union_id,
        NodeKind::Box(BoxParams {
            half_extents: Vec3::new(1.0, 0.2, 1.0),
            rounding: 0.05,
            material_id: 0,
            color: Vec3::ZERO,
        }),
    );
    obj.set_transform(base, Affine3A::from_translation(Vec3::new(0.0, 0.0, 0.0)));
    for (sx, sz) in [(-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0), (1.0, 1.0)] {
        let p = obj.add_child(
            union_id,
            NodeKind::Cylinder(CylinderParams {
                radius: 0.15,
                half_height: 1.0,
                material_id: 0,
                color: Vec3::ZERO,
            }),
        );
        obj.set_transform(
            p,
            Affine3A::from_translation(Vec3::new(sx * 0.85, 1.0, sz * 0.85)),
        );
    }
    let cap = obj.add_child(
        union_id,
        NodeKind::Box(BoxParams {
            half_extents: Vec3::new(1.1, 0.15, 1.1),
            rounding: 0.05,
            material_id: 0,
            color: Vec3::ZERO,
        }),
    );
    obj.set_transform(cap, Affine3A::from_translation(Vec3::new(0.0, 2.0, 0.0)));
    obj
}

// ── .obj writer (visual sanity check) ───────────────────────────────

fn dump_obj(path: &str, mesh: &SurfaceMesh) -> std::io::Result<()> {
    use std::fmt::Write as _;
    use std::io::Write as _;
    let mut out = String::with_capacity(mesh.vertices.len() * 32 + mesh.indices.len() * 12);
    writeln!(out, "# proc_surface_nets spike output").unwrap();
    writeln!(
        out,
        "# verts={} tris={}",
        mesh.vertices.len(),
        mesh.indices.len() / 3
    )
    .unwrap();
    for v in &mesh.vertices {
        writeln!(out, "v {} {} {}", v.local_pos[0], v.local_pos[1], v.local_pos[2]).unwrap();
    }
    // .obj indices are 1-based
    for tri in mesh.indices.chunks_exact(3) {
        writeln!(out, "f {} {} {}", tri[0] + 1, tri[1] + 1, tri[2] + 1).unwrap();
    }
    let mut f = std::fs::File::create(path)?;
    f.write_all(out.as_bytes())?;
    Ok(())
}
