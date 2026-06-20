//! Geometry-truth repro of the in-editor terrain-sculpt "double-sheet" z-fight.
//!
//! Reproduces, on a flat voxelized slab and at EDITOR-DEFAULT brush params, the
//! exact editor sculpt sequence that produces two coincident surface sheets:
//!
//!   1. voxelize a flat ground slab (with per-leaf distances → DC/QEF)
//!   2. extract the FULL pre-edit mesh = the "old tris"
//!   3. apply an Inflate stamp through the REAL brush kernel
//!   4. SphereTouch-filter the old tris by `op.radius` (mirrors the editor —
//!      remesh_region.rs RemeshFilter::SphereTouch)
//!   5. re-extract the PATCH over the brush cell range (mirrors the editor)
//!   6. detect cross-source coincident sheets: an up-facing kept-old ground
//!      triangle and an up-facing patch triangle stacked in the SAME xz column
//!
//! The bug: the patch re-extract emits tris up to ~1 voxel PAST region_max
//! (mesh_extract.rs:1548-1553), so the dome-rim/ground annulus lands in the
//! kept-old ground ring that SphereTouch (keyed on `op.radius`) did not drop →
//! two ground sheets at ~the same place → z-fight in the editor.
//!
//! IMPORTANT: uses editor-default brush params (radius≈2 vox / strength≈8), NOT
//! the inverted radius=8/strength=6 of the `sculpt_geom_truth` example — the bug
//! is regime-specific and the example's params mask it.
//!
//! This is the editor-path repro the z-fight saga lacked: it exercises the real
//! filter+patch geometry headlessly. The fix is to make the SphereTouch drop and
//! the patch re-extract cover the SAME reach so the annulus can't double up.

use arvx_core::mesh_extract::{
    collect_cell_map_in_region, extract_mesh_region_from_cells_pooled_haloed,
    extract_surface_mesh_density_haloed, MeshVertex, SculptExtractScratch,
};
use arvx_core::sculpt::{
    apply_delta, brush_cell_range, compute_brush_edits_in_stroke, BrushMode, BrushOp, FalloffCurve,
};
use arvx_core::voxelize_octree::voxelize_octree;
use arvx_core::{Aabb, BrickPool, LeafAttrPool};
use glam::{IVec3, Vec3};
use std::collections::{HashMap, HashSet};

const VS: f32 = 0.25;
const N: u32 = 64; // 64³ grid (2^6), 16 m tile

/// Octahedral normal pack (local copy — matches `arvx_core::leaf_attr::pack_oct`,
/// which may be private; mirrors the `sculpt_geom_truth` example).
fn pack_oct(n: Vec3) -> u32 {
    let n = n / (n.x.abs() + n.y.abs() + n.z.abs()).max(1e-6);
    let (mut x, mut y) = (n.x, n.y);
    if n.z < 0.0 {
        let ox = (1.0 - y.abs()) * if x >= 0.0 { 1.0 } else { -1.0 };
        let oy = (1.0 - x.abs()) * if y >= 0.0 { 1.0 } else { -1.0 };
        x = ox;
        y = oy;
    }
    let xi = (x.clamp(-1.0, 1.0) * 32767.0).round() as i32 as i16 as u16 as u32;
    let yi = (y.clamp(-1.0, 1.0) * 32767.0).round() as i32 as i16 as u16 as u32;
    xi | (yi << 16)
}

/// Per-triangle (centroid, face-normal) for the given mesh.
fn tris(verts: &[MeshVertex], idx: &[u32]) -> Vec<(Vec3, Vec3)> {
    idx.chunks_exact(3)
        .filter_map(|t| {
            let p0 = Vec3::from(verts[t[0] as usize].local_pos);
            let p1 = Vec3::from(verts[t[1] as usize].local_pos);
            let p2 = Vec3::from(verts[t[2] as usize].local_pos);
            let n = (p1 - p0).cross(p2 - p0).normalize_or_zero();
            if n == Vec3::ZERO {
                None
            } else {
                Some(((p0 + p1 + p2) / 3.0, n))
            }
        })
        .collect()
}

// CAPTURED OPEN BUG. RED today: the editor's SphereTouch(op.radius) drop leaves a
// sub-voxel-offset patch/kept-old overlap on sloped surfaces (z-fight). Un-ignore
// once the drop reach is unified with the patch re-extract reach in the editor
// (arvx_scene_manager/sculpt.rs::rebuild_dirty_clusters) + the test mirror updated.
#[ignore = "reproduces open bug: sculpt double-sheet z-fight on slopes (48 distinct coincident sheets); un-ignore when the SphereTouch drop reach is unified with the patch re-extract reach"]
#[test]
fn terrain_inflate_does_not_double_sheet() {
    // ── 1. Flat ground slab with per-leaf distances. sdf = y - ground.
    let extent = N as f32 * VS;
    let ground = extent * 0.5;
    let aabb = Aabb::new(Vec3::ZERO, Vec3::splat(extent));
    // Sloped surface. A FLAT slab gives bit-identical region/full DC placement
    // (coincident but harmless — no z-fight); a slope is where region-vs-full
    // sub-voxel placement can diverge (the real editor regime). Brush centre x =
    // tile centre, so the surface height there is exactly `ground`.
    let slope = 0.35_f32;
    let cx = extent * 0.5;
    let mut sdf = |ps: &[Vec3]| -> Vec<(f32, u16, u16, u8, u32, Option<Vec3>)> {
        ps.iter().map(|p| (p.y - (ground + slope * (p.x - cx)), 0u16, 0u16, 0u8, 0u32, None)).collect()
    };
    let mut leaf = LeafAttrPool::new(1 << 16);
    let mut bricks = BrickPool::new(1 << 12);
    let res = voxelize_octree(&mut sdf, &aabb, VS, &mut leaf, &mut bricks, 0).expect("voxelize");
    let mut octree = res.octree;
    let grid_origin = res.grid_origin;
    let depth = octree.depth();

    // ── 2. Full pre-edit extract = the "old tris".
    let (old_verts, old_idx) = extract_surface_mesh_density_haloed(
        octree.as_slice(),
        depth,
        VS,
        grid_origin,
        bricks.as_slice(),
        leaf.as_slice(),
        leaf.bones_as_slice(),
        &[],
        0,
        None,
        leaf.dists_as_slice(),
        None,
    );

    // ── 3. Inflate stamp at the surface centre, EDITOR DEFAULTS.
    let radius = 2.0_f32; // voxels — editor default (NOT the example's 8)
    let strength = 8.0_f32; // finest-voxel amplitude — editor default
    let cg = Vec3::splat(N as f32 * 0.5); // (32,32,32): on the ground at the centre
    let op = BrushOp {
        center: cg,
        segment_start: cg,
        radius,
        falloff_curve: FalloffCurve::Smooth,
        strength,
        mode: BrushMode::Inflate,
        material: 0,
    };
    // Editor-faithful patch region = the brush cell range (sculpt.rs:1045),
    // NOT a hand-rolled radius+2 box. The +1 extract pad is internal to
    // extract_mesh_region; the SphereTouch drop below uses op.radius. The
    // mismatch between this region's reach and the drop radius is the bug.
    let bextent = 1u32 << depth;
    let (blo, bhi) = brush_cell_range(&op, bextent);
    let rmin = IVec3::new(blo.x as i32, blo.y as i32, blo.z as i32).max(IVec3::ZERO);
    let rmax = IVec3::new(bhi.x as i32, bhi.y as i32, bhi.z as i32).min(IVec3::splat(bextent as i32));
    let delta = compute_brush_edits_in_stroke(
        &octree,
        &bricks,
        leaf.as_slice(),
        leaf.dists_as_slice(),
        op,
        |_| false,
    );
    assert!(!delta.is_empty(), "Inflate produced no edits — brush placement is off the surface");

    let n_added = delta.count_added() as u32;
    let base = if n_added > 0 {
        leaf.allocate_contiguous_bump(n_added).expect("alloc slots")
    } else {
        0
    };
    let mut next = base;
    let applied = apply_delta(&mut octree, &mut bricks, &delta, || {
        let s = next;
        next += 1;
        s
    });
    for (slot, attrs) in &applied.allocated_slots {
        *leaf.get_mut(*slot) = attrs.to_leaf_attr();
        leaf.set_dist(*slot, attrs.dist);
    }
    for (slot, normal) in &applied.renormalized_slots {
        leaf.get_mut(*slot).normal_oct = pack_oct(*normal);
    }
    for (slot, normal, dist) in &applied.redist_slots {
        leaf.get_mut(*slot).normal_oct = pack_oct(*normal);
        leaf.set_dist(*slot, *dist);
    }

    // ── 4. Re-extract the PATCH over the brush cell range (mirrors the editor:
    // cell map padded +3, extract over (rmin, rmax) computed above).
    let cells = collect_cell_map_in_region(
        octree.as_slice(),
        depth,
        bricks.as_slice(),
        rmin - IVec3::splat(3),
        rmax + IVec3::splat(3),
    );
    let mut scratch = SculptExtractScratch::new();
    let (patch_verts, patch_idx) = extract_mesh_region_from_cells_pooled_haloed(
        &mut scratch,
        &cells,
        rmin,
        rmax,
        octree.as_slice(),
        depth,
        VS,
        grid_origin,
        bricks.as_slice(),
        leaf.as_slice(),
        leaf.bones_as_slice(),
        &[],
        None,
        None::<&fn(Vec3) -> f32>,
        leaf.dists_as_slice(),
    );

    // ── 5. Two drop filters on the old tris:
    //   (a) SphereTouch(op.radius)  — what the editor does TODAY (the bug):
    //       keep a tri iff all 3 verts are outside the brush sphere.
    //   (b) BoxTouch(patch emit box) — the candidate fix: drop everything the
    //       patch re-extract re-emits (incl. the +1 extract pad), so no
    //       kept-old can overlap the patch. (Mirrors RemeshFilter::BoxTouch.)
    let center_world = grid_origin + cg * VS;
    let r2 = (radius * VS) * (radius * VS);
    let box_min = grid_origin + (rmin.as_vec3() - Vec3::ONE) * VS;
    let box_max = grid_origin + (rmax.as_vec3() + Vec3::ONE) * VS;
    let in_box = |p: Vec3| p.cmpge(box_min).all() && p.cmple(box_max).all();
    let collect_kept = |keep: &dyn Fn(Vec3, Vec3, Vec3) -> bool| -> Vec<(Vec3, Vec3)> {
        old_idx
            .chunks_exact(3)
            .filter_map(|t| {
                let p0 = Vec3::from(old_verts[t[0] as usize].local_pos);
                let p1 = Vec3::from(old_verts[t[1] as usize].local_pos);
                let p2 = Vec3::from(old_verts[t[2] as usize].local_pos);
                if !keep(p0, p1, p2) {
                    return None;
                }
                let n = (p1 - p0).cross(p2 - p0).normalize_or_zero();
                (n != Vec3::ZERO).then_some(((p0 + p1 + p2) / 3.0, n))
            })
            .collect()
    };
    let kept_sphere = collect_kept(&|p0, p1, p2| {
        (p0 - center_world).length_squared() > r2
            && (p1 - center_world).length_squared() > r2
            && (p2 - center_world).length_squared() > r2
    });
    let kept_box = collect_kept(&|p0, p1, p2| !in_box(p0) && !in_box(p1) && !in_box(p2));
    let patch_tris = tris(&patch_verts, &patch_idx);

    // ── 6. Cross-source coincident-sheet detector. Two up-facing surfaces in
    // the same xz column with a SUB-VOXEL NON-ZERO gap z-fight; a bit-identical
    // overlap (gap≈0) is harmless. Returns (bit_identical, z_fight, max_gap_vox).
    let up = |n: Vec3| n.y > 0.5;
    let cell = |p: Vec3| (((p.x - grid_origin.x) / VS).floor() as i32, ((p.z - grid_origin.z) / VS).floor() as i32);
    let detect = |kept: &[(Vec3, Vec3)]| -> (usize, usize, f32) {
        let mut old_bins: HashMap<(i32, i32), Vec<(Vec3, Vec3)>> = HashMap::new();
        for (c, n) in kept {
            if up(*n) {
                old_bins.entry(cell(*c)).or_default().push((*c, *n));
            }
        }
        let xz_tol = 0.5 * VS;
        let (mut exact, mut zfight, mut maxgap) = (0usize, 0usize, 0.0f32);
        for (pc, pn) in &patch_tris {
            if !up(*pn) {
                continue;
            }
            let (cx, cz) = cell(*pc);
            let mut best: Option<(f32, Vec3, Vec3)> = None;
            for dx in -1..=1 {
                for dz in -1..=1 {
                    if let Some(olds) = old_bins.get(&(cx + dx, cz + dz)) {
                        for (oc, on) in olds {
                            let dxz = ((oc.x - pc.x).powi(2) + (oc.z - pc.z).powi(2)).sqrt();
                            if dxz < xz_tol && best.map(|(b, _, _)| dxz < b).unwrap_or(true) {
                                best = Some((dxz, *oc, *on));
                            }
                        }
                    }
                }
            }
            if let Some((_, oc, on)) = best {
                let oh = if on.y.abs() > 1e-4 {
                    oc.y - (on.x * (pc.x - oc.x) + on.z * (pc.z - oc.z)) / on.y
                } else {
                    oc.y
                };
                let gap = (pc.y - oh).abs() / VS;
                if gap < 0.02 {
                    exact += 1;
                } else if gap < 1.5 {
                    zfight += 1;
                    maxgap = maxgap.max(gap);
                }
            }
        }
        (exact, zfight, maxgap)
    };

    let (es, zs, gs) = detect(&kept_sphere);
    let (eb, zb, gb) = detect(&kept_box);
    eprintln!("[double-sheet] SphereTouch(op.radius, EDITOR TODAY):   bit-identical={es:>3}  z-FIGHT={zs:>3}  (max gap {gs:.3} vox)");
    eprintln!("[double-sheet] BoxTouch(patch emit box, FIX CANDIDATE): bit-identical={eb:>3}  z-FIGHT={zb:>3}  (max gap {gb:.3} vox)");

    // Hole check for the BoxTouch candidate: every xz column the FULL pre-edit
    // surface covered must still be covered by (kept_box ∪ patch) — else
    // dropping the whole box punched a hole the patch didn't refill.
    let surf_cols = |tris_cn: &[(Vec3, Vec3)]| -> HashSet<(i32, i32)> {
        tris_cn.iter().filter(|(_, n)| up(*n)).map(|(c, _)| cell(*c)).collect()
    };
    let full_old = tris(&old_verts, &old_idx);
    let full_old_cols = surf_cols(&full_old);
    let mut covered = surf_cols(&kept_box);
    covered.extend(surf_cols(&patch_tris));
    let box_holes = full_old_cols.iter().filter(|c| !covered.contains(c)).count();
    eprintln!("[double-sheet] BoxTouch hole-check: {box_holes} surface columns lost (of {} pre-edit)", full_old_cols.len());

    assert_eq!(
        zs, 0,
        "DOUBLE SHEET (z-fight): editor's SphereTouch leaves {zs} sub-voxel-offset coincident patch/old \
         ground tris (max {gs:.3} vox). BoxTouch(patch region) candidate leaves {zb} (max {gb:.3} vox). \
         Unify the drop reach with the patch re-extract reach in rebuild_dirty_clusters."
    );
}
