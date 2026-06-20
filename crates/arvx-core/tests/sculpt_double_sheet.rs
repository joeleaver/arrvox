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

/// Up-facing triangles' full world-space vertices (ground tris face +Y).
fn up_verts(verts: &[MeshVertex], idx: &[u32]) -> Vec<[Vec3; 3]> {
    idx.chunks_exact(3)
        .filter_map(|t| {
            let p = [
                Vec3::from(verts[t[0] as usize].local_pos),
                Vec3::from(verts[t[1] as usize].local_pos),
                Vec3::from(verts[t[2] as usize].local_pos),
            ];
            let n = (p[1] - p[0]).cross(p[2] - p[0]).normalize_or_zero();
            (n.y > 0.5).then_some(p)
        })
        .collect()
}

/// Up-facing triangles surviving `keep`, from the OLD mesh (mirrors a drop filter).
fn kept_up_verts(verts: &[MeshVertex], idx: &[u32], keep: &dyn Fn(Vec3, Vec3, Vec3) -> bool) -> Vec<[Vec3; 3]> {
    idx.chunks_exact(3)
        .filter_map(|t| {
            let p = [
                Vec3::from(verts[t[0] as usize].local_pos),
                Vec3::from(verts[t[1] as usize].local_pos),
                Vec3::from(verts[t[2] as usize].local_pos),
            ];
            if !keep(p[0], p[1], p[2]) {
                return None;
            }
            let n = (p[1] - p[0]).cross(p[2] - p[0]).normalize_or_zero();
            (n.y > 0.5).then_some(p)
        })
        .collect()
}

/// Surface height at `(x,z)` from a triangle via XZ barycentric interpolation,
/// or None if `(x,z)` is outside the triangle's XZ projection.
fn bary_height(tri: &[Vec3; 3], x: f32, z: f32) -> Option<f32> {
    let (a, b, c) = (tri[0], tri[1], tri[2]);
    let (v0x, v0z) = (b.x - a.x, b.z - a.z);
    let (v1x, v1z) = (c.x - a.x, c.z - a.z);
    let (v2x, v2z) = (x - a.x, z - a.z);
    let d00 = v0x * v0x + v0z * v0z;
    let d01 = v0x * v1x + v0z * v1z;
    let d11 = v1x * v1x + v1z * v1z;
    let d20 = v2x * v0x + v2z * v0z;
    let d21 = v2x * v1x + v2z * v1z;
    let denom = d00 * d11 - d01 * d01;
    if denom.abs() < 1e-12 {
        return None;
    }
    let v = (d11 * d20 - d01 * d21) / denom;
    let w = (d00 * d21 - d01 * d20) / denom;
    let u = 1.0 - v - w;
    let e = -1e-4;
    (u >= e && v >= e && w >= e).then_some(u * a.y + v * b.y + w * c.y)
}

/// THE TRUSTWORTHY VALIDATOR. Cast a vertical ray through every grid-cell-center
/// column over the union mesh's xz bbox and count the up-facing surfaces it hits.
/// Heights within 0.02 vox merge into one layer (a welded seam, or two
/// bit-identical coincident surfaces, reads as ONE surface). A column with ≥2
/// layers separated by a z-fighting gap (0.02..1.5 vox) is a TRUE double sheet.
/// Returns (double_sheet_columns, max_gap_vox). Unlike column-coincidence, this
/// cannot be fooled by a welded region/full seam.
fn zfight_columns(union_up: &[[Vec3; 3]], grid_origin: Vec3, vs: f32) -> (usize, f32) {
    if union_up.is_empty() {
        return (0, 0.0);
    }
    let (mut lo_x, mut lo_z, mut hi_x, mut hi_z) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
    for t in union_up {
        for p in t {
            lo_x = lo_x.min(p.x);
            lo_z = lo_z.min(p.z);
            hi_x = hi_x.max(p.x);
            hi_z = hi_z.max(p.z);
        }
    }
    let cxa = ((lo_x - grid_origin.x) / vs).floor() as i32;
    let cxb = ((hi_x - grid_origin.x) / vs).ceil() as i32;
    let cza = ((lo_z - grid_origin.z) / vs).floor() as i32;
    let czb = ((hi_z - grid_origin.z) / vs).ceil() as i32;
    let (mut count, mut maxgap) = (0usize, 0.0f32);
    let merge = 0.02 * vs;
    for cz in cza..=czb {
        for cx in cxa..=cxb {
            let px = grid_origin.x + (cx as f32 + 0.5) * vs;
            let pz = grid_origin.z + (cz as f32 + 0.5) * vs;
            let mut hs: Vec<f32> = union_up.iter().filter_map(|t| bary_height(t, px, pz)).collect();
            if hs.len() < 2 {
                continue;
            }
            hs.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let mut layers = vec![hs[0]];
            for &h in &hs[1..] {
                if (h - layers.last().unwrap()).abs() > merge {
                    layers.push(h);
                }
            }
            let mut zf = false;
            for w in layers.windows(2) {
                let g = (w[1] - w[0]).abs();
                if g > merge && g < 1.5 * vs {
                    zf = true;
                    maxgap = maxgap.max(g / vs);
                }
            }
            if zf {
                count += 1;
            }
        }
    }
    (count, maxgap)
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

    // ── ISOLATION: extract the SAME (pre-brush) octree via the REGION path
    // over (rmin,rmax) and compare to the FULL pre-edit extract in the overlap.
    // No brush touches this data, so any z-fight here is PURE mesher
    // inconsistency (region extract != full extract on identical input). If
    // this is 0, the double-sheet is the brush modifying annulus distances, not
    // a mesher bug — a completely different fix.
    let cells_pre = collect_cell_map_in_region(
        octree.as_slice(),
        depth,
        bricks.as_slice(),
        rmin - IVec3::splat(3),
        rmax + IVec3::splat(3),
    );
    let mut scratch_pre = SculptExtractScratch::new();
    let (region_pre_v, region_pre_i) = extract_mesh_region_from_cells_pooled_haloed(
        &mut scratch_pre,
        &cells_pre,
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

    // ── 4b. FIX CANDIDATE: re-extract over an EXPANDED region whose boundary
    // sits in UNCHANGED territory (margin EX beyond the brush influence), so the
    // patch's outer seam welds bit-identically to the kept-old (isolation proved
    // region==full there). Drop = Box over the expanded emit extent. Expect: the
    // boundary z-fight + holes both vanish.
    const EX: i32 = 3;
    let rmin_ex = (rmin - IVec3::splat(EX)).max(IVec3::ZERO);
    let rmax_ex = (rmax + IVec3::splat(EX)).min(IVec3::splat(bextent as i32));
    let cells_ex = collect_cell_map_in_region(
        octree.as_slice(),
        depth,
        bricks.as_slice(),
        rmin_ex - IVec3::splat(3),
        rmax_ex + IVec3::splat(3),
    );
    let mut scratch_ex = SculptExtractScratch::new();
    let (patch_ex_v, patch_ex_i) = extract_mesh_region_from_cells_pooled_haloed(
        &mut scratch_ex,
        &cells_ex,
        rmin_ex,
        rmax_ex,
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
    let detect = |kept: &[(Vec3, Vec3)], patch: &[(Vec3, Vec3)]| -> (usize, usize, f32) {
        let mut old_bins: HashMap<(i32, i32), Vec<(Vec3, Vec3)>> = HashMap::new();
        for (c, n) in kept {
            if up(*n) {
                old_bins.entry(cell(*c)).or_default().push((*c, *n));
            }
        }
        let xz_tol = 0.5 * VS;
        let (mut exact, mut zfight, mut maxgap) = (0usize, 0usize, 0.0f32);
        for (pc, pn) in patch {
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

    // Surface-column coverage helper for hole checks.
    let surf_cols = |tris_cn: &[(Vec3, Vec3)]| -> HashSet<(i32, i32)> {
        tris_cn.iter().filter(|(_, n)| up(*n)).map(|(c, _)| cell(*c)).collect()
    };
    let full_old_cols = surf_cols(&tris(&old_verts, &old_idx));
    let holes_of = |kept: &[(Vec3, Vec3)], patch: &[(Vec3, Vec3)]| -> usize {
        let mut cov = surf_cols(kept);
        cov.extend(surf_cols(patch));
        full_old_cols.iter().filter(|c| !cov.contains(c)).count()
    };

    // (a) SphereTouch(op.radius) — the editor TODAY (the bug).
    let (es, zs, gs) = detect(&kept_sphere, &patch_tris);
    // (b) BoxTouch(tight patch region) — drop the whole emit box.
    let (eb, zb, gb) = detect(&kept_box, &patch_tris);
    let box_holes = holes_of(&kept_box, &patch_tris);
    // ISOLATION — full vs region on the IDENTICAL pre-brush octree (pure mesher).
    let region_pre = tris(&region_pre_v, &region_pre_i);
    let full_pre_in_box: Vec<(Vec3, Vec3)> = tris(&old_verts, &old_idx)
        .into_iter()
        .filter(|(c, _)| {
            c.cmpge(grid_origin + rmin.as_vec3() * VS).all() && c.cmple(grid_origin + rmax.as_vec3() * VS).all()
        })
        .collect();
    let (ei, zi, gi) = detect(&full_pre_in_box, &region_pre);
    // (c) FIX: expanded region (boundary in UNCHANGED territory) + Box drop.
    let ex_box_min = grid_origin + (rmin_ex.as_vec3() - Vec3::ONE) * VS;
    let ex_box_max = grid_origin + (rmax_ex.as_vec3() + Vec3::ONE) * VS;
    let in_ex_box = |p: Vec3| p.cmpge(ex_box_min).all() && p.cmple(ex_box_max).all();
    let kept_ex = collect_kept(&|p0, p1, p2| !in_ex_box(p0) && !in_ex_box(p1) && !in_ex_box(p2));
    let patch_ex_tris = tris(&patch_ex_v, &patch_ex_i);
    let (ex_e, ex_z, ex_g) = detect(&kept_ex, &patch_ex_tris);
    let ex_holes = holes_of(&kept_ex, &patch_ex_tris);

    eprintln!("[double-sheet] SphereTouch(op.radius, EDITOR TODAY):   bit-identical={es:>3}  z-FIGHT={zs:>3}  (max gap {gs:.3} vox)");
    eprintln!("[double-sheet] BoxTouch(tight patch region):           bit-identical={eb:>3}  z-FIGHT={zb:>3}  (max gap {gb:.3} vox)  holes={box_holes}");
    eprintln!("[double-sheet] ISOLATION full-vs-region (pre-brush):   bit-identical={ei:>3}  z-FIGHT={zi:>3}  (max gap {gi:.3} vox)  <- centroid detector UNRELIABLE (see VERTEX-SET)");
    eprintln!("[double-sheet] FIX expanded(+{EX}) region + Box drop:    bit-identical={ex_e:>3}  z-FIGHT={ex_z:>3}  (max gap {ex_g:.3} vox)  holes={ex_holes}");

    // ── TRUSTWORTHY VALIDATOR: cast a vertical ray per column over each
    // RENDERED union mesh (kept-old ∪ patch) and count true double sheets.
    // Immune to the welded-seam confound that fooled the column-coincidence
    // detector above.
    let patch_up = up_verts(&patch_verts, &patch_idx);
    let patch_ex_up = up_verts(&patch_ex_v, &patch_ex_i);
    let union_of = |kept: Vec<[Vec3; 3]>, patch: &[[Vec3; 3]]| {
        let mut u = kept;
        u.extend_from_slice(patch);
        u
    };
    let sphere_keep =
        |p0: Vec3, p1: Vec3, p2: Vec3| (p0 - center_world).length_squared() > r2 && (p1 - center_world).length_squared() > r2 && (p2 - center_world).length_squared() > r2;
    let box_keep = |p0: Vec3, p1: Vec3, p2: Vec3| !in_box(p0) && !in_box(p1) && !in_box(p2);
    let ex_keep = |p0: Vec3, p1: Vec3, p2: Vec3| !in_ex_box(p0) && !in_ex_box(p1) && !in_ex_box(p2);
    let (rz_sphere, rg_sphere) = zfight_columns(&union_of(kept_up_verts(&old_verts, &old_idx, &sphere_keep), &patch_up), grid_origin, VS);
    let (rz_box, rg_box) = zfight_columns(&union_of(kept_up_verts(&old_verts, &old_idx, &box_keep), &patch_up), grid_origin, VS);
    let (rz_ex, rg_ex) = zfight_columns(&union_of(kept_up_verts(&old_verts, &old_idx, &ex_keep), &patch_ex_up), grid_origin, VS);
    let (rz_iso, rg_iso) = zfight_columns(&union_of(up_verts(&old_verts, &old_idx), &up_verts(&region_pre_v, &region_pre_i)), grid_origin, VS);
    eprintln!("[RAY-VALIDATOR] true rendered double-sheet columns (gap in vox):");
    eprintln!("    SphereTouch(today)={rz_sphere} (g{rg_sphere:.3})   BoxTouch(tight)={rz_box} (g{rg_box:.3})   expanded+Box={rz_ex} (g{rg_ex:.3})   isolation={rz_iso} (g{rg_iso:.3})");

    // ── DECISIVE mesher check: are the region-pre and full-pre VERTEX SETS
    // identical? For each region-pre vertex, 3D distance to the nearest full-pre
    // vertex. seam_far > 0 ⇒ the region extract genuinely produces DIFFERENT
    // boundary vertices than the full extract (a REAL mesher seam, NOT a
    // triangulation/interpolation artifact) — exactly what the column-coincidence
    // detector hid by matching centroids loosely.
    let fv: Vec<Vec3> = old_verts.iter().map(|v| Vec3::from(v.local_pos)).collect();
    let mut seam_max = 0.0f32;
    let mut seam_far = 0usize;
    let pos_dbg = std::env::var("ARVX_SEAM_POS").is_ok();
    let (mut n_pad, mut n_ring, mut n_interior) = (0usize, 0usize, 0usize);
    for rv in region_pre_v.iter().map(|v| Vec3::from(v.local_pos)) {
        let mut best = f32::MAX;
        for &f in &fv {
            best = best.min((f - rv).length_squared());
        }
        let d = best.sqrt();
        seam_max = seam_max.max(d);
        if d > 0.01 * VS {
            seam_far += 1;
            if pos_dbg {
                // grid coords + signed cells-from-nearest-box-face (>0 = outside box = pad)
                let g = (rv - grid_origin) / VS;
                let gi = [g.x, g.y, g.z];
                let mn = [rmin.x as f32, rmin.y as f32, rmin.z as f32];
                let mx = [rmax.x as f32, rmax.y as f32, rmax.z as f32];
                // max over axes of how far OUTSIDE the box (negative = inside)
                let mut outside = f32::MIN;
                let mut inside_margin = f32::MAX;
                for a in 0..3 {
                    outside = outside.max((mn[a] - gi[a]).max(gi[a] - mx[a]));
                    inside_margin = inside_margin.min((gi[a] - mn[a]).min(mx[a] - gi[a]));
                }
                if outside > 0.01 {
                    n_pad += 1;
                } else if inside_margin < 2.0 {
                    n_ring += 1;
                } else {
                    n_interior += 1;
                }
            }
        }
    }
    if pos_dbg {
        eprintln!("[SEAM-POS] of {seam_far} divergent verts: pad(outside box)={n_pad}  boundary-ring(<2 cells in)={n_ring}  interior={n_interior}");
    }
    eprintln!(
        "[VERTEX-SET] region-pre vs full-pre: {} verts, max nearest-full {:.4} vox, {} verts >0.01 vox apart  <- REAL region!=full boundary divergence",
        region_pre_v.len(),
        seam_max / VS,
        seam_far,
    );

    // Everything above is documented via prints; these are the measured landscape.
    let _ = (es, eb, gb, zb, ei, zi, gi, ex_e, ex_z, ex_g, ex_holes, box_holes, rz_box, rg_box, rz_ex, rg_ex, rg_sphere);

    // PART B FIXED + GATED (region-grid widened to the collect halo): the region
    // extract is now BIT-IDENTICAL to the full extract — no boundary divergence,
    // no pure-mesher double sheet. These two assertions guard against a
    // regression of that fix.
    assert_eq!(seam_far, 0, "part B regressed: region != full at boundary — {seam_far} verts up to {seam_max:.4}/VS apart");
    assert_eq!(rz_iso, 0, "part B regressed: pure-mesher (no-brush) double-sheet columns = {rz_iso}");

    // PART A REMAINING (separate fix): the editor's SphereTouch path still has
    // brush-near-op.radius double sheets (kept-old = pre-brush, patch =
    // post-brush). Drop the brush's changed region to eliminate. Flip this to
    // `assert_eq!(rz_sphere, 0)` + un-ignore once part A lands.
    assert!(
        rz_sphere > 0,
        "expected the brush near-radius double-sheet to still reproduce on the SphereTouch path, got {rz_sphere}"
    );
}
