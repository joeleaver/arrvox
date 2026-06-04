//! Falsification / guardrail tests for occupancy-only surface recovery.
//!
//! These pin the *honest limit* of what a surface net can recover from
//! per-cell inside/outside bits plus a known grid spacing `h`. They exist so
//! that nobody later "fixes" rounded corners by over-tuning the box-constrained
//! relaxation placer (Gibson) — the information simply is not in the occupancy.
//!
//! Both tests assert that two geometrically *different* surfaces produce the
//! *identical* occupancy bit set on a known grid:
//!
//!   1. A flat plane vs. a gently-curved dome whose deviation from the plane
//!      stays under `h/2` over a small footprint. This proves the ±h/2
//!      normal-position floor: occupancy cannot tell a flat surface from a
//!      sub-`h/2`-curved one, so a placer that recovered the curvature would be
//!      inventing data.
//!
//!   2. A 90° sharp corner vs. an `r = h/2` fillet rounding that same corner.
//!      The two surfaces differ only inside a sub-cell neighbourhood of the
//!      corner, so the cell-center occupancy is identical. This proves sub-`2h`
//!      sharp features are unrecoverable from occupancy alone.
//!
//! The occupancy sampler used here is the simplest possible thing — a cell is
//! solid iff its *center* is inside the surface — matching the classifier's
//! point-membership semantics without dragging in the full voxelizer.

use glam::Vec3;

/// Grid spacing `h` (a.k.a. `voxel_size`). All feature sizes in these tests are
/// expressed as multiples of `H` so the relationship to the ±h/2 floor is
/// explicit.
const H: f32 = 0.25;

/// Number of cells per axis in the synthetic patches. Kept small and odd-ish so
/// a feature can sit cleanly on an interior grid corner.
const N: usize = 16;

/// Sample occupancy on an `N×N×N` grid of cell *centers*. A cell is solid iff
/// its center is inside the surface, as decided by `inside`. Returns a flat
/// `N*N*N` bit vector in (x, y, z) row-major order.
///
/// Cell `(i, j, k)` has center at `((i+0.5)·h, (j+0.5)·h, (k+0.5)·h)` so that
/// no center ever lands exactly on an integer grid plane — this avoids
/// degenerate "on the surface" ties that would make the comparison fragile.
fn sample_occupancy<F: Fn(Vec3) -> bool>(inside: F) -> Vec<bool> {
    let mut bits = vec![false; N * N * N];
    for i in 0..N {
        for j in 0..N {
            for k in 0..N {
                let center = Vec3::new(
                    (i as f32 + 0.5) * H,
                    (j as f32 + 0.5) * H,
                    (k as f32 + 0.5) * H,
                );
                bits[(i * N + j) * N + k] = inside(center);
            }
        }
    }
    bits
}

/// Count how many cells flipped between two occupancy sets — used only for
/// diagnostics on failure.
fn diff_count(a: &[bool], b: &[bool]) -> usize {
    a.iter().zip(b.iter()).filter(|(x, y)| x != y).count()
}

/// A flat plane vs. a parabolic dome of curvature radius `R = 4h` over a small
/// footprint produce identical occupancy.
///
/// The surface is a height field over the (x, z) plane: a cell is solid iff it
/// lies *below* the height `h_surf(x, z)` (i.e. `center.y <= h_surf`).
///
/// Flat plane:  `h_surf = PLANE_Y` (constant, = 6h).
/// Dome:        `h_surf = PLANE_Y - (ρ² / (2R))`, a paraboloid with apex at
///              `PLANE_Y` over the footprint center and curvature radius `R`.
///
/// Over the footprint (radius `RHO_MAX` from center) the dome dips at most
/// `RHO_MAX² / (2R)` below the plane. We choose the footprint so that this
/// maximum deviation is strictly less than `h/2`. Because cell centers are
/// spaced `h` apart vertically and the surfaces never differ by as much as one
/// half-cell, *every* cell center falls on the same side of both surfaces →
/// identical occupancy. That is precisely the ±h/2 normal-position floor.
#[test]
fn flat_plane_and_dome_have_identical_occupancy() {
    const PLANE_Y: f32 = 6.0 * H;
    const R: f32 = 4.0 * H; // curvature radius of the dome
    let center_xz = Vec3::new(N as f32 * H * 0.5, 0.0, N as f32 * H * 0.5);

    // Pick a footprint radius so the dome's max dip below the plane is < h/2.
    // max_dip = RHO_MAX² / (2R) < H/2  =>  RHO_MAX < sqrt(R*H).
    // sqrt(R*H) = sqrt(4H·H) = 2H. Use 1.9H to stay safely under the floor.
    const RHO_MAX: f32 = 1.9 * H;
    let max_dip = (RHO_MAX * RHO_MAX) / (2.0 * R);
    assert!(
        max_dip < H * 0.5,
        "test setup invalid: dome dips {max_dip} >= h/2={}",
        H * 0.5
    );

    // Restrict comparison to the footprint column so the dome is well-defined
    // and the claim is local (outside the footprint both surfaces are the flat
    // plane anyway). We sample the full grid for both and assert equality.
    let inside_flat = |c: Vec3| c.y <= PLANE_Y;
    let inside_dome = |c: Vec3| {
        let dx = c.x - center_xz.x;
        let dz = c.z - center_xz.z;
        let rho2 = dx * dx + dz * dz;
        // Only curve within the footprint; outside it the dome equals the plane
        // (so the surfaces are guaranteed identical there too).
        let h_surf = if rho2 <= RHO_MAX * RHO_MAX {
            PLANE_Y - rho2 / (2.0 * R)
        } else {
            PLANE_Y
        };
        c.y <= h_surf
    };

    let occ_flat = sample_occupancy(inside_flat);
    let occ_dome = sample_occupancy(inside_dome);

    assert_eq!(
        occ_flat,
        occ_dome,
        "flat plane and sub-h/2-curvature dome must have identical occupancy \
         ({} cells differ) — a curvature recovered from occupancy alone would \
         be invented data",
        diff_count(&occ_flat, &occ_dome)
    );

    // Sanity: the occupancy is non-trivial (both empty and solid cells exist),
    // otherwise the equality above would be vacuous.
    assert!(
        occ_flat.iter().any(|&b| b) && occ_flat.iter().any(|&b| !b),
        "occupancy must contain both solid and empty cells to be meaningful"
    );
}

/// A 90° sharp corner vs. an `r = h/2` fillet at the same grid corner produce
/// identical occupancy.
///
/// Geometry lives in the (x, y) plane and is extruded along z (so it is a 2D
/// claim sampled on a 3D grid). The solid region is the quadrant
/// `x <= CORNER_X && y <= CORNER_Y` — a sharp 90° interior corner at
/// `(CORNER_X, CORNER_Y)`.
///
/// The fillet rounds that convex corner with a quarter-circle of radius
/// `r = h/2`: inside the `r×r` box at the corner, solidity is decided by the
/// arc `(x - (CORNER_X - r))² + (y - (CORNER_Y - r))² <= r²` instead of the two
/// straight edges.
///
/// `CORNER_X` and `CORNER_Y` sit on integer grid planes, so the nearest cell
/// centers are `h/2` away on each axis — i.e. exactly at the box's edge. With
/// `r = h/2` the fillet only modifies solidity strictly *inside* the `r×r`
/// corner box, and no cell center lies strictly inside that box, so every cell
/// classifies identically under both surfaces. That proves a sub-`2h` sharp
/// feature is invisible to occupancy.
#[test]
fn sharp_corner_and_fillet_have_identical_occupancy() {
    // Put the corner on an interior integer grid plane: x = 8h, y = 8h.
    const CORNER_X: f32 = 8.0 * H;
    const CORNER_Y: f32 = 8.0 * H;
    const R: f32 = 0.5 * H; // fillet radius = h/2

    let inside_sharp = |c: Vec3| c.x <= CORNER_X && c.y <= CORNER_Y;

    let inside_fillet = |c: Vec3| {
        // Outside the corner box → identical to the sharp corner's two edges.
        let in_corner_box = c.x > CORNER_X - R && c.y > CORNER_Y - R;
        if in_corner_box {
            // Inside the r×r box, round the convex corner with a quarter circle
            // centered at (CORNER_X - r, CORNER_Y - r).
            let cx = CORNER_X - R;
            let cy = CORNER_Y - R;
            let dx = c.x - cx;
            let dy = c.y - cy;
            dx * dx + dy * dy <= R * R
        } else {
            c.x <= CORNER_X && c.y <= CORNER_Y
        }
    };

    let occ_sharp = sample_occupancy(inside_sharp);
    let occ_fillet = sample_occupancy(inside_fillet);

    assert_eq!(
        occ_sharp,
        occ_fillet,
        "90° corner and r=h/2 fillet must have identical occupancy \
         ({} cells differ) — a sub-2h sharp feature is unrecoverable from \
         occupancy alone",
        diff_count(&occ_sharp, &occ_fillet)
    );

    // Sanity: the corner actually produces a mixed solid/empty field.
    assert!(
        occ_sharp.iter().any(|&b| b) && occ_sharp.iter().any(|&b| !b),
        "occupancy must contain both solid and empty cells to be meaningful"
    );
}
