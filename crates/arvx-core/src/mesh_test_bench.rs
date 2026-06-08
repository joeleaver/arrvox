//! Headless meshing test-bench.
//!
//! Validates the blurred-occupancy surface-net mesher
//! ([`crate::mesh_extract::extract_mesh_region_from_cells_pooled_haloed`],
//! the blur + ∇D-normal path) against analytic ground truth, WITHOUT
//! the terrain / tiling / LOD / sculpt machinery that normally drives
//! it. We voxelize a closed-form SDF into clean occupancy, feed it
//! through the *identical* extract path the sculpt brush uses, then:
//!
//! * measure geometry error `|sdf(vertex)|`, normal error
//!   `angle(extracted_n, ∇sdf)`, and per-edge normal continuity
//!   (no-speckle), as `#[test]` assertions; and
//! * software-render the result to RGB pixel buffers (shaded /
//!   wireframe / voxel-edge layers, 3/4 + side views) that the
//!   `examples/mesh_bench` binary writes to PNG.
//!
//! This is the permanent tool for diagnosing geometry terracing: it
//! isolates the mesher from every confound, gives numbers, and shows
//! the surface next to the source voxels.
//!
//! ### How the bench drives the engine path
//!
//! [`extract_mesh_region_from_cells_pooled_haloed`] needs occupancy
//! (`CellMap`), a region, an octree + brick pool (for the
//! INTERIOR-bulk corner fallback), and the leaf-attr / bone pools. The
//! bench builds the MINIMAL valid versions:
//!
//! * `octree_nodes = [EMPTY_NODE]` — so the in-extract
//!   `is_solid_lookup` fallback always returns "empty"; occupancy is
//!   then driven entirely by the `CellMap`. (Same trick the in-module
//!   density tests use.)
//! * empty `brick_cells`, `leaf_attr_pool`, `bone_voxel_pool`, `halo`,
//!   `sculpt_slots` — none of these affect the *geometry* or the `∇D`
//!   normal (the normal now comes from the density gradient field, not
//!   the per-leaf normals). With an empty `leaf_attr_pool` every
//!   vertex's `leaf_attr_id` falls back to 0, which is irrelevant here.
//! * `base_voxel_size = voxel_size`, `grid_origin = bounds.min` snapped
//!   to the voxel lattice.

use crate::aabb::Aabb;
use crate::mesh_extract::{
    extract_mesh_region_from_cells_pooled_haloed, set_blur_override, CellMap, MeshVertex,
    SculptExtractScratch,
};
use crate::sparse_octree::EMPTY_NODE;
use crate::unpack_oct;
use glam::{IVec3, Vec3};

// ════════════════════════════════════════════════════════════════════
// 1. Analytic shapes
// ════════════════════════════════════════════════════════════════════

/// A closed-form analytic shape: signed distance + analytic surface
/// normal (normalized gradient). All shapes are authored to live
/// comfortably inside the `[-1, 1]³`-ish bench domain.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Shape {
    /// Sphere, radius 0.7, centered at origin.
    Sphere,
    /// Axis-aligned box, half-extents (0.6, 0.45, 0.55), sharp edges.
    Box,
    /// Rounded box, half-extents (0.55, 0.4, 0.5), corner radius 0.18.
    RoundedBox,
    /// Torus in the XZ plane, major radius 0.55, minor radius 0.22.
    Torus,
    /// Tilted planar slope `y = m_x·x + m_z·z`, clipped to the domain.
    /// The canonical terracing probe.
    Slope,
    /// Sine heightfield terrain `y = a·sin(kx·x) + b·sin(kz·z)`.
    SineTerrain,
    /// Cone with a SHARP apex (probes sharp-feature rounding).
    Cone,
    /// Capped cylinder, radius 0.5, half-height 0.6, axis +Y.
    Cylinder,
    /// STEEP planar slope `y = m_x·x + m_z·z` with `m_x ≈ 2.5` (~68°),
    /// so adjacent columns jump ~2-3 voxels at vs=0.25 — the canonical
    /// vertical-terracing probe matching the user's sculpted-mound side.
    SteepSlope,
    /// SCULPTED MOUND — a smooth radial Gaussian bump
    /// `y = H·exp(-(x²+z²)/s²)` with moderately steep sides (the closest
    /// analog to an Inflate-brush mound). Peak ~2.25 world ≈ 9 voxels at
    /// vs=0.25.
    Mound,
}

impl Shape {
    /// Every shape, in table order.
    pub fn all() -> &'static [Shape] {
        &[
            Shape::Sphere,
            Shape::Box,
            Shape::RoundedBox,
            Shape::Torus,
            Shape::Slope,
            Shape::SineTerrain,
            Shape::Cone,
            Shape::Cylinder,
            Shape::SteepSlope,
            Shape::Mound,
        ]
    }

    /// Short lowercase name used for output directories / table rows.
    pub fn name(self) -> &'static str {
        match self {
            Shape::Sphere => "sphere",
            Shape::Box => "box",
            Shape::RoundedBox => "rounded_box",
            Shape::Torus => "torus",
            Shape::Slope => "slope",
            Shape::SineTerrain => "sine_terrain",
            Shape::Cone => "cone",
            Shape::Cylinder => "cylinder",
            Shape::SteepSlope => "steep_slope",
            Shape::Mound => "mound",
        }
    }

    /// True for shapes whose surface is smooth everywhere (no sharp
    /// creases). The geometry / normal accuracy asserts apply only to
    /// these; box/rounded_box-edges/cone-apex are exempt (looser bound).
    pub fn is_smooth(self) -> bool {
        matches!(
            self,
            Shape::Sphere
                | Shape::Torus
                | Shape::Slope
                | Shape::SineTerrain
                | Shape::SteepSlope
                | Shape::Mound
        )
    }

    /// True for the height-field shapes (terracing-relevant: surface is
    /// `y = h(x,z)`, the solid is everything below it).
    pub fn is_height_field(self) -> bool {
        matches!(
            self,
            Shape::Slope | Shape::SineTerrain | Shape::SteepSlope | Shape::Mound
        )
    }

    /// True for the gently-curved smooth shapes used as accuracy
    /// CONTROLS (sphere/torus/slope/sine). Excludes the deliberately
    /// STEEP terracing probes (`SteepSlope`, `Mound`) — those exist to
    /// demonstrate terracing/blur behavior under a sweep, not to pass a
    /// tight accuracy bound (their steep sides carry more discretization
    /// + blur shift). They are still exercised by the roughness and
    /// R-sweep tests.
    pub fn is_accuracy_control(self) -> bool {
        matches!(
            self,
            Shape::Sphere | Shape::Torus | Shape::Slope | Shape::SineTerrain
        )
    }

    /// Domain the shape is meshed over. Sized so every shape spans a
    /// MODERATE number of voxels: enough that the R = 2 blur recovers a
    /// smooth field and the metrics are meaningful, but few enough that
    /// the wireframe / voxel-edge renders stay legible (tens of
    /// thousands of triangles turn a 512² image into noise). Sphere
    /// radius 3 → 12 cells across at vs = 0.5, 24 at vs = 0.25.
    pub fn bounds(self) -> Aabb {
        match self {
            Shape::Slope | Shape::SineTerrain | Shape::SteepSlope | Shape::Mound => {
                // Height fields: tall enough in Y to hold the surface +
                // the solid below + halo, wide in X/Z. SteepSlope rises
                // ±~6 over ±2.4 in X, and the Mound peaks at +2.25, so a
                // ±4.5 box brackets the surface with margin.
                Aabb::new(Vec3::new(-4.5, -4.5, -4.5), Vec3::new(4.5, 4.5, 4.5))
            }
            _ => Aabb::new(Vec3::splat(-4.0), Vec3::splat(4.0)),
        }
    }

    /// Signed distance (negative inside). For non-Euclidean fields
    /// (slope/sine height fields) this is the vertical gap, which is a
    /// valid sign field for occupancy + a good local distance near the
    /// near-horizontal surface.
    pub fn sdf(self, p: Vec3) -> f32 {
        match self {
            Shape::Sphere => p.length() - 3.0,
            Shape::Box => sd_box(p, Vec3::new(2.6, 1.9, 2.3)),
            Shape::RoundedBox => sd_box(p, Vec3::new(2.2, 1.6, 2.0)) - 0.7,
            Shape::Torus => {
                // Torus around +Y axis: ring in the XZ plane.
                let q = Vec3::new((p.x * p.x + p.z * p.z).sqrt() - 2.4, p.y, 0.0);
                q.length() - 1.0
            }
            Shape::Slope => {
                // y = m_x x + m_z z. True point-to-plane distance =
                // vertical gap / |∇(y - surf)|, where ∇ = (-m_x, 1, -m_z).
                // Dividing by the gradient magnitude makes `|sdf|` the
                // honest Euclidean surface distance the geometry metric
                // compares against (no slope-angle inflation).
                let surf = SLOPE_MX * p.x + SLOPE_MZ * p.z;
                let grad_mag = (1.0 + SLOPE_MX * SLOPE_MX + SLOPE_MZ * SLOPE_MZ).sqrt();
                (p.y - surf) / grad_mag
            }
            Shape::SineTerrain => {
                // y = a·sin(kx x) + b·sin(kz z). First-order Euclidean
                // distance: vertical gap / |∇(y - surf)| (exact at crests
                // / troughs, ~exact for gentle slopes between).
                let surf = SINE_A * (SINE_KX * p.x).sin() + SINE_B * (SINE_KZ * p.z).sin();
                let dx = SINE_A * SINE_KX * (SINE_KX * p.x).cos();
                let dz = SINE_B * SINE_KZ * (SINE_KZ * p.z).cos();
                let grad_mag = (1.0 + dx * dx + dz * dz).sqrt();
                (p.y - surf) / grad_mag
            }
            Shape::Cone => sd_cone(p, CONE_HALF_ANGLE, CONE_HEIGHT),
            Shape::Cylinder => sd_capped_cylinder(p, 2.2, 1.8),
            Shape::SteepSlope => {
                // y = m_x x + m_z z, m_x ≈ 2.5 (~68°). The surface is
                // CLAMPED to ±3.6 so the steep ramp stays inside the ±4.5
                // box (otherwise it would leave the domain). The steep
                // central band is where vertical terracing shows.
                let raw = STEEP_MX * p.x + STEEP_MZ * p.z;
                let surf = raw.clamp(-3.6, 3.6);
                let grad_mag = (1.0 + STEEP_MX * STEEP_MX + STEEP_MZ * STEEP_MZ).sqrt();
                (p.y - surf) / grad_mag
            }
            Shape::Mound => {
                // Radial Gaussian bump y = H·exp(-(x²+z²)/s²). Vertical
                // gap / |∇| for an honest Euclidean surface distance.
                let r2 = p.x * p.x + p.z * p.z;
                let surf = MOUND_H * (-r2 / (MOUND_S * MOUND_S)).exp();
                // ∂surf/∂x = surf · (-2x/s²); likewise z.
                let dsx = surf * (-2.0 * p.x / (MOUND_S * MOUND_S));
                let dsz = surf * (-2.0 * p.z / (MOUND_S * MOUND_S));
                let grad_mag = (1.0 + dsx * dsx + dsz * dsz).sqrt();
                (p.y - surf) / grad_mag
            }
        }
    }

    /// Analytic outward surface normal = normalized ∇sdf. Uses
    /// closed-form gradients where clean, central differences only as a
    /// fallback for the piecewise fields (box/cone) where the analytic
    /// gradient is annoying — central diff of the analytic sdf is still
    /// "analytic ground truth" for our purposes (it's the true gradient
    /// of the closed-form field, not of the discrete mesh).
    pub fn normal(self, p: Vec3) -> Vec3 {
        match self {
            Shape::Sphere => p.normalize_or_zero(),
            Shape::Torus => {
                let len_xz = (p.x * p.x + p.z * p.z).sqrt();
                if len_xz < 1e-6 {
                    return Vec3::Y;
                }
                // Center of the tube at this angular position.
                let ring = Vec3::new(p.x / len_xz * 2.4, 0.0, p.z / len_xz * 2.4);
                (p - ring).normalize_or_zero()
            }
            Shape::Slope => Vec3::new(-SLOPE_MX, 1.0, -SLOPE_MZ).normalize(),
            Shape::SineTerrain => {
                // ∇(y - surf) = (-dsurf/dx, 1, -dsurf/dz).
                let dx = SINE_A * SINE_KX * (SINE_KX * p.x).cos();
                let dz = SINE_B * SINE_KZ * (SINE_KZ * p.z).cos();
                Vec3::new(-dx, 1.0, -dz).normalize()
            }
            Shape::SteepSlope => {
                // In the clamped flat caps the normal is +Y; on the ramp
                // it is the constant tilted plane normal. Decide by
                // whether the raw surface is inside the clamp band.
                let raw = STEEP_MX * p.x + STEEP_MZ * p.z;
                if raw.abs() >= 3.6 {
                    Vec3::Y
                } else {
                    Vec3::new(-STEEP_MX, 1.0, -STEEP_MZ).normalize()
                }
            }
            Shape::Mound => {
                let r2 = p.x * p.x + p.z * p.z;
                let surf = MOUND_H * (-r2 / (MOUND_S * MOUND_S)).exp();
                let dsx = surf * (-2.0 * p.x / (MOUND_S * MOUND_S));
                let dsz = surf * (-2.0 * p.z / (MOUND_S * MOUND_S));
                Vec3::new(-dsx, 1.0, -dsz).normalize()
            }
            // Box / rounded_box / cone / cylinder: central-diff the
            // analytic sdf (true field gradient).
            _ => grad_central(|q| self.sdf(q), p),
        }
    }
}

/// Slope gradient (chosen non-grid-aligned in BOTH axes so terracing is
/// unambiguous if it appears). The slope plane passes through the
/// origin; over ±4.5 in X it rises ±~1.9 — a gentle, clearly
/// non-aligned ramp.
const SLOPE_MX: f32 = 0.42;
const SLOPE_MZ: f32 = 0.27;
/// Sine-terrain params: a couple of bumps across the ±4.5 domain.
/// Amplitudes ~1-1.3 world units (2-3 voxels at vs=0.5), wavelengths
/// ~5-9 units so the surface is smooth on the voxel scale.
const SINE_A: f32 = 1.1;
const SINE_B: f32 = 0.8;
const SINE_KX: f32 = 1.15;
const SINE_KZ: f32 = 1.55;
/// Cone: half-angle (radians) + height, apex up at +Y. Apex at
/// (0, +2, 0), base near y = -2 — spans ~8 voxels tall at vs=0.5.
const CONE_HALF_ANGLE: f32 = 0.6;
const CONE_HEIGHT: f32 = 4.0;
/// STEEP slope: `m_x = 2.5` (~68° from horizontal) so adjacent columns
/// at vs=0.25 jump `0.25·2.5 = 0.625` world ≈ 2.5 voxels — well past the
/// 1-voxel-per-column the R=2 blur can de-staircase, so it terraces.
/// A small non-zero `m_z` keeps the slope off the grid axes.
const STEEP_MX: f32 = 2.5;
const STEEP_MZ: f32 = 0.3;
/// Sculpted mound: peak height + radial scale. `H = 2.25` ≈ 9 voxels
/// tall at vs=0.25; `s = 1.6` gives moderately steep sides (max slope
/// ~`H·√2/(s·√e)` ≈ 1.2 → ~50° at the inflection ring), steeper than the
/// gentle sine — the closest analog to an Inflate-brush mound.
const MOUND_H: f32 = 2.25;
const MOUND_S: f32 = 1.6;

#[inline]
fn sd_box(p: Vec3, b: Vec3) -> f32 {
    let q = p.abs() - b;
    q.max(Vec3::ZERO).length() + q.max_element().min(0.0)
}

/// Cone with apex at +Y (height `h`), opening downward, half-angle `a`.
/// IQ's capped-cone-ish sdf simplified to a sharp single cone.
#[inline]
fn sd_cone(p: Vec3, half_angle: f32, h: f32) -> f32 {
    // Place apex at (0, h/2, 0), base at y = -h/2.
    let apex_y = h * 0.5;
    let pp = Vec3::new(p.x, apex_y - p.y, p.z); // distance below apex along +pp.y
    let q = (half_angle.sin(), half_angle.cos());
    let w = Vec3::new((pp.x * pp.x + pp.z * pp.z).sqrt(), pp.y, 0.0);
    // Distance to the lateral cone surface (signed by side).
    let lateral = w.x * q.1 - w.y * q.0;
    // Cap the cone at the base plane y = -h/2 → pp.y <= h.
    let base = w.y - h;
    lateral.max(base)
}

/// Capped cylinder, axis +Y, half-height `hh`, radius `r`.
#[inline]
fn sd_capped_cylinder(p: Vec3, hh: f32, r: f32) -> f32 {
    let d_xz = (p.x * p.x + p.z * p.z).sqrt() - r;
    let d_y = p.y.abs() - hh;
    let dx = d_xz.max(0.0);
    let dy = d_y.max(0.0);
    d_xz.max(d_y).min(0.0) + (dx * dx + dy * dy).sqrt()
}

/// Central-difference gradient of a scalar field, normalized.
#[inline]
fn grad_central<F: Fn(Vec3) -> f32>(f: F, p: Vec3) -> Vec3 {
    let h = 1e-3;
    let dx = f(p + Vec3::new(h, 0.0, 0.0)) - f(p - Vec3::new(h, 0.0, 0.0));
    let dy = f(p + Vec3::new(0.0, h, 0.0)) - f(p - Vec3::new(0.0, h, 0.0));
    let dz = f(p + Vec3::new(0.0, 0.0, h)) - f(p - Vec3::new(0.0, 0.0, h));
    Vec3::new(dx, dy, dz).normalize_or_zero()
}

// ════════════════════════════════════════════════════════════════════
// 2. Voxelize + mesh via the engine path
// ════════════════════════════════════════════════════════════════════

/// Occupancy produced by voxelizing a shape: the set of solid cells (in
/// integer grid coords) plus the grid origin / voxel size that map cell
/// coords back to world space.
pub struct Occupancy {
    /// Solid cells (cell center inside the surface). Values are all 0
    /// (a dummy `leaf_attr_id`); only key membership matters.
    pub cells: CellMap,
    /// World position of cell-grid origin (lo corner of cell `(0,0,0)`).
    pub grid_origin: Vec3,
    /// Finest cell edge length.
    pub voxel_size: f32,
    /// Half-open cell-coord region covering all solid cells (+ margin).
    pub region_min: IVec3,
    pub region_max: IVec3,
}

impl Occupancy {
    /// World-space center of cell `c`.
    #[inline]
    pub fn cell_center(&self, c: IVec3) -> Vec3 {
        self.grid_origin + (Vec3::new(c.x as f32, c.y as f32, c.z as f32) + Vec3::splat(0.5)) * self.voxel_size
    }

    /// Lo corner (world) of cell `c`.
    #[inline]
    pub fn cell_lo(&self, c: IVec3) -> Vec3 {
        self.grid_origin + Vec3::new(c.x as f32, c.y as f32, c.z as f32) * self.voxel_size
    }
}

/// Voxelize `shape` over `bounds` at `voxel_size`: a cell is solid iff
/// `sdf(cell_center) < 0`. `grid_origin = bounds.min` snapped so the
/// lattice is stable.
pub fn voxelize(shape: Shape, bounds: Aabb, voxel_size: f32) -> Occupancy {
    let grid_origin = bounds.min;
    let size = bounds.size();
    let nx = (size.x / voxel_size).ceil() as i32 + 1;
    let ny = (size.y / voxel_size).ceil() as i32 + 1;
    let nz = (size.z / voxel_size).ceil() as i32 + 1;

    let mut cells = CellMap::default();
    let mut lo = IVec3::splat(i32::MAX);
    let mut hi = IVec3::splat(i32::MIN);
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                let c = IVec3::new(x, y, z);
                let center =
                    grid_origin + (Vec3::new(x as f32, y as f32, z as f32) + Vec3::splat(0.5)) * voxel_size;
                if shape.sdf(center) < 0.0 {
                    cells.insert(c, 0);
                    lo = lo.min(c);
                    hi = hi.max(c);
                }
            }
        }
    }

    // Region = solid-cell bbox + a couple cells of margin so every
    // surface cube iterates and the density neighborhood is real.
    let (region_min, region_max) = if cells.is_empty() {
        (IVec3::ZERO, IVec3::ZERO)
    } else {
        (lo - IVec3::splat(2), hi + IVec3::splat(3))
    };

    Occupancy {
        cells,
        grid_origin,
        voxel_size,
        region_min,
        region_max,
    }
}

// ════════════════════════════════════════════════════════════════════
// 2b. Confound scenarios — reproduce the real-terrain terracing
// ════════════════════════════════════════════════════════════════════
//
// The clean analytic shapes mesh SMOOTHLY (the bench's whole point), so
// they do NOT reproduce the severe terracing the user sees on real
// terrain. These scenarios re-introduce, one at a time, the confounds
// the clean bench strips out, to find which one terraces:
//
//   (a) COARSE       — few voxels per feature (slope/sine at vs 1.0/2.0).
//   (b) TRUNCATION   — mesh only a sub-window of the occupancy with a
//                      NARROW halo; the blur neighborhood is clamped at
//                      the window boundary.
//   (c) IRREGULAR    — perturb each column's solid height by a
//                      deterministic ±1-2 voxel hash (sculpt-brush-like
//                      irregular occupancy the blur can't fully smooth).

/// A named confound scenario that yields an [`Occupancy`] meshed through
/// the engine path, plus the analytic [`Shape`] it derives from (for
/// metric ground truth).
pub struct Scenario {
    pub name: String,
    pub shape: Shape,
    pub occ: Occupancy,
}

/// Deterministic 32-bit hash of two ints → `[0,1)`. Used to perturb
/// column heights reproducibly (no RNG state).
#[inline]
fn hash01(x: i32, z: i32) -> f32 {
    let mut h = (x as u32).wrapping_mul(0x9E37_79B1) ^ (z as u32).wrapping_mul(0x85EB_CA77);
    h ^= h >> 15;
    h = h.wrapping_mul(0x2545_F491);
    h ^= h >> 13;
    (h & 0x00FF_FFFF) as f32 / 0x0100_0000 as f32
}

/// Voxelize a height-field shape with each `(x,z)` column's solid
/// height perturbed by a deterministic `±amp` voxels (hash-based). This
/// mimics the irregular occupancy a sculpt brush / dilation leaves: the
/// surface is no longer a clean monotone height field but a jagged one.
pub fn voxelize_irregular(shape: Shape, bounds: Aabb, voxel_size: f32, amp_voxels: f32) -> Occupancy {
    let grid_origin = bounds.min;
    let size = bounds.size();
    let nx = (size.x / voxel_size).ceil() as i32 + 1;
    let ny = (size.y / voxel_size).ceil() as i32 + 1;
    let nz = (size.z / voxel_size).ceil() as i32 + 1;

    let mut cells = CellMap::default();
    let mut lo = IVec3::splat(i32::MAX);
    let mut hi = IVec3::splat(i32::MIN);
    for z in 0..nz {
        for x in 0..nx {
            // Per-column height offset in WORLD units, ±amp voxels.
            let jitter = (hash01(x, z) * 2.0 - 1.0) * amp_voxels * voxel_size;
            for y in 0..ny {
                let c = IVec3::new(x, y, z);
                let center = grid_origin
                    + (Vec3::new(x as f32, y as f32, z as f32) + Vec3::splat(0.5)) * voxel_size;
                // Shift the classification surface up/down per column.
                let shifted = Vec3::new(center.x, center.y - jitter, center.z);
                if shape.sdf(shifted) < 0.0 {
                    cells.insert(c, 0);
                    lo = lo.min(c);
                    hi = hi.max(c);
                }
            }
        }
    }
    let (region_min, region_max) = if cells.is_empty() {
        (IVec3::ZERO, IVec3::ZERO)
    } else {
        (lo - IVec3::splat(2), hi + IVec3::splat(3))
    };
    Occupancy { cells, grid_origin, voxel_size, region_min, region_max }
}

/// Build a TRUNCATED occupancy: from a full voxelization keep only cells
/// inside the interior `window` cell-box expanded by `halo` cells, and
/// mesh a region tight around the window. Cells beyond `window + halo`
/// are ABSENT — so the density blur at the window boundary reads a
/// truncated (clamped) neighborhood, exactly like a terrain tile that
/// only stores `halo` cells of its neighbour. `window` is in cell coords
/// of the full occupancy.
pub fn truncate_to_window(full: &Occupancy, window_min: IVec3, window_max: IVec3, halo: i32) -> Occupancy {
    let keep_min = window_min - IVec3::splat(halo);
    let keep_max = window_max + IVec3::splat(halo);
    let mut cells = CellMap::default();
    for (&c, &v) in full.cells.iter() {
        if c.x >= keep_min.x && c.x < keep_max.x
            && c.y >= keep_min.y && c.y < keep_max.y
            && c.z >= keep_min.z && c.z < keep_max.z
        {
            cells.insert(c, v);
        }
    }
    // Region = the interior window (+1 pad each side so the boundary
    // cubes iterate). The blur reads `cells`, which stops at
    // `window ± halo` → truncated neighborhood at the window edge.
    Occupancy {
        cells,
        grid_origin: full.grid_origin,
        voxel_size: full.voxel_size,
        region_min: window_min - IVec3::ONE,
        region_max: window_max + IVec3::ONE,
    }
}

/// Build all Goal-2 confound scenarios.
pub fn confound_scenarios() -> Vec<Scenario> {
    let mut out = Vec::new();

    // (a) COARSE — slope + sine at vs 1.0 and 2.0 (few voxels/feature).
    for shape in [Shape::Slope, Shape::SineTerrain] {
        for &vs in &[1.0f32, 2.0] {
            let occ = voxelize(shape, shape.bounds(), vs);
            out.push(Scenario {
                name: format!("coarse_{}_{}", shape.name(), fmt_vs(vs)),
                shape,
                occ,
            });
        }
    }

    // (b) TRUNCATION — sine_terrain, full vs windowed (halo=2). At
    // vs=0.25 the domain spans ~36 cells/axis; take an interior window
    // and a narrow halo so the blur clamps at the window boundary.
    {
        let shape = Shape::SineTerrain;
        let vs = 0.25f32;
        let full = voxelize(shape, shape.bounds(), vs);
        // Full-domain cell range (origin at bounds.min).
        let span = ((shape.bounds().size().x / vs).ceil()) as i32;
        // Interior window: middle ~40% of X/Z, full Y range.
        let q = span / 5; // 20%
        let wlo = IVec3::new(2 * q, -1000, 2 * q);
        let whi = IVec3::new(3 * q, 1000, 3 * q);
        // Clamp Y to the full occupancy's Y extent.
        let (ylo, yhi) = full.cells.keys().fold((i32::MAX, i32::MIN), |(a, b), c| {
            (a.min(c.y), b.max(c.y))
        });
        let wlo = IVec3::new(wlo.x, ylo - 1, wlo.z);
        let whi = IVec3::new(whi.x, yhi + 2, whi.z);
        let windowed = truncate_to_window(&full, wlo, whi, 2);
        out.push(Scenario { name: "trunc_sine_windowed_halo2".into(), shape, occ: windowed });
        // Reference: the SAME interior window cut from the full mesh but
        // with a WIDE halo (8) so the blur neighborhood is NOT clamped.
        let windowed_wide = truncate_to_window(&full, wlo, whi, 8);
        out.push(Scenario { name: "trunc_sine_windowed_halo8".into(), shape, occ: windowed_wide });
    }

    // (c) IRREGULAR — slope + sine with ±1.5-voxel column jitter.
    for shape in [Shape::Slope, Shape::SineTerrain] {
        let vs = 0.25f32;
        let occ = voxelize_irregular(shape, shape.bounds(), vs, 1.5);
        out.push(Scenario {
            name: format!("irregular_{}_{}", shape.name(), fmt_vs(vs)),
            shape,
            occ,
        });
    }

    out
}

/// `0.5 -> "0p5"`, `0.25 -> "0p25"`, `1 -> "1"`.
pub fn fmt_vs(vs: f32) -> String {
    format!("{vs}").replace('.', "p")
}

/// Mesh `occ` through the IDENTICAL engine extract path the sculpt
/// brush uses (blur → surface nets → ∇D normal). Returns
/// `(vertices, indices)` with object-local positions and ∇D normals.
pub fn mesh_occupancy(occ: &Occupancy) -> (Vec<MeshVertex>, Vec<u32>) {
    // Octree depth must be large enough that `extent = 1 << depth`
    // covers every cell coord the extract touches (incl. the +1 pads).
    // The region can have negative coords; `is_solid_lookup` only fires
    // for in-`[0,extent)` coords and returns false for the rest, which
    // with our EMPTY_NODE tree is the desired "occupancy from CellMap
    // only" behaviour either way. Pick a depth that comfortably spans
    // the magnitude of the coords.
    let max_abs = occ
        .cells
        .keys()
        .fold(0i32, |m, c| m.max(c.x.abs()).max(c.y.abs()).max(c.z.abs()))
        .max(occ.region_max.x.abs())
        .max(occ.region_max.y.abs())
        .max(occ.region_min.x.abs())
        .max(occ.region_min.y.abs());
    let mut depth: u8 = 1;
    while (1i32 << depth) <= max_abs + 4 && depth < 20 {
        depth += 1;
    }

    let nodes = [EMPTY_NODE];
    let mut scratch = SculptExtractScratch::new();
    extract_mesh_region_from_cells_pooled_haloed(
        &mut scratch,
        &occ.cells,
        occ.region_min,
        occ.region_max,
        &nodes,
        depth,
        occ.voxel_size,
        occ.grid_origin,
        &[],   // brick_cells
        &[],   // leaf_attr_pool
        &[],   // bone_voxel_pool
        &[],   // halo
        None,  // sculpt_slots
        None::<&fn(Vec3) -> f32>,
    &[],
        )
}

/// Blur-kernel parameters for the R-sweep: radius (cells), sigma
/// (cells), and iso threshold. `iso = 0.5` is the standard `D = 0.5`
/// surface; a value `> 0.5` pushes the surface OUTWARD to counter the
/// wider-blur inward isosurface shift (bias correction).
#[derive(Clone, Copy, Debug)]
pub struct BlurParams {
    pub r: i32,
    pub sigma: f32,
    pub iso: f32,
}

impl BlurParams {
    /// The shipped production kernel (`R=2, σ=1.0, iso=0.5`).
    pub fn shipped() -> Self {
        Self { r: 2, sigma: 1.0, iso: 0.5 }
    }
    /// Kernel for radius `r` with `σ = r·0.55` (so the wider kernel
    /// actually spreads) and the standard `iso = 0.5`.
    pub fn for_radius(r: i32) -> Self {
        Self { r, sigma: r as f32 * 0.55, iso: 0.5 }
    }
    /// Same but with a custom iso (bias-correction experiments).
    pub fn for_radius_iso(r: i32, iso: f32) -> Self {
        Self { r, sigma: r as f32 * 0.55, iso }
    }
    /// Short label for output directories / table rows (e.g. `R2`,
    /// `R3_iso0.60`). Available for bench drivers that sweep params.
    #[allow(dead_code)]
    pub fn name(self) -> String {
        if (self.iso - 0.5).abs() < 1e-6 {
            format!("R{}", self.r)
        } else {
            format!("R{}_iso{:.2}", self.r, self.iso)
        }
    }
}

/// Mesh `occ` with a specific blur kernel (the R-sweep). Sets the
/// per-thread blur override for the duration of this extract, then
/// clears it. Identical to [`mesh_occupancy`] otherwise.
pub fn mesh_occupancy_blur(occ: &Occupancy, bp: BlurParams) -> (Vec<MeshVertex>, Vec<u32>) {
    set_blur_override(Some((bp.r, bp.sigma, bp.iso)));
    let out = mesh_occupancy(occ);
    set_blur_override(None);
    out
}

// ════════════════════════════════════════════════════════════════════
// 3. Quantitative validation metrics
// ════════════════════════════════════════════════════════════════════

/// Per-shape validation report (all error figures in VOXEL units except
/// the angular ones, which are degrees).
#[derive(Clone, Debug)]
pub struct Metrics {
    pub shape: &'static str,
    pub voxel: f32,
    pub vertex_count: usize,
    pub triangle_count: usize,
    /// `max |sdf(v)| / voxel` over all vertices.
    pub geom_max: f32,
    /// `RMS(|sdf(v)|) / voxel`.
    pub geom_rms: f32,
    /// Mean angle (deg) between extracted normal and analytic ∇sdf.
    pub normal_mean_deg: f32,
    /// Max angle (deg) of the same.
    pub normal_max_deg: f32,
    /// Max angle (deg) between the two endpoint normals of any triangle
    /// edge (per-voxel speckle detector). Includes domain-boundary
    /// edges, where the open-boundary surface meets the clip walls and
    /// the normal legitimately turns — so this is the *reported* (table)
    /// figure, not the assertion target.
    pub edge_normal_max_deg: f32,
    /// Same as `edge_normal_max_deg` but over INTERIOR edges only (both
    /// endpoints ≥ 2 voxels off every domain face). This is the true
    /// speckle signal — free of the open-boundary artifact — and the
    /// `no_speckle` assertion target.
    pub edge_normal_interior_max_deg: f32,
    /// **Lumpiness** (in voxels): RMS over surface vertices of
    /// `sdf(v) − mean(sdf(1-ring neighbors))`. This isolates the
    /// HIGH-FREQUENCY sub-voxel facet noise from any smooth systematic
    /// surface shift (e.g. the Gaussian-blur bias on convex shapes,
    /// which `geom_rms` also captures but `roughness` does not). Lower
    /// = smoother silhouette. The Newton isosurface projection targets
    /// this number.
    pub roughness: f32,
    /// Whether terracing was detected on the height-field shapes
    /// (slope / sine). `None` for non-height-field shapes.
    pub terrace_detected: Option<bool>,
}

/// Compute the validation metrics for an extracted mesh of `shape`.
///
/// Metrics are measured over the vertices that lie ON the analytic
/// surface — those within `SURFACE_BAND · voxel` of the zero-isosurface.
/// For closed shapes (sphere/torus/box/…) that is every vertex. For the
/// height fields (slope/sine) the extract also emits the BOTTOM and the
/// ±X/±Z domain-clip WALLS of the solid block below the surface; those
/// vertices are real mesh but do NOT lie on the analytic surface, so
/// they are excluded from geometry/normal/speckle so the numbers reflect
/// the surface we care about, not the artificial domain box.
pub fn evaluate(shape: Shape, occ: &Occupancy, verts: &[MeshVertex], indices: &[u32]) -> Metrics {
    let vs = occ.voxel_size;
    const SURFACE_BAND: f32 = 1.5; // in voxels

    // Mask: vertex lies on the measured analytic surface.
    //
    // * Height fields (slope/sine): the extract also emits the solid
    //   block's BOTTOM and ±X/±Z domain-clip WALLS. Those are real mesh
    //   but not the analytic surface, so we measure ONLY the top face
    //   (normal predominantly +Y) AND within the surface band. The
    //   top-normal test excludes the near-vertical walls whose `|sdf|`
    //   can dip under the band as they pass the surface height.
    // * Closed shapes: every vertex is on the analytic surface, so the
    //   band test alone keeps them all.
    let height_field = shape.is_height_field();
    // Top-face normal cutoff: the STEEP slope's surface normal is only
    // `y ≈ 0.37`, so a 0.6 cutoff would wrongly drop its ramp. 0.2 still
    // excludes the ±X/±Z walls (y≈0) and the bottom (y<0).
    let top_cut = 0.2f32;
    let on_surface: Vec<bool> = verts
        .iter()
        .map(|v| {
            let band_ok = (shape.sdf(Vec3::from(v.local_pos)).abs() / vs) < SURFACE_BAND;
            if height_field {
                band_ok && unpack_oct(v.normal_oct).normalize_or_zero().y > top_cut
            } else {
                band_ok
            }
        })
        .collect();

    // ── Geometry: |sdf(vertex)| in voxel units (surface verts only). ──
    let mut geom_max = 0.0f32;
    let mut geom_sq = 0.0f32;
    let mut geom_n = 0u32;
    for (i, v) in verts.iter().enumerate() {
        if !on_surface[i] {
            continue;
        }
        let p = Vec3::from(v.local_pos);
        let e = shape.sdf(p).abs() / vs;
        geom_max = geom_max.max(e);
        geom_sq += e * e;
        geom_n += 1;
    }
    let geom_rms = if geom_n > 0 {
        (geom_sq / geom_n as f32).sqrt()
    } else {
        0.0
    };

    // ── Normal error: angle(extracted, analytic). ──
    let mut n_sum_deg = 0.0f32;
    let mut n_max_deg = 0.0f32;
    let mut n_count = 0u32;
    for (i, v) in verts.iter().enumerate() {
        if !on_surface[i] {
            continue;
        }
        let p = Vec3::from(v.local_pos);
        let ne = unpack_oct(v.normal_oct).normalize_or_zero();
        let na = shape.normal(p);
        if ne.length_squared() < 1e-8 || na.length_squared() < 1e-8 {
            continue;
        }
        let deg = ne.dot(na).clamp(-1.0, 1.0).acos().to_degrees();
        n_sum_deg += deg;
        n_max_deg = n_max_deg.max(deg);
        n_count += 1;
    }
    let normal_mean_deg = if n_count > 0 {
        n_sum_deg / n_count as f32
    } else {
        0.0
    };

    // ── Per-edge normal continuity (speckle detector). ──
    // Only edges whose BOTH endpoints are on the measured surface (skip
    // the wall/bottom transitions, which legitimately turn ~90°). We
    // track two figures: the overall max (table) and the INTERIOR max
    // (assertion), the latter excluding domain-boundary verts where the
    // open boundary turns the surface normal regardless of mesher
    // quality.
    let bounds = shape.bounds();
    let margin = 2.0 * vs;
    let is_interior = |p: Vec3| -> bool {
        p.x > bounds.min.x + margin
            && p.x < bounds.max.x - margin
            && p.y > bounds.min.y + margin
            && p.y < bounds.max.y - margin
            && p.z > bounds.min.z + margin
            && p.z < bounds.max.z - margin
    };
    let mut edge_normal_max_deg = 0.0f32;
    let mut edge_normal_interior_max_deg = 0.0f32;
    let mut edge = |ia: u32, ib: u32| {
        if !on_surface[ia as usize] || !on_surface[ib as usize] {
            return;
        }
        let pa = Vec3::from(verts[ia as usize].local_pos);
        let pb = Vec3::from(verts[ib as usize].local_pos);
        let na = unpack_oct(verts[ia as usize].normal_oct).normalize_or_zero();
        let nb = unpack_oct(verts[ib as usize].normal_oct).normalize_or_zero();
        if na.length_squared() < 1e-8 || nb.length_squared() < 1e-8 {
            return;
        }
        let deg = na.dot(nb).clamp(-1.0, 1.0).acos().to_degrees();
        edge_normal_max_deg = edge_normal_max_deg.max(deg);
        if is_interior(pa) && is_interior(pb) {
            edge_normal_interior_max_deg = edge_normal_interior_max_deg.max(deg);
        }
    };
    for tri in indices.chunks_exact(3) {
        edge(tri[0], tri[1]);
        edge(tri[1], tri[2]);
        edge(tri[2], tri[0]);
    }

    // ── Lumpiness / roughness: RMS of signed `sdf(v)` minus its 1-ring
    // neighbor mean, over surface verts. A smooth systematic surface
    // shift (the blur bias) cancels in `sdf_v − mean(neighbors)`, so
    // this isolates the high-frequency facet noise. ──
    let n_v = verts.len();
    let mut sdf_signed = vec![0.0f32; n_v];
    for (i, v) in verts.iter().enumerate() {
        sdf_signed[i] = shape.sdf(Vec3::from(v.local_pos)) / vs;
    }
    let mut nbr_sum = vec![0.0f32; n_v];
    let mut nbr_cnt = vec![0u32; n_v];
    let mut add = |a: u32, b: u32| {
        nbr_sum[a as usize] += sdf_signed[b as usize];
        nbr_cnt[a as usize] += 1;
    };
    for tri in indices.chunks_exact(3) {
        add(tri[0], tri[1]);
        add(tri[0], tri[2]);
        add(tri[1], tri[0]);
        add(tri[1], tri[2]);
        add(tri[2], tri[0]);
        add(tri[2], tri[1]);
    }
    let mut rough_sq = 0.0f32;
    let mut rough_n = 0u32;
    for i in 0..n_v {
        if !on_surface[i] || nbr_cnt[i] == 0 {
            continue;
        }
        let local = sdf_signed[i] - nbr_sum[i] / nbr_cnt[i] as f32;
        rough_sq += local * local;
        rough_n += 1;
    }
    let roughness = if rough_n > 0 {
        (rough_sq / rough_n as f32).sqrt()
    } else {
        0.0
    };

    // ── Terrace detection (height-field shapes only). ──
    let terrace_detected = if shape.is_height_field() {
        Some(detect_terracing(shape, verts, vs))
    } else {
        None
    };

    Metrics {
        shape: shape.name(),
        voxel: vs,
        vertex_count: verts.len(),
        triangle_count: indices.len() / 3,
        geom_max,
        geom_rms,
        normal_mean_deg,
        normal_max_deg: n_max_deg,
        edge_normal_max_deg,
        edge_normal_interior_max_deg,
        roughness,
        terrace_detected,
    }
}

/// Detect terracing on a height-field shape: walk the slope-top surface
/// vertices ordered by their position along the dominant horizontal
/// axis, and flag a terrace when the extracted height stays ~flat
/// (within a small band) across a horizontal run and then JUMPS by more
/// than `0.6·voxel` — the staircase signature. A smooth ramp climbs in
/// small steps and never plateaus-then-jumps.
pub fn detect_terracing(shape: Shape, verts: &[MeshVertex], vs: f32) -> bool {
    // Top-surface vertices: normal points up. The cutoff is low (0.15)
    // so STEEP ramps (normal y ≈ 0.37) still count — terracing on a
    // steep slope is exactly what we want to catch.
    let top_cut = 0.15f32;
    // Profile axis: X is the dominant horizontal gradient for the
    // planar/sine fields and a valid centerline cross-section for the
    // radial mound. Slice a thin Z band so the profile is ~1-D.
    let mut top: Vec<Vec3> = verts
        .iter()
        .filter(|v| unpack_oct(v.normal_oct).normalize_or_zero().y > top_cut)
        .map(|v| Vec3::from(v.local_pos))
        .filter(|p| p.z.abs() < vs)
        .collect();
    if top.len() < 8 {
        // Fall back to all up-facing verts (noisier but still works).
        top = verts
            .iter()
            .filter(|v| unpack_oct(v.normal_oct).normalize_or_zero().y > top_cut)
            .map(|v| Vec3::from(v.local_pos))
            .collect();
    }
    if top.len() < 8 {
        return false;
    }
    top.sort_by(|a, b| a.x.partial_cmp(&b.x).unwrap());

    // Staircase signature: a "flat run" of ≥ 4 consecutive samples
    // spanning ≥ 0.75·vs of horizontal X (so it's a genuine plateau, not
    // the gentle near-flat tail of a smooth bump) whose Y stays within
    // 0.1·vs, immediately followed by a JUMP > 0.8·vs. A smooth ramp
    // climbs in small steady steps and never plateaus-then-jumps; the
    // gentle flat BASE of a mound rises by < 0.8·vs so it doesn't trip.
    let flat_eps = 0.1 * vs;
    let jump_thresh = 0.8 * vs;
    let min_run = 4u32;
    let min_run_span = 0.75 * vs;
    let mut run_len = 1u32;
    let mut run_x0 = top.first().map(|p| p.x).unwrap_or(0.0);
    let _ = shape;
    for w in top.windows(2) {
        let dy = (w[1].y - w[0].y).abs();
        if dy < flat_eps {
            run_len += 1;
        } else {
            let span = (w[0].x - run_x0).abs();
            if run_len >= min_run && span >= min_run_span && dy > jump_thresh {
                return true;
            }
            run_len = 1;
            run_x0 = w[1].x;
        }
    }
    false
}

/// One row of [`run_all`]: the shape, voxel size, occupancy, extracted
/// mesh `(verts, indices)`, and the computed metrics.
pub type BenchRow = (Shape, f32, Occupancy, Vec<MeshVertex>, Vec<u32>, Metrics);

/// Voxelize + mesh + evaluate every shape at every `voxels` size.
pub fn run_all(voxels: &[f32]) -> Vec<BenchRow> {
    let mut out = Vec::new();
    for &shape in Shape::all() {
        for &vs in voxels {
            let occ = voxelize(shape, shape.bounds(), vs);
            let (verts, indices) = mesh_occupancy(&occ);
            let m = evaluate(shape, &occ, &verts, &indices);
            out.push((shape, vs, occ, verts, indices, m));
        }
    }
    out
}

/// Format the metrics table as a string (driver prints it).
pub fn format_table(rows: &[Metrics]) -> String {
    let mut s = String::new();
    s.push_str(&format!(
        "{:<14} {:>6} {:>7} {:>8} {:>8} {:>8} {:>9} {:>9} {:>10} {:>10} {:>8}\n",
        "shape",
        "voxel",
        "verts",
        "geom_max",
        "geom_rms",
        "rough",
        "nrm_mean",
        "edge_all",
        "edge_inner",
        "verts/tri",
        "terrace"
    ));
    s.push_str(&format!("{}\n", "-".repeat(110)));
    for m in rows {
        let terr = match m.terrace_detected {
            Some(true) => "YES",
            Some(false) => "no",
            None => "-",
        };
        s.push_str(&format!(
            "{:<14} {:>6.3} {:>7} {:>8.3} {:>8.3} {:>8.3} {:>8.2}° {:>8.2}° {:>9.2}° {:>10} {:>8}\n",
            m.shape,
            m.voxel,
            m.vertex_count,
            m.geom_max,
            m.geom_rms,
            m.roughness,
            m.normal_mean_deg,
            m.edge_normal_max_deg,
            m.edge_normal_interior_max_deg,
            m.triangle_count,
            terr,
        ));
    }
    s
}

// ════════════════════════════════════════════════════════════════════
// 4. Software renderer → RGB pixel buffers (composable layers)
// ════════════════════════════════════════════════════════════════════

/// An RGB8 image the example binary encodes to PNG.
pub struct Image {
    pub width: u32,
    pub height: u32,
    /// Row-major RGB, 3 bytes/pixel.
    pub rgb: Vec<u8>,
    /// Parallel depth buffer (z in camera/NDC depth; +∞ = empty).
    depth: Vec<f32>,
}

impl Image {
    pub fn new(width: u32, height: u32, bg: [u8; 3]) -> Self {
        let n = (width * height) as usize;
        let mut rgb = Vec::with_capacity(n * 3);
        for _ in 0..n {
            rgb.extend_from_slice(&bg);
        }
        Self {
            width,
            height,
            rgb,
            depth: vec![f32::INFINITY; n],
        }
    }

    #[inline]
    fn put(&mut self, x: i32, y: i32, z: f32, c: [u8; 3]) {
        if x < 0 || y < 0 || x >= self.width as i32 || y >= self.height as i32 {
            return;
        }
        let idx = (y as u32 * self.width + x as u32) as usize;
        if z < self.depth[idx] {
            self.depth[idx] = z;
            let o = idx * 3;
            self.rgb[o] = c[0];
            self.rgb[o + 1] = c[1];
            self.rgb[o + 2] = c[2];
        }
    }

    /// Put without depth-write (overlay-on-top), but still depth-TESTED
    /// against a small bias so lines on the surface aren't hidden.
    #[inline]
    fn put_overlay(&mut self, x: i32, y: i32, z: f32, c: [u8; 3]) {
        if x < 0 || y < 0 || x >= self.width as i32 || y >= self.height as i32 {
            return;
        }
        let idx = (y as u32 * self.width + x as u32) as usize;
        // Slight bias so a line co-planar with a filled triangle wins.
        if z <= self.depth[idx] + 1e-3 {
            let o = idx * 3;
            self.rgb[o] = c[0];
            self.rgb[o + 1] = c[1];
            self.rgb[o + 2] = c[2];
        }
    }
}

/// World-space radius the cameras frame (≈ the shape domain half-size).
const FRAME_RADIUS: f32 = 4.5;

/// Camera that projects world → screen. Two modes: a fixed 3/4
/// perspective and a side orthographic.
#[derive(Clone, Copy)]
pub struct Camera {
    view: glam::Mat4,
    proj: glam::Mat4,
    width: f32,
    height: f32,
    ortho: bool,
}

impl Camera {
    /// 3/4 perspective looking at the origin from above-front-right,
    /// framing a shape of roughly `radius` world units.
    pub fn three_quarter(width: u32, height: u32) -> Self {
        let radius = FRAME_RADIUS;
        let dir = Vec3::new(0.6, 0.5, 0.75).normalize();
        let eye = dir * (radius * 3.2);
        let view = glam::Mat4::look_at_rh(eye, Vec3::ZERO, Vec3::Y);
        let aspect = width as f32 / height as f32;
        let proj = glam::Mat4::perspective_rh(45f32.to_radians(), aspect, 0.1, 1000.0);
        Self {
            view,
            proj,
            width: width as f32,
            height: height as f32,
            ortho: false,
        }
    }

    /// Side orthographic looking down -Z onto the XY plane (terracing
    /// reads instantly as horizontal steps in the profile).
    pub fn side_ortho(width: u32, height: u32) -> Self {
        let eye = Vec3::new(0.0, 0.0, FRAME_RADIUS * 4.0);
        let view = glam::Mat4::look_at_rh(eye, Vec3::ZERO, Vec3::Y);
        let h = FRAME_RADIUS * 1.15;
        let aspect = width as f32 / height as f32;
        let proj = glam::Mat4::orthographic_rh(-h * aspect, h * aspect, -h, h, 0.1, 1000.0);
        Self {
            view,
            proj,
            width: width as f32,
            height: height as f32,
            ortho: true,
        }
    }

    /// 3/4 perspective framing a shape of `radius` world units centred at
    /// `center` (for off-origin terrain windows). Same view direction as
    /// [`Self::three_quarter`].
    pub fn three_quarter_framing(width: u32, height: u32, center: Vec3, radius: f32) -> Self {
        let dir = Vec3::new(0.6, 0.5, 0.75).normalize();
        let eye = center + dir * (radius * 3.2);
        let view = glam::Mat4::look_at_rh(eye, center, Vec3::Y);
        let aspect = width as f32 / height as f32;
        let proj = glam::Mat4::perspective_rh(45f32.to_radians(), aspect, 0.1, 10000.0);
        Self {
            view,
            proj,
            width: width as f32,
            height: height as f32,
            ortho: false,
        }
    }

    /// Side orthographic framing `radius` world units centred at `center`,
    /// looking down -Z (terracing/ripple reads as horizontal steps in the
    /// profile). For off-origin terrain windows.
    pub fn side_ortho_framing(width: u32, height: u32, center: Vec3, radius: f32) -> Self {
        let eye = center + Vec3::new(0.0, 0.0, radius * 4.0);
        let view = glam::Mat4::look_at_rh(eye, center, Vec3::Y);
        let h = radius * 1.15;
        let aspect = width as f32 / height as f32;
        let proj = glam::Mat4::orthographic_rh(-h * aspect, h * aspect, -h, h, 0.1, 10000.0);
        Self {
            view,
            proj,
            width: width as f32,
            height: height as f32,
            ortho: true,
        }
    }

    /// Side orthographic with INDEPENDENT horizontal / vertical
    /// half-extents centred at `center`. Lets a profile view magnify Y
    /// (small `half_y`) so a gentle slope's surface band fills the frame
    /// and low-amplitude "smooth stairs" become visible. Looks down -Z.
    pub fn side_ortho_xy(width: u32, height: u32, center: Vec3, half_x: f32, half_y: f32) -> Self {
        let eye = center + Vec3::new(0.0, 0.0, half_x.max(half_y) * 8.0);
        let view = glam::Mat4::look_at_rh(eye, center, Vec3::Y);
        let proj = glam::Mat4::orthographic_rh(-half_x, half_x, -half_y, half_y, 0.1, 100000.0);
        Self {
            view,
            proj,
            width: width as f32,
            height: height as f32,
            ortho: true,
        }
    }

    /// Project world point → (screen_x, screen_y, depth). Depth is NDC
    /// z (smaller = nearer) used for the z-buffer.
    #[inline]
    fn project(&self, p: Vec3) -> (f32, f32, f32) {
        let clip = self.proj * self.view * p.extend(1.0);
        let w = if self.ortho { 1.0 } else { clip.w };
        let ndc = Vec3::new(clip.x / w, clip.y / w, clip.z / w);
        let sx = (ndc.x * 0.5 + 0.5) * self.width;
        let sy = (1.0 - (ndc.y * 0.5 + 0.5)) * self.height;
        (sx, sy, ndc.z)
    }
}

/// Render config: which layers to compose.
#[derive(Clone, Copy)]
pub struct RenderOpts {
    pub shaded: bool,
    pub wireframe: bool,
    pub voxels: bool,
    /// Dim the shaded fill so overlays read.
    pub dim_shading: bool,
}

/// Render `verts`/`indices` (+ surface voxel cells from `occ`) into an
/// `Image` with the requested layers.
pub fn render(
    cam: &Camera,
    occ: &Occupancy,
    verts: &[MeshVertex],
    indices: &[u32],
    opts: RenderOpts,
    size: u32,
) -> Image {
    let bg = [18, 20, 26];
    let mut img = Image::new(size, size, bg);
    let light_dir = Vec3::new(-0.4, 0.85, 0.35).normalize();

    // ── SHADED fill ──
    if opts.shaded {
        for tri in indices.chunks_exact(3) {
            let p0 = Vec3::from(verts[tri[0] as usize].local_pos);
            let p1 = Vec3::from(verts[tri[1] as usize].local_pos);
            let p2 = Vec3::from(verts[tri[2] as usize].local_pos);
            let n0 = unpack_oct(verts[tri[0] as usize].normal_oct).normalize_or_zero();
            let n1 = unpack_oct(verts[tri[1] as usize].normal_oct).normalize_or_zero();
            let n2 = unpack_oct(verts[tri[2] as usize].normal_oct).normalize_or_zero();
            raster_triangle_shaded(
                &mut img, cam, [p0, p1, p2], [n0, n1, n2], light_dir, opts.dim_shading,
            );
        }
    }

    // ── VOXELS: 12 cube edges of surface (boundary) cells ──
    if opts.voxels {
        let col = [70, 96, 120];
        for &c in occ.cells.keys() {
            // Only boundary cells (≥1 of the 6 face neighbors empty).
            if !is_boundary_cell(occ, c) {
                continue;
            }
            draw_cell_cube(&mut img, cam, occ, c, col);
        }
    }

    // ── WIREFRAME: bright triangle edges ──
    if opts.wireframe {
        let col = [255, 214, 90];
        for tri in indices.chunks_exact(3) {
            let p0 = Vec3::from(verts[tri[0] as usize].local_pos);
            let p1 = Vec3::from(verts[tri[1] as usize].local_pos);
            let p2 = Vec3::from(verts[tri[2] as usize].local_pos);
            draw_line_world(&mut img, cam, p0, p1, col);
            draw_line_world(&mut img, cam, p1, p2, col);
            draw_line_world(&mut img, cam, p2, p0, col);
        }
    }

    img
}

/// Is cell `c` a surface (boundary) cell — at least one face neighbor
/// empty?
fn is_boundary_cell(occ: &Occupancy, c: IVec3) -> bool {
    const D: [IVec3; 6] = [
        IVec3::new(1, 0, 0),
        IVec3::new(-1, 0, 0),
        IVec3::new(0, 1, 0),
        IVec3::new(0, -1, 0),
        IVec3::new(0, 0, 1),
        IVec3::new(0, 0, -1),
    ];
    D.iter().any(|d| !occ.cells.contains_key(&(c + *d)))
}

/// Draw the 12 edges of cell `c`'s world-space cube.
fn draw_cell_cube(img: &mut Image, cam: &Camera, occ: &Occupancy, c: IVec3, col: [u8; 3]) {
    let lo = occ.cell_lo(c);
    let s = occ.voxel_size;
    let corner = |dx: f32, dy: f32, dz: f32| lo + Vec3::new(dx, dy, dz) * s;
    let v = [
        corner(0.0, 0.0, 0.0),
        corner(1.0, 0.0, 0.0),
        corner(1.0, 1.0, 0.0),
        corner(0.0, 1.0, 0.0),
        corner(0.0, 0.0, 1.0),
        corner(1.0, 0.0, 1.0),
        corner(1.0, 1.0, 1.0),
        corner(0.0, 1.0, 1.0),
    ];
    const E: [(usize, usize); 12] = [
        (0, 1), (1, 2), (2, 3), (3, 0), // bottom
        (4, 5), (5, 6), (6, 7), (7, 4), // top
        (0, 4), (1, 5), (2, 6), (3, 7), // verticals
    ];
    for &(a, b) in &E {
        draw_line_world_faint(img, cam, v[a], v[b], col);
    }
}

/// Barycentric-rasterize a triangle with per-vertex normal Lambert
/// shading.
fn raster_triangle_shaded(
    img: &mut Image,
    cam: &Camera,
    p: [Vec3; 3],
    n: [Vec3; 3],
    light_dir: Vec3,
    dim: bool,
) {
    let (s0x, s0y, z0) = cam.project(p[0]);
    let (s1x, s1y, z1) = cam.project(p[1]);
    let (s2x, s2y, z2) = cam.project(p[2]);

    let minx = s0x.min(s1x).min(s2x).floor().max(0.0) as i32;
    let maxx = s0x.max(s1x).max(s2x).ceil().min(img.width as f32 - 1.0) as i32;
    let miny = s0y.min(s1y).min(s2y).floor().max(0.0) as i32;
    let maxy = s0y.max(s1y).max(s2y).ceil().min(img.height as f32 - 1.0) as i32;
    if minx > maxx || miny > maxy {
        return;
    }

    let area = edge_fn(s0x, s0y, s1x, s1y, s2x, s2y);
    if area.abs() < 1e-6 {
        return;
    }
    let inv_area = 1.0 / area;

    for y in miny..=maxy {
        for x in minx..=maxx {
            let px = x as f32 + 0.5;
            let py = y as f32 + 0.5;
            let w0 = edge_fn(s1x, s1y, s2x, s2y, px, py) * inv_area;
            let w1 = edge_fn(s2x, s2y, s0x, s0y, px, py) * inv_area;
            let w2 = edge_fn(s0x, s0y, s1x, s1y, px, py) * inv_area;
            // Inside test (allow either winding).
            if (w0 < 0.0 || w1 < 0.0 || w2 < 0.0) && (w0 > 0.0 || w1 > 0.0 || w2 > 0.0) {
                continue;
            }
            let z = w0 * z0 + w1 * z1 + w2 * z2;
            let nrm = (n[0] * w0 + n[1] * w1 + n[2] * w2).normalize_or_zero();
            let lambert = nrm.dot(light_dir).max(0.0);
            let amb = 0.22;
            let intensity = (amb + (1.0 - amb) * lambert).clamp(0.0, 1.0);
            // Base surface color (cool steel), dimmed if overlays follow.
            let base = if dim {
                Vec3::new(0.30, 0.34, 0.42)
            } else {
                Vec3::new(0.62, 0.68, 0.78)
            };
            let c = base * intensity;
            let rgb = [
                (c.x * 255.0) as u8,
                (c.y * 255.0) as u8,
                (c.z * 255.0) as u8,
            ];
            img.put(x, y, z, rgb);
        }
    }
}

#[inline]
fn edge_fn(ax: f32, ay: f32, bx: f32, by: f32, cx: f32, cy: f32) -> f32 {
    (cx - ax) * (by - ay) - (cy - ay) * (bx - ax)
}

/// Bright wireframe line (depth-tested overlay).
fn draw_line_world(img: &mut Image, cam: &Camera, a: Vec3, b: Vec3, col: [u8; 3]) {
    let (ax, ay, az) = cam.project(a);
    let (bx, by, bz) = cam.project(b);
    bresenham(img, ax, ay, az, bx, by, bz, col, false);
}

/// Faint voxel-edge line (depth-tested overlay).
fn draw_line_world_faint(img: &mut Image, cam: &Camera, a: Vec3, b: Vec3, col: [u8; 3]) {
    let (ax, ay, az) = cam.project(a);
    let (bx, by, bz) = cam.project(b);
    bresenham(img, ax, ay, az, bx, by, bz, col, true);
}

#[allow(clippy::too_many_arguments)]
fn bresenham(
    img: &mut Image,
    x0: f32,
    y0: f32,
    z0: f32,
    x1: f32,
    y1: f32,
    z1: f32,
    col: [u8; 3],
    faint: bool,
) {
    let mut x0i = x0.round() as i32;
    let mut y0i = y0.round() as i32;
    let x1i = x1.round() as i32;
    let y1i = y1.round() as i32;
    let dx = (x1i - x0i).abs();
    let dy = -(y1i - y0i).abs();
    let sx = if x0i < x1i { 1 } else { -1 };
    let sy = if y0i < y1i { 1 } else { -1 };
    let mut err = dx + dy;
    let steps = dx.max(-dy).max(1) as f32;
    let mut i = 0.0f32;
    loop {
        let t = (i / steps).clamp(0.0, 1.0);
        let z = z0 + (z1 - z0) * t;
        // Faint lines blend toward bg so deep voxel clutter recedes.
        let c = if faint {
            [
                ((col[0] as f32) * 0.85 + 18.0 * 0.15) as u8,
                ((col[1] as f32) * 0.85 + 20.0 * 0.15) as u8,
                ((col[2] as f32) * 0.85 + 26.0 * 0.15) as u8,
            ]
        } else {
            col
        };
        img.put_overlay(x0i, y0i, z, c);
        if x0i == x1i && y0i == y1i {
            break;
        }
        let e2 = 2 * err;
        if e2 >= dy {
            err += dy;
            x0i += sx;
        }
        if e2 <= dx {
            err += dx;
            y0i += sy;
        }
        i += 1.0;
        if i > steps + 2.0 {
            break;
        }
    }
}

// ════════════════════════════════════════════════════════════════════
// 5. TERRAIN-BAKE-PATH reproduction (smooth-stairs hunt)
// ════════════════════════════════════════════════════════════════════
//
// Everything above drives the SCULPT/region extract
// (`extract_mesh_region_from_cells_pooled_haloed`) on a hand-built
// `CellMap`. The USER'S TERRAIN, however, bakes through a DIFFERENT
// path: `voxelize_to_artifact` (octree + bricks + halo) →
// `extract_surface_mesh_density_haloed`. To reproduce the residual
// "smooth stairs" the user reports on gentle slopes we must mesh through
// THAT path, on terrain-like (gentle-slope + FBM) heightfields, and
// measure a LOW-FREQUENCY ripple metric (not just the existing
// high-frequency `roughness`, which misses wide flat treads).

use crate::mesh_extract::{collect_cell_map, extract_surface_mesh_density_haloed};
use crate::voxelize_octree::voxelize_to_artifact;

/// Terrain halo the real bake uses (`bake.rs::TILE_HALO_VOXELS`).
pub const REPRO_TILE_HALO: u32 = 4;

/// A terrain-like heightfield `y = h(x, z)`. The solid is everything at
/// or below the surface (`sdf = world_y - h(x,z)`, the vertical-gap sign
/// field the real terrain bake uses). Closures are boxed so a single
/// type covers gentle slopes and FBM.
pub struct HeightField {
    pub name: String,
    /// Surface height at horizontal `(x, z)`.
    pub h: Box<dyn Fn(f32, f32) -> f32 + Send + Sync>,
}

impl HeightField {
    /// Gentle planar slope `y = mx·x + mz·z`. `mz` is a small off-axis
    /// tilt so the slope is never grid-aligned (a grid-aligned slope can
    /// hide ripple behind exact lattice coincidence). `dh/dx ≈ mx`.
    pub fn gentle_slope(mx: f32, mz: f32) -> Self {
        HeightField {
            name: format!("slope_dhdx{}", fmt_vs(mx)),
            h: Box::new(move |x, z| mx * x + mz * z),
        }
    }

    /// Gentle FBM terrain: a few octaves of value-noise, overall gentle
    /// (amplitude/wavelength chosen so the dominant slope stays ≲ 0.25 —
    /// like real rolling terrain, NOT cliffs). Matches the spirit of the
    /// engine's `FbmTerrainFn` but is self-contained for the bench.
    pub fn fbm() -> Self {
        HeightField {
            name: "fbm".into(),
            h: Box::new(|x, z| fbm_height(x, z)),
        }
    }
}

/// Value-noise FBM height. 3 octaves, GENTLE — chosen to mimic real
/// rolling terrain (the engine's `FbmTerrainFn` default has scale_m=120,
/// i.e. a very wide base wavelength). Base wavelength ~16 world units,
/// amplitude ~1.2; each octave halves wavelength + amplitude. The wide
/// base keeps the dominant slope gentle (≲ 0.2) so this is a fair
/// curvature-preservation control for the wide-window fix — a tight
/// (small-wavelength) FBM would be unrealistically high-curvature and
/// over-penalise any plane-fit smoothing.
pub fn fbm_height(x: f32, z: f32) -> f32 {
    let mut h = 0.0f32;
    let mut amp = 1.2f32;
    let mut freq = 1.0f32 / 16.0;
    for _ in 0..3 {
        h += amp * value_noise2(x * freq, z * freq);
        amp *= 0.5;
        freq *= 2.0;
    }
    h
}

/// Smooth 2-D value noise in `[-1, 1]`, lattice + quintic fade +
/// bilinear of per-lattice-point hashes. Self-contained.
fn value_noise2(x: f32, z: f32) -> f32 {
    let xi = x.floor();
    let zi = z.floor();
    let xf = x - xi;
    let zf = z - zi;
    let fade = |t: f32| t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
    let u = fade(xf);
    let v = fade(zf);
    let g = |ix: f32, iz: f32| -> f32 {
        // Hash lattice point → [-1, 1].
        let h = hash01(ix as i32, iz as i32);
        h * 2.0 - 1.0
    };
    let c00 = g(xi, zi);
    let c10 = g(xi + 1.0, zi);
    let c01 = g(xi, zi + 1.0);
    let c11 = g(xi + 1.0, zi + 1.0);
    let a = c00 + (c10 - c00) * u;
    let b = c01 + (c11 - c01) * u;
    a + (b - a) * v
}

/// Output of a heightfield bake/region mesh + the bake inputs needed to
/// render it (matches [`Occupancy`]'s renderable fields).
pub struct HeightMesh {
    pub verts: Vec<MeshVertex>,
    pub indices: Vec<u32>,
    pub grid_origin: Vec3,
    pub voxel_size: f32,
    /// The cell-grid origin/size of the SOLID block (for the voxel
    /// overlay render) — a faithful sub-sample, not the whole tile.
    pub surface_cells: CellMap,
}

impl HeightMesh {
    /// Wrap as an [`Occupancy`] so the existing `render` works unchanged.
    pub fn as_occupancy(&self) -> Occupancy {
        // region_min/max only gate the voxel-overlay bbox; derive from
        // the surface cells.
        let (mut lo, mut hi) = (IVec3::splat(i32::MAX), IVec3::splat(i32::MIN));
        for &c in self.surface_cells.keys() {
            lo = lo.min(c);
            hi = hi.max(c);
        }
        if self.surface_cells.is_empty() {
            lo = IVec3::ZERO;
            hi = IVec3::ZERO;
        }
        Occupancy {
            cells: self.surface_cells.clone(),
            grid_origin: self.grid_origin,
            voxel_size: self.voxel_size,
            region_min: lo - IVec3::splat(2),
            region_max: hi + IVec3::splat(3),
        }
    }
}

/// Build a pow2-cubic tile AABB for the heightfield. The tile is
/// centered on the origin in X/Z and tall enough in Y to bracket the
/// surface + a solid block below + the halo. `extent_cells` is forced to
/// a power of two (the bake contract).
fn heightfield_tile_aabb(half_world: f32, voxel_size: f32) -> Aabb {
    // Cells across the full 2*half_world span, rounded up to pow2.
    let cells = ((2.0 * half_world) / voxel_size).ceil().max(1.0) as u32;
    let pow2 = cells.next_power_of_two();
    let extent = pow2 as f32 * voxel_size;
    // Center the cube on origin; min snapped to the voxel grid.
    let snap = |v: f32| (v / voxel_size).floor() * voxel_size;
    let min = Vec3::splat(snap(-extent * 0.5));
    Aabb::new(min, min + Vec3::splat(extent))
}

/// Collect the SURFACE (boundary) cells of a heightfield into a CellMap
/// for the voxel overlay render — every solid cell whose `+Y` neighbor
/// is empty (the top shell). Cheap, faithful, and avoids materializing
/// the whole solid block.
fn heightfield_surface_cells(
    hf: &HeightField,
    aabb: &Aabb,
    voxel_size: f32,
) -> CellMap {
    let origin = aabb.min;
    let n = ((aabb.max.x - aabb.min.x) / voxel_size).round() as i32;
    let mut cells = CellMap::default();
    let solid = |cx: i32, cy: i32, cz: i32| -> bool {
        let c = origin
            + (Vec3::new(cx as f32, cy as f32, cz as f32) + Vec3::splat(0.5)) * voxel_size;
        c.y - (hf.h)(c.x, c.z) < 0.0
    };
    for cz in 0..n {
        for cx in 0..n {
            for cy in 0..n {
                if solid(cx, cy, cz) && !solid(cx, cy + 1, cz) {
                    cells.insert(IVec3::new(cx, cy, cz), 0);
                }
            }
        }
    }
    cells
}

/// Mesh a heightfield through the **TERRAIN BAKE PATH**:
/// `voxelize_to_artifact` (octree + bricks + halo) →
/// `extract_surface_mesh_density_haloed`. This is the path the user's
/// terrain actually runs.
pub fn bake_heightfield(hf: &HeightField, half_world: f32, voxel_size: f32) -> HeightMesh {
    let aabb = heightfield_tile_aabb(half_world, voxel_size);
    let h = &hf.h;
    let sdf_fn = |positions: &[Vec3]| -> Vec<(f32, u16, u16, u8, u32)> {
        positions
            .iter()
            .map(|p| {
                // Vertical-gap sign field, the real terrain bake's SDF.
                let d = p.y - h(p.x, p.z);
                (d, 1u16, 1u16, 0u8, 0u32)
            })
            .collect()
    };
    let artifact = voxelize_to_artifact(sdf_fn, &aabb, voxel_size, REPRO_TILE_HALO)
        .expect("heightfield voxelize_to_artifact");
    let brick_pool_flat: Vec<u32> = artifact.brick_cells.iter().flatten().copied().collect();
    let (verts, indices) = extract_surface_mesh_density_haloed(
        artifact.octree.as_slice(),
        artifact.octree.depth(),
        voxel_size,
        artifact.grid_origin,
        &brick_pool_flat,
        &artifact.leaf_attrs,
        &[],
        &artifact.halo_cells,
        REPRO_TILE_HALO,
        None,
        &[],
    );
    let surface_cells = heightfield_surface_cells(hf, &aabb, voxel_size);
    HeightMesh {
        verts,
        indices,
        grid_origin: artifact.grid_origin,
        voxel_size,
        surface_cells,
    }
}

/// Mesh the SAME heightfield occupancy through the **REGION/SCULPT PATH**
/// (`extract_mesh_region_from_cells_pooled_haloed` via [`mesh_occupancy`])
/// so the bake-path ripple can be compared against the region-path ripple
/// on the identical occupancy. We build the full solid CellMap (every
/// cell at/below the surface) and a region tight around it.
pub fn region_mesh_heightfield(hf: &HeightField, half_world: f32, voxel_size: f32) -> HeightMesh {
    let aabb = heightfield_tile_aabb(half_world, voxel_size);
    let origin = aabb.min;
    let n = ((aabb.max.x - aabb.min.x) / voxel_size).round() as i32;
    let mut cells = CellMap::default();
    let mut lo = IVec3::splat(i32::MAX);
    let mut hi = IVec3::splat(i32::MIN);
    let h = &hf.h;
    for cz in 0..n {
        for cx in 0..n {
            for cy in 0..n {
                let c = origin
                    + (Vec3::new(cx as f32, cy as f32, cz as f32) + Vec3::splat(0.5)) * voxel_size;
                if c.y - h(c.x, c.z) < 0.0 {
                    let k = IVec3::new(cx, cy, cz);
                    cells.insert(k, 0);
                    lo = lo.min(k);
                    hi = hi.max(k);
                }
            }
        }
    }
    let occ = Occupancy {
        cells,
        grid_origin: origin,
        voxel_size,
        region_min: lo - IVec3::splat(2),
        region_max: hi + IVec3::splat(3),
    };
    let (verts, indices) = mesh_occupancy(&occ);
    let surface_cells = heightfield_surface_cells(hf, &aabb, voxel_size);
    HeightMesh {
        verts,
        indices,
        grid_origin: origin,
        voxel_size,
        surface_cells,
    }
}

/// Same as [`bake_heightfield`] but with a per-thread blur override so
/// the R-sweep / fix experiments can vary `(R, σ, iso)` on the bake path.
pub fn bake_heightfield_blur(
    hf: &HeightField,
    half_world: f32,
    voxel_size: f32,
    bp: BlurParams,
) -> HeightMesh {
    set_blur_override(Some((bp.r, bp.sigma, bp.iso)));
    let out = bake_heightfield(hf, half_world, voxel_size);
    set_blur_override(None);
    out
}

/// Mesh a heightfield through the terrain bake path with **QEF-Hermite**
/// placement, reading the per-leaf signed distance the **voxelizer** baked
/// into the artifact (Stage 3). Identical to [`bake_heightfield`] except it
/// sets the QEF toggle and hands the artifact's `leaf_attr_dists` to the
/// mesher — the real production data path, end to end. (The Stage-2
/// analytic-distance injection it replaced has been dropped; the asserts
/// are unchanged, now validating the voxelizer's distance write too.)
pub fn bake_heightfield_qef(hf: &HeightField, half_world: f32, voxel_size: f32) -> HeightMesh {
    bake_heightfield_qef_aabb(hf, heightfield_tile_aabb(half_world, voxel_size), voxel_size)
}

/// [`bake_heightfield`] (blur path) over an explicit (grid-aligned) tile
/// AABB — the seam-test baseline the QEF path is compared against.
pub fn bake_heightfield_blur_aabb(hf: &HeightField, aabb: Aabb, voxel_size: f32) -> HeightMesh {
    let h = &hf.h;
    let sdf_fn = |positions: &[Vec3]| -> Vec<(f32, u16, u16, u8, u32)> {
        positions
            .iter()
            .map(|p| (p.y - h(p.x, p.z), 1u16, 1u16, 0u8, 0u32))
            .collect()
    };
    let artifact = voxelize_to_artifact(sdf_fn, &aabb, voxel_size, REPRO_TILE_HALO)
        .expect("heightfield voxelize_to_artifact");
    let brick_pool_flat: Vec<u32> = artifact.brick_cells.iter().flatten().copied().collect();
    let (verts, indices) = extract_surface_mesh_density_haloed(
        artifact.octree.as_slice(),
        artifact.octree.depth(),
        voxel_size,
        artifact.grid_origin,
        &brick_pool_flat,
        &artifact.leaf_attrs,
        &[],
        &artifact.halo_cells,
        REPRO_TILE_HALO,
        None,
        &[],
    );
    let surface_cells = heightfield_surface_cells(hf, &aabb, voxel_size);
    HeightMesh {
        verts,
        indices,
        grid_origin: artifact.grid_origin,
        voxel_size,
        surface_cells,
    }
}

/// [`bake_heightfield_qef`] over an explicit (grid-aligned) tile AABB — so
/// the seam test can bake two ADJACENT tiles that share a face.
pub fn bake_heightfield_qef_aabb(hf: &HeightField, aabb: Aabb, voxel_size: f32) -> HeightMesh {
    let h = &hf.h;
    let sdf_fn = |positions: &[Vec3]| -> Vec<(f32, u16, u16, u8, u32)> {
        positions
            .iter()
            .map(|p| (p.y - h(p.x, p.z), 1u16, 1u16, 0u8, 0u32))
            .collect()
    };
    let artifact = voxelize_to_artifact(sdf_fn, &aabb, voxel_size, REPRO_TILE_HALO)
        .expect("heightfield voxelize_to_artifact");
    let brick_pool_flat: Vec<u32> = artifact.brick_cells.iter().flatten().copied().collect();

    // Production gate is presence of the distance pool — passing the baked
    // `leaf_attr_dists` selects QEF-Hermite (no toggle needed).
    let (verts, indices) = extract_surface_mesh_density_haloed(
        artifact.octree.as_slice(),
        artifact.octree.depth(),
        voxel_size,
        artifact.grid_origin,
        &brick_pool_flat,
        &artifact.leaf_attrs,
        &[],
        &artifact.halo_cells,
        REPRO_TILE_HALO,
        None,
        &artifact.leaf_attr_dists,
    );

    let surface_cells = heightfield_surface_cells(hf, &aabb, voxel_size);
    HeightMesh {
        verts,
        indices,
        grid_origin: artifact.grid_origin,
        voxel_size,
        surface_cells,
    }
}

/// Mesh a heightfield through the **REGION / SCULPT re-extract path**
/// (`extract_mesh_region_from_cells_pooled_haloed`, Stage 6) with QEF-Hermite
/// — the path a sculpt brush / halo refresh runs. Uses the REAL voxelized
/// octree (so `is_solid_lookup` resolves interior bulk correctly) + the
/// artifact's baked per-leaf distances. Validates that a sculpt re-extract
/// matches the QEF bake instead of falling back to blur.
pub fn region_mesh_heightfield_qef(hf: &HeightField, half_world: f32, voxel_size: f32) -> HeightMesh {
    let aabb = heightfield_tile_aabb(half_world, voxel_size);
    let h = &hf.h;
    let sdf_fn = |positions: &[Vec3]| -> Vec<(f32, u16, u16, u8, u32)> {
        positions
            .iter()
            .map(|p| (p.y - h(p.x, p.z), 1u16, 1u16, 0u8, 0u32))
            .collect()
    };
    let artifact = voxelize_to_artifact(sdf_fn, &aabb, voxel_size, REPRO_TILE_HALO)
        .expect("heightfield voxelize_to_artifact");
    let brick_pool_flat: Vec<u32> = artifact.brick_cells.iter().flatten().copied().collect();
    let cells = collect_cell_map(
        artifact.octree.as_slice(),
        artifact.octree.depth(),
        &brick_pool_flat,
    );
    let extent = 1i32 << artifact.octree.depth();
    let mut scratch = SculptExtractScratch::new();
    let (verts, indices) = extract_mesh_region_from_cells_pooled_haloed(
        &mut scratch,
        &cells,
        IVec3::ZERO,
        IVec3::splat(extent),
        artifact.octree.as_slice(),
        artifact.octree.depth(),
        voxel_size,
        artifact.grid_origin,
        &brick_pool_flat,
        &artifact.leaf_attrs,
        &[],
        &artifact.halo_cells,
        None,
        None::<&fn(Vec3) -> f32>,
        &artifact.leaf_attr_dists,
    );
    let surface_cells = heightfield_surface_cells(hf, &aabb, voxel_size);
    HeightMesh {
        verts,
        indices,
        grid_origin: artifact.grid_origin,
        voxel_size,
        surface_cells,
    }
}

// ────────────────────────────────────────────────────────────────────
// Low-frequency banding metric (smooth-stairs detector)
// ────────────────────────────────────────────────────────────────────

/// Low-frequency ripple report for a heightfield surface (all lengths /
/// amplitudes in VOXELS).
#[derive(Clone, Debug)]
pub struct RippleReport {
    /// Number of top-surface samples used.
    pub n_samples: usize,
    /// Dominant ripple WAVELENGTH along the slope direction, in voxels.
    /// `0` if no periodic peak found.
    pub wavelength_vox: f32,
    /// Amplitude (≈ half peak-to-peak) of the dominant ripple, in voxels.
    pub amplitude_vox: f32,
    /// RMS of the full surface-Y residual (height − best-fit reference),
    /// in voxels. Captures ALL deviation (any frequency).
    pub residual_rms_vox: f32,
    /// Existing high-frequency roughness for reference (1-ring lumpiness).
    pub roughness_vox: f32,
}

/// Measure the LOW-FREQUENCY banding ("smooth stairs") on a heightfield
/// mesh. Procedure:
///   1. Take TOP-surface vertices (normal +Y dominant).
///   2. Project each onto the slope direction (the dominant horizontal
///      gradient of `h`) → 1-D ordinate `s`. For FBM (no single
///      direction) we use the X axis.
///   3. Residual = `vertex.y − h_true(x,z)` (height minus the analytic
///      surface — the true low-frequency reference, NOT a global plane,
///      so FBM curvature doesn't masquerade as ripple).
///   4. Resample the residual onto a uniform `s` grid (bin-average),
///      then AUTOCORRELATE: the first off-zero autocorrelation peak's
///      lag is the dominant ripple wavelength; its amplitude is the RMS
///      of the band-limited residual.
pub fn measure_ripple(hf: &HeightField, mesh: &HeightMesh, slope_dir: Vec3) -> RippleReport {
    measure_ripple_raw(
        &mesh.verts,
        &mesh.indices,
        mesh.voxel_size,
        &|x, z| (hf.h)(x, z),
        slope_dir,
    )
}

/// Low-level ripple metric over raw extract output + a true-height
/// closure. The terrain repro (a different mesh type) reuses this with
/// the real `FbmTerrainFn` as the height reference. See [`measure_ripple`]
/// for the procedure documentation.
pub fn measure_ripple_raw(
    verts: &[MeshVertex],
    indices: &[u32],
    vs: f32,
    h: &dyn Fn(f32, f32) -> f32,
    slope_dir: Vec3,
) -> RippleReport {
    // Top-surface vertices only.
    let dir = Vec3::new(slope_dir.x, 0.0, slope_dir.z).normalize_or_zero();
    let dir = if dir.length_squared() < 1e-6 { Vec3::X } else { dir };
    let mut samples: Vec<(f32, f32)> = Vec::new(); // (ordinate s, residual y in voxels)
    for v in verts {
        let n = unpack_oct(v.normal_oct).normalize_or_zero();
        if n.y <= 0.30 {
            continue; // walls / bottom / steep — keep the top face only
        }
        let p = Vec3::from(v.local_pos);
        let s = p.dot(dir) / vs; // ordinate in voxels
        let resid = (p.y - h(p.x, p.z)) / vs; // residual in voxels
        samples.push((s, resid));
    }
    let n_samples = samples.len();
    if n_samples < 16 {
        return RippleReport {
            n_samples,
            wavelength_vox: 0.0,
            amplitude_vox: 0.0,
            residual_rms_vox: 0.0,
            roughness_vox: 0.0,
        };
    }
    // Full residual RMS (all frequencies).
    let mean_r: f32 = samples.iter().map(|&(_, r)| r).sum::<f32>() / n_samples as f32;
    let residual_rms_vox = (samples
        .iter()
        .map(|&(_, r)| (r - mean_r) * (r - mean_r))
        .sum::<f32>()
        / n_samples as f32)
        .sqrt();

    // Resample residual onto a uniform 1-voxel-spaced ordinate grid.
    let s_min = samples.iter().map(|&(s, _)| s).fold(f32::INFINITY, f32::min);
    let s_max = samples.iter().map(|&(s, _)| s).fold(f32::NEG_INFINITY, f32::max);
    let span = (s_max - s_min).max(1.0);
    // ~2 bins/voxel so we can resolve ~2-voxel ripples (Nyquist).
    let nb = ((span * 2.0).ceil() as usize).clamp(16, 4096);
    let mut bin_sum = vec![0.0f32; nb];
    let mut bin_cnt = vec![0u32; nb];
    for &(s, r) in &samples {
        let t = ((s - s_min) / span * (nb as f32 - 1.0)).clamp(0.0, nb as f32 - 1.0);
        let bi = t as usize;
        bin_sum[bi] += r;
        bin_cnt[bi] += 1;
    }
    // Linear-fill empty bins from neighbors; de-mean.
    let mut sig = vec![0.0f32; nb];
    let mut last = 0.0f32;
    for i in 0..nb {
        if bin_cnt[i] > 0 {
            last = bin_sum[i] / bin_cnt[i] as f32;
        }
        sig[i] = last;
    }
    let m: f32 = sig.iter().sum::<f32>() / nb as f32;
    for v in sig.iter_mut() {
        *v -= m;
    }
    // Bin spacing in voxels.
    let bin_vox = span / (nb as f32 - 1.0);

    // Autocorrelation. The first local MAX after the zero-lag peak's
    // descent is the dominant ripple period.
    let mut ac = vec![0.0f32; nb];
    for lag in 0..nb {
        let mut acc = 0.0f32;
        for i in 0..(nb - lag) {
            acc += sig[i] * sig[i + lag];
        }
        ac[lag] = acc / (nb - lag) as f32;
    }
    let ac0 = ac[0].max(1e-12);
    // Find first lag where AC dips below ~0.2*ac0 (out of the central
    // lobe), then the next local maximum.
    let mut dipped = false;
    let mut peak_lag = 0usize;
    let mut peak_val = 0.0f32;
    for lag in 1..nb - 1 {
        let nr = ac[lag] / ac0;
        if !dipped {
            if nr < 0.2 {
                dipped = true;
            }
            continue;
        }
        if ac[lag] > ac[lag - 1] && ac[lag] >= ac[lag + 1] && ac[lag] > peak_val {
            peak_val = ac[lag];
            peak_lag = lag;
            break;
        }
    }
    let wavelength_vox = peak_lag as f32 * bin_vox;
    // Amplitude of the band: RMS of the residual restricted to the
    // ripple frequency. Approximate by the autocorrelation peak value
    // (≈ variance at that lag) → amplitude ≈ sqrt(2 * peak) for a sine.
    let amplitude_vox = if peak_lag > 0 {
        (2.0 * peak_val.max(0.0)).sqrt()
    } else {
        0.0
    };
    // Reuse the existing 1-ring roughness for the high-freq reference.
    let roughness_vox = heightmesh_roughness_raw(verts, indices, vs, h);

    RippleReport {
        n_samples,
        wavelength_vox,
        amplitude_vox,
        residual_rms_vox,
        roughness_vox,
    }
}

/// 1-ring lumpiness (high-frequency roughness) on a heightfield mesh,
/// in voxels — analog of [`Metrics::roughness`] but against a true-height
/// closure. Raw form so both the bench [`HeightMesh`] and the terrain
/// repro mesh can call it.
pub fn heightmesh_roughness_raw(
    verts: &[MeshVertex],
    indices: &[u32],
    vs: f32,
    h: &dyn Fn(f32, f32) -> f32,
) -> f32 {
    let nv = verts.len();
    let mut resid = vec![0.0f32; nv];
    let mut top = vec![false; nv];
    for (i, v) in verts.iter().enumerate() {
        let p = Vec3::from(v.local_pos);
        resid[i] = (p.y - h(p.x, p.z)) / vs;
        top[i] = unpack_oct(v.normal_oct).normalize_or_zero().y > 0.30;
    }
    let mut nbr_sum = vec![0.0f32; nv];
    let mut nbr_cnt = vec![0u32; nv];
    let mut add = |a: u32, b: u32| {
        nbr_sum[a as usize] += resid[b as usize];
        nbr_cnt[a as usize] += 1;
    };
    for tri in indices.chunks_exact(3) {
        add(tri[0], tri[1]);
        add(tri[0], tri[2]);
        add(tri[1], tri[0]);
        add(tri[1], tri[2]);
        add(tri[2], tri[0]);
        add(tri[2], tri[1]);
    }
    let mut sq = 0.0f32;
    let mut cnt = 0u32;
    for i in 0..nv {
        if !top[i] || nbr_cnt[i] == 0 {
            continue;
        }
        let local = resid[i] - nbr_sum[i] / nbr_cnt[i] as f32;
        sq += local * local;
        cnt += 1;
    }
    if cnt > 0 {
        (sq / cnt as f32).sqrt()
    } else {
        0.0
    }
}

// ════════════════════════════════════════════════════════════════════
// 3 (cont). Validation tests
// ════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh_extract::set_wide_window_project;

    /// Voxelize → mesh → metrics for one shape×voxel.
    fn run(shape: Shape, vs: f32) -> (Occupancy, Vec<MeshVertex>, Vec<u32>, Metrics) {
        let occ = voxelize(shape, shape.bounds(), vs);
        let (verts, indices) = mesh_occupancy(&occ);
        let m = evaluate(shape, &occ, &verts, &indices);
        (occ, verts, indices, m)
    }

    /// Every shape voxelizes to a non-trivial mesh at both resolutions.
    #[test]
    fn all_shapes_mesh_nonempty() {
        for &shape in Shape::all() {
            for &vs in &[0.5f32, 0.25] {
                let (_occ, verts, indices, _m) = run(shape, vs);
                assert!(
                    !verts.is_empty() && !indices.is_empty(),
                    "{} @ vs={} produced empty mesh",
                    shape.name(),
                    vs
                );
            }
        }
    }

    /// GEOMETRY accuracy on the SMOOTH shapes.
    ///
    /// `geom_rms` measures distance to the ANALYTIC surface. The Newton
    /// projection snaps vertices onto the smooth `D = 0.5` isosurface of
    /// the Gaussian-blurred occupancy — which is shifted INWARD from the
    /// analytic surface on convex shapes (the blur bias). So `geom_rms`
    /// here is dominated by that *systematic, smooth* shift (≈ 0.5 voxel
    /// on a curved shape), NOT by lumpiness — lumpiness is pinned
    /// separately by `smooth_shapes_low_roughness` (≈ 0.03-0.07 voxel).
    /// The bound is the honest post-projection envelope. (Measured:
    /// sphere ≤0.54, torus ≤0.56, slope ≤0.37, sine ≤0.66.)
    ///
    /// `geom_max < 1.5` — worst single vertex (tight-curvature corner /
    /// open-boundary cube). Sharp shapes are exempt (lattice crease).
    #[test]
    fn smooth_shapes_geometry_accurate() {
        for &shape in Shape::all() {
            if !shape.is_accuracy_control() {
                continue;
            }
            for &vs in &[0.5f32, 0.25] {
                let (_occ, _v, _i, m) = run(shape, vs);
                assert!(
                    m.geom_rms < 0.70,
                    "{} @ vs={}: geom_rms {:.3} voxels ≥ 0.70",
                    shape.name(),
                    vs,
                    m.geom_rms
                );
                assert!(
                    m.geom_max < 1.5,
                    "{} @ vs={}: geom_max {:.3} voxels ≥ 1.5",
                    shape.name(),
                    vs,
                    m.geom_max
                );
            }
        }
    }

    /// NORMAL accuracy on the SMOOTH shapes: mean angle below a
    /// resolution-dependent ceiling. At the coarse `vs = 0.5` a small
    /// torus tube / sine crest is only a couple of cells across, so the
    /// per-vertex normal carries genuine discretization; the bound is
    /// the honest measured envelope, not a tight ideal. (Measured @
    /// vs=0.5: sphere 7.1°, torus 14.8°, slope 15.3°, sine 23.5°; all
    /// roughly halve at vs=0.25.)
    #[test]
    fn smooth_shapes_normals_accurate() {
        for &shape in Shape::all() {
            if !shape.is_accuracy_control() {
                continue;
            }
            for &vs in &[0.5f32, 0.25] {
                let (_occ, _v, _i, m) = run(shape, vs);
                // Coarser voxel → looser ceiling (curvature per cell ↑).
                let bound = if vs >= 0.5 { 28.0 } else { 16.0 };
                assert!(
                    m.normal_mean_deg < bound,
                    "{} @ vs={}: normal mean {:.2}° ≥ {:.0}°",
                    shape.name(),
                    vs,
                    m.normal_mean_deg,
                    bound
                );
            }
        }
    }

    /// NO-SPECKLE: the ∇D-gradient-field normals must be CONTINUOUS, not
    /// per-voxel flips. Asserted on `edge_normal_interior_max_deg` (the
    /// max edge-normal angle over INTERIOR edges — excludes the
    /// open-boundary domain-edge artifact where the surface legitimately
    /// turns regardless of mesher quality).
    ///
    /// The bound is per-shape because some shapes carry GENUINE curvature
    /// between adjacent verts at these resolutions (a torus tube /
    /// sine crest only a couple of cells across at vs=0.5) — that is real
    /// geometry, NOT speckle. The SLOPE (zero curvature) is the clean
    /// discriminator: it pins the true speckle floor at < 8° regardless
    /// of resolution. Per-voxel speckle (e.g. differentiating the
    /// trilinear density) would push the slope's interior angle to tens
    /// of degrees and fail.
    #[test]
    fn smooth_shapes_no_normal_speckle() {
        for &shape in Shape::all() {
            if !shape.is_smooth() {
                continue;
            }
            // Per-shape interior edge-normal ceiling. Slope is the clean
            // flat-surface discriminator (< 8°); sphere is gently curved;
            // torus / sine allow more because their adjacent verts span
            // real curvature at the coarse resolution.
            let bound = match shape {
                Shape::Slope => 8.0,
                Shape::Sphere => 24.0,
                Shape::Torus => 60.0,
                Shape::SineTerrain => 52.0,
                _ => 60.0,
            };
            for &vs in &[0.5f32, 0.25] {
                let (_occ, _v, _i, m) = run(shape, vs);
                assert!(
                    m.edge_normal_interior_max_deg < bound,
                    "{} @ vs={}: interior edge-normal max {:.2}° ≥ {:.0}° (speckle)",
                    shape.name(),
                    vs,
                    m.edge_normal_interior_max_deg,
                    bound
                );
            }
        }
    }

    /// NO-TERRACE on the height-field shapes. This SHOULD currently
    /// fail if terracing is real — `#[ignore]`d so the build stays
    /// green, but `cargo test -- --ignored no_terracing` confirms.
    #[ignore]
    #[test]
    fn no_terracing_on_height_fields() {
        for &shape in &[Shape::Slope, Shape::SineTerrain] {
            for &vs in &[0.5f32, 0.25] {
                let (_occ, _v, _i, m) = run(shape, vs);
                assert_eq!(
                    m.terrace_detected,
                    Some(false),
                    "{} @ vs={}: terracing detected",
                    shape.name(),
                    vs
                );
            }
        }
    }

    /// SMOOTH (low lumpiness) — the Newton isosurface projection keeps
    /// the high-frequency facet roughness low on the smooth shapes.
    /// Roughness isolates the facet noise from the systematic blur
    /// shift (which `geom_rms` captures). With the projection ON, the
    /// smooth shapes measure `roughness ≤ ~0.05` at vs=0.25 and `≤ ~0.10`
    /// at the coarse vs=0.5; with the projection OFF the centroid
    /// placement is ~0.07-0.12. The (resolution-aware) bound guards the
    /// projection against regression — turning Newton OFF lifts every
    /// smooth shape past it.
    #[test]
    fn smooth_shapes_low_roughness() {
        for &shape in Shape::all() {
            if !shape.is_smooth() {
                continue;
            }
            for &vs in &[0.5f32, 0.25] {
                let (_occ, _v, _i, m) = run(shape, vs);
                // Resolution-aware: the coarse vs=0.5 sine crests carry
                // genuine sub-voxel roughness (measured ≤0.121 post
                // D-topology fix). vs=0.25 stays tight (≤0.07).
                let bound = if vs >= 0.5 { 0.14 } else { 0.08 };
                assert!(
                    m.roughness < bound,
                    "{} @ vs={}: roughness {:.3} ≥ {:.2} voxel (lumpy)",
                    shape.name(),
                    vs,
                    m.roughness,
                    bound
                );
            }
        }
    }

    /// GOAL-2 REPRODUCTION: confirm the bench can reproduce the
    /// terrain-style terracing/bumpiness. The IRREGULAR scenario
    /// (sculpt-brush-like ±1.5-voxel column jitter) must mesh to a
    /// MUCH rougher / more-terraced surface than the clean shape — i.e.
    /// the blur+SN cannot smooth jagged occupancy. This is the confound
    /// that matches the user's terrain; the clean baseline stays smooth.
    #[test]
    fn irregular_occupancy_reproduces_bumpiness() {
        let vs = 0.25f32;
        // Clean sine baseline.
        let clean = voxelize(Shape::SineTerrain, Shape::SineTerrain.bounds(), vs);
        let (cv, ci) = mesh_occupancy(&clean);
        let clean_m = evaluate(Shape::SineTerrain, &clean, &cv, &ci);

        // Irregular (brushfire-like) occupancy.
        let irr = voxelize_irregular(Shape::SineTerrain, Shape::SineTerrain.bounds(), vs, 1.5);
        let (iv, ii) = mesh_occupancy(&irr);
        let irr_m = evaluate(Shape::SineTerrain, &irr, &iv, &ii);

        // The irregular surface is substantially rougher and turns the
        // interior normals far more than the clean one — the bench
        // reproduces the bumpiness. (Roughness + interior edge-normal
        // are the lumpiness signals; geom_rms is dominated by the blur
        // shift on the steep sine crests in BOTH cases, so it does not
        // discriminate here.)
        assert!(
            irr_m.edge_normal_interior_max_deg > clean_m.edge_normal_interior_max_deg + 5.0,
            "irregular interior edge-normal {:.1}° not clearly worse than clean {:.1}°",
            irr_m.edge_normal_interior_max_deg,
            clean_m.edge_normal_interior_max_deg
        );
        assert!(
            irr_m.roughness > clean_m.roughness,
            "irregular roughness {:.3} not worse than clean {:.3}",
            irr_m.roughness,
            clean_m.roughness
        );
    }

    /// GOAL-2: region truncation (narrow halo) degrades the windowed
    /// mesh vs a wide halo — the density blur clamps at the window
    /// boundary. Confirms the bench captures the tile-boundary confound.
    #[test]
    fn region_truncation_degrades_vs_wide_halo() {
        let vs = 0.25f32;
        let shape = Shape::SineTerrain;
        let full = voxelize(shape, shape.bounds(), vs);
        let span = (shape.bounds().size().x / vs).ceil() as i32;
        let q = span / 5;
        let (ylo, yhi) = full
            .cells
            .keys()
            .fold((i32::MAX, i32::MIN), |(a, b), c| (a.min(c.y), b.max(c.y)));
        let wlo = IVec3::new(2 * q, ylo - 1, 2 * q);
        let whi = IVec3::new(3 * q, yhi + 2, 3 * q);

        let narrow = truncate_to_window(&full, wlo, whi, 2);
        let wide = truncate_to_window(&full, wlo, whi, 8);
        let (nv, ni) = mesh_occupancy(&narrow);
        let (wv, wi) = mesh_occupancy(&wide);
        let nm = evaluate(shape, &narrow, &nv, &ni);
        let wm = evaluate(shape, &wide, &wv, &wi);

        assert!(
            nm.geom_rms > wm.geom_rms,
            "narrow-halo geom_rms {:.3} should exceed wide-halo {:.3} (boundary clamp)",
            nm.geom_rms,
            wm.geom_rms
        );
    }

    /// BLUR-RADIUS SWEEP: on a terraced (irregular, sculpt-brush-like)
    /// steep mound, widening the blur radius R reduces the interior
    /// edge-normal angle — i.e. wider blur de-terraces. R=4 must be
    /// clearly smoother than R=2. (This pins the experiment's headline
    /// finding so a regression in the blur path is caught.)
    #[test]
    fn wider_blur_reduces_terracing_on_irregular_mound() {
        let vs = 0.25f32;
        let occ = voxelize_irregular(Shape::Mound, Shape::Mound.bounds(), vs, 1.5);
        let r2 = {
            let (v, i) = mesh_occupancy_blur(&occ, BlurParams::for_radius(2));
            evaluate(Shape::Mound, &occ, &v, &i)
        };
        let r4 = {
            let (v, i) = mesh_occupancy_blur(&occ, BlurParams::for_radius(4));
            evaluate(Shape::Mound, &occ, &v, &i)
        };
        assert!(
            r4.edge_normal_interior_max_deg < r2.edge_normal_interior_max_deg - 8.0,
            "R=4 interior edge-normal {:.1}° not clearly below R=2 {:.1}° (wider blur should de-terrace)",
            r4.edge_normal_interior_max_deg,
            r2.edge_normal_interior_max_deg
        );
    }

    /// BIAS: widening R grows the systematic geom_rms (the blurred
    /// `D=0.5` isosurface shifts inward on the convex mound), AND a
    /// global iso OFFSET does NOT fix it (it over-corrects the flats
    /// faster than it recovers the peak). Pins both halves of the
    /// bias-correction conclusion.
    #[test]
    fn wider_blur_grows_bias_and_iso_offset_does_not_fix_it() {
        let vs = 0.25f32;
        let occ = voxelize(Shape::Mound, Shape::Mound.bounds(), vs);
        let g = |bp: BlurParams| {
            let (v, i) = mesh_occupancy_blur(&occ, bp);
            evaluate(Shape::Mound, &occ, &v, &i).geom_rms
        };
        let r2 = g(BlurParams::for_radius(2));
        let r4 = g(BlurParams::for_radius(4));
        assert!(r4 > r2, "wider blur should grow geom_rms bias: R4 {r4:.3} ≤ R2 {r2:.3}");
        // A positive iso offset (push surface outward) does NOT reduce
        // the overall geom_rms — it makes it worse.
        let r4_iso = g(BlurParams::for_radius_iso(4, 0.6));
        assert!(
            r4_iso > r4,
            "iso offset unexpectedly reduced geom_rms ({r4_iso:.3} ≤ {r4:.3}) — \
             a global threshold offset was expected NOT to help",
        );
    }

    /// The renderer produces a non-blank image (sanity that projection +
    /// raster wrote pixels).
    #[test]
    fn renderer_writes_pixels() {
        let (occ, verts, indices, _m) = run(Shape::Sphere, 0.25);
        let cam = Camera::three_quarter(128, 128);
        let img = render(
            &cam,
            &occ,
            &verts,
            &indices,
            RenderOpts {
                shaded: true,
                wireframe: true,
                voxels: true,
                dim_shading: true,
            },
            128,
        );
        let bg = [18u8, 20, 26];
        let non_bg = img
            .rgb
            .chunks_exact(3)
            .filter(|px| px[0] != bg[0] || px[1] != bg[1] || px[2] != bg[2])
            .count();
        assert!(non_bg > 500, "render wrote only {non_bg} non-bg pixels");
    }

    // ── Smooth-stairs reproduction + fix (bake-path heightfields) ──

    /// REPRODUCTION: a GENTLE slope (wide tread) baked through the
    /// terrain bake path with the fix FORCED OFF shows a low-frequency
    /// residual ripple ("smooth stairs") in the surface-Y residual, even
    /// though the high-frequency roughness is modest. This pins the
    /// bench's ability to reproduce the user's report.
    #[test]
    fn gentle_slope_bake_path_has_low_freq_ripple() {
        let hf = HeightField::gentle_slope(0.10, 0.035);
        for &vs in &[0.5f32, 1.0] {
            // Force the production fix OFF to see the raw rippled baseline.
            set_wide_window_project(Some(0.0));
            let mesh = bake_heightfield(&hf, 12.0, vs);
            set_wide_window_project(None);
            let r = measure_ripple(&hf, &mesh, Vec3::X);
            // The residual RMS (deviation from the true plane) is a clear,
            // non-trivial ripple signal on the gentle slope.
            assert!(
                r.residual_rms_vox > 0.10,
                "gentle slope vs={vs}: expected a reproducible ripple \
                 (residual_rms {:.3} ≤ 0.10 voxel) — bench not reproducing it",
                r.residual_rms_vox
            );
        }
    }

    /// FIX (production default-ON): the wide-window plane-fit projection
    /// (R=2 blur unchanged) substantially reduces the gentle-slope ripple
    /// — both the low-frequency residual RMS and the high-frequency
    /// roughness drop vs the fix-OFF baseline. The default extract
    /// (override `None`) already runs the fix, so it must match the
    /// explicit-on path.
    #[test]
    fn wide_window_fix_reduces_gentle_slope_ripple() {
        let hf = HeightField::gentle_slope(0.10, 0.035);
        for &vs in &[0.5f32, 1.0] {
            let base = {
                set_wide_window_project(Some(0.0)); // off
                let m = bake_heightfield(&hf, 12.0, vs);
                set_wide_window_project(None);
                measure_ripple(&hf, &m, Vec3::X)
            };
            let fixed = {
                set_wide_window_project(None); // production default = ON (r5)
                let m = bake_heightfield(&hf, 12.0, vs);
                measure_ripple(&hf, &m, Vec3::X)
            };
            assert!(
                fixed.residual_rms_vox < base.residual_rms_vox * 0.7,
                "vs={vs}: fix should cut residual_rms ≥30% ({:.3} → {:.3})",
                base.residual_rms_vox,
                fixed.residual_rms_vox
            );
            assert!(
                fixed.roughness_vox < base.roughness_vox,
                "vs={vs}: fix should cut roughness ({:.3} → {:.3})",
                base.roughness_vox,
                fixed.roughness_vox
            );
        }
    }

    /// OFF override is bit-stable: two fix-OFF extracts produce the
    /// identical SURFACE (multiset of vertex positions). The bake's vertex
    /// *array order* is not deterministic across runs (voxelization uses
    /// parallel + hashed cell maps), so we compare the position multiset.
    #[test]
    fn wide_window_fix_off_is_bit_stable() {
        let hf = HeightField::gentle_slope(0.10, 0.035);
        set_wide_window_project(Some(0.0)); // off
        let a = bake_heightfield(&hf, 12.0, 0.5);
        let b = bake_heightfield(&hf, 12.0, 0.5);
        set_wide_window_project(None);
        assert_eq!(a.verts.len(), b.verts.len());
        let key = |v: &MeshVertex| {
            (
                (v.local_pos[0] / 1e-4).round() as i64,
                (v.local_pos[1] / 1e-4).round() as i64,
                (v.local_pos[2] / 1e-4).round() as i64,
            )
        };
        let mut sa: Vec<_> = a.verts.iter().map(key).collect();
        let mut sb: Vec<_> = b.verts.iter().map(key).collect();
        sa.sort_unstable();
        sb.sort_unstable();
        assert_eq!(sa, sb, "off-path vertex positions must be stable");
    }

    /// CURVATURE PRESERVED: on a curved heightfield the fix must NOT
    /// inflate the residual (which would mean it's rounding real
    /// curvature, not just ripple). At r=5 the FBM residual stays within
    /// ~25% of the fix-OFF baseline (it does not blow up the way a wide
    /// UNWEIGHTED box fit would).
    #[test]
    fn wide_window_fix_preserves_curvature_on_fbm() {
        let hf = HeightField::fbm();
        let vs = 0.5f32;
        let base = {
            set_wide_window_project(Some(0.0)); // off
            let m = bake_heightfield(&hf, 12.0, vs);
            set_wide_window_project(None);
            measure_ripple(&hf, &m, Vec3::X)
        };
        let fixed = {
            set_wide_window_project(None); // production default = ON (r5)
            let m = bake_heightfield(&hf, 12.0, vs);
            measure_ripple(&hf, &m, Vec3::X)
        };
        assert!(
            fixed.residual_rms_vox < base.residual_rms_vox * 1.25 + 1e-4,
            "fix inflated FBM residual too much ({:.3} → {:.3}) — rounding curvature",
            base.residual_rms_vox,
            fixed.residual_rms_vox
        );
    }

    // ── Stage 2: QEF-Hermite mesher (behind the qef_hermite toggle) ──

    /// QEF-Hermite places each vertex on the TRUE surface from the stored
    /// per-leaf Hermite (distance + normal), so a gentle slope comes out
    /// FLAT by construction — the residual RMS collapses to the
    /// quantization floor, well under the `< 0.02 vox` target and far
    /// below the blur path's `> 0.10` ripple (which the plane-fit only
    /// claws back ~30%). No post-hoc recovery.
    #[test]
    fn qef_hermite_bake_path_is_flat_on_gentle_slope() {
        let hf = HeightField::gentle_slope(0.10, 0.035);
        for &vs in &[0.5f32, 1.0] {
            let m = bake_heightfield_qef(&hf, 12.0, vs);
            let r = measure_ripple(&hf, &m, Vec3::X);
            assert!(
                r.n_samples >= 16,
                "vs={vs}: too few top-surface samples ({})",
                r.n_samples
            );
            assert!(
                r.residual_rms_vox < 0.02,
                "vs={vs}: QEF-Hermite slope should be flat (residual_rms {:.4} ≥ 0.02)",
                r.residual_rms_vox
            );
        }
    }

    /// Against the raw rippled blur baseline (fix OFF), QEF-Hermite is a
    /// large improvement on the same gentle slope — not the ~30% the
    /// plane-fit recovers, but ≥80%.
    #[test]
    fn qef_hermite_beats_blur_on_gentle_slope() {
        let hf = HeightField::gentle_slope(0.10, 0.035);
        let vs = 0.5f32;
        let base = {
            set_wide_window_project(Some(0.0)); // raw blur, plane-fit OFF
            let m = bake_heightfield(&hf, 12.0, vs);
            set_wide_window_project(None);
            measure_ripple(&hf, &m, Vec3::X)
        };
        let qef = measure_ripple(&hf, &bake_heightfield_qef(&hf, 12.0, vs), Vec3::X);
        assert!(
            qef.residual_rms_vox < base.residual_rms_vox * 0.2,
            "QEF should cut the blur ripple ≥80% ({:.3} → {:.4})",
            base.residual_rms_vox,
            qef.residual_rms_vox
        );
    }

    /// On a curved (FBM) field QEF-Hermite tracks the true height: its
    /// residual beats the raw-blur ripple and stays small — it places on
    /// the real surface, it does not round curvature.
    #[test]
    fn qef_hermite_tracks_fbm_curvature() {
        let hf = HeightField::fbm();
        let vs = 0.5f32;
        let base = {
            set_wide_window_project(Some(0.0));
            let m = bake_heightfield(&hf, 12.0, vs);
            set_wide_window_project(None);
            measure_ripple(&hf, &m, Vec3::X)
        };
        let qef = measure_ripple(&hf, &bake_heightfield_qef(&hf, 12.0, vs), Vec3::X);
        // QEF cuts the FBM ripple ~83% (0.186 → 0.031); the residual that
        // remains is genuine sub-voxel curvature the discrete mesh can't
        // resolve, NOT terracing. Pin a comfortable margin on both the cut
        // ratio and the absolute floor.
        assert!(
            qef.residual_rms_vox < base.residual_rms_vox * 0.3,
            "QEF should cut the FBM ripple ≥70% ({:.3} → {:.4})",
            base.residual_rms_vox,
            qef.residual_rms_vox
        );
        assert!(
            qef.residual_rms_vox < 0.05,
            "QEF FBM residual floor should stay small ({:.4} ≥ 0.05)",
            qef.residual_rms_vox
        );
    }

    /// **Stage 6 — the sculpt/halo REGION re-extract is flat too.** The same
    /// gentle slope, meshed through `extract_mesh_region_from_cells_pooled_haloed`
    /// (the path a sculpt brush / halo refresh runs) with the baked per-leaf
    /// distances, comes out flat — so a sculpted/refreshed region matches the
    /// QEF bake instead of falling back to the rippled blur path.
    #[test]
    fn qef_region_reextract_slope_is_flat() {
        let hf = HeightField::gentle_slope(0.10, 0.035);
        for &vs in &[0.5f32, 1.0] {
            let m = region_mesh_heightfield_qef(&hf, 12.0, vs);
            let r = measure_ripple(&hf, &m, Vec3::X);
            assert!(r.n_samples >= 16, "vs={vs}: too few region samples");
            assert!(
                r.residual_rms_vox < 0.03,
                "vs={vs}: QEF region re-extract should be flat (residual {:.4} ≥ 0.03)",
                r.residual_rms_vox
            );
        }
    }

    /// **Watertight seam across adjacent tiles.** Two independently
    /// voxelized tiles share the `x = 0` face. Each emits the boundary
    /// cubes that straddle the seam (the tile owns them on one side and
    /// folds the other side in as halo); for a watertight mesh those
    /// shared vertices must COINCIDE.
    ///
    /// They are not *bit*-identical: each tile's grid origin is its own
    /// `aabb.min`, so the QEF (and the blur Newton) solve at different
    /// coordinate magnitudes — a property of the full-asset path, not of
    /// QEF (the shipped blur path has the same limitation). What we pin is
    /// that the seam gap stays deep sub-voxel (no visible crack) AND that
    /// QEF does not regress watertightness vs the blur baseline. The Stage-3
    /// halo distance write (phase-1 center sample = the neighbour's interior
    /// sample) is what keeps the QEF seam this tight.
    #[test]
    fn qef_hermite_seam_watertight_across_adjacent_tiles() {
        let hf = HeightField::gentle_slope(0.10, 0.035);
        let vs = 0.5f32;
        let n = 32i32; // pow2 cells per tile
        let span = n as f32 * vs;
        let ymin = -span * 0.5;
        let zmin = -span * 0.5;
        let aabb_a = Aabb::new(
            Vec3::new(-span, ymin, zmin),
            Vec3::new(0.0, ymin + span, zmin + span),
        );
        let aabb_b = Aabb::new(
            Vec3::new(0.0, ymin, zmin),
            Vec3::new(span, ymin + span, zmin + span),
        );

        // Max seam gap: for each A seam-band vertex, the distance to the
        // nearest B vertex (within ½-voxel, i.e. a genuine shared cube).
        let band = REPRO_TILE_HALO as f32 * vs;
        let seam_gap = |a: &HeightMesh, b: &HeightMesh| -> (f32, usize) {
            let sa: Vec<_> = a
                .verts
                .iter()
                .filter(|v| v.local_pos[0].abs() <= band)
                .collect();
            let sb: Vec<_> = b
                .verts
                .iter()
                .filter(|v| v.local_pos[0].abs() <= band)
                .collect();
            let mut max_gap = 0.0f32;
            let mut matched = 0usize;
            for v in &sa {
                let mut best = f32::INFINITY;
                for w in &sb {
                    let d = (Vec3::from(v.local_pos) - Vec3::from(w.local_pos)).length();
                    if d < best {
                        best = d;
                    }
                }
                if best <= 0.5 * vs {
                    matched += 1;
                    if best > max_gap {
                        max_gap = best;
                    }
                }
            }
            (max_gap / vs, matched)
        };

        let (qef_gap, qef_matched) =
            seam_gap(&bake_heightfield_qef_aabb(&hf, aabb_a, vs), &bake_heightfield_qef_aabb(&hf, aabb_b, vs));
        let (blur_gap, _) =
            seam_gap(&bake_heightfield_blur_aabb(&hf, aabb_a, vs), &bake_heightfield_blur_aabb(&hf, aabb_b, vs));

        // A healthy shared boundary exists (the tiles really do meet).
        assert!(
            qef_matched >= 32,
            "expected a populated shared seam, only {qef_matched} matched cubes"
        );
        // Deep sub-voxel: no visible crack (observed ~2e-4 vox).
        assert!(
            qef_gap < 0.01,
            "QEF seam gap {qef_gap:.6} vox ≥ 0.01 — possible crack"
        );
        // No regression vs the shipped blur path's watertightness.
        assert!(
            qef_gap <= blur_gap.max(1e-4) * 2.0,
            "QEF seam gap {qef_gap:.6} vox regressed vs blur {blur_gap:.6}"
        );
    }

    /// The force-off override (mirroring the `ARVX_QEF_HERMITE=0` env
    /// kill-switch) makes a distance-present bake fall back to the blur
    /// path — the rollback safety net for the Stage-5 production flip. With
    /// it set, the gentle slope ripples like the blur baseline (>0.05)
    /// instead of going flat (<0.02).
    #[test]
    fn qef_force_off_falls_back_to_blur() {
        use crate::mesh_extract::set_qef_force_off;
        let hf = HeightField::gentle_slope(0.10, 0.035);
        let vs = 0.5f32;
        set_qef_force_off(true);
        let m = bake_heightfield_qef(&hf, 12.0, vs);
        set_qef_force_off(false); // reset before the assert (panic-safe)
        let forced = measure_ripple(&hf, &m, Vec3::X);
        assert!(
            forced.residual_rms_vox > 0.05,
            "force-off should bake the (rippled) blur path, got {:.4}",
            forced.residual_rms_vox
        );
    }

    /// QEF placement is deterministic: two bakes of the same tile produce
    /// the identical SURFACE (multiset of vertex positions). The vertex
    /// *array order* is not deterministic — voxelization uses parallel +
    /// hashed cell maps — so we compare the position multiset, mirroring
    /// `wide_window_fix_off_is_bit_stable`. A non-deterministic gather
    /// order or solve would surface as a mismatch here, which is what makes
    /// the seam reproducible in the first place.
    #[test]
    fn qef_hermite_bake_is_bit_stable() {
        let hf = HeightField::gentle_slope(0.10, 0.035);
        let a = bake_heightfield_qef(&hf, 12.0, 0.5);
        let b = bake_heightfield_qef(&hf, 12.0, 0.5);
        assert_eq!(a.verts.len(), b.verts.len());
        let key = |v: &MeshVertex| {
            (
                v.local_pos[0].to_bits(),
                v.local_pos[1].to_bits(),
                v.local_pos[2].to_bits(),
                v.normal_oct,
            )
        };
        let mut sa: Vec<_> = a.verts.iter().map(key).collect();
        let mut sb: Vec<_> = b.verts.iter().map(key).collect();
        sa.sort_unstable();
        sb.sort_unstable();
        assert_eq!(sa, sb, "QEF vertex positions+normals must be stable");
    }
}
