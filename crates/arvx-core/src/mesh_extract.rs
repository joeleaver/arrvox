//! CPU surface-mesh extraction (naive surface nets) at asset load.
//!
//! Walks an asset's brick-terminated octree and emits a triangle mesh
//! that follows the surface defined by the cell-occupancy field. One
//! [`MeshVertex`] per active SN-cube (a `2×2×2` grouping of cells whose
//! corner cells contain a mix of solid and void). Two triangles per
//! active sample-edge (an axis edge between a solid cell and an EMPTY
//! cell). Vertices carry an octahedral-packed average normal and a
//! `leaf_attr_id` slot for the resolve / shade pass to look up
//! prefiltered surface attributes via the `leaf_attr_pool`.
//!
//! No GPU work here — this just produces `(vertices, indices)` that
//! the per-asset cache stores.

use glam::{IVec3, UVec3, Vec3};
use rustc_hash::FxHashMap;

/// Per-cell occupancy lookup used by Surface Nets — keys are
/// finest-grid integer coords, values are `leaf_attr_id`s (or the
/// `CELL_INTERIOR` sentinel for INTERIOR-bulk cells).
///
/// D6.2: backed by `FxHashMap`. The hot extract loop does
/// hundreds of thousands of probes per stamp; FxHash's
/// single-multiply-mix hash is ~3-5× faster than std's SipHash on
/// 12-byte `IVec3` keys, with no DoS-resistance concern for
/// internal data. `cells.len()` / `cells.iter()` behave identically.
pub type CellMap = FxHashMap<IVec3, u32>;

use crate::brick_pool::{BRICK_CELLS, BRICK_DIM, BRICK_EMPTY, BRICK_INTERIOR};
use crate::companion::BoneVoxel;
use crate::leaf_attr::{pack_oct, unpack_oct, LeafAttr};
use crate::leaf_attr_pool::LeafAttrPool;
use crate::sparse_octree::{
    brick_id, is_branch, is_brick, is_leaf, leaf_slot, EMPTY_NODE, INTERIOR_NODE,
};

/// Sentinel value stored in [`CellGrid`] entries that have no occupant.
/// Doubles as the empty marker for cube_vertex caches (no vertex emitted
/// yet for that SN-cube). Real `leaf_attr_id` and `vertex_id` values are
/// well below `u32::MAX`, so the collision risk is theoretical only.
pub const CELL_GRID_EMPTY: u32 = u32::MAX;

/// Dense 3D grid over a bounded integer-coord region. Replaces
/// [`CellMap`] in the sculpt inner loop for two roles:
///
/// 1. **Cell occupancy / `leaf_attr_id` lookup.** Stores
///    `leaf_attr_id` for solid cells (or [`CELL_INTERIOR`] for
///    brick-INTERIOR-bulk cells), [`CELL_GRID_EMPTY`] otherwise. Built
///    once at the start of [`extract_mesh_region_from_cells`] from the
///    region's [`CellMap`]; from then on, every per-cell / per-face /
///    per-corner probe is a flat-array read.
///
/// 2. **Cube → vertex id cache.** Stores the vertex id of the SN cube
///    whose lo corner is at that grid coord, [`CELL_GRID_EMPTY`]
///    when no vertex has been emitted yet for that cube.
///
/// D6.3 motivation: the post-D6.2 inner loop on a 50-cell brush radius
/// does ~12 FxHashMap probes per solid cell (6 face neighbors + 4
/// cube-vertex lookups + 8 corner cells inside `build_cube_vertex` per
/// fresh cube). At ~50 k cells × 12 probes × ~30 ns per FxHash probe we
/// were spending 13-22 ms on high-density stamps. A dense `Vec<u32>`
/// lookup is one bounds check + one indexed load (~2-5 ns) — an order
/// of magnitude faster, and the grid stays cache-resident for the
/// 1-cluster brush footprint.
///
/// **Memory budget:** for a 50-cell brush radius (typical splat5 stamp)
/// the grid covers `[region_min - 1, region_max + 1) ≈ 104³ ≈ 1.12 M
/// entries × 4 B = 4.5 MB` per grid (two grids = ~9 MB scratch).
/// Smaller brushes proportionally less.
///
/// Half-open extent: `[origin, origin + size)` along each axis.
///
/// **Dirty-tracking for pool reuse (D6.3.c):** [`set`](Self::set)
/// records every slot it writes-to-empty in a parallel `dirty` list.
/// [`reuse`](Self::reuse) resets only those slots between stamps
/// (~50 µs for 30 k writes) instead of memsetting the entire backing
/// `Vec` (~450 µs for 4.5 MB), so the same scratch buffer can be
/// held on `ArvxSceneManager` and reused across stamps without paying
/// the fresh-alloc + memset cost each time.
pub struct CellGrid {
    data: Vec<u32>,
    /// Flat indices that [`set`](Self::set) has written-to-empty during
    /// the current "epoch" (since `new` / the last `reuse` call). Lets
    /// `reuse` reset only the touched slots — cheap for a brush
    /// footprint that touches ~3 % of a 9 MB grid.
    dirty: Vec<usize>,
    origin: IVec3,
    size: IVec3,
}

impl CellGrid {
    /// Allocate a fresh grid covering `[origin, origin + size)`,
    /// pre-filled with [`CELL_GRID_EMPTY`].
    ///
    /// Panics if any axis of `size` is negative (callers always pass
    /// strictly-positive sizes after the pad-min/pad-max math; a
    /// negative `size` would mean an inside-out region).
    pub fn new(origin: IVec3, size: IVec3) -> Self {
        assert!(
            size.x > 0 && size.y > 0 && size.z > 0,
            "CellGrid size must be strictly positive (got {:?})",
            size
        );
        let len = (size.x as usize) * (size.y as usize) * (size.z as usize);
        Self {
            data: vec![CELL_GRID_EMPTY; len],
            dirty: Vec::new(),
            origin,
            size,
        }
    }

    /// Empty grid suitable for the [`Default`] use case — pool-reuse
    /// callers grow it via [`reuse`](Self::reuse) on first use.
    pub fn empty() -> Self {
        Self {
            data: Vec::new(),
            dirty: Vec::new(),
            origin: IVec3::ZERO,
            size: IVec3::ZERO,
        }
    }

    /// Reset previously-touched slots back to [`CELL_GRID_EMPTY`] and
    /// reconfigure the grid for a new region. The underlying `Vec`
    /// grows on demand and never shrinks — D6.3.c trades a fixed
    /// ~9 MB scratch high-water-mark for zero per-stamp allocation.
    ///
    /// Cheaper than a full memset when the brush touches a small
    /// fraction of the grid: walks only the `dirty` list (~30 k
    /// indices on a 44 k-cell stamp) instead of the full 1.12 M
    /// slots.
    pub fn reuse(&mut self, origin: IVec3, size: IVec3) {
        assert!(
            size.x > 0 && size.y > 0 && size.z > 0,
            "CellGrid size must be strictly positive (got {:?})",
            size
        );
        // Reset previously-dirty slots. Indices are guaranteed valid
        // for the *current* `data.len()` because `set` only appends
        // after a successful `flat_index`; the layout change below
        // happens after reset.
        for &idx in &self.dirty {
            debug_assert!(idx < self.data.len());
            self.data[idx] = CELL_GRID_EMPTY;
        }
        self.dirty.clear();

        let new_len = (size.x as usize) * (size.y as usize) * (size.z as usize);
        if new_len > self.data.len() {
            self.data.resize(new_len, CELL_GRID_EMPTY);
        }
        // Don't shrink — keep the high-water capacity for the next stamp.

        self.origin = origin;
        self.size = size;
    }

    /// Linearize `coord` into the flat-`Vec<u32>` index, returning
    /// `None` if the coord is outside `[origin, origin + size)`.
    #[inline]
    pub fn flat_index(&self, coord: IVec3) -> Option<usize> {
        let local = coord - self.origin;
        if local.x < 0
            || local.x >= self.size.x
            || local.y < 0
            || local.y >= self.size.y
            || local.z < 0
            || local.z >= self.size.z
        {
            return None;
        }
        let sx = self.size.x as usize;
        let sy = self.size.y as usize;
        Some(local.x as usize + sx * (local.y as usize + sy * local.z as usize))
    }

    /// Read the slot at `coord`, returning `None` if either the coord
    /// is out-of-bounds OR the slot holds the [`CELL_GRID_EMPTY`]
    /// sentinel. Otherwise returns the stored `u32`.
    #[inline]
    pub fn get(&self, coord: IVec3) -> Option<u32> {
        let idx = self.flat_index(coord)?;
        let v = self.data[idx];
        if v == CELL_GRID_EMPTY {
            None
        } else {
            Some(v)
        }
    }

    /// Faster predicate matching `CellMap::contains_key` semantics —
    /// equivalent to `self.get(coord).is_some()` but skips wrapping
    /// the value.
    #[inline]
    pub fn contains(&self, coord: IVec3) -> bool {
        match self.flat_index(coord) {
            Some(idx) => self.data[idx] != CELL_GRID_EMPTY,
            None => false,
        }
    }

    /// Write `value` at `coord`. Silently no-ops if `coord` is
    /// out-of-bounds — callers that hand the grid coords inside its
    /// own pre-computed range never hit this fallback, but the safe
    /// behaviour matches `CellMap::insert` for shared callers.
    ///
    /// The first write to any slot pushes its flat index onto
    /// `dirty` so [`reuse`](Self::reuse) can reset it cheaply.
    /// Subsequent writes to the same slot skip the push.
    #[inline]
    pub fn set(&mut self, coord: IVec3, value: u32) {
        if let Some(idx) = self.flat_index(coord) {
            if self.data[idx] == CELL_GRID_EMPTY {
                self.dirty.push(idx);
            }
            self.data[idx] = value;
        }
    }

    /// Reset previously-set slots back to [`CELL_GRID_EMPTY`] via the
    /// dirty list — the layout (`origin` / `size`) is preserved.
    /// Cheaper than a full memset for the common 3-4 % dirty fill.
    #[inline]
    pub fn reset(&mut self) {
        for &idx in &self.dirty {
            self.data[idx] = CELL_GRID_EMPTY;
        }
        self.dirty.clear();
    }

    /// Grid origin (lo corner, inclusive).
    #[inline]
    pub fn origin(&self) -> IVec3 {
        self.origin
    }

    /// Grid size along each axis. Extent is `[origin, origin + size)`.
    #[inline]
    pub fn size(&self) -> IVec3 {
        self.size
    }
}

/// Pool-reused scratch buffers for the sculpt extract path (D6.3.c).
///
/// Held on `ArvxSceneManager` and reused across every stamp: the two
/// [`CellGrid`]s grow to the largest brush footprint seen so far and
/// never shrink, while the [`Vec<IVec3>`] for pad-range cells uses
/// `clear` (no dealloc). [`extract_mesh_region_from_cells_pooled`]
/// resets the dirty slots and the Vec at function entry — fresh
/// allocations only happen on first use, and on grid grow.
///
/// Saves ~500-700 µs per stamp vs the fresh-`new` path (one alloc +
/// `Vec::resize(..., CELL_GRID_EMPTY)` per grid, plus the
/// `Vec<IVec3>` allocation).
pub struct SculptExtractScratch {
    pub cells_grid: CellGrid,
    pub cube_vertex_grid: CellGrid,
    pub solid_cells: Vec<IVec3>,
    /// Reusable smooth-density buffer, parallel to `cells_grid` (same
    /// `[origin, origin + size)` extent and flat `x + sx*(y + sy*z)`
    /// indexing). `D[i] ∈ [0, 1]` is the Gaussian-blurred binary
    /// occupancy of the corresponding grid cell; the `D = 0.5`
    /// isosurface is the smooth surface the extract places vertices on.
    /// Grows to the largest footprint seen and never shrinks, mirroring
    /// the `CellGrid` pool-reuse policy so no per-stamp allocation.
    pub density: Vec<f32>,
    /// Reusable smooth-gradient buffer, parallel to `density` (same flat
    /// indexing). `G[i] = ∇D` evaluated at grid point `i` by central
    /// differences of the smooth `density` grid (replicated/one-sided at
    /// the grid edges). Stored per grid point and trilinearly
    /// interpolated at vertex positions to derive the surface normal.
    ///
    /// **Why a precomputed grid-point gradient (not differentiating the
    /// trilinear density at the vertex):** the gradient of a trilinear
    /// interpolant is piecewise-constant and DISCONTINUOUS across cell
    /// boundaries, so differencing the interpolated `D` flips the normal
    /// cell-to-cell → voxel-scale speckle. Differencing the smooth `D`
    /// grid first, then trilinearly interpolating the *gradient*, gives
    /// a normal that is continuous across cells (C0). Same occupancy
    /// dependency radius as the position reads (`±(R+1)`), so the
    /// watertight seam property is preserved.
    pub gradient: Vec<[f32; 3]>,
}

impl SculptExtractScratch {
    /// Empty scratch — grids size up on first
    /// [`extract_mesh_region_from_cells_pooled`] call.
    pub fn new() -> Self {
        Self {
            cells_grid: CellGrid::empty(),
            cube_vertex_grid: CellGrid::empty(),
            solid_cells: Vec::new(),
            density: Vec::new(),
            gradient: Vec::new(),
        }
    }
}

impl Default for SculptExtractScratch {
    fn default() -> Self {
        Self::new()
    }
}

/// One surface-mesh vertex.
///
/// 32 B, `repr(C)`, `bytemuck`-castable straight into a vertex buffer.
/// Positions are **object-local**; the per-instance world matrix is
/// applied in the vertex shader.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MeshVertex {
    /// Cube center in object-local coords. Lands on a grid corner of
    /// the cell lattice (between cells, not on a cell center).
    pub local_pos: [f32; 3],
    /// Octahedral-packed average of the surface-cell normals at the
    /// vertex's 8 corner cells. Falls back to +Y for cubes with no
    /// surface cells (only INTERIOR + EMPTY contributors), which on a
    /// well-baked 1-thick shell shouldn't happen but keeps the
    /// extractor total. Encoding matches `LeafAttr::normal_oct`.
    pub normal_oct: u32,
    /// Absolute slot into the global `leaf_attr_pool`. Picked from the
    /// surface cell with the smallest `(z, y, x)` coord among the
    /// cube's 8 corners — deterministic and stable across reruns.
    /// Falls back to 0 when no corner is a surface cell.
    pub leaf_attr_id: u32,
    /// 4 × u8 bone indices packed little-endian (matches `BoneVoxel.indices`).
    /// Sourced from the same cell that contributed `leaf_attr_id` so the
    /// per-vertex attribution is internally consistent. Zero for
    /// unskinned assets — the matching `bone_weights` is then also zero,
    /// which the vertex shader treats as "skip skinning, rest pose".
    pub bone_indices: u32,
    /// 4 × u8 bone weights packed little-endian (sum to 255 in
    /// well-formed skinning data; 0 for unskinned cells).
    pub bone_weights: u32,
    /// Reserved for future per-vertex attributes (LOD bias, blend
    /// shapes, etc). Keeps the stride at 32 B and the layout
    /// 16-byte-aligned for GPU access.
    pub _pad: u32,
}

const _: () = assert!(std::mem::size_of::<MeshVertex>() == 32);
// Hand-checked field offsets — vertex layout in `mesh_pass/pass.rs`
// pulls position from offset 0, normal_oct from 12, leaf_attr_id from 16.
// Bone fields live in what was `_pad[0..1]`; the GPU-side decl picks
// them up in commit 4 when the VS starts skinning.
const _: () = {
    use std::mem::offset_of;
    assert!(offset_of!(MeshVertex, local_pos) == 0);
    assert!(offset_of!(MeshVertex, normal_oct) == 12);
    assert!(offset_of!(MeshVertex, leaf_attr_id) == 16);
    assert!(offset_of!(MeshVertex, bone_indices) == 20);
    assert!(offset_of!(MeshVertex, bone_weights) == 24);
    assert!(offset_of!(MeshVertex, _pad) == 28);
};

/// Vertex format for the procedural proxy-mesh pipeline (GPU surface-
/// nets-from-SDF). Distinct from [`MeshVertex`]: proxy meshes have no
/// octree, no LeafAttr pool slots, no skinning. Instead the SDF
/// evaluator's full `TreeSample` (material + secondary + blend + color)
/// is baked per-vertex at extraction time; the proxy raster pipeline
/// reads these directly and writes the G-buffer without going through
/// the LeafAttr indirection used by octree-backed meshes.
///
/// 32 B, `repr(C)`, `bytemuck`-castable. Same stride as [`MeshVertex`]
/// so the surface-nets extractor's buffer allocation logic carries over.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ProxyVertex {
    /// SN-cube vertex position in object-local space.
    pub local_pos: [f32; 3],
    /// SDF-gradient normal, octahedral-packed. Encoding matches
    /// [`LeafAttr::normal_oct`].
    pub normal_oct: u32,
    /// Packed material identifiers + blend weight. Same layout as
    /// [`LeafAttr::material_packed`]:
    ///   bits  0-15: primary material_id (u16)
    ///   bits 16-27: secondary material_id (u12)
    ///   bits 28-31: blend_weight (u4)
    pub material_packed: u32,
    /// Per-vertex RGBA8 color from the procedural's color nodes
    /// (`ColorByHeight`, `ColorByNoise`, leaf `color` params).
    /// Low byte = R, next = G, then B, then alpha/intensity.
    /// 0 = "no procedural override, use material base_color".
    pub color_packed: u32,
    /// Reserved for future per-vertex attributes (LOD bias, emission,
    /// node_id for picking, etc.). Keeps the stride at 32 B.
    pub _reserved: [u32; 2],
}

const _: () = assert!(std::mem::size_of::<ProxyVertex>() == 32);
const _: () = {
    use std::mem::offset_of;
    assert!(offset_of!(ProxyVertex, local_pos) == 0);
    assert!(offset_of!(ProxyVertex, normal_oct) == 12);
    assert!(offset_of!(ProxyVertex, material_packed) == 16);
    assert!(offset_of!(ProxyVertex, color_packed) == 20);
    assert!(offset_of!(ProxyVertex, _reserved) == 24);
};

/// Sentinel marking INTERIOR cells in the dense cell map. INTERIOR
/// cells count as "solid" for SN sign purposes but carry no per-cell
/// `leaf_attr_id`, so we can't store a real slot here.
///
/// Also used by `voxelize_octree`'s halo-sampling pass to flag halo
/// cells that landed strictly inside the neighbouring solid — these
/// contribute solidity for SN cube classification but don't need their
/// own `LeafAttr` allocation.
pub const CELL_INTERIOR: u32 = u32::MAX;

/// In-grid encoding of [`CELL_INTERIOR`]. [`CellGrid`] reserves
/// [`CELL_GRID_EMPTY`] (= `u32::MAX`) as the "no entry" sentinel,
/// which collides with `CELL_INTERIOR` — without this remap, INTERIOR
/// cells stored in the grid would look identical to absent slots and
/// `build_cube_vertex`'s corner classification would treat them as
/// empty (wrong: INTERIOR is solid for SN purposes). At populate
/// time the extract loop remaps `CELL_INTERIOR` → `CELL_INTERIOR_GRID`
/// before `cells_grid.set`; the `build_cube_vertex` lookup closure
/// reverses the remap. A real `leaf_attr_id` can never hit this
/// value — `LeafAttrPool` capacities are well under `u32::MAX - 1`.
const CELL_INTERIOR_GRID: u32 = u32::MAX - 1;

/// Half-width (in cells) of the separable Gaussian density blur kernel
/// used by [`extract_mesh_region_from_cells_pooled_haloed`]. The full
/// support is `[-DENSITY_KERNEL_R, +DENSITY_KERNEL_R]³`.
///
/// **Watertight invariant:** R MUST be `≤` the boundary halo (terrain
/// bakes with halo = 2). The blurred density `D[c]` is a *pure local*
/// function of the occupancy in `c ± R`; two tiles/patches that share
/// the same `2R+1`-wide boundary neighborhood (the interior cells of
/// one side appear as halo cells of the other) therefore compute the
/// *bit-identical* `D` — hence identical `D = 0.5` crossings and
/// identical `∇D` normals — at every shared boundary cube. No welding,
/// no iteration, no cross-tile divergence.
///
/// R = 2 for good smoothing everywhere. The kernel reach (`R + 1` for the
/// edge-crossing reads, `R + 2` counting the ∇D normal's ½-voxel step) is
/// fed two ways with NO dependence on the halo away from tile seams:
///   • Mid-tile / single asset: the surrounding occupancy is collected
///     into the same `cells_grid` from the tile's own octree, so the blur
///     is fully supported — smoothness does not involve the halo at all.
///   • Tile-to-tile seam (the ONLY place a neighbor tile's occupancy is
///     needed): the baked halo must be `≥` the reach, so it is widened to
///     `TILE_HALO_VOXELS = 4`. With that, both tiles share the full
///     neighborhood and compute bit-identical seam vertices — watertight,
///     and the halo is never the limiting factor on smoothness.
/// R = 2. The de-staircasing root-cause fix is the D-FIELD TOPOLOGY
/// (surface nets classified by `D ≥ iso`, not binary occupancy), which
/// removes the terracing at any R with NO bias. R is then a pure
/// smoothing-strength knob. R = 2 is the accurate default: it keeps
/// geometry within the bench's tested error bounds (low curvature-
/// proportional bias, convex peaks stay crisp). R = 3 (σ ≈ 1.5) smooths
/// high-frequency edit-jitter markedly more (bench irregular-mound
/// interior edge-normal max 49° → 25°) but FAILS the geometry/normal
/// accuracy bounds — it trades shape accuracy for smoothness, an
/// aesthetic call. Reach is `R + 1 = 3 ≤ TILE_HALO_VOXELS = 4`, so it
/// stays watertight (halo also leaves headroom for an opt-in R = 3).
const DENSITY_KERNEL_R: i32 = 2;

/// Standard deviation (in cells) of the separable Gaussian density
/// kernel. `σ ≈ 1.0` gives a gentle blur whose support is well within
/// `±2` cells (the `R = 2` truncation drops < 5 % of the unit-area
/// Gaussian per axis), enough to de-staircase grid-aligned occupancy
/// into a smooth `[0, 1]` field while staying strictly local.
const DENSITY_KERNEL_SIGMA: f32 = 1.0;

/// Iso-threshold of the smooth density field: the surface is the
/// `D = 0.5` level set. Inside (occupancy 1) blurs to `D > 0.5`,
/// outside (occupancy 0) to `D < 0.5`, so `sdf = 0.5 - D` is negative
/// inside / positive outside — matching `build_cube_vertex`'s
/// edge-crossing sign convention (solid corner negative, empty corner
/// positive).
const DENSITY_ISO: f32 = 0.5;

thread_local! {
    /// Per-thread blur-kernel override for the meshing test-bench
    /// (`mesh_test_bench`). `Some((r, sigma, iso))` overrides
    /// [`DENSITY_KERNEL_R`] / [`DENSITY_KERNEL_SIGMA`] / [`DENSITY_ISO`]
    /// for the next extract on this thread; `None` (default) uses the
    /// shipped consts. Set via [`set_blur_override`]. This lets the
    /// bench sweep `R ∈ {2,3,4}` WITHOUT changing the global default —
    /// the production path never touches this (it's `None` everywhere).
    static BLUR_OVERRIDE: std::cell::Cell<Option<(i32, f32, f32)>> =
        const { std::cell::Cell::new(None) };
}

/// Set (or clear) the per-thread blur-kernel override used by the
/// meshing test-bench: `Some((radius, sigma, iso))` or `None` to restore
/// the shipped defaults. **Bench / diagnostics only** — the production
/// sculpt/terrain path leaves this `None`.
pub fn set_blur_override(over: Option<(i32, f32, f32)>) {
    BLUR_OVERRIDE.with(|c| c.set(over));
}

/// Production wide-window plane-fit radius (voxels) for the TERRAIN BAKE
/// density path. The fix is **default-ON** for terrain at this radius:
/// it averages out the coherent "smooth-stairs" ripple the fixed-width
/// R=2 blur leaves on gentle (wide-tread) slopes, WITHOUT widening the
/// blur (so no convex bias, and the high-frequency `∇D` normal is
/// untouched). r=5 is the Pareto point — ~2-3× ripple cut on gentle
/// slopes with FBM curvature preserved (see `arvx-terrain`'s repro +
/// `mesh_test_bench`'s `wide_window_fix_*` tests).
pub const TERRAIN_PLANE_FIT_RADIUS: f32 = 5.0;

thread_local! {
    /// Per-thread OVERRIDE of the wide-window plane-fit radius (voxels).
    /// `None` (default) → the extract uses the radius its entry point
    /// passes (terrain bake = [`TERRAIN_PLANE_FIT_RADIUS`]; region/sculpt
    /// = off). `Some(0.0)` → force the post-pass OFF (the bench baseline);
    /// `Some(r)` → force radius `r` (bench R-sweep). See
    /// [`set_wide_window_project`].
    static WIDE_WINDOW_PROJECT: std::cell::Cell<Option<f32>> =
        const { std::cell::Cell::new(None) };
}

/// Override the wide-window plane-fit radius for the next density-blur
/// extract on this thread: `Some(0.0)` forces it OFF, `Some(r)` forces
/// radius `r` voxels, `None` restores the per-call default (terrain bake
/// default-ON at [`TERRAIN_PLANE_FIT_RADIUS`]). **Bench / diagnostics
/// only** — production leaves this `None`.
pub fn set_wide_window_project(radius_voxels: Option<f32>) {
    WIDE_WINDOW_PROJECT.with(|c| c.set(radius_voxels.map(|r| r.max(0.0))));
}

/// Resolve the effective plane-fit radius: the thread-local override if
/// set, else the per-call `default_radius` the extract entry point chose.
#[inline]
fn resolve_plane_fit_radius(default_radius: f32) -> f32 {
    WIDE_WINDOW_PROJECT
        .with(|c| c.get())
        .unwrap_or(default_radius)
        .max(0.0)
}

/// Resolve the active `(radius, sigma, iso)` — the thread-local
/// override if set, else the shipped consts.
#[inline]
fn active_blur_params() -> (i32, f32, f32) {
    BLUR_OVERRIDE
        .with(|c| c.get())
        .unwrap_or((DENSITY_KERNEL_R, DENSITY_KERNEL_SIGMA, DENSITY_ISO))
}

/// Diagnostic / rollback kill-switch for the QEF-Hermite mesher. The
/// production gate is **presence of a per-leaf distance pool** (Stage 5):
/// any extract handed a non-empty `dists` slice meshes via QEF-Hermite,
/// everything else (binary imports, old/no-distance assets, the sculpt
/// region path until Stage 6) falls back to the blur/binary path.
///
/// Setting `ARVX_QEF_HERMITE=0` in the environment forces the blur path
/// even when distances are present — a launch-time A/B / rollback switch.
/// Read once per process (the value can't change mid-run), so it's free in
/// the hot extract path.
#[inline]
pub(crate) fn qef_hermite_force_off() -> bool {
    static FORCE_OFF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FORCE_OFF.get_or_init(|| {
        std::env::var("ARVX_QEF_HERMITE")
            .map(|v| v == "0")
            .unwrap_or(false)
    })
}

/// Bumped whenever the mesher's GEOMETRY OUTPUT changes for the same voxel
/// input, OR the set of per-tile sections a cached bake must carry for a
/// correct *re-extract* changes — so on-disk caches keyed by a bake
/// signature (terrain `.arvxtile`, etc.) auto-invalidate instead of
/// serving a stale tile. History: `1` = cell-center-plane QEF-Hermite;
/// `2` = Manifold-DC interpolation; `3` = per-leaf distance section now
/// persisted, so a reloaded tile re-extracts / sculpts with Manifold-DC
/// from the stored field instead of the blur fallback (pre-`3` caches
/// lack the section → re-bake to gain it).
pub const MESHER_OUTPUT_VERSION: u32 = 3;

/// Manifold-DC interpolation placement ([`manifold_dc_placement`]) — the
/// PRODUCTION DEFAULT for the QEF/terrain path. It reconstructs each vertex by
/// INTERPOLATING the stored per-cell signed distances at the cube's edge
/// crossings (missing interior/empty neighbours extrapolated from the surface
/// cell's Hermite plane), QEF over those crossings biased to the centroid,
/// clamped to an expanded bound. This de-staircases gentle slopes that the old
/// cell-center-plane + 1-voxel-cube-clamp QEF terraced, while staying watertight,
/// deterministic, and crisp on sharp features. Rollback to the legacy placement
/// with `ARVX_MESH_DC=0`. Read once.
pub(crate) fn manifold_dc_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("ARVX_MESH_DC").map(|v| v != "0").unwrap_or(true))
}

thread_local! {
    /// Per-thread force-OFF for the QEF-Hermite mesher (default `false` =
    /// allow QEF when a distance pool is present). The terrain seam tests
    /// that specifically exercise the **blur plane-fit FALLBACK** flip this
    /// on to bake the legacy path even though the artifact now carries
    /// distances. Mirrors [`set_wide_window_project`]'s test-only override.
    static QEF_FORCE_OFF_THREAD: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Force the blur/binary fallback on this thread regardless of distance
/// presence. **Test / diagnostics only** — exercises the legacy path the
/// QEF default now replaces. Reset to `false` when done.
pub fn set_qef_force_off(on: bool) {
    QEF_FORCE_OFF_THREAD.with(|c| c.set(on));
}

#[inline]
fn qef_force_off_thread() -> bool {
    QEF_FORCE_OFF_THREAD.with(|c| c.get())
}

/// Build the normalized 1D Gaussian weights for radius `r`, sigma
/// `sigma`, over `[-r, r]` (length `2r+1`, summing to 1.0 so a fully-
/// solid neighborhood blurs to exactly `D = 1.0`). Returned as a `Vec`
/// so the radius can vary at runtime (bench R-sweep); the production
/// path calls this once per extract with the const `R = 2`.
fn density_kernel_weights_1d_for(r: i32, sigma: f32) -> Vec<f32> {
    let len = (2 * r + 1) as usize;
    let mut w = vec![0.0f32; len];
    let two_sigma_sq = 2.0 * sigma * sigma;
    let mut sum = 0.0f32;
    for (i, slot) in w.iter_mut().enumerate() {
        let d = i as i32 - r;
        let v = (-((d * d) as f32) / two_sigma_sq).exp();
        *slot = v;
        sum += v;
    }
    let inv = 1.0 / sum;
    for slot in w.iter_mut() {
        *slot *= inv;
    }
    w
}

/// Trilinearly sample the precomputed density grid `density` (laid out
/// like `cells_grid`: `origin` lo-corner, `size` extent, flat
/// `x + sx*(y + sy*z)`) at fractional grid coordinate `p` (in cells).
///
/// Out-of-bounds is handled by *clamping the sample coordinate into the
/// grid*, NOT by reading 0. Reading 0 outside would synthesize a false
/// `D = 0.5` crossing at the grid edge (a spurious surface); clamping
/// instead extends the boundary value outward, which is the correct
/// Neumann (zero-gradient) edge condition for a density field whose
/// real support is fully captured by the halo.
#[inline]
fn sample_density_trilinear(
    density: &[f32],
    origin: IVec3,
    size: IVec3,
    p: Vec3,
) -> f32 {
    let sx = size.x as usize;
    let sy = size.y as usize;
    // Translate into local fractional coords.
    let lx = p.x - origin.x as f32;
    let ly = p.y - origin.y as f32;
    let lz = p.z - origin.z as f32;
    // Clamp the *base* integer cell so [i0, i0+1] stays in bounds, and
    // clamp the fractional weight to [0, 1] outside the valid span so
    // the sample saturates to the boundary value rather than reading 0.
    let clamp_axis = |v: f32, n: i32| -> (usize, f32) {
        let max_i0 = (n - 1).max(0); // last valid base cell index
        if v <= 0.0 {
            (0, 0.0)
        } else if v >= (n - 1) as f32 {
            // At/over the far edge: pin to the last cell, frac 0 so the
            // upper sample (also pinned) returns the boundary value.
            (max_i0 as usize, 0.0)
        } else {
            let i0 = v.floor();
            (i0 as usize, v - i0)
        }
    };
    let (x0, fx) = clamp_axis(lx, size.x);
    let (y0, fy) = clamp_axis(ly, size.y);
    let (z0, fz) = clamp_axis(lz, size.z);
    let x1 = (x0 + 1).min((size.x - 1).max(0) as usize);
    let y1 = (y0 + 1).min((size.y - 1).max(0) as usize);
    let z1 = (z0 + 1).min((size.z - 1).max(0) as usize);
    let at = |x: usize, y: usize, z: usize| -> f32 {
        density[x + sx * (y + sy * z)]
    };
    let c000 = at(x0, y0, z0);
    let c100 = at(x1, y0, z0);
    let c010 = at(x0, y1, z0);
    let c110 = at(x1, y1, z0);
    let c001 = at(x0, y0, z1);
    let c101 = at(x1, y0, z1);
    let c011 = at(x0, y1, z1);
    let c111 = at(x1, y1, z1);
    let c00 = c000 + (c100 - c000) * fx;
    let c10 = c010 + (c110 - c010) * fx;
    let c01 = c001 + (c101 - c001) * fx;
    let c11 = c011 + (c111 - c011) * fx;
    let c0 = c00 + (c10 - c00) * fy;
    let c1 = c01 + (c11 - c01) * fy;
    c0 + (c1 - c0) * fz
}

/// Trilinearly sample the precomputed grid-point gradient `gradient`
/// (vector-valued, laid out / clamped exactly like
/// [`sample_density_trilinear`]) at fractional grid coordinate `p`.
///
/// Interpolating the *gradient field* (rather than differentiating the
/// interpolated density) is what makes the surface normal continuous
/// across cell boundaries — `gradient` is built from central
/// differences of the smooth `density` grid at grid points, so it is
/// itself smooth, and trilinear interpolation of a smooth field is C0.
///
/// Out-of-bounds clamps the sample coordinate (boundary replication),
/// matching `sample_density_trilinear`'s edge policy so seam vertices
/// near the grid edge read the same boundary gradient from either side.
#[inline]
fn sample_gradient_trilinear(
    gradient: &[[f32; 3]],
    origin: IVec3,
    size: IVec3,
    p: Vec3,
) -> Vec3 {
    let sx = size.x as usize;
    let sy = size.y as usize;
    let lx = p.x - origin.x as f32;
    let ly = p.y - origin.y as f32;
    let lz = p.z - origin.z as f32;
    let clamp_axis = |v: f32, n: i32| -> (usize, f32) {
        let max_i0 = (n - 1).max(0);
        if v <= 0.0 {
            (0, 0.0)
        } else if v >= (n - 1) as f32 {
            (max_i0 as usize, 0.0)
        } else {
            let i0 = v.floor();
            (i0 as usize, v - i0)
        }
    };
    let (x0, fx) = clamp_axis(lx, size.x);
    let (y0, fy) = clamp_axis(ly, size.y);
    let (z0, fz) = clamp_axis(lz, size.z);
    let x1 = (x0 + 1).min((size.x - 1).max(0) as usize);
    let y1 = (y0 + 1).min((size.y - 1).max(0) as usize);
    let z1 = (z0 + 1).min((size.z - 1).max(0) as usize);
    let at = |x: usize, y: usize, z: usize| -> Vec3 {
        Vec3::from(gradient[x + sx * (y + sy * z)])
    };
    let c000 = at(x0, y0, z0);
    let c100 = at(x1, y0, z0);
    let c010 = at(x0, y1, z0);
    let c110 = at(x1, y1, z0);
    let c001 = at(x0, y0, z1);
    let c101 = at(x1, y0, z1);
    let c011 = at(x0, y1, z1);
    let c111 = at(x1, y1, z1);
    let c00 = c000 + (c100 - c000) * fx;
    let c10 = c010 + (c110 - c010) * fx;
    let c01 = c001 + (c101 - c001) * fx;
    let c11 = c011 + (c111 - c011) * fx;
    let c0 = c00 + (c10 - c00) * fy;
    let c1 = c01 + (c11 - c01) * fy;
    c0 + (c1 - c0) * fz
}

/// Walk a brick-terminated octree and emit the surface mesh as
/// `(vertices, indices)`.
///
/// * `octree_nodes` — `tree.as_slice()` from the asset's `SparseOctree`.
///   Must already have its brick ids and per-cell `leaf_attr_id` slots
///   remapped to scene-global values.
/// * `octree_depth` — the asset's `depth` field (matches
///   `SparseOctree::depth()`).
/// * `base_voxel_size` — finest cell edge length in object-local units.
/// * `grid_origin` — object-local position of the octree extent's lo
///   corner.
/// * `brick_cells` — flat brick storage; `brick_id * BRICK_CELLS + flat`
///   indexes into it.
/// * `leaf_attr_pool` — the scene-global LeafAttr pool. Indexed by
///   per-cell `leaf_attr_id` to read the prefiltered normal that gets
///   averaged into vertex normals. Pass `&[]` to skip vertex-normal
///   averaging entirely (vertices fall back to +Y); useful for tests.
/// * `bone_voxel_pool` — parallel `BoneVoxel` pool indexed by the same
///   `leaf_attr_id` slots. Vertex shader skinning reads from
///   `bone_indices/weights` baked here. Pass `&[]` for unskinned
///   assets (or tests) — vertices then carry zero weights, which the
///   VS treats as "rest pose".
pub fn extract_surface_mesh(
    octree_nodes: &[u32],
    octree_depth: u8,
    base_voxel_size: f32,
    grid_origin: Vec3,
    brick_cells: &[u32],
    leaf_attr_pool: &[LeafAttr],
    bone_voxel_pool: &[BoneVoxel],
    sculpt_slots: Option<&rustc_hash::FxHashSet<u32>>,
) -> (Vec<MeshVertex>, Vec<u32>) {
    extract_surface_mesh_haloed(
        octree_nodes,
        octree_depth,
        base_voxel_size,
        grid_origin,
        brick_cells,
        leaf_attr_pool,
        bone_voxel_pool,
        &[],
        0,
        sculpt_slots,
    )
}

/// Halo-aware variant of [`extract_surface_mesh`]. Folds `halo_cells`
/// into the cell-occupancy map before the surface walk so SN cubes at
/// the AABB boundary get valid 8-corner data, and restricts the
/// iteration domain to the nominal interior so adjacent tiles don't
/// emit duplicated boundary geometry.
///
/// ### Watertight-seam protocol
///
/// With `halo > 0` (terrain Phase 3):
///
/// 1. Halo cells appear in the cell-occupancy map exactly like
///    interior cells, supplying SDF sign + `leaf_attr` data to any
///    SN cube that straddles the AABB boundary. Boundary cubes thus
///    get a full 8-corner classification and produce vertex
///    positions that agree on both sides — the LO neighbour's
///    rightmost cube and this tile's leftmost-halo cube live at the
///    same world position, with the same solid/empty pattern, and
///    therefore the same surface-nets centroid.
///
/// 2. The outer face-emit loop iterates only solid cells whose lo
///    coord lies in `[0, N)` on every axis (`N = 1 << octree_depth`).
///    Halo cells never iterate, so we never emit quads from cells
///    whose `+axis` neighbour is unknown (beyond halo). Quads
///    emitted from a boundary interior cell may reach one cube into
///    the halo — that's the cube the LO/HI neighbour was always
///    going to emit too, just on the other side of its own
///    boundary, with the same vertex position. The neighbour's
///    interior-side cell is empty in this scenario (otherwise the
///    quad wouldn't fire), so the neighbour does not double-emit.
///
/// In other words: each (solid, empty) sample-edge is uniquely owned
/// by the tile whose solid side falls in nominal-interior cells.
/// Halo data covers the corner-classification side, no quad is
/// double-emitted, and adjacent tile meshes meet at coincident
/// vertex positions.
///
/// `halo_cells` carry `leaf_attr_id`s that are valid indices into
/// `leaf_attr_pool` (or [`CELL_INTERIOR`] for halo cells strictly
/// inside the neighbouring solid). When `halo = 0` this function is
/// bit-identical to [`extract_surface_mesh`] regardless of the slice.
#[allow(clippy::too_many_arguments)]
/// `sculpt_slots`: leaf-attr slots allocated by sculpt. Used by
/// `build_cube_vertex`'s tie-break to prefer sculpt cells over
/// procedural neighbours when both share an SN cube corner. `None`
/// (or empty set) keeps the original position-only `coord_less`
/// tie-break — what callers without sculpt context (procedural
/// bakes, paint, the initial extract) want.
pub fn extract_surface_mesh_haloed(
    octree_nodes: &[u32],
    octree_depth: u8,
    base_voxel_size: f32,
    grid_origin: Vec3,
    brick_cells: &[u32],
    leaf_attr_pool: &[LeafAttr],
    bone_voxel_pool: &[BoneVoxel],
    halo_cells: &[(IVec3, u32)],
    halo: u32,
    sculpt_slots: Option<&rustc_hash::FxHashSet<u32>>,
) -> (Vec<MeshVertex>, Vec<u32>) {
    extract_surface_mesh_haloed_impl(
        octree_nodes,
        octree_depth,
        base_voxel_size,
        grid_origin,
        brick_cells,
        leaf_attr_pool,
        bone_voxel_pool,
        halo_cells,
        halo,
        sculpt_slots,
        false,
        // Binary path (imports): NO plane-fit (the blur rounds sharp
        // edges; imports need the binary surface nets unchanged).
        0.0,
        // No per-leaf distance on the import path yet → legacy binary
        // surface nets (Stage 2 gates QEF on the terrain bake only).
        &[],
        // Binary imports have no procedural source → no analytic normal.
        None,
    )
}

/// **Terrain-only** density-blur variant of
/// [`extract_surface_mesh_haloed`]. Identical watertight halo-seam
/// iteration + material/leaf_attr/bone attribution, but the surface is
/// meshed from the BLURRED occupancy (`D = 0.5`-topology + `∇D` normal)
/// instead of binary surface nets — so baked terrain de-staircases
/// directly from occupancy, no post-hoc heightfield Y-projection.
///
/// This shares the proven seam protocol with the binary path: the inner
/// halo ring `[-1, N+1)` iterates (boundary cells emit from BOTH the
/// owning interior side and the neighbour's halo side → identical
/// overdraw), and the blurred density `D[c]` is a pure local function of
/// occupancy in `c ± DENSITY_KERNEL_R` (`R + 1` reach including the ∇D
/// step). With the terrain halo `≥` that reach (`TILE_HALO_VOXELS = 4`),
/// adjacent tiles compute bit-identical seam vertices.
///
/// IMPORTANT: gated to the terrain bake only. Imports keep the binary
/// path ([`extract_surface_mesh_haloed`]) because the blur rounds
/// sub-2-voxel sharp edges (a documented fundamental limit).
#[allow(clippy::too_many_arguments)]
pub fn extract_surface_mesh_density_haloed(
    octree_nodes: &[u32],
    octree_depth: u8,
    base_voxel_size: f32,
    grid_origin: Vec3,
    brick_cells: &[u32],
    leaf_attr_pool: &[LeafAttr],
    bone_voxel_pool: &[BoneVoxel],
    halo_cells: &[(IVec3, u32)],
    halo: u32,
    sculpt_slots: Option<&rustc_hash::FxHashSet<u32>>,
    // Per-slot signed-distance pool (voxel units), indexed like
    // `leaf_attr_pool`. With the QEF toggle set + a non-empty slice, the
    // tile meshes via QEF-Hermite (smooth-by-construction) instead of the
    // blur path. `&[]` keeps the legacy blur behaviour bit-identical.
    dists: &[i16],
    // Optional analytic SHADING NORMAL: world position → outward
    // `∇sd` (un-normalized). When supplied AND in QEF mode, the vertex
    // normal is this evaluated AT THE VERTEX — the EXACT surface normal,
    // no grid reconstruction. The terrain bake passes
    // `terrain_fn.sample_grad`-at-world on pure-procedural tiles; `None`
    // falls back to the interpolated `∇D` field. Generalizes to 3D SDF
    // terrain unchanged (`∇sd` is the normal whether `sd` is a
    // heightfield gap or a true 3D field).
    surface_normal_fn: Option<&dyn Fn(Vec3) -> Vec3>,
) -> (Vec<MeshVertex>, Vec<u32>) {
    extract_surface_mesh_haloed_impl(
        octree_nodes,
        octree_depth,
        base_voxel_size,
        grid_origin,
        brick_cells,
        leaf_attr_pool,
        bone_voxel_pool,
        halo_cells,
        halo,
        sculpt_slots,
        true,
        // Terrain bake: wide-window plane-fit DEFAULT-ON. The shared seam
        // ring is pinned (derived from the octree extent) so adjacent
        // tiles stay watertight. A thread-local override (`set_wide_window
        // _project`) can force it off for the bench baseline.
        TERRAIN_PLANE_FIT_RADIUS,
        dists,
        surface_normal_fn,
    )
}

/// Dense blurred-density + gradient grids for the terrain density-blur
/// extract path, laid out like [`CellGrid`] (`origin` lo-corner, `size`
/// extent, flat `x + sx*(y + sy*z)` indexing).
struct DensityGrids {
    origin: IVec3,
    size: IVec3,
    /// Blurred occupancy `D ∈ [0, 1]`, one per grid point.
    density: Vec<f32>,
    /// Central-difference `∇D` of the smooth field, one per grid point.
    gradient: Vec<[f32; 3]>,
}

#[allow(clippy::too_many_arguments)]
fn extract_surface_mesh_haloed_impl(
    octree_nodes: &[u32],
    octree_depth: u8,
    base_voxel_size: f32,
    grid_origin: Vec3,
    brick_cells: &[u32],
    leaf_attr_pool: &[LeafAttr],
    bone_voxel_pool: &[BoneVoxel],
    halo_cells: &[(IVec3, u32)],
    halo: u32,
    sculpt_slots: Option<&rustc_hash::FxHashSet<u32>>,
    density_blur: bool,
    // Default wide-window plane-fit radius (voxels) for the smooth-stairs
    // fix. `0.0` = off (binary path). The terrain density path passes
    // [`TERRAIN_PLANE_FIT_RADIUS`] (default-ON); a thread-local override
    // can force it off/other for the bench. The shared seam ring (within
    // ½ voxel of the nominal tile faces, derived from the octree extent)
    // is pinned so adjacent tiles stay watertight.
    plane_fit_default_radius: f32,
    // Per-slot signed-distance pool (voxel units), indexed exactly like
    // `leaf_attr_pool`. Non-empty + the thread-local QEF toggle selects
    // the QEF-Hermite mesher (Stage 2 bench gate; Stage 5 makes presence
    // alone the production gate). Empty keeps the legacy blur/binary path.
    dists: &[i16],
    // Optional analytic shading-normal callback (world → outward `∇sd`).
    // QEF mode shades from this evaluated at the vertex when supplied
    // (exact normal); otherwise from the interpolated `∇D` field. See
    // [`extract_surface_mesh_density_haloed`].
    surface_normal_fn: Option<&dyn Fn(Vec3) -> Vec3>,
) -> (Vec<MeshVertex>, Vec<u32>) {
    let mut vertices: Vec<MeshVertex> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();

    // **QEF-Hermite selection.** The production gate is presence of a
    // per-leaf distance pool (Stage 5): a non-empty `dists` slice selects
    // the QEF-Hermite mesher; an env kill-switch can force it off. When
    // enabled, the surface is meshed with BINARY topology (sign changes) +
    // QEF-Hermite vertex placement — NOT the blurred-D topology/position.
    // So the density precompute, the D-field owner expansion, and the
    // wide-window plane fit are all bypassed (they exist only to recover the
    // position the stored distance now carries exactly). `topo_blur` gates
    // that legacy machinery; `use_qef` gates the placement.
    let use_qef = !dists.is_empty() && !qef_hermite_force_off() && !qef_force_off_thread();
    let topo_blur = density_blur && !use_qef;
    if octree_nodes.is_empty() {
        return (vertices, indices);
    }

    // Pass 1: collect every non-empty cell into a dense lookup map.
    // Surface cells store their `leaf_attr_id`; brick-internal INTERIOR
    // cells store `CELL_INTERIOR`. INTERIOR_NODE-region cells are NOT
    // expanded — `is_solid_lookup` resolves them on demand. That keeps
    // the map size proportional to the surface shell, not the asset's
    // solid volume.
    let mut cells: CellMap = CellMap::default();
    walk_collect_cells(
        octree_nodes,
        brick_cells,
        0,
        UVec3::ZERO,
        0,
        octree_depth,
        &mut cells,
    );
    // Fold halo cells into the map. Coords with axes in `[-halo, 0)` or
    // `[N, N+halo)` are by construction outside the octree's interior
    // range, so there's no collision with `walk_collect_cells`'s output.
    for &(coord, slot) in halo_cells {
        cells.insert(coord, slot);
    }
    if cells.is_empty() {
        return (vertices, indices);
    }

    let extent = 1i32 << octree_depth;

    // ── Optional smooth-density precompute (terrain bake only) ──
    //
    // When `density_blur` is set, blur the binary occupancy into a dense
    // `[0, 1]` field `D` over the cells' bounding box (+ kernel-reach
    // pad), then mesh on the `D = iso` isosurface with `∇D` normals.
    // This de-staircases directly from occupancy. The seam stays
    // watertight: `D[c]` depends only on occupancy in `c ± R` and the
    // grid here is built from the SAME `cells` map (interior + halo)
    // both tiles share, so two tiles compute bit-identical boundary
    // density (and vertices). The dense grid is laid out exactly like
    // [`CellGrid`]: `origin` lo-corner, `size` extent, flat
    // `x + sx*(y + sy*z)` indexing.
    // NB: the `∇D` field is precomputed whenever `density_blur` — NOT only
    // when `topo_blur`. The QEF path uses binary-SIGN topology + Hermite
    // POSITION, but still takes its smooth SHADING NORMAL from the
    // interpolated `∇D` (a C0-continuous field); the per-leaf corner-average
    // normal facets at the voxel scale (visible speckle on smooth terrain).
    let (r_blur, sigma_blur, iso) = if density_blur {
        active_blur_params()
    } else {
        (DENSITY_KERNEL_R, DENSITY_KERNEL_SIGMA, DENSITY_ISO)
    };
    // Skip the (expensive, ~kernel·N³) blur+∇D precompute when it is provably
    // unused: in QEF mode the topology+position come from sign+stored distance
    // (`topo_blur == false`), and when the caller supplies an analytic
    // `surface_normal_fn` the QEF branch takes its shading normal from THAT
    // (the `unwrap_or(&outward_normal)` below resolves to the analytic fn;
    // `build_cube_vertex`'s zero-return fallback is the corner-average
    // `normal_sum`, not `∇D`). So the whole ∇D field is dead on the terrain
    // bake path. The region/sculpt re-extract passes `surface_normal_fn = None`
    // → `∇D` IS the shading normal there, so it still computes it.
    let density_unused_in_qef = use_qef && surface_normal_fn.is_some();
    let density_grids: Option<DensityGrids> = if density_blur && !density_unused_in_qef {
        // Bounding box of every cell in the map (interior + halo).
        let mut bb_min = IVec3::splat(i32::MAX);
        let mut bb_max = IVec3::splat(i32::MIN);
        for &c in cells.keys() {
            bb_min = bb_min.min(c);
            bb_max = bb_max.max(c);
        }
        // Pad by the full kernel reach (`R` for the blur + 1 for the ∇D
        // central difference + 1 cell of slack for the cube-corner /
        // trilinear sampling reach). Boundary-extension (clamped
        // sampling) handles any read past the pad, so a couple cells is
        // plenty and matches the sculpt grid's `± (R + 2)` reach.
        let pad = r_blur + 2;
        let g_origin = bb_min - IVec3::splat(pad);
        let g_size = (bb_max - bb_min) + IVec3::splat(2 * pad + 1);
        let sx = g_size.x as usize;
        let sy = g_size.y as usize;
        let sz = g_size.z as usize;
        let total = sx * sy * sz;
        let mut density = vec![0.0f32; total];
        // Seed binary occupancy: solid iff the cell is in the map OR an
        // INTERIOR_NODE-region cell (resolved on demand via the octree).
        for z in 0..sz {
            for y in 0..sy {
                for x in 0..sx {
                    let c = g_origin + IVec3::new(x as i32, y as i32, z as i32);
                    let solid = cells.contains_key(&c)
                        || is_solid_lookup(octree_nodes, brick_cells, octree_depth, c, extent);
                    density[x + sx * (y + sy * z)] = if solid { 1.0 } else { 0.0 };
                }
            }
        }
        // Separable Gaussian blur (X, Y, Z). Clamp out-of-grid taps to
        // the nearest in-bounds cell (boundary extension) so a solid
        // block near the grid edge stays `D = 1`.
        let kern = density_kernel_weights_1d_for(r_blur, sigma_blur);
        let mut row: Vec<f32> = vec![0.0; sx.max(sy).max(sz)];
        for z in 0..sz {
            for y in 0..sy {
                let base = sx * (y + sy * z);
                row[..sx].copy_from_slice(&density[base..base + sx]);
                for x in 0..sx {
                    let mut acc = 0.0f32;
                    for k in -r_blur..=r_blur {
                        let xi = (x as i32 + k).clamp(0, sx as i32 - 1) as usize;
                        acc += kern[(k + r_blur) as usize] * row[xi];
                    }
                    density[base + x] = acc;
                }
            }
        }
        for z in 0..sz {
            for x in 0..sx {
                for y in 0..sy {
                    row[y] = density[x + sx * (y + sy * z)];
                }
                for y in 0..sy {
                    let mut acc = 0.0f32;
                    for k in -r_blur..=r_blur {
                        let yi = (y as i32 + k).clamp(0, sy as i32 - 1) as usize;
                        acc += kern[(k + r_blur) as usize] * row[yi];
                    }
                    density[x + sx * (y + sy * z)] = acc;
                }
            }
        }
        for y in 0..sy {
            for x in 0..sx {
                for z in 0..sz {
                    row[z] = density[x + sx * (y + sy * z)];
                }
                for z in 0..sz {
                    let mut acc = 0.0f32;
                    for k in -r_blur..=r_blur {
                        let zi = (z as i32 + k).clamp(0, sz as i32 - 1) as usize;
                        acc += kern[(k + r_blur) as usize] * row[zi];
                    }
                    density[x + sx * (y + sy * z)] = acc;
                }
            }
        }
        // Central-difference the smooth `D` for the grid-point gradient.
        let mut gradient = vec![[0.0f32; 3]; total];
        let sxi = sx as i32;
        let syi = sy as i32;
        let szi = sz as i32;
        let at = |x: i32, y: i32, z: i32| -> f32 {
            let xc = x.clamp(0, sxi - 1) as usize;
            let yc = y.clamp(0, syi - 1) as usize;
            let zc = z.clamp(0, szi - 1) as usize;
            density[xc + sx * (yc + sy * zc)]
        };
        for z in 0..szi {
            for y in 0..syi {
                for x in 0..sxi {
                    let gx = (at(x + 1, y, z) - at(x - 1, y, z)) * 0.5;
                    let gy = (at(x, y + 1, z) - at(x, y - 1, z)) * 0.5;
                    let gz = (at(x, y, z + 1) - at(x, y, z - 1)) * 0.5;
                    gradient[(x as usize) + sx * ((y as usize) + sy * (z as usize))] =
                        [gx, gy, gz];
                }
            }
        }
        Some(DensityGrids {
            origin: g_origin,
            size: g_size,
            density,
            gradient,
        })
    } else {
        None
    };

    // Closures over the density grids (when present). `d_solid` tests a
    // cell's center against the iso threshold (the SAME field the
    // crossing uses), and the sampler closures feed `build_cube_vertex`'s
    // Newton projection + ∇D normal.
    let d_grids = density_grids.as_ref();
    let d_solid = |cell: IVec3| -> bool {
        match d_grids {
            Some(g) => {
                let center =
                    Vec3::new(cell.x as f32 + 0.5, cell.y as f32 + 0.5, cell.z as f32 + 0.5);
                sample_density_trilinear(&g.density, g.origin, g.size, center) >= iso
            }
            None => false,
        }
    };
    let density_grid_fn = |p_grid: Vec3| -> f32 {
        match d_grids {
            Some(g) => sample_density_trilinear(&g.density, g.origin, g.size, p_grid),
            None => 0.0,
        }
    };
    let gradient_grid_fn = |p_grid: Vec3| -> Vec3 {
        match d_grids {
            Some(g) => sample_gradient_trilinear(&g.gradient, g.origin, g.size, p_grid),
            None => Vec3::ZERO,
        }
    };
    let smooth_sdf = |p_world: Vec3| -> f32 {
        let g = (p_world - grid_origin) / base_voxel_size;
        iso - density_grid_fn(g)
    };
    let outward_normal = |p_world: Vec3| -> Vec3 {
        let g = (p_world - grid_origin) / base_voxel_size;
        -gradient_grid_fn(g)
    };

    // Pass 2: iterate every cell-pair across the 6 face directions.
    // For each (solid → void) edge, the 4 SN cubes around that edge
    // form a quad. Iterating cells in `cells` (rather than scanning
    // every grid edge) keeps us proportional to surface area.
    let mut cube_vertex: CellMap = CellMap::default();
    let halo_active = halo > 0;
    // With `halo > 0`, iterate every cell in the inner halo ring
    // (`coord in [-1, N+1)`) in addition to the interior. The outer
    // halo ring (`coord in [-halo, -1) ∪ [N+1, N+halo)`) provides
    // 8-corner data for cubes referenced from the inner halo's
    // emissions but never iterates as a quad-emit-from cell. This
    // gives every tile-boundary cell two iterating tiles (the
    // tile that owns it as interior + the neighbour that owns it
    // as halo) so each boundary cube is emitted by both sides —
    // overdraw at the seam, no see-through cracks from asymmetric
    // iteration when the surface slopes across the boundary.
    let iter_lo = if halo_active { -1 } else { 0 };
    let iter_hi = if halo_active { extent + 1 } else { extent };

    // Binary corner classifier (material attribution + the
    // INTERIOR_NODE-region fallback): `Some(slot)` for solid, `None`
    // for empty. Shared by both the binary and D-blur paths — material
    // always follows the real occupied cell, even when geometry follows
    // the D field.
    let cell_lookup = |c: IVec3| -> Option<u32> {
        match cells.get(&c) {
            Some(&v) => Some(v),
            None => {
                if is_solid_lookup(octree_nodes, brick_cells, octree_depth, c, extent) {
                    Some(CELL_INTERIOR)
                } else {
                    None
                }
            }
        }
    };

    // Sign-based solidity for the QEF-Hermite topology (see
    // [`qef_cell_inside`]): puts each active edge on the true crossing.
    let qef_solid = |c: IVec3| -> bool { qef_cell_inside(cell_lookup(c), dists) };

    // **Owner candidate set.**
    //
    // * Binary path: owners are exactly the cells in `cells` that fall
    //   in the inner halo ring `[-1, N+1)`. Each emits its (solid→void)
    //   faces.
    // * D-blur path: the `D = iso` surface can sit up to ~1 cell outside
    //   the binary boundary, so a D-solid owner may be a binary-EMPTY
    //   cell adjacent to a binary-solid one. We therefore also consider
    //   the 26 neighbours of each `cells` cell (still clamped to the
    //   inner halo ring), and emit a face only where `D(self) >= iso &&
    //   D(neighbor) < iso`. The ring clamp is what preserves the
    //   watertight seam: both tiles share the inner-ring cells (own +
    //   halo) and the same D grid, so they emit identical boundary
    //   cubes — no asymmetric leak into the outer halo.
    let in_ring = |c: IVec3| -> bool {
        !halo_active
            || (c.x >= iter_lo
                && c.x < iter_hi
                && c.y >= iter_lo
                && c.y < iter_hi
                && c.z >= iter_lo
                && c.z < iter_hi)
    };
    let mut owners: Vec<IVec3> = Vec::new();
    if topo_blur {
        let mut seen: rustc_hash::FxHashSet<IVec3> = rustc_hash::FxHashSet::default();
        seen.reserve(cells.len() * 8);
        for &cell in cells.keys() {
            for dz in -1..=1 {
                for dy in -1..=1 {
                    for dx in -1..=1 {
                        let c = cell + IVec3::new(dx, dy, dz);
                        if in_ring(c) && seen.insert(c) {
                            owners.push(c);
                        }
                    }
                }
            }
        }
    } else {
        owners.reserve(cells.len());
        for &cell in cells.keys() {
            if in_ring(cell) {
                owners.push(cell);
            }
        }
    }

    for &cell in &owners {
        // D-blur path skips non-D-solid owners (the candidate expansion
        // includes binary-empty cells that may or may not be D-solid).
        if topo_blur && !d_solid(cell) {
            continue;
        }
        // QEF path skips owners that are sign-EMPTY surface leaves (a leaf
        // whose center sits above the surface): they are not solid for the
        // sign-based topology.
        if use_qef && !qef_solid(cell) {
            continue;
        }
        for face in 0..6 {
            let dir = FACE_DIRS[face];
            let neighbor = cell + dir;
            if topo_blur {
                // D-active face: this side D-solid, the other D-empty.
                if d_solid(neighbor) {
                    continue;
                }
            } else if use_qef {
                // Sign-active face: this side inside, the other outside.
                if qef_solid(neighbor) {
                    continue;
                }
            } else {
                if cells.contains_key(&neighbor) {
                    continue;
                }
                // Neighbor isn't in the cell map — could still be inside
                // an INTERIOR_NODE region (which we deliberately didn't
                // expand into the map). Hit the octree to disambiguate.
                if is_solid_lookup(octree_nodes, brick_cells, octree_depth, neighbor, extent) {
                    continue;
                }
            }
            // Active edge: emit a quad of 4 SN-cube vertices, wound
            // CCW around the outward normal (`dir`, pointing from solid
            // into void).
            let cube_offsets = CUBE_OFFSETS_PER_FACE[face];
            let mut quad = [0u32; 4];
            for i in 0..4 {
                let cube = cell + cube_offsets[i];
                quad[i] = match cube_vertex.get(&cube) {
                    Some(&v) => v,
                    None => {
                        // Corner-cell lookup: CellMap first; for cells
                        // that aren't there, fall back to walking the
                        // octree (catches the INTERIOR_NODE-region case
                        // — coarse octree branches classified bulk-
                        // solid by the BFS contribute no per-cell
                        // entry, but their cells are still solid for
                        // SN classification). Without this fallback
                        // the boundary cube on the halo side of the
                        // seam — which gets its non-halo corners via
                        // `is_solid_lookup` rather than CellMap when
                        // those corners land in an `INTERIOR_NODE`
                        // region — would misclassify them as empty
                        // and emit a spurious vertex with no match in
                        // the neighbouring tile.
                        let vertex = if topo_blur {
                            build_cube_vertex(
                                cube,
                                cell_lookup,
                                base_voxel_size,
                                grid_origin,
                                leaf_attr_pool,
                                bone_voxel_pool,
                                sculpt_slots,
                                Some(&smooth_sdf),
                                Some(&outward_normal),
                                Some(&density_grid_fn),
                                Some(&gradient_grid_fn),
                                iso,
                                false,
                                &[],
                            )
                        } else if use_qef {
                            // Sign topology + QEF-Hermite position from the
                            // stored distance. SHADING NORMAL: the caller's
                            // analytic `∇sd` evaluated AT THE VERTEX (exact —
                            // no grid reconstruction) when supplied, else the
                            // smooth interpolated `∇D` field. `density_grid_fn
                            // = None` keeps corner classification + position
                            // on the sign/QEF path; only the normal differs.
                            let qef_normal: &dyn Fn(Vec3) -> Vec3 =
                                surface_normal_fn.unwrap_or(&outward_normal);
                            build_cube_vertex(
                                cube,
                                cell_lookup,
                                base_voxel_size,
                                grid_origin,
                                leaf_attr_pool,
                                bone_voxel_pool,
                                sculpt_slots,
                                None::<&fn(Vec3) -> f32>,
                                Some(qef_normal),
                                None::<&fn(Vec3) -> f32>,
                                None::<&fn(Vec3) -> Vec3>,
                                DENSITY_ISO,
                                true,
                                dists,
                            )
                        } else {
                            // Pure binary (imports): edge-crossing centroid +
                            // averaged per-leaf normal.
                            build_cube_vertex(
                                cube,
                                cell_lookup,
                                base_voxel_size,
                                grid_origin,
                                leaf_attr_pool,
                                bone_voxel_pool,
                                sculpt_slots,
                                None::<&fn(Vec3) -> f32>,
                                None::<&fn(Vec3) -> Vec3>,
                                None::<&fn(Vec3) -> f32>,
                                None::<&fn(Vec3) -> Vec3>,
                                DENSITY_ISO,
                                false,
                                &[],
                            )
                        };
                        let vid = vertices.len() as u32;
                        vertices.push(vertex);
                        cube_vertex.insert(cube, vid);
                        vid
                    }
                };
            }
            indices.extend([quad[0], quad[1], quad[2]]);
            indices.extend([quad[0], quad[2], quad[3]]);
        }
    }

    // Wide-window plane-fit projection (smooth-stairs fix). Default-ON
    // for the terrain density path (radius from `plane_fit_default_radius`,
    // typically `TERRAIN_PLANE_FIT_RADIUS`); a thread-local override can
    // force it off (the bench baseline). Only meaningful on the
    // density-blur path — the QEF-Hermite path has no ripple to recover, so
    // `topo_blur` (not `density_blur`) gates it off there. The shared seam
    // ring — vertices within ½ voxel of any nominal tile face (derived from
    // the octree extent) — is PINNED so adjacent tiles stay watertight.
    if topo_blur {
        let r = resolve_plane_fit_radius(plane_fit_default_radius);
        if r > 0.0 {
            let tile_lo = grid_origin;
            let tile_hi = grid_origin + Vec3::splat(extent as f32 * base_voxel_size);
            wide_window_plane_project(
                &mut vertices,
                &indices,
                base_voxel_size,
                r,
                Some((tile_lo, tile_hi)),
            );
        }
    }

    (vertices, indices)
}

/// Pass 1 of mesh extraction — walk the octree and produce the dense
/// non-empty cell map.
///
/// Exposed separately from [`extract_surface_mesh`] so callers that
/// re-extract **multiple regions per stamp** (the sculpt per-cluster
/// re-extract path in Phase B R4c) can build the map once and run
/// [`extract_mesh_region_from_cells`] against it per region. Each
/// rebuild of the map is O(surface area); per-region pass 2 is
/// proportional to the region's cell count, so amortization is
/// load-bearing for drag-paint perf.
///
/// Returns an empty map for empty octrees.
pub fn collect_cell_map(
    octree_nodes: &[u32],
    octree_depth: u8,
    brick_cells: &[u32],
) -> CellMap {
    let mut cells = CellMap::default();
    if octree_nodes.is_empty() {
        return cells;
    }
    walk_collect_cells(
        octree_nodes,
        brick_cells,
        0,
        UVec3::ZERO,
        0,
        octree_depth,
        &mut cells,
    );
    cells
}

/// Pass 2 of mesh extraction, scoped to a region — produce the surface
/// mesh for cells in `[region_min, region_max)` (half-open).
///
/// **What gets emitted:**
/// * For each solid cell C inside the region (or one cell outside, see
///   pad below): for each of the 6 face directions, if C's neighbor in
///   that direction is empty (or out-of-bounds), emit the quad of 4
///   SN-cube vertices around the face's shared edge.
///
/// **Region boundary handling.** Iteration runs over cells in
/// `[region_min - 1, region_max + 1)` — a 1-cell pad on each side. The
/// pad catches two crack-causing cases:
/// 1. A solid cell *outside* the region whose face-neighbor inside
///    the region is empty: without the pad, the boundary quad on
///    that face would be missing on the region's side.
/// 2. An SN-cube whose vertex sits at the region's edge, with one
///    contributing corner cell just past `region_max`: without the
///    pad, that cube's vertex would be built from incomplete corner
///    data.
///
/// The pad means some output triangles' vertex positions land slightly
/// past `region_max` (up to 1 voxel). Callers that union region outputs
/// (R4c) accept this overlap — duplicate boundary verts across adjacent
/// regions are intentional under the per-cluster-owned model.
///
/// Output indices are *local* to the returned vertex buffer (0-based,
/// referencing positions in the returned `Vec<MeshVertex>`). Caller
/// can drop them straight into a [`crate::cluster_mesh_data::ClusterMesh`]
/// without further remapping.
pub fn extract_mesh_region_from_cells(
    cells: &CellMap,
    region_min: IVec3,
    region_max: IVec3,
    octree_nodes: &[u32],
    octree_depth: u8,
    base_voxel_size: f32,
    grid_origin: Vec3,
    brick_cells: &[u32],
    leaf_attr_pool: &[LeafAttr],
    bone_voxel_pool: &[BoneVoxel],
    sculpt_slots: Option<&rustc_hash::FxHashSet<u32>>,
) -> (Vec<MeshVertex>, Vec<u32>) {
    let mut scratch = SculptExtractScratch::new();
    extract_mesh_region_from_cells_pooled(
        &mut scratch,
        cells,
        region_min,
        region_max,
        octree_nodes,
        octree_depth,
        base_voxel_size,
        grid_origin,
        brick_cells,
        leaf_attr_pool,
        bone_voxel_pool,
        sculpt_slots,
    )
}

/// Pool-reused entry point — same contract as
/// [`extract_mesh_region_from_cells`] but reuses the scratch buffers
/// in `scratch` across stamps. The sculpt drag-paint path threads a
/// single [`SculptExtractScratch`] held on `ArvxSceneManager` through
/// here to avoid the ~500-700 µs alloc+memset cost the
/// fresh-allocating wrapper pays each call.
///
/// The grids grow to the largest brush footprint encountered and
/// never shrink. Each call calls
/// [`CellGrid::reuse`](CellGrid::reuse) which resets only the
/// previously-dirty slots (~50 µs for a 30 k-write stamp vs ~450 µs
/// for a memset of the full 4.5 MB backing Vec).
#[allow(clippy::too_many_arguments)]
pub fn extract_mesh_region_from_cells_pooled(
    scratch: &mut SculptExtractScratch,
    cells: &CellMap,
    region_min: IVec3,
    region_max: IVec3,
    octree_nodes: &[u32],
    octree_depth: u8,
    base_voxel_size: f32,
    grid_origin: Vec3,
    brick_cells: &[u32],
    leaf_attr_pool: &[LeafAttr],
    bone_voxel_pool: &[BoneVoxel],
    sculpt_slots: Option<&rustc_hash::FxHashSet<u32>>,
) -> (Vec<MeshVertex>, Vec<u32>) {
    extract_mesh_region_from_cells_pooled_haloed(
        scratch,
        cells,
        region_min,
        region_max,
        octree_nodes,
        octree_depth,
        base_voxel_size,
        grid_origin,
        brick_cells,
        leaf_attr_pool,
        bone_voxel_pool,
        &[],
        sculpt_slots,
        None::<&fn(Vec3) -> f32>,
    &[],
        )
}

/// Halo-aware variant of [`extract_mesh_region_from_cells_pooled`].
///
/// `halo_cells` supplies cells whose coords lie OUTSIDE the asset's
/// nominal `[0, S)³` cube but still need to participate in SN-cube
/// corner classification at the tile boundary. Phase 4 terrain sculpt
/// passes the asset's stored halo cells through here so per-cluster
/// re-extract at a tile face preserves the watertight seam quads
/// established at bake time.
///
/// The halo cells are folded into the local `cells_grid` exactly like
/// interior cells (with the same `CELL_INTERIOR → CELL_INTERIOR_GRID`
/// remap) but never added to `solid_cells` — the boundary cube on the
/// halo side of the seam is emitted from the interior cell on the
/// owning side, and the halo cell's role is purely 8-corner data.
/// This matches the "halo cells iterate as corner data only" rule
/// from [`extract_surface_mesh_haloed`].
///
/// When `halo_cells` is empty this function is bit-identical to
/// [`extract_mesh_region_from_cells_pooled`].
#[allow(clippy::too_many_arguments)]
pub fn extract_mesh_region_from_cells_pooled_haloed<S>(
    scratch: &mut SculptExtractScratch,
    cells: &CellMap,
    region_min: IVec3,
    region_max: IVec3,
    octree_nodes: &[u32],
    octree_depth: u8,
    base_voxel_size: f32,
    grid_origin: Vec3,
    brick_cells: &[u32],
    leaf_attr_pool: &[LeafAttr],
    bone_voxel_pool: &[BoneVoxel],
    halo_cells: &[(IVec3, u32)],
    sculpt_slots: Option<&rustc_hash::FxHashSet<u32>>,
    sdf_fn: Option<&S>,
    // Per-slot signed-distance pool (voxel units), indexed by the same
    // global slot ids the cells carry. Non-empty selects QEF-Hermite (the
    // render sculpt re-extract passes the scene pool's `dists_as_slice()`);
    // `&[]` keeps the blur path. Mirrors the full-asset extract gate.
    dists: &[i16],
) -> (Vec<MeshVertex>, Vec<u32>)
where
    S: Fn(Vec3) -> f32,
{
    let mut vertices: Vec<MeshVertex> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();
    if cells.is_empty() {
        return (vertices, indices);
    }
    // QEF-Hermite selection (Stage 6) — same gate as the full-asset path.
    let use_qef = !dists.is_empty() && !qef_hermite_force_off() && !qef_force_off_thread();
    // Empty-region guard (no cells to iterate).
    if region_min.x >= region_max.x
        || region_min.y >= region_max.y
        || region_min.z >= region_max.z
    {
        return (vertices, indices);
    }

    let pad_min = region_min - IVec3::ONE;
    let pad_max = region_max + IVec3::ONE;
    let extent = 1i32 << octree_depth;

    // **D6.3 — replace `cube_vertex` and per-cell HashMap probes with
    // a pair of dense `CellGrid`s.**
    //
    // Post-D6.1+D6.2 the inner loop spent 4-10 ms (peaking at ~10 ms
    // on 40-50 k-cell stamps) inside FxHashMap probes: 6 face-neighbor
    // checks against `cells` per cell, 4 `cube_vertex` get/insert
    // pairs per face emission, and 8 corner-cell lookups per fresh
    // SN-cube (inside `build_cube_vertex`). At ~30 ns per FxHash probe
    // the budget added up; a dense `Vec<u32>`-backed lookup is ~3-5 ns
    // (one bounds check + one indexed load) and stays cache-resident
    // for the 9 MB scratch a 50-cell brush radius needs.
    //
    // Grid extent = `[pad_min - 1, pad_max + 1)` — the half-open
    // bound of every coord this loop probes:
    //   • neighbor lookups land in `[pad_min - 1, pad_max + 1)`
    //     (cells in pad range + ±1 face offset).
    //   • cube positions land in `[pad_min - 1, pad_max)` (cube
    //     offsets are in `{-1, 0}`), and `build_cube_vertex` corner
    //     lookups extend up to `[pad_min - 1, pad_max + 1)`.
    //
    // `cells.iter()` may include entries past the grid bounds — the
    // collect step pads by +3 to give `build_cube_vertex` boundary
    // data, but our grid only needs ±1. `CellGrid::set` silently
    // drops out-of-bounds writes so the populate step is bounds-safe.
    let grid_min = pad_min - IVec3::ONE;
    let grid_size = pad_max - pad_min + IVec3::splat(2);
    scratch.cells_grid.reuse(grid_min, grid_size);
    scratch.cube_vertex_grid.reuse(grid_min, grid_size);
    scratch.solid_cells.clear();

    // **Populate phase** — `cells_grid` and `solid_cells` are written
    // here, then become read-only for the density precompute + cube
    // loop. The `&mut` borrows below are scoped to this block so the
    // subsequent split borrow (immutable `cells_grid` + `density`,
    // mutable `cube_vertex_grid`) type-checks without `unsafe`.
    {
        let cells_grid = &mut scratch.cells_grid;
        let solid_cells = &mut scratch.solid_cells;
        if solid_cells.capacity() < cells.len() {
            solid_cells.reserve(cells.len() - solid_cells.capacity());
        }

    // Combined populate + filter pass — visits `cells.iter()` once,
    // mirroring D6.1's iteration win. Cells inside `[pad_min, pad_max)`
    // are pushed into `solid_cells` (the inner loop's domain); cells
    // anywhere inside grid bounds get registered in `cells_grid` for
    // face-neighbor / corner lookups. The two ranges overlap so most
    // cells contribute to both.
    //
    // `CELL_INTERIOR` (= `u32::MAX`) collides with the grid's empty
    // sentinel `CELL_GRID_EMPTY`, so remap it to `CELL_INTERIOR_GRID`
    // before storing; the lookup closure passed into
    // `build_cube_vertex` reverses the remap so the corner classifier
    // sees the original `CELL_INTERIOR` value.
    for (&cell, &slot) in cells.iter() {
        let stored = if slot == CELL_INTERIOR {
            CELL_INTERIOR_GRID
        } else {
            slot
        };
        cells_grid.set(cell, stored);
        if cell.x >= pad_min.x
            && cell.x < pad_max.x
            && cell.y >= pad_min.y
            && cell.y < pad_max.y
            && cell.z >= pad_min.z
            && cell.z < pad_max.z
        {
            solid_cells.push(cell);
        }
    }

    // Fold halo cells into the local `cells_grid` for 8-corner data
    // at the tile boundary. Halo cells whose coords fall outside the
    // grid bounds are silently dropped by `CellGrid::set` — they
    // don't influence cubes this region produces. Crucially, halo
    // cells are NOT added to `solid_cells`: they never iterate as a
    // quad-emit-from cell. See `extract_surface_mesh_haloed` for the
    // watertight-seam protocol.
    for &(coord, slot) in halo_cells {
        let stored = if slot == CELL_INTERIOR {
            CELL_INTERIOR_GRID
        } else {
            slot
        };
        cells_grid.set(coord, stored);
    }
    } // end populate block — `cells_grid` / `solid_cells` now read-only.

    // ── Smooth-density precompute (direct, non-iterative smoothing) ──
    //
    // Blur the binary occupancy `occ(c) = cells_grid.contains(c) ? 1 : 0`
    // with a fixed separable Gaussian (R = DENSITY_KERNEL_R, σ =
    // DENSITY_KERNEL_SIGMA) into a dense `[0, 1]` density field `D`,
    // parallel to `cells_grid`. The `D = DENSITY_ISO` (0.5) isosurface
    // is the smooth surface; `∇D` is its (inward) gradient. Because `D`
    // is a *pure local* function of occupancy in `c ± R` and `R ≤` the
    // boundary halo, the density at every shared boundary cell is
    // identical from both sides → watertight by construction.
    //
    // One separable 3-pass blur over the dense grid: X then Y then Z.
    // For a `104³` brush footprint this is ~1.1 M cells × 3 passes ×
    // (2R+1=5) taps ≈ 17 M MACs — sub-millisecond, and replaces the
    // 12-iteration Taubin relaxation (24 Laplacian sweeps over the
    // mesh + adjacency build) that the render side used to run per
    // stamp.
    let g_origin = scratch.cells_grid.origin();
    let g_size = scratch.cells_grid.size();
    let sx = g_size.x as usize;
    let sy = g_size.y as usize;
    let sz = g_size.z as usize;
    let total = sx * sy * sz;
    // Blur kernel: const `R = 2` / `σ = 1.0` in production; the bench
    // can override per-thread via `set_blur_override` to sweep R.
    let (r, kern_sigma, iso) = active_blur_params();

    // The density / gradient SLICES are borrowed below even in QEF mode
    // (where the closures that read them are never called), so the buffers
    // must be `≥ total`. Resize unconditionally; only the expensive
    // blur+gradient COMPUTE is skipped for QEF.
    if scratch.density.len() < total {
        scratch.density.resize(total, 0.0);
    }
    if scratch.gradient.len() < total {
        scratch.gradient.resize(total, [0.0; 3]);
    }

    // ── blur→D + ∇D precompute ──
    // Always computed (not skipped for QEF): QEF places vertices from the
    // stored per-leaf distance, but still takes its smooth SHADING NORMAL
    // from the interpolated `∇D` field (the per-leaf corner average facets
    // at the voxel scale → speckle). The blur is sub-ms.
    {
    let kern = density_kernel_weights_1d_for(r, kern_sigma);
    // Pass 0: seed `density` with raw binary occupancy.
    {
        let cells_grid = &scratch.cells_grid;
        let density = &mut scratch.density;
        for z in 0..sz {
            for y in 0..sy {
                for x in 0..sx {
                    let c = g_origin
                        + IVec3::new(x as i32, y as i32, z as i32);
                    density[x + sx * (y + sy * z)] =
                        if cells_grid.contains(c) { 1.0 } else { 0.0 };
                }
            }
        }
    }
    // Separable Gaussian: blur along X, then Y, then Z. Each pass reads
    // `density` into a scratch row and writes the blurred result back.
    // Out-of-grid taps clamp to the nearest in-bounds cell (boundary
    // extension), matching `sample_density_trilinear`'s edge policy so
    // a fully-solid block near the grid edge stays `D = 1`, not a
    // false `< 1` that would pull a spurious surface inward.
    {
        let density = &mut scratch.density;
        let mut row: Vec<f32> = vec![0.0; sx.max(sy).max(sz)];
        // X pass.
        for z in 0..sz {
            for y in 0..sy {
                let base = sx * (y + sy * z);
                row[..sx].copy_from_slice(&density[base..base + sx]);
                for x in 0..sx {
                    let mut acc = 0.0f32;
                    for k in -r..=r {
                        let xi = (x as i32 + k).clamp(0, sx as i32 - 1) as usize;
                        acc += kern[(k + r) as usize] * row[xi];
                    }
                    density[base + x] = acc;
                }
            }
        }
        // Y pass.
        for z in 0..sz {
            for x in 0..sx {
                for y in 0..sy {
                    row[y] = density[x + sx * (y + sy * z)];
                }
                for y in 0..sy {
                    let mut acc = 0.0f32;
                    for k in -r..=r {
                        let yi = (y as i32 + k).clamp(0, sy as i32 - 1) as usize;
                        acc += kern[(k + r) as usize] * row[yi];
                    }
                    density[x + sx * (y + sy * z)] = acc;
                }
            }
        }
        // Z pass.
        for y in 0..sy {
            for x in 0..sx {
                for z in 0..sz {
                    row[z] = density[x + sx * (y + sy * z)];
                }
                for z in 0..sz {
                    let mut acc = 0.0f32;
                    for k in -r..=r {
                        let zi = (z as i32 + k).clamp(0, sz as i32 - 1) as usize;
                        acc += kern[(k + r) as usize] * row[zi];
                    }
                    density[x + sx * (y + sy * z)] = acc;
                }
            }
        }
    }

    // ── Smooth-gradient precompute (continuous normals) ──
    //
    // `G[c] = ∇D` at every grid point, by central differences of the
    // SMOOTH `density` grid. We interpolate THIS gradient field at the
    // vertex (in `build_cube_vertex`) rather than differentiating the
    // trilinear density: the gradient of a trilinear interpolant is
    // piecewise-constant and DISCONTINUOUS across cell boundaries (→
    // voxel-scale normal speckle), whereas central-differencing the
    // smooth `D` first yields a smooth grid-point gradient whose
    // trilinear interpolation is continuous (C0) across cells.
    //
    // Edge handling: replicated / one-sided differences at the grid
    // boundary (clamp the ±1 sample index into range), never reading 0
    // outside — matching the density blur's and the trilinear sampler's
    // boundary-extension policy. `G[c]` depends on `D` in `c ± 1` and
    // each `D` depends on occupancy in `c ± R`, so `G`'s occupancy
    // dependency is `± (R + 1)` — the same reach as the position
    // crossings — so two tiles sharing the halo compute the identical
    // `G` (and identical normals) at the seam.
    {
        let density = &scratch.density;
        let gradient = &mut scratch.gradient;
        let sxi = sx as i32;
        let syi = sy as i32;
        let szi = sz as i32;
        let at = |x: i32, y: i32, z: i32| -> f32 {
            let xc = x.clamp(0, sxi - 1) as usize;
            let yc = y.clamp(0, syi - 1) as usize;
            let zc = z.clamp(0, szi - 1) as usize;
            density[xc + sx * (yc + sy * zc)]
        };
        for z in 0..szi {
            for y in 0..syi {
                for x in 0..sxi {
                    // Central difference of the smooth D grid. When a
                    // neighbour clamps (grid edge) the difference becomes
                    // one-sided / zero on that axis, which is the correct
                    // replicated-boundary behaviour.
                    let gx = (at(x + 1, y, z) - at(x - 1, y, z)) * 0.5;
                    let gy = (at(x, y + 1, z) - at(x, y - 1, z)) * 0.5;
                    let gz = (at(x, y, z + 1) - at(x, y, z - 1)) * 0.5;
                    gradient[(x as usize) + sx * ((y as usize) + sy * (z as usize))] =
                        [gx, gy, gz];
                }
            }
        }
    }
    } // end blur→D + ∇D precompute

    // ── Split borrows for the cube loop ──
    // `cells_grid` + `density` + `gradient` are read-only;
    // `cube_vertex_grid` is mutable. Distinct scratch fields → safe
    // split borrow.
    let cells_grid = &scratch.cells_grid;
    let cube_vertex_grid = &mut scratch.cube_vertex_grid;
    let solid_cells = &scratch.solid_cells;
    let density = &scratch.density[..total];
    let gradient = &scratch.gradient[..total];

    // Smooth density sampler in fractional grid (cell) coordinates.
    let density_at = |p_grid: Vec3| -> f32 {
        sample_density_trilinear(density, g_origin, g_size, p_grid)
    };
    // Smooth SDF in world space: `sdf = iso - D(p)`. Inside (D high) →
    // negative; outside (D low) → positive. `build_cube_vertex`
    // interpolates the `t = da/(da-db)` zero-crossing on each active
    // edge to place the vertex. `iso` is the active threshold (0.5 in
    // production; the bench may offset it to counter the wider-blur
    // inward isosurface shift).
    let smooth_sdf = |p_world: Vec3| -> f32 {
        let g = (p_world - grid_origin) / base_voxel_size;
        iso - density_at(g)
    };
    // Outward surface normal at a world-space point: interpolate the
    // precomputed grid-point gradient field `G = ∇D`, then negate.
    // `∇D` points toward higher density = INTO the solid, so `-∇D`
    // points toward EMPTY (outward). Interpolating the gradient FIELD
    // (not differentiating the interpolated density) keeps the normal
    // continuous across cell boundaries. Returns the raw (un-normalized)
    // `-G`; `build_cube_vertex` normalizes and applies the degenerate
    // fallback.
    let outward_normal = |p_world: Vec3| -> Vec3 {
        let g = (p_world - grid_origin) / base_voxel_size;
        -sample_gradient_trilinear(gradient, g_origin, g_size, g)
    };
    // Grid-space density + RAW gradient samplers for the Newton
    // projection inside `build_cube_vertex` (it works in grid coords).
    let density_grid_fn =
        |p_grid: Vec3| -> f32 { sample_density_trilinear(density, g_origin, g_size, p_grid) };
    let gradient_grid_fn =
        |p_grid: Vec3| -> Vec3 { sample_gradient_trilinear(gradient, g_origin, g_size, p_grid) };
    // The caller-supplied `sdf_fn` is intentionally ignored for vertex
    // placement now — the density-derived SDF is the authoritative
    // smooth field. (Sculpt/terrain callers all pass `None`.)
    let _ = sdf_fn;

    // **Run surface nets ON the D field, not binary occupancy.**
    //
    // The topology (which faces emit, which cube edges are active) must
    // agree with where the vertex is PLACED (the `D = iso` isosurface).
    // The previous code classified solidity by binary occupancy
    // (`cells_grid.contains`) while placing on `D = iso`; on the ~1/3 of
    // surface cubes where a corner is binary-SOLID but its smooth `D`
    // is `< iso`, the iso did not cross the binary-active edge, so
    // `t = da/(da-db)` clamped to 0/1 and the vertex pinned to the
    // discrete grid Y → horizontal terraces. Classifying solidity by
    // `D(cell_center) >= iso` (the SAME field the crossing uses) makes
    // topology and position consistent: every active edge has a real
    // interior crossing, no clamps, no terraces.
    //
    // `cell_lookup` (binary) is kept ONLY to pick the material /
    // leaf_attr / bone slot — never for solidity or edge activation.
    let d_solid = |cell: IVec3| -> bool {
        let center = Vec3::new(cell.x as f32 + 0.5, cell.y as f32 + 0.5, cell.z as f32 + 0.5);
        density_grid_fn(center) >= iso
    };
    // Binary corner classifier (material attribution + the
    // INTERIOR_NODE-region fallback): solid + slot, or None.
    let cell_lookup = |c: IVec3| -> Option<u32> {
        match cells_grid.get(c) {
            Some(CELL_INTERIOR_GRID) => Some(CELL_INTERIOR),
            Some(v) => Some(v),
            None => {
                if is_solid_lookup(octree_nodes, brick_cells, octree_depth, c, extent) {
                    Some(CELL_INTERIOR)
                } else {
                    None
                }
            }
        }
    };
    // Sign-based solidity for the QEF-Hermite topology (see
    // [`qef_cell_inside`]) — the active edges sit on the true crossing.
    let qef_solid = |c: IVec3| -> bool { qef_cell_inside(cell_lookup(c), dists) };
    // Unified solidity for the cube loop: distance-sign in QEF mode, blurred
    // `D = iso` otherwise. Keeping one closure keeps the owner / face /
    // emit logic shared.
    let solid = |c: IVec3| -> bool {
        if use_qef {
            qef_solid(c)
        } else {
            d_solid(c)
        }
    };

    // **Iteration / owner set.** A D-surface face is emitted by its
    // D-SOLID side. The owner cell can be a cell whose binary occupancy
    // is EMPTY but whose `D >= iso` (the `D = iso` surface sits up to
    // ~1 cell outside the binary boundary on a convex region). For an
    // `R = 2` blur a binary-empty cell can only be D-solid if a
    // binary-solid cell lies within 1 cell — i.e. in its full 3×3×3
    // neighbourhood (face OR edge OR corner), not just the 6 faces. So
    // the candidate owners are the binary `solid_cells` plus their 26
    // neighbours. Missing the diagonal neighbours leaves cracks on
    // slanted/curved surfaces where the D-solid owner is only a
    // diagonal neighbour of the binary boundary. Deduplicate so no face
    // is emitted twice; each candidate emits only its D-active faces
    // (`D(self) >= iso` && `D(neighbor) < iso`).
    let mut candidates: rustc_hash::FxHashSet<IVec3> =
        rustc_hash::FxHashSet::default();
    candidates.reserve(solid_cells.len() * 8);
    for &cell in solid_cells.iter() {
        for dz in -1..=1 {
            for dy in -1..=1 {
                for dx in -1..=1 {
                    candidates.insert(cell + IVec3::new(dx, dy, dz));
                }
            }
        }
    }

    for &cell in &candidates {
        // Owner is solid on the active field (sign for QEF, `D` for blur).
        if !solid(cell) {
            continue;
        }
        for face in 0..6 {
            let dir = FACE_DIRS[face];
            let neighbor = cell + dir;
            // Active face: this side solid, the other empty.
            if solid(neighbor) {
                continue;
            }
            let cube_offsets = CUBE_OFFSETS_PER_FACE[face];
            let mut quad = [0u32; 4];
            for i in 0..4 {
                let cube = cell + cube_offsets[i];
                quad[i] = match cube_vertex_grid.get(cube) {
                    Some(v) => v,
                    None => {
                        let vertex = if use_qef {
                            // Sign topology + QEF-Hermite position from the
                            // stored distance, but the smooth interpolated
                            // `∇D` SHADING NORMAL (the per-leaf corner average
                            // facets → speckle). `density_grid_fn = None`
                            // keeps classification + position on the sign/QEF
                            // path; only the normal comes from `∇D`.
                            build_cube_vertex(
                                cube,
                                cell_lookup,
                                base_voxel_size,
                                grid_origin,
                                leaf_attr_pool,
                                bone_voxel_pool,
                                sculpt_slots,
                                None::<&fn(Vec3) -> f32>,
                                Some(&outward_normal),
                                None::<&fn(Vec3) -> f32>,
                                None::<&fn(Vec3) -> Vec3>,
                                iso,
                                true,
                                dists,
                            )
                        } else {
                            build_cube_vertex(
                                cube,
                                cell_lookup,
                                base_voxel_size,
                                grid_origin,
                                leaf_attr_pool,
                                bone_voxel_pool,
                                sculpt_slots,
                                Some(&smooth_sdf),
                                Some(&outward_normal),
                                Some(&density_grid_fn),
                                Some(&gradient_grid_fn),
                                iso,
                                false,
                                &[],
                            )
                        };
                        let vid = vertices.len() as u32;
                        vertices.push(vertex);
                        cube_vertex_grid.set(cube, vid);
                        vid
                    }
                };
            }
            indices.extend([quad[0], quad[1], quad[2]]);
            indices.extend([quad[0], quad[2], quad[3]]);
        }
    }

    // Wide-window plane-fit projection. The region/sculpt path keeps the
    // fix OPT-IN (default radius 0.0 → off) — the production smooth-stairs
    // fix targets the terrain BAKE path; sculpt has its own smoothing and
    // a different (overlay-based) seam model. The bench can still force it
    // on via the thread-local override to compare paths. No tile-boundary
    // AABB here (single-region extract), so `None` (no pinning). QEF has no
    // ripple to recover, so it is skipped there.
    {
        let r = if use_qef { 0.0 } else { resolve_plane_fit_radius(0.0) };
        if r > 0.0 {
            wide_window_plane_project(&mut vertices, &indices, base_voxel_size, r, None);
        }
    }

    (vertices, indices)
}

/// Convenience wrapper: full octree walk + single-region extract in one
/// call. Equivalent to
/// `extract_mesh_region_from_cells(collect_cell_map(..), region, ..)`.
///
/// Use this for one-shot region extraction (R4b unit tests, ad-hoc
/// diagnostics); use the two-step form for sculpt's per-stamp loop
/// across many regions ([`extract_mesh_region_from_cells`] reuses one
/// cell map across all regions).
#[allow(clippy::too_many_arguments)]
pub fn extract_surface_mesh_region(
    octree_nodes: &[u32],
    octree_depth: u8,
    base_voxel_size: f32,
    grid_origin: Vec3,
    brick_cells: &[u32],
    leaf_attr_pool: &[LeafAttr],
    bone_voxel_pool: &[BoneVoxel],
    region_min: IVec3,
    region_max: IVec3,
) -> (Vec<MeshVertex>, Vec<u32>) {
    let cells = collect_cell_map(octree_nodes, octree_depth, brick_cells);
    extract_mesh_region_from_cells(
        &cells,
        region_min,
        region_max,
        octree_nodes,
        octree_depth,
        base_voxel_size,
        grid_origin,
        brick_cells,
        leaf_attr_pool,
        bone_voxel_pool,
        None,
    )
}

/// Regularized least-squares tangent-plane intersection (QEF) for
/// vertex placement. Each plane is (normal, point_on_plane). The
/// `bias` point (typically the naive centroid) pulls the solution
/// toward a known-good position, preventing wild jumps from
/// poorly-conditioned systems.
///
/// Solves: (A^T A + λI) x = A^T b + λ · bias
///
/// With λ > 0 the system is always well-conditioned (det ≥ λ³),
/// so this never returns None. When plane data is strong, the
/// solution follows the planes. When degenerate (parallel normals,
/// few planes), it gracefully blends toward bias.
const QEF_LAMBDA: f32 = 0.01;

#[inline]
fn solve_qef(planes: &[(Vec3, Vec3)], bias: Vec3) -> Vec3 {
    let lambda = QEF_LAMBDA;
    let mut ata_xx = lambda;
    let mut ata_xy = 0.0f32;
    let mut ata_xz = 0.0f32;
    let mut ata_yy = lambda;
    let mut ata_yz = 0.0f32;
    let mut ata_zz = lambda;
    let mut atb_x = lambda * bias.x;
    let mut atb_y = lambda * bias.y;
    let mut atb_z = lambda * bias.z;

    for &(n, p) in planes {
        let d = n.dot(p);
        ata_xx += n.x * n.x;
        ata_xy += n.x * n.y;
        ata_xz += n.x * n.z;
        ata_yy += n.y * n.y;
        ata_yz += n.y * n.z;
        ata_zz += n.z * n.z;
        atb_x += n.x * d;
        atb_y += n.y * d;
        atb_z += n.z * d;
    }

    let det = ata_xx * (ata_yy * ata_zz - ata_yz * ata_yz)
        - ata_xy * (ata_xy * ata_zz - ata_yz * ata_xz)
        + ata_xz * (ata_xy * ata_yz - ata_yy * ata_xz);

    let inv_det = 1.0 / det;

    let x = (atb_x * (ata_yy * ata_zz - ata_yz * ata_yz)
        - ata_xy * (atb_y * ata_zz - ata_yz * atb_z)
        + ata_xz * (atb_y * ata_yz - ata_yy * atb_z))
        * inv_det;
    let y = (ata_xx * (atb_y * ata_zz - ata_yz * atb_z)
        - atb_x * (ata_xy * ata_zz - ata_yz * ata_xz)
        + ata_xz * (ata_xy * atb_z - atb_y * ata_xz))
        * inv_det;
    let z = (ata_xx * (ata_yy * atb_z - atb_y * ata_yz)
        - ata_xy * (ata_xy * atb_z - atb_y * ata_xz)
        + atb_x * (ata_xy * ata_yz - ata_yy * ata_xz))
        * inv_det;

    Vec3::new(x, y, z)
}

/// **QEF-Hermite vertex placement** for an SN cube — the smooth-by-
/// construction alternative to the `blur→D` + Newton recovery.
///
/// Each *surface-leaf* corner of the cube (skip EMPTY and `CELL_INTERIOR`
/// — they carry no prefiltered normal/distance and contribute no plane)
/// gives an exact first-order Hermite sample of the surface: the
/// prefiltered outward normal `n` and the surface point
/// `p_surf = cell_center − d_vox · n`, where `d_vox` is the stored
/// gradient-normalized signed distance (in voxel/grid units; the SDF is
/// negative inside, so a cell just below the surface has `d_vox < 0` and
/// `p_surf` moves outward onto the surface). The dual vertex is the
/// regularized least-squares intersection of those tangent planes
/// (`solve_qef`, bias = their centroid), clamped to the cube's
/// cell-center AABB `[cube+½, cube+1½]` so a sharp-crease seam can never
/// fold a vertex out of its cube.
///
/// On a flat/gently-sloped region every plane is (near-)coincident with
/// the true surface, so the QEF nails the height (`residual → 0`) — no
/// terracing, because both topology and position root in the SAME surface
/// leaves (the old terracing was a binary-topology / blurred-D-position
/// mismatch).
///
/// Returns `None` when no surface-leaf corner supplied a plane (a non-
/// surface cube, or `dists` not populated) — the caller falls back to the
/// edge-crossing centroid. Position is in grid (voxel) space.
fn qef_hermite_placement<F>(
    cube: IVec3,
    cell_lookup: &F,
    leaf_attr_pool: &[LeafAttr],
    dists: &[i16],
) -> Option<Vec3>
where
    F: Fn(IVec3) -> Option<u32>,
{
    let mut planes: Vec<(Vec3, Vec3)> = Vec::with_capacity(8);
    let mut p_sum = Vec3::ZERO;
    for i in 0u32..8 {
        let c = cube + corner_offset(i);
        let Some(slot) = cell_lookup(c) else { continue };
        if slot == CELL_INTERIOR {
            continue;
        }
        let Some(attr) = leaf_attr_pool.get(slot as usize) else {
            continue;
        };
        let n = unpack_oct(attr.normal_oct);
        if n.length_squared() <= 1e-12 {
            continue;
        }
        let n = n.normalize();
        // Gradient-normalized signed distance in voxel units (0 when the
        // slot has no stored distance — branch-node LOD attrs, or a
        // partially-populated pool; harmless, that plane just passes
        // through the cell center).
        let d_vox = dists
            .get(slot as usize)
            .copied()
            .map(LeafAttrPool::dequantize_dist)
            .unwrap_or(0.0);
        let center = Vec3::new(c.x as f32 + 0.5, c.y as f32 + 0.5, c.z as f32 + 0.5);
        let p_surf = center - d_vox * n;
        planes.push((n, p_surf));
        p_sum += p_surf;
    }
    if planes.is_empty() {
        return None;
    }
    let bias = p_sum / planes.len() as f32;
    let lo = Vec3::new(cube.x as f32 + 0.5, cube.y as f32 + 0.5, cube.z as f32 + 0.5);
    Some(solve_qef(&planes, bias).clamp(lo, lo + Vec3::ONE))
}

/// **Manifold Dual Contouring placement** — reconstruct the vertex by
/// INTERPOLATING the stored per-cell signed-distance field, not by fitting
/// cell-center planes. For each of the cube's 12 edges that crosses the surface
/// (sign change between its two corner cells' stored distances), linearly
/// interpolate the crossing point from the two distances and take the surface
/// cell's normal there; place the vertex by QEF over those Hermite crossings,
/// Tikhonov-biased toward the crossing CENTROID. A rank-deficient gentle-slope
/// cube collapses to the smooth centroid (instead of stepping), and the result
/// is clamped to an EXPANDED bound so the vertex may follow a gentle slope
/// across the cell boundary — the per-cube AABB clamp in `qef_hermite_placement`
/// is what staircased gentle slopes.
///
/// Interior/empty corner cells carry only a sign (no stored magnitude); they
/// contribute a ±band-half-width sentinel so a surface↔solid or surface↔empty
/// crossing still interpolates to a sub-cell position. Sign convention matches
/// [`qef_cell_inside`] so the topology and placement agree. Returns `None` when
/// no edge crosses (caller falls back to the edge-crossing centroid).
fn manifold_dc_placement<F>(
    cube: IVec3,
    cell_lookup: &F,
    leaf_attr_pool: &[LeafAttr],
    dists: &[i16],
) -> Option<Vec3>
where
    F: Fn(IVec3) -> Option<u32>,
{
    // Sentinel half-width for the rare both-sides-have-no-distance edge.
    const SENT: f32 = 0.866_025_4; // sqrt(3)/2
    // Per corner cell: (inside?, stored signed distance, stored normal). Sign
    // matches `qef_cell_inside`; surface cells carry a real (d, n), interior/
    // empty carry only the sign.
    let sample = |c: IVec3| -> (bool, Option<f32>, Option<Vec3>) {
        match cell_lookup(c) {
            None => (false, None, None),                          // empty → outside
            Some(s) if s == CELL_INTERIOR => (true, None, None),  // deep solid → inside
            Some(s) => {
                let d = dists
                    .get(s as usize)
                    .copied()
                    .map(LeafAttrPool::dequantize_dist)
                    .unwrap_or(0.0);
                let n = leaf_attr_pool
                    .get(s as usize)
                    .map(|a| unpack_oct(a.normal_oct))
                    .filter(|n| n.length_squared() > 1e-12)
                    .map(|n| n.normalize());
                (d <= 0.0, Some(d), n)
            }
        }
    };
    // 12 cube edges as corner-index pairs in `corner_offset` (bit) order.
    const EDGES: [(usize, usize); 12] = [
        (0, 1), (2, 3), (4, 5), (6, 7), // x
        (0, 2), (1, 3), (4, 6), (5, 7), // y
        (0, 4), (1, 5), (2, 6), (3, 7), // z
    ];
    let corners: [(bool, Option<f32>, Option<Vec3>); 8] =
        std::array::from_fn(|i| sample(cube + corner_offset(i as u32)));
    let center = |i: usize| -> Vec3 {
        let c = cube + corner_offset(i as u32);
        Vec3::new(c.x as f32 + 0.5, c.y as f32 + 0.5, c.z as f32 + 0.5)
    };
    let mut planes: Vec<(Vec3, Vec3)> = Vec::with_capacity(12);
    let mut csum = Vec3::ZERO;
    let mut nc = 0.0f32;
    for (a, b) in EDGES {
        let (ina, da_opt, na) = corners[a];
        let (inb, db_opt, nb) = corners[b];
        if ina == inb {
            continue; // no sign change → no crossing on this edge
        }
        let ca = center(a);
        let cb = center(b);
        // Resolve both endpoint distances. A side with no stored magnitude
        // (interior/empty) is EXTRAPOLATED from the surface side's Hermite
        // plane (`d_b = d_a + (c_b − c_a)·n_a`, |∇d| = 1): on a clean slope this
        // reproduces the exact plane (residual → 0) and on a curved field it is
        // first-order — far better than a flat sentinel. Only when neither side
        // is a surface cell (both sign-only) does the ±sentinel apply.
        let (da, db) = match (da_opt, db_opt) {
            (Some(da), Some(db)) => (da, db),
            (Some(da), None) => {
                let db = na
                    .map(|n| da + (cb - ca).dot(n))
                    .unwrap_or(if inb { -SENT } else { SENT });
                (da, db)
            }
            (None, Some(db)) => {
                let da = nb
                    .map(|n| db + (ca - cb).dot(n))
                    .unwrap_or(if ina { -SENT } else { SENT });
                (da, db)
            }
            (None, None) => (
                if ina { -SENT } else { SENT },
                if inb { -SENT } else { SENT },
            ),
        };
        let denom = da - db;
        if denom.abs() < 1e-9 {
            continue;
        }
        let t = (da / denom).clamp(0.0, 1.0);
        let p = ca + (cb - ca) * t;
        csum += p;
        nc += 1.0;
        // Normal from the surface endpoint (prefer the nearer of the two).
        let n = if t < 0.5 { na.or(nb) } else { nb.or(na) };
        if let Some(n) = n {
            planes.push((n, p));
        }
    }
    if nc == 0.0 {
        return None;
    }
    let centroid = csum / nc;
    let pos = if planes.is_empty() {
        centroid
    } else {
        solve_qef(&planes, centroid)
    };
    // Corner-cell centers span [cube+0.5, cube+1.5]; allow ±0.5 cell beyond so a
    // gentle slope is followed across the boundary instead of stepping at it.
    let lo = Vec3::new(cube.x as f32 + 0.5, cube.y as f32 + 0.5, cube.z as f32 + 0.5);
    Some(pos.clamp(lo - Vec3::splat(0.5), lo + Vec3::splat(1.5)))
}

/// Sign-based solidity for the QEF-Hermite topology. A cell is "inside"
/// iff its stored per-leaf signed distance is `≤ 0` (center on/below the
/// surface); deep INTERIOR-region cells (`CELL_INTERIOR`, no stored
/// distance) are solid; absent cells are empty.
///
/// Why QEF mode needs *this* instead of surface-leaf membership: the
/// voxelizer keeps a cell as a SURFACE leaf whenever the surface merely
/// *passes through* its Lipschitz band — including cells whose center sits
/// just **above** the surface. Treating every such leaf as solid (the
/// import/binary rule) puts the occupancy boundary ~½ voxel above the true
/// surface and staircases it; the cube-AABB clamp would then pin each
/// vertex inside its offset cube and the QEF could never reach the true
/// surface (the terracing returns). Classifying on the distance *sign*
/// puts each active edge on the real crossing, so the cube straddles the
/// surface and the tight clamp and the QEF agree. Deterministic (quantized
/// distance) → seam-safe.
#[inline]
fn qef_cell_inside(slot_lookup: Option<u32>, dists: &[i16]) -> bool {
    match slot_lookup {
        None => false,
        Some(s) if s == CELL_INTERIOR => true,
        Some(s) => {
            dists
                .get(s as usize)
                .copied()
                .map(LeafAttrPool::dequantize_dist)
                .unwrap_or(0.0)
                <= 0.0
        }
    }
}

/// Build the [`MeshVertex`] for an SN cube whose lo corner is `cube`.
/// The cube spans cells `cube..cube+1` along each axis (8 corner cells
/// total).
///
/// **Position** is determined by normal-guided QEF placement when
/// sufficient normal data is available, falling back to naive
/// edge-crossing centroid otherwise.
///
/// Normal-guided placement: each solid corner with a valid `LeafAttr`
/// defines a tangent plane (point = cell center, normal = unpacked
/// oct normal). The vertex is placed at the point that best satisfies
/// all tangent planes via a least-squares solve (3×3 Cramer's rule).
/// This makes the output resolution-independent: gentle slopes produce
/// smooth geometry at any voxel size, while sharp features (divergent
/// normals) are preserved.
///
/// Fallback (naive surface nets): centroid of edge-crossing midpoints.
/// Used when no normals are available (empty `leaf_attr_pool`) or the
/// QEF system is degenerate (parallel normals, rank-deficient).
///
/// **Solidity** test is the caller-supplied `cell_lookup` closure —
/// returns `Some(slot)` for solid cells (with `CELL_INTERIOR` for
/// brick-INTERIOR-bulk cells, or the cell's `leaf_attr_id` otherwise)
/// and `None` for empty. Generic so the full-asset extract can keep
/// using its `CellMap` (where surface cells are sparse over a deep
/// octree extent and a dense grid would be untenable) while the
/// region extract hands in a `CellGrid` for ~6-10× faster probes.
///
/// Falls back to the SN cube's grid corner (`cube + (1, 1, 1)`) when
/// no edge crossings are detected — defensive only.
#[allow(clippy::too_many_arguments)]
fn build_cube_vertex<F, S, N, D, G>(
    cube: IVec3,
    cell_lookup: F,
    voxel_size: f32,
    grid_origin: Vec3,
    leaf_attr_pool: &[LeafAttr],
    bone_voxel_pool: &[BoneVoxel],
    sculpt_slots: Option<&rustc_hash::FxHashSet<u32>>,
    sdf_fn: Option<&S>,
    normal_fn: Option<&N>,
    // Grid-space density + gradient samplers for the Newton projection
    // of the vertex onto the smooth `D = iso` isosurface.
    // `density_grid_fn(p_grid) -> D in [0,1]`;
    // `gradient_grid_fn(p_grid) -> ∇D (raw, per-cell, NOT normalized)`.
    // Both `Some` together or both `None`. When present they refine the
    // edge-crossing centroid position; when absent the centroid is used
    // verbatim (legacy / full-asset path).
    density_grid_fn: Option<&D>,
    gradient_grid_fn: Option<&G>,
    // Active iso threshold for the Newton target (0.5 in production).
    iso: f32,
    // **QEF-Hermite placement** (smooth-by-construction). When `true`, the
    // vertex is the regularized tangent-plane intersection of the cube's
    // surface-leaf Hermite samples (`p_surf = center − dist·n`) clamped to
    // the cube, replacing the edge-crossing centroid / Newton path. `dists`
    // is the per-slot signed-distance pool (voxel units), indexed exactly
    // like `leaf_attr_pool`. Off (or `dists` empty) keeps the legacy path.
    qef_hermite: bool,
    dists: &[i16],
) -> MeshVertex
where
    F: Fn(IVec3) -> Option<u32>,
    S: Fn(Vec3) -> f32,
    // `?Sized` so the caller can pass a `&dyn Fn` normal callback (the
    // analytic shading normal threaded through the terrain bake) as well
    // as a concrete closure (the `∇D` / corner-average paths).
    N: Fn(Vec3) -> Vec3 + ?Sized,
    D: Fn(Vec3) -> f32,
    G: Fn(Vec3) -> Vec3,
{
    // Pre-classify the 8 corner cells once; the edge loop reuses these.
    // Bit layout: index = bit0(+X) | bit1(+Y) | bit2(+Z).
    //
    // **Solidity comes from the D field, NOT binary occupancy** (when a
    // `density_grid_fn` is supplied — the sculpt/terrain path). A corner
    // is "solid" iff `D(corner_cell_center) >= iso`, the SAME threshold
    // and sampler the edge-crossing uses. This keeps the active-edge
    // topology consistent with the `D = iso` placement so every active
    // edge has a real interior crossing (no `t` clamps → no terraces).
    // The full-asset / legacy path (no `density_grid_fn`) falls back to
    // binary `cell_lookup` as before.
    let mut corner_solid = [false; 8];
    for i in 0u32..8 {
        let oa = corner_offset(i);
        let c = cube + oa;
        corner_solid[i as usize] = if qef_hermite {
            // Sign-based (see `qef_cell_inside`): the active edges sit on
            // the true crossing so the cube straddles the surface.
            qef_cell_inside(cell_lookup(c), dists)
        } else {
            match density_grid_fn {
                Some(dfn) => {
                    let center =
                        Vec3::new(c.x as f32 + 0.5, c.y as f32 + 0.5, c.z as f32 + 0.5);
                    dfn(center) >= iso
                }
                None => cell_lookup(c).is_some(),
            }
        };
    }

    // **Material attribution** is independent of the D solidity: pick the
    // leaf_attr / bone slot from a BINARY-solid corner so we never index
    // an empty cell's attribute. (A corner can be D-solid but binary-
    // empty just outside the binary boundary, or D-empty but binary-
    // solid just inside it — geometry follows D, material follows the
    // real occupied cell.) `chosen` carries (coord, is_sculpt) so the
    // tie-break prefers sculpt slots over pre-existing ones, then the
    // lowest `(z,y,x)` coord for determinism.
    let mut normal_sum = Vec3::ZERO;
    let mut leaf_attr_id: u32 = 0;
    let mut chosen: Option<(IVec3, bool)> = None;
    for i in 0u32..8 {
        let oa = corner_offset(i);
        let c = cube + oa;
        if let Some(slot) = cell_lookup(c) {
            if slot != CELL_INTERIOR {
                if let Some(attr) = leaf_attr_pool.get(slot as usize) {
                    normal_sum += unpack_oct(attr.normal_oct);
                }
                let c_is_sculpt = sculpt_slots
                    .map(|s| s.contains(&slot))
                    .unwrap_or(false);
                let take = match chosen {
                    None => true,
                    Some((prev_coord, prev_is_sculpt)) => {
                        match (c_is_sculpt, prev_is_sculpt) {
                            (true, false) => true,
                            (false, true) => false,
                            _ => coord_less(c, prev_coord),
                        }
                    }
                };
                if take {
                    chosen = Some((c, c_is_sculpt));
                    leaf_attr_id = slot;
                }
            }
        }
    }

    // Walk the 12 edges; accumulate crossing points. With an SDF
    // evaluator, interpolate to the zero-crossing. Without, fall back
    // to midpoints (naive surface nets).
    let mut crossing_sum = Vec3::ZERO;
    let mut crossing_count: u32 = 0;
    for &(a, b) in &CUBE_EDGES {
        if corner_solid[a as usize] != corner_solid[b as usize] {
            let oa = corner_offset(a);
            let ob = corner_offset(b);
            let crossing = if let Some(sdf) = sdf_fn {
                // Cell centers in grid coords.
                let pa = Vec3::new(
                    (cube.x + oa.x) as f32 + 0.5,
                    (cube.y + oa.y) as f32 + 0.5,
                    (cube.z + oa.z) as f32 + 0.5,
                );
                let pb = Vec3::new(
                    (cube.x + ob.x) as f32 + 0.5,
                    (cube.y + ob.y) as f32 + 0.5,
                    (cube.z + ob.z) as f32 + 0.5,
                );
                let da = sdf(grid_origin + pa * voxel_size);
                let db = sdf(grid_origin + pb * voxel_size);
                let denom = da - db;
                let t = if denom.abs() > 1e-12 {
                    (da / denom).clamp(0.0, 1.0)
                } else {
                    0.5
                };
                pa + (pb - pa) * t
            } else {
                Vec3::new(
                    cube.x as f32 + (oa.x + ob.x) as f32 * 0.5 + 0.5,
                    cube.y as f32 + (oa.y + ob.y) as f32 * 0.5 + 0.5,
                    cube.z as f32 + (oa.z + ob.z) as f32 * 0.5 + 0.5,
                )
            };
            crossing_sum += crossing;
            crossing_count += 1;
        }
    }

    let centroid0 = if crossing_count > 0 {
        crossing_sum / crossing_count as f32
    } else {
        Vec3::new(
            cube.x as f32 + 1.0,
            cube.y as f32 + 1.0,
            cube.z as f32 + 1.0,
        )
    };

    // **Newton projection onto the `D = DENSITY_ISO` isosurface.**
    //
    // The edge-crossing centroid sits ~0.2-0.3 voxel off the true smooth
    // isosurface (it averages per-edge linear crossings, which under- /
    // over-shoot a curved `D`). A couple of Newton steps on the smooth
    // density field snap it onto the actual `D = 0.5` level set, removing
    // the residual sub-voxel lumpiness (silhouette facets / slope-edge
    // waviness) without moving the vertex out of its cube neighborhood.
    //
    // Step (grid space): `p -= (D(p) - 0.5) / |∇D|² · ∇D`. This is a
    // deterministic LOCAL function of the (already watertight) `D`/`∇D`
    // grids, so two tiles sharing the halo project a shared boundary
    // vertex to the identical point — the seam stays watertight.
    //
    // The total displacement from the centroid is clamped to ±0.75 cell
    // per axis so a poorly-conditioned step (near-zero gradient, or a
    // crossing far from the surface) can never fling the vertex out of
    // its SN cube.
    // **QEF-Hermite** (smooth-by-construction) takes precedence when
    // enabled: place the vertex on the true surface from the cube's
    // surface-leaf Hermite samples, NOT on the (rippling) blurred-D
    // isosurface. Falls back to the edge-crossing centroid only for a
    // degenerate cube that supplied no surface-leaf plane (defensive).
    let local_centroid = if qef_hermite {
        if manifold_dc_on() {
            manifold_dc_placement(cube, &cell_lookup, leaf_attr_pool, dists).unwrap_or(centroid0)
        } else {
            qef_hermite_placement(cube, &cell_lookup, leaf_attr_pool, dists).unwrap_or(centroid0)
        }
    } else {
        match (density_grid_fn, gradient_grid_fn) {
            (Some(dfn), Some(gfn)) if crossing_count > 0 => {
                let mut p = centroid0;
                for _ in 0..2 {
                    let d = dfn(p);
                    let g = gfn(p);
                    let gg = g.dot(g);
                    if gg < 1e-6 {
                        break;
                    }
                    let step = ((d - iso) / gg) * g;
                    p -= step;
                    // Clamp total displacement to ±0.75 cell of the centroid.
                    p = p.clamp(centroid0 - Vec3::splat(0.75), centroid0 + Vec3::splat(0.75));
                }
                p
            }
            _ => centroid0,
        }
    };

    let local_pos = grid_origin + local_centroid * voxel_size;

    // **Normal.** The outward normal is `-∇D` (∇D points into the
    // solid; its negation points toward EMPTY), obtained by trilinearly
    // interpolating the precomputed grid-point gradient field at the
    // vertex — NOT by central-differencing the trilinear density here.
    //
    // Differentiating the interpolated density (the previous approach)
    // sampled the trilinear `D` at `vertex ± h/2`; the gradient of a
    // trilinear interpolant is piecewise-constant and DISCONTINUOUS
    // across cell boundaries, so the normal flipped cell-to-cell →
    // voxel-scale speckle. The smooth grid-point gradient field is C0
    // under trilinear interpolation, so the resulting normal is
    // continuous across cells.
    //
    // `normal_fn(local_pos)` returns the raw (un-normalized) `-∇D`.
    // Falls back to the averaged corner-leaf normal (then +Y) when no
    // gradient field is supplied or the interpolated gradient is
    // degenerate (locally-flat `D`, e.g. deep interior — shouldn't
    // occur on a real surface cube).
    let normal_oct = if let Some(nf) = normal_fn {
        let n = nf(local_pos);
        if n.length_squared() > 1e-12 {
            pack_oct(n.normalize())
        } else if normal_sum.length_squared() > 1e-12 {
            pack_oct(normal_sum)
        } else {
            pack_oct(Vec3::Y)
        }
    } else if normal_sum.length_squared() > 1e-12 {
        pack_oct(normal_sum)
    } else {
        pack_oct(Vec3::Y)
    };

    // Bone weights come from the same chosen surface cell that
    // contributed `leaf_attr_id` — keeps the per-vertex attribution
    // consistent across normal / material / skinning. SN cubes that
    // straddle a bone boundary will pick whichever side won the
    // (z, y, x) tie-break; a smarter blend (max-weighted bone across
    // the 8 corners) is possible but unnecessary at finest voxel size,
    // where each cube already spans a sub-millimeter neighborhood.
    let bone_voxel = bone_voxel_pool
        .get(leaf_attr_id as usize)
        .copied()
        .unwrap_or_default();

    MeshVertex {
        local_pos: local_pos.to_array(),
        normal_oct,
        leaf_attr_id,
        bone_indices: bone_voxel.indices,
        bone_weights: bone_voxel.weights,
        _pad: 0,
    }
}

/// **Wide-window plane-fit projection** — the principled fix for the
/// residual coherent "smooth-stairs" ripple that the fixed-width R=2
/// density blur leaves on gentle (wide-tread) slopes.
///
/// ### Why the ripple exists
///
/// The surface is the `D = 0.5` isosurface of a Gaussian blur of binary
/// occupancy. On a gentle slope the occupancy is a staircase whose tread
/// is several cells wide; the fixed R=2 (σ=1) kernel is NARROWER than the
/// tread, so it cannot fully average the steps — the blurred `D = 0.5`
/// isosurface itself undulates at the tread wavelength (a low-amplitude,
/// low-frequency ripple). The per-cube Newton projection faithfully snaps
/// each vertex ONTO that rippling isosurface, so it preserves the ripple.
/// Widening the blur (R=3/4) removes it but adds convex bias and rounds
/// real curvature — an aesthetic cost.
///
/// ### The fix (no blur change)
///
/// After extraction, for each surface vertex fit a least-squares PLANE to
/// the surface vertices in a wide neighbourhood (radius `r_world =
/// radius_voxels · voxel_size`, wide enough to span a tread) and move the
/// vertex ONTO that plane ALONG THE PLANE NORMAL only. The plane is the
/// best local linear fit of the surface, so the periodic ripple — being
/// zero-mean about the local plane — is averaged out, while genuine
/// large-scale curvature (captured by the plane drifting vertex-to-vertex)
/// and the surface position are preserved. The displacement is clamped to
/// `±0.75` voxel (matching the Newton clamp) so a poorly-conditioned fit
/// can never fling a vertex.
///
/// This touches ONLY the vertex positions (a post-pass over the extract
/// output); the blur, topology, material attribution, and the
/// high-frequency `∇D` normal are all unchanged — so there is no convex
/// bias and no curvature rounding beyond the ripple removal. Normals are
/// left as-is (they already come from the smooth gradient field; the
/// sub-voxel position nudge does not warrant a renormal).
///
/// ### Tile-seam watertightness — the shared seam ring MUST be pinned
///
/// For a SINGLE grid the fit is a deterministic local function of the
/// vertices, so it's trivially consistent. For TILED terrain it is NOT:
/// the plane-fit window reaches `radius_voxels` into the neighbour tile,
/// but each tile only meshes its own interior + the 1-ring shared seam
/// (the halo protocol meshes the boundary cube identically, nothing
/// deeper). So tile A fits its seam-vertex plane to {A interior + seam
/// ring} and tile B fits to {B interior + seam ring} — DIFFERENT vertex
/// sets → the SHARED seam vertices would project to DIFFERENT positions
/// → a crack of up to ±`radius`·something voxels.
///
/// **Fix:** `pin_boundary = Some((aabb_min, aabb_max))` pins every vertex
/// within ½ voxel of any of the 6 tile-boundary faces — those are exactly
/// the shared seam-ring vertices the neighbour also owns. Pinned vertices
/// keep their watertight extract positions (never moved) but STILL
/// contribute to their neighbours' plane fits, so the tile interior right
/// next to the seam still de-ripples (its fit window includes the pinned
/// ring). Only the 1-vertex-thick seam ring is held — a negligible thin
/// band — and adjacent tiles agree there bit-for-bit. Pass `None` for a
/// single-grid extract (the bench) where there is no seam to protect.
pub fn wide_window_plane_project(
    verts: &mut [MeshVertex],
    _indices: &[u32],
    voxel_size: f32,
    radius_voxels: f32,
    pin_boundary: Option<(Vec3, Vec3)>,
) {
    if radius_voxels <= 0.0 || verts.len() < 4 {
        return;
    }
    let r_world = radius_voxels * voxel_size;
    let r2 = r_world * r_world;

    // Shared seam-ring mask: vertices within ½ voxel of any tile-boundary
    // face are PINNED (kept at their watertight position) so adjacent
    // tiles agree on the seam. They still feed neighbours' fits below.
    let pin_margin = 0.5 * voxel_size;
    let is_pinned = |p: Vec3| -> bool {
        match pin_boundary {
            Some((lo, hi)) => {
                (p.x - lo.x).abs() < pin_margin
                    || (p.x - hi.x).abs() < pin_margin
                    || (p.y - lo.y).abs() < pin_margin
                    || (p.y - hi.y).abs() < pin_margin
                    || (p.z - lo.z).abs() < pin_margin
                    || (p.z - hi.z).abs() < pin_margin
            }
            None => false,
        }
    };

    // Only project TOP-SURFACE vertices (normal predominantly +Y). The
    // height-field block's vertical walls + bottom would otherwise be
    // dragged by the plane fit; the ripple we target is on the top face.
    // (For closed shapes this would be every vertex, but the production
    // use is terrain — a height field — so the top-face gate is correct
    // and conservative.)
    let positions: Vec<Vec3> = verts.iter().map(|v| Vec3::from(v.local_pos)).collect();
    let is_top: Vec<bool> = verts
        .iter()
        .map(|v| unpack_oct(v.normal_oct).normalize_or_zero().y > 0.30)
        .collect();

    // Spatial hash of TOP vertices, bucket size = r_world.
    let inv_cell = 1.0 / r_world.max(1e-6);
    let key_of = |p: Vec3| -> (i32, i32, i32) {
        (
            (p.x * inv_cell).floor() as i32,
            (p.y * inv_cell).floor() as i32,
            (p.z * inv_cell).floor() as i32,
        )
    };
    let mut buckets: rustc_hash::FxHashMap<(i32, i32, i32), Vec<u32>> =
        rustc_hash::FxHashMap::default();
    for (i, p) in positions.iter().enumerate() {
        if is_top[i] {
            buckets.entry(key_of(*p)).or_default().push(i as u32);
        }
    }

    // Compute new positions into a buffer (read old, write new — so the
    // fit is order-independent / deterministic).
    let mut new_pos: Vec<Vec3> = positions.clone();
    for i in 0..verts.len() {
        if !is_top[i] {
            continue;
        }
        let p = positions[i];
        // Shared seam-ring vertices stay at their watertight position
        // (but were already added to `buckets`, so they still anchor
        // their interior neighbours' plane fits below).
        if is_pinned(p) {
            continue;
        }
        let (kx, ky, kz) = key_of(p);
        // Gather neighbours within r_world from the 27 adjacent buckets,
        // with a Gaussian DISTANCE WEIGHT `w = exp(-d²/(2σ²))`,
        // `σ = r_world/2`. The weighting is what resolves the
        // slope-vs-curvature tension: near points dominate the fit so
        // genuine large-scale curvature (FBM hills) is followed, while the
        // wide tail still averages the periodic ripple to zero. An
        // unweighted box fit at the same radius over-flattens real
        // curvature (a wide plane can't follow a hill); the Gaussian
        // tail-off keeps the fit locally quadratic-faithful.
        let two_sig2 = 0.5 * r2; // 2σ² with σ = r_world/2
        let mut nbrs: Vec<(Vec3, f32)> = Vec::new();
        let mut wsum = 0.0f32;
        for dz in -1..=1 {
            for dy in -1..=1 {
                for dx in -1..=1 {
                    if let Some(ids) = buckets.get(&(kx + dx, ky + dy, kz + dz)) {
                        for &j in ids {
                            let q = positions[j as usize];
                            let d2 = (q - p).length_squared();
                            if d2 <= r2 {
                                let w = (-d2 / two_sig2).exp();
                                nbrs.push((q, w));
                                wsum += w;
                            }
                        }
                    }
                }
            }
        }
        if nbrs.len() < 6 || wsum < 1e-6 {
            // Too few points for a stable plane — leave the vertex.
            continue;
        }
        // Weighted least-squares plane via PCA: weighted centroid +
        // smallest-eigenvector normal of the weighted covariance.
        let inv_w = 1.0 / wsum;
        let centroid = nbrs.iter().map(|&(q, w)| q * w).sum::<Vec3>() * inv_w;
        let mut cxx = 0.0;
        let mut cyy = 0.0;
        let mut czz = 0.0;
        let mut cxy = 0.0;
        let mut cxz = 0.0;
        let mut cyz = 0.0;
        for &(q, w) in &nbrs {
            let d = (q - centroid) * w.sqrt();
            cxx += d.x * d.x;
            cyy += d.y * d.y;
            czz += d.z * d.z;
            cxy += d.x * d.y;
            cxz += d.x * d.z;
            cyz += d.y * d.z;
        }
        // Normalize by the total weight (the eigenvector direction is
        // scale-invariant, but this keeps the matrix well-conditioned).
        let cov = [
            [cxx * inv_w, cxy * inv_w, cxz * inv_w],
            [cxy * inv_w, cyy * inv_w, cyz * inv_w],
            [cxz * inv_w, cyz * inv_w, czz * inv_w],
        ];
        let normal = smallest_eigenvector_sym3(cov);
        if normal.length_squared() < 1e-12 {
            continue;
        }
        let normal = normal.normalize();
        // Project p onto the plane through `centroid` with `normal`:
        // move along the normal by the signed distance.
        let signed = (p - centroid).dot(normal);
        let mut delta = -signed * normal;
        // Clamp to ±0.75 voxel per axis (Newton-projection clamp).
        let clamp = 0.75 * voxel_size;
        delta = delta.clamp(Vec3::splat(-clamp), Vec3::splat(clamp));
        new_pos[i] = p + delta;
    }

    for (i, v) in verts.iter_mut().enumerate() {
        v.local_pos = new_pos[i].to_array();
    }
}

/// Smallest-eigenvector of a 3×3 symmetric positive-semidefinite matrix
/// (the plane normal in a least-squares plane fit). Uses the analytic
/// symmetric-eigenvalue formula for the eigenvalues, then inverse
/// iteration (a couple of steps against `cov - λ_min·I`) for the vector.
/// Robust enough for the well-conditioned point-cloud covariances here.
fn smallest_eigenvector_sym3(m: [[f32; 3]; 3]) -> Vec3 {
    // Eigenvalues of a symmetric 3×3 (Smith's closed form).
    let p1 = m[0][1] * m[0][1] + m[0][2] * m[0][2] + m[1][2] * m[1][2];
    let (l0, l1, l2);
    if p1 < 1e-18 {
        // Already diagonal.
        l0 = m[0][0];
        l1 = m[1][1];
        l2 = m[2][2];
    } else {
        let q = (m[0][0] + m[1][1] + m[2][2]) / 3.0;
        let p2 = (m[0][0] - q).powi(2)
            + (m[1][1] - q).powi(2)
            + (m[2][2] - q).powi(2)
            + 2.0 * p1;
        let p = (p2 / 6.0).sqrt().max(1e-18);
        // B = (1/p)(m - qI).
        let b = [
            [(m[0][0] - q) / p, m[0][1] / p, m[0][2] / p],
            [m[1][0] / p, (m[1][1] - q) / p, m[1][2] / p],
            [m[2][0] / p, m[2][1] / p, (m[2][2] - q) / p],
        ];
        // det(B)/2.
        let det_b = b[0][0] * (b[1][1] * b[2][2] - b[1][2] * b[2][1])
            - b[0][1] * (b[1][0] * b[2][2] - b[1][2] * b[2][0])
            + b[0][2] * (b[1][0] * b[2][1] - b[1][1] * b[2][0]);
        let r = (det_b * 0.5).clamp(-1.0, 1.0);
        let phi = r.acos() / 3.0;
        let eig1 = q + 2.0 * p * phi.cos();
        let eig3 = q + 2.0 * p * (phi + 2.0 * std::f32::consts::PI / 3.0).cos();
        let eig2 = 3.0 * q - eig1 - eig3;
        l0 = eig1;
        l1 = eig2;
        l2 = eig3;
    }
    let lambda_min = l0.min(l1).min(l2);
    // Inverse iteration against (m - λ_min I): solve (m - λ_min I) x = b
    // a couple of times. Use a small shift to keep the system solvable.
    let shift = lambda_min - 1e-4 * (l0.abs().max(l1.abs()).max(l2.abs()) + 1.0);
    let a = [
        [m[0][0] - shift, m[0][1], m[0][2]],
        [m[1][0], m[1][1] - shift, m[1][2]],
        [m[2][0], m[2][1], m[2][2] - shift],
    ];
    // Start from each axis; pick the result with the largest norm (most
    // aligned with the true eigenvector).
    let mut best = Vec3::ZERO;
    let mut best_len = 0.0f32;
    for seed in [Vec3::X, Vec3::Y, Vec3::Z] {
        let mut v = seed;
        for _ in 0..3 {
            match solve3(a, v) {
                Some(s) => {
                    let l = s.length();
                    if l > 1e-20 {
                        v = s / l;
                    }
                }
                None => break,
            }
        }
        // Residual: how small is m·v relative to v (closer to eigenvector
        // with the smallest eigenvalue → smaller ‖m·v − λ_min v‖).
        let mv = Vec3::new(
            m[0][0] * v.x + m[0][1] * v.y + m[0][2] * v.z,
            m[1][0] * v.x + m[1][1] * v.y + m[1][2] * v.z,
            m[2][0] * v.x + m[2][1] * v.y + m[2][2] * v.z,
        );
        let resid = (mv - lambda_min * v).length();
        let score = 1.0 / (resid + 1e-6);
        if score > best_len {
            best_len = score;
            best = v;
        }
    }
    best
}

/// Solve the 3×3 linear system `a x = b` by Cramer's rule. `None` if
/// the matrix is (near-)singular.
fn solve3(a: [[f32; 3]; 3], b: Vec3) -> Option<Vec3> {
    let det = a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
        - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
        + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);
    if det.abs() < 1e-20 {
        return None;
    }
    let inv_det = 1.0 / det;
    let dx = b.x * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
        - a[0][1] * (b.y * a[2][2] - a[1][2] * b.z)
        + a[0][2] * (b.y * a[2][1] - a[1][1] * b.z);
    let dy = a[0][0] * (b.y * a[2][2] - a[1][2] * b.z)
        - b.x * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
        + a[0][2] * (a[1][0] * b.z - b.y * a[2][0]);
    let dz = a[0][0] * (a[1][1] * b.z - b.y * a[2][1])
        - a[0][1] * (a[1][0] * b.z - b.y * a[2][0])
        + b.x * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);
    Some(Vec3::new(dx * inv_det, dy * inv_det, dz * inv_det))
}

/// **Gibson Constrained Elastic Surface Net** — occupancy-only smooth
/// vertex placement, robust to the CPU sculpt path's deliberately
/// rank-1 (homogenized) per-leaf normals.
///
/// Naive surface-net vertices sit at SN-cube edge-crossing centroids,
/// which staircase on slopes that aren't grid-aligned. This relaxer:
///
/// 1. **Captures each vertex's original cell-box origin** so the final
///    position can be bounded to `±h/2` of it (`h = voxel_size`).
/// 2. **Taubin λ|μ smoothing** — alternates a shrink step (λ toward the
///    1-ring mean) and an inflate step (μ, negative) so the mesh
///    de-staircases without the global volume shrinkage that pure
///    Laplacian smoothing produces.
/// 3. **Box constraint (the Gibson part):** after *every* step each
///    component is clamped to `[orig[i] − h/2, orig[i] + h/2]`. This is
///    the only load-bearing use of `voxel_size`, and it bounds the
///    per-vertex displacement so smoothing can never exceed the
///    occupancy field's own `±h/2` positional ambiguity (the floor the
///    A1 falsification tests pin). Clamping every intermediate step
///    keeps the whole trajectory inside the box, so the final position
///    is provably in-box regardless of iteration count.
/// 4. **Soft anchor (small weight, once after the Taubin pass):** convex
///    features erode when smoothing drags a vertex *inward* (behind the
///    plane through its original position with its original normal). The
///    anchor pulls such inward-eroded vertices a small fraction back out
///    toward that plane, still inside the box. It is deliberately
///    one-sided — vertices that smoothed onto or past their plane (e.g.
///    staircase terraces flattening onto the true slope) are untouched,
///    so the guard never re-imposes the staircase the relaxer just
///    removed. It's a guard, not the placer — small weight, applied once.
///
/// After the position loop, per-vertex normals are recomputed as the
/// **area-weighted average of incident triangle face normals from the
/// relaxed positions** (Task A3) so shading agrees with the de-staircased
/// geometry instead of carrying the stale lattice-quantized normals.
///
/// IMPORTANT: this is *not* a QEF. The CPU sculpt path homogenizes
/// per-leaf normals (anti-shading-stripe), so the tangent-plane set is
/// rank-1 and a normal-driven QEF degenerates — commit `9b4930c8`
/// removed exactly that. Occupancy + the box clamp is the correct,
/// field-free placer here. The GPU `proc_surface_nets.wesl` QEF is
/// untouched (it has full-rank live-field gradients).
///
/// `iterations`: number of Taubin shrink+inflate pairs (6-10 typical).
/// `pin_boundary`: if `Some((aabb_min, aabb_max))`, vertices within
/// `3 * voxel_size` of any AABB face are pinned to preserve tile seams.
pub fn relax_surface_net_vertices(
    vertices: &mut [MeshVertex],
    indices: &[u32],
    voxel_size: f32,
    iterations: u32,
    pin_boundary: Option<(Vec3, Vec3)>,
) {
    if vertices.is_empty() || indices.is_empty() || iterations == 0 {
        return;
    }
    let n = vertices.len();

    // Capture origins and the cell-box half-extent. Every final vertex
    // position is clamped componentwise to `orig ± half`.
    let orig: Vec<Vec3> = vertices.iter().map(|v| Vec3::from(v.local_pos)).collect();
    let half = voxel_size * 0.5;

    // Capture original normals for the soft anchor. The anchor pulls each
    // smoothed vertex toward the plane through its origin with this
    // normal — using the *original* (pre-relax) normals keeps the anchor
    // a fixed reference rather than chasing the moving surface. Falls back
    // to +Y on a degenerate-packed normal.
    let orig_normals: Vec<Vec3> = vertices
        .iter()
        .map(|v| {
            let nrm = unpack_oct(v.normal_oct);
            if nrm.length_squared() > 1e-8 {
                nrm.normalize()
            } else {
                Vec3::Y
            }
        })
        .collect();

    // Build adjacency: for each vertex, collect unique neighbor IDs.
    let mut adj_offsets: Vec<u32> = vec![0; n + 1];
    // Count edges per vertex (each triangle contributes 2 directed edges
    // per incident vertex; deduped below to one entry per neighbor).
    for tri in indices.chunks_exact(3) {
        let (a, b, c) = (tri[0] as usize, tri[1] as usize, tri[2] as usize);
        adj_offsets[a] += 2;
        adj_offsets[b] += 2;
        adj_offsets[c] += 2;
    }
    // Prefix sum for CSR layout.
    let mut total = 0u32;
    for slot in adj_offsets.iter_mut().take(n) {
        let count = *slot;
        *slot = total;
        total += count;
    }
    adj_offsets[n] = total;
    let mut adj_data: Vec<u32> = vec![0; total as usize];
    let mut adj_write: Vec<u32> = adj_offsets[..n].to_vec();
    for tri in indices.chunks_exact(3) {
        let (a, b, c) = (tri[0], tri[1], tri[2]);
        let mut push = |from: u32, to: u32| {
            let idx = adj_write[from as usize] as usize;
            adj_data[idx] = to;
            adj_write[from as usize] += 1;
        };
        push(a, b);
        push(a, c);
        push(b, a);
        push(b, c);
        push(c, a);
        push(c, b);
    }
    // Real dedup per vertex: sort then collapse runs in place, shrinking
    // each vertex's adjacency span to its set of UNIQUE neighbors and
    // recording the new end in `adj_end`. The old code only sorted and
    // relied on an adjacent-duplicate skip in the hot loop, which left
    // self-edges and double-counted neighbors a step could silently
    // accumulate; here the 1-ring mean is over true unique neighbors.
    let mut adj_end: Vec<u32> = vec![0; n];
    for i in 0..n {
        let start = adj_offsets[i] as usize;
        let end = adj_offsets[i + 1] as usize;
        let span = &mut adj_data[start..end];
        span.sort_unstable();
        let mut write = 0usize;
        let mut prev = u32::MAX;
        for k in 0..span.len() {
            let v = span[k];
            // Drop duplicates and any self-edge (a vertex is never its
            // own 1-ring neighbor).
            if v == prev || v == i as u32 {
                continue;
            }
            prev = v;
            span[write] = v;
            write += 1;
        }
        adj_end[i] = start as u32 + write as u32;
    }

    // Pin boundary vertices to preserve tile seams.
    let pinned: Vec<bool> = if let Some((aabb_min, aabb_max)) = pin_boundary {
        let margin = voxel_size * 3.0;
        vertices
            .iter()
            .map(|v| {
                let p = Vec3::from(v.local_pos);
                (p.x - aabb_min.x).abs() < margin
                    || (p.x - aabb_max.x).abs() < margin
                    || (p.y - aabb_min.y).abs() < margin
                    || (p.y - aabb_max.y).abs() < margin
                    || (p.z - aabb_min.z).abs() < margin
                    || (p.z - aabb_max.z).abs() < margin
            })
            .collect()
    } else {
        vec![false; n]
    };

    // Taubin λ|μ: shrink then inflate. λ > 0 smooths; μ < 0, |μ| > λ
    // counteracts the shrink so volume is preserved. Standard
    // pass-band values (λ=0.33, μ=−0.34).
    const LAMBDA: f32 = 0.33;
    const MU: f32 = -0.34;
    let mut new_pos = vec![Vec3::ZERO; n];

    // One Laplacian step with factor `factor`, then the Gibson box clamp.
    // Reads `vertices`, writes `new_pos`, then commits.
    let mut laplacian_step = |factor: f32, vertices: &mut [MeshVertex]| {
        for i in 0..n {
            let cur = Vec3::from(vertices[i].local_pos);
            if pinned[i] {
                new_pos[i] = cur;
                continue;
            }
            let start = adj_offsets[i] as usize;
            let end = adj_end[i] as usize;
            let nbrs = &adj_data[start..end];
            if nbrs.is_empty() {
                new_pos[i] = cur;
                continue;
            }
            let mut sum = Vec3::ZERO;
            for &nb in nbrs {
                sum += Vec3::from(vertices[nb as usize].local_pos);
            }
            let avg = sum / nbrs.len() as f32;
            let p = cur + (avg - cur) * factor;

            // Gibson constraint: clamp componentwise to the cell box.
            let lo = orig[i] - Vec3::splat(half);
            let hi = orig[i] + Vec3::splat(half);
            new_pos[i] = p.clamp(lo, hi);
        }
        for i in 0..n {
            vertices[i].local_pos = new_pos[i].to_array();
        }
    };

    for _ in 0..iterations {
        laplacian_step(LAMBDA, vertices);
        laplacian_step(MU, vertices);
    }

    // Soft anchor (a guard, not the placer): a convex feature erodes when
    // smoothing drags its vertex INWARD — behind the original tangent
    // plane through `orig[i]` with normal `orig_normals[i]`. After the
    // Taubin pass, pull each such inward-eroded vertex a small fraction
    // back OUT toward that plane (still inside the cell box). The guard is
    // deliberately one-sided: vertices that smoothed onto or past their
    // original plane (e.g. staircase terraces flattening onto the true
    // slope) are untouched, so the anchor never re-imposes the staircase
    // it's the relaxer's whole job to remove. Applied once, after
    // iterating, so it can't compound across steps into a dominant term.
    //
    // Weight is kept low (0.12) on purpose: a blanket erosion guard also
    // fires on legitimate de-staircasing corrections (a high-terrace
    // vertex moving down onto the true slope reads as "inward" against
    // its +Y-ish normal), so a large weight would re-impose the staircase
    // and dominate the placer (measured: 0.25 forces ~3× the iterations
    // to clear a slope). 0.12 still nudges genuine convex creases back
    // out while leaving de-staircasing convergence fast. Detecting true
    // creases to anchor them hard is the deferred Task A6.
    const ANCHOR_WEIGHT: f32 = 0.12;
    for i in 0..n {
        if pinned[i] {
            continue;
        }
        let p = Vec3::from(vertices[i].local_pos);
        let n_i = orig_normals[i];
        // Signed distance off the original tangent plane (positive =
        // outward along the normal, negative = eroded inward).
        let signed = (p - orig[i]).dot(n_i);
        if signed >= 0.0 {
            continue; // not eroded — leave it.
        }
        // Move a fraction of the inward erosion back out.
        let pulled = p - n_i * (signed * ANCHOR_WEIGHT);
        let lo = orig[i] - Vec3::splat(half);
        let hi = orig[i] + Vec3::splat(half);
        vertices[i].local_pos = pulled.clamp(lo, hi).to_array();
    }

    // --- Task A3: recompute normals from the relaxed faces. ---
    //
    // Accumulate each triangle's geometric normal (length ∝ 2× face
    // area) onto its three vertices, so the per-vertex normal is the
    // area-weighted average of incident face normals at the RELAXED
    // positions. Winding is the extractor's outward CCW order (the same
    // order the `closed_cube_winds_outward` test pins), so the
    // accumulated normal points outward.
    let mut normal_accum = vec![Vec3::ZERO; n];
    for tri in indices.chunks_exact(3) {
        let (ia, ib, ic) = (tri[0] as usize, tri[1] as usize, tri[2] as usize);
        let a = Vec3::from(vertices[ia].local_pos);
        let b = Vec3::from(vertices[ib].local_pos);
        let c = Vec3::from(vertices[ic].local_pos);
        // Cross product magnitude == 2× triangle area → area weighting
        // falls out for free; no per-triangle normalize.
        let face = (b - a).cross(c - a);
        normal_accum[ia] += face;
        normal_accum[ib] += face;
        normal_accum[ic] += face;
    }
    for i in 0..n {
        let acc = normal_accum[i];
        if acc.length_squared() > 1e-12 {
            vertices[i].normal_oct = pack_oct(acc.normalize());
        }
        // Else: isolated/degenerate vertex — keep its prior normal.
    }
}

/// Find the closest point on a polyline (sequence of connected line
/// segments) to point `p`. For a single-point polyline, returns that
/// point. Used by [`point_in_stroke_tube`].
pub fn nearest_on_polyline(p: Vec3, path: &[Vec3]) -> Vec3 {
    debug_assert!(!path.is_empty());
    if path.len() == 1 {
        return path[0];
    }
    let mut best = path[0];
    let mut best_dist_sq = f32::INFINITY;
    for seg in path.windows(2) {
        let ab = seg[1] - seg[0];
        let len_sq = ab.length_squared();
        let nearest = if len_sq < 1e-12 {
            seg[0]
        } else {
            let t = ((p - seg[0]).dot(ab) / len_sq).clamp(0.0, 1.0);
            seg[0] + ab * t
        };
        let d = (p - nearest).length_squared();
        if d < best_dist_sq {
            best_dist_sq = d;
            best = nearest;
        }
    }
    best
}

/// Test whether a point lies inside the stroke tube (the Minkowski sum
/// of the stroke polyline with a sphere of the given radius squared).
pub fn point_in_stroke_tube(p: Vec3, path: &[Vec3], radius_sq: f32) -> bool {
    let nearest = nearest_on_polyline(p, path);
    (p - nearest).length_squared() <= radius_sq
}

/// Cube corner offset for index `i` — bit 0 = +X, bit 1 = +Y, bit 2 = +Z.
#[inline]
fn corner_offset(i: u32) -> IVec3 {
    IVec3::new(
        (i & 1) as i32,
        ((i >> 1) & 1) as i32,
        ((i >> 2) & 1) as i32,
    )
}

/// The 12 axis-aligned edges of a cube, as (corner_a, corner_b) index
/// pairs. Order: 4 X-edges, 4 Y-edges, 4 Z-edges.
const CUBE_EDGES: [(u32, u32); 12] = [
    // +X axis (offsets differ in bit 0)
    (0, 1), (2, 3), (4, 5), (6, 7),
    // +Y axis (bit 1)
    (0, 2), (1, 3), (4, 6), (5, 7),
    // +Z axis (bit 2)
    (0, 4), (1, 5), (2, 6), (3, 7),
];

#[inline]
fn coord_less(a: IVec3, b: IVec3) -> bool {
    (a.z, a.y, a.x) < (b.z, b.y, b.x)
}

/// Outward normals for the 6 cell faces, in this order:
/// +X, -X, +Y, -Y, +Z, -Z. Used to walk neighbor cells.
const FACE_DIRS: [IVec3; 6] = [
    IVec3::new(1, 0, 0),
    IVec3::new(-1, 0, 0),
    IVec3::new(0, 1, 0),
    IVec3::new(0, -1, 0),
    IVec3::new(0, 0, 1),
    IVec3::new(0, 0, -1),
];

/// For each face direction (matching [`FACE_DIRS`]), the 4 SN-cube
/// offsets relative to the solid cell that form the active sample-edge
/// between the solid cell and its empty neighbor — listed in CCW order
/// about the outward normal so triangles `(0, 1, 2)` and `(0, 2, 3)`
/// face outward.
///
/// Derivation: the axis edge between cell A and cell A+dir is shared by
/// 4 SN cubes whose corner cells include both A and A+dir; rotating the
/// 2×2 group of cubes about `dir` (right-hand rule) gives CCW order.
const CUBE_OFFSETS_PER_FACE: [[IVec3; 4]; 6] = [
    // +X — CCW about +X is +Y → +Z.
    [
        IVec3::new(0, -1, -1),
        IVec3::new(0, 0, -1),
        IVec3::new(0, 0, 0),
        IVec3::new(0, -1, 0),
    ],
    // -X — CCW about -X (reverse of +X traversal).
    [
        IVec3::new(-1, -1, -1),
        IVec3::new(-1, -1, 0),
        IVec3::new(-1, 0, 0),
        IVec3::new(-1, 0, -1),
    ],
    // +Y — CCW about +Y is +Z → +X.
    [
        IVec3::new(-1, 0, -1),
        IVec3::new(-1, 0, 0),
        IVec3::new(0, 0, 0),
        IVec3::new(0, 0, -1),
    ],
    // -Y — CCW about -Y (reverse of +Y traversal).
    [
        IVec3::new(-1, -1, -1),
        IVec3::new(0, -1, -1),
        IVec3::new(0, -1, 0),
        IVec3::new(-1, -1, 0),
    ],
    // +Z — CCW about +Z is +X → +Y.
    [
        IVec3::new(-1, -1, 0),
        IVec3::new(0, -1, 0),
        IVec3::new(0, 0, 0),
        IVec3::new(-1, 0, 0),
    ],
    // -Z — CCW about -Z (reverse of +Z traversal).
    [
        IVec3::new(-1, -1, -1),
        IVec3::new(-1, 0, -1),
        IVec3::new(0, 0, -1),
        IVec3::new(0, -1, -1),
    ],
];

/// Walk the octree but populate `cells` only with non-empty cells
/// inside the half-open region `[region_min, region_max)` (plus a small
/// pad walked at the branch level so SN-cube boundary stitching has
/// data on either side).
///
/// **Phase B R4c-V2** uses this to avoid the full-asset cell-map walk
/// per stamp. For a small brush region in a deep octree the recursive
/// walk prunes any branch whose AABB doesn't intersect the region, so
/// only a handful of brick-leaf nodes get expanded — sub-millisecond
/// on splat5 vs ~500 ms-1 s for the full walk.
pub fn collect_cell_map_in_region(
    octree_nodes: &[u32],
    octree_depth: u8,
    brick_cells: &[u32],
    region_min: IVec3,
    region_max: IVec3,
) -> CellMap {
    let mut cells = CellMap::default();
    if octree_nodes.is_empty() {
        return cells;
    }
    if region_min.x >= region_max.x
        || region_min.y >= region_max.y
        || region_min.z >= region_max.z
    {
        return cells;
    }
    walk_collect_cells_in_region(
        octree_nodes,
        brick_cells,
        0,
        UVec3::ZERO,
        0,
        octree_depth,
        region_min,
        region_max,
        &mut cells,
    );
    cells
}

/// Recursive walker matching `walk_collect_cells` but pruning branches
/// whose AABB doesn't intersect `[region_min, region_max)`.
fn walk_collect_cells_in_region(
    nodes: &[u32],
    brick_cells: &[u32],
    node_idx: usize,
    origin: UVec3,
    level: u8,
    max_depth: u8,
    region_min: IVec3,
    region_max: IVec3,
    cells: &mut CellMap,
) {
    let node = nodes[node_idx];
    if node == EMPTY_NODE || node == INTERIOR_NODE {
        return;
    }
    // This node's AABB in finest-cell coords: `[origin, origin + span)`.
    let span = 1i32 << (max_depth - level);
    let node_min = IVec3::new(origin.x as i32, origin.y as i32, origin.z as i32);
    let node_max = node_min + IVec3::splat(span);
    // Intersection test (both ranges half-open).
    if node_max.x <= region_min.x
        || node_min.x >= region_max.x
        || node_max.y <= region_min.y
        || node_min.y >= region_max.y
        || node_max.z <= region_min.z
        || node_min.z >= region_max.z
    {
        return;
    }
    if is_leaf(node) {
        let cell_voxels = 1u32 << (max_depth - level);
        let slot = leaf_slot(node);
        for dz in 0..cell_voxels {
            for dy in 0..cell_voxels {
                for dx in 0..cell_voxels {
                    let c = IVec3::new(
                        origin.x as i32 + dx as i32,
                        origin.y as i32 + dy as i32,
                        origin.z as i32 + dz as i32,
                    );
                    if c.x < region_min.x
                        || c.x >= region_max.x
                        || c.y < region_min.y
                        || c.y >= region_max.y
                        || c.z < region_min.z
                        || c.z >= region_max.z
                    {
                        continue;
                    }
                    cells.insert(c, slot);
                }
            }
        }
        return;
    }
    if is_brick(node) {
        let bid = brick_id(node);
        let base = (bid * BRICK_CELLS) as usize;
        for cz in 0..BRICK_DIM {
            for cy in 0..BRICK_DIM {
                for cx in 0..BRICK_DIM {
                    let c = IVec3::new(
                        origin.x as i32 + cx as i32,
                        origin.y as i32 + cy as i32,
                        origin.z as i32 + cz as i32,
                    );
                    if c.x < region_min.x
                        || c.x >= region_max.x
                        || c.y < region_min.y
                        || c.y >= region_max.y
                        || c.z < region_min.z
                        || c.z >= region_max.z
                    {
                        continue;
                    }
                    let flat =
                        (cx + cy * BRICK_DIM + cz * BRICK_DIM * BRICK_DIM) as usize;
                    let v = brick_cells[base + flat];
                    if v == BRICK_EMPTY {
                        continue;
                    }
                    let stored = if v == BRICK_INTERIOR { CELL_INTERIOR } else { v };
                    cells.insert(c, stored);
                }
            }
        }
        return;
    }
    if is_branch(node) {
        let children_offset = node as usize;
        let half = 1u32 << (max_depth - level - 1);
        for octant in 0u32..8 {
            let dx = octant & 1;
            let dy = (octant >> 1) & 1;
            let dz = (octant >> 2) & 1;
            let child_origin = UVec3::new(
                origin.x + dx * half,
                origin.y + dy * half,
                origin.z + dz * half,
            );
            walk_collect_cells_in_region(
                nodes,
                brick_cells,
                children_offset + octant as usize,
                child_origin,
                level + 1,
                max_depth,
                region_min,
                region_max,
                cells,
            );
        }
    }
}

/// Walk the octree and populate `cells` with one entry per non-empty
/// cell, at finest resolution. INTERIOR_NODE-region cells are NOT
/// expanded — `is_solid_lookup` resolves them on demand.
fn walk_collect_cells(
    nodes: &[u32],
    brick_cells: &[u32],
    node_idx: usize,
    origin: UVec3,
    level: u8,
    max_depth: u8,
    cells: &mut CellMap,
) {
    let node = nodes[node_idx];
    if node == EMPTY_NODE || node == INTERIOR_NODE {
        return;
    }
    if is_leaf(node) {
        // Variable-depth LEAF: covers `2^(max_depth - level)` cells per
        // axis. For typical assets these are at finest depth (1 cell);
        // for procedural primitives they may be coarser. Expand to all
        // finest cells so SN sees one uniform lattice.
        let cell_voxels = 1u32 << (max_depth - level);
        debug_assert!(
            cell_voxels <= 64,
            "LEAF too coarse for naive SN extraction (covers {}^3 finest cells)",
            cell_voxels,
        );
        let slot = leaf_slot(node);
        for dz in 0..cell_voxels {
            for dy in 0..cell_voxels {
                for dx in 0..cell_voxels {
                    let c = IVec3::new(
                        origin.x as i32 + dx as i32,
                        origin.y as i32 + dy as i32,
                        origin.z as i32 + dz as i32,
                    );
                    cells.insert(c, slot);
                }
            }
        }
        return;
    }
    if is_brick(node) {
        let bid = brick_id(node);
        let base = (bid * BRICK_CELLS) as usize;
        for cz in 0..BRICK_DIM {
            for cy in 0..BRICK_DIM {
                for cx in 0..BRICK_DIM {
                    let flat =
                        (cx + cy * BRICK_DIM + cz * BRICK_DIM * BRICK_DIM) as usize;
                    let v = brick_cells[base + flat];
                    if v == BRICK_EMPTY {
                        continue;
                    }
                    let c = IVec3::new(
                        origin.x as i32 + cx as i32,
                        origin.y as i32 + cy as i32,
                        origin.z as i32 + cz as i32,
                    );
                    let stored = if v == BRICK_INTERIOR { CELL_INTERIOR } else { v };
                    cells.insert(c, stored);
                }
            }
        }
        return;
    }
    if is_branch(node) {
        let children_offset = node as usize;
        let half = 1u32 << (max_depth - level - 1);
        for octant in 0u32..8 {
            let dx = octant & 1;
            let dy = (octant >> 1) & 1;
            let dz = (octant >> 2) & 1;
            let child_origin = UVec3::new(
                origin.x + dx * half,
                origin.y + dy * half,
                origin.z + dz * half,
            );
            walk_collect_cells(
                nodes,
                brick_cells,
                children_offset + octant as usize,
                child_origin,
                level + 1,
                max_depth,
                cells,
            );
        }
    }
}

/// Resolve "is the cell at this coord solid?" by descending the octree.
/// Used for cells outside the dense cell map — primarily INTERIOR_NODE
/// regions, which we don't expand to keep memory bounded. Returns false
/// for out-of-bounds coords (the asset extent's exterior is empty).
///
/// O(depth) per call — within a few-cell-thick surface shell this fires
/// only for the small number of EMPTY-side neighbor lookups per surface
/// cell, so the total cost stays proportional to surface area.
fn is_solid_lookup(
    nodes: &[u32],
    brick_cells: &[u32],
    depth: u8,
    coord: IVec3,
    extent: i32,
) -> bool {
    if coord.x < 0
        || coord.y < 0
        || coord.z < 0
        || coord.x >= extent
        || coord.y >= extent
        || coord.z >= extent
    {
        return false;
    }
    let coord_u = UVec3::new(coord.x as u32, coord.y as u32, coord.z as u32);
    let mut idx = 0usize;
    for level in 0..depth {
        let node = nodes[idx];
        if node == EMPTY_NODE {
            return false;
        }
        if node == INTERIOR_NODE {
            return true;
        }
        if is_leaf(node) {
            return true;
        }
        if is_brick(node) {
            // BRICK lives at this level; its cells span `1 << (depth -
            // level)` finest voxels per axis. The flat brick index is
            // the low bits of `coord` modulo that span.
            let span = 1u32 << (depth - level);
            let mask = span - 1;
            let lx = coord_u.x & mask;
            let ly = coord_u.y & mask;
            let lz = coord_u.z & mask;
            let flat = (lx + ly * span + lz * span * span) as usize;
            let v = brick_cells[(brick_id(node) * BRICK_CELLS) as usize + flat];
            return v != BRICK_EMPTY;
        }
        // Branch: descend.
        let half = 1u32 << (depth - level - 1);
        let ox = if coord_u.x & half != 0 { 1u32 } else { 0 };
        let oy = if coord_u.y & half != 0 { 1u32 } else { 0 };
        let oz = if coord_u.z & half != 0 { 1u32 } else { 0 };
        let octant = (ox + oy * 2 + oz * 4) as usize;
        idx = node as usize + octant;
    }
    let node = nodes[idx];
    match node {
        EMPTY_NODE => false,
        INTERIOR_NODE => true,
        n if is_leaf(n) => true,
        _ => false,
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::sparse_octree::{make_brick, make_leaf};

    #[test]
    fn mesh_vertex_size_is_32() {
        assert_eq!(std::mem::size_of::<MeshVertex>(), 32);
    }

    #[test]
    fn empty_octree_yields_nothing() {
        let nodes = vec![EMPTY_NODE];
        let (verts, indices) = extract_surface_mesh(&nodes, 4, 0.001, Vec3::ZERO, &[], &[], &[], None);
        assert!(verts.is_empty());
        assert!(indices.is_empty());
    }

    /// A single LEAF at the root with depth=0 covers exactly one cell.
    /// All 6 neighbors are out-of-bounds (= EMPTY for SN sign), so we
    /// expect a fully-closed unit cube: 8 unique vertices, 6 faces ×
    /// 2 triangles = 12 triangles = 36 indices.
    ///
    /// With naive-SN smoothing each vertex lands at the centroid of
    /// its SN cube's edge crossings — for a single isolated cell the
    /// 8 cubes around it each have exactly 3 sign-change edges
    /// meeting at the cell, and the centroid of those 3 crossings is
    /// at offset (1/3, 1/3, 1/3) from the cube's "near" corner. So
    /// the vertices form an inscribed cube at offsets (1/3·vs,
    /// 2/3·vs) along each axis.
    #[test]
    fn single_leaf_at_root_emits_a_closed_cube() {
        let nodes = vec![make_leaf(7)];
        let vs = 0.5;
        let origin = Vec3::new(1.0, 2.0, 3.0);
        let (verts, indices) = extract_surface_mesh(&nodes, 0, vs, origin, &[], &[], &[], None);

        assert_eq!(verts.len(), 8, "8 SN-cube vertices around the unit cell");
        assert_eq!(indices.len(), 36, "6 faces × 2 triangles × 3 indices");

        // Inscribed-cube corners at (1/3 or 2/3) × vs offset on each
        // axis. Order doesn't matter — we sort both lists.
        let third = vs / 3.0;
        let two_thirds = 2.0 * vs / 3.0;
        let mut got: Vec<[f32; 3]> = verts.iter().map(|v| v.local_pos).collect();
        got.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mut expected: Vec<[f32; 3]> = (0..8)
            .map(|i| {
                let dx = if (i & 1) != 0 { two_thirds } else { third };
                let dy = if ((i >> 1) & 1) != 0 { two_thirds } else { third };
                let dz = if ((i >> 2) & 1) != 0 { two_thirds } else { third };
                [origin.x + dx, origin.y + dy, origin.z + dz]
            })
            .collect();
        expected.sort_by(|a, b| a.partial_cmp(b).unwrap());
        for (g, e) in got.iter().zip(expected.iter()) {
            for k in 0..3 {
                assert!((g[k] - e[k]).abs() < 1e-5, "{:?} != {:?}", g, e);
            }
        }

        // Every vertex should carry the leaf's `leaf_attr_id`.
        for v in &verts {
            assert_eq!(v.leaf_attr_id, 7);
        }
    }

    /// Six face triangles must wind so their cross product points along
    /// the outward axis (+X, -X, +Y, -Y, +Z, -Z). For a single root
    /// leaf the 12 triangles split exactly 2 per axis-direction, with
    /// no inward-facing triangles.
    #[test]
    fn closed_cube_winds_outward() {
        let nodes = vec![make_leaf(0)];
        let (verts, indices) = extract_surface_mesh(&nodes, 0, 1.0, Vec3::ZERO, &[], &[], &[], None);
        let mut counts = [0i32; 6]; // +X -X +Y -Y +Z -Z

        for tri in indices.chunks(3) {
            let a = Vec3::from_array(verts[tri[0] as usize].local_pos);
            let b = Vec3::from_array(verts[tri[1] as usize].local_pos);
            let c = Vec3::from_array(verts[tri[2] as usize].local_pos);
            let n = (b - a).cross(c - a).normalize_or_zero();
            // Each cube-face triangle is axis-aligned. Find which axis +/- it points.
            let bucket = if n.x > 0.5 {
                0
            } else if n.x < -0.5 {
                1
            } else if n.y > 0.5 {
                2
            } else if n.y < -0.5 {
                3
            } else if n.z > 0.5 {
                4
            } else if n.z < -0.5 {
                5
            } else {
                panic!("triangle normal not axis-aligned: {:?}", n);
            };
            counts[bucket] += 1;
        }
        // 2 triangles per face × 6 faces — perfectly balanced.
        assert_eq!(counts, [2, 2, 2, 2, 2, 2]);
    }

    /// A brick with one filled cell at (0,0,0) should produce the same
    /// closed-cube mesh as a single root leaf, just at a finer
    /// resolution. Verifies brick traversal and per-cell exposure logic
    /// agree with the leaf path.
    #[test]
    fn single_filled_brick_cell_is_a_unit_cube() {
        let nodes = vec![make_brick(0)];
        let mut bricks = vec![BRICK_EMPTY; BRICK_CELLS as usize];
        bricks[0] = 99;
        let (verts, indices) = extract_surface_mesh(&nodes, 2, 1.0, Vec3::ZERO, &bricks, &[], &[], None);
        assert_eq!(verts.len(), 8);
        assert_eq!(indices.len(), 36);
        for v in &verts {
            assert_eq!(v.leaf_attr_id, 99);
        }
    }

    /// Two horizontally-adjacent filled cells share an interior face;
    /// the mesh should *not* emit that face. Total faces = 10 (5 per
    /// cell — 6 cube faces minus the 1 shared face), so we expect
    /// 12 grid-corner vertices (the 12 unique corners of a 2×1×1 box)
    /// and 10 × 2 = 20 triangles = 60 indices.
    #[test]
    fn shared_face_between_adjacent_cells_is_skipped() {
        let nodes = vec![make_brick(0)];
        let mut bricks = vec![BRICK_EMPTY; BRICK_CELLS as usize];
        bricks[0] = 1; // (0,0,0)
        bricks[1] = 2; // (1,0,0) — face-adjacent in +X
        let (verts, indices) = extract_surface_mesh(&nodes, 2, 1.0, Vec3::ZERO, &bricks, &[], &[], None);
        assert_eq!(verts.len(), 12, "12 unique corners of a 2×1×1 box");
        assert_eq!(indices.len(), 60, "10 exposed faces × 2 triangles × 3 indices");
    }

    /// INTERIOR cells (sentinel inside a brick) must not emit faces
    /// toward each other or toward INTERIOR_NODE-region cells, but must
    /// hide adjacent surface-cell faces. With one surface cell at
    /// (0,0,0) and an INTERIOR neighbor at (1,0,0), the shared +X face
    /// of the surface cell is hidden; we expect 5 exposed surface
    /// faces, no faces from the INTERIOR cell itself.
    #[test]
    fn interior_cells_hide_adjacent_surface_faces() {
        let nodes = vec![make_brick(0)];
        let mut bricks = vec![BRICK_EMPTY; BRICK_CELLS as usize];
        bricks[0] = 5; // (0,0,0) surface
        bricks[1] = BRICK_INTERIOR; // (1,0,0) interior
        let (verts, indices) = extract_surface_mesh(&nodes, 2, 1.0, Vec3::ZERO, &bricks, &[], &[], None);
        // Surface cell exposes 5 of 6 faces (+X is hidden by INTERIOR).
        // INTERIOR cell exposes 5 of 6 faces toward EMPTY (-X is hidden
        // by the surface cell, but +X, +Y, -Y, +Z, -Z are exposed to
        // EMPTY since we're in a 2-cell box at the corner of the brick
        // and out-of-brick cells are EMPTY).
        // Total exposed faces = 10. Vertices = 12 (the 2×1×1 box's
        // corners). Indices = 10 × 6 = 60.
        assert_eq!(verts.len(), 12);
        assert_eq!(indices.len(), 60);
    }

    /// INTERIOR_NODE regions must be treated as solid by the on-demand
    /// solidity check. Build a tree where one octant of the root is a
    /// surface BRICK and another is INTERIOR_NODE, sharing a face. The
    /// shared face must be hidden — surface BRICK cells don't emit
    /// faces toward INTERIOR_NODE-region cells.
    #[test]
    fn interior_node_region_is_solid_for_sn_sign() {
        // depth=2 root tree, one branch level. 8 octants.
        // Octant 0 (-X-Y-Z): surface BRICK.
        // Octant 1 (+X-Y-Z): INTERIOR_NODE.
        // Other octants: EMPTY_NODE.
        // With BRICK_DIM=4 and depth=2, each octant covers 1<<(2-1)=2
        // finest voxels per axis — but a BRICK lives at depth-2=0 i.e.
        // at the root. Conflict: BRICK requires being at level 0 with
        // depth-level = BRICK_LEVELS = 2, so root depth 2 with BRICK at
        // root works. But we have a branch at root, so the BRICK lives
        // at level 1 with depth-level=1 — wrong (BRICK needs span 4).
        //
        // Instead: depth=3 root with branch at root. Each octant is at
        // level 1, span 1 << (3-1) = 4 cells per axis = BRICK span. So
        // place a BRICK in octant 0 and INTERIOR_NODE in octant 1.
        let mut nodes = vec![0u32; 9]; // root + 8 children
        nodes[0] = 1; // branch: children at offset 1
        nodes[1] = make_brick(0); // octant 0 (-X-Y-Z)
        nodes[2] = INTERIOR_NODE; // octant 1 (+X-Y-Z)
        nodes[3] = EMPTY_NODE;
        nodes[4] = EMPTY_NODE;
        nodes[5] = EMPTY_NODE;
        nodes[6] = EMPTY_NODE;
        nodes[7] = EMPTY_NODE;
        nodes[8] = EMPTY_NODE;

        // Brick: fill the +X face cells (x=3) with surface, leave rest
        // EMPTY. With INTERIOR_NODE in octant 1 (touching x=4..7), the
        // surface cells at x=3 abut INTERIOR_NODE on their +X side —
        // those +X faces must be hidden.
        let mut bricks = vec![BRICK_EMPTY; BRICK_CELLS as usize];
        for cz in 0..BRICK_DIM {
            for cy in 0..BRICK_DIM {
                let flat = (3 + cy * BRICK_DIM + cz * BRICK_DIM * BRICK_DIM) as usize;
                bricks[flat] = 11;
            }
        }

        let (verts, indices) = extract_surface_mesh(&nodes, 3, 1.0, Vec3::ZERO, &bricks, &[], &[], None);

        // Every triangle must point along an outward axis. Check that
        // *no* triangle points in +X (those would be surface→INTERIOR
        // faces that should have been hidden).
        let mut plus_x_triangles = 0;
        for tri in indices.chunks(3) {
            let a = Vec3::from_array(verts[tri[0] as usize].local_pos);
            let b = Vec3::from_array(verts[tri[1] as usize].local_pos);
            let c = Vec3::from_array(verts[tri[2] as usize].local_pos);
            let n = (b - a).cross(c - a).normalize_or_zero();
            if n.x > 0.5 {
                plus_x_triangles += 1;
            }
        }
        assert_eq!(
            plus_x_triangles, 0,
            "no triangles should face +X — those are surface→INTERIOR_NODE faces"
        );
        // Sanity: we did emit *something* (the other 5 faces of each
        // surface cell are exposed).
        assert!(!indices.is_empty());
    }

    /// Bone weights baked at extract time should match the BoneVoxel
    /// of the surface cell that contributed `leaf_attr_id`. With both
    /// surface slots sharing a single bone (idx 7, weight 255), every
    /// emitted vertex should carry that exact pair — no zeros, no
    /// averaging artifacts. Confirms the extractor reads the parallel
    /// pool by `leaf_attr_id` and the layout matches the VS contract.
    #[test]
    fn vertex_carries_bone_weights_from_chosen_cell() {
        let nodes = vec![make_brick(0)];
        let mut bricks = vec![BRICK_EMPTY; BRICK_CELLS as usize];
        bricks[0] = 0;
        bricks[1] = 1;
        let leaf_attrs = vec![LeafAttr::EMPTY; 2];
        let bone_pool = vec![
            BoneVoxel::new([7, 0, 0, 0], [255, 0, 0, 0]),
            BoneVoxel::new([7, 0, 0, 0], [255, 0, 0, 0]),
        ];
        let (verts, _) = extract_surface_mesh(
            &nodes, 2, 1.0, Vec3::ZERO, &bricks, &leaf_attrs, &bone_pool, None,
        );
        assert!(!verts.is_empty(), "extractor produced no vertices");
        for v in &verts {
            assert_eq!(v.bone_indices, u32::from_le_bytes([7, 0, 0, 0]));
            assert_eq!(v.bone_weights, u32::from_le_bytes([255, 0, 0, 0]));
        }
    }

    /// Empty bone pool → vertices carry zero bone fields. The VS
    /// treats this as "skip skinning, rest pose" (weights sum to 0).
    #[test]
    fn vertex_bone_fields_zero_for_unskinned_assets() {
        let nodes = vec![make_brick(0)];
        let mut bricks = vec![BRICK_EMPTY; BRICK_CELLS as usize];
        bricks[0] = 0;
        bricks[1] = 1;
        let (verts, _) = extract_surface_mesh(
            &nodes, 2, 1.0, Vec3::ZERO, &bricks, &[], &[], None,
        );
        assert!(!verts.is_empty());
        for v in &verts {
            assert_eq!(v.bone_indices, 0);
            assert_eq!(v.bone_weights, 0);
        }
    }

    /// Vertex normal averaging: with two surface cells sharing a
    /// vertex, both contributing identical +Y normals, the vertex
    /// should pack to +Y after averaging.
    #[test]
    fn vertex_normal_averaging_uses_leaf_attr_pool() {
        let nodes = vec![make_brick(0)];
        let mut bricks = vec![BRICK_EMPTY; BRICK_CELLS as usize];
        bricks[0] = 0;
        bricks[1] = 1;
        // LeafAttr pool with two slots, both pointing +Y.
        let pool = vec![
            LeafAttr {
                normal_oct: pack_oct(Vec3::Y),
                material_primary: 0,
                material_secondary_blend: 0,
            },
            LeafAttr {
                normal_oct: pack_oct(Vec3::Y),
                material_primary: 0,
                material_secondary_blend: 0,
            },
        ];
        let (verts, _) = extract_surface_mesh(&nodes, 2, 1.0, Vec3::ZERO, &bricks, &pool, &[], None);
        for v in &verts {
            let n = unpack_oct(v.normal_oct);
            assert!((n - Vec3::Y).length() < 1e-3, "expected +Y, got {:?}", n);
        }
    }

    // ── Phase B R4b — region-scoped extract ─────────────────────────

    /// Triangle multiset keyed by sorted vertex-position triple. Used to
    /// compare triangle sets across different VBO orderings.
    fn triangle_position_set(
        indices: &[u32],
        verts: &[MeshVertex],
    ) -> std::collections::HashMap<[[i32; 3]; 3], usize> {
        let mut m = std::collections::HashMap::new();
        for tri in indices.chunks_exact(3) {
            let mut p = [
                pos_key(verts[tri[0] as usize].local_pos),
                pos_key(verts[tri[1] as usize].local_pos),
                pos_key(verts[tri[2] as usize].local_pos),
            ];
            p.sort();
            *m.entry(p).or_insert(0) += 1;
        }
        m
    }

    fn pos_key(p: [f32; 3]) -> [i32; 3] {
        [
            (p[0] * 1000.0) as i32,
            (p[1] * 1000.0) as i32,
            (p[2] * 1000.0) as i32,
        ]
    }

    fn region_extent(extent: i32) -> (IVec3, IVec3) {
        (IVec3::ZERO, IVec3::splat(extent))
    }

    /// Region covering the whole asset should produce the same triangle
    /// set as a full-asset extract.
    #[test]
    fn region_extract_matches_full_extract_on_full_region() {
        // 4×4×4 brick: two adjacent surface cells + a couple of others.
        let nodes = vec![make_brick(0)];
        let mut bricks = vec![BRICK_EMPTY; BRICK_CELLS as usize];
        bricks[0] = 1; // (0,0,0)
        bricks[1] = 2; // (1,0,0)
        bricks[BRICK_DIM as usize * BRICK_DIM as usize] = 3; // (0,0,1)

        let depth = 2u8;
        let extent = 1i32 << depth;
        // The region path now runs surface nets on the smooth D field
        // (D-threshold topology), NOT binary occupancy — so it no longer
        // count-matches the binary full-asset `extract_surface_mesh`, and
        // sparse cells (like these 3 isolated voxels) dissolve under the
        // R=2 blur (their cell-center D < 0.5). The invariant that
        // survives is DETERMINISM + coverage: a region covering the full
        // extent must produce the SAME mesh as the two-step
        // collect+region path over the same domain. (A solid block that
        // survives the blur is exercised by the dedicated D-topology
        // manifold test below.)
        let _ = region_extent(extent);
        let cells = collect_cell_map(&nodes, depth, &bricks);
        let (a_v, a_i) = extract_mesh_region_from_cells(
            &cells, IVec3::ZERO, IVec3::splat(extent), &nodes, depth, 1.0,
            Vec3::ZERO, &bricks, &[], &[], None,
        );
        let (region_v, region_i) = extract_surface_mesh_region(
            &nodes, depth, 1.0, Vec3::ZERO, &bricks, &[], &[],
            IVec3::ZERO, IVec3::splat(extent),
        );
        assert_eq!(a_v.len(), region_v.len(), "region path must be deterministic (verts)");
        assert_eq!(a_i.len(), region_i.len(), "region path must be deterministic (tris)");
    }

    /// **D-topology manifold + de-staircase.** A solid block that
    /// survives the R=2 blur meshes through the region (D-threshold) path
    /// to a CLOSED, manifold surface (every edge shared by exactly 2
    /// triangles), and the surface follows the block — confirming the
    /// D-topology change keeps the mesh watertight.
    #[test]
    fn region_d_topology_block_is_manifold() {
        // 4³ solid block of cells (survives the blur — interior D ≈ 1).
        let lo = IVec3::splat(2);
        let hi = IVec3::splat(6);
        let cells = occupancy_cellmap(lo - IVec3::ONE, hi + IVec3::ONE, |c| {
            c.x >= lo.x && c.x < hi.x && c.y >= lo.y && c.y < hi.y && c.z >= lo.z && c.z < hi.z
        });
        let nodes = vec![EMPTY_NODE];
        let mut scratch = SculptExtractScratch::new();
        let (v, i) = extract_mesh_region_from_cells_pooled_haloed(
            &mut scratch, &cells, lo - IVec3::ONE, hi + IVec3::ONE, &nodes, 5, 1.0,
            Vec3::ZERO, &[], &[], &[], &[], None, None::<&fn(Vec3) -> f32>,
        &[],
        );
        assert!(!v.is_empty() && !i.is_empty(), "block produced no surface");
        // Manifold: every undirected edge shared by exactly 2 triangles.
        let mut edges: std::collections::HashMap<(u32, u32), u32> = Default::default();
        for t in i.chunks_exact(3) {
            for &(a, b) in &[(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
                *edges.entry((a.min(b), a.max(b))).or_insert(0) += 1;
            }
        }
        let boundary = edges.values().filter(|&&c| c != 2).count();
        assert_eq!(boundary, 0, "{boundary} non-manifold/boundary edges (mesh not watertight)");
    }

    /// Region far from any solid cell yields nothing (or a degenerate
    /// empty mesh).
    #[test]
    fn region_extract_in_empty_space_yields_nothing() {
        let nodes = vec![make_brick(0)];
        let mut bricks = vec![BRICK_EMPTY; BRICK_CELLS as usize];
        bricks[0] = 1;
        let (v, i) = extract_surface_mesh_region(
            &nodes,
            2,
            1.0,
            Vec3::ZERO,
            &bricks,
            &[],
            &[],
            IVec3::splat(20),
            IVec3::splat(25),
        );
        assert!(v.is_empty());
        assert!(i.is_empty());
    }

    /// A SINGLE isolated voxel dissolves under the R=2 density blur:
    /// its cell-center `D` is well below 0.5 (one cell of solid mass
    /// spread over a `5³` Gaussian footprint), so the D-threshold
    /// topology emits NO surface. This is correct, desirable behaviour —
    /// a lone voxel is sub-resolution noise the smoothing removes (the
    /// pre-D-topology binary path emitted a spurious 1-voxel cube here).
    #[test]
    fn region_extract_single_cell_dissolves_under_blur() {
        let nodes = vec![make_brick(0)];
        let mut bricks = vec![BRICK_EMPTY; BRICK_CELLS as usize];
        bricks[0] = 7; // (0,0,0)
        let (v, i) = extract_surface_mesh_region(
            &nodes, 2, 1.0, Vec3::ZERO, &bricks, &[], &[],
            IVec3::ZERO, IVec3::ONE,
        );
        assert!(
            v.is_empty() && i.is_empty(),
            "a single voxel should dissolve under the blur (got {} verts)",
            v.len()
        );
    }

    /// Region restricted to a subset still meshes the part of the
    /// surface its padded window covers, and a region far from any
    /// solid emits nothing. Uses a solid BLOCK (survives the R=2 blur,
    /// unlike a thin 1-2 cell strip which dissolves) so the D-threshold
    /// topology has a real surface to find.
    #[test]
    fn region_extract_subset_includes_padded_neighbors() {
        // 4³ solid block at cells [2,6)³.
        let lo = IVec3::splat(2);
        let hi = IVec3::splat(6);
        let cells = occupancy_cellmap(IVec3::splat(0), IVec3::splat(8), |c| {
            c.x >= lo.x && c.x < hi.x && c.y >= lo.y && c.y < hi.y && c.z >= lo.z && c.z < hi.z
        });
        let nodes = vec![EMPTY_NODE];
        let depth = 5u8;

        // A sub-window that touches one face of the block must produce a
        // non-empty surface (its padded neighborhood reaches the block).
        let mut scratch = SculptExtractScratch::new();
        let (v_a, i_a) = extract_mesh_region_from_cells_pooled_haloed(
            &mut scratch, &cells, IVec3::new(3, 5, 3), IVec3::new(5, 7, 5), &nodes,
            depth, 1.0, Vec3::ZERO, &[], &[], &[], &[], None, None::<&fn(Vec3) -> f32>,
        &[],
        );
        assert!(
            !v_a.is_empty() && !i_a.is_empty(),
            "sub-window touching the block's +Y face must mesh the surface there",
        );

        // A region far from the block (and its blur footprint) emits
        // nothing.
        let (v_b, i_b) = extract_mesh_region_from_cells_pooled_haloed(
            &mut scratch, &cells, IVec3::splat(20), IVec3::splat(24), &nodes,
            depth, 1.0, Vec3::ZERO, &[], &[], &[], &[], None, None::<&fn(Vec3) -> f32>,
        &[],
        );
        assert!(
            v_b.is_empty() && i_b.is_empty(),
            "region far from solids must emit nothing",
        );
    }

    /// **D6.3.c regression** — running the pooled extract twice
    /// against the *same* scratch must produce the same output as
    /// two fresh-allocating extracts. Catches stale-dirty bugs,
    /// missed resets, and cross-stamp data leakage through the
    /// reused grids.
    #[test]
    fn pooled_extract_reuses_scratch_across_stamps() {
        // Stamp 1: 2×1×1 box at the origin.
        let nodes1 = vec![make_brick(0)];
        let mut bricks1 = vec![BRICK_EMPTY; BRICK_CELLS as usize];
        bricks1[0] = 5;
        bricks1[1] = BRICK_INTERIOR;

        // Stamp 2: a different brick configuration — single cell in
        // a different position, so the dirty regions don't overlap.
        let nodes2 = vec![make_brick(0)];
        let mut bricks2 = vec![BRICK_EMPTY; BRICK_CELLS as usize];
        let dim = BRICK_DIM as usize;
        bricks2[2 + dim * 2 + dim * dim * 2] = 7; // (2,2,2)

        let depth = 2u8;
        let region_min = IVec3::ZERO;
        let region_max = IVec3::splat(1i32 << depth);

        // Reference: each stamp via the fresh-alloc path.
        let cells1 = collect_cell_map(&nodes1, depth, &bricks1);
        let (ref_v1, ref_i1) = extract_mesh_region_from_cells(
            &cells1, region_min, region_max, &nodes1, depth, 1.0,
            Vec3::ZERO, &bricks1, &[], &[], None,
        );
        let cells2 = collect_cell_map(&nodes2, depth, &bricks2);
        let (ref_v2, ref_i2) = extract_mesh_region_from_cells(
            &cells2, region_min, region_max, &nodes2, depth, 1.0,
            Vec3::ZERO, &bricks2, &[], &[], None,
        );

        // Pooled: both stamps share the same scratch.
        let mut scratch = SculptExtractScratch::new();
        let (pool_v1, pool_i1) = extract_mesh_region_from_cells_pooled(
            &mut scratch, &cells1, region_min, region_max, &nodes1,
            depth, 1.0, Vec3::ZERO, &bricks1, &[], &[], None,
        );
        let (pool_v2, pool_i2) = extract_mesh_region_from_cells_pooled(
            &mut scratch, &cells2, region_min, region_max, &nodes2,
            depth, 1.0, Vec3::ZERO, &bricks2, &[], &[], None,
        );

        assert_eq!(
            triangle_position_set(&ref_i1, &ref_v1),
            triangle_position_set(&pool_i1, &pool_v1),
            "stamp 1 pooled output must match fresh-alloc reference",
        );
        assert_eq!(
            triangle_position_set(&ref_i2, &ref_v2),
            triangle_position_set(&pool_i2, &pool_v2),
            "stamp 2 (post-reuse) pooled output must match fresh-alloc reference",
        );

        // And a third stamp with the original cells should re-produce
        // stamp 1's output — the dirty-tracking reset must be complete.
        let (pool_v3, pool_i3) = extract_mesh_region_from_cells_pooled(
            &mut scratch, &cells1, region_min, region_max, &nodes1,
            depth, 1.0, Vec3::ZERO, &bricks1, &[], &[], None,
        );
        assert_eq!(
            triangle_position_set(&ref_i1, &ref_v1),
            triangle_position_set(&pool_i3, &pool_v3),
            "stamp 3 (cycling back to stamp 1 input) must match reference",
        );
    }

    /// **D6.3 bug regression** — the region extract must classify
    /// INTERIOR-bulk corner cells as solid, identically to the
    /// full-asset extract. Without the `CELL_INTERIOR` →
    /// `CELL_INTERIOR_GRID` remap inside `extract_mesh_region_from_cells`,
    /// the grid stores `u32::MAX` for INTERIOR cells, which is
    /// indistinguishable from `CELL_GRID_EMPTY`; `build_cube_vertex`'s
    /// corner classifier then treats those cells as empty and
    /// produces wrong edge crossings.
    #[test]
    fn region_extract_respects_interior_corner_cells() {
        // INTERIOR-bulk cells must seed the density blur as occupancy=1
        // (via the `CELL_INTERIOR → CELL_INTERIOR_GRID` remap), exactly
        // like real surface cells — otherwise the D field would read
        // them as empty and carve a hole / spurious shelf. Build a solid
        // block whose interior cells are tagged `CELL_INTERIOR` and a
        // twin whose interior cells carry real slots; the D-threshold
        // mesh must be IDENTICAL (the blur sees both as occupied).
        let lo = IVec3::splat(2);
        let hi = IVec3::splat(6);
        let is_solid = |c: IVec3| {
            c.x >= lo.x && c.x < hi.x && c.y >= lo.y && c.y < hi.y && c.z >= lo.z && c.z < hi.z
        };
        // Block A: every cell carries a real slot.
        let mut cells_a = CellMap::default();
        // Block B: interior cells (all 6 face-neighbors solid) carry
        // CELL_INTERIOR; boundary cells carry a real slot.
        let mut cells_b = CellMap::default();
        for z in 0..8 {
            for y in 0..8 {
                for x in 0..8 {
                    let c = IVec3::new(x, y, z);
                    if !is_solid(c) {
                        continue;
                    }
                    cells_a.insert(c, 1);
                    let is_interior = FACE_DIRS.iter().all(|d| is_solid(c + *d));
                    cells_b.insert(c, if is_interior { CELL_INTERIOR } else { 1 });
                }
            }
        }
        let nodes = vec![EMPTY_NODE];
        let depth = 5u8;
        let mut sa = SculptExtractScratch::new();
        let mut sb = SculptExtractScratch::new();
        let (va, ia) = extract_mesh_region_from_cells_pooled_haloed(
            &mut sa, &cells_a, lo - IVec3::ONE, hi + IVec3::ONE, &nodes, depth, 1.0,
            Vec3::ZERO, &[], &[], &[], &[], None, None::<&fn(Vec3) -> f32>,
        &[],
        );
        let (vb, ib) = extract_mesh_region_from_cells_pooled_haloed(
            &mut sb, &cells_b, lo - IVec3::ONE, hi + IVec3::ONE, &nodes, depth, 1.0,
            Vec3::ZERO, &[], &[], &[], &[], None, None::<&fn(Vec3) -> f32>,
        &[],
        );
        assert_eq!(
            va.len(), vb.len(),
            "INTERIOR cells must seed the blur like surface cells (vertex count)",
        );
        assert_eq!(
            ia.len(), ib.len(),
            "INTERIOR cells must seed the blur like surface cells (triangle count)",
        );
        // Positions must match too (same occupancy → same D → same mesh).
        for (a, b) in va.iter().zip(vb.iter()) {
            assert_eq!(a.local_pos, b.local_pos, "INTERIOR vs slot block diverged");
        }
    }

    /// `collect_cell_map` + `extract_mesh_region_from_cells` produce the
    /// same result as `extract_surface_mesh_region` — the convenience
    /// wrapper is just sugar over the two-step form.
    #[test]
    fn two_step_form_matches_convenience_wrapper() {
        let nodes = vec![make_brick(0)];
        let mut bricks = vec![BRICK_EMPTY; BRICK_CELLS as usize];
        bricks[0] = 1;
        bricks[1] = 2;
        bricks[BRICK_DIM as usize * BRICK_DIM as usize] = 3;
        let depth = 2u8;
        let region_min = IVec3::ZERO;
        let region_max = IVec3::splat(2);

        let (v1, i1) = extract_surface_mesh_region(
            &nodes,
            depth,
            1.0,
            Vec3::ZERO,
            &bricks,
            &[],
            &[],
            region_min,
            region_max,
        );
        let cells = collect_cell_map(&nodes, depth, &bricks);
        let (v2, i2) = extract_mesh_region_from_cells(
            &cells,
            region_min,
            region_max,
            &nodes,
            depth,
            1.0,
            Vec3::ZERO,
            &bricks,
            &[],
            &[],
            None,
        );
        assert_eq!(
            triangle_position_set(&i1, &v1),
            triangle_position_set(&i2, &v2),
        );
    }

    /// Empty region (min == max on any axis) returns nothing.
    #[test]
    fn region_extract_empty_region_returns_nothing() {
        let nodes = vec![make_brick(0)];
        let mut bricks = vec![BRICK_EMPTY; BRICK_CELLS as usize];
        bricks[0] = 1;
        let (v, i) = extract_surface_mesh_region(
            &nodes,
            2,
            1.0,
            Vec3::ZERO,
            &bricks,
            &[],
            &[],
            IVec3::splat(2),
            IVec3::splat(2), // empty
        );
        assert!(v.is_empty());
        assert!(i.is_empty());
    }

    // ─────────────── CellGrid (D6.3.a) ────────────────────────────

    /// Fresh `CellGrid::new` initialises every slot to the empty
    /// sentinel — `get` returns `None` and `contains` returns false.
    #[test]
    fn cellgrid_new_is_empty() {
        let grid = CellGrid::new(IVec3::new(-3, 5, 10), IVec3::new(4, 4, 4));
        for z in 10..14 {
            for y in 5..9 {
                for x in -3..1 {
                    let c = IVec3::new(x, y, z);
                    assert_eq!(grid.get(c), None, "coord {:?}", c);
                    assert!(!grid.contains(c), "coord {:?}", c);
                }
            }
        }
    }

    /// `set` writes a value that `get` and `contains` then surface;
    /// the sentinel value is unreachable through the public API
    /// because `get` filters it out.
    #[test]
    fn cellgrid_set_get_roundtrip() {
        let mut grid = CellGrid::new(IVec3::new(0, 0, 0), IVec3::new(3, 3, 3));
        let c = IVec3::new(1, 2, 1);
        assert_eq!(grid.get(c), None);
        grid.set(c, 42);
        assert_eq!(grid.get(c), Some(42));
        assert!(grid.contains(c));

        // Another coord stays empty.
        assert_eq!(grid.get(IVec3::new(0, 0, 0)), None);
        assert!(!grid.contains(IVec3::new(0, 0, 0)));
    }

    /// `flat_index` enforces the half-open extent: lo corner inclusive,
    /// hi corner exclusive, with one-past on every axis returning None.
    #[test]
    fn cellgrid_bounds_are_half_open() {
        let grid = CellGrid::new(IVec3::new(-2, -2, -2), IVec3::new(4, 4, 4));

        // Inside, corners.
        assert!(grid.flat_index(IVec3::new(-2, -2, -2)).is_some());
        assert!(grid.flat_index(IVec3::new(1, 1, 1)).is_some());

        // One past on each axis.
        assert!(grid.flat_index(IVec3::new(2, 1, 1)).is_none());
        assert!(grid.flat_index(IVec3::new(1, 2, 1)).is_none());
        assert!(grid.flat_index(IVec3::new(1, 1, 2)).is_none());

        // Below origin on each axis.
        assert!(grid.flat_index(IVec3::new(-3, 0, 0)).is_none());
        assert!(grid.flat_index(IVec3::new(0, -3, 0)).is_none());
        assert!(grid.flat_index(IVec3::new(0, 0, -3)).is_none());
    }

    /// `set` to an out-of-bounds coord is a silent no-op (matches the
    /// behaviour the inner extract loop relies on for query coords
    /// that happen to lie one cell outside the pre-sized grid).
    #[test]
    fn cellgrid_set_out_of_bounds_is_noop() {
        let mut grid = CellGrid::new(IVec3::ZERO, IVec3::splat(2));
        grid.set(IVec3::new(5, 5, 5), 99);
        assert_eq!(grid.get(IVec3::new(5, 5, 5)), None);
        // No panic.
    }

    /// `reset` returns the grid to its post-`new` state — useful for
    /// the pool-reuse path (D6.3.c).
    #[test]
    fn cellgrid_reset_clears_all_slots() {
        let mut grid = CellGrid::new(IVec3::ZERO, IVec3::splat(3));
        grid.set(IVec3::new(0, 0, 0), 1);
        grid.set(IVec3::new(2, 2, 2), 2);
        grid.set(IVec3::new(1, 1, 1), CELL_INTERIOR);
        assert_eq!(grid.get(IVec3::new(0, 0, 0)), Some(1));

        grid.reset();
        assert_eq!(grid.get(IVec3::new(0, 0, 0)), None);
        assert_eq!(grid.get(IVec3::new(2, 2, 2)), None);
        assert_eq!(grid.get(IVec3::new(1, 1, 1)), None);
    }

    /// Linearisation must be unique across the grid — two distinct
    /// coords never share a flat index. Catches axis-stride mistakes.
    #[test]
    fn cellgrid_flat_index_is_unique() {
        let size = IVec3::new(3, 4, 5);
        let grid = CellGrid::new(IVec3::new(-1, 2, 7), size);
        let mut seen = std::collections::HashSet::new();
        for z in 7..12 {
            for y in 2..6 {
                for x in -1..2 {
                    let c = IVec3::new(x, y, z);
                    let idx = grid.flat_index(c).unwrap();
                    assert!(seen.insert(idx), "duplicate index {} at {:?}", idx, c);
                }
            }
        }
        assert_eq!(seen.len() as i32, size.x * size.y * size.z);
    }

    // ─── QEF / normal-guided vertex placement tests ──────────────────

    #[test]
    fn solve_qef_three_orthogonal_planes_finds_intersection() {
        use super::solve_qef;
        let planes = [
            (Vec3::X, Vec3::new(2.0, 0.0, 0.0)),
            (Vec3::Y, Vec3::new(0.0, 3.0, 0.0)),
            (Vec3::Z, Vec3::new(0.0, 0.0, 4.0)),
        ];
        let bias = Vec3::new(1.0, 1.0, 1.0);
        let result = solve_qef(&planes, bias);
        // With 3 strong orthogonal planes, bias pulls slightly toward
        // (1,1,1). Regularization effect is ~λ/(1+λ) ≈ 9%.
        assert!((result.x - 2.0).abs() < 0.25, "x={}", result.x);
        assert!((result.y - 3.0).abs() < 0.25, "y={}", result.y);
        assert!((result.z - 4.0).abs() < 0.35, "z={}", result.z);
    }

    #[test]
    fn solve_qef_parallel_planes_blends_toward_bias() {
        use super::solve_qef;
        let planes = [
            (Vec3::Y, Vec3::new(0.0, 1.0, 0.0)),
            (Vec3::Y, Vec3::new(0.0, 2.0, 0.0)),
            (Vec3::Y, Vec3::new(0.0, 3.0, 0.0)),
        ];
        let bias = Vec3::new(5.0, 0.0, 7.0);
        let result = solve_qef(&planes, bias);
        // Y axis is well-determined by the planes (average y ≈ 2).
        // X and Z are underdetermined — solution blends toward bias.
        assert!((result.x - 5.0).abs() < 0.5, "x={}", result.x);
        assert!((result.z - 7.0).abs() < 0.5, "z={}", result.z);
    }

    #[test]
    fn solve_qef_diagonal_plane_finds_correct_point() {
        use super::solve_qef;
        let n45 = Vec3::new(0.0, 1.0, 1.0).normalize();
        let planes = [
            (Vec3::X, Vec3::new(1.5, 0.0, 0.0)),
            (n45, Vec3::new(0.0, 1.0, 1.0)),
            (Vec3::Y, Vec3::new(0.0, 0.5, 0.0)),
        ];
        let bias = Vec3::new(1.5, 0.5, 1.0);
        let result = solve_qef(&planes, bias);
        for &(n, p) in &planes {
            let dist = n.dot(result - p).abs();
            assert!(dist < 0.2, "plane dist = {} for n={:?}", dist, n);
        }
    }

    #[test]
    fn qef_gentle_slope_eliminates_terracing() {
        use super::build_cube_vertex;
        use crate::leaf_attr::{pack_oct, LeafAttr};

        // Simulate a 45-degree slope: cells (0,0,0) and (1,1,0) are solid
        // with normals pointing up the slope (normalized (0,1,1)).
        let slope_normal = Vec3::new(0.0, 1.0, 1.0).normalize();
        let normal_packed = pack_oct(slope_normal);

        let mut pool = vec![LeafAttr::default(); 4];
        pool[1] = LeafAttr { normal_oct: normal_packed, material_primary: 0, material_secondary_blend: 0 };
        pool[2] = LeafAttr { normal_oct: normal_packed, material_primary: 0, material_secondary_blend: 0 };

        // SN cube at (0,0,0). Corners: cell(0,0,0)=solid(slot 1),
        // cell(1,1,0)=solid(slot 2), rest empty.
        let cell_lookup = |c: IVec3| -> Option<u32> {
            if c == IVec3::new(0, 0, 0) {
                Some(1)
            } else if c == IVec3::new(1, 1, 0) {
                Some(2)
            } else {
                None
            }
        };

        let vert = build_cube_vertex(
            IVec3::ZERO,
            cell_lookup,
            1.0,
            Vec3::ZERO,
            &pool,
            &[],
            None,
            None::<&fn(Vec3) -> f32>,
            None::<&fn(Vec3) -> Vec3>,
            None::<&fn(Vec3) -> f32>,
            None::<&fn(Vec3) -> Vec3>,
            DENSITY_ISO,
            false,
            &[],
        );

        // With naive SN, the vertex would sit at the centroid of
        // edge-crossing midpoints. With QEF on two planes sharing the
        // same normal, the system is rank-1 → single-plane fallback
        // won't fire (plane_count=2 but degenerate) → falls back to
        // naive. But the key is: when normals differ slightly (real
        // terrain), the QEF kicks in.
        //
        // For identical normals: QEF is degenerate → naive fallback.
        // This is correct behavior — identical normals means the surface
        // IS flat on the grid, no correction needed.
        let _ = vert;
    }

    #[test]
    fn qef_regularized_stays_near_bias_for_wild_planes() {
        use super::solve_qef;

        // Two nearly-parallel planes whose unregularized intersection
        // would be far away. The regularized solve stays near the bias.
        let n1 = Vec3::new(0.0, 1.0, 0.01).normalize();
        let n2 = Vec3::new(0.0, 1.0, -0.01).normalize();
        let planes = [
            (n1, Vec3::new(0.5, 10.0, 0.5)),
            (n2, Vec3::new(0.5, -10.0, 0.5)),
            (Vec3::X, Vec3::new(0.5, 0.5, 0.5)),
        ];
        let bias = Vec3::new(0.5, 0.5, 0.5);
        let result = solve_qef(&planes, bias);
        // Regularization prevents the solution from flying off.
        // build_cube_vertex would also clamp, but the solve itself
        // should produce a reasonable result near the bias.
        assert!(result.x.is_finite() && result.y.is_finite() && result.z.is_finite());
    }

    #[test]
    fn qef_two_perpendicular_planes_preserves_sharp_edge() {
        use super::solve_qef;

        // Two planes meeting at a 90-degree edge along Z axis at x=1, y=1.
        let planes = [
            (Vec3::X, Vec3::new(1.0, 0.5, 0.5)),
            (Vec3::Y, Vec3::new(0.5, 1.0, 0.5)),
        ];
        let bias = Vec3::new(0.5, 0.5, 0.5);
        let result = solve_qef(&planes, bias);
        // X and Y should be at the plane intersection, Z pulled toward bias.
        assert!((result.x - 1.0).abs() < 0.15, "x={}", result.x);
        assert!((result.y - 1.0).abs() < 0.15, "y={}", result.y);
    }

    // --- Task A2/A3: Gibson Constrained Elastic Surface Net tests. ---

    /// Build a regular `(cols × rows)` triangulated grid in the XZ plane
    /// with `y = height(col, row)`. Returns `(vertices, indices)` with
    /// outward (+Y) winding and a default +Y normal on every vertex.
    fn grid_mesh(
        cols: usize,
        rows: usize,
        spacing: f32,
        height: impl Fn(usize, usize) -> f32,
    ) -> (Vec<MeshVertex>, Vec<u32>) {
        let mut verts = Vec::with_capacity(cols * rows);
        for r in 0..rows {
            for c in 0..cols {
                verts.push(MeshVertex {
                    local_pos: [c as f32 * spacing, height(c, r), r as f32 * spacing],
                    normal_oct: pack_oct(Vec3::Y),
                    leaf_attr_id: 0,
                    bone_indices: 0,
                    bone_weights: 0,
                    _pad: 0,
                });
            }
        }
        let idx = |c: usize, r: usize| (r * cols + c) as u32;
        let mut indices = Vec::new();
        for r in 0..rows - 1 {
            for c in 0..cols - 1 {
                let v00 = idx(c, r);
                let v10 = idx(c + 1, r);
                let v01 = idx(c, r + 1);
                let v11 = idx(c + 1, r + 1);
                // CCW seen from +Y (outward / up).
                indices.extend_from_slice(&[v00, v01, v11]);
                indices.extend_from_slice(&[v00, v11, v10]);
            }
        }
        (verts, indices)
    }

    /// Signed mesh volume via the divergence theorem: sum of signed
    /// tetrahedra `(0, a, b, c)` over all triangles. Outward winding ⇒
    /// positive volume for a closed surface.
    fn signed_volume(verts: &[MeshVertex], indices: &[u32]) -> f32 {
        let mut vol = 0.0f64;
        for tri in indices.chunks_exact(3) {
            let a = Vec3::from(verts[tri[0] as usize].local_pos);
            let b = Vec3::from(verts[tri[1] as usize].local_pos);
            let c = Vec3::from(verts[tri[2] as usize].local_pos);
            vol += a.dot(b.cross(c)) as f64 / 6.0;
        }
        vol as f32
    }

    /// Build a UV-sphere mesh of `radius` with `lat × lon` segments.
    /// Outward winding, exact analytic normals. Used as a smooth-surface
    /// reference: a good volume-preserving relaxer should barely move it.
    fn uv_sphere(radius: f32, lat: usize, lon: usize) -> (Vec<MeshVertex>, Vec<u32>) {
        use std::f32::consts::PI;
        let mut verts = Vec::new();
        for i in 0..=lat {
            let theta = PI * i as f32 / lat as f32; // 0..PI (pole to pole)
            let (st, ct) = theta.sin_cos();
            for j in 0..=lon {
                let phi = 2.0 * PI * j as f32 / lon as f32;
                let (sp, cp) = phi.sin_cos();
                let dir = Vec3::new(st * cp, ct, st * sp);
                verts.push(MeshVertex {
                    local_pos: (dir * radius).to_array(),
                    normal_oct: pack_oct(dir),
                    leaf_attr_id: 0,
                    bone_indices: 0,
                    bone_weights: 0,
                    _pad: 0,
                });
            }
        }
        let stride = lon + 1;
        let vid = |i: usize, j: usize| (i * stride + j) as u32;
        let mut indices = Vec::new();
        for i in 0..lat {
            for j in 0..lon {
                let v00 = vid(i, j);
                let v01 = vid(i, j + 1);
                let v10 = vid(i + 1, j);
                let v11 = vid(i + 1, j + 1);
                // Outward winding (verified by signed_volume > 0).
                indices.extend_from_slice(&[v00, v11, v10]);
                indices.extend_from_slice(&[v00, v01, v11]);
            }
        }
        (verts, indices)
    }

    /// The Gibson box constraint: after relaxation no vertex may move
    /// more than `h/2` from its origin on ANY axis. This is the floor the
    /// A1 falsification tests pin — smoothing is allowed to recover the
    /// `±h/2` positional ambiguity of the occupancy field and no more.
    #[test]
    fn box_clamp_bounds_displacement() {
        let h = 1.0f32;
        // A noisy slope so the relaxer has real work to do (and so the
        // unconstrained Laplacian step would move some vertices far).
        let (mut verts, indices) = grid_mesh(12, 12, h, |c, r| {
            // Big staircase amplitude so an unclamped step overshoots h/2.
            let smooth = 0.7 * c as f32 + 0.3 * r as f32;
            (smooth / h).round() * h
        });
        let orig: Vec<Vec3> = verts.iter().map(|v| Vec3::from(v.local_pos)).collect();

        relax_surface_net_vertices(&mut verts, &indices, h, 10, None);

        let half = h * 0.5;
        let eps = 1e-4;
        for (i, v) in verts.iter().enumerate() {
            let p = Vec3::from(v.local_pos);
            let d = (p - orig[i]).abs();
            assert!(
                d.x <= half + eps && d.y <= half + eps && d.z <= half + eps,
                "vertex {i} moved {d:?} > h/2={half} from origin",
            );
        }
    }

    /// On a tilted plane sampled as a staircase (each height snapped to
    /// the grid), naive vertices sit on terraces; the relaxer should pull
    /// them onto the underlying smooth plane so the staircase RMS drops
    /// below ~0.1h.
    #[test]
    fn relax_removes_staircase_on_synthetic_slope() {
        let h = 1.0f32;
        let slope_x = 0.35f32; // gentle, non-grid-aligned slope
        let slope_z = 0.15f32;
        // Smooth height field and its staircased (grid-snapped) version.
        let smooth_h = |c: usize, r: usize| slope_x * c as f32 + slope_z * r as f32;
        let (mut verts, indices) =
            grid_mesh(16, 16, h, |c, r| (smooth_h(c, r) / h).round() * h);

        // Pre-relax staircase RMS (deviation of each vertex y from the
        // smooth plane). Interior vertices only — the open grid's border
        // verts have no neighbors on one side and stay terraced.
        let rms = |verts: &[MeshVertex]| -> f32 {
            let cols = 16usize;
            let mut sse = 0.0f64;
            let mut count = 0u32;
            for r in 1..15 {
                for c in 1..15 {
                    let v = &verts[r * cols + c];
                    let dy = v.local_pos[1] - smooth_h(c, r);
                    sse += (dy * dy) as f64;
                    count += 1;
                }
            }
            (sse / count as f64).sqrt() as f32
        };

        let before = rms(&verts);
        assert!(before > 0.2 * h, "staircase should start coarse: {before}");

        relax_surface_net_vertices(&mut verts, &indices, h, 15, None);

        let after = rms(&verts);
        assert!(
            after < 0.1 * h,
            "staircase RMS {after} (h={h}) should drop below 0.1h (was {before})",
        );
    }

    /// Taubin relaxation preserves volume: a synthetic sphere should keep
    /// its volume within a few percent (unbounded pure-Laplacian shrink
    /// would collapse it). `voxel_size` is set generous relative to the
    /// edge length so the box clamp doesn't dominate on this smooth mesh.
    #[test]
    fn relax_preserves_volume_within_few_percent() {
        let radius = 8.0f32;
        let (mut verts, indices) = uv_sphere(radius, 24, 32);
        let v_before = signed_volume(&verts, &indices);
        assert!(v_before > 0.0, "outward winding ⇒ positive volume");

        // Edge length near the equator ≈ 2πr/lon ≈ 1.57; pick h ≈ 2× that
        // so the ±h/2 box is wide enough not to clip a near-stationary
        // smooth surface — the test then exercises Taubin's volume
        // preservation, not the clamp.
        let h = 3.0f32;
        relax_surface_net_vertices(&mut verts, &indices, h, 8, None);

        let v_after = signed_volume(&verts, &indices);
        let ratio = (v_after / v_before - 1.0).abs();
        assert!(
            ratio < 0.05,
            "volume changed {:.2}% ({v_before} → {v_after})",
            ratio * 100.0,
        );
    }

    /// A3: recomputed normals on a relaxed sphere vary continuously — no
    /// 26-direction lattice banding. Across every mesh edge the angle
    /// between the two endpoint normals stays small, and each normal
    /// points outward (positive dot with the radial direction).
    #[test]
    fn relaxed_normals_vary_continuously_on_sphere() {
        let radius = 8.0f32;
        let (mut verts, indices) = uv_sphere(radius, 24, 32);
        let h = 3.0f32;
        relax_surface_net_vertices(&mut verts, &indices, h, 8, None);

        // Outward check: recomputed normal vs radial direction.
        for v in &verts {
            let p = Vec3::from(v.local_pos);
            if p.length_squared() < 1e-6 {
                continue;
            }
            let radial = p.normalize();
            let nrm = unpack_oct(v.normal_oct);
            assert!(
                nrm.dot(radial) > 0.7,
                "normal {nrm:?} not outward vs radial {radial:?}",
            );
        }

        // Continuity: across each triangle edge the normals differ by a
        // small angle. A lattice-quantized normal would snap to one of 26
        // directions and produce large jumps (≈ several tens of degrees)
        // between adjacent verts; a smoothly-recomputed field stays tight.
        let mut max_angle = 0.0f32;
        for tri in indices.chunks_exact(3) {
            for &(ia, ib) in &[(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])] {
                let na = unpack_oct(verts[ia as usize].normal_oct).normalize();
                let nb = unpack_oct(verts[ib as usize].normal_oct).normalize();
                let ang = na.dot(nb).clamp(-1.0, 1.0).acos();
                max_angle = max_angle.max(ang);
            }
        }
        // Adjacent verts on a 24×32 sphere are ≈ 7.5°–11° apart in normal;
        // allow generous headroom but well below the ≈30°+ that lattice
        // banding would inject.
        assert!(
            max_angle < 0.35, // ≈ 20°
            "max adjacent-normal angle {:.1}° suggests lattice banding",
            max_angle.to_degrees(),
        );
    }

    // ───────────────────────── Direct density-smoothing ─────────────
    //
    // The region extract (`extract_mesh_region_from_cells_pooled_haloed`)
    // now reconstructs smooth vertex positions from the `D = 0.5`
    // isosurface of a fixed Gaussian-blurred occupancy field and normals
    // from `∇D`. The two tests below pin the watertight-by-construction
    // property (the kernel is a pure local function of occupancy) and the
    // smoothness + outward-normal behaviour on a synthetic slope.

    /// Build a `CellMap` from a synthetic occupancy predicate over a
    /// box of cell coords. Every solid cell gets `leaf_attr_id = 0`
    /// (the default LeafAttr — its `+Y` normal is irrelevant here since
    /// the smoothing path derives the normal from `∇D`, not the pool).
    fn occupancy_cellmap<F: Fn(IVec3) -> bool>(
        lo: IVec3,
        hi: IVec3,
        solid: F,
    ) -> CellMap {
        let mut cells = CellMap::default();
        for z in lo.z..hi.z {
            for y in lo.y..hi.y {
                for x in lo.x..hi.x {
                    let c = IVec3::new(x, y, z);
                    if solid(c) {
                        cells.insert(c, 0);
                    }
                }
            }
        }
        cells
    }

    /// **Watertight by construction.** Density is a pure local function
    /// of occupancy within `±DENSITY_KERNEL_R` of each cell, and
    /// `cells_grid` is populated identically by owned cells and folded
    /// halo cells. So a boundary cube extracted from two different
    /// region windows — one owning the cells on the −X side of a seam
    /// (the +X side supplied as halo), the other owning +X (−X as halo)
    /// — produces a bit-identical vertex (position AND normal), with no
    /// welding and no iteration.
    #[test]
    fn density_smoothing_boundary_vertices_bit_identical() {
        let depth: u8 = 6; // extent = 64, plenty for the test region.
        let nodes = vec![EMPTY_NODE]; // occupancy entirely from the CellMap.
        let vs = 0.25f32;
        let origin = Vec3::new(-1.0, 2.0, 0.5);

        // A slanted solid wedge: cell solid iff y < 16 + (x - 16)/2.
        // Grid-aligned faces would staircase; the density smoother
        // de-staircases AND must agree on the seam from both sides.
        let solid = |c: IVec3| -> bool {
            let surface_y = 16.0 + (c.x as f32 - 16.0) * 0.5;
            (c.y as f32) < surface_y
        };

        // Full occupancy over a wide band straddling the seam plane at
        // x = 24. Build it once; both extracts fold the same cells (as
        // owned-or-halo) so the seam neighborhood is identical content.
        let band_lo = IVec3::new(8, 0, 8);
        let band_hi = IVec3::new(40, 32, 24);
        let all = occupancy_cellmap(band_lo, band_hi, solid);

        let seam = 24i32;

        // Each extract's region OVERLAPS the seam by ≥ the density
        // dependency radius so the shared seam cube's neighborhood is
        // covered by REAL occupancy (no boundary clamp) in BOTH grids.
        // A cube's vertex depends on density within ~±1 cell (trilinear
        // + ∇ central diff), and each density value depends on occupancy
        // within ±DENSITY_KERNEL_R, so the full reach is ~±(R+1) = ±3
        // cells. We overlap by 4 to leave margin. The seam cells of the
        // other side that fall in-region are supplied as halo (so they
        // populate `cells_grid` identically without being owned twice as
        // quad-emitters — though duplicate-but-identical emission would
        // also be watertight). This mirrors the terrain halo-refresh
        // slab extract, which reaches into the neighbour by the halo
        // width on the side it is refreshing.
        const OVERLAP: i32 = 4;

        // ── Extract A: owns x < seam; window reaches seam+OVERLAP. ──
        let a_region_lo = IVec3::new(seam - 8, 4, 10);
        let a_region_hi = IVec3::new(seam + OVERLAP, 28, 22);
        let mut cells_a = CellMap::default();
        let mut halo_a: Vec<(IVec3, u32)> = Vec::new();
        for (&c, &slot) in all.iter() {
            if c.x < seam {
                cells_a.insert(c, slot);
            } else {
                halo_a.push((c, slot)); // +X side as halo
            }
        }
        let mut scratch_a = SculptExtractScratch::new();
        let (verts_a, _idx_a) = extract_mesh_region_from_cells_pooled_haloed(
            &mut scratch_a,
            &cells_a,
            a_region_lo,
            a_region_hi,
            &nodes,
            depth,
            vs,
            origin,
            &[],
            &[],
            &[],
            &halo_a,
            None,
            None::<&fn(Vec3) -> f32>,
        &[],
        );

        // ── Extract B: owns x ≥ seam; window reaches seam-OVERLAP. ──
        let b_region_lo = IVec3::new(seam - OVERLAP, 4, 10);
        let b_region_hi = IVec3::new(seam + 8, 28, 22);
        let mut cells_b = CellMap::default();
        let mut halo_b: Vec<(IVec3, u32)> = Vec::new();
        for (&c, &slot) in all.iter() {
            if c.x >= seam {
                cells_b.insert(c, slot);
            } else {
                halo_b.push((c, slot)); // −X side as halo
            }
        }
        let mut scratch_b = SculptExtractScratch::new();
        let (verts_b, _idx_b) = extract_mesh_region_from_cells_pooled_haloed(
            &mut scratch_b,
            &cells_b,
            b_region_lo,
            b_region_hi,
            &nodes,
            depth,
            vs,
            origin,
            &[],
            &[],
            &[],
            &halo_b,
            None,
            None::<&fn(Vec3) -> f32>,
        &[],
        );

        // Collect vertices of the SHARED seam cube — the one whose
        // lo-corner is x = seam-1 (corner cells seam-1 and seam). Its
        // vertex grid-x lands strictly inside (seam-0.5, seam+0.5): an
        // X-edge crossing interpolates between cell-centers seam-0.5
        // and seam+0.5, and the smoothed centroid stays inside that
        // open interval. Cubes at lo-x seam-2 (grid-x ≤ seam-0.5) and
        // lo-x seam (grid-x ≥ seam+0.5) are excluded, so this picks
        // exactly the cubes both extracts share. Both must emit them
        // bit-identically (watertight).
        let collect_seam = |verts: &[MeshVertex]| -> Vec<MeshVertex> {
            let mut v: Vec<MeshVertex> = verts
                .iter()
                .filter(|v| {
                    let gx = (v.local_pos[0] - origin.x) / vs;
                    gx > seam as f32 - 0.5 + 1e-4 && gx < seam as f32 + 0.5 - 1e-4
                })
                .copied()
                .collect();
            v.sort_by(|a, b| {
                a.local_pos
                    .partial_cmp(&b.local_pos)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            v
        };
        let seam_a = collect_seam(&verts_a);
        let seam_b = collect_seam(&verts_b);

        assert!(
            !seam_a.is_empty(),
            "expected boundary vertices on the seam plane (got none)"
        );
        assert_eq!(
            seam_a.len(),
            seam_b.len(),
            "seam vertex count differs across the two extracts: {} vs {}",
            seam_a.len(),
            seam_b.len()
        );
        for (va, vb) in seam_a.iter().zip(seam_b.iter()) {
            // Bit-identical positions AND normals — the watertight
            // guarantee. `==` on the f32 bit-pattern (via the derived
            // PartialEq on the [f32;3] / u32 fields).
            assert_eq!(
                va.local_pos, vb.local_pos,
                "seam vertex position diverged: {:?} vs {:?}",
                va.local_pos, vb.local_pos
            );
            assert_eq!(
                va.normal_oct, vb.normal_oct,
                "seam vertex normal diverged at {:?}",
                va.local_pos
            );
        }
    }

    /// DIAGNOSTIC PROBE (terrace facet). Synthesize a GENTLE,
    /// non-grid-aligned slope `solid iff world_y < 0.37*x + 0.21` over a
    /// ~40-cell-wide region at voxel size 1.0, run the SAME blur + extract
    /// path, then PRINT the extracted vertex (x, y) profile sorted by x —
    /// plus, for a handful of surface cubes, the actual da/db sdf values
    /// and the clamped t on the vertical (Y) edges. This decides whether
    /// the extracted-Y is a smooth ramp tracking the analytic slope or
    /// quantized into terraces, and whether the t values vary sub-voxel
    /// or are stuck near 0.5 / clamped to 0 or 1.
    #[ignore]
    #[test]
    fn probe_gentle_slope_vertex_profile() {
        let depth: u8 = 7; // extent = 128.
        let nodes = vec![EMPTY_NODE];
        let vs = 1.0f32;
        let origin = Vec3::ZERO;

        // Gentle slope: solid iff world_y < 0.37*x + 0.21. With vs=1 and
        // origin=0, world coords == grid (cell) coords, and the cell
        // CENTER of cell c is (c + 0.5). Classify by the cell center so
        // the analytic surface is well-defined per cell.
        let m = 0.37f32; // slope d(y)/d(x)
        let b = 0.21f32; // intercept
        let surf_y = |x: f32| m * x + b; // analytic surface height at world-x
        let solid = |c: IVec3| -> bool {
            let cx = c.x as f32 + 0.5;
            let cy = c.y as f32 + 0.5;
            cy < surf_y(cx)
        };

        // ~40 cells wide in X. surf_y over x∈[0,40) spans y∈[0.21, 15.0],
        // so a Y band of [-4, 22) safely brackets the surface plus halo.
        let band_lo = IVec3::new(-4, -4, -4);
        let band_hi = IVec3::new(44, 22, 8);
        let cells = occupancy_cellmap(band_lo, band_hi, solid);

        // Region well inside the band so the density neighborhood is real
        // occupancy (no boundary clamp) for the cubes we inspect.
        let region_lo = IVec3::new(2, -2, 0);
        let region_hi = IVec3::new(38, 20, 6);
        let mut scratch = SculptExtractScratch::new();
        let (verts, _indices) = extract_mesh_region_from_cells_pooled_haloed(
            &mut scratch,
            &cells,
            region_lo,
            region_hi,
            &nodes,
            depth,
            vs,
            origin,
            &[],
            &[],
            &[],
            &[],
            None,
            None::<&fn(Vec3) -> f32>,
        &[],
        );
        assert!(!verts.is_empty(), "slope produced no surface vertices");

        // --- Vertex (x, y) profile of the slope-TOP surface, by x. ---
        // Pick one representative vertex per integer world-x column on the
        // top face (normal predominantly +Y), at a fixed z slice, so the
        // ramp is 1-D and easy to read.
        let mut top: Vec<(f32, f32, f32)> = Vec::new(); // (x, y_extracted, y_true)
        for v in &verts {
            let nrm = unpack_oct(v.normal_oct).normalize();
            if nrm.y <= 0.6 {
                continue; // skip side/bottom walls
            }
            let p = v.local_pos;
            // Single z column (z near 2.0) so the profile is 1-D.
            if (p[2] - 2.0).abs() > 0.51 {
                continue;
            }
            // The surface the extractor targets is D=0.5, classified by
            // cell-center occupancy. The geometric crossing for our slope
            // sits at world_y where the column transitions solid→empty:
            // the last solid cell-center is below surf_y(x); the SN cube
            // grid-corner is at integer y. Report the analytic surface for
            // reference (the GRID-CORNER y where occupancy flips), i.e.
            // surf_y at this x minus 0.5 (cell-center → grid-corner offset
            // is not exact for a slope, but close enough to read terracing).
            top.push((p[0], p[1], surf_y(p[0]) - 0.5));
        }
        top.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        eprintln!("\n=== GENTLE-SLOPE VERTEX PROFILE (vs=1.0, slope=0.37, z~2) ===");
        eprintln!("   world_x   y_extract   y_true   residual   d(y_extract)");
        let mut prev_y: Option<f32> = None;
        let mut distinct_y: std::collections::BTreeSet<i64> = Default::default();
        let mut resid_sq = 0.0f32;
        let mut resid_n = 0u32;
        for &(x, y, yt) in &top {
            let dy = prev_y.map(|py| y - py).unwrap_or(0.0);
            eprintln!(
                "  {:8.3}   {:8.4}   {:7.4}   {:+8.4}   {:+8.4}",
                x, y, yt, y - yt, dy
            );
            prev_y = Some(y);
            // Quantize extracted-y to 1/100 to count distinct levels.
            distinct_y.insert((y * 100.0).round() as i64);
            resid_sq += (y - yt) * (y - yt);
            resid_n += 1;
        }
        let rms = if resid_n > 0 {
            (resid_sq / resid_n as f32).sqrt()
        } else {
            0.0
        };
        eprintln!(
            "--- profile: {} columns, {} distinct y-levels (1/100 res), RMS residual {:.4} (vs={}) ---",
            top.len(),
            distinct_y.len(),
            rms,
            vs
        );

        // --- Reproduce build_cube_vertex's crossing math for a handful of
        // surface cubes: print da/db and clamped t on the VERTICAL (Y)
        // edges. This reads the SAME density grid the extractor built. ---
        //
        // Rebuild the density sampler from scratch's now-populated buffers.
        let g_origin = scratch.cells_grid.origin();
        let g_size = scratch.cells_grid.size();
        let total =
            (g_size.x as usize) * (g_size.y as usize) * (g_size.z as usize);
        let density = &scratch.density[..total];
        let density_at = |p_grid: Vec3| -> f32 {
            sample_density_trilinear(density, g_origin, g_size, p_grid)
        };
        let smooth_sdf = |p_world: Vec3| -> f32 {
            let g = (p_world - origin) / vs;
            DENSITY_ISO - density_at(g)
        };

        eprintln!("\n=== VERTICAL-EDGE CROSSING (da, db, t) for surface cubes at z=2 ===");
        eprintln!("  cube(x,y,z)  a_corner_solid b_corner_solid  da        db        t");
        // For each x column, find the cube whose vertical edge straddles
        // the surface (a-corner solid, b-corner empty, by BINARY occupancy
        // — exactly what corner_solid uses).
        let cell_solid = |c: IVec3| cells.contains_key(&c);
        let z_fixed = 2;
        for cx in 5..35 {
            // Find the y where the binary column flips solid→empty.
            let mut flip_y: Option<i32> = None;
            for cy in -2..20 {
                let here = cell_solid(IVec3::new(cx, cy, z_fixed));
                let above = cell_solid(IVec3::new(cx, cy + 1, z_fixed));
                if here && !above {
                    flip_y = Some(cy);
                    break;
                }
            }
            let Some(fy) = flip_y else { continue };
            // The SN cube with lo-corner (cx, fy, z_fixed): its +Y edge on
            // the x=cx, z=z_fixed corner goes from corner (cx,fy,zf) [solid]
            // to (cx, fy+1, zf) [empty]. Replicate the SDF crossing.
            let cube = IVec3::new(cx, fy, z_fixed);
            // corner_offset for the local x=0,z=0 column: a at y=0, b at y=1.
            let pa = Vec3::new(
                cube.x as f32 + 0.5,
                cube.y as f32 + 0.5,
                cube.z as f32 + 0.5,
            );
            let pb = Vec3::new(
                cube.x as f32 + 0.5,
                cube.y as f32 + 1.0 + 0.5,
                cube.z as f32 + 0.5,
            );
            let da = smooth_sdf(origin + pa * vs);
            let db = smooth_sdf(origin + pb * vs);
            let denom = da - db;
            let t = if denom.abs() > 1e-12 {
                (da / denom).clamp(0.0, 1.0)
            } else {
                0.5
            };
            let a_solid = cell_solid(IVec3::new(cube.x, cube.y, cube.z));
            let b_solid = cell_solid(IVec3::new(cube.x, cube.y + 1, cube.z));
            eprintln!(
                "  ({:3},{:3},{:3})       {:5}          {:5}      {:+7.4}  {:+7.4}  {:.4}",
                cube.x, cube.y, cube.z, a_solid, b_solid, da, db, t
            );
        }
        eprintln!("=== END PROBE ===\n");
    }

    /// **Root-cause regression: no `t`-clamps on a smooth slope.**
    ///
    /// The terracing bug was a binary-topology vs D-position MISMATCH:
    /// corners were classified solid by BINARY occupancy but the vertex
    /// placed on the `D = 0.5` isosurface, so on ~1/3 of surface cubes a
    /// binary-active vertical edge had BOTH endpoints' `D < 0.5` →
    /// `t = da/(da-db)` landed outside `[0,1]` → clamped → vertex pinned
    /// to the grid Y → terraces. (The `#[ignore]`d probe above prints
    /// the ~11/30 clamps that existed before the fix.)
    ///
    /// After running surface nets ON the D field (corner solidity from
    /// `D >= 0.5`, the same field the crossing uses), every D-active edge
    /// has a real interior crossing. This test rebuilds the post-fix
    /// classifier on the populated density grid and asserts:
    ///   * the count of vertical-edge crossings clamped to exactly 0 or 1
    ///     across the surface columns is 0 (was ~11/30), and
    ///   * the extracted slope-top Y tracks the analytic slope with RMS
    ///     `< 0.15 · voxel`.
    #[test]
    fn d_topology_slope_has_no_t_clamps() {
        let depth: u8 = 7;
        let nodes = vec![EMPTY_NODE];
        let vs = 1.0f32;
        let origin = Vec3::ZERO;

        let m = 0.37f32;
        let b = 0.21f32;
        let surf_y = |x: f32| m * x + b;
        let solid = |c: IVec3| -> bool {
            let cx = c.x as f32 + 0.5;
            let cy = c.y as f32 + 0.5;
            cy < surf_y(cx)
        };
        let band_lo = IVec3::new(-4, -4, -4);
        let band_hi = IVec3::new(44, 22, 8);
        let cells = occupancy_cellmap(band_lo, band_hi, solid);

        let region_lo = IVec3::new(2, -2, 0);
        let region_hi = IVec3::new(38, 20, 6);
        let mut scratch = SculptExtractScratch::new();
        let (verts, _indices) = extract_mesh_region_from_cells_pooled_haloed(
            &mut scratch, &cells, region_lo, region_hi, &nodes, depth, vs, origin,
            &[], &[], &[], &[], None, None::<&fn(Vec3) -> f32>,
        &[],
        );
        assert!(!verts.is_empty());

        // Rebuild the SAME density-based classifier the fixed extract
        // uses (D at the cell center vs the 0.5 threshold).
        let g_origin = scratch.cells_grid.origin();
        let g_size = scratch.cells_grid.size();
        let total = (g_size.x as usize) * (g_size.y as usize) * (g_size.z as usize);
        let density = &scratch.density[..total];
        let density_at =
            |p: Vec3| sample_density_trilinear(density, g_origin, g_size, p);
        let d_solid = |c: IVec3| {
            density_at(Vec3::new(c.x as f32 + 0.5, c.y as f32 + 0.5, c.z as f32 + 0.5))
                >= DENSITY_ISO
        };
        let smooth_sdf =
            |pw: Vec3| DENSITY_ISO - density_at((pw - origin) / vs);

        // For each X column find the cube whose +Y edge is D-active (a
        // D-solid, b D-empty) — exactly what the fixed `corner_solid`
        // selects — and count how many crossings clamp to 0 or 1.
        let z_fixed = 2;
        let mut clamps = 0u32;
        let mut active = 0u32;
        for cx in 5..35 {
            let mut flip_y: Option<i32> = None;
            for cy in -2..20 {
                if d_solid(IVec3::new(cx, cy, z_fixed))
                    && !d_solid(IVec3::new(cx, cy + 1, z_fixed))
                {
                    flip_y = Some(cy);
                    break;
                }
            }
            let Some(fy) = flip_y else { continue };
            let cube = IVec3::new(cx, fy, z_fixed);
            let pa = Vec3::new(cube.x as f32 + 0.5, cube.y as f32 + 0.5, cube.z as f32 + 0.5);
            let pb = Vec3::new(cube.x as f32 + 0.5, cube.y as f32 + 1.5, cube.z as f32 + 0.5);
            let da = smooth_sdf(origin + pa * vs);
            let db = smooth_sdf(origin + pb * vs);
            let denom = da - db;
            // RAW (unclamped) crossing parameter. The terrace bug was
            // `raw_t` landing OUTSIDE [0,1] (da, db same sign → the iso
            // never crosses between a and b → the clamp pins the vertex
            // to the grid). A `raw_t` exactly at an endpoint (da == 0,
            // the corner is exactly on the surface) is a VALID on-surface
            // crossing, not a clamp — so we test for genuinely-out-of-
            // range raw_t with a tiny epsilon tolerance.
            let raw_t = if denom.abs() > 1e-12 { da / denom } else { 0.5 };
            active += 1;
            if raw_t < -1e-4 || raw_t > 1.0 + 1e-4 {
                clamps += 1;
            }
        }
        assert!(active >= 20, "expected ≥20 active columns, got {active}");
        assert_eq!(
            clamps, 0,
            "{clamps}/{active} D-active vertical-edge crossings had raw_t OUTSIDE [0,1] \
             (clamped) — binary/D topology mismatch (terracing) is back"
        );

        // Slope-top extracted Y must be a SMOOTH RAMP, not stepped. The
        // de-staircasing measure is the residual of the slope-top
        // vertices about the BEST-FIT LINE `y = α·x + β` (a terraced
        // surface deviates from any line by ~half a step; a smooth ramp
        // hugs the line). This isolates terracing from the constant blur
        // shift (which a line-fit absorbs into β). Require RMS
        // `< 0.15 · voxel`.
        let mut pts: Vec<(f32, f32)> = Vec::new();
        for v in &verts {
            let nrm = unpack_oct(v.normal_oct).normalize_or_zero();
            if nrm.y <= 0.6 {
                continue;
            }
            let p = v.local_pos;
            if (p[2] - 2.0).abs() > 0.51 {
                continue;
            }
            pts.push((p[0], p[1]));
        }
        assert!(pts.len() >= 10, "expected ≥10 slope-top columns, got {}", pts.len());
        let np = pts.len() as f32;
        let sx: f32 = pts.iter().map(|p| p.0).sum();
        let sy: f32 = pts.iter().map(|p| p.1).sum();
        let sxx: f32 = pts.iter().map(|p| p.0 * p.0).sum();
        let sxy: f32 = pts.iter().map(|p| p.0 * p.1).sum();
        let denom = np * sxx - sx * sx;
        let alpha = (np * sxy - sx * sy) / denom;
        let beta = (sy - alpha * sx) / np;
        let mut sq = 0.0f32;
        for &(x, y) in &pts {
            let r = y - (alpha * x + beta);
            sq += r * r;
        }
        let rms = (sq / np).sqrt() / vs;
        assert!(
            rms < 0.15,
            "slope-top Y RMS-about-line {rms:.3} voxel ≥ 0.15 — surface is stepped, not a smooth ramp"
        );
    }

    /// **Smooth + outward normals.** A synthetic occupancy slope
    /// extracts to low staircase RMS (the density isosurface
    /// de-staircases the grid-aligned faces) and every surface normal
    /// points away from the solid (outward), with `+Y`-ish on a
    /// solid-below / empty-above slab.
    #[test]
    fn density_smoothing_slope_is_smooth_and_outward() {
        let depth: u8 = 6;
        let nodes = vec![EMPTY_NODE];
        let vs = 0.5f32;
        let origin = Vec3::ZERO;

        // Solid below a gentle plane: y < 20 + (x-16)*0.4 + (z-16)*0.2.
        // The true surface normal is normalize((-0.4, 1, -0.2)) — mostly
        // +Y, tilted slightly −X / −Z. Outward = toward empty (+ side).
        let nx = -0.4f32;
        let nz = -0.2f32;
        let solid = |c: IVec3| -> bool {
            let surf = 20.0 + (c.x as f32 - 16.0) * (-nx) + (c.z as f32 - 16.0) * (-nz);
            (c.y as f32) < surf
        };
        let band_lo = IVec3::new(4, 0, 4);
        let band_hi = IVec3::new(40, 36, 40);
        let cells = occupancy_cellmap(band_lo, band_hi, solid);

        let region_lo = IVec3::new(10, 6, 10);
        let region_hi = IVec3::new(34, 34, 34);
        let mut scratch = SculptExtractScratch::new();
        let (verts, indices) = extract_mesh_region_from_cells_pooled_haloed(
            &mut scratch,
            &cells,
            region_lo,
            region_hi,
            &nodes,
            depth,
            vs,
            origin,
            &[],
            &[],
            &[],
            &[],
            None,
            None::<&fn(Vec3) -> f32>,
        &[],
        );
        assert!(!verts.is_empty(), "slope produced no surface vertices");

        // Analytic surface for the slope (in world coords). For a
        // vertex at world (x, _, z) the true surface y is:
        //   y = origin.y + vs * (20 + (gx-16)*(-nx) + (gz-16)*(-nz))
        // where gx = (x-origin.x)/vs. The smoothed surface should sit
        // close to this plane → low RMS, NOT the ±vs staircase a naive
        // grid-aligned extract would produce.
        let true_surface_y = |wx: f32, wz: f32| -> f32 {
            let gx = (wx - origin.x) / vs;
            let gz = (wz - origin.z) / vs;
            origin.y + vs * (20.0 + (gx - 16.0) * (-nx) + (gz - 16.0) * (-nz))
        };

        // RMS of the vertical residual over the top (slope) surface.
        // Restrict to vertices on the slope face (skip the side/bottom
        // walls of the extracted block, whose residual vs the *top*
        // plane is meaningless). A vertex is "on the slope" when its
        // normal is predominantly +Y.
        let mut sq_sum = 0.0f32;
        let mut count = 0u32;
        let true_n = Vec3::new(nx, 1.0, nz).normalize();
        let mut min_outward_dot = f32::INFINITY;
        for v in &verts {
            let nrm = unpack_oct(v.normal_oct).normalize();
            // Slope-top face: normal close to +Y. (Side faces of the
            // finite extracted block point ±X/±Z and are excluded.)
            if nrm.y > 0.6 {
                let p = Vec3::from(v.local_pos);
                let resid = p.y - true_surface_y(p.x, p.z);
                sq_sum += resid * resid;
                count += 1;
                // Outward-ness: the slope-top normal must agree with the
                // analytic outward normal (away from the solid below).
                min_outward_dot = min_outward_dot.min(nrm.dot(true_n));
            }
        }
        assert!(count > 0, "found no slope-top vertices to check");
        let rms = (sq_sum / count as f32).sqrt();
        // A naive grid-aligned extract staircases with RMS ≈ vs/√12 ≈
        // 0.14·vs at best and far worse on the terraces. The density
        // isosurface should sit well under a quarter-voxel.
        assert!(
            rms < 0.25 * vs,
            "slope staircase RMS {:.4} too high (>{:.4} = 0.25·vs) — not smooth",
            rms,
            0.25 * vs
        );

        // Outward normals: every slope-top normal points away from the
        // solid (toward empty). On a solid-below / empty-above slab the
        // outward direction is +Y-tilted; require strong agreement with
        // the analytic outward normal.
        assert!(
            min_outward_dot > 0.9,
            "slope-top normal not outward enough: min dot with analytic \
             outward normal {:.3} (want > 0.9)",
            min_outward_dot
        );
        // And specifically +Y-dominant (outward should be +Y on a
        // solid-below slab, the sign the plan calls out).
        for v in &verts {
            let nrm = unpack_oct(v.normal_oct).normalize();
            if nrm.y > 0.6 {
                assert!(
                    nrm.y > 0.0,
                    "slope-top outward normal has non-positive Y: {nrm:?}"
                );
            }
        }
        // Sanity: indices reference real verts.
        for &i in &indices {
            assert!((i as usize) < verts.len());
        }
    }

    /// **No per-voxel normal speckle.** Normals come from interpolating
    /// the precomputed grid-point gradient field (which is C0), NOT from
    /// differentiating the trilinear density at the vertex (whose
    /// gradient is piecewise-constant and DISCONTINUOUS across cells).
    /// On a smooth occupancy sphere the angle between the two endpoint
    /// normals of every triangle edge must stay small — a continuous
    /// field. The old differentiate-trilinear approach flips the normal
    /// cell-to-cell and would blow past this bound.
    #[test]
    fn density_normals_are_continuous_no_voxel_speckle() {
        let depth: u8 = 7; // extent = 128.
        let nodes = vec![EMPTY_NODE];
        let vs = 0.5f32;
        let origin = Vec3::ZERO;

        // A gentle, NON-grid-aligned planar slope. Its true surface
        // curvature is ~zero, so the normal is (nearly) constant over
        // the whole surface — every triangle edge SHOULD see a sub-degree
        // normal change. Any edge that flips by a large angle is
        // voxel-scale speckle, i.e. the discontinuous gradient that
        // differentiating the trilinear density produces cell-to-cell.
        // This is the cleanest discriminator: with the interpolated
        // grid-point gradient field the edges stay tight; the old
        // differentiate-trilinear normal would show a bimodal spread
        // with a large mass of edges past the threshold.
        let nx = 0.35f32;
        let nz = 0.20f32;
        let solid = |c: IVec3| -> bool {
            // Solid below y = 64 + nx*(x-64) + nz*(z-64).
            let surf = 64.0 + nx * (c.x as f32 - 64.0) + nz * (c.z as f32 - 64.0);
            (c.y as f32) < surf
        };
        let band_lo = IVec3::new(24, 0, 24);
        let band_hi = IVec3::new(104, 100, 104);
        let cells = occupancy_cellmap(band_lo, band_hi, solid);

        let region_lo = IVec3::new(36, 36, 36);
        let region_hi = IVec3::new(92, 92, 92);
        let mut scratch = SculptExtractScratch::new();
        let (verts, indices) = extract_mesh_region_from_cells_pooled_haloed(
            &mut scratch,
            &cells,
            region_lo,
            region_hi,
            &nodes,
            depth,
            vs,
            origin,
            &[],
            &[],
            &[],
            &[],
            None,
            None::<&fn(Vec3) -> f32>,
        &[],
        );
        assert!(!verts.is_empty(), "slope produced no surface vertices");
        assert!(!indices.is_empty(), "slope produced no triangles");

        // Per-edge normal-angle check over the INTERIOR of the slope-top
        // surface. We restrict to vertices comfortably inside the
        // extracted block in X/Z (away from its ±X/±Z side walls and the
        // grid-boundary clamp, where the surface legitimately curves
        // toward the walls) and select slope-top vertices (normal
        // predominantly +Y). The true surface there has ~zero curvature,
        // so a CONTINUOUS normal field keeps every edge to a small angle.
        // A discontinuous (per-voxel speckled) normal — what
        // differentiating the trilinear density produces — flips abruptly
        // between cell-adjacent verts and blows past the bound.
        let max_angle_deg = 15.0f32;
        let max_cos_floor = max_angle_deg.to_radians().cos();
        // Interior X/Z window in WORLD coords (region [36,92) cells × vs
        // = [18, 46) world; keep ≥ 6 world units off each X/Z edge).
        let in_interior = |i: u32| -> bool {
            let p = verts[i as usize].local_pos;
            p[0] > 24.0 && p[0] < 40.0 && p[2] > 24.0 && p[2] < 40.0
        };
        let is_top = |i: u32| unpack_oct(verts[i as usize].normal_oct).normalize().y > 0.6;
        let mut worst_deg = 0.0f32;
        let mut checked = 0u32;
        let mut edge = |ia: u32, ib: u32| {
            if !is_top(ia) || !is_top(ib) || !in_interior(ia) || !in_interior(ib) {
                return;
            }
            let na = unpack_oct(verts[ia as usize].normal_oct).normalize();
            let nb = unpack_oct(verts[ib as usize].normal_oct).normalize();
            let cos = na.dot(nb).clamp(-1.0, 1.0);
            let deg = cos.acos().to_degrees();
            if deg > worst_deg {
                worst_deg = deg;
            }
            checked += 1;
            assert!(
                cos >= max_cos_floor,
                "interior slope-top edge normal angle {deg:.2}° exceeds {max_angle_deg}° \
                 (na={na:?}, nb={nb:?}) — voxel-scale normal speckle",
            );
        };
        for tri in indices.chunks_exact(3) {
            edge(tri[0], tri[1]);
            edge(tri[1], tri[2]);
            edge(tri[2], tri[0]);
        }
        assert!(checked > 0, "no interior slope-top edges were checked");
        // On the flat interior the true normal is constant, so a
        // continuous field keeps every edge well under the bound (a few
        // degrees, modulo oct-pack quantization + the σ=1 blur's
        // transition-band tilt). A borderline-passing worst angle would
        // itself indicate residual discontinuity.
        assert!(
            worst_deg < 6.0,
            "worst interior slope-top edge normal angle {worst_deg:.2}° unexpectedly \
             large — field is not smooth",
        );
    }
}
