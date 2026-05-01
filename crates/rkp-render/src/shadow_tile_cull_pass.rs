//! Phase 7d — shadow-tile cull infrastructure.
//!
//! Builds a 2D bitmap in directional-light-space that says "any
//! shadow-casting primitive lies along this column?" Per-pixel
//! shadow trace looks up the bitmap to short-circuit the BVH walk
//! for pixels whose ray doesn't pass near any caster.
//!
//! ## Why
//!
//! The TLAS makes BVH traversal O(log N), but the per-pixel
//! invocation cost is a floor — every shadow ray pays the cost of
//! reading `tlas_node_count`, doing the root-node AABB cull, and
//! managing the traversal stack. For half-res 1080p that's ~1 M
//! pixels paying ~1 ns/pixel = ~1 ms regardless of how many casters
//! actually exist. With ~30 grass blades, that 1 ms is far more
//! than the BVH traversal cost itself. The tile cull replaces the
//! per-pixel root-AABB cull with a per-pixel single-bit lookup.
//! Most of the screen has no casters in their light-space tile, so
//! the BVH walk is skipped entirely.
//!
//! ## V1 limitations
//!
//! * **Single bitmap**, sized for one directional light. Engine
//!   picks the first directional light it finds; other lights
//!   (point/spot, additional directional) take the full BVH path.
//! * **Static tile resolution** (`SHADOW_TILE_GRID_W × _H` =
//!   256 × 256). 8 KB bitmap regardless of scene size.

use crate::tlas_build_pass::TlasPrim;

/// Tile-grid resolution. 256 × 256 = 65 536 tiles → 2048 u32
/// bitmap = 8 KB. CPU-side computes per-frame `tile_size` in
/// world units to fit the directional light's projected scene
/// AABB into this grid.
pub const SHADOW_TILE_GRID_W: u32 = 256;
pub const SHADOW_TILE_GRID_H: u32 = 256;

/// Bitmap length in u32s = `SHADOW_TILE_GRID_W * SHADOW_TILE_GRID_H / 32`.
pub const SHADOW_TILE_BITMAP_U32S: u32 =
    (SHADOW_TILE_GRID_W * SHADOW_TILE_GRID_H) / 32;

/// Per-dispatch uniform for the mark pass. 64 B — matches
/// `ShadowTileUniform` in `shadow_tile_mark.wgsl`. The shadow trace
/// reads the same fields from `MarchParams` (with shorthand names);
/// the duplication keeps the mark pass independent of the march's
/// uniform layout.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ShadowTileUniform {
    pub light_origin: [f32; 3],   // 0..12
    pub tile_size: f32,             // 12..16
    pub light_right: [f32; 3],     // 16..28
    pub grid_w: u32,                // 28..32
    pub light_up: [f32; 3],         // 32..44
    pub grid_h: u32,                // 44..48
    pub prim_count: u32,            // 48..52
    pub _pad0: u32,                 // 52..56
    pub _pad1: u32,                 // 56..60
    pub _pad2: u32,                 // 60..64
}

const _: () = assert!(std::mem::size_of::<ShadowTileUniform>() == 64);

/// Pipeline holder for the shadow-tile mark pass. Owns the
/// compute pipeline, the persistent bitmap storage, and the
/// per-dispatch uniform buffer. The bitmap is bound from here
/// into both the mark pass (write) and the shadow trace (read);
/// the engine wires the bind group entries directly.
pub struct ShadowTileCullPass {
    pub mark_pipeline: wgpu::ComputePipeline,
    pub g0_layout: wgpu::BindGroupLayout,
    pub g1_layout: wgpu::BindGroupLayout,
    /// Tile occupancy bitmap. `SHADOW_TILE_BITMAP_U32S` u32s.
    /// Engine zeroes per frame before dispatching the mark pass;
    /// the shadow trace reads after.
    pub bitmap_buffer: wgpu::Buffer,
    pub uniform_buffer: wgpu::Buffer,
}

impl ShadowTileCullPass {
    pub fn new(device: &wgpu::Device) -> Self {
        let g0_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("shadow_tile_mark g0"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let g1_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("shadow_tile_mark g1"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: std::num::NonZeroU64::new(
                        std::mem::size_of::<ShadowTileUniform>() as u64,
                    ),
                },
                count: None,
            }],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("shadow_tile_mark pipeline layout"),
            bind_group_layouts: &[Some(&g0_layout), Some(&g1_layout)],
            immediate_size: 0,
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("shadow_tile_mark"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("shaders/shadow_tile_mark.wgsl").into(),
            ),
        });
        let mark_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("shadow_tile_mark"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("mark_main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let bitmap_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("shadow_tile_bitmap"),
            size: (SHADOW_TILE_BITMAP_U32S as u64) * 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("shadow_tile_mark uniform"),
            size: std::mem::size_of::<ShadowTileUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            mark_pipeline,
            g0_layout,
            g1_layout,
            bitmap_buffer,
            uniform_buffer,
        }
    }
}

/// Pick a tile size in world units that fits the directional
/// light's light-space scene AABB into a `SHADOW_TILE_GRID_W ×
/// SHADOW_TILE_GRID_H` grid with a small safety margin. The
/// resulting `(light_origin, tile_size)` pair places light-space
/// `(0, 0)` at the projected-AABB minimum so all valid tiles fall
/// in `[0, grid_w) × [0, grid_h)`.
///
/// Returns `(tile_size, light_origin)`. `light_origin` is in
/// world-space — the mark pass and shadow trace subtract it from
/// every world-space query before projecting.
///
/// `light_right` and `light_up` are passed in pre-computed (engine
/// derives them from the sun direction).
pub fn fit_tile_grid(
    scene_min: [f32; 3],
    scene_max: [f32; 3],
    light_right: [f32; 3],
    light_up: [f32; 3],
) -> (f32, [f32; 3]) {
    // Project all 8 corners of the world AABB into light-space;
    // find 2D bounds.
    let mut min_2d = [f32::INFINITY; 2];
    let mut max_2d = [f32::NEG_INFINITY; 2];
    for c in 0..8u32 {
        let corner = [
            if (c & 1) != 0 { scene_max[0] } else { scene_min[0] },
            if (c & 2) != 0 { scene_max[1] } else { scene_min[1] },
            if (c & 4) != 0 { scene_max[2] } else { scene_min[2] },
        ];
        let xl = light_right[0] * corner[0] + light_right[1] * corner[1] + light_right[2] * corner[2];
        let yl = light_up[0] * corner[0] + light_up[1] * corner[1] + light_up[2] * corner[2];
        if xl < min_2d[0] { min_2d[0] = xl; }
        if yl < min_2d[1] { min_2d[1] = yl; }
        if xl > max_2d[0] { max_2d[0] = xl; }
        if yl > max_2d[1] { max_2d[1] = yl; }
    }
    let extent_x = (max_2d[0] - min_2d[0]).max(1e-3);
    let extent_y = (max_2d[1] - min_2d[1]).max(1e-3);
    // Tile size to fit the larger extent into the grid; smaller
    // extent gets finer granularity but doesn't waste tiles.
    let tile_size = (extent_x / SHADOW_TILE_GRID_W as f32)
        .max(extent_y / SHADOW_TILE_GRID_H as f32)
        .max(1e-3);
    // `light_origin` in world-space such that
    // `dot(p - light_origin, light_right) - min_2d.x >= 0` for any
    // p in the AABB. Subtract `min_2d` worth of (right, up) from
    // the world origin so projected coords come out in [0, extent].
    let light_origin = [
        light_right[0] * min_2d[0] + light_up[0] * min_2d[1],
        light_right[1] * min_2d[0] + light_up[1] * min_2d[1],
        light_right[2] * min_2d[0] + light_up[2] * min_2d[1],
    ];
    (tile_size, light_origin)
}

/// Derive a right/up orthonormal basis perpendicular to the light
/// direction `L`. World up is +Y by convention; if `L` is too
/// close to ±Y, falls back to using world forward (+Z) as the
/// generator.
pub fn light_space_basis(light_dir: [f32; 3]) -> ([f32; 3], [f32; 3]) {
    let l = normalize(light_dir);
    let world_up = if l[1].abs() < 0.99 {
        [0.0_f32, 1.0, 0.0]
    } else {
        [0.0_f32, 0.0, 1.0]
    };
    let right = normalize(cross(world_up, l));
    let up = cross(l, right);
    (right, up)
}

fn normalize(v: [f32; 3]) -> [f32; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt().max(1e-10);
    [v[0] / len, v[1] / len, v[2] / len]
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// CPU reference for `shadow_tile_mark.wgsl::mark_main`. Returns
/// the tile bitmap that the GPU dispatch is expected to produce
/// for the same input. Used by the integration test for a
/// bit-for-bit comparison.
pub fn cpu_reference_mark(
    prims: &[TlasPrim],
    uniform: &ShadowTileUniform,
) -> Vec<u32> {
    let mut bitmap = vec![0u32; SHADOW_TILE_BITMAP_U32S as usize];
    for prim in prims {
        let mut min_2d = [f32::INFINITY; 2];
        let mut max_2d = [f32::NEG_INFINITY; 2];
        for c in 0..8u32 {
            let corner = [
                if (c & 1) != 0 { prim.aabb_max[0] } else { prim.aabb_min[0] },
                if (c & 2) != 0 { prim.aabb_max[1] } else { prim.aabb_min[1] },
                if (c & 4) != 0 { prim.aabb_max[2] } else { prim.aabb_min[2] },
            ];
            let off = [
                corner[0] - uniform.light_origin[0],
                corner[1] - uniform.light_origin[1],
                corner[2] - uniform.light_origin[2],
            ];
            let xl = off[0] * uniform.light_right[0]
                + off[1] * uniform.light_right[1]
                + off[2] * uniform.light_right[2];
            let yl = off[0] * uniform.light_up[0]
                + off[1] * uniform.light_up[1]
                + off[2] * uniform.light_up[2];
            if xl < min_2d[0] { min_2d[0] = xl; }
            if yl < min_2d[1] { min_2d[1] = yl; }
            if xl > max_2d[0] { max_2d[0] = xl; }
            if yl > max_2d[1] { max_2d[1] = yl; }
        }
        // 1-tile halo dilation — see `shadow_tile_mark.wgsl` for
        // the anti-flicker rationale.
        let tile_min_x = (((min_2d[0] / uniform.tile_size).floor() as i32) - 1).max(0);
        let tile_min_y = (((min_2d[1] / uniform.tile_size).floor() as i32) - 1).max(0);
        let tile_max_x = (((max_2d[0] / uniform.tile_size).ceil() as i32) + 1).min(uniform.grid_w as i32);
        let tile_max_y = (((max_2d[1] / uniform.tile_size).ceil() as i32) + 1).min(uniform.grid_h as i32);
        if tile_min_x >= tile_max_x || tile_min_y >= tile_max_y {
            continue;
        }
        for ty in tile_min_y..tile_max_y {
            for tx in tile_min_x..tile_max_x {
                let tile_idx = (ty as u32) * uniform.grid_w + (tx as u32);
                let word = (tile_idx >> 5) as usize;
                let bit = tile_idx & 31;
                if word < bitmap.len() {
                    bitmap[word] |= 1u32 << bit;
                }
            }
        }
    }
    bitmap
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_prim(min: [f32; 3], max: [f32; 3]) -> TlasPrim {
        TlasPrim {
            aabb_min: min,
            asset_id: 0,
            aabb_max: max,
            instance_state_offset: 0,
            material_id: 0,
            instance_index: 0,
            _pad0: 0,
            _pad1: 0,
        }
    }

    #[test]
    fn shadow_tile_uniform_size_is_64() {
        assert_eq!(std::mem::size_of::<ShadowTileUniform>(), 64);
    }

    #[test]
    fn light_space_basis_is_orthonormal() {
        let l = [0.0, -1.0, 0.0]; // sun overhead; light_dir points down
        let (right, up) = light_space_basis(l);
        // right perpendicular to l
        let dot_rl = right[0] * l[0] + right[1] * l[1] + right[2] * l[2];
        assert!(dot_rl.abs() < 1e-5);
        // up perpendicular to l and right
        let dot_ul = up[0] * l[0] + up[1] * l[1] + up[2] * l[2];
        let dot_ur = up[0] * right[0] + up[1] * right[1] + up[2] * right[2];
        assert!(dot_ul.abs() < 1e-5);
        assert!(dot_ur.abs() < 1e-5);
        // Lengths ~ 1
        let r_len = (right[0].powi(2) + right[1].powi(2) + right[2].powi(2)).sqrt();
        let u_len = (up[0].powi(2) + up[1].powi(2) + up[2].powi(2)).sqrt();
        assert!((r_len - 1.0).abs() < 1e-5);
        assert!((u_len - 1.0).abs() < 1e-5);
    }

    #[test]
    fn light_space_basis_handles_y_aligned_light() {
        // L pointing straight up — the y-up fallback kicks in.
        let l = [0.0, 1.0, 0.0];
        let (right, up) = light_space_basis(l);
        let r_len = (right[0].powi(2) + right[1].powi(2) + right[2].powi(2)).sqrt();
        let u_len = (up[0].powi(2) + up[1].powi(2) + up[2].powi(2)).sqrt();
        assert!((r_len - 1.0).abs() < 1e-5, "right not unit-length: {:?}", right);
        assert!((u_len - 1.0).abs() < 1e-5);
    }

    #[test]
    fn fit_tile_grid_centers_origin() {
        // Sun straight down; world-up basis.
        let l = [0.0, -1.0, 0.0];
        let (right, up) = light_space_basis(l);
        let (tile_size, _origin) = fit_tile_grid(
            [-10.0, 0.0, -10.0],
            [10.0, 5.0, 10.0],
            right,
            up,
        );
        // Extent in light-space is 20×20 (X×Z, since L = -Y); tile
        // grid 256² ⇒ tile_size ≈ 20/256 = 0.078.
        assert!((tile_size - 20.0 / 256.0).abs() < 1e-3, "tile_size = {tile_size}");
    }

    #[test]
    fn cpu_reference_mark_marks_known_tile() {
        let l = [0.0, -1.0, 0.0];
        let (right, up) = light_space_basis(l);
        // Scene: 0..10 in X, 0..10 in Z. tile_size = 10/256.
        let (tile_size, origin) = fit_tile_grid(
            [0.0, 0.0, 0.0],
            [10.0, 1.0, 10.0],
            right,
            up,
        );
        let prims = vec![
            // Centered at (5, 0.5, 5), AABB ±0.05 in each axis.
            // light-space coords (right, up) for a y-down light:
            //   right = +X, up = +Z (or -Z depending on cross
            //   handedness — test what the basis produces).
            make_prim([4.95, 0.45, 4.95], [5.05, 0.55, 5.05]),
        ];
        let uniform = ShadowTileUniform {
            light_origin: origin,
            tile_size,
            light_right: right,
            grid_w: SHADOW_TILE_GRID_W,
            light_up: up,
            grid_h: SHADOW_TILE_GRID_H,
            prim_count: prims.len() as u32,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        let bitmap = cpu_reference_mark(&prims, &uniform);

        // 0.1m AABB, tile_size ~0.039m → AABB spans ~3 tiles per
        // axis (with floor/ceil → up to 4). Plus 1-tile halo on
        // every side → up to 6×6 = 36 tiles.
        let total: u32 = bitmap.iter().map(|w| w.count_ones()).sum();
        assert!(
            total >= 1 && total <= 36,
            "expected 1-36 tiles set for one small AABB w/ halo, got {total}"
        );
        // Tiles should cluster near the center of the grid (the
        // prim is at world (5, _, 5), middle of a 0..10 scene).
        // Count bits in the [120, 140) × [120, 140) tile rect —
        // every set bit must fall there.
        let mut center_count: u32 = 0;
        for ty in 120u32..140 {
            for tx in 120u32..140 {
                let idx = ty * SHADOW_TILE_GRID_W + tx;
                let word = (idx >> 5) as usize;
                let bit = idx & 31;
                if (bitmap[word] >> bit) & 1 != 0 {
                    center_count += 1;
                }
            }
        }
        assert_eq!(center_count, total, "set bits weren't all in the central region");
    }

    #[test]
    fn cpu_reference_mark_empty_input_yields_zero_bitmap() {
        let l = [0.0, -1.0, 0.0];
        let (right, up) = light_space_basis(l);
        let (tile_size, origin) = fit_tile_grid([0.0; 3], [10.0; 3], right, up);
        let uniform = ShadowTileUniform {
            light_origin: origin,
            tile_size,
            light_right: right,
            grid_w: SHADOW_TILE_GRID_W,
            light_up: up,
            grid_h: SHADOW_TILE_GRID_H,
            prim_count: 0,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        let bitmap = cpu_reference_mark(&[], &uniform);
        assert!(bitmap.iter().all(|w| *w == 0));
    }
}
