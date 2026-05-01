// Phase 7d Session 1 — shadow-tile mark compute pass.
//
// One thread per TLAS primitive (= one per shadow-casting blade /
// host instance). Projects the prim's 8 AABB corners into the
// directional light's light-space (axes `right`, `up` perpendicular
// to the light direction `L`), finds the resulting 2D AABB, and
// `atomicOr`-marks every covered tile in the bitmap.
//
// The shadow trace per-pixel pass reads this bitmap to short-circuit
// the BVH descent: for directional lights, every world-space ray's
// (x_l, y_l) is constant along the ray (the projection axis is L
// itself), so a single tile lookup tells us whether ANY shadow
// caster lies along the ray's path. Empty tile → no shadow → skip
// the entire BVH walk.
//
// Tile resolution is chosen CPU-side (`grid_w × grid_h`); 256×256
// is the V1 default — bitmap = 8 KB, plenty fine for typical paint
// patches projected against the scene's overall light-space extent.
//
// V1 limitation: only ONE bitmap, sized for ONE directional light
// (the first one the engine finds). Multi-directional scenes only
// get the cull benefit on that one light; others fall back to the
// full per-pixel BVH descent.

struct TlasPrim {
    aabb_min: vec3<f32>,
    asset_id: u32,
    aabb_max: vec3<f32>,
    instance_state_offset: u32,
    material_id: u32,
    instance_index: u32,
    _pad0: u32,
    _pad1: u32,
}

struct ShadowTileUniform {
    // World-space origin of the light-space coordinate system.
    // Per-frame: scene min projected back, or any fixed point in
    // the scene — only relative coordinates matter.
    light_origin: vec3<f32>,
    // Tile size in light-space units (= world units, since the
    // basis is orthonormal).
    tile_size: f32,
    // Light-space right basis vector. Perpendicular to L.
    light_right: vec3<f32>,
    // Tile grid width.
    grid_w: u32,
    // Light-space up basis vector. Perpendicular to L and right.
    light_up: vec3<f32>,
    // Tile grid height.
    grid_h: u32,
    // Number of primitives in `tlas_prims` this frame. Mirrors the
    // post-readback value from the TLAS build.
    prim_count: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read> tlas_prims: array<TlasPrim>;
@group(0) @binding(1) var<storage, read_write> tile_bitmap: array<atomic<u32>>;
@group(1) @binding(0) var<uniform> u: ShadowTileUniform;

fn project_to_light_space(p: vec3<f32>) -> vec2<f32> {
    let offset = p - u.light_origin;
    return vec2<f32>(dot(offset, u.light_right), dot(offset, u.light_up));
}

@compute @workgroup_size(64, 1, 1)
fn mark_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= u.prim_count) {
        return;
    }
    let prim = tlas_prims[i];

    // Project all 8 AABB corners; compute 2D bounds. The full
    // 8-corner sweep is exact (avoids the conservative-but-loose
    // shortcut of projecting just min/max).
    var min_2d = vec2<f32>(1e30, 1e30);
    var max_2d = vec2<f32>(-1e30, -1e30);
    for (var c: u32 = 0u; c < 8u; c = c + 1u) {
        let cx = select(prim.aabb_min.x, prim.aabb_max.x, (c & 1u) != 0u);
        let cy = select(prim.aabb_min.y, prim.aabb_max.y, (c & 2u) != 0u);
        let cz = select(prim.aabb_min.z, prim.aabb_max.z, (c & 4u) != 0u);
        let p = project_to_light_space(vec3<f32>(cx, cy, cz));
        min_2d = min(min_2d, p);
        max_2d = max(max_2d, p);
    }

    // Tile range, clamped to grid bounds. Both bounds are floor /
    // ceil into integer tile coordinates so partial-tile coverage
    // marks the tile.
    //
    // V1 anti-flicker dilation: extend the marked range by one
    // tile in every direction. Animated shaders move blade AABBs
    // frame-to-frame; pixels right on a tile boundary would
    // otherwise flip between "tile occupied" and "tile empty" as
    // the AABB wobbles, causing shadows to flicker. The 1-tile
    // halo absorbs sub-tile motion. Cost: more bits set → fewer
    // pixels short-circuit the BVH walk, but for typical grass
    // density the dilation is a small constant factor.
    let tile_min_x = max(i32(floor(min_2d.x / u.tile_size)) - 1, 0);
    let tile_min_y = max(i32(floor(min_2d.y / u.tile_size)) - 1, 0);
    let tile_max_x = min(i32(ceil(max_2d.x / u.tile_size)) + 1, i32(u.grid_w));
    let tile_max_y = min(i32(ceil(max_2d.y / u.tile_size)) + 1, i32(u.grid_h));

    if (tile_min_x >= tile_max_x || tile_min_y >= tile_max_y) {
        return;
    }

    for (var ty = tile_min_y; ty < tile_max_y; ty = ty + 1) {
        for (var tx = tile_min_x; tx < tile_max_x; tx = tx + 1) {
            let tile_idx = u32(ty) * u.grid_w + u32(tx);
            let word = tile_idx >> 5u;
            let bit = tile_idx & 31u;
            atomicOr(&tile_bitmap[word], 1u << bit);
        }
    }
}
