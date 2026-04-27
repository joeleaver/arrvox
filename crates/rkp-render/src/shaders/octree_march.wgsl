// Octree-accelerated compute ray marcher.
//
// Step-and-query: advance along the ray, query the octree at each position.
// EMPTY nodes at coarse depth levels let us skip large regions in one step.
// Surface detected at first occupied voxel (opacity > threshold).

const OCTREE_EMPTY: u32 = 0xFFFFFFFFu;
const OCTREE_INTERIOR: u32 = 0xFFFFFFFEu;
const OCTREE_LEAF_BIT: u32 = 0x80000000u;
const OCTREE_BRICK_BIT: u32 = 0x40000000u;
const OCTREE_PAYLOAD_MASK: u32 = 0x3FFFFFFFu;
const OPACITY_THRESHOLD: f32 = 0.05;
// Safety-net ceiling for the outer ray-march loop. The ray already
// terminates naturally when `t > t_range.y` (exits the octree); this
// cap exists only to avoid a GPU hang if FP precision or a brick-
// chain bug ever prevented `t` from advancing.
//
// Sized for the largest octree we expect (depth 12 = 4096 voxels
// per axis = 512 bricks). Worst-case diagonal is ~1775 nodes.
// 4096 leaves plenty of headroom and is still negligible vs the
// GPU watchdog timeout.
//
// TODO: replace with an adaptive limit derived from each object's
// uploaded octree depth — `MAX_STEPS_PER_OBJ = 8 * bricks_per_axis`
// would be exact. Requires adding a field to `RkpGpuObject`.
const MAX_STEPS: u32 = 4096u;
// Brick layout — must match rkp_core::brick_pool constants.
const BRICK_DIM: u32 = 4u;
const BRICK_DIM_F: f32 = 4.0;
const BRICK_CELLS: u32 = 64u; // 4³
const BRICK_CELL_EMPTY: u32 = 0xFFFFFFFFu;
// Interior-of-solid sentinel (see rkp_core::brick_pool::BRICK_INTERIOR).
// Stored by the mesh import for cells that are inside the solid but
// aren't the visible shell. The march skips past these identically to
// BRICK_CELL_EMPTY (rays hit the shell first, never see the interior);
// neighborhood kernels count them as occupied mass so centroid-based
// normal reconstruction has something to bias against.
const BRICK_CELL_INTERIOR: u32 = 0xFFFFFFFDu;
// A 4³ brick has at most ~12 cells along the longest diagonal traversal,
// so capping inner-DDA at 16 keeps a misbehaving loop from melting the
// frame. Real traversals never come close to this cap.
// Raised from 16 to 128: the inner DDA chains across adjacent bricks
// via brick_face_links, so a single inner loop can traverse many bricks
// before the outer loop needs to run again.
// Inner brick-chain limit — the DDA walks from cell to cell, chaining
// across brick boundaries via `brick_face_links`. Sized the same as
// `MAX_STEPS` since the inner chain can cover the full octree diagonal
// in pathological cases.
const BRICK_MAX_STEPS: u32 = 4096u;

struct RkpObject {
    world: mat4x4<f32>,
    aabb_min: vec3<f32>, octree_root: u32,
    aabb_max: vec3<f32>, octree_depth: u32,
    octree_extent_bits: u32, voxel_size: f32,
    material_id: u32, object_id: u32,
    geom_type: u32, is_skinned: u32,
    bone_count: u32, bone_buffer_offset: u32,
    rest_octree_root: u32, rest_octree_depth: u32,
    rest_octree_extent_bits: u32, bone_field_offset: u32,
    layer_mask: u32,
    bone_field_dim_x: u32, bone_field_dim_y: u32, bone_field_dim_z: u32,
    bone_field_origin_x: f32, bone_field_origin_y: f32, bone_field_origin_z: f32,
    bone_field_occ_offset: u32,
    grid_origin: vec3<f32>,
    _post_grid: u32,
    inverse_world: mat4x4<f32>,
}

struct CameraUniforms {
    position: vec4<f32>, forward: vec4<f32>,
    right: vec4<f32>, up: vec4<f32>,
    resolution: vec2<f32>, jitter: vec2<f32>,
    layer_mask: u32, focus_object_id: u32,
    _cam_pad0: u32, _cam_pad1: u32,
    prev_vp: mat4x4<f32>, view_proj: mat4x4<f32>,
}

// Render-layer + focus gate. An object is visible in this viewport iff its
// layer mask intersects the camera's mask, OR its object_id matches the
// camera's focus entity. Default (u32::MAX / u32::MAX) passes everything.
fn rkp_object_visible(obj: RkpObject) -> bool {
    return (obj.layer_mask & camera.layer_mask) != 0u
        || obj.object_id == camera.focus_object_id;
}

struct MarchParams {
    object_count: u32,
    mode: u32,
    shadow_max_steps: u32,
    num_lights: u32,
    // Prefiltered-LOD early-exit gate. `1` → check each branch's `.y`
    // (prefilter attr id) plus a screen-footprint threshold and
    // terminate descent when the node would occupy <1 pixel. `0` →
    // always descend to a terminator (pre-LOD behavior).
    lod_enabled: u32,
    // Surface-Nets normal reconstruction gate. `1` → at each brick-cell
    // hit, replace the baked octahedral normal with one reconstructed
    // from the 3³ in-brick occupancy neighborhood (centroid-outward).
    // Proof-of-concept: brick boundaries fall back to baked; isolated
    // or fully-surrounded voxels fall back to baked too.
    surfacenet_enabled: u32,
    // Per-tile list grid width in tiles (render_width / 8, rounded up).
    // Shader looks up its tile's object-list slice as
    // `tile_object_ids[tile_offsets[tile_idx]..tile_offsets[tile_idx+1]]`
    // where `tile_idx = ty * tile_count_x + tx`.
    tile_count_x: u32,
    // Pad to 32 bytes (uniform size must be a multiple of 16). Plain
    // u32s, not a vec3<u32> — vec3 would promote struct alignment to 16
    // and inflate the total to 48 bytes, breaking the binding-size
    // check against the 32-byte Rust struct.
    _pad0: u32,
}

const INTERNAL_ATTR_NONE: u32 = 0xFFFFFFFFu;
// Footprint threshold below which we treat a branch as small enough to
// represent with its prefilter attr. Strict `<` (not `<=`) so the
// fallback path still runs for exactly-1px nodes. See the LOD plan for
// hysteresis discussion: at depth N+1 the footprint is half of N's, so
// the sharp cutoff produces a monotonic "descend/terminate" decision
// per ray with no ping-pong under camera motion at the sub-pixel scale.
// Terminate descent when the *child* we'd descend into would be smaller
// than a pixel. Since a child's footprint is half the current node's,
// the cutoff on the current node is 2.0 px — at 2.0 the child is at 1.0.
// Rationale: "one sample per screen pixel" is the right mip cutoff;
// below that the per-pixel value is an aliased pick from one of 8
// children. The prefiltered attr (bottom-up average) is the correct
// low-pass reconstruction, so using it when the child would be
// sub-pixel is strictly better than descending.
//
// Note: an earlier 0.9-px draft was too conservative — at typical
// viewing distances for meter-scale assets with mm-scale voxels, every
// branch was several px on screen so LOD never fired.
const LOD_CUTOFF_PX: f32 = 2.0;

struct GpuLight {
    position: vec4<f32>,   // xyz = position, w = type (0=dir, 1=point, 2=spot)
    color: vec4<f32>,      // rgb = color, w = intensity
    direction: vec4<f32>,  // xyz = direction, w = spot angle
    params: vec4<f32>,     // x = range, y = inner_angle, z = shadow_softness, w = cast_shadow
}

// vec3 fields flattened to f32 components — see rkp_shade.wgsl for the
// full rationale (WGSL vec3 alignment would pad this to 128 bytes, but
// the Rust/GpuMaterial is tightly packed at 96).
struct GpuMaterial {
    albedo_r: f32, albedo_g: f32, albedo_b: f32,
    roughness: f32,
    metallic: f32,
    emission_r: f32, emission_g: f32, emission_b: f32,
    emission_strength: f32,
    subsurface: f32,
    subsurface_r: f32, subsurface_g: f32, subsurface_b: f32,
    opacity: f32,
    ior: f32,
    noise_scale: f32,
    noise_strength: f32,
    noise_channels: u32,
    shader_id: u32,
    _pad1: f32, _pad2: f32, _pad3: f32, _pad4: f32, _pad5: f32,
}

struct OctreeResult {
    slot: u32,
    depth: u32,
    // Spatial bounds of the terminating cell (in object-local oc-space).
    // For BRICK results these are the brick's bounds; the brick DDA loop
    // uses them to compute local cell coords without re-descending.
    cell_center: vec3<f32>,
    cell_half: f32,
}

// --- Bindings ---

// Brick storage at binding 0 — flat array of u32 cells, indexed by
// `brick_id * BRICK_CELLS + flat_cell_index`. Each cell is either
// BRICK_CELL_EMPTY or a leaf_attr_id. (Binding 0 was a dummy voxel_pool
// before bricks landed; we reused the slot to stay under the 12
// storage-buffer limit per shader stage.)
@group(0) @binding(0) var<storage, read> brick_pool: array<u32>;
// Each slot is (node_value, prefilter_attr_id). `.x` holds the existing
// node encoding (EMPTY / INTERIOR / BRANCH offset / LEAF id / BRICK id);
// `.y` holds a prefiltered leaf_attr_id for LOD-cutoff early-exit, or
// INTERNAL_ATTR_NONE (0xFFFFFFFF) when unavailable. Interleaved into a
// single `vec2<u32>` binding to stay under the 12-storage-buffer limit.
@group(0) @binding(1) var<storage, read> octree_nodes: array<vec2<u32>>;
@group(0) @binding(2) var<storage, read> objects: array<RkpObject>;
@group(0) @binding(3) var<uniform> camera: CameraUniforms;
// color_pool[leaf_attr_id] → packed R|G|B|A u32, 0 = no override (use
// material base_color). Parallel to leaf_attr_pool.
@group(0) @binding(4) var<storage, read> color_pool_data: array<u32>;
// leaf_attr[leaf_id] carries normal + material IDs. One 8-byte read per
// hit; everything needed to shade the leaf.
struct LeafAttr {
    normal_oct: u32,                 // 2× snorm16 octahedral
    material_packed: u32,            // low 16: material_primary
                                     // mid 12:  material_secondary (shifted 16)
                                     // high 4:  blend_weight (shifted 28)
}
// bone_matrices[offset + i] = forward skinning palette (world × inv_bind).
// bone_matrices[offset + bone_count + i] = inverse skinning palette
// used by the Phase-3b skinned march to invert deformed samples back
// to rest space. Packed by rkp-engine::scene_sync::BoneMatrixAllocator.
@group(0) @binding(5) var<storage, read> bone_matrices: array<mat4x4<f32>>;
// bone_weights[leaf_attr_id * 2 + 0] = packed bone indices (4 × u8)
// bone_weights[leaf_attr_id * 2 + 1] = packed bone weights (4 × u8)
// Baked at import, uploaded via LeafAttrPool::bone_bytes.
@group(0) @binding(6) var<storage, read> bone_weights: array<u32>;
// brick_face_links[brick_id * 6 + face] → adjacent brick_id, or one of
// FACE_EMPTY / FACE_INTERIOR. Face order: −X, +X, −Y, +Y, −Z, +Z.
// Populated by `rkp_core::brick_face_links::compute_brick_face_links`.
@group(0) @binding(7) var<storage, read> brick_face_links: array<u32>;
@group(0) @binding(8) var<storage, read> leaf_attr_pool: array<LeafAttr>;
// Deformed-space bone field — vec2<u32> (indices, weights) per voxel
// cell. Scatter pass writes; the skinned march branch reads. Empty
// cells are (0, 0) (zero-cleared each frame before scatter).
@group(0) @binding(9) var<storage, read> bone_field: array<vec2<u32>>;
// Per-brick occupancy bitmap for the bone field — one bit per 4³ cell
// brick, packed 32 per u32. Same underlying buffer the scatter pass
// writes via `atomicOr` (declared atomic there); read-only plain u32s
// here because the main bind group declares this read-only and the
// scatter's bind group is separate.
@group(0) @binding(10) var<storage, read> bone_field_occ: array<u32>;

const FACE_INTERIOR: u32 = 0xFFFFFFFEu;
const FACE_EMPTY_LINK: u32 = 0xFFFFFFFFu;
const FACE_NX: u32 = 0u;
const FACE_PX: u32 = 1u;
const FACE_NY: u32 = 2u;
const FACE_PY: u32 = 3u;
const FACE_NZ: u32 = 4u;
const FACE_PZ: u32 = 5u;

fn leaf_attr_material_primary(a: LeafAttr) -> u32 { return a.material_packed & 0xFFFFu; }
fn leaf_attr_material_secondary(a: LeafAttr) -> u32 { return (a.material_packed >> 16u) & 0x0FFFu; }
fn leaf_attr_blend_weight(a: LeafAttr) -> u32 { return (a.material_packed >> 28u) & 0x0Fu; }

fn is_brick_node(node: u32) -> bool {
    return (node & OCTREE_LEAF_BIT) != 0u
        && (node & OCTREE_BRICK_BIT) != 0u
        && node != OCTREE_EMPTY
        && node != OCTREE_INTERIOR;
}

fn brick_id_of(node: u32) -> u32 {
    return node & OCTREE_PAYLOAD_MASK;
}

@group(1) @binding(0) var gbuf_position: texture_storage_2d<rgba32float, write>;
@group(1) @binding(1) var gbuf_normal: texture_storage_2d<rgba16float, write>;
@group(1) @binding(2) var gbuf_material: texture_storage_2d<rg32uint, write>;
// Dedicated 32-bit pick channel — stores `gpu_idx` of the hit entity
// (`0xFFFFFFFFu` for sky / miss). Replaces the old 8-bit object_id
// slot in `gbuf_material`'s G channel, which capped pickable scenes at
// 255 entries.
@group(1) @binding(3) var gbuf_pick: texture_storage_2d<r32uint, write>;
// Glass info — oct-packed normal in R, (thickness_mm << 16 | material_id) in G.
// Zero in both channels = "no glass at this pixel."
@group(1) @binding(4) var gbuf_glass: texture_storage_2d<rg32uint, write>;
// Leaf-slot target — primary hit's scene-global leaf_attr_slot, or 0
// for sky / no-hit pixels. Consumed by rkp_shade's geodesic paint
// cursor; indexes into `brush_overlay_distances`.
@group(1) @binding(5) var gbuf_leaf_slot: texture_storage_2d<r32uint, write>;

@group(2) @binding(0) var<uniform> march_params: MarchParams;
@group(2) @binding(1) var<storage, read> materials: array<GpuMaterial>;
@group(2) @binding(2) var<storage, read_write> stats: array<atomic<u32>, 64>;
// stats[0]       = total steps across all pixels
// stats[1]       = (reserved — was total_lookups; retained slot for layout stability)
// stats[2]       = pixels that found a hit
// stats[3]       = max steps for any single pixel
// stats[4..16]   = descent depth histogram, surface march (buckets L0..L11)
// stats[16..28]  = descent depth histogram, normal        (buckets L0..L11)
// stats[28..40]  = descent depth histogram, shadow        (buckets L0..L11)
// stats[40..44]  = hit footprint: <1px, [1,2), [2,4), >=4px
// stats[44]      = leaf_attr_pool reads   (8 B each)
// stats[45]      = voxel_pool reads       (8 B each; word0+word1 same cache line)
// stats[46]      = color_pool_data reads  (4 B each)
// stats[47]      = materials reads        (32 B each — WGSL storage layout)
// stats[48..52]  = LOD early-exit depth histogram (levels 0-2, 3-5, 6-8, 9+)
// stats[52]      = surfacenet normal reconstructions (brick-hit path)
// stats[53]      = (unused — was brick-boundary fallback pre-face-links)
// stats[54]      = surfacenet degenerate fallbacks (isolated or surrounded)
// stats[55]      = skinned march entries (object-level — per skinned obj per pixel)
// stats[56]      = skinned march hits (pixel produced a deformed G-buffer write)
// stats[57]      = skinned march bone-field populated-cell reads
// stats[58..64]  = reserved
//
// octree_nodes reads are derived CPU-side from the per-phase depth histograms:
// sum(bucket[i] * (i + 1)) since each lookup descends `depth+1` nodes.
const PHASE_MARCH: u32 = 0u;
const PHASE_NORMAL: u32 = 1u;
const PHASE_SHADOW: u32 = 2u;
@group(2) @binding(3) var<storage, read> lights: array<GpuLight>;
// Per-tile object-list offsets (prefix sum). Length = num_tiles + 1.
// Tile `t`'s object-id slice is `tile_object_ids[tile_offsets[t]..tile_offsets[t+1]]`.
@group(2) @binding(4) var<storage, read> tile_offsets: array<u32>;
// Flat list of object indices, grouped by tile. Unbounded object count
// per scene — replaces the retired 32-object bitmask culling scheme.
@group(2) @binding(5) var<storage, read> tile_object_ids: array<u32>;

// Workgroup-shared tile range so thread 0 reads `tile_offsets[t]`
// + `tile_offsets[t+1]` once and every thread in the tile reuses them.
var<workgroup> tile_range_start: u32;
var<workgroup> tile_range_end: u32;

// --- Helpers ---

// (Removed legacy `extract_opacity` / `extract_*_id` / `extract_blend_weight`
// helpers — they unpacked the old 8-byte VoxelSample. The active path reads
// material data directly from LeafAttr via `leaf_attr_material_*` instead.)

fn invert_rigid(m: mat4x4<f32>) -> mat4x4<f32> {
    let s2 = dot(m[0].xyz, m[0].xyz);
    let inv_s2 = 1.0 / s2;
    let c0 = vec3<f32>(m[0].x, m[1].x, m[2].x) * inv_s2;
    let c1 = vec3<f32>(m[0].y, m[1].y, m[2].y) * inv_s2;
    let c2 = vec3<f32>(m[0].z, m[1].z, m[2].z) * inv_s2;
    let t = m[3].xyz;
    let inv_t = -vec3<f32>(dot(c0, t), dot(c1, t), dot(c2, t));
    return mat4x4<f32>(
        vec4<f32>(c0, 0.0), vec4<f32>(c1, 0.0),
        vec4<f32>(c2, 0.0), vec4<f32>(inv_t, 1.0),
    );
}

fn intersect_aabb(origin: vec3<f32>, inv_dir: vec3<f32>, box_min: vec3<f32>, box_max: vec3<f32>) -> vec2<f32> {
    let t0 = (box_min - origin) * inv_dir;
    let t1 = (box_max - origin) * inv_dir;
    let tmin = min(t0, t1);
    let tmax = max(t0, t1);
    return vec2<f32>(max(max(max(tmin.x, tmin.y), tmin.z), 0.0),
                     min(min(tmax.x, tmax.y), tmax.z));
}

fn bucket_depth(phase: u32, level: u32) {
    // 12 buckets per phase starting at stats[4]. Levels beyond 11 clamp to 11.
    let base = 4u + phase * 12u;
    atomicAdd(&stats[base + min(level, 11u)], 1u);
}

/// Look up a single neighbor cell's occupancy state given an offset
/// from the hit cell. Resolves cross-brick reads by chaining 1–3
/// face-link hops (one per axis that crosses a brick boundary). Pure
/// indirect memory reads — no octree descent.
///
/// Returns one of:
/// - `BRICK_CELL_EMPTY` — neighbor is empty (not occupied).
/// - Any other value — neighbor is occupied. For cells in a real
///   brick this is the neighbor's `leaf_attr_id`; for cells in an
///   INTERIOR bulk region we return a non-EMPTY sentinel
///   (`FACE_INTERIOR`) since "there's solid there" is all the
///   centroid needs to know.
fn resolve_neighbor_cell(
    start_brick: u32, cx: u32, cy: u32, cz: u32,
    dx: i32, dy: i32, dz: i32,
) -> u32 {
    var current = start_brick;
    var wx: i32 = i32(cx) + dx;
    var wy: i32 = i32(cy) + dy;
    var wz: i32 = i32(cz) + dz;

    // For each axis the neighbor wants to step outside the brick, walk
    // one face link. If that link is FACE_EMPTY, the whole direction is
    // empty air — neighbor is empty. If FACE_INTERIOR, the whole region
    // is solid — report occupied with a sentinel. Otherwise hop into
    // the adjacent brick and wrap the coordinate.
    if wx < 0 {
        let f = brick_face_links[current * 6u + FACE_NX];
        if f == FACE_EMPTY_LINK { return BRICK_CELL_EMPTY; }
        if f == FACE_INTERIOR { return FACE_INTERIOR; }
        current = f;
        wx = wx + i32(BRICK_DIM);
    } else if wx >= i32(BRICK_DIM) {
        let f = brick_face_links[current * 6u + FACE_PX];
        if f == FACE_EMPTY_LINK { return BRICK_CELL_EMPTY; }
        if f == FACE_INTERIOR { return FACE_INTERIOR; }
        current = f;
        wx = wx - i32(BRICK_DIM);
    }
    if wy < 0 {
        let f = brick_face_links[current * 6u + FACE_NY];
        if f == FACE_EMPTY_LINK { return BRICK_CELL_EMPTY; }
        if f == FACE_INTERIOR { return FACE_INTERIOR; }
        current = f;
        wy = wy + i32(BRICK_DIM);
    } else if wy >= i32(BRICK_DIM) {
        let f = brick_face_links[current * 6u + FACE_PY];
        if f == FACE_EMPTY_LINK { return BRICK_CELL_EMPTY; }
        if f == FACE_INTERIOR { return FACE_INTERIOR; }
        current = f;
        wy = wy - i32(BRICK_DIM);
    }
    if wz < 0 {
        let f = brick_face_links[current * 6u + FACE_NZ];
        if f == FACE_EMPTY_LINK { return BRICK_CELL_EMPTY; }
        if f == FACE_INTERIOR { return FACE_INTERIOR; }
        current = f;
        wz = wz + i32(BRICK_DIM);
    } else if wz >= i32(BRICK_DIM) {
        let f = brick_face_links[current * 6u + FACE_PZ];
        if f == FACE_EMPTY_LINK { return BRICK_CELL_EMPTY; }
        if f == FACE_INTERIOR { return FACE_INTERIOR; }
        current = f;
        wz = wz - i32(BRICK_DIM);
    }
    let flat = u32(wx) + u32(wy) * BRICK_DIM + u32(wz) * BRICK_DIM * BRICK_DIM;
    return brick_pool[current * BRICK_CELLS + flat];
}

/// Reconstruct a surface normal at the given brick cell from the 3³
/// centroid of occupied neighbors. Cross-brick neighbors are resolved
/// via chained face-link hops — no octree descent. The resulting
/// normal is the direction away from the centroid of occupied mass.
///
/// `fallback` is the baked octahedral normal, returned when the
/// neighborhood is uninformative (isolated voxel, fully surrounded).
fn reconstruct_normal_surfacenet(
    brick_id: u32,
    cx: u32, cy: u32, cz: u32,
    fallback: vec3<f32>,
) -> vec3<f32> {
    // 3³ kernel (26 neighbors) with inverse-distance weighting via
    // unit-vector accumulation. Each occupied neighbor contributes a
    // unit vector pointing from the hit cell toward it — so face,
    // edge, and corner neighbors all contribute the same magnitude of
    // "direction evidence", but farther cells' offsets are normalized
    // to 1 before summing. Equivalent to `w_i = 1/|offset_i|` in a
    // weighted centroid.
    //
    // Rationale vs. uniform-weighted larger kernels: fewer samples (26
    // vs 124 for 5³) and the outer ring of samples isn't
    // over-contributing just because they happen to span more cells at
    // the same distance band.
    var direction_sum = vec3<f32>(0.0);
    var count = 0.0;
    for (var dz = -1; dz <= 1; dz = dz + 1) {
        for (var dy = -1; dy <= 1; dy = dy + 1) {
            for (var dx = -1; dx <= 1; dx = dx + 1) {
                if dx == 0 && dy == 0 && dz == 0 { continue; }
                let ncell = resolve_neighbor_cell(brick_id, cx, cy, cz, dx, dy, dz);
                if ncell == BRICK_CELL_EMPTY { continue; }
                let offset = vec3<f32>(f32(dx), f32(dy), f32(dz));
                let inv_len = inverseSqrt(f32(dx * dx + dy * dy + dz * dz));
                direction_sum = direction_sum + offset * inv_len;
                count = count + 1.0;
            }
        }
    }

    if count < 0.5 {
        atomicAdd(&stats[54], 1u);
        return fallback;
    }
    let len = length(direction_sum);
    if len < 1e-3 {
        atomicAdd(&stats[54], 1u);
        return fallback;
    }
    return -direction_sum / len;
}

fn bucket_lod_exit(level: u32) {
    // 4 buckets: 0-2, 3-5, 6-8, 9+.
    var b = 3u;
    if level <= 2u { b = 0u; }
    else if level <= 5u { b = 1u; }
    else if level <= 8u { b = 2u; }
    atomicAdd(&stats[48u + b], 1u);
}

/// Descend the octree from `root` toward `pos` (in oc-space) and return
/// the terminating node.
///
/// Prefiltered-LOD early exit: if `lod_enabled` is on and the current
/// branch's projected screen footprint drops below [`LOD_CUTOFF_PX`],
/// we stop descending and return the branch's prefilter attr id (a
/// `leaf_attr_id` into `leaf_attr_pool`) as if it were a LEAF at the
/// current level. The caller shades it with exactly the same path as a
/// regular leaf hit — the prefiltered attr is by construction a valid
/// `LeafAttr` pointing at an averaged (normal, material, color) for
/// the subtree.
///
/// Parameters:
/// * `t_current` — ray parameter at the descent entry in oc-space
///   units (same units as `extent`). Used to compute distance.
/// * `local_to_world_scale` — multiplier converting oc-space length to
///   world units. Same scalar for both `node_size` and `dist` so it
///   cancels at the threshold — but we keep it explicit because the
///   footprint histogram in the caller already works in world units.
/// * `focal_px_y` — vertical pixels per world unit at unit depth. Read
///   once in the caller from `camera`.
fn octree_lookup(
    root: u32,
    max_depth: u32,
    extent: f32,
    pos: vec3<f32>,
    phase: u32,
    t_current: f32,
    local_to_world_scale: f32,
    focal_px_y: f32,
) -> OctreeResult {
    var offset = root;
    var half = extent * 0.5;
    var center = vec3<f32>(half);
    for (var level = 0u; level < max_depth; level++) {
        let packed = octree_nodes[offset];
        let node = packed.x;
        if node == OCTREE_EMPTY {
            bucket_depth(phase, level);
            return OctreeResult(OCTREE_EMPTY, level, center, half);
        }
        if node == OCTREE_INTERIOR {
            bucket_depth(phase, level);
            return OctreeResult(OCTREE_INTERIOR, level, center, half);
        }
        if (node & OCTREE_LEAF_BIT) != 0u {
            bucket_depth(phase, level);
            // Preserve BRICK_BIT in the returned slot so the caller can
            // distinguish a regular leaf from a brick (both arrive via the
            // same code path; only their payload-mask interpretation
            // differs).
            return OctreeResult(node & OCTREE_PAYLOAD_MASK | (node & OCTREE_BRICK_BIT), level, center, half);
        }

        // Branch — check the prefiltered-LOD cutoff before descending.
        // Gated on `phase == PHASE_MARCH`: the shadow path uses a cone-
        // footprint LOD (Phase 3) and must not pixel-footprint-exit;
        // the normal path doesn't need LOD (normals baked into leaves).
        // The node's side in world units is `(half * 2.0) * scale`; the
        // distance from the ray origin is `t * scale`. Both pull from the
        // same `scale`, so the ratio matches the existing world-space
        // footprint histogram's formula (`vs * focal_px_y / dist`).
        if march_params.lod_enabled != 0u
            && phase == PHASE_MARCH
            && packed.y != INTERNAL_ATTR_NONE
        {
            let node_size_world = (half * 2.0) * local_to_world_scale;
            let dist_world = max(t_current * local_to_world_scale, 1e-3);
            let footprint_px = node_size_world * focal_px_y / dist_world;
            if footprint_px < LOD_CUTOFF_PX {
                bucket_depth(phase, level);
                bucket_lod_exit(level);
                // `packed.y` is a `leaf_attr_id` (< BRICK_BIT). Return it
                // as a regular leaf — no BRICK_BIT, callers shade it via
                // the standard leaf-hit path.
                return OctreeResult(packed.y, level, center, half);
            }
        }

        let gt = vec3<u32>(pos >= center);
        offset = node + gt.x + gt.y * 2u + gt.z * 4u;
        half *= 0.5;
        center += vec3<f32>(
            select(-half, half, pos.x >= center.x),
            select(-half, half, pos.y >= center.y),
            select(-half, half, pos.z >= center.z),
        );
    }
    bucket_depth(phase, max_depth);
    let node = octree_nodes[offset].x;
    if node == OCTREE_EMPTY { return OctreeResult(OCTREE_EMPTY, max_depth, center, half); }
    if node == OCTREE_INTERIOR { return OctreeResult(OCTREE_INTERIOR, max_depth, center, half); }
    if (node & OCTREE_LEAF_BIT) != 0u {
        return OctreeResult(node & OCTREE_PAYLOAD_MASK | (node & OCTREE_BRICK_BIT), max_depth, center, half);
    }
    return OctreeResult(OCTREE_EMPTY, max_depth, center, half);
}

/// Detect a BRICK result from `octree_lookup`: BRICK_BIT preserved in slot.
fn slot_is_brick(slot: u32) -> bool {
    return (slot & OCTREE_BRICK_BIT) != 0u
        && slot != OCTREE_EMPTY
        && slot != OCTREE_INTERIOR;
}

/// Strip the BRICK_BIT marker from a slot to get the actual brick_id.
fn slot_brick_id(slot: u32) -> u32 {
    return slot & OCTREE_PAYLOAD_MASK;
}

// Skip past an empty/interior node's region along the ray.
// Uses DDA exit: find the nearest axis-aligned plane the ray crosses to leave this node.
fn skip_node(pos: vec3<f32>, dir: vec3<f32>, inv_dir: vec3<f32>, node_depth: u32, extent: f32, vs: f32) -> f32 {
    let node_size = extent / f32(1u << node_depth);
    let node_min = floor(pos / node_size) * node_size;
    let node_max = node_min + node_size;
    // Exit planes: for positive dir, exit through max; for negative, through min.
    let t_exit = select((node_min - pos) * inv_dir, (node_max - pos) * inv_dir, dir > vec3<f32>(0.0));
    // Smallest positive exit = nearest boundary crossing.
    let t_pos = max(t_exit, vec3<f32>(1e-6));
    return min(min(t_pos.x, t_pos.y), t_pos.z) + vs * 0.01;
}

// Decode a packed 2× snorm16 octahedral normal. Mirror of rkp_core::unpack_oct.
fn unpack_oct_normal(packed: u32) -> vec3<f32> {
    let ui_raw = i32(packed & 0xFFFFu);
    let vi_raw = i32((packed >> 16u) & 0xFFFFu);
    // snorm16: interpret as i16 (sign-extend the 16-bit value).
    let ui = select(ui_raw, ui_raw - 65536, ui_raw >= 32768);
    let vi = select(vi_raw, vi_raw - 65536, vi_raw >= 32768);
    let u = clamp(f32(ui) / 32767.0, -1.0, 1.0);
    let v = clamp(f32(vi) / 32767.0, -1.0, 1.0);
    var n = vec3<f32>(u, v, 1.0 - abs(u) - abs(v));
    if n.z < 0.0 {
        let nx0 = n.x;
        n.x = (1.0 - abs(n.y)) * select(-1.0, 1.0, nx0 >= 0.0);
        n.y = (1.0 - abs(nx0)) * select(-1.0, 1.0, n.y >= 0.0);
    }
    let len = length(n);
    if len < 1e-8 { return vec3<f32>(0.0, 1.0, 0.0); }
    return n / len;
}

// --- Accumulating march (per object) ---
//
// Front-to-back opacity accumulation within a single object. Accumulates
// position and color (cheap). Normal computed ONCE at the end (expensive).

struct MarchResult {
    oc_pos: vec3<f32>,
    color: vec3<f32>,
    // Accumulated local-space normal — weighted by sample contribution the
    // same way color and position are. Pulled from the leaf_attr payload
    // rather than reconstructed from the opacity-field gradient.
    normal: vec3<f32>,
    alpha: f32,
    t: f32,
    first_slot: u32,        // voxel_pool slot (already dereferenced from leaf_attr)
    valid: bool,
    steps: u32,             // total steps taken (for profiling)
    // Glass tracking. Set when the ray traverses at least one
    // transparent voxel (material opacity < 0.99) before landing on
    // the opaque `first_slot` (or exiting the object without one).
    // `glass_normal` is the front-face normal in oc-space, same
    // basis as `normal`. `glass_enter_t` is the ray parameter at
    // first glass contact, `glass_exit_t` is updated on each
    // subsequent glass cell; together they yield a thickness proxy
    // (entry → last-glass-cell). World-space conversion happens in
    // `main()` where the object-to-world scale is in scope.
    glass_valid: bool,
    glass_normal: vec3<f32>,
    glass_material: u32,
    glass_enter_t: f32,
    glass_exit_t: f32,
    // Leaf_attr slot of the front-face glass voxel — the first
    // transparent cell the ray entered. Used so paint picking can
    // target the glass voxel itself instead of the opaque voxel
    // behind it (which is what `first_slot` records). 0 when the
    // ray didn't hit glass.
    glass_slot: u32,
}

// Pack a unit normal into an oct u32 — mirror of `unpack_oct_normal`
// above, same basis rkp_core / skin_deform use.
fn pack_oct_normal(n: vec3<f32>) -> u32 {
    let l1 = abs(n.x) + abs(n.y) + abs(n.z);
    let n1 = n / max(l1, 1e-8);
    var u = n1.x;
    var v = n1.y;
    if n1.z < 0.0 {
        let ox = u;
        u = (1.0 - abs(v)) * select(-1.0, 1.0, ox >= 0.0);
        v = (1.0 - abs(ox)) * select(-1.0, 1.0, v  >= 0.0);
    }
    let ui = i32(clamp(u, -1.0, 1.0) * 32767.0);
    let vi = i32(clamp(v, -1.0, 1.0) * 32767.0);
    let ul = u32(ui & 0xFFFF);
    let vl = u32(vi & 0xFFFF);
    return ul | (vl << 16u);
}

// ── Phase-3b: skinned march — deformed-field lookup ──────────────────
//
// The scatter pass has pre-rotated each surface voxel's normal into
// deformed space and written `(leaf_slot, normal_oct)` into the bone
// field at the forward-skinned cell (plus a 2×2×2 splat to close
// sparse-scatter gaps). The march's job is a plain walk of deformed
// space: step voxel-by-voxel; first populated cell is the hit.
//
// No inverse skinning, no rest-octree descent. LBS is non-invertible
// at joints — descending the rest octree at the (imperfect) inverse-
// skinned position was picking neighbouring leaves' data or empty
// cells, producing the "tears" + wrong-looking normals the user was
// seeing. Forward splat dodges the whole class of issues.

const SKINNED_MAX_STEPS: u32 = 512u;
const OCC_BRICK_DIM: i32 = 4;

/// Look up the bone field at a deformed-cell coordinate. Returns
/// `(0u, 0u)` when out of bounds or unpopulated.
fn sample_bone_field(cell: vec3<i32>, dims: vec3<i32>, offset: u32) -> vec2<u32> {
    if any(cell < vec3<i32>(0)) || any(cell >= dims) {
        return vec2<u32>(0u);
    }
    let ux = u32(cell.x);
    let uy = u32(cell.y);
    let uz = u32(cell.z);
    let idx = ux + uy * u32(dims.x) + uz * u32(dims.x) * u32(dims.y);
    return bone_field[offset + idx];
}

/// Test whether the 4³-cell brick containing `cell` has any populated
/// cell (scatter sets the bit via `atomicOr`). Returns `false` for
/// bricks outside the grid so the march treats them as empty and skips
/// past.
fn bone_field_brick_populated(
    cell: vec3<i32>,
    cell_dims: vec3<i32>,
    occ_offset: u32,
) -> bool {
    if any(cell < vec3<i32>(0)) || any(cell >= cell_dims) {
        return false;
    }
    let brick = cell / vec3<i32>(OCC_BRICK_DIM);
    let bx_dim = u32((cell_dims.x + OCC_BRICK_DIM - 1) / OCC_BRICK_DIM);
    let by_dim = u32((cell_dims.y + OCC_BRICK_DIM - 1) / OCC_BRICK_DIM);
    let brick_idx = u32(brick.x)
        + u32(brick.y) * bx_dim
        + u32(brick.z) * bx_dim * by_dim;
    let word = bone_field_occ[occ_offset + (brick_idx >> 5u)];
    return (word & (1u << (brick_idx & 31u))) != 0u;
}

/// Ray-t at which the ray exits the 4³ brick containing `cell`.
/// Uses slab intersection against the brick's oc-space bounds. Caller
/// nudges past the returned t by a small epsilon to land in the next
/// brick's interior.
fn skinned_brick_exit_t(
    origin: vec3<f32>,
    inv_dir: vec3<f32>,
    cell: vec3<i32>,
    grid_origin: vec3<f32>,
    vs: f32,
) -> f32 {
    let brick = cell / vec3<i32>(OCC_BRICK_DIM);
    let brick_min = grid_origin + vec3<f32>(brick * OCC_BRICK_DIM) * vs;
    let brick_max = brick_min + vec3<f32>(f32(OCC_BRICK_DIM) * vs);
    let t0 = (brick_min - origin) * inv_dir;
    let t1 = (brick_max - origin) * inv_dir;
    let t_far = max(t0, t1);
    return min(t_far.x, min(t_far.y, t_far.z));
}

/// Skinned march — direct deformed-field lookup. See the helper
/// doc-block above for the architecture rationale.
fn march_object_skinned(
    world_origin: vec3<f32>, world_dir: vec3<f32>, obj: RkpObject,
) -> MarchResult {
    var result = MarchResult(
        vec3<f32>(0.0), vec3<f32>(0.0), vec3<f32>(0.0),
        0.0, 0.0, 0u, false, 0u,
        false, vec3<f32>(0.0), 0u, 0.0, 0.0,
        0u, // glass_slot
    );

    let inv_world = obj.inverse_world;
    let local_origin_mesh = (inv_world * vec4<f32>(world_origin, 1.0)).xyz;
    let local_dir_unnorm = (inv_world * vec4<f32>(world_dir, 0.0)).xyz;
    let local_dir = normalize(local_dir_unnorm);
    let vs = obj.voxel_size;

    let rest_extent = bitcast<f32>(obj.rest_octree_extent_bits);
    // Scatter + bone field all live in grid frame (origin at octree
    // corner, range [0, extent]); the ray enters in mesh frame from
    // `inverse_world`. Shift once up front.
    let half_rest_ext = rest_extent * 0.5;
    let local_origin = local_origin_mesh + vec3<f32>(half_rest_ext);

    let grid_dim = vec3<i32>(
        i32(obj.bone_field_dim_x),
        i32(obj.bone_field_dim_y),
        i32(obj.bone_field_dim_z),
    );
    if grid_dim.x <= 0 || grid_dim.y <= 0 || grid_dim.z <= 0 {
        return result; // no scatter this frame; caller falls back to rigid path
    }
    atomicAdd(&stats[55], 1u); // skinned-branch entry

    let grid_origin = vec3<f32>(
        obj.bone_field_origin_x,
        obj.bone_field_origin_y,
        obj.bone_field_origin_z,
    );
    let grid_max = grid_origin + vec3<f32>(grid_dim) * vs;

    let safe_dir = vec3<f32>(
        select(local_dir.x, select(-1e-10, 1e-10, local_dir.x >= 0.0), abs(local_dir.x) < 1e-10),
        select(local_dir.y, select(-1e-10, 1e-10, local_dir.y >= 0.0), abs(local_dir.y) < 1e-10),
        select(local_dir.z, select(-1e-10, 1e-10, local_dir.z >= 0.0), abs(local_dir.z) < 1e-10),
    );
    let inv_dir = 1.0 / safe_dir;

    let t_range = intersect_aabb(local_origin, inv_dir, grid_origin, grid_max);
    if t_range.x > t_range.y { return result; }

    var t = max(t_range.x, 0.0) + vs * 0.001;
    var step_count = 0u;

    for (var step = 0u; step < SKINNED_MAX_STEPS; step++) {
        step_count += 1u;
        if t > t_range.y { break; }

        let p_local = local_origin + safe_dir * t;
        let cell_f = (p_local - grid_origin) / vs;
        let cell_i = vec3<i32>(floor(cell_f));

        // Brick-level empty-space skip. The scatter pass tags every
        // 4³ brick that got any cell write; if this brick is clear,
        // fast-forward `t` to the brick's far-side exit so we skip
        // up to 64 cell reads with a single bit test. `atomicAdd`s
        // 58/59 are the telemetry for brick-skip hit rate — read with
        // the [skin march] stats line in engine.rs.
        if !bone_field_brick_populated(cell_i, grid_dim, obj.bone_field_occ_offset) {
            atomicAdd(&stats[58], 1u); // empty-brick skip
            let t_exit = skinned_brick_exit_t(local_origin, inv_dir, cell_i, grid_origin, vs);
            t = max(t + vs * 0.01, t_exit + vs * 0.001);
            continue;
        }
        atomicAdd(&stats[59], 1u); // populated-brick sample

        let cell = sample_bone_field(cell_i, grid_dim, obj.bone_field_offset);

        let leaf_slot = cell.x;
        let normal_oct = cell.y;
        if leaf_slot == 0u {
            // Empty cell within a populated brick — scatter's 2×2×2
            // splat keeps rigid regions covered; step one voxel.
            t += vs;
            continue;
        }
        atomicAdd(&stats[57], 1u); // populated cell read
        atomicAdd(&stats[56], 1u); // hit

        let deformed_normal = unpack_oct_normal(normal_oct);

        // Per-voxel color — mirror the rigid path's lookup. Indexed
        // by the scatter-time leaf_slot so we get the right voxel's
        // albedo regardless of LBS skew.
        atomicAdd(&stats[46], 1u);
        let cp = color_pool_data[leaf_slot];
        var color = vec3<f32>(0.5);
        if cp != 0u {
            color = vec3<f32>(
                f32(cp & 0xFFu) / 255.0,
                f32((cp >> 8u) & 0xFFu) / 255.0,
                f32((cp >> 16u) & 0xFFu) / 255.0,
            );
        }

        result.oc_pos = p_local;
        result.normal = deformed_normal;
        result.color = color;
        result.alpha = 1.0;
        result.t = t;
        result.first_slot = leaf_slot;
        result.valid = true;
        result.steps = step_count;
        return result;
    }

    result.steps = step_count;
    return result;
}

fn march_object(
    world_origin: vec3<f32>, world_dir: vec3<f32>, obj: RkpObject,
) -> MarchResult {
    // Phase-3b: skinned objects inverse-skin at march time. Unskinned
    // objects fall through to the existing rest-octree DDA.
    if obj.is_skinned != 0u && obj.bone_count > 0u && obj.bone_field_dim_x > 0u {
        return march_object_skinned(world_origin, world_dir, obj);
    }
    var result = MarchResult(
        vec3<f32>(0.0), vec3<f32>(0.0), vec3<f32>(0.0),
        0.0, 0.0, 0u, false, 0u,
        false, vec3<f32>(0.0), 0u, 0.0, 0.0,
        0u, // glass_slot
    );

    let inv_world = obj.inverse_world;
    let local_origin = (inv_world * vec4<f32>(world_origin, 1.0)).xyz;
    let local_dir_unnorm = (inv_world * vec4<f32>(world_dir, 0.0)).xyz;
    let local_dir = normalize(local_dir_unnorm);
    // Conversion from oc-space (where `t` marches) to world units.
    // `length(local_dir_unnorm) = 1/S` for uniform scale S, so the
    // reciprocal gives world_distance = oc_distance * local_to_world.
    let local_to_world = 1.0 / max(length(local_dir_unnorm), 1e-8);
    // camera.up.xyz encodes tan(half_fov_y) — same decoding as the
    // post-hit footprint histogram.
    let focal_px_y = 0.5 * camera.resolution.y / max(length(camera.up.xyz), 1e-6);

    let root = obj.octree_root;
    let max_depth = obj.octree_depth;
    let extent = bitcast<f32>(obj.octree_extent_bits);
    let vs = obj.voxel_size;
    let half_ext = extent * 0.5;

    let oc_origin = local_origin - obj.grid_origin;
    let safe_dir = vec3<f32>(
        select(local_dir.x, select(-1e-10, 1e-10, local_dir.x >= 0.0), abs(local_dir.x) < 1e-10),
        select(local_dir.y, select(-1e-10, 1e-10, local_dir.y >= 0.0), abs(local_dir.y) < 1e-10),
        select(local_dir.z, select(-1e-10, 1e-10, local_dir.z >= 0.0), abs(local_dir.z) < 1e-10),
    );
    let inv_dir = 1.0 / safe_dir;

    let t_range = intersect_aabb(oc_origin, inv_dir, vec3<f32>(0.0), vec3<f32>(extent));
    if t_range.x > t_range.y {
        return result;
    }

    var t = t_range.x;
    var step_count = 0u;
    // Forward bias for octree_lookup / skip_node — disambiguates the
    // pos-on-exact-boundary case where `pos >= center` would otherwise
    // round into an EMPTY sibling subtree and miss the brick we're
    // entering. See rkp_shadow_trace.wgsl.
    let lookup_bias = vs * 1.0e-3;

    for (var step = 0u; step < MAX_STEPS; step++) {
        step_count += 1u;
        if t > t_range.y { break; }
        if result.alpha > 0.99 { break; }

        let pos = clamp(oc_origin + safe_dir * (t + lookup_bias), vec3<f32>(vs * 0.01), vec3<f32>(extent - vs * 0.01));
        let r = octree_lookup(root, max_depth, extent, pos, PHASE_MARCH, t, local_to_world, focal_px_y);

        if r.slot == OCTREE_EMPTY {
            t += skip_node(pos, safe_dir, inv_dir, r.depth, extent, vs);
            continue;
        }

        // BRICK: descend into a flat 4³ cell array. The DDA below stays in
        // this brick until the ray exits its bounds or the accumulator
        // saturates. Each step inside the brick is one flat read — no more
        // octree descent until we leave the brick.
        if slot_is_brick(r.slot) {
            var brick_id = slot_brick_id(r.slot);
            let cell_size = (r.cell_half * 2.0) / BRICK_DIM_F;
            let inv_cell_size = 1.0 / cell_size;
            var brick_origin = r.cell_center - vec3<f32>(r.cell_half);
            var brick_base = brick_id * BRICK_CELLS;

            // Amanatides-Woo DDA + brick_face_links chaining. When the
            // ray exits a brick face, we consult the face-link table in
            // one indirect read rather than re-querying the octree,
            // bypassing the FP-ambiguous `pos >= center` comparisons at
            // brick boundaries that produced visible seams.
            let p0 = oc_origin + safe_dir * t;
            let local0 = (p0 - brick_origin) * inv_cell_size;
            var cell = clamp(
                vec3<i32>(floor(local0)),
                vec3<i32>(0),
                vec3<i32>(3),
            );
            let step_i = vec3<i32>(
                select(-1, 1, safe_dir.x >= 0.0),
                select(-1, 1, safe_dir.y >= 0.0),
                select(-1, 1, safe_dir.z >= 0.0),
            );
            let step_gt = vec3<f32>(
                select(0.0, 1.0, safe_dir.x >= 0.0),
                select(0.0, 1.0, safe_dir.y >= 0.0),
                select(0.0, 1.0, safe_dir.z >= 0.0),
            );
            let next_b = brick_origin + (vec3<f32>(cell) + step_gt) * cell_size;
            var t_max = t + (next_b - p0) * inv_dir;
            let t_delta = abs(vec3<f32>(cell_size) * inv_dir);
            let dda_eps = cell_size * 1.0e-3;

            var brick_done = false;
            for (var bs = 0u; bs < BRICK_MAX_STEPS; bs++) {
                step_count += 1u;
                if t > t_range.y { brick_done = true; break; }
                if result.alpha > 0.99 { brick_done = true; break; }

                if cell.x < 0 || cell.x >= 4
                    || cell.y < 0 || cell.y >= 4
                    || cell.z < 0 || cell.z >= 4 {
                    var face_idx: u32;
                    if cell.x < 0 { face_idx = FACE_NX; }
                    else if cell.x >= 4 { face_idx = FACE_PX; }
                    else if cell.y < 0 { face_idx = FACE_NY; }
                    else if cell.y >= 4 { face_idx = FACE_PY; }
                    else if cell.z < 0 { face_idx = FACE_NZ; }
                    else { face_idx = FACE_PZ; }
                    let link = brick_face_links[brick_id * 6u + face_idx];
                    if link == FACE_INTERIOR {
                        // Ray is about to enter a solid-bulk region
                        // beyond this brick's face. For a glass object
                        // (or a ray that has already entered glass),
                        // the bulk is part of the glass body — we
                        // want to skip through it. Bail out of the
                        // brick DDA and let the outer loop handle the
                        // adjacent INTERIOR_NODE, which my OCTREE_
                        // INTERIOR handler converts into a glass skip
                        // with thickness tracking. For non-glass
                        // objects, keep the original opaque-fallback
                        // behavior so solid meshes with interior bulk
                        // still terminate the march here.
                        let obj_opacity = materials[obj.material_id].opacity;
                        if obj_opacity < 0.99 || result.glass_valid {
                            if !result.glass_valid {
                                result.glass_valid = true;
                                result.glass_normal = -safe_dir;
                                result.glass_material = obj.material_id;
                                result.glass_enter_t = t;
                            }
                            result.glass_exit_t = t;
                            break; // outer loop takes over via skip_node on INTERIOR
                        }
                        let p = oc_origin + safe_dir * t;
                        result.oc_pos = p;
                        result.normal = -safe_dir;
                        result.alpha = 1.0;
                        result.t = t;
                        result.first_slot = 0u;
                        result.valid = true;
                        // No leaf_slot for interior hit → no per-voxel
                        // colour override. Write 0 so the gbuffer's
                        // RGB565 channel stays 0, which `rkp_shade.wgsl`
                        // reads as "use material albedo" (writing 0.5
                        // would pack as a non-zero RGB565 and override
                        // the material with grey).
                        result.color = vec3<f32>(0.0);
                        result.steps = step_count;
                        brick_done = true;
                        break;
                    }
                    if link == FACE_EMPTY_LINK {
                        // No same-depth brick adjacent — fall back to
                        // the outer loop's skip_node.
                        break;
                    }
                    brick_id = link;
                    brick_base = link * BRICK_CELLS;
                    // Shift brick_origin to the neighbor brick's world-space
                    // corner and reset the crossed-axis cell coord to its
                    // entry edge in the new brick.
                    let brick_extent = BRICK_DIM_F * cell_size;
                    if face_idx == FACE_NX { cell.x = 3; brick_origin.x -= brick_extent; }
                    else if face_idx == FACE_PX { cell.x = 0; brick_origin.x += brick_extent; }
                    else if face_idx == FACE_NY { cell.y = 3; brick_origin.y -= brick_extent; }
                    else if face_idx == FACE_PY { cell.y = 0; brick_origin.y += brick_extent; }
                    else if face_idx == FACE_NZ { cell.z = 3; brick_origin.z -= brick_extent; }
                    else { cell.z = 0; brick_origin.z += brick_extent; }
                    // Re-anchor `t_max` from the current ray position to
                    // the new brick's cell boundaries. The incremental
                    // `t_max += t_delta` updates accumulate FP rounding
                    // over many iterations; letting that drift carry across
                    // brick chains at large octree extents eventually
                    // causes `t_max.x < t_max.y` to pick the wrong axis,
                    // producing grid-aligned cell skips (visible as the
                    // scale-dependent voxel-hole artifact). Re-anchoring
                    // at every face-link crossing caps the drift to one
                    // brick's worth of steps.
                    let p_now = oc_origin + safe_dir * t;
                    let next_b = brick_origin + (vec3<f32>(cell) + step_gt) * cell_size;
                    t_max = t + (next_b - p_now) * inv_dir;
                }

                let cx = u32(cell.x);
                let cy = u32(cell.y);
                let cz = u32(cell.z);
                let flat = cx + cy * BRICK_DIM + cz * BRICK_DIM * BRICK_DIM;
                let c = brick_pool[brick_base + flat];

                // BRICK_CELL_INTERIOR cells are mesh-import solid bulk
                // markers with no per-cell leaf_attr. For glass objects
                // they still want to contribute thickness — treat
                // them the same way we treat an OCTREE_INTERIOR node:
                // if the object is glass (or we're already in glass),
                // record / extend glass and move on. Non-glass objects
                // keep the original "skip like empty air" semantics.
                if c == BRICK_CELL_INTERIOR {
                    let obj_opacity = materials[obj.material_id].opacity;
                    if obj_opacity < 0.99 || result.glass_valid {
                        if !result.glass_valid {
                            result.glass_valid = true;
                            result.glass_normal = -safe_dir;
                            result.glass_material = obj.material_id;
                            result.glass_enter_t = t;
                        }
                        result.glass_exit_t = t;
                    }
                    // Fall through to DDA step (skip either way).
                }

                // BRICK_CELL_INTERIOR cells are solid-bulk markers set
                // by mesh imports; skip them identically to empty air
                // so the march only ever stops on the visible shell.
                if c != BRICK_CELL_EMPTY && c != BRICK_CELL_INTERIOR {
                    atomicAdd(&stats[44], 1u); // leaf_attr read
                    let attr = leaf_attr_pool[c];
                    let baked_normal = unpack_oct_normal(attr.normal_oct);
                    var cell_normal: vec3<f32>;
                    if march_params.surfacenet_enabled != 0u {
                        cell_normal = reconstruct_normal_surfacenet(
                            brick_id, cx, cy, cz, baked_normal,
                        );
                        atomicAdd(&stats[52], 1u);
                    } else {
                        cell_normal = baked_normal;
                    }
                    let mid = leaf_attr_material_primary(attr);
                    atomicAdd(&stats[47], 1u); // materials read
                    let m_opacity = materials[mid].opacity;

                    if m_opacity >= 0.99 {
                        // Opaque — this is the primary hit (the
                        // "behind" the glass, if any).
                        let p = oc_origin + safe_dir * t;
                        result.oc_pos = p;
                        result.normal = cell_normal;
                        result.alpha = 1.0;
                        result.t = t;
                        result.first_slot = c;
                        result.valid = true;
                        // Default to 0 (no override) so the gbuffer's
                        // RGB565 stays 0 and `rkp_shade.wgsl` falls back
                        // to material albedo. A non-zero default would
                        // pack as a non-zero RGB565 and override the
                        // material colour with whatever default we set.
                        var color = vec3<f32>(0.0);
                        atomicAdd(&stats[46], 1u); // color_pool read
                        let cp = color_pool_data[c];
                        if cp != 0u {
                            color = vec3<f32>(
                                f32(cp & 0xFFu) / 255.0,
                                f32((cp >> 8u) & 0xFFu) / 255.0,
                                f32((cp >> 16u) & 0xFFu) / 255.0,
                            );
                        }
                        result.color = color;
                        result.steps = step_count;
                        brick_done = true;
                        break;
                    } else {
                        // Glass cell — record entry the first time,
                        // update exit every time, but keep marching
                        // to find the opaque behind. Normal of the
                        // glass surface = the entry cell's normal,
                        // which points OUT of the glass body (toward
                        // the camera for a front-face hit). Subsequent
                        // glass cells don't overwrite the normal —
                        // the front face is what matters for Fresnel.
                        if !result.glass_valid {
                            result.glass_valid = true;
                            result.glass_normal = cell_normal;
                            result.glass_material = mid;
                            result.glass_enter_t = t;
                            result.glass_slot = c;
                        }
                        result.glass_exit_t = t;
                        // Fall through to DDA step below.
                    }
                }

                // DDA step to next cell along axis with smallest t_max.
                if t_max.x < t_max.y && t_max.x < t_max.z {
                    t = t_max.x + dda_eps;
                    cell.x += step_i.x;
                    t_max.x += t_delta.x;
                } else if t_max.y < t_max.z {
                    t = t_max.y + dda_eps;
                    cell.y += step_i.y;
                    t_max.y += t_delta.y;
                } else {
                    t = t_max.z + dda_eps;
                    cell.z += step_i.z;
                    t_max.z += t_delta.z;
                }
            }
            if brick_done { break; }
            continue;
        }

        // Leaf / INTERIOR hit. Opaque cells are a first-surface stop
        // (primary hit / behind-glass). Transparent cells record
        // themselves as glass-in-front and the march continues so
        // we can deliver the opaque behind to the G-buffer.
        //
        // INTERIOR handling is split by whether we've already
        // entered glass:
        //  - Not in glass yet: an INTERIOR node means we entered
        //    solid bulk directly (e.g. a non-glass opaque primitive
        //    collapsed to an INTERIOR subtree); treat as an opaque
        //    hit with a ray-opposite normal fallback, same as the
        //    pre-glass code.
        //  - Already in glass: the INTERIOR subtree is part of the
        //    glass body (solid voxelized primitives fill their
        //    interior). Skip past it via `skip_node` so we reach
        //    the opaque surface behind instead of stopping here.
        // INTERIOR subtrees (fully-solid bulk regions produced by the
        // voxelizer when a whole subtree is inside the object) carry
        // no per-voxel material — `RemapMaterial` can't reach them.
        // Treat them as the object's default material so a glass
        // cube whose interior collapsed to INTERIOR_NODE still reads
        // as glass throughout. The object's `material_id` IS updated
        // by `AssignMaterial` / scene load / drag-drop, so this
        // reliably reflects the user's intent.
        if r.slot == OCTREE_INTERIOR {
            let obj_opacity = materials[obj.material_id].opacity;
            if obj_opacity < 0.99 || result.glass_valid {
                // Glass bulk — skip through, growing thickness.
                if !result.glass_valid {
                    result.glass_valid = true;
                    result.glass_normal = -safe_dir;
                    result.glass_material = obj.material_id;
                    result.glass_enter_t = t;
                }
                t += skip_node(pos, safe_dir, inv_dir, r.depth, extent, vs);
                result.glass_exit_t = max(result.glass_exit_t, t);
                continue;
            }
        }

        var leaf_id = 0u;                  // leaf_attr_id for this hit (for main())
        // For INTERIOR (fully opaque bulk region) we have no stored normal —
        // the ray-opposite is a cheap safe default.
        var sample_normal = -safe_dir;
        var mid = 0u;
        var m_opacity = 1.0;
        if r.slot != OCTREE_INTERIOR {
            atomicAdd(&stats[44], 1u); // leaf_attr read
            let attr = leaf_attr_pool[r.slot];
            leaf_id = r.slot;
            sample_normal = unpack_oct_normal(attr.normal_oct);
            mid = leaf_attr_material_primary(attr);
            atomicAdd(&stats[47], 1u); // materials read
            m_opacity = materials[mid].opacity;
        }

        if m_opacity < 0.99 {
            // Glass — record entry, keep marching. Leaf path
            // advances by half a voxel at a time (see below), so
            // updating glass_exit_t every visit gives thickness.
            if !result.glass_valid {
                result.glass_valid = true;
                result.glass_normal = sample_normal;
                result.glass_material = mid;
                result.glass_enter_t = t;
                result.glass_slot = leaf_id;
            }
            result.glass_exit_t = t;
            t += vs * 0.5;
            continue;
        }

        result.oc_pos = pos;
        result.normal = sample_normal;
        result.alpha = 1.0;
        result.t = t;
        result.first_slot = leaf_id;
        result.valid = true;
        // Default 0 = "no override" — packs to RGB565 = 0, which
        // `rkp_shade.wgsl` reads as "fall back to material albedo".
        // Both INTERIOR hits (no leaf_slot to look up) and LEAF hits
        // whose color_pool entry is 0 (never painted, or fully erased)
        // route to material albedo via this path.
        var color = vec3<f32>(0.0);
        if r.slot != OCTREE_INTERIOR {
            atomicAdd(&stats[46], 1u); // color_pool read
            let cp = color_pool_data[leaf_id];
            if cp != 0u {
                color = vec3<f32>(
                    f32(cp & 0xFFu) / 255.0,
                    f32((cp >> 8u) & 0xFFu) / 255.0,
                    f32((cp >> 16u) & 0xFFu) / 255.0,
                );
            }
        }
        result.color = color;
        result.steps = step_count;
        break; // done — first-surface hit
    }

    result.steps = step_count;
    return result;
}

// --- Main ---

@compute @workgroup_size(8, 8, 1)
fn main(
    @builtin(global_invocation_id) pixel: vec3<u32>,
    @builtin(local_invocation_index) local_idx: u32,
) {
    // Per-tile object list: thread 0 fetches this tile's slice bounds
    // once; every thread in the 8x8 workgroup then iterates the same
    // range. Culling already happened CPU-side — the list only contains
    // objects whose screen AABB overlaps this tile, so no per-object
    // AABB re-check needed here.
    if local_idx == 0u {
        let tx = pixel.x / 8u;
        let ty = pixel.y / 8u;
        let tile_idx = ty * march_params.tile_count_x + tx;
        tile_range_start = tile_offsets[tile_idx];
        tile_range_end = tile_offsets[tile_idx + 1u];
    }
    workgroupBarrier();

    let dims = textureDimensions(gbuf_position);
    if pixel.x >= dims.x || pixel.y >= dims.y { return; }

    // No objects overlap this tile — write background and skip.
    if tile_range_start == tile_range_end {
        let coord = vec2<i32>(pixel.xy);
        textureStore(gbuf_position, coord, vec4<f32>(0.0, 0.0, 0.0, 1e10));
        textureStore(gbuf_normal, coord, vec4<f32>(0.0, 0.0, 0.0, 0.0));
        textureStore(gbuf_material, coord, vec4<u32>(0u, 0u, 0u, 0u));
        textureStore(gbuf_pick, coord, vec4<u32>(0xFFFFFFFFu, 0u, 0u, 0u));
        textureStore(gbuf_leaf_slot, coord, vec4<u32>(0u, 0u, 0u, 0u));
        return;
    }

    let coord = vec2<i32>(pixel.xy);
    let uv = (vec2<f32>(pixel.xy) + 0.5 + camera.jitter) / camera.resolution;
    let ndc = vec2<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    let ray_origin = camera.position.xyz;
    let ray_dir = normalize(camera.forward.xyz + ndc.x * camera.right.xyz + ndc.y * camera.up.xyz);

    // Single-pass: march every object once, keep closest opaque hit. O(N) instead
    // of O(N²) selection sort. AABB culling skips objects behind the closest hit.

    var accum_pos = vec3<f32>(0.0);
    var accum_normal = vec3<f32>(0.0);
    var accum_color = vec3<f32>(0.0);
    var accum_alpha = 0.0;
    var first_dist = 0.0;
    var first_mat_id = 0u;
    var first_sec_mat = 0u;
    var first_blend = 0u;
    // Leaf_attr slot of the primary hit — written to `gbuf_leaf_slot`
    // for the geodesic paint cursor to look up per-voxel brush overlay
    // distances. 0 stays 0 on sky / no-hit pixels.
    var first_leaf_slot: u32 = 0u;
    var first_obj_id = 0u;
    var have_first = false;
    var max_world_dist = 1e20; // world-space distance to closest opaque hit
    var closest_obj_idx = 0xFFFFFFFFu; // index of closest hit object (for shadow skip)

    // Pick tracking — distinct from the shaded "first opaque hit"
    // accumulators above. Picking wants the FIRST surface the ray
    // touched, whether it was glass or opaque, so clicking on a
    // glass cube selects the cube rather than punching through to
    // whatever's behind it. Seeded to infinity; replaced on each
    // closer glass entry or opaque hit.
    var pick_obj_id = 0xFFFFFFFFu;
    var pick_dist = 1e20;
    // Paint-cursor leaf_slot tied to `pick_dist` — whichever is
    // nearest along the ray (glass front face vs opaque hit) wins the
    // `gbuf_leaf_slot` write so paint clicks target the voxel the
    // user can actually see, not whatever's hidden behind glass.
    var pick_leaf_slot: u32 = 0u;

    // Glass accumulator — tracked across objects so a glass pane on
    // object A properly tints the shaded surface of object B behind
    // it. `glass_enter_dist` is the nearest glass front-face world
    // distance, `glass_exit_dist` the deepest glass-cell world
    // distance from that object; their difference is the thickness
    // proxy (entry → last-glass-cell, over-counts air gaps in
    // nested glass but is correct for the single-pane case). Glass
    // only contributes to the final pixel if it sits in FRONT of
    // the closest opaque hit (`glass_enter_dist < max_world_dist`).
    var glass_have = false;
    var glass_enter_dist = 1e20;
    var glass_exit_dist = 0.0;
    var glass_normal_world = vec3<f32>(0.0, 1.0, 0.0);
    var glass_material_id = 0u;
    var total_steps = 0u;

    for (var k = tile_range_start; k < tile_range_end; k++) {
        let i = tile_object_ids[k];
        let obj = objects[i];
        if obj.geom_type == 0u { continue; }
        // Layer + focus gate — retained here since the CPU tile-list
        // builder runs before layer state is known to it. Default
        // uniforms (u32::MAX) make this a cheap no-op.
        if !rkp_object_visible(obj) { continue; }

        // AABB check: compute world-space entry distance, skip if behind closest hit.
        let inv_world = obj.inverse_world;
        let local_origin = (inv_world * vec4<f32>(ray_origin, 1.0)).xyz;
        let local_dir_unnorm = (inv_world * vec4<f32>(ray_dir, 0.0)).xyz;
        let local_to_world_scale = 1.0 / max(length(local_dir_unnorm), 1e-10);
        let local_dir = normalize(local_dir_unnorm);
        let extent = bitcast<f32>(obj.octree_extent_bits);
        let half_ext = extent * 0.5;
        let oc_origin = local_origin - obj.grid_origin;
        let safe_d = vec3<f32>(
            select(local_dir.x, select(-1e-10, 1e-10, local_dir.x >= 0.0), abs(local_dir.x) < 1e-10),
            select(local_dir.y, select(-1e-10, 1e-10, local_dir.y >= 0.0), abs(local_dir.y) < 1e-10),
            select(local_dir.z, select(-1e-10, 1e-10, local_dir.z >= 0.0), abs(local_dir.z) < 1e-10),
        );
        let t_range = intersect_aabb(oc_origin, 1.0 / safe_d, vec3<f32>(0.0), vec3<f32>(extent));
        if t_range.x > t_range.y { continue; } // ray misses AABB
        let world_entry = t_range.x * local_to_world_scale;
        if world_entry > max_world_dist { continue; } // AABB entirely behind closest hit

        // March this object.
        let r = march_object(ray_origin, ray_dir, obj);
        total_steps += r.steps;

        // Pull glass info out of this object's march, if any. Glass
        // can be present even when `r.valid == false` (ray passed
        // through glass and exited the object without finding an
        // opaque cell behind) — useful when the opaque surface is
        // in a DIFFERENT object behind this one. The winning glass
        // is the nearest entry-point ahead of the closest opaque
        // hit.
        if r.glass_valid {
            let g_enter = r.glass_enter_t * local_to_world_scale;
            let g_exit = r.glass_exit_t * local_to_world_scale;
            if g_enter < glass_enter_dist && g_enter < max_world_dist {
                glass_have = true;
                glass_enter_dist = g_enter;
                glass_exit_dist = g_exit;
                let world_n = normalize((obj.world * vec4<f32>(r.glass_normal, 0.0)).xyz);
                glass_normal_world = world_n;
                glass_material_id = r.glass_material;
            }
            // Pick also considers glass hits — click on a glass
            // cube should select the cube, not the opaque object
            // behind it.
            if g_enter < pick_dist {
                pick_dist = g_enter;
                pick_obj_id = obj.object_id;
                pick_leaf_slot = r.glass_slot;
            }
        }

        if !r.valid { continue; }

        // Compute world-space hit position and distance.
        let inv_a = 1.0 / max(r.alpha, 0.001);
        let oc_pos = r.oc_pos * inv_a;
        let color = r.color * inv_a;
        // Normal accumulated in march_object from per-leaf stored normals,
        // weighted by the same coverage that weights position/color. Single
        // normalize here replaces the old 48-tap trilinear gradient — this
        // is where the perf cliff used to sit.
        let local_normal_raw = r.normal * inv_a;
        let local_normal = normalize(local_normal_raw);

        // Convert octree-local hit back to entity-local, then world.
        let local_hit = oc_pos + obj.grid_origin;
        let world_pos = (obj.world * vec4<f32>(local_hit, 1.0)).xyz;
        let hit_dist = length(world_pos - ray_origin);

        // Skip hits beyond the closest opaque surface.
        if hit_dist > max_world_dist { continue; }

        let world_normal = normalize((obj.world * vec4<f32>(local_normal, 0.0)).xyz);

        // Opaque hit closer than current best: replace the accumulator entirely.
        if r.alpha > 0.99 {
            accum_pos = world_pos;
            accum_normal = world_normal;
            accum_color = color;
            accum_alpha = 1.0;
            first_dist = hit_dist;
            first_obj_id = obj.object_id;
            if r.first_slot != 0u {
                let attr = leaf_attr_pool[r.first_slot];
                first_mat_id = leaf_attr_material_primary(attr);
                first_sec_mat = leaf_attr_material_secondary(attr);
                first_blend = leaf_attr_blend_weight(attr);
                first_leaf_slot = r.first_slot;
            } else {
                first_mat_id = obj.material_id;
            }
            have_first = true;
            max_world_dist = hit_dist;
            closest_obj_idx = i;
            if hit_dist < pick_dist {
                pick_dist = hit_dist;
                pick_obj_id = obj.object_id;
                pick_leaf_slot = r.first_slot;
            }
            continue;
        }

        // Transparent: accumulate (approximate — not depth-sorted across objects).
        let remaining = 1.0 - accum_alpha;
        let weight = r.alpha * remaining;
        accum_pos += world_pos * weight;
        accum_normal += world_normal * weight;
        accum_color += color * weight;
        accum_alpha += weight;

        if !have_first {
            first_dist = hit_dist;
            first_obj_id = obj.object_id;
            closest_obj_idx = i;
            if r.first_slot != 0u {
                let attr = leaf_attr_pool[r.first_slot];
                first_mat_id = leaf_attr_material_primary(attr);
                first_sec_mat = leaf_attr_material_secondary(attr);
                first_blend = leaf_attr_blend_weight(attr);
                first_leaf_slot = r.first_slot;
            } else {
                first_mat_id = obj.material_id;
            }
            have_first = true;
        }
    }

    // Stats.
    atomicAdd(&stats[0], total_steps);
    atomicMax(&stats[3], total_steps);

    // Footprint histogram: size in pixels of the finest voxel at the hit point.
    // <1px means we walked to a mip level finer than the screen can resolve.
    // camera.up.xyz encodes tan(half_fov_y), so focal_px_y = 0.5 * H / |up|.
    if have_first && closest_obj_idx != 0xFFFFFFFFu {
        let focal_px_y = 0.5 * camera.resolution.y / max(length(camera.up.xyz), 1e-6);
        let hit_vs = objects[closest_obj_idx].voxel_size;
        let footprint = hit_vs * focal_px_y / max(first_dist, 1e-3);
        var bucket = 3u;
        if footprint < 1.0 { bucket = 0u; }
        else if footprint < 2.0 { bucket = 1u; }
        else if footprint < 4.0 { bucket = 2u; }
        atomicAdd(&stats[40u + bucket], 1u);
    }

    // Pack glass info if the ray passed through any glass in front
    // of the closest opaque hit (or with no opaque hit, any glass
    // at all). `thickness_mm` caps at u16::MAX (≈65.5 m) — any
    // deeper glass clamps harmlessly. `material_id` is u16; it
    // shares the lower 16 bits of G.
    // Final depth gate: glass may have been recorded earlier in the
    // tile-list iteration, before a CLOSER opaque hit updated
    // `max_world_dist`. The per-record check at line 1402 only sees
    // `max_world_dist` as it stood AT THAT POINT, so glass packed
    // ahead of a later-iterated closer opaque would leak through
    // (thickness collapses to 0, but the 1.0 clamp floor below used
    // to promote that to `thickness_mm = 1`, which the compositor
    // still rendered — glass drawn in front of closer geometry).
    var glass_packed = vec2<u32>(0u, 0u);
    if glass_have && glass_enter_dist < max_world_dist {
        let thickness = max(0.0, min(glass_exit_dist, max_world_dist) - glass_enter_dist);
        let thickness_mm_raw = u32(clamp(thickness * 1000.0, 1.0, 65535.0));
        let packed_g = (glass_material_id & 0xFFFFu) | (thickness_mm_raw << 16u);
        glass_packed = vec2<u32>(pack_oct_normal(glass_normal_world), packed_g);
    }

    if !have_first {
        // No opaque hit — but the ray may still have touched glass
        // on its way to the horizon, in which case clicking that
        // pixel should select the glass object (not pass through
        // to "sky miss"). Pick channel: use the tracked glass
        // pick_obj_id when present, else the sky-miss sentinel.
        let miss_pick_id = select(0xFFFFFFFFu, pick_obj_id, glass_have);
        textureStore(gbuf_position, coord, vec4<f32>(0.0, 0.0, 0.0, 1e10));
        textureStore(gbuf_normal, coord, vec4<f32>(0.0, 0.0, 0.0, 0.0));
        textureStore(gbuf_material, coord, vec4<u32>(0u, 0u, 0u, 0u));
        textureStore(gbuf_pick, coord, vec4<u32>(miss_pick_id, 0u, 0u, 0u));
        textureStore(gbuf_glass, coord, vec4<u32>(glass_packed.x, glass_packed.y, 0u, 0u));
        textureStore(gbuf_leaf_slot, coord, vec4<u32>(0u, 0u, 0u, 0u));
        return;
    }

    let inv_alpha = 1.0 / max(accum_alpha, 0.001);
    let final_pos = accum_pos * inv_alpha;
    let final_color = accum_color * inv_alpha;

    // Shadow tracing moved to rkp_shadow_trace.wgsl (half-res pass).
    let final_normal_n = normalize(accum_normal);

    let cr = u32(clamp(final_color.r, 0.0, 1.0) * 31.0);
    let cg = u32(clamp(final_color.g, 0.0, 1.0) * 63.0);
    let cb = u32(clamp(final_color.b, 0.0, 1.0) * 31.0);
    let color_rgb565 = cr | (cg << 5u) | (cb << 11u);

    let packed_r = (first_mat_id & 0xFFFFu) | ((first_sec_mat & 0xFFFFu) << 16u);
    // Remap the 4-bit LeafAttr blend (0..15) to the 8-bit G-buffer
    // channel (0..255) via `b << 4 | b` — hits both endpoints (0 → 0,
    // 15 → 255) and spaces the intermediate values evenly. Without
    // this rkp_shade's `blend / 255.0` would cap dual-material lerp
    // at ~5.9 %, which is what showed up as "still looks hard" on
    // MAIN even after the shade-pass fix.
    let first_blend_8 = (first_blend & 0x0Fu) << 4u | (first_blend & 0x0Fu);
    // Bits 8-15 are free after the 8-bit object_id was retired in favor
    // of the dedicated `gbuf_pick` channel below; left as 0 for any
    // future packing that can use 8 bits.
    let packed_g = (first_blend_8 & 0xFFu)
                 | (color_rgb565 << 16u);

    atomicAdd(&stats[2], 1u);
    textureStore(gbuf_position, coord, vec4<f32>(final_pos, first_dist));
    textureStore(gbuf_normal, coord, vec4<f32>(final_normal_n, accum_alpha));
    textureStore(gbuf_material, coord, vec4<u32>(packed_r, packed_g, 0u, 0u));
    // Paint cursor should target whatever voxel is CLOSEST along the
    // ray — if a glass voxel was hit before the opaque, record the
    // glass slot. Same select() shape as `pick_id` below.
    let paint_slot = select(first_leaf_slot, pick_leaf_slot, pick_dist < max_world_dist);
    textureStore(gbuf_leaf_slot, coord, vec4<u32>(paint_slot, 0u, 0u, 0u));
    // Pick uses the nearest visible surface — either the opaque
    // hit or a glass entry, whichever came first along the ray.
    // Falls back to `first_obj_id` (the opaque accumulator) when
    // no glass was touched, which keeps the no-glass path identical
    // to the pre-glass behaviour.
    let pick_id = select(first_obj_id, pick_obj_id, pick_dist < max_world_dist);
    textureStore(gbuf_pick, coord, vec4<u32>(pick_id, 0u, 0u, 0u));
    textureStore(gbuf_glass, coord, vec4<u32>(glass_packed.x, glass_packed.y, 0u, 0u));
}
