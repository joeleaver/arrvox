// Phase-3 skin-deform scatter pass — writes a per-object deformed
// geometry field.
//
// One workgroup per populated octree brick (see `SkinBrickEntry`);
// 4×4×4 = 64 threads, one per cell in the brick. Each thread:
//
//   1. Reads its brick cell from the scene's brick_pool. Cells that
//      carry `BRICK_EMPTY` / `BRICK_INTERIOR` sentinels are skipped
//      (neither is a shell leaf).
//   2. Looks up the cell's 4-bone LBS influences from the scene's
//      bone_weights_buffer at the cell's leaf_attr slot.
//   3. Forward-skins the cell's rest-pose voxel centre and rest
//      normal using the object's forward-pose bone matrices
//      (weighted LBS for position, weighted LBS of upper-3×3 for
//      direction).
//   4. Writes `(leaf_slot, deformed_normal_oct)` into the scene bone
//      field at the nearest deformed-voxel cell plus its 2×2×2
//      positive neighbours — the 8-neighbour splat closes small
//      gaps between stretched rest voxels without needing
//      march-side DDA.
//
// The ray marcher's skinned branch (Phase 3b) just reads the bone
// field: a non-zero `leaf_slot` cell IS a surface hit. No inverse-
// skinning, no rest-octree descent. Normals come out of the field
// pre-rotated so shading is exact at the position the march sees —
// no chance of picking a neighbouring leaf's normal due to LBS
// inexactness at joints.

// ---- Scene bindings (reused from the main octree_march bind group) --
// We don't need the full scene layout — just brick_pool, bone_matrices,
// bone_weights. Using their stable binding numbers from `rkp_scene.rs`
// (see the doc comment at the top of that file) so the pipeline layout
// below can bind RkpScene.bind_group as group 0.

@group(0) @binding(0) var<storage, read> brick_pool: array<u32>;
@group(0) @binding(5) var<storage, read> bone_matrices: array<mat4x4<f32>>;
@group(0) @binding(6) var<storage, read> bone_weights: array<u32>;
// Also need the leaf-attr pool so we can fetch each leaf's rest
// normal to forward-rotate into the deformed field. Same binding as
// the main march reads.
struct LeafAttr {
    normal_oct: u32,
    material_packed: u32,
}
@group(0) @binding(8) var<storage, read> leaf_attr_pool: array<LeafAttr>;
// Bone field — scatter destination. Read-write here, vs. read-only in
// the main march bind group (`octree_march.wgsl`); both groups
// reference the same underlying `RkpScene::bone_field_buffer` but
// never bind it in the same dispatch, so wgpu's usage-scope check is
// satisfied. Cell payload: `(leaf_slot, deformed_normal_oct)`.
// `leaf_slot == 0` = empty (matches the main-shader convention that
// slot 0 is treated as "no slot").
@group(0) @binding(9) var<storage, read_write> bone_field: array<vec2<u32>>;
// Per-brick occupancy bitmap. One bit per 4³ cell brick, packed 32
// bricks per u32. The scatter `atomicOr`s into this bitmap whenever
// it writes a cell; the skinned march reads it via `atomicLoad` to
// skip empty bricks with one lookup instead of 64 cell reads. Same
// buffer is bound read-only into the main bind group as binding 10.
@group(0) @binding(10) var<storage, read_write> bone_field_occ: array<atomic<u32>>;

// ---- Per-dispatch bindings (one dispatch per skinned entity) --------

struct SkinUniforms {
    // Offset into `bone_matrices` where this entity's forward palette
    // starts. Inverse palette sits at `bone_buffer_offset + bone_count`
    // (unused in the scatter path — only the march consumes it).
    bone_buffer_offset: u32,
    bone_count: u32,
    // Cell index in `bone_field` where this entity's grid begins.
    bone_field_offset: u32,
    // Deformed bone-field grid dimensions in voxel cells.
    bone_field_dim_x: u32,
    bone_field_dim_y: u32,
    bone_field_dim_z: u32,
    // Object-local origin of the deformed grid's (0,0,0) cell.
    // `(deformed_pos - grid_origin) / voxel_size` → cell idx.
    grid_origin_x: f32,
    grid_origin_y: f32,
    grid_origin_z: f32,
    voxel_size: f32,
    // Offset into `bone_field_occ` (in u32 words) where this entity's
    // packed brick bitmap begins. Brick dims = ceil(cell_dims / 4).
    bone_field_occ_offset: u32,
    // 5 × u32 pad → 64 B total (Rust struct kept matching).
    _pad0: u32, _pad1: u32, _pad2: u32, _pad3: u32, _pad4: u32,
}

struct SkinBrickEntry {
    // Scene-global brick id (matches BRICK nodes in octree).
    brick_id: u32,
    // Brick-corner grid coord (finest-voxel units).
    origin_x: u32,
    origin_y: u32,
    origin_z: u32,
    // Index of this brick's owning entity within the per-frame
    // `uniforms_arr`. Baked by `SkinBatchScratch::push`.
    uniform_idx: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(1) @binding(0) var<storage, read> uniforms_arr: array<SkinUniforms>;
@group(1) @binding(1) var<storage, read> brick_list: array<SkinBrickEntry>;

// ---- Constants mirroring rkp-core::brick_pool -----------------------

const BRICK_DIM: u32 = 4u;
const BRICK_CELLS: u32 = 64u;               // 4^3
const BRICK_EMPTY: u32 = 0xFFFFFFFFu;
const BRICK_INTERIOR: u32 = 0xFFFFFFFDu;

// Forward-skin a rest-pose position by weighted LBS. Weights are u8
// (0..=255) packed into a u32; indices are u8s packed into another u32.
fn forward_skin(
    rest_pos: vec3<f32>,
    packed_indices: u32,
    packed_weights: u32,
    bone_buffer_offset: u32,
) -> vec3<f32> {
    var acc = vec3<f32>(0.0);
    var total_w = 0.0;
    for (var i = 0u; i < 4u; i = i + 1u) {
        let bone_idx = (packed_indices >> (i * 8u)) & 0xFFu;
        let w = f32((packed_weights >> (i * 8u)) & 0xFFu);
        if w < 1.0 { continue; }
        let mat = bone_matrices[bone_buffer_offset + bone_idx];
        let tp = (mat * vec4<f32>(rest_pos, 1.0)).xyz;
        acc = acc + tp * w;
        total_w = total_w + w;
    }
    if total_w > 0.0 { return acc / total_w; }
    return rest_pos;
}

// Rotate a rest-pose normal by the weighted LBS blend (upper-3×3 of
// the forward matrices applied to a direction via `w=0` vec4). After
// the weighted sum we renormalise — kills any scale the conjugated
// pose introduces.
fn forward_rotate_normal(
    rest_normal: vec3<f32>,
    packed_indices: u32,
    packed_weights: u32,
    bone_buffer_offset: u32,
) -> vec3<f32> {
    var acc = vec3<f32>(0.0);
    var total_w = 0.0;
    for (var i = 0u; i < 4u; i = i + 1u) {
        let bone_idx = (packed_indices >> (i * 8u)) & 0xFFu;
        let w = f32((packed_weights >> (i * 8u)) & 0xFFu);
        if w < 1.0 { continue; }
        let mat = bone_matrices[bone_buffer_offset + bone_idx];
        let rn = (mat * vec4<f32>(rest_normal, 0.0)).xyz;
        acc = acc + rn * w;
        total_w = total_w + w;
    }
    if total_w > 0.0 {
        let out = acc / total_w;
        let l = length(out);
        if l > 1e-6 { return out / l; }
    }
    return rest_normal;
}

// Pack a unit normal into an oct u32 — matches
// `rkp_core::leaf_attr::pack_oct` so the main march can decode it
// with its existing `unpack_oct_normal`.
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
    // Quantise to snorm16 (sign-extended i16 values round-tripped
    // through `unpack_oct_normal` on the read side).
    let ui = i32(clamp(u, -1.0, 1.0) * 32767.0);
    let vi = i32(clamp(v, -1.0, 1.0) * 32767.0);
    let ul = u32(ui & 0xFFFF);
    let vl = u32(vi & 0xFFFF);
    return ul | (vl << 16u);
}

@compute @workgroup_size(4, 4, 4)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let brick_idx = wid.x;
    // Batched scatter: all entities' bricks are concatenated; the
    // dispatch launches exactly `total_bricks` workgroups so there is
    // no num_bricks bound to check.
    let info = brick_list[brick_idx];
    let skin = uniforms_arr[info.uniform_idx];

    // Cell within the 4³ brick.
    let cx = lid.x;
    let cy = lid.y;
    let cz = lid.z;
    let cell_flat = cx + cy * BRICK_DIM + cz * BRICK_DIM * BRICK_DIM;

    // Read brick cell; skip empty and interior sentinels.
    let pool_idx = info.brick_id * BRICK_CELLS + cell_flat;
    let slot = brick_pool[pool_idx];
    if slot == BRICK_EMPTY { return; }
    if slot == BRICK_INTERIOR { return; }

    // `slot` is the leaf_attr_id. Fetch this cell's bone influences.
    let bw_base = slot * 2u;
    let packed_indices = bone_weights[bw_base];
    let packed_weights = bone_weights[bw_base + 1u];
    if packed_weights == 0u { return; } // no skin influence on this voxel

    let rest_pos = (vec3<f32>(
        f32(info.origin_x + cx),
        f32(info.origin_y + cy),
        f32(info.origin_z + cz),
    ) + vec3<f32>(0.5)) * skin.voxel_size;

    let deformed_pos = forward_skin(rest_pos, packed_indices, packed_weights, skin.bone_buffer_offset);

    // Fetch the leaf's rest-pose normal and rotate it into deformed
    // space once. The march reads this pre-rotated value directly so
    // it never has to touch `rotate_rest_normal` at shade time.
    let rest_normal = unpack_oct_normal(leaf_attr_pool[slot].normal_oct);
    let deformed_normal = forward_rotate_normal(rest_normal, packed_indices, packed_weights, skin.bone_buffer_offset);
    let normal_packed = pack_oct_normal(deformed_normal);

    // Cell payload — (leaf_slot, deformed_normal_oct). The march
    // treats `leaf_slot == 0` as empty, matching the main-shader
    // convention. Real leaf slots start at `leaf_attr_slot_start`
    // for each asset, which is > 0 after any geometry has been
    // uploaded.
    let payload = vec2<u32>(slot, normal_packed);

    // Grid coord in the deformed bone field.
    let gp = deformed_pos - vec3<f32>(skin.grid_origin_x, skin.grid_origin_y, skin.grid_origin_z);
    let cont = gp / skin.voxel_size;
    let base = vec3<i32>(floor(cont));
    let dims = vec3<i32>(
        i32(skin.bone_field_dim_x),
        i32(skin.bone_field_dim_y),
        i32(skin.bone_field_dim_z),
    );

    // 2×2×2 splat. Every surface voxel writes to 8 cells regardless
    // of joint/rigid status — at shell resolution the scatter IS the
    // rendered surface, and gaps between sparse scatter land as
    // tears. Rkifield can afford a 1-cell scatter in rigid regions
    // because its march samples a continuous SDF between cells;
    // rkipatch has no such fallback.
    for (var oz = 0i; oz <= 1i; oz = oz + 1i) {
        for (var oy = 0i; oy <= 1i; oy = oy + 1i) {
            for (var ox = 0i; ox <= 1i; ox = ox + 1i) {
                let dx = base.x + ox;
                let dy = base.y + oy;
                let dz = base.z + oz;
                if dx < 0 || dy < 0 || dz < 0 { continue; }
                if dx >= dims.x || dy >= dims.y || dz >= dims.z { continue; }
                let ux = u32(dx);
                let uy = u32(dy);
                let uz = u32(dz);
                let cell = ux + uy * skin.bone_field_dim_x
                    + uz * skin.bone_field_dim_x * skin.bone_field_dim_y;
                bone_field[skin.bone_field_offset + cell] = payload;
            }
        }
    }

    // Flag every 4³-cell brick the splat touched as populated. Doing
    // this once per thread (instead of once per splat cell) collapses
    // 8 atomicOrs → 1–8, typically 1 — the 2×2×2 splat almost always
    // fits inside a single brick (only straddles when base aligns with
    // the last cell of a brick on some axis).
    let bx_dim = (skin.bone_field_dim_x + 3u) / 4u;
    let by_dim = (skin.bone_field_dim_y + 3u) / 4u;
    let bz_dim = (skin.bone_field_dim_z + 3u) / 4u;
    if base.x >= 0 && base.y >= 0 && base.z >= 0
        && base.x < dims.x && base.y < dims.y && base.z < dims.z
    {
        let bx_base = u32(base.x) >> 2u;
        let by_base = u32(base.y) >> 2u;
        let bz_base = u32(base.z) >> 2u;
        let straddle_x = select(0u, 1u, (u32(base.x) & 3u) == 3u && (base.x + 1) < dims.x);
        let straddle_y = select(0u, 1u, (u32(base.y) & 3u) == 3u && (base.y + 1) < dims.y);
        let straddle_z = select(0u, 1u, (u32(base.z) & 3u) == 3u && (base.z + 1) < dims.z);
        for (var dz = 0u; dz <= straddle_z; dz = dz + 1u) {
            for (var dy = 0u; dy <= straddle_y; dy = dy + 1u) {
                for (var dx = 0u; dx <= straddle_x; dx = dx + 1u) {
                    let bx = bx_base + dx;
                    let by = by_base + dy;
                    let bz = bz_base + dz;
                    if bx >= bx_dim || by >= by_dim || bz >= bz_dim { continue; }
                    let brick_idx = bx + by * bx_dim + bz * bx_dim * by_dim;
                    let occ_word = skin.bone_field_occ_offset + (brick_idx >> 5u);
                    let occ_bit = 1u << (brick_idx & 31u);
                    atomicOr(&bone_field_occ[occ_word], occ_bit);
                }
            }
        }
    }
}

// `unpack_oct_normal` — same as the one in `octree_march.wgsl`.
// Duplicated so the scatter shader can decode the leaf's stored
// rest-pose normal without dragging the whole march file in.
fn unpack_oct_normal(packed: u32) -> vec3<f32> {
    let ui_raw = i32(packed & 0xFFFFu);
    let vi_raw = i32((packed >> 16u) & 0xFFFFu);
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
