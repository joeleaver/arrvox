// user_shader_instance_march_helpers.wgsl — Stage 5a/5b WGSL helper library.
//
// Pure (mostly) functions composed by Stage 5a's test entries and
// Stage 5b's `instance_march_main`. The three "pure" helpers do no
// memory access:
//
// * `inst_ray_aabb_intersect(ro, inv_dir, aabb_min, aabb_max) -> vec2<f32>`
//   — slab method, returns `(t_near, t_far)`. Miss = `t_near > t_far`.
// * `inst_world_to_local(world_pos, instance_pos, instance_scale) -> vec3<f32>`
//   — transforms world-space → prototype canonical `[0, 1]³`. V1 = uniform
//   scale only (TRS-only locked decision).
//
// The descent helper does read the pool bindings declared below:
//
// * `inst_proto_descend(...)` — descends a prototype octree from
//   `(local_origin, local_dir)` and returns the first cell hit.
//
// ## Bindings owned here
//
// `octree_nodes` / `brick_pool` / `leaf_attr_pool` at @group(0)
// @binding(0/1/2). Both Stage 5a's test pipeline and Stage 5b's march
// pipeline bind THE SAME group(0) layout, so the helpers compile
// against either pipeline without modification.
//
// Pipeline-specific bindings (test inputs/outputs, march regions, etc.)
// live in the file the Rust composer concatenates after this one.

// ── Shared constants (mirror octree_march.wgsl + user_shader_proto.wgsl) ──

const OCTREE_EMPTY: u32 = 0xFFFFFFFFu;
const OCTREE_INTERIOR: u32 = 0xFFFFFFFEu;
const OCTREE_LEAF_BIT: u32 = 0x80000000u;
const OCTREE_BRICK_BIT: u32 = 0x40000000u;
const OCTREE_PAYLOAD_MASK: u32 = 0x3FFFFFFFu;

const BRICK_DIM: u32 = 4u;
const BRICK_DIM_F: f32 = 4.0;
const BRICK_CELLS: u32 = 64u;
const BRICK_CELL_EMPTY: u32 = 0xFFFFFFFFu;
const BRICK_CELL_INTERIOR: u32 = 0xFFFFFFFDu;

// ── Types ─────────────────────────────────────────────────────────────

struct LeafAttr {
    normal_oct: u32,
    material_packed: u32,
}

/// Result of a prototype descent. `hit == 1u` when a populated cell was
/// found, `0u` otherwise. On hit, `t` is the ray parameter at which the
/// ray entered the cell (in canonical-space units), `normal` is the
/// unpacked octahedral normal in canonical space, and `material_local`
/// is the leaf-attr's packed material word.
struct InstProtoHit {
    hit: u32,
    t: f32,
    normal: vec3<f32>,
    material_local: u32,
    leaf_attr_slot: u32,
}

// ── Pool bindings (shared between test + march pipelines) ─────────────

@group(0) @binding(0) var<storage, read> octree_nodes: array<vec2<u32>>;
@group(0) @binding(1) var<storage, read> brick_pool: array<u32>;
@group(0) @binding(2) var<storage, read> leaf_attr_pool: array<LeafAttr>;

// ── Pure helpers ─────────────────────────────────────────────────────

/// Slab-method ray-AABB intersection. Returns `(t_near, t_far)`. Miss
/// detection is `t_near > t_far` OR `t_far < 0` (whole interval behind
/// the origin); callers do whichever check fits their context.
///
/// `inv_dir = 1.0 / direction` is taken as a parameter so callers that
/// have already computed it (the march does, for skip_node) don't pay
/// twice. Direction components must be non-zero; the caller is
/// responsible for guarding against that with the same `select`-based
/// 1e-10 nudge used in `octree_march.wgsl::intersect_aabb`.
///
/// `t_near` is clamped to 0 — same convention as `intersect_aabb` in
/// octree_march.wgsl, so callers that march only forward from the
/// origin use the returned `t_near` directly.
fn inst_ray_aabb_intersect(
    ro: vec3<f32>, inv_dir: vec3<f32>,
    aabb_min: vec3<f32>, aabb_max: vec3<f32>,
) -> vec2<f32> {
    let t0 = (aabb_min - ro) * inv_dir;
    let t1 = (aabb_max - ro) * inv_dir;
    let tmin = min(t0, t1);
    let tmax = max(t0, t1);
    let t_near = max(max(max(tmin.x, tmin.y), tmin.z), 0.0);
    let t_far = min(min(tmax.x, tmax.y), tmax.z);
    return vec2<f32>(t_near, t_far);
}

/// Transform a world-space point into the prototype's canonical
/// `[0, 1]³` space. V1 = uniform scale only — `instance_scale` is the
/// scalar side length of the cube the instance occupies in world units.
///
/// The instance's world-space AABB is centered at `instance_pos` with
/// side `instance_scale`, so the canonical-to-world map is
/// `world = instance_pos + (canonical - 0.5) * instance_scale`. The
/// inverse — what we want — is
/// `canonical = (world - instance_pos) / instance_scale + 0.5`.
///
/// Returned points may fall outside `[0, 1]³` — the caller decides what
/// to do (reject as out-of-bounds, clamp, or proceed because the
/// canonical-space descent will hit OCTREE_EMPTY anyway).
fn inst_world_to_local(
    world_pos: vec3<f32>, instance_pos: vec3<f32>, instance_scale: f32,
) -> vec3<f32> {
    let inv_s = 1.0 / max(instance_scale, 1e-10);
    return (world_pos - instance_pos) * inv_s + vec3<f32>(0.5);
}

/// Descend a prototype octree from `pos` (in canonical `[0, 1]³`) and
/// resolve the terminator. Mirrors `octree_lookup` in `octree_march.wgsl`
/// but trimmed for the prototype case:
///
/// * No prefiltered-LOD (prototypes are tiny by construction; leaves are
///   reached in `max_depth` steps).
/// * No diagnostic `bucket_depth` calls.
/// * Operates in canonical extent = 1.0.
///
/// `octree_root` is the absolute pool index of the prototype's root
/// (level 0). `max_depth` is the prototype's depth. Caller passes the
/// shared `octree_nodes` storage buffer at module scope; this function
/// reads it directly.
///
/// Returns the terminating slot value tagged with the BRICK bit
/// preserved when the leaf is a brick reference. EMPTY/INTERIOR
/// terminators return their sentinels. Callers detect "is brick" via
/// the BRICK bit.
fn inst_octree_lookup(
    root: u32, max_depth: u32, pos: vec3<f32>,
) -> u32 {
    var offset = root;
    var half_extent: f32 = 0.5;
    var center: vec3<f32> = vec3<f32>(0.5);
    for (var level: u32 = 0u; level < max_depth; level = level + 1u) {
        let packed = octree_nodes[offset];
        let node = packed.x;
        if node == OCTREE_EMPTY { return OCTREE_EMPTY; }
        if node == OCTREE_INTERIOR { return OCTREE_INTERIOR; }
        if (node & OCTREE_LEAF_BIT) != 0u {
            return (node & OCTREE_PAYLOAD_MASK) | (node & OCTREE_BRICK_BIT);
        }
        let gt = vec3<u32>(pos >= center);
        offset = node + gt.x + gt.y * 2u + gt.z * 4u;
        half_extent = half_extent * 0.5;
        center = center + vec3<f32>(
            select(-half_extent, half_extent, pos.x >= center.x),
            select(-half_extent, half_extent, pos.y >= center.y),
            select(-half_extent, half_extent, pos.z >= center.z),
        );
    }
    let node = octree_nodes[offset].x;
    if node == OCTREE_EMPTY { return OCTREE_EMPTY; }
    if node == OCTREE_INTERIOR { return OCTREE_INTERIOR; }
    if (node & OCTREE_LEAF_BIT) != 0u {
        return (node & OCTREE_PAYLOAD_MASK) | (node & OCTREE_BRICK_BIT);
    }
    return OCTREE_EMPTY;
}

/// Detect a BRICK terminator from `inst_octree_lookup`.
fn inst_slot_is_brick(slot: u32) -> bool {
    return (slot & OCTREE_BRICK_BIT) != 0u
        && slot != OCTREE_EMPTY
        && slot != OCTREE_INTERIOR;
}

fn inst_slot_brick_id(slot: u32) -> u32 {
    return slot & OCTREE_PAYLOAD_MASK;
}

/// Decode a packed 2× snorm16 octahedral normal. Mirror of
/// `unpack_oct_normal` in octree_march.wgsl. Kept inline so this
/// helper file stands alone.
fn inst_unpack_oct_normal(packed: u32) -> vec3<f32> {
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

/// Walk the prototype octree along the canonical-space ray
/// `(local_origin, local_dir)` and return the first cell hit. Steps via
/// DDA at the brick level: descend octree → on brick, walk cells → on
/// empty, skip past the brick / node.
///
/// This is the V1 prototype descent — no glass, no LOD, no neighbour
/// reconstruction. It mirrors the rigid path of `march_object` in
/// octree_march.wgsl but operates in fixed `[0,1]³` extent so the
/// per-instance setup is constant.
///
/// `max_steps_outer` and `max_steps_brick` cap the outer + inner DDA
/// loops respectively. The instance-march caller picks small numbers
/// (prototypes are tiny) — typically 256 / 64.
fn inst_proto_descend(
    local_origin: vec3<f32>, local_dir: vec3<f32>,
    octree_root: u32, max_depth: u32,
    max_steps_outer: u32, max_steps_brick: u32,
) -> InstProtoHit {
    var hit: InstProtoHit;
    hit.hit = 0u;
    hit.t = 0.0;
    hit.normal = vec3<f32>(0.0, 1.0, 0.0);
    hit.material_local = 0u;
    hit.leaf_attr_slot = 0u;

    let safe_dir = vec3<f32>(
        select(local_dir.x, select(-1e-10, 1e-10, local_dir.x >= 0.0), abs(local_dir.x) < 1e-10),
        select(local_dir.y, select(-1e-10, 1e-10, local_dir.y >= 0.0), abs(local_dir.y) < 1e-10),
        select(local_dir.z, select(-1e-10, 1e-10, local_dir.z >= 0.0), abs(local_dir.z) < 1e-10),
    );
    let inv_dir = 1.0 / safe_dir;

    let t_range = inst_ray_aabb_intersect(
        local_origin, inv_dir, vec3<f32>(0.0), vec3<f32>(1.0),
    );
    if t_range.x > t_range.y { return hit; }

    let extent: f32 = 1.0;
    let cells_per_axis_f = f32(BRICK_DIM << max_depth);
    let vs = 1.0 / cells_per_axis_f;
    let lookup_bias = vs * 1.0e-3;

    var t = t_range.x;

    for (var step: u32 = 0u; step < max_steps_outer; step = step + 1u) {
        if t > t_range.y { break; }

        let pos = clamp(
            local_origin + safe_dir * (t + lookup_bias),
            vec3<f32>(vs * 0.01),
            vec3<f32>(extent - vs * 0.01),
        );
        let slot = inst_octree_lookup(octree_root, max_depth, pos);

        if slot == OCTREE_EMPTY {
            // Skip past the leaf-level cell containing pos. We don't
            // know the precise depth `inst_octree_lookup` terminated at
            // (it could have hit an EMPTY internal-level node). For V1,
            // step by a leaf-level cell — correctness over speed; Stage
            // 5b can tighten this with depth tracking.
            t = t + vs;
            continue;
        }
        if slot == OCTREE_INTERIOR {
            // Prototypes don't currently emit INTERIOR, but treat the
            // same as a hit on the bounds for safety: report hit at
            // current t with an arbitrary +Y normal.
            hit.hit = 1u;
            hit.t = t;
            hit.normal = vec3<f32>(0.0, 1.0, 0.0);
            hit.material_local = 0u;
            hit.leaf_attr_slot = 0u;
            return hit;
        }

        if inst_slot_is_brick(slot) {
            let brick_id = inst_slot_brick_id(slot);
            // Locate the brick in canonical space via the leaf-level
            // cell that contains `pos`. Each brick covers (1/2^max_depth)
            // along each axis; the cell at the leaf level is at
            // `floor(pos * 2^max_depth) / 2^max_depth`.
            let bricks_per_axis_f = f32(1u << max_depth);
            let brick_size = 1.0 / bricks_per_axis_f;
            let brick_xyz = floor(pos * bricks_per_axis_f);
            let brick_origin = brick_xyz * brick_size;
            let cell_size = brick_size / BRICK_DIM_F;
            let inv_cell_size = 1.0 / cell_size;

            let p0 = local_origin + safe_dir * t;
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

            for (var bs: u32 = 0u; bs < max_steps_brick; bs = bs + 1u) {
                if t > t_range.y { break; }
                if cell.x < 0 || cell.x >= 4
                    || cell.y < 0 || cell.y >= 4
                    || cell.z < 0 || cell.z >= 4 {
                    // Exited the brick — fall back to outer loop. V1
                    // doesn't chain across bricks (no face_links bound
                    // for prototype pool). The outer loop's octree
                    // descent picks up the next brick.
                    break;
                }
                let cell_idx = u32(cell.x) + u32(cell.y) * BRICK_DIM
                    + u32(cell.z) * BRICK_DIM * BRICK_DIM;
                let cell_value = brick_pool[brick_id * BRICK_CELLS + cell_idx];

                if cell_value != BRICK_CELL_EMPTY && cell_value != BRICK_CELL_INTERIOR {
                    // Hit a populated cell. `cell_value` is the
                    // leaf-attr slot.
                    let attr = leaf_attr_pool[cell_value];
                    hit.hit = 1u;
                    hit.t = t;
                    hit.normal = inst_unpack_oct_normal(attr.normal_oct);
                    hit.material_local = attr.material_packed;
                    hit.leaf_attr_slot = cell_value;
                    return hit;
                }

                // DDA step.
                let mn = min(t_max.x, min(t_max.y, t_max.z));
                t = mn + dda_eps;
                if mn == t_max.x {
                    cell.x = cell.x + step_i.x;
                    t_max.x = t_max.x + t_delta.x;
                } else if mn == t_max.y {
                    cell.y = cell.y + step_i.y;
                    t_max.y = t_max.y + t_delta.y;
                } else {
                    cell.z = cell.z + step_i.z;
                    t_max.z = t_max.z + t_delta.z;
                }
            }
            // Fall through — outer loop re-descends at advanced `t`.
            continue;
        }

        // Non-brick LEAF (leaf without brick bit) — V1 doesn't expect
        // these on prototypes (the bake always emits with BRICK_BIT
        // set). Treat as no hit and step.
        t = t + vs;
    }
    return hit;
}
