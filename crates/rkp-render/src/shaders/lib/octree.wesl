import package::lib::types::OctreeResult;
import package::lib::types::ProtoHit;
import package::lib::types::RkpAsset;
import package::lib::types::LeafAttr;
import package::lib::math::intersect_aabb;
import package::lib::oct_normal::unpack_oct_normal;
import package::lib::octree_slot::OCTREE_EMPTY;
import package::lib::octree_slot::OCTREE_INTERIOR;
import package::lib::octree_slot::OCTREE_LEAF_BIT;
import package::lib::octree_slot::OCTREE_BRICK_BIT;
import package::lib::octree_slot::OCTREE_PAYLOAD_MASK;
import package::lib::octree_slot::slot_is_brick;
import package::lib::octree_slot::slot_brick_id;
import package::lib::brick::INTERNAL_ATTR_NONE;
import package::lib::brick::BRICK_DIM;
import package::lib::brick::BRICK_DIM_F;
import package::lib::brick::BRICK_CELLS;
import package::lib::brick::BRICK_CELL_EMPTY;
import package::lib::brick::BRICK_CELL_INTERIOR;
import package::lib::brick::BRICK_MAX_STEPS;
import package::lib::brick::FACE_INTERIOR;
import package::lib::brick::FACE_EMPTY_LINK;
import package::lib::brick::FACE_NX;
import package::lib::brick::FACE_PX;
import package::lib::brick::FACE_NY;
import package::lib::brick::FACE_PY;
import package::lib::brick::FACE_NZ;
import package::lib::brick::FACE_PZ;

// Octree-traversal helpers shared across the host march and shadow
// passes.
//
// `octree_lookup` calls two stats helpers — `bucket_depth(phase, level)`
// and `bucket_lod_exit(level)` — by free name. Each consumer file
// declares its own version (real if it has a `stats` storage binding,
// no-op stubs otherwise). The lib fn also reads `octree_nodes` as a
// free name; every consumer binds it as `array<vec2<u32>>` at the
// same group/binding so the read is consistent.
//
// `dispatch_user_instance_descend` stays per-consumer — it's the
// splice target whose declaration `splice_const_marker` replaces.
//
// `descend_proto_octree` stays per-consumer until its three sites
// converge on a single `octree_lookup` call shape (in progress;
// see Wave AC notes in project_wesl_phase_2_endpoint).
//
// `octree_nodes` is `array<vec4<u32>>` (16 B per node): `.x` = node
// value, `.y` = prefiltered-LOD attr, `.z` / `.w` = quantized tight
// occupancy AABB lo / hi (zeroed during Step 1 of the per-node
// tight-bounds rollout).

// March/shadow phase ids — used as a stats-bucket selector and as a
// gate for the LOD branch (only PHASE_MARCH triggers prefiltered-LOD
// early-exit). Values are stable wire format with the CPU stats
// reader; do not renumber.
const PHASE_MARCH: u32 = 0u;
const PHASE_SHADOW: u32 = 2u;

// Prefiltered-LOD cutoff in pixels. A descent terminates early when
// the current node's projected screen footprint (in vertical pixels)
// drops below this threshold and a prefiltered attr id is available
// at `octree_nodes[offset].y`. 2.0 px is a perceptual sweet spot
// (well below Nyquist for the LUT material lookups).
const LOD_CUTOFF_PX: f32 = 2.0;

// Iteration cap for `descend_proto_octree`'s outer DDA. Proto octrees
// are tiny (1-3 levels), so 4096 covers any plausible walk; bump
// when geometry-shader prototypes grow extents that need it.
const MAX_STEPS_PROTO: u32 = 4096u;

// Voxel-size-relative kick applied past a node-exit plane in
// `skip_node` so the next iteration steps into the new node rather
// than re-hitting the boundary. 1% of voxel size is well above
// FP-rounding noise at metre scale and well below the next interior
// feature, so it can't accidentally skip a thin shell.
const OCTREE_SKIP_EPSILON_FACTOR: f32 = 0.01;

// Voxel-size-relative bias applied to a ray's lookup position before
// `octree_lookup` to disambiguate ties at exact brick boundaries.
// Same scale rationale as `OCTREE_SKIP_EPSILON_FACTOR` but an order
// of magnitude smaller — the lookup only needs to break exact ties,
// not push past a boundary the DDA already crossed.
const OCTREE_LOOKUP_BIAS_FACTOR: f32 = 1.0e-3;

// Slab-test denominator floor for `skip_node`. Floats below this are
// numerically indistinguishable from zero at metre scale; clamping
// keeps the per-component reciprocal `1/d` finite.
const OCTREE_SLAB_EPS: f32 = 1.0e-6;

// Compute the t-distance to the next octree node boundary along a
// ray. `vs` is the voxel size (smallest cell dimension); the
// `+ vs * OCTREE_SKIP_EPSILON_FACTOR` kick keeps the next iteration
// inside the new node.
fn skip_node(pos: vec3<f32>, dir: vec3<f32>, inv_dir: vec3<f32>, node_depth: u32, extent: f32, vs: f32) -> f32 {
    let node_size = extent / f32(1u << node_depth);
    let node_min = floor(pos / node_size) * node_size;
    let node_max = node_min + node_size;
    // Exit planes: for positive dir, exit through max; for negative, through min.
    let t_exit = select((node_min - pos) * inv_dir, (node_max - pos) * inv_dir, dir > vec3<f32>(0.0));
    // Smallest positive exit = nearest boundary crossing.
    let t_pos = max(t_exit, vec3<f32>(OCTREE_SLAB_EPS));
    return min(min(t_pos.x, t_pos.y), t_pos.z) + vs * OCTREE_SKIP_EPSILON_FACTOR;
}

// Descend the octree from its root and return the terminating slot +
// depth + cell-bounds. Unified across the camera march and the two
// shadow paths — each caller controls per-pass behaviour through
// `phase` (the stats bucket) and `lod_enabled` (gates the LOD branch
// entirely; non-march callers pass 0).
//
// The LOD branch terminates early when the current node's projected
// screen footprint drops below `LOD_CUTOFF_PX` AND a prefiltered
// `leaf_attr_id` is available at `octree_nodes[offset].y`. It only
// fires when `lod_enabled != 0` AND `phase == PHASE_MARCH`; the
// shadow paths use a different (cone-footprint) LOD model so they
// must not trigger pixel-footprint LOD here. Non-march callers can
// pass `(0u, 0.0, 1.0, 0.0)` for the LOD args — values are unused
// when `lod_enabled == 0`.
//
// Stats: every empty/interior/leaf return calls `bucket_depth` so the
// CPU-side stats reader can build per-phase depth histograms. The
// LOD-exit return additionally calls `bucket_lod_exit`. Consumers
// without a `stats` storage binding declare these as no-op stubs.
//
// `ray_origin` and `inv_dir` carry the marching ray. At each descent
// level we test the ray against the node's tight occupancy AABB
// decoded from `packed.zw`; when the ray misses, we return EMPTY for
// that level so the outer DDA's `skip_node` advances past the cell
// without descending into a brick whose actual content the ray
// doesn't intersect. Callers without a meaningful ray (none today,
// but pass `(vec3(0), vec3(1))` to disable the test — every node's
// tight AABB will pass).
fn octree_lookup(
    root: u32,
    max_depth: u32,
    extent: f32,
    pos: vec3<f32>,
    ray_origin: vec3<f32>,
    inv_dir: vec3<f32>,
    phase: u32,
    lod_enabled: u32,
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
        // Per-node tight-bounds ray cull. Skip on `(z, w) == (0, 0)`
        // (the Step-1 sentinel meaning "tight bounds not set"; pre-
        // bake state). Otherwise decode the quantized bounds (8 bits
        // per axis) from normalized cell-fractions to oc-space, and
        // ray-AABB-test. Miss → treat as EMPTY at this level so the
        // outer DDA's `skip_node` advances past the cell.
        if (packed.z != 0u || packed.w != 0u) {
            let cell_min = center - vec3<f32>(half);
            let cell_size = half * 2.0;
            let lo_norm = vec3<f32>(
                f32(packed.z & 0xFFu),
                f32((packed.z >> 8u) & 0xFFu),
                f32((packed.z >> 16u) & 0xFFu),
            ) * (1.0 / 255.0);
            let hi_norm = vec3<f32>(
                f32(packed.w & 0xFFu),
                f32((packed.w >> 8u) & 0xFFu),
                f32((packed.w >> 16u) & 0xFFu),
            ) * (1.0 / 255.0);
            let tight_min = cell_min + lo_norm * cell_size;
            let tight_max = cell_min + hi_norm * cell_size;
            let t_range = intersect_aabb(ray_origin, inv_dir, tight_min, tight_max);
            if t_range.x > t_range.y {
                bucket_depth(phase, level);
                return OctreeResult(OCTREE_EMPTY, level, center, half);
            }
        }
        if (node & OCTREE_LEAF_BIT) != 0u {
            bucket_depth(phase, level);
            // Preserve BRICK_BIT in the returned slot so the caller
            // can distinguish a leaf-attr leaf from a brick leaf.
            return OctreeResult(
                (node & OCTREE_PAYLOAD_MASK) | (node & OCTREE_BRICK_BIT),
                level, center, half,
            );
        }

        // Prefiltered-LOD early-exit. See `LOD_CUTOFF_PX` doc for the
        // full math; in short: terminate when the node would occupy
        // less than ~2 vertical pixels on screen.
        if lod_enabled != 0u
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
    let leaf_packed = octree_nodes[offset];
    let node = leaf_packed.x;
    if node == OCTREE_EMPTY { return OctreeResult(OCTREE_EMPTY, max_depth, center, half); }
    if node == OCTREE_INTERIOR { return OctreeResult(OCTREE_INTERIOR, max_depth, center, half); }
    // Leaf-level tight-bounds cull. Same encoding as the descent-
    // level test above.
    if (leaf_packed.z != 0u || leaf_packed.w != 0u) {
        let cell_min = center - vec3<f32>(half);
        let cell_size = half * 2.0;
        let lo_norm = vec3<f32>(
            f32(leaf_packed.z & 0xFFu),
            f32((leaf_packed.z >> 8u) & 0xFFu),
            f32((leaf_packed.z >> 16u) & 0xFFu),
        ) * (1.0 / 255.0);
        let hi_norm = vec3<f32>(
            f32(leaf_packed.w & 0xFFu),
            f32((leaf_packed.w >> 8u) & 0xFFu),
            f32((leaf_packed.w >> 16u) & 0xFFu),
        ) * (1.0 / 255.0);
        let tight_min = cell_min + lo_norm * cell_size;
        let tight_max = cell_min + hi_norm * cell_size;
        let t_range = intersect_aabb(ray_origin, inv_dir, tight_min, tight_max);
        if t_range.x > t_range.y {
            return OctreeResult(OCTREE_EMPTY, max_depth, center, half);
        }
    }
    if (node & OCTREE_LEAF_BIT) != 0u {
        return OctreeResult(
            (node & OCTREE_PAYLOAD_MASK) | (node & OCTREE_BRICK_BIT),
            max_depth, center, half,
        );
    }
    return OctreeResult(OCTREE_EMPTY, max_depth, center, half);
}

// Walk a prototype octree from `oc_origin` along `safe_dir` and
// return the first opaque hit. Used by `dispatch_user_instance_descend`
// to march per-blade prototypes (grass, foliage, etc.) when a
// host-cell hit reveals a band-cell pointing at an instance shader.
//
// The phase used for stats bucketing comes in as `phase` — callers
// pass `PHASE_MARCH` for octree_march and `PHASE_SHADOW` for the
// shadow paths. Proto octrees are tiny (1-3 levels deep), so the
// LOD branch in `octree_lookup` won't fire — `lod_enabled` is
// hardcoded 0u and the LOD-arg slots receive zeros.
//
// Reads `brick_pool`, `leaf_attr_pool`, `brick_face_links` as free
// names (every consumer binds them at the same group/binding).
//
// Splice contract — `shader_composer/compose.rs` emits text that
// calls this with eight args (asset, oc_origin, safe_dir, inv_dir,
// t_start, t_end, local_to_world, PROTO_PHASE). The splice references
// `PROTO_PHASE` as a free name; each consumer file declares it with
// the appropriate `PHASE_MARCH` / `PHASE_SHADOW` value. Don't change
// the signature without updating compose.rs in lockstep.
fn descend_proto_octree(
    asset: RkpAsset,
    oc_origin: vec3<f32>,
    safe_dir: vec3<f32>,
    inv_dir: vec3<f32>,
    t_start: f32,
    t_end: f32,
    local_to_world: f32,
    phase: u32,
) -> ProtoHit {
    var r: ProtoHit;
    r.valid = false;
    r.local_t = t_end;
    r.oc_pos = vec3<f32>(0.0);
    r.local_normal = vec3<f32>(0.0, 1.0, 0.0);
    r.leaf_slot = 0u;

    let extent = bitcast<f32>(asset.octree_extent_bits);
    let vs = asset.voxel_size;
    let root = asset.octree_root;
    let max_depth = asset.octree_depth;
    let lookup_bias = vs * 1.0e-3;

    var t = t_start;
    for (var step = 0u; step < MAX_STEPS_PROTO; step++) {
        if t > t_end { break; }

        let pos = clamp(
            oc_origin + safe_dir * (t + lookup_bias),
            vec3<f32>(vs * 0.01),
            vec3<f32>(extent - vs * 0.01),
        );
        let lookup = octree_lookup(
            root, max_depth, extent, pos,
            oc_origin, inv_dir,
            phase, 0u, 0.0, 1.0, 0.0,
        );

        if lookup.slot == OCTREE_EMPTY {
            t += skip_node(pos, safe_dir, inv_dir, lookup.depth, extent, vs);
            continue;
        }
        if lookup.slot == OCTREE_INTERIOR {
            // Proto bakes shouldn't have interior bulk (thin shell only),
            // but if one shows up treat it as opaque-with-flat-normal.
            r.valid = true;
            r.local_t = t;
            r.oc_pos = oc_origin + safe_dir * t;
            r.local_normal = -safe_dir;
            r.leaf_slot = 0u;
            return r;
        }

        if slot_is_brick(lookup.slot) {
            var brick_id = slot_brick_id(lookup.slot);
            let cell_size = (lookup.cell_half * 2.0) / BRICK_DIM_F;
            let inv_cell_size = 1.0 / cell_size;
            var brick_origin = lookup.cell_center - vec3<f32>(lookup.cell_half);
            var brick_base = brick_id * BRICK_CELLS;

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

            for (var bs = 0u; bs < BRICK_MAX_STEPS; bs++) {
                if t > t_end { break; }

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
                        r.valid = true;
                        r.local_t = t;
                        r.oc_pos = oc_origin + safe_dir * t;
                        r.local_normal = -safe_dir;
                        r.leaf_slot = 0u;
                        return r;
                    }
                    if link == FACE_EMPTY_LINK { break; }
                    brick_id = link;
                    brick_base = link * BRICK_CELLS;
                    let brick_extent = BRICK_DIM_F * cell_size;
                    if face_idx == FACE_NX { cell.x = 3; brick_origin.x -= brick_extent; }
                    else if face_idx == FACE_PX { cell.x = 0; brick_origin.x += brick_extent; }
                    else if face_idx == FACE_NY { cell.y = 3; brick_origin.y -= brick_extent; }
                    else if face_idx == FACE_PY { cell.y = 0; brick_origin.y += brick_extent; }
                    else if face_idx == FACE_NZ { cell.z = 3; brick_origin.z -= brick_extent; }
                    else { cell.z = 0; brick_origin.z += brick_extent; }
                    let p_now = oc_origin + safe_dir * t;
                    let next_b2 = brick_origin + (vec3<f32>(cell) + step_gt) * cell_size;
                    t_max = t + (next_b2 - p_now) * inv_dir;
                }

                let cx = u32(cell.x);
                let cy = u32(cell.y);
                let cz = u32(cell.z);
                let flat = cx + cy * BRICK_DIM + cz * BRICK_DIM * BRICK_DIM;
                let c = brick_pool[brick_base + flat];

                if c != BRICK_CELL_EMPTY && c != BRICK_CELL_INTERIOR {
                    let attr = leaf_attr_pool[c];
                    r.valid = true;
                    r.local_t = t;
                    r.oc_pos = oc_origin + safe_dir * t;
                    r.local_normal = unpack_oct_normal(attr.normal_oct);
                    r.leaf_slot = c;
                    return r;
                }

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
            // Brick walk exhausted without a hit — fall through to outer
            // loop's skip_node so we don't infinite-loop on the outer step.
            t += vs;
            continue;
        }

        // Plain leaf — single-voxel slot.
        let leaf_slot = lookup.slot & OCTREE_PAYLOAD_MASK;
        let attr = leaf_attr_pool[leaf_slot];
        r.valid = true;
        r.local_t = t;
        r.oc_pos = oc_origin + safe_dir * t;
        r.local_normal = unpack_oct_normal(attr.normal_oct);
        r.leaf_slot = leaf_slot;
        return r;
    }
    return r;
}
