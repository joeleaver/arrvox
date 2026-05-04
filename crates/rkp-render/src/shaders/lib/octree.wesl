import package::lib::types::OctreeResult;
import package::lib::octree_slot::OCTREE_EMPTY;
import package::lib::octree_slot::OCTREE_INTERIOR;
import package::lib::octree_slot::OCTREE_LEAF_BIT;
import package::lib::octree_slot::OCTREE_BRICK_BIT;
import package::lib::octree_slot::OCTREE_BAND_BIT;
import package::lib::octree_slot::OCTREE_PAYLOAD_MASK;
import package::lib::brick::INTERNAL_ATTR_NONE;

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

// Compute the t-distance to the next octree node boundary along a
// ray. `vs` is the voxel size (smallest cell dimension); the
// `+ vs * 0.01` epsilon kicks the ray just past the boundary so the
// next iteration steps into the new node rather than re-hitting the
// boundary.
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
fn octree_lookup(
    root: u32,
    max_depth: u32,
    extent: f32,
    pos: vec3<f32>,
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
        if (node & OCTREE_LEAF_BIT) != 0u {
            bucket_depth(phase, level);
            // Preserve BRICK_BIT and BAND_BIT in the returned slot so
            // the caller can distinguish a leaf-attr leaf from a brick
            // leaf or a band cell. Band cells trigger
            // `dispatch_user_instance_descend` for proper instance-
            // shader rendering.
            return OctreeResult(
                (node & OCTREE_PAYLOAD_MASK) | (node & OCTREE_BRICK_BIT) | (node & OCTREE_BAND_BIT),
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
    let node = octree_nodes[offset].x;
    if node == OCTREE_EMPTY { return OctreeResult(OCTREE_EMPTY, max_depth, center, half); }
    if node == OCTREE_INTERIOR { return OctreeResult(OCTREE_INTERIOR, max_depth, center, half); }
    if (node & OCTREE_LEAF_BIT) != 0u {
        return OctreeResult(
            (node & OCTREE_PAYLOAD_MASK) | (node & OCTREE_BRICK_BIT) | (node & OCTREE_BAND_BIT),
            max_depth, center, half,
        );
    }
    return OctreeResult(OCTREE_EMPTY, max_depth, center, half);
}
