// user_shader_instance_march_main.wgsl — Stage 5b instance-march entry.
//
// Composes the Stage 5a helpers (defined in
// `user_shader_instance_march_helpers.wgsl`, concatenated ahead of this
// file by the Rust pipeline owner) into a per-ray march:
//
//   1. For each entry in the flat `tile_index_buffer`:
//      a. Look up the region's AABB (`regions_buffer[entry.region_index]`).
//      b. Slab-cull the world-space ray against the region AABB.
//      c. Look up the region's prototype in `proto_lookup_buffer` by
//         `region.shader_id`. Skip if not found.
//      d. For each instance slot in the region:
//         i.   Read the instance's `pos` (and `scale`, if present).
//         ii.  Compute the instance's world-space AABB.
//         iii. Slab-cull the ray against the instance AABB.
//         iv.  Transform the ray into prototype `[0, 1]³` canonical
//              space via `inst_world_to_local`.
//         v.   Call `inst_proto_descend` to find the first cell hit.
//         vi.  Track the closest hit so far (smallest world-space `t`).
//   2. Write the closest hit to `output_hits[ray_index]`.
//
// V1 simplifications (per the locked Stage 5 design — TRS-only, single
// ray dispatch):
//   - Linear iteration over the flat tile-index, NOT a per-pixel
//     TileIndex lookup.
//   - Uniform scale only. `scale_kind == PerAxis` falls back to the
//     `pos_offset_u32` field's first component as a uniform scalar (
//     authors warned via parser; per-axis honors come in a later stage).
//   - One workgroup per ray, one thread. The Rust pass dispatches as
//     `dispatch_workgroups(num_rays, 1, 1)`. Stage 6 will batch by
//     screen tile.
//
// ## Hit output
//
// Each ray writes one `InstanceMarchHit` — `hit == 1u` on a populated
// cell, with `t_world` (world-space ray parameter), `normal` (world-
// space-aligned via `inst_unpack_oct_normal` in canonical basis — V1
// keeps this in the prototype's basis since uniform scale doesn't
// rotate normals), `material_packed` (host-material × user-shader
// material blend later — V1 returns the leaf-attr's word as-is), and
// the source `region_index` + `instance_index` so the renderer can
// look up shading state. `hit == 0u` is "no instance occluded this ray
// in any tested region."

struct EmitRegionUniformLayout {
    aabb_min: vec3<f32>,
    cell_size: f32,
    aabb_max: vec3<f32>,
    shader_id: u32,
    time: f32,
    material_id: u32,
    region_thickness: f32,
    instance_block_offset: u32,
    instance_block_size: u32,
    instance_stride_u32: u32,
    host_octree_root: u32,
    host_octree_depth: u32,
    host_octree_extent: f32,
    _pad_host: u32,
    host_grid_origin: vec3<f32>,
    _pad_grid: f32,
    params: array<vec4<f32>, 2>,
    host_inverse_world: mat4x4<f32>,
}

struct GpuTileIndexEntryLayout {
    host_object_id: u32,
    material_id: u32,
    tile_x: i32,
    tile_y: i32,
    tile_z: i32,
    region_index: u32,
    _pad0: u32,
    _pad1: u32,
}

struct GpuPrototypeEntryLayout {
    shader_id: u32,
    octree_root: u32,
    max_depth: u32,
    instance_stride_u32: u32,
    pos_offset_u32: u32,
    scale_offset_u32: u32,
    scale_kind: u32,
    _pad0: u32,
}

const NO_SCALE_OFFSET: u32 = 0xFFFFFFFFu;
const SCALE_KIND_NONE: u32 = 0u;
const SCALE_KIND_UNIFORM: u32 = 1u;
const SCALE_KIND_PER_AXIS: u32 = 2u;

/// Camera-derived per-pixel ray, constructed inside the shader from
/// `MarchCameraUniform`. Held as a transient struct so the rest of the
/// march code (`instance_march_one_ray`) reads from a single value
/// rather than passing 4-5 args.
struct MarchRay {
    origin: vec3<f32>,
    max_steps_outer: u32,
    direction: vec3<f32>,
    max_steps_brick: u32,
}

/// Per-frame uniform — buffer sizes the shader needs to bound-check
/// + screen dimensions for per-pixel indexing + outer/brick step caps.
///
/// `march_max_steps_outer/brick` are uniform-driven so the host can
/// tune without recompiling the shader. Defaults pulled from the
/// Rust mirror.
struct MarchUniforms {
    tile_index_count: u32,
    proto_lookup_count: u32,
    screen_width: u32,
    screen_height: u32,
    march_max_steps_outer: u32,
    march_max_steps_brick: u32,
    _pad0: u32,
    _pad1: u32,
}

/// Camera state for per-pixel ray construction. Layout is the FIRST
/// 80 BYTES of `rkp_scene::CameraUniforms` — same field order, same
/// offsets, so Stage 6c can bind a slice of the existing camera
/// buffer without a translation step. The remaining
/// `layer_mask`/`focus_object_id`/`prev_vp`/`view_proj` fields aren't
/// needed by the V1 march and are deliberately omitted from this
/// uniform; binding them would inflate the min-binding-size for no
/// benefit.
struct MarchCameraUniform {
    position: vec4<f32>,
    forward: vec4<f32>,
    right: vec4<f32>,
    up: vec4<f32>,
    resolution: vec2<f32>,
    jitter: vec2<f32>,
}

struct InstanceMarchHit {
    hit: u32,
    region_index: u32,
    instance_index: u32,
    leaf_attr_slot: u32,
    t_world: f32,
    material_packed: u32,
    _pad0: u32,
    _pad1: u32,
    normal: vec3<f32>,
    _pad2: f32,
}

@group(1) @binding(0) var<storage, read> regions_buffer: array<EmitRegionUniformLayout>;
@group(1) @binding(1) var<storage, read> instance_pool: array<u32>;
@group(1) @binding(2) var<storage, read> tile_index_buffer: array<GpuTileIndexEntryLayout>;
@group(1) @binding(3) var<storage, read> instance_alloc: array<u32>;

@group(2) @binding(0) var<storage, read> proto_lookup_buffer: array<GpuPrototypeEntryLayout>;

@group(3) @binding(0) var<uniform> march_uniforms: MarchUniforms;
@group(3) @binding(1) var<uniform> camera: MarchCameraUniform;
@group(3) @binding(2) var<storage, read_write> output_hits: array<InstanceMarchHit>;

/// Linear scan `proto_lookup_buffer` for `shader_id`. Returns the entry
/// pointer (as a copied struct, since arrays-of-storage aren't address-
/// of-able in WGSL). `found == 0u` when missing.
struct ProtoLookupResult {
    found: u32,
    entry: GpuPrototypeEntryLayout,
}

fn find_prototype(shader_id: u32) -> ProtoLookupResult {
    var r: ProtoLookupResult;
    r.found = 0u;
    let count = march_uniforms.proto_lookup_count;
    for (var i: u32 = 0u; i < count; i = i + 1u) {
        let e = proto_lookup_buffer[i];
        if e.shader_id == shader_id {
            r.found = 1u;
            r.entry = e;
            return r;
        }
    }
    return r;
}

/// Read the instance's `pos: vec3<f32>` from `instance_pool` at the
/// given record base (in u32 units) + the prototype's `pos_offset_u32`.
fn read_instance_pos(base_u32: u32, pos_offset_u32: u32) -> vec3<f32> {
    let bx = bitcast<f32>(instance_pool[base_u32 + pos_offset_u32]);
    let by = bitcast<f32>(instance_pool[base_u32 + pos_offset_u32 + 1u]);
    let bz = bitcast<f32>(instance_pool[base_u32 + pos_offset_u32 + 2u]);
    return vec3<f32>(bx, by, bz);
}

/// Read the instance's scale per the prototype's `scale_kind`. Uniform
/// returns the f32 scalar; per-axis returns the first component as a
/// scalar fallback (V1 limitation); none returns 1.0.
fn read_instance_scale(
    base_u32: u32, scale_offset_u32: u32, scale_kind: u32,
) -> f32 {
    if scale_kind == SCALE_KIND_NONE || scale_offset_u32 == NO_SCALE_OFFSET {
        return 1.0;
    }
    if scale_kind == SCALE_KIND_UNIFORM {
        return bitcast<f32>(instance_pool[base_u32 + scale_offset_u32]);
    }
    // SCALE_KIND_PER_AXIS — V1 falls back to the first component.
    // Stage 5c (or later) will honor per-axis by passing a vec3<f32>
    // through `inst_world_to_local`.
    return bitcast<f32>(instance_pool[base_u32 + scale_offset_u32]);
}

/// March one ray against every instance in every region — V1's
/// per-pixel inner loop. Reusable from a future per-tile or per-pixel
/// outer that batches across many rays.
fn instance_march_one_ray(ray: MarchRay) -> InstanceMarchHit {
    var best: InstanceMarchHit;
    best.hit = 0u;
    best.region_index = 0u;
    best.instance_index = 0u;
    best.leaf_attr_slot = 0u;
    best.t_world = 1.0e30;
    best.material_packed = 0u;
    best.normal = vec3<f32>(0.0, 1.0, 0.0);

    // Pre-compute world-space inv_dir with the same 1e-10 nudge
    // convention as `octree_march.wgsl`.
    let safe_dir = vec3<f32>(
        select(ray.direction.x, select(-1e-10, 1e-10, ray.direction.x >= 0.0), abs(ray.direction.x) < 1e-10),
        select(ray.direction.y, select(-1e-10, 1e-10, ray.direction.y >= 0.0), abs(ray.direction.y) < 1e-10),
        select(ray.direction.z, select(-1e-10, 1e-10, ray.direction.z >= 0.0), abs(ray.direction.z) < 1e-10),
    );
    let inv_dir = 1.0 / safe_dir;

    let tile_count = march_uniforms.tile_index_count;
    for (var ti: u32 = 0u; ti < tile_count; ti = ti + 1u) {
        let tile_entry = tile_index_buffer[ti];
        let region_index = tile_entry.region_index;
        let region = regions_buffer[region_index];

        // Slab-cull the world-space ray against the region's AABB.
        let region_t = inst_ray_aabb_intersect(
            ray.origin, inv_dir, region.aabb_min, region.aabb_max,
        );
        if region_t.x > region_t.y { continue; }
        if region_t.x > best.t_world { continue; }

        // Find the prototype for this region's shader.
        let lookup = find_prototype(region.shader_id);
        if lookup.found == 0u { continue; }
        let proto = lookup.entry;

        // How many instances actually scattered into this region. The
        // emit pass writes the count at `instance_alloc[region_index]`;
        // capped to `region.instance_block_size` since the atomic-add
        // overflowed past that point and bumped the overflow counter.
        let written = min(
            instance_alloc[region_index],
            region.instance_block_size,
        );

        for (var ii: u32 = 0u; ii < written; ii = ii + 1u) {
            let base = region.instance_block_offset
                + ii * region.instance_stride_u32;

            let inst_pos = read_instance_pos(base, proto.pos_offset_u32);
            let inst_scale = read_instance_scale(
                base, proto.scale_offset_u32, proto.scale_kind,
            );
            let half = inst_scale * 0.5;
            let inst_aabb_min = inst_pos - vec3<f32>(half);
            let inst_aabb_max = inst_pos + vec3<f32>(half);

            let inst_t = inst_ray_aabb_intersect(
                ray.origin, inv_dir, inst_aabb_min, inst_aabb_max,
            );
            if inst_t.x > inst_t.y { continue; }
            if inst_t.x > best.t_world { continue; }

            // Transform ray to canonical space. Origin: world-space
            // start point at the AABB entry. Direction: world dir
            // scaled by `1/scale` (uniform).
            let world_entry = ray.origin + safe_dir * inst_t.x;
            let local_entry = inst_world_to_local(
                world_entry, inst_pos, inst_scale,
            );
            let inv_s = 1.0 / max(inst_scale, 1e-10);
            let local_dir = safe_dir * inv_s;

            let hit = inst_proto_descend(
                local_entry, local_dir,
                proto.octree_root, proto.max_depth,
                ray.max_steps_outer, ray.max_steps_brick,
            );

            if hit.hit == 0u { continue; }

            // Convert canonical-space `t` to world-space. The local
            // direction was scaled by `1/scale`, so canonical `t`
            // corresponds to a world-space distance of
            // `world_t = inst_t.x + hit.t * scale`.
            let world_t = inst_t.x + hit.t * inst_scale;
            if world_t >= best.t_world { continue; }

            best.hit = 1u;
            best.region_index = region_index;
            best.instance_index = ii;
            best.leaf_attr_slot = hit.leaf_attr_slot;
            best.t_world = world_t;
            best.material_packed = hit.material_local;
            best.normal = hit.normal;
        }
    }

    if best.hit == 0u {
        best.t_world = 0.0;
    }
    return best;
}

/// Construct the per-pixel world-space ray. Mirrors the convention in
/// `octree_march.wgsl`'s main entry point — same uv/ndc derivation —
/// so a Stage 6c renderer integration can rely on instance-march hits
/// being on identical pixel-space rays as the host march.
fn camera_pixel_ray(pixel_xy: vec2<u32>) -> MarchRay {
    let uv = (vec2<f32>(pixel_xy) + 0.5 + camera.jitter) / camera.resolution;
    let ndc = vec2<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    let dir = normalize(
        camera.forward.xyz
        + ndc.x * camera.right.xyz
        + ndc.y * camera.up.xyz
    );
    var ray: MarchRay;
    ray.origin = camera.position.xyz;
    ray.direction = dir;
    ray.max_steps_outer = march_uniforms.march_max_steps_outer;
    ray.max_steps_brick = march_uniforms.march_max_steps_brick;
    return ray;
}

@compute @workgroup_size(8, 8, 1)
fn instance_march_main(@builtin(global_invocation_id) pixel: vec3<u32>) {
    let w = march_uniforms.screen_width;
    let h = march_uniforms.screen_height;
    if pixel.x >= w || pixel.y >= h { return; }
    let ray = camera_pixel_ray(pixel.xy);
    let hit = instance_march_one_ray(ray);
    let out_index = pixel.x + pixel.y * w;
    output_hits[out_index] = hit;
}
