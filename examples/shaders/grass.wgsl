// grass.wgsl — Option B (instance pipeline) demo shader.
//
// Drop this into your project's `assets/shaders/grass.wgsl`. Then
// create a material whose `shader` field is `"grass"` and paint that
// material onto an entity's surface. The engine's per-frame
// `tick_instance_pipeline` (rkp-engine/src/render_worker.rs) detects
// the painted material, bakes the blade prototype, scatters blades on
// every painted leaf, marches them per-pixel, and composites into the
// merged G-buffer that shade reads from.
//
// Locked Option B authoring API: WGSL-only, `// @instance_proto
// <Struct>` directive + struct decl + the four hooks (proto / emit /
// optional inst_aabb / optional inst_to_local). With the inst_to_local
// hook wired through the march, the V1 transform contract is AFFINE:
// translate + rotation + uniform scale + linear skew. Grass uses all
// four: position, yaw rotation, blade-height scale, and a per-blade
// lean that linearly displaces the tip. Per-blade width scaling rides
// on the same hook (affine X-axis dilation around the blade's
// centerline).

// ── Region directives ───────────────────────────────────────────────
// max_depth 5 → 128 canonical cells per axis. With tile_size 5.12 m
// (= 0.04 × 4 × 2^5), each painted tile keeps its cells at the
// intended ~4 cm grain regardless of how big a patch the user
// paints — paint splits into multiple regions instead of one giant
// region whose cell size grows with paint extent.

// @max_depth 5
// @tile_size 5.12
// @region_thickness 1.5
// @animated
// `density` can request up to 5 blades per host position; the
// instance_at hook short-circuits to `false` once k exceeds the
// density-driven count.
// @max_emits_per_thread 5

// ── Per-material params ─────────────────────────────────────────────

// @param blade_height:  f32 = 0.35, range = [0.05, 1.5]
// @param blade_width:   f32 = 1.0,  range = [0.2, 3.0]
// @param height_jitter: f32 = 0.25, range = [0.0, 0.8]
// @param density:       f32 = 1.0,  range = [0.01, 4.0]
// @param pos_jitter:    f32 = 0.5,  range = [0.0, 1.0]
// @param lean_amount:   f32 = 0.15, range = [0.0, 0.6]
// @param wind_amp:      f32 = 0.08, range = [0.0, 0.3]
// @param wind_freq:     f32 = 1.5,  range = [0.0, 6.0]

// ── Per-instance state struct ───────────────────────────────────────
// Tagged-field discovery (see `crates/rkp-render/src/instance_proto.rs`):
//   * `pos: vec3<f32>` — required. Center of the blade's AABB cube.
//   * `scale: f32` — uniform world-space side of the AABB. Drives blade
//     height (with jitter); also the proto's overall extent.
//   * `yaw: f32` — Y-axis rotation in radians, per-blade random.
//   * `width: f32` — per-blade canonical-X dilation. >1 widens, <1
//     narrows; applied around the blade centerline (canonical x = 0.5)
//     in `inst_to_local`. Materializes the `@param blade_width` slider.
//   * `lean: vec2<f32>` — per-blade tip lean in canonical units, post-
//     yaw. lean ∈ [-0.5, 0.5] is roughly "tip moves up to 0.5 × scale m
//     horizontally." Animated each frame for wind sway.
// Total: 32 B (`width` packs into the natural pad before vec2's 8 B
// alignment, so the struct stays at the soft cap).

// @instance_proto Blade
struct Blade {
    pos: vec3<f32>,
    scale: f32,
    yaw: f32,
    width: f32,
    lean: vec2<f32>,
}

// ── Helpers ─────────────────────────────────────────────────────────
fn grass_hash_u01(seed: u32) -> f32 {
    var x = seed;
    x = x ^ (x >> 16u);
    x = x * 0x7feb352du;
    x = x ^ (x >> 15u);
    x = x * 0x846ca68bu;
    x = x ^ (x >> 16u);
    return f32(x) / 4294967295.0;
}

fn grass_seed_from_pos(p: vec3<f32>) -> u32 {
    return bitcast<u32>(p.x) ^ (bitcast<u32>(p.y) * 0x9E3779B9u)
        ^ (bitcast<u32>(p.z) * 0x85EBCA6Bu);
}

// ── Prototype hook ──────────────────────────────────────────────────
// Bakes the static blade silhouette in canonical [0,1]³. Real grass-
// blade dimensions: ~3-5 mm wide at base, tapering to a point, ~0.5 mm
// thick. At `blade_height = 0.4 m` (the canonical 1-unit), 3 mm = 0.0075
// canonical, so half-width 0.025 covers a ~20 mm-wide blade tapering to
// a point. Thickness `half_thick_z = 0.008` is just above the cell size
// at depth 5, so the blade is one cell deep — properly thin.
//
// Bake-time `bake_curve` leans the tip in +X canonical. Per-instance
// `lean` adds linear skew on top via inst_to_local; per-instance `width`
// dilates X around the centerline.
//
// Resolution floor for the tip taper: at @max_depth 5 the canonical
// cell size is 1/128 ≈ 0.0078, so a half-width below ~half a cell
// (0.0039) lives between cell centers and produces speckly gaps as
// the blade tapers. Clipping the blade where `raw_half_width` drops
// below `min_half_width` makes the silhouette end cleanly above the
// resolution floor — short of an actual point, but visually clean
// (vs. a noisy stippled tip). For higher @max_depth, lower this
// constant proportionally.
fn user_grass_proto(uvw: vec3<f32>) -> VoxelEmit {
    var v: VoxelEmit;
    v.occupancy = 0u;

    let bake_curve = 0.10;
    let curve_x = bake_curve * uvw.y * uvw.y;

    // Tapered ribbon: base half-width 0.025, linear taper toward 0;
    // clipped at the resolution-limited tip rather than letting it
    // disintegrate.
    let raw_half_width = 0.025 * (1.0 - uvw.y);
    let min_half_width = 0.005;
    if (raw_half_width < min_half_width) {
        return v;
    }
    let half_width_x = raw_half_width;
    let half_thick_z = 0.008;

    let dx = uvw.x - 0.5 - curve_x;
    let dz = uvw.z - 0.5;

    if (uvw.y < 1.0 && abs(dx) < half_width_x && abs(dz) < half_thick_z) {
        v.occupancy = 1u;
        // Outward-radial normal in the XZ plane with a small +Y tilt
        // — gives the shade pass a top-vs-side cue. Length is normalized
        // by the bake's pack_oct so magnitude is irrelevant here.
        v.normal = normalize(vec3<f32>(dx, 0.35, dz));
        v.material_primary = 0u;
        v.material_secondary = 0u;
        v.blend_weight = 0u;
    }
    return v;
}

// ── inst_world_matrix hook ──────────────────────────────────────────
// Forward affine: canonical [0, 1]³ → world. Returned matrix is
// column-major (WGSL convention). The new emit pass writes this
// directly into `RkpInstance.world` for each emitted blade so the
// host march can descend the proto via its standard `inv_world × ray`
// flow.
//
// Composition (canonical → world):
//   1. Apply width: `pre_lean.x = 0.5 + width × (canonical.x - 0.5)`
//   2. Apply linear lean: `unscaled.xz = pre_lean.xz + lean × canonical.y`
//      `unscaled.y = canonical.y`
//   3. Centre + scale: `unrot = (unscaled - 0.5) × scale`
//   4. Yaw rotation around Y, then translate by `inst.pos`.
//
// Building the 4x4 columns by transforming the canonical basis:
//   col0 = forward(1,0,0) - forward(0,0,0)   (mapped X axis in world)
//   col1 = forward(0,1,0) - forward(0,0,0)
//   col2 = forward(0,0,1) - forward(0,0,0)
//   col3 = forward(0,0,0)                    (translation; projective row 1)
fn user_grass_inst_world_matrix(inst: Blade) -> mat4x4<f32> {
    let cy = cos(inst.yaw);
    let sy = sin(inst.yaw);
    // canonical (0,0,0) maps to world: pre_lean.x = 0.5 - 0.5*width,
    //   us.x = pre_lean.x + 0, us.y = 0, us.z = 0,
    //   ur = (us - 0.5) * scale = (-0.5, -0.5, -0.5) * scale shifted by width
    //   world = R(yaw) * ur + pos
    let w = inst.width;
    let s = inst.scale;
    let lx = inst.lean.x;
    let lz = inst.lean.y;
    // Column 0: ∂world/∂canonical.x, evaluated at any canonical point
    // (the map is affine, so the derivative is constant).
    //   ∂pre_lean.x/∂canon.x = w
    //   ∂us.x/∂canon.x = w; ∂us.y = 0; ∂us.z = 0
    //   ∂ur = (w*s, 0, 0)
    //   ∂world = R(yaw) * (w*s, 0, 0) = (w*s*cy, 0, w*s*sy)
    let col0 = vec4<f32>(w * s * cy, 0.0, w * s * sy, 0.0);
    // Column 1: ∂world/∂canonical.y
    //   ∂pre_lean.x = 0; ∂us.x = lx; ∂us.y = 1; ∂us.z = lz
    //   ∂ur = (lx*s, s, lz*s)
    //   ∂world = R(yaw) * (lx*s, s, lz*s)
    //          = (lx*s*cy - lz*s*sy, s, lx*s*sy + lz*s*cy)
    let col1 = vec4<f32>(lx * s * cy - lz * s * sy, s, lx * s * sy + lz * s * cy, 0.0);
    // Column 2: ∂world/∂canonical.z
    //   ∂pre_lean.x = 0; ∂us.x = 0; ∂us.y = 0; ∂us.z = 1
    //   ∂ur = (0, 0, s)
    //   ∂world = R(yaw) * (0, 0, s) = (-s*sy, 0, s*cy)
    let col2 = vec4<f32>(-s * sy, 0.0, s * cy, 0.0);
    // Column 3: world at canonical (0,0,0)
    //   pre_lean.x = 0.5 - 0.5*w; us = (pre_lean.x, 0, 0)
    //   ur = (us - 0.5) * s = ((0.5 - 0.5*w - 0.5) * s, -0.5*s, -0.5*s)
    //      = (-0.5*w*s, -0.5*s, -0.5*s)
    //   world = R(yaw) * ur + pos
    let urx = -0.5 * w * s;
    let ury = -0.5 * s;
    let urz = -0.5 * s;
    let wx = urx * cy - urz * sy + inst.pos.x;
    let wy = ury + inst.pos.y;
    let wz = urx * sy + urz * cy + inst.pos.z;
    let col3 = vec4<f32>(wx, wy, wz, 1.0);
    return mat4x4<f32>(col0, col1, col2, col3);
}

// ── inst_to_local hook ──────────────────────────────────────────────
// Map a world-space point into the blade's canonical [0, 1]³ space.
// Inverse of `inst_world_matrix`. Kept on the API for future use
// (e.g. paint-cursor hit-testing within an emitted instance).
fn user_grass_inst_to_local(world_pos: vec3<f32>, inst: Blade) -> vec3<f32> {
    let local = world_pos - inst.pos;
    let cy = cos(-inst.yaw);
    let sy = sin(-inst.yaw);
    let unrot = vec3<f32>(
        local.x * cy - local.z * sy,
        local.y,
        local.x * sy + local.z * cy,
    );
    let unscaled = unrot / inst.scale + vec3<f32>(0.5);
    let canon_y = unscaled.y;
    let pre_lean_x = unscaled.x - inst.lean.x * canon_y;
    let canon_z = unscaled.z - inst.lean.y * canon_y;
    // Undo width dilation: pre_lean is in "world-canonical-after-lean"
    // coordinates with the blade centered at 0.5 and dilated by width.
    // Map back to proto-canonical so the proto query lands on the
    // narrower pre-baked silhouette.
    let canon_x = (pre_lean_x - 0.5) / max(inst.width, 1e-3) + 0.5;
    return vec3<f32>(canon_x, canon_y, canon_z);
}

// ── inst_aabb hook ──────────────────────────────────────────────────
// Tight world-space bounds for the bent + rotated + width-dilated
// blade. Iterates 8 canonical corners through the FORWARD map; the
// convex hull of those corners encloses everything in [0, 1]³.
fn user_grass_inst_aabb(inst: Blade) -> Aabb {
    let cy_yaw = cos(inst.yaw);
    let sy_yaw = sin(inst.yaw);
    var amin = vec3<f32>(1.0e30);
    var amax = vec3<f32>(-1.0e30);
    for (var i: u32 = 0u; i < 8u; i = i + 1u) {
        let cx = f32((i >> 0u) & 1u);
        let cyc = f32((i >> 1u) & 1u);
        let cz = f32((i >> 2u) & 1u);
        // Forward (canonical → world):
        //   pre_lean.x = 0.5 + width × (canonical.x - 0.5)
        //   unscaled.xz = pre_lean.xz + lean × canonical.y
        //   unrot = (unscaled - 0.5) × scale
        //   world = R(yaw) × unrot + pos
        let pre_lean_x = 0.5 + inst.width * (cx - 0.5);
        let us_x = pre_lean_x + inst.lean.x * cyc;
        let us_y = cyc;
        let us_z = cz + inst.lean.y * cyc;
        let ur_x = (us_x - 0.5) * inst.scale;
        let ur_y = (us_y - 0.5) * inst.scale;
        let ur_z = (us_z - 0.5) * inst.scale;
        let wx = ur_x * cy_yaw - ur_z * sy_yaw + inst.pos.x;
        let wy = ur_y + inst.pos.y;
        let wz = ur_x * sy_yaw + ur_z * cy_yaw + inst.pos.z;
        let p = vec3<f32>(wx, wy, wz);
        amin = min(amin, p);
        amax = max(amax, p);
    }
    var a: Aabb;
    a.min = amin;
    a.max = amax;
    return a;
}

// ── instance_at hook (Phase B-redux) ────────────────────────────────
// Returns the k-th instance (here, blade) for a given host position
// via *out_instance, or `false` if there's no instance at index k (k
// beyond density-driven count, or host normal disqualifies the cell).
// Called per-pixel-per-leaf from the host march on band-cell hits;
// stateless — every input but `k` derives from `host_pos` + `ctx.time`
// via hashing, so no per-frame state writes are needed.
fn user_grass_instance_at(
    host_pos: vec3<f32>,
    host: HostSample,
    ctx: UserCtx,
    k: u32,
    out_instance: ptr<function, Blade>,
) -> bool {
    if (host.normal.y < 0.5) { return false; }

    let blade_height  = ctx.params[0];
    let blade_width   = ctx.params[1];
    let height_jitter = ctx.params[2];
    let density       = ctx.params[3];
    let pos_jitter    = ctx.params[4];
    let lean_amount   = ctx.params[5];
    let wind_amp      = ctx.params[6];
    let wind_freq     = ctx.params[7];

    let base_seed = grass_seed_from_pos(host_pos);
    let count_full = u32(floor(density));
    let extra_p = density - f32(count_full);

    // Hard cap at @max_emits_per_thread; same as emit's `i >= 5u` exit.
    if (k >= count_full + 1u || k >= 5u) { return false; }

    let s0 = base_seed ^ (k * 0x9E3779B9u);
    let r_density = grass_hash_u01(s0);
    // Density's fractional part probabilistically spawns the (count_full)-th
    // blade. This branch is identical to the emit-loop's break.
    if (k == count_full && r_density >= extra_p) { return false; }

    let r_jx      = grass_hash_u01(s0 ^ 0xBF58476Du);
    let r_jz      = grass_hash_u01(s0 ^ 0x94D049BBu);
    let r_height  = grass_hash_u01(s0 ^ 0xCBF29CE4u);
    let r_yaw     = grass_hash_u01(s0 ^ 0xA2B5C7D9u);
    let r_lean_x  = grass_hash_u01(s0 ^ 0xC2B2AE35u);
    let r_lean_z  = grass_hash_u01(s0 ^ 0xD3B5C7E1u);
    let r_phase   = grass_hash_u01(s0 ^ 0xFEEDFACEu);

    let jitter_radius = ctx.cell_size * pos_jitter;
    let jx = (r_jx - 0.5) * 2.0 * jitter_radius;
    let jz = (r_jz - 0.5) * 2.0 * jitter_radius;

    let h_factor = 1.0 + (r_height - 0.5) * 2.0 * height_jitter;
    let h = max(blade_height * h_factor, 1.0e-3);

    let phase = r_phase * 6.28318530718;
    let wind_x = sin(ctx.time * wind_freq + phase) * wind_amp;
    let wind_z = cos(ctx.time * wind_freq + phase * 0.73) * wind_amp;
    let base_lean_x = (r_lean_x - 0.5) * 2.0 * lean_amount;
    let base_lean_z = (r_lean_z - 0.5) * 2.0 * lean_amount;

    var b: Blade;
    b.pos = vec3<f32>(host_pos.x + jx, host_pos.y + 0.5 * h, host_pos.z + jz);
    b.scale = h;
    b.yaw = r_yaw * 6.28318530718;
    b.width = max(blade_width, 1.0e-3);
    b.lean = vec2<f32>(base_lean_x + wind_x, base_lean_z + wind_z);
    *out_instance = b;
    return true;
}
