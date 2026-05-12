// grass.wgsl — V1 mesh-path reference user shader (tile-based anchors).
//
// Each painted material gets one anchor per painted tile, where tile
// size is set by `@tile_size` below. The shader's `spawn_count` decides
// how many blades to emit for the tile based on its world-space AABB
// area; `vs` places blades inside that AABB.
//
// V1.1: anchor carries `paint_min/max` (the painted-leaf BB in world)
// alongside the stable `tile_min/max` cube. Spawn placement uses
// paint bounds so blades land on the painted region, not in unpainted
// parts of the tile cube. Trade-off: paint bounds grow when paint
// extends within an existing tile, so blade XZ shifts during active
// drag-paint. For a left-alone painted region the bounds are stable.
// V1 remaining limit:
//   · Blades grow +Y. No per-tile normal yet; slopes will look
//     incorrect.

// ── Manifest ──────────────────────────────────────────────────────
// @geometry procedural { vertex_count: 6 }
// @tile_size 0.5
// @animated

// ── Per-material params ───────────────────────────────────────────
// `density` is blades-per-m² of painted surface. Probabilistic
// rounding spreads sub-integer expected counts across spawns.
// At the V1 cap of 64 blades per anchor with tile_size = 0.5 m
// (area 0.25 m²), saturation hits at density = 256 blades/m².
// @param blade_height: f32 = 0.35,  range = [0.05, 1.5]
// @param blade_width:  f32 = 0.04,  range = [0.01, 0.2]
// @param density:      f32 = 100.0, range = [1.0, 256.0]
// @param wind_amp:     f32 = 0.08,  range = [0.0, 0.3]
// @param wind_freq:    f32 = 1.5,   range = [0.0, 6.0]

// ── Helpers ───────────────────────────────────────────────────────
fn grass_hash_u01(seed: u32) -> f32 {
    var x = seed;
    x = x ^ (x >> 16u);
    x = x * 0x7feb352du;
    x = x ^ (x >> 15u);
    x = x * 0x846ca68bu;
    x = x ^ (x >> 16u);
    return f32(x) / 4294967295.0;
}

// Per-spawn blade base position in world space. Spawn XZ falls inside
// the anchor's PAINT bounds (te.aabb world-transformed) so blades
// land on the actual painted region. Y is `anchor.surface_y` =
// world top of painted leaves.
fn grass_blade_base(anchor: AnchorContext, spawn_idx: u32) -> vec3<f32> {
    let s0 = anchor.seed ^ (spawn_idx * 0x9E3779B9u);
    let r_px = grass_hash_u01(s0 ^ 0xBF58476Du);
    let r_pz = grass_hash_u01(s0 ^ 0x94D049BBu);
    return vec3<f32>(
        mix(anchor.paint_min.x, anchor.paint_max.x, r_px),
        anchor.surface_y,
        mix(anchor.paint_min.z, anchor.paint_max.z, r_pz),
    );
}

// ── spawn_count ───────────────────────────────────────────────────
// Density per painted XZ ground area (`paint_min/max` is the painted-
// leaf BB, not the tile cube — so density scales on the actual
// painted m², not on tile area that may be mostly empty).
// Probabilistic rounding so sub-integer counts don't floor to 0.
// Capped at the engine's per-anchor spawn ceiling (V1 = 64,
// matching `MAX_SPAWNS_PER_ANCHOR_V1`).
fn spawn_count(anchor: AnchorContext, frame: FrameContext) -> u32 {
    let density = ctx_param(2);
    let extent_x = max(anchor.paint_max.x - anchor.paint_min.x, 0.0);
    let extent_z = max(anchor.paint_max.z - anchor.paint_min.z, 0.0);
    let xz_area = extent_x * extent_z;
    let raw = density * xz_area;

    let base = u32(floor(raw));
    let frac = raw - f32(base);
    let r = grass_hash_u01(anchor.seed ^ 0xA341316Cu);
    var n = base;
    if (r < frac) { n = n + 1u; }
    return min(n, 64u);
}

// ── spawn_alive ───────────────────────────────────────────────────
// Cull blades whose XZ position doesn't sit above any painted host
// voxel inside the anchor's tile. The blade base Y (`anchor.surface_y`)
// is the tile floor — stable for blade placement, but typically BELOW
// the painted ground voxel — so a single point-probe at that Y would
// always miss. Instead we scan the column from tile_max.y down to
// tile_min.y, returning true on the first paint_probe hit.
//
// 8 samples covers tiles up to `tile_size / voxel_size = 8` deep,
// which fits the typical 0.5m tile / 0.05–0.1m voxel ratio. Cost is
// 8 octree descents per spawn at worst — negligible against the
// per-vertex math the blade would otherwise trigger.
fn spawn_alive(anchor: AnchorContext, spawn_idx: u32, frame: FrameContext) -> bool {
    // Spawn region is the painted-leaf BB on the anchor — every
    // blade is already inside the painted area, no probe needed.
    return true;
}

// ── vs ────────────────────────────────────────────────────────────
// Places one blade per spawn_idx inside the tile's XZ AABB; blade base
// sits at `aabb_max.y` (the painted surface top, V1 approximation).
// Six-vertex tapered quad oriented around a per-blade yaw with wind
// sway driven by `frame.time`.
fn vs(anchor: AnchorContext, spawn_idx: u32, vid: u32, frame: FrameContext) -> VsOut {
    let blade_height = ctx_param(0);
    let blade_width  = ctx_param(1);
    let wind_amp     = ctx_param(3);
    let wind_freq    = ctx_param(4);

    // Per-spawn deterministic seeds (stable across frames). Position
    // shared with `spawn_alive`'s probe so dead blades don't render.
    let s0 = anchor.seed ^ (spawn_idx * 0x9E3779B9u);
    let r_yaw    = grass_hash_u01(s0 ^ 0xA2B5C7D9u);
    let r_height = grass_hash_u01(s0 ^ 0xCBF29CE4u);
    let r_phase  = grass_hash_u01(s0 ^ 0xFEEDFACEu);

    let base = grass_blade_base(anchor, spawn_idx);
    let base_x = base.x;
    let base_y = base.y;
    let base_z = base.z;

    let h = blade_height * (0.7 + r_height * 0.6);
    let yaw = r_yaw * 6.28318530718;
    let c = cos(yaw);
    let s = sin(yaw);

    // vid → (u, v) in blade-local quad space.
    var u: f32 = 0.0;
    var v: f32 = 0.0;
    if (vid == 0u)      { u = 0.0; v = 0.0; }
    else if (vid == 1u) { u = 1.0; v = 0.0; }
    else if (vid == 2u) { u = 1.0; v = 1.0; }
    else if (vid == 3u) { u = 0.0; v = 0.0; }
    else if (vid == 4u) { u = 1.0; v = 1.0; }
    else                { u = 0.0; v = 1.0; }

    // Tapered local geometry. Width narrows toward the tip.
    let local_x = (u - 0.5) * blade_width * (1.0 - v * 0.8);
    let local_y = v * h;
    let local_z = 0.0;

    // Wind sway — tip drifts in world XZ as sinusoid of frame.time.
    let wind_phase = r_phase * 6.28318530718;
    let wind_x = sin(frame.time * wind_freq + wind_phase) * wind_amp * v;
    let wind_z = cos(frame.time * wind_freq + wind_phase * 0.73) * wind_amp * v;

    let rotated_x = local_x * c - local_z * s + wind_x;
    let rotated_z = local_x * s + local_z * c + wind_z;

    let world_pos = vec3<f32>(
        base_x + rotated_x,
        base_y + local_y,
        base_z + rotated_z,
    );

    let clip = camera.view_proj * vec4<f32>(world_pos, 1.0);

    var out: VsOut;
    out.clip_pos = clip;
    out.world_pos = world_pos;
    // V1: assume +Y normal (flat ground). Per-tile normal averaging
    // is a V1.1 follow-up so blades on slopes don't look wrong.
    out.world_normal = vec3<f32>(0.0, 1.0, 0.0);
    out.material_packed = anchor.material_id;
    out.color_rgb = vec3<f32>(1.0);
    out.blend_f = 0.0;
    out.intensity = 0u;
    return out;
}
