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
//
// V1.1+normals: anchor carries `surface_normal`, the world-space unit
// normal averaged from every painted leaf in the tile (LeafAttr.normal
// is the prefiltered SDF gradient baked at voxelize time). Blades
// build a TBN frame around N, so:
//   · blade "up" axis follows the surface
//   · random yaw spins around N (stays in the tangent plane)
//   · base Y is projected onto the surface plane so blades sit on
//     slopes instead of floating from paint_max.y
//   · wind sway displaces the tip in the tangent plane

// ── Manifest ──────────────────────────────────────────────────────
// @geometry procedural { vertex_count: 6 }
// @tile_size 0.5
// @max_distance 150.0
// @animated

// ── Per-material params ───────────────────────────────────────────
// `density` is blades-per-m² of painted surface. Probabilistic
// rounding spreads sub-integer expected counts across spawns.
// At the V1 cap of 32 blades per anchor with tile_size = 0.5 m
// (area 0.25 m²), saturation hits at density = 128 blades/m².
// @param blade_height: f32 = 0.35,  range = [0.05, 1.5]
// @param blade_width:  f32 = 0.04,  range = [0.01, 0.2]
// @param density:      f32 = 100.0, range = [1.0, 128.0]
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

// Robust orthonormal tangent frame around N. Pick the reference axis
// as world-X when N is too close to world-Y (cross(Y,Y)=0 would
// degenerate), else world-Y. Returns (T, B) with N = cross(T, B).
fn grass_tangent_frame(n: vec3<f32>) -> mat2x3<f32> {
    let ref_axis = select(
        vec3<f32>(0.0, 1.0, 0.0),
        vec3<f32>(1.0, 0.0, 0.0),
        abs(n.y) > 0.95,
    );
    let t = normalize(cross(ref_axis, n));
    let b = cross(n, t);
    return mat2x3<f32>(t, b);
}

// Per-spawn blade base position in world space.
//
// XZ is randomized inside the anchor's TILE cube (tile_coord ×
// tile_size, world-transformed) — that's LOD-stable: the same world
// tile_coord maps to the same world XZ rectangle no matter which
// terrain LOD entity covers it. Earlier versions used `paint_min/max`
// (the painted-leaf AABB) which grew with voxel size and made blade
// XZ shimmer on LOD swap.
//
// Y is projected onto the per-tile surface plane through
// (tile_center_xz, surface_y) with normal `surface_normal`. Both come
// from the leaf nearest to the tile-center XZ (CPU pre-sample), so
// they're LOD-stable too. On a flat tile this returns `surface_y`
// unchanged; on a slope it tracks the surface so blades aren't
// floating. Falls back to `surface_y` when the normal is too tipped
// (|N.y| < eps) — flat horizontal plane semantics for vertical
// surfaces.
fn grass_blade_base(anchor: AnchorContext, spawn_idx: u32) -> vec3<f32> {
    let s0 = anchor.seed ^ (spawn_idx * 0x9E3779B9u);
    let r_px = grass_hash_u01(s0 ^ 0xBF58476Du);
    let r_pz = grass_hash_u01(s0 ^ 0x94D049BBu);
    let pos_x = mix(anchor.tile_min.x, anchor.tile_max.x, r_px);
    let pos_z = mix(anchor.tile_min.z, anchor.tile_max.z, r_pz);

    let cx = 0.5 * (anchor.tile_min.x + anchor.tile_max.x);
    let cz = 0.5 * (anchor.tile_min.z + anchor.tile_max.z);
    let n  = anchor.surface_normal;
    var pos_y = anchor.surface_y;
    if (abs(n.y) > 1e-3) {
        pos_y = anchor.surface_y
              - (n.x * (pos_x - cx) + n.z * (pos_z - cz)) / n.y;
    }
    return vec3<f32>(pos_x, pos_y, pos_z);
}

// ── spawn_count ───────────────────────────────────────────────────
// Density per painted XZ ground area (`paint_min/max` is the painted-
// leaf BB, not the tile cube — so density scales on the actual
// painted m², not on tile area that may be mostly empty).
// Probabilistic rounding so sub-integer counts don't floor to 0.
// Capped at the engine's per-anchor spawn ceiling (V1 = 32,
// matching `MAX_SPAWNS_PER_ANCHOR_V1`). The host WESL also clamps
// `out_counts[i]` to the same value, so returning more here is safe
// but wasted work.
fn spawn_count(anchor: AnchorContext, frame: FrameContext) -> u32 {
    let density = ctx_param(2);
    // Use the LOD-stable tile cube area, not the painted-leaf BB —
    // see grass_blade_base. For a fully-painted tile this is identical
    // (cube == leaf BB); for partly-painted tiles the cube may
    // over-count, but `spawn_alive`'s paint_probe cuts the unpainted
    // spawns, so the visible blade count tracks paint regardless.
    let extent_x = max(anchor.tile_max.x - anchor.tile_min.x, 0.0);
    let extent_z = max(anchor.tile_max.z - anchor.tile_min.z, 0.0);
    let xz_area = extent_x * extent_z;
    let raw = density * xz_area;

    let base = u32(floor(raw));
    let frac = raw - f32(base);
    let r = grass_hash_u01(anchor.seed ^ 0xA341316Cu);
    var n = base;
    if (r < frac) { n = n + 1u; }
    return min(n, 32u);
}

// ── spawn_alive ───────────────────────────────────────────────────
fn spawn_alive(anchor: AnchorContext, spawn_idx: u32, frame: FrameContext) -> bool {
    // Tile-cube placement (LOD-stable) puts blades anywhere inside
    // the tile, including sub-cells that hold a different material
    // (FBM terrain mixes grass with sand/rock/snow inside any one
    // shader tile). `anchor.paint_mask` is a 4×4 bitmap, set CPU-
    // side from the leaves that actually carry this material; we
    // map the blade's XZ to its sub-cell and keep only the spawns
    // whose bit is set. This is the cheap CPU-pre-computed
    // alternative to the per-spawn paint_probe octree descent.
    let base = grass_blade_base(anchor, spawn_idx);
    let extent_x = max(anchor.tile_max.x - anchor.tile_min.x, 1e-6);
    let extent_z = max(anchor.tile_max.z - anchor.tile_min.z, 1e-6);
    let u = clamp(
        (base.x - anchor.tile_min.x) / extent_x,
        0.0,
        0.999999,
    );
    let v = clamp(
        (base.z - anchor.tile_min.z) / extent_z,
        0.0,
        0.999999,
    );
    let cx = u32(u * 4.0);
    let cz = u32(v * 4.0);
    let bit = cz * 4u + cx;
    return (anchor.paint_mask & (1u << bit)) != 0u;
}

// ── vs ────────────────────────────────────────────────────────────
// Places one blade per spawn_idx inside the tile's XZ AABB; blade base
// lifts onto the per-tile surface plane (see `grass_blade_base`).
// Six-vertex tapered quad built in the surface-tangent frame: blade
// "up" = anchor.surface_normal, random yaw around that axis, wind
// sway in the tangent plane.
fn vs(anchor: AnchorContext, spawn_idx: u32, vid: u32, frame: FrameContext) -> VsOut {
    let blade_height = ctx_param(0);
    let blade_width  = ctx_param(1);
    let wind_amp     = ctx_param(3);
    let wind_freq    = ctx_param(4);

    // Per-spawn deterministic seeds (stable across frames).
    let s0 = anchor.seed ^ (spawn_idx * 0x9E3779B9u);
    let r_yaw    = grass_hash_u01(s0 ^ 0xA2B5C7D9u);
    let r_height = grass_hash_u01(s0 ^ 0xCBF29CE4u);
    let r_phase  = grass_hash_u01(s0 ^ 0xFEEDFACEu);

    let base = grass_blade_base(anchor, spawn_idx);

    // Tangent frame on the surface. T/B span the tangent plane; N is
    // the blade up-axis.
    let n = anchor.surface_normal;
    let tb = grass_tangent_frame(n);
    let t = tb[0];
    let b = tb[1];

    let h   = blade_height * (0.7 + r_height * 0.6);
    let yaw = r_yaw * 6.28318530718;
    let c   = cos(yaw);
    let s   = sin(yaw);

    // vid → (u, v) in blade-local quad space (u across width, v up).
    var u: f32 = 0.0;
    var v: f32 = 0.0;
    if (vid == 0u)      { u = 0.0; v = 0.0; }
    else if (vid == 1u) { u = 1.0; v = 0.0; }
    else if (vid == 2u) { u = 1.0; v = 1.0; }
    else if (vid == 3u) { u = 0.0; v = 0.0; }
    else if (vid == 4u) { u = 1.0; v = 1.0; }
    else                { u = 0.0; v = 1.0; }

    // Tapered quad in local frame. local_w = width axis (in tangent
    // plane), local_h = up along N.
    let local_w = (u - 0.5) * blade_width * (1.0 - v * 0.8);
    let local_h = v * h;

    // Wind sway — tip drifts in the tangent plane.
    let wind_phase = r_phase * 6.28318530718;
    let wind_a = sin(frame.time * wind_freq + wind_phase) * wind_amp * v;
    let wind_b = cos(frame.time * wind_freq + wind_phase * 0.73) * wind_amp * v;

    // Pick blade facing direction by yawing T/B around N. `dir_w`
    // is the chosen width axis; `dir_perp` is the in-plane sway axis.
    let dir_w    = t * c + b * s;
    let dir_perp = -t * s + b * c;

    let offset = dir_w * (local_w + wind_a)
               + dir_perp * wind_b
               + n * local_h;

    let world_pos = base + offset;

    let clip = camera.view_proj * vec4<f32>(world_pos, 1.0);

    var out: VsOut;
    out.clip_pos = clip;
    out.world_pos = world_pos;
    out.world_normal = n;
    out.material_packed = anchor.material_id;
    out.color_rgb = vec3<f32>(1.0);
    out.blend_f = 0.0;
    out.intensity = 0u;
    return out;
}
