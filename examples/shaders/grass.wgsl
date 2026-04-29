// grass.wgsl — Option B (instance pipeline) demo shader.
//
// Drop this into your project's `assets/shaders/grass.wgsl`. Then
// create a material whose `shader` field is `"grass"` and paint that
// material onto an entity's surface. The engine's per-frame
// `tick_instance_pipeline` (rkp-engine/src/render_worker.rs) will
// detect the painted material, bake the blade prototype, scatter one
// blade per surface sample, march it per-pixel, and composite into
// the merged G-buffer that shade reads from.
//
// Locked Option B authoring API (Stage 1 memo): WGSL-only,
// `// @instance_proto <Struct>` directive + struct decl + the four
// hooks (proto / emit / optional inst_aabb / optional inst_to_local).
// See `crates/rkp-render/src/shaders/user_shader_proto.wgsl` and
// `..._emit.wgsl` for the in-tree templates that surround this shader
// at compose time.

// ── Metadata directives ─────────────────────────────────────────────
// Region inputs the engine uses when emitting per-region scatter
// requests. See `lifecycle.rs::submit_render_frame` (the painted-AABB
// walk + emit loop).
//
// max_depth: bake-time prototype octree depth. Each level doubles
//            per-axis voxel count; depth 3 = 32³ voxels per blade.
// region_thickness: world-space band above the painted surface inside
//                   which blades scatter. 0.6 m here = blades up to
//                   ~60 cm reach.
// cell_size: host-space sample-grid cell (= brick-parent / 4 on the
//            emit path). Smaller = denser blade carpet.

// @max_depth 3
// @region_thickness 0.6
// @cell_size 0.04

// ── Per-instance state struct ───────────────────────────────────────
// Tagged-field discovery (see `crates/rkp-render/src/instance_proto.rs`):
//   * `pos: vec3<f32>` — required, sets the instance's world origin.
//   * `yaw: f32` — optional, here just rides along (V1 march doesn't
//     use it; reserved for V2 rotation).
// 16 bytes total (well under the 32 B soft cap, 64 B hard cap).

// @instance_proto Blade
struct Blade {
    pos: vec3<f32>,
    yaw: f32,
}

// ── Prototype hook ──────────────────────────────────────────────────
// Called for every cell in canonical [0,1]³ during the bake compute
// dispatch. Returns a `VoxelEmit` describing whether this cell is
// inside the prototype shape (occupancy=1) and, if so, its surface
// normal + materials. The bake writes occupied cells into the
// prototype octree's leaf level.
//
// The blade is a thin column tapering from the base. Centered on
// (0.5, *, 0.5) with radius lerping from 0.10 at the base to 0.015
// at the tip. y=0 is the painted surface; y=1 is the tip.
fn user_grass_proto(uvw: vec3<f32>) -> VoxelEmit {
    var v: VoxelEmit;
    v.occupancy = 0u;

    let to_axis = uvw.xz - vec2<f32>(0.5, 0.5);
    let r = length(to_axis);
    let max_radius = mix(0.10, 0.015, uvw.y);
    if (uvw.y < 1.0 && r < max_radius) {
        v.occupancy = 1u;
        // Outward-radial normal with a small upward bias — reads as
        // the side of a thin blade leaning slightly toward the tip.
        v.normal = normalize(vec3<f32>(to_axis.x, 0.3, to_axis.y));
        // Material primary = 1 means "use the host material's slot 1
        // shading parameters." This isn't host material inheritance
        // (the engine handles that elsewhere); it's a placeholder.
        // V2: the composer will splice in `region.material_id` so
        // each instance picks up the painted host's shading.
        v.material_primary = 1u;
        v.material_secondary = 0u;
        v.blend_weight = 0u;
    }
    return v;
}

// ── Emit hook ───────────────────────────────────────────────────────
// Called once per `(host_pos, host)` sample during the per-region
// scatter dispatch. Calls `emit_instance(blade)` zero or more times
// to atomic-append blades into this region's slice of the global
// instance pool.
//
// Strategy: emit exactly one blade per surface sample where the
// painted material matches this shader's material_id and the surface
// is upward-facing. Higher densities can come from either:
//   * a smaller `@cell_size` directive above (more samples per area);
//   * emitting multiple blades per sample with a sub-grid offset.
// V1 sticks to one-per-sample for simplicity.
fn user_grass_emit(host_pos: vec3<f32>, host: HostSample, ctx: UserCtx) {
    // Off the host surface — nothing to grow on.
    if (host.distance > ctx.cell_size) {
        return;
    }
    // Painted material must match this shader's material — otherwise
    // the sample is an unpainted bit of the host where we shouldn't
    // grow grass.
    if (host.material != ctx.material_id) {
        return;
    }
    // Don't grow grass on ceilings or steep walls. y > 0.5 → angle
    // less than ~60° from vertical.
    if (host.normal.y < 0.5) {
        return;
    }

    var b: Blade;
    b.pos = host_pos;
    b.yaw = 0.0;
    emit_instance(b);
}
