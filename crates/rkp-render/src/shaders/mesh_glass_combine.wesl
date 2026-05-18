// Glass G-buffer pack — third stage of the mesh-mode glass pipeline.
//
// Reads the entry-pass and exit-pass color targets that
// `mesh_glass.wesl::frag_front` / `frag_back` wrote, plus the primary
// G-buffer's `position.w` (opaque hit distance), and assembles the
// final `gbuf_glass` Rg32Uint output that `rkp_glass.wesl` consumes.
//
// Output layout that `rkp_glass.wesl` consumes:
//   · R = oct-packed world entry normal.
//   · G = `(thickness_mm << 16) | material_id` where
//         `thickness_mm = clamp((exit - entry) * 1000, 1, 65535)`.
//   · `R==0, G==0` ⇒ "no glass at this pixel" sentinel —
//     `rkp_glass.wesl` gates on `thickness_mm == 0`.
//
// Three filter cases produce the no-glass sentinel:
//   1. The entry FS never ran here (clear value preserved). Detected
//      by `entry_dist >= ENTRY_SENTINEL`.
//   2. Glass is behind the closest opaque hit (entry past
//      `gbuf_position.w`). The march has the same gate at line 1625.
//   3. The back-face FS didn't run (no exit recorded — happens for
//      cards / non-watertight glass). Sentinel: exit_dist == 0.
//      We pack a tiny 1 mm thickness so the composite still draws
//      something rather than collapsing to no-glass.

const ENTRY_SENTINEL: f32 = 1.0e30;

struct CombineParams {
    /// `RKP_MESH_GLASS_DEBUG_FORCE = 1` ⇒ ignore the entry pass and
    /// force every pixel with an opaque mesh hit into a 100 mm
    /// glass shell with material 1 and an outward +Z normal. Used
    /// to bisect "no glass visible" — if force-mode shows glass
    /// everywhere, the entry/back FS discard path is at fault; if
    /// not, the combine writes or the rkp_glass composite read are.
    debug_force: u32,
    _pad0: u32, _pad1: u32, _pad2: u32,
}

@group(0) @binding(0) var glass_entry_packed: texture_2d<u32>;
@group(0) @binding(1) var glass_exit_dist:    texture_2d<f32>;
@group(0) @binding(2) var gbuf_position:      texture_2d<f32>;
@group(0) @binding(3) var gbuf_glass_out:     texture_storage_2d<rg32uint, write>;
@group(0) @binding(4) var<uniform> params:    CombineParams;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(gbuf_glass_out);
    if gid.x >= dims.x || gid.y >= dims.y { return; }
    let coord = vec2<i32>(gid.xy);

    let packed = textureLoad(glass_entry_packed, coord, 0);
    var oct_normal = packed.x;
    var material_id = packed.y & 0xFFFFu;
    var entry_dist = bitcast<f32>(packed.z);

    // Debug: force-glass every opaque mesh hit so we can verify the
    // combine→composite chain in isolation from the FS discard path.
    if params.debug_force != 0u {
        let opaque_dist_dbg = textureLoad(gbuf_position, coord, 0).w;
        if opaque_dist_dbg < 1.0e9 {
            // +Z normal in oct-pack: oct(0,0,1) ≈ packed = 0x00000000
            // since the encoding is `(snorm16(0), snorm16(0))`. We use
            // a vec3(0,1,0) instead so the pack isn't all-zero.
            // pack_oct_normal((0,1,0)) ≈ 0x0000_7FFF in 2×snorm16.
            oct_normal = 0x00007FFFu;
            material_id = 1u;
            // Place the entry 5 mm in front of the opaque hit so
            // thickness is positive.
            entry_dist = max(opaque_dist_dbg - 0.005, 0.001);
            // Spoof a 100 mm shell.
            let thickness_mm_dbg = 100u;
            let packed_g_dbg = (material_id & 0xFFFFu) | (thickness_mm_dbg << 16u);
            textureStore(gbuf_glass_out, coord, vec4<u32>(oct_normal, packed_g_dbg, 0u, 0u));
        } else {
            textureStore(gbuf_glass_out, coord, vec4<u32>(0u, 0u, 0u, 0u));
        }
        return;
    }

    if entry_dist >= ENTRY_SENTINEL {
        textureStore(gbuf_glass_out, coord, vec4<u32>(0u, 0u, 0u, 0u));
        return;
    }

    // Opaque-hit gate: drop glass that lives behind the closest opaque
    // surface. `gbuf_position.w` carries the opaque hit distance from
    // mesh raster (and `1e10` on miss, which never gates anything).
    let opaque_dist = textureLoad(gbuf_position, coord, 0).w;
    if entry_dist >= opaque_dist {
        textureStore(gbuf_glass_out, coord, vec4<u32>(0u, 0u, 0u, 0u));
        return;
    }

    let raw_exit = textureLoad(glass_exit_dist, coord, 0).r;
    // Clamp the exit to the closer of the back-face hit or the opaque
    // hit — same as the march's `min(glass_exit_dist, max_world_dist)`
    // on line 1626. If no back-face hit (raw_exit == 0), fall back to
    // 1 mm so the composite still renders glass rather than the
    // no-glass sentinel.
    var exit_dist: f32;
    if raw_exit <= entry_dist {
        exit_dist = entry_dist + 0.001;
    } else {
        exit_dist = min(raw_exit, opaque_dist);
    }

    let thickness_world = max(0.0, exit_dist - entry_dist);
    let thickness_mm = u32(clamp(thickness_world * 1000.0, 1.0, 65535.0));
    let packed_g = material_id | (thickness_mm << 16u);
    textureStore(gbuf_glass_out, coord, vec4<u32>(oct_normal, packed_g, 0u, 0u));
}
