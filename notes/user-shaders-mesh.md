# User Shaders on Mesh Path — V1 Plan

## Goal

Replace the voxel-pipeline user-shader path (proto-bake octrees + emit pass + tile-bin pass + per-pixel march scan) with a **vertex-shader-driven path** that runs through the existing mesh raster + shadow infrastructure. The user authors a small WGSL module — a manifest, a spawn-count function, a vertex shader, and an optional spawn-alive predicate — and the engine handles cardinality, instancing, indirect draws, shadows, and G-buffer integration.

User-shader geometry becomes "decorative procedural geometry attached to a host surface, scaling to high cardinality" — covers grass, leaves, scattered groundcover, rocks-as-cover, and the open class of "things the user hasn't thought of yet."

## Background and scope

The current path (`project_user_shader_emit_rebuild_2026_05_05`, tip `7cc6230`) is:

1. CPU collects `EmitLeaf` records from painted leaves whose material has an `instance_at` hook.
2. `user_shader_proto_pass` bakes each shader to a canonical [0,1]³ octree prototype.
3. `user_shader_emit_pass` walks painted leaves, dispatches per-leaf `k = 0..MAX_EMITS (=8)` calls to `instance_at`, packs accepted instances as `RkpGpuInstance` records with affine `world` matrices.
4. `user_shader_tile_bin_pass` projects each instance's world AABB to screen and bins into per-tile lists.
5. Primary march scans each pixel's tile list, descends the per-instance proto octree, writes the hit into the visibility buffer.
6. Shadow trace path was disabled when the band-cell BFS was stripped — emitted blades cast no shadows today.

This shape was forced by the voxel pipeline: the march only walks octree leaves, so any emitted blade had to be a baked prototype octree; memory pressure forced HW-instancing of a single canonical mesh; shadows had to ride the TLAS path. With mesh primary visibility + CSM + Nanite-style LOD now shipped, all three constraints are gone.

After discussion (this session), we're committing to a **vertex-shader user-shader path** that runs alongside the mesh raster. The existing emit / tile-bin / proto-bake / proto-rollup / per-pixel-tile-scan retires.

**In scope for V1:**
- Anchor source = painted leaves on host octree (same source as today; single anchor source, fixed in V1).
- Composable-primitives API: manifest declares geometry source + animation subscription + spawn caching mode; user authors `spawn_count`, `vs`, and optional `spawn_alive`.
- Dynamic per-anchor spawn count (compute pass → prefix sum → fill → indirect draw).
- Geometry source = `procedural { vertex_count, index_count }` (vid-driven, no input mesh) **or** `mesh { asset_handle }` (HW-instance a single mesh asset).
- Full G-buffer write from the user-shader VS+FS — own raster pass with `LoadOp::Load`, depth-tested against the shared depth attachment.
- Shadow VS shares the same path, drawn against the existing CSM cascades.
- Engine-provided `AnchorContext` (with `surface_area`, `leaf_extent`, stable `seed`) and `FrameContext` (time, wind, camera).
- FS defaults to "pack `AnchorContext.material` + interpolated color into G-buffer"; user can override.
- Spawn-count caching: manifest declares `static` (paint-epoch-keyed) or `per_frame`; engine caches the spawn-count + fill output accordingly.
- Picking returns `USER_SHADER_PICK_SENTINEL` per the load-bearing rule (`feedback_no_user_shader_picking`).

**Out of scope (explicitly):**
- Multiple anchor sources (mesh-surface points, ECS lists, GPU compute scatter). Painted-leaves only in V1.
- Parent skinning (leaves following a tree-branch's bones). Anchors are static-bound; deferred to V2.
- Multi-topology per material (mesh library with per-spawn `which_mesh` index).
- Compute-driven emission (user-authored compute pass writing arbitrary vertex buffers).
- Per-spawn picking. Sentinel return is load-bearing.
- Glass on user-shader output. If a user shader's material is glass, emit opaque-uniform in V1; full glass routing is a follow-up.
- Backwards-compatibility shim for the old API. Old `instance_at` + `inst_world_matrix` + `inst_aabb` + `inst_to_local` hooks are gone; existing `grass.wgsl` gets rewritten as the V1 reference.

## Architecture

### Pipeline ordering per frame

```
1. Octree-mesh raster      → vis buffer + depth                  (existing)
2. Splat raster            → vis buffer + depth                  (existing)
3. splat_resolve compute   → gbuf_normal/material/glass          (existing)
4. Proxy-mesh raster       → full G-buffer + depth               (existing)
5. user_shader spawn_count → per-anchor counts                   (NEW, per shader)
6. user_shader prefix_sum  → per-anchor offsets + total          (NEW, per shader)
7. user_shader fill        → instance records                    (NEW, per shader)
8. user_shader raster      → full G-buffer + depth (indirect)    (NEW, per shader)
9. user_shader shadow VS   → CSM cascades (indirect)             (NEW, per shader, per cascade)
```

Steps 5–8 run **once per active user-shader material** with painted leaves. Step 9 runs once per cascade per shader. All four are skipped when the shader has no anchors (paint-epoch gate).

### Authoring surface

A V1 user shader is a single `.wgsl` file with:

```wgsl
// ── Manifest (top-of-file directives) ─────────────────────────────
// @geometry procedural { vertex_count: 14, index_count: 36 }   // OR
// @geometry mesh { asset: "blade.glb" }
// @spawn_count_cache static                                    // or per_frame
// @animated                                                    // subscribe to FrameContext.time + wind
// @param blade_height: f32 = 0.35, range = [0.05, 1.5]
// ... (existing @param syntax unchanged)

// ── Spawn-count function ──────────────────────────────────────────
// Returns N spawns to allocate for this anchor. Engine prefix-sums
// counts across all anchors → total spawn count → indirect args.
fn spawn_count(anchor: AnchorContext, frame: FrameContext) -> u32 {
    let density = ctx_param(3);  // params accessed via engine helper
    return u32(density * anchor.surface_area);
}

// ── Vertex shader (required) ──────────────────────────────────────
// Runs once per (spawn × vertex_index). Returns G-buffer-ready output.
fn vs(anchor: AnchorContext, spawn_idx: u32, vid: u32, frame: FrameContext) -> VsOut {
    // user computes clip_pos, world_pos, world_normal, material, color
}

// ── Spawn predicate (optional) ────────────────────────────────────
// Runs once per allocated spawn. Returning false short-circuits the
// VS (no triangles emitted for this spawn). Use for:
//   · paint-coverage edge filter (probe host material at jittered pos)
//   · random rejection (Poisson-disc thinning)
//   · per-frame culling cheaper than vs-level discards
fn spawn_alive(anchor: AnchorContext, spawn_idx: u32, frame: FrameContext) -> bool {
    return true;  // default if omitted
}

// ── Fragment shader (optional) ────────────────────────────────────
// Defaults to "pack AnchorContext.material + interpolated color".
// Override only if the shader needs custom material logic per-fragment.
fn fs(in: FsIn) -> FsOut {
    // default body
}
```

That's it — no proto bake, no canonical [0,1]³ box, no inverse maps, no AABB hook. The vertex shader writes world-space positions directly.

### AnchorContext

Engine-provided per-anchor uniform. Computed at anchor-collection time, cached on `paint_epoch + geometry_epoch`.

```wgsl
struct AnchorContext {
    world_pos:        vec3<f32>,    // leaf center, world space
    surface_normal:   vec3<f32>,    // host surface normal at this leaf
    leaf_extent:      f32,          // half-size of leaf in world units
    surface_area:     f32,          // leaf_size² × |dot(n, up)| or similar projection
    material_id:      u32,          // primary material on this leaf
    material_blend:   u32,          // packed: secondary (16) | blend4 (4) | ...
    host_color:       vec4<f32>,    // unpacked host color at this leaf (paint overlay)
    leaf_slot:        u32,          // for host_sample probes (optional user use)
    seed:             u32,          // stable per-anchor random seed (hashed from world_pos)
    _pad:             u32,
}
```

Engine populates this from painted leaves; user reads but doesn't write. Per-anchor records are packed in a tight buffer; the fill pass writes one instance record per allocated spawn referencing its anchor index.

### FrameContext

Engine-provided per-frame uniform. Subscribed via manifest directives so it's small and explicit.

```wgsl
struct FrameContext {
    time:           f32,
    delta_time:     f32,
    wind_dir:       vec3<f32>,
    wind_strength:  f32,
    camera_pos:     vec3<f32>,
    _pad:           f32,
}
```

V1 fills all fields unconditionally; manifest `@animated` is informational (gates whether `per_frame` caching is allowed for `spawn_count`).

### Engine pipeline detail

#### Step 5 — spawn_count compute pass

- One workgroup thread per anchor.
- Composes the user's `spawn_count` function into a single dispatched compute shader (one shader per active user-shader material).
- Bindings: `anchors[]` (storage, RO), `out_count[]: u32` (storage, RW), `FrameContext` uniform, `params` uniform.
- Output: `count[i] = spawn_count(anchors[i], frame)`.

#### Step 6 — prefix sum

- Standard exclusive prefix scan over `out_count[]` → `offset[]` + `total: u32`.
- Single-pass shared-memory scan for ≤4096 anchors; multi-pass otherwise. Either reuse an existing prefix-sum utility in the tree or stand up a minimal one (the spawn-count buffer is bounded by `MAX_ANCHORS_PER_SHADER` ~ 64k for V1).
- Writes `instance_records.total_count` and (via copy) the indirect draw args buffer: `(vertex_count_per_spawn × total, instance_count = 1, ...)`.

#### Step 7 — fill compute pass

- One workgroup thread per anchor; per-thread loop `for k in 0..count[i]` writes instance records at `offset[i] + k`.
- An instance record is minimal — just `{ anchor_idx: u32, spawn_idx: u32 }` (8 B). VS reads it via `instance_index` and dereferences `anchors[record.anchor_idx]`.
- If `spawn_alive` is defined: per-instance call → write a "skip" sentinel into the record (or write 0xFFFFFFFF; VS short-circuits without emitting any vertex output). Cheaper than discarding in the FS.

#### Step 8 — user_shader raster pass

- Pipeline: own raster pipeline per user shader. Same color attachment set as `mesh_proxy_pass` (position, pick, normal, material, glass; load+store; depth load+store; LessEqual; CCW; Back cull).
- Vertex input: depends on geometry source.
  - `procedural`: no vertex buffer; VS reads `@builtin(vertex_index)` and `@builtin(instance_index)`. Geometry is fully procedural in WGSL.
  - `mesh`: a vertex buffer (the asset's mesh vertices); VS reads vertex attributes + instance_index.
- Draw call: `draw_indexed_indirect` for the procedural path (with an engine-baked unit index buffer per `vertex_count_per_spawn`) or `draw_indexed_indirect` against the asset's index buffer for `mesh`. Indirect args were written during step 6.
- VS receives `(AnchorContext, spawn_idx, vid, FrameContext)` via splice — engine wraps the user's `vs` function with an entry point that does the instance-record deref + uniform binds.
- FS writes the same packed G-buffer format as `mesh_proxy.wesl`. Pick attachment writes `USER_SHADER_PICK_SENTINEL` unconditionally (load-bearing).

#### Step 9 — user_shader shadow VS

- One draw per active shader per cascade. Same indirect-draw args reused (they don't depend on viewport).
- Vertex-shader entry runs the user's `vs` to get world position, transforms by cascade's view-proj instead of main camera's, depth-only output.
- If the user wants a different shadow VS (e.g., simplified animation), manifest can declare `@shadow_vs custom`; default is "same VS, depth-only output." V1 ships the default only.

### Spawn-count caching

Two modes, declared in manifest:

- **`@spawn_count_cache static`** (default) — `spawn_count` and the fill pass run only when `paint_epoch` or `geometry_epoch` changes. Output instance buffer persists across frames. The raster + shadow steps still run every frame, but the upstream three passes are skipped.
- **`@spawn_count_cache per_frame`** — spawn_count + prefix-sum + fill run every frame. Required if `spawn_count` reads `frame.time` / `frame.camera_pos` (distance LOD, time-varying density).

Engine validates at composition time: if `@spawn_count_cache static` is declared but `spawn_count` references `FrameContext` fields, refuse to compile (build-time error).

### Composer integration

The shader composer's user-shader splice points change shape:
- Out: `compose_proto_chunk`, `compose_emit_chunk`, `compose_inst_world_matrix`, `compose_inst_aabb`, `compose_inst_to_local`.
- In: `compose_spawn_count`, `compose_vs`, `compose_spawn_alive` (optional), `compose_fs` (optional).

Each composed function gets wrapped in an engine-authored entry-point shell (compute or vertex/fragment) and emitted as a self-contained WGSL module per user-shader. One pipeline per shader per role (spawn_count compute, fill compute, raster, shadow).

The composer's existing infrastructure (lib_symbols, hash, parser) is largely reusable — the parts that go away are the proto-bake-specific splices, the band-cell BFS chunks, and the tile-bin chunks.

## Rust changes

| File | Change |
|------|--------|
| `crates/rkp-render/src/user_shader_mesh_pass.rs` (new) | The four-pass pipeline owner: spawn_count compute, prefix_sum, fill compute, raster. Borrows shape from `mesh_proxy_pass.rs` for the raster portion. Owns per-shader pipeline objects (one set per active user-shader material). |
| `crates/rkp-render/src/user_shader_mesh_shadow.rs` (new) | Shadow-VS variant of the raster pipeline; one per shader. Reused indirect args buffer. |
| `crates/rkp-render/src/shaders/user_shader_mesh.wesl` (new) | Hand-authored skeleton: WGSL structs (AnchorContext, FrameContext, VsOut, FsIn, FsOut), the engine-side helpers (instance-record deref, default FS), and the splice anchors for the user's functions. |
| `crates/rkp-render/src/shaders/user_shader_mesh_compute.wesl` (new) | Compute-pass skeletons for spawn_count + fill + (small) prefix-sum. |
| `crates/rkp-render/src/shader_composer/compose.rs` | New `compose_spawn_count`, `compose_vs`, `compose_spawn_alive`, `compose_fs` paths. Drop `compose_proto_chunk`, `compose_emit_chunk`, `compose_inst_*`. |
| `crates/rkp-render/src/shader_composer/parser.rs` | Parse new manifest directives: `@geometry`, `@spawn_count_cache`. Drop `@instance_proto`, `@max_emits_per_thread`, `@tile_size`, `@region_thickness`, `@max_depth`. |
| `crates/rkp-render/src/shader_composer/hash.rs` | Hash inputs change to match new splice points. |
| `crates/rkp-engine/src/render_worker/user_shader_tick.rs` | Replace `tick_emit_pass` / proto-bake bookkeeping with the new four-pass orchestration. Per active shader: collect anchors, upload to anchor buffer, dispatch the four passes (gated on paint-epoch for static-cache shaders), record draws. |
| `crates/rkp-engine/src/render_frame.rs` | Replace `painted_leaves: Vec<EmitLeaf>` with `user_shader_anchors: HashMap<ShaderId, Vec<AnchorRecord>>` (per-shader, since each shader sees only its own painted leaves). Drop `user_shader_emit_chunk`. |
| `crates/rkp-engine/src/engine/lifecycle.rs::scan_painted_aabbs` | Output per-shader anchor lists (not just a flat `EmitLeaf` Vec). Anchor record carries the new fields (`surface_area`, `leaf_extent`, `seed`, etc.). |
| `crates/rkp-render/src/rkp_renderer.rs` | Schedule the four-pass user-shader pipeline after `mesh_proxy_pass`. Per active shader: spawn_count → prefix_sum → fill → raster → (per cascade) shadow. |
| `crates/rkp-render/src/viewport_renderer.rs` | Per-VR bind groups for the user-shader raster (camera g0 + per-pass-uniforms). |
| `examples/shaders/grass.wgsl`, `splat5/assets/shaders/grass.wgsl` | Rewrite from scratch against the V1 API. Single VS computes blade geometry from `(spawn_idx, vid, anchor, frame)`. |
| **Retire** | `crates/rkp-render/src/user_shader_emit_pass.rs` (453 lines), `user_shader_tile_bin_pass.rs` (210), `user_shader_proto_pass.rs` + `user_shader_proto_pass/*.rs` (~853), `shaders/user_shader_emit.wesl` (179), `shaders/user_shader_tile_bin.wesl` (169), `shaders/user_shader_proto.wesl` (291), `shaders/user_shader_proto_rollup.wesl` (129), `shaders/shadow_scatter_emit.wesl`. Drop the band-cell scan + tile-list iteration from `octree_march.wesl` and `rkp_shadow_trace.wesl`. ~2.5k+ lines retire. |

## Validation

In order, each a before/after check:

1. **Empty scene smoke test** — engine boots, no user shaders active, no regressions on existing scenes (mesh raster + proxy + shadows still correct).
2. **Grass on a flat painted patch** (small region, ~100 anchors). Blades visible, animated with wind, cast shadows onto the host surface and onto themselves.
3. **Grass on a large painted region** (~10k anchors). Density stays consistent — boundary anchors don't over- or under-emit relative to interior anchors.
4. **Grass on a sloped / curved surface.** Anchors on different surface normals get different spawn counts (via `surface_area`); blades still oriented correctly via `anchor.surface_normal`.
5. **Non-convex paint shape** (L's, rings, isolated patches). Grass follows paint exactly — equivalent to V1.1's per-anchor host-material probe.
6. **Multiple user-shader materials in one scene** (e.g., grass + scattered rocks). Each shader runs its own four-pass pipeline; per-shader anchor sets correctly partition by material.
7. **Static vs per-frame caching.** Static-cache grass: spawn_count pass runs only on paint changes (verify via GPU timestamps). Per-frame cache: runs every frame.
8. **Mesh-source user shader.** Author a "scattered rocks" shader using `@geometry mesh { asset: "rock.glb" }`. Verify mesh vertex data flows through VS correctly; rocks scatter across painted region.
9. **CSM shadows on user-shader output.** All four cascades produce blade shadows; transitions are clean.
10. **Picking.** Click on a blade — pick buffer returns `USER_SHADER_PICK_SENTINEL`, host entity underneath is selected instead.
11. **Perf regression check.** With ~10k grass instances, total user-shader cost should beat the current path's ~53 ms / 1k instance ceiling (`project_user_shader_emit_rebuild_2026_05_05`). Hardware-instanced VS is the expected big win.

## Risks / open questions

- **Indirect-arg layout / draw type for procedural geometry.** Indexed draws need an index buffer. For procedural geometry without an explicit index buffer, two options: (a) engine bakes a "unit index buffer" `[0, 1, 2, ..., N-1]` of length = `index_count_per_spawn`, instanced via `instance_index` mapping; (b) use non-indexed `draw_indirect` and let VS compute everything from vertex_index. Option (a) is friendlier to triangle-strip-ish topologies; (b) is simpler. Decide during implementation; lean toward (b).
- **Spawn-record vs reverse-lookup.** Materializing per-spawn records (anchor_idx + spawn_idx) is one extra compute pass + memory but gives O(1) VS lookup. Alternative: skip the fill pass, VS does binary search on the offset[] prefix-sum array. For V1, materialize — simpler and the per-spawn buffer is small (8 B × total_spawns).
- **Anchor partitioning by shader.** Current `scan_painted_aabbs` produces a flat `Vec<EmitLeaf>` across all shaders. V1 needs per-shader anchor lists. Either filter at collection time (one pass per shader, sparse) or one collection pass that bins into per-shader Vecs. Latter is probably right.
- **`AnchorContext.surface_area` definition.** `leaf_size² × |dot(host_normal, up)|` works for flat-ish surfaces; on highly curved surfaces it under-counts. May need to project leaf footprint along an axis the user can pick. Start simple; iterate if grass density looks wrong on curves.
- **Max anchors per shader.** With per-shader anchor partitioning, a heavily painted scene could exceed reasonable prefix-sum-pass limits. V1: cap at ~64k anchors per shader, warn (eprintln) if exceeded. Multi-pass prefix sum is a follow-up.
- **Composer error surface.** New manifest + 3-function API is a smaller surface than the old 4-hook + many-directive API, but unknown-WGSL-feature support gaps may surface. Try grass first.
- **Animation determinism for shadow VS.** If `vs` reads `frame.time`, the shadow VS must use the same time value (no temporal jitter between camera and shadow). Engine binds the same `FrameContext` for both — built-in.
- **Static-cache invalidation when params change.** A `@spawn_count_cache static` shader whose `spawn_count` reads a `@param` (e.g., density slider) needs to invalidate the cache on param change. Add the param-epoch to the cache key alongside paint-epoch + geometry-epoch.
- **Backwards compat for `grass.wgsl`.** Existing demo grass shader uses the old API. Rewrite it as part of V1; don't try to maintain the old API behind a flag. The two existing copies (`examples/shaders/grass.wgsl`, `splat5/assets/shaders/grass.wgsl`) update together.

## What gets retired

Confirmed retirements once V1 lands and the rewritten `grass.wgsl` validates:

- **Rust files (~1775 lines):** `user_shader_emit_pass.rs`, `user_shader_tile_bin_pass.rs`, `user_shader_proto_pass.rs` + `user_shader_proto_pass/{cache,pass,types,tests}.rs`.
- **WGSL files (~770 lines):** `shaders/user_shader_emit.wesl`, `shaders/user_shader_tile_bin.wesl`, `shaders/user_shader_proto.wesl`, `shaders/user_shader_proto_rollup.wesl`, `shaders/shadow_scatter_emit.wesl`.
- **In-place deletions:** band-cell descend + tile-list iteration in `octree_march.wesl`, `rkp_shadow_trace.wesl`. The `OCTREE_BAND_BIT` (bit 29) frees up.
- **Engine plumbing:** `RenderFrame.painted_leaves`, `RenderFrame.user_shader_emit_chunk`, the `EmitLeaf` type, the proto-bake cache fields on `EngineState`, the `tick_emit_pass` orchestration in `user_shader_tick.rs` (replaced).
- **Composer splices:** `compose_proto_chunk`, `compose_emit_chunk`, `compose_inst_world_matrix`, `compose_inst_aabb`, `compose_inst_to_local`, `splice_emit_chunks`.
- **Manifest directives:** `@instance_proto`, `@max_emits_per_thread`, `@max_depth` (region), `@tile_size`, `@region_thickness`.

Total: ~2.5k lines deleted; net engine code goes down even after the new pipeline lands.

## After V1

Follow-ups in priority order, each its own session:

- **Multi-topology per material.** `@geometry mesh_library { assets: [...] }`; per-spawn `which_mesh` index selectable in `spawn_count` or `spawn_alive` output.
- **Glass on user-shader output.** Route glass-material user shaders through `mesh_glass` instead of opaque. Mixed glass/opaque per shader is still voxel-bake-only.
- **Additional anchor sources.** Mesh-surface points (poisson on a triangle mesh); ECS-driven point lists; GPU compute scatter.
- **Parent skinning.** Leaves-on-a-tree-branch — anchors derived per-frame from a host mesh's bone transforms. The big V2 architectural extension.
- **Compute-driven emission.** User-authored compute pass writing arbitrary vertex buffers + indirect args. Cloth, fluid surfaces, sim-coupled effects.
- **LOD authoring helpers.** `spawn_count` can already do distance-based LOD via `frame.camera_pos`, but a built-in helper + per-LOD vertex-count override would be ergonomic.
- **March-path retirement.** Once user shaders + procedural + paint + glass all run cleanly on the mesh path, retire the march. Separate doc.
