# RKIPatch — Surface-Mesh Voxel Graphics Engine

## What This Is

A real-time graphics engine where **every object is a sparse octree of surface voxels** that gets meshed (via per-cluster Karis-Nanite-style surface nets + DAG LOD) into triangles for rendering. A leaf in the tree IS a point on the surface; the tree's existence defines the geometry. Each leaf carries a prefiltered octahedrally-packed normal + material references — everything needed to shade it in a single 8-byte read.

There is no per-voxel opacity field. Transparency is a per-material property. Surfaces are defined by leaf existence in the octree, not by an isosurface of a scalar field.

This is a sibling project to [RKIField](../rkifield/) (SDF engine). Both share core infrastructure (editor UI, ECS, MCP, physics, animation) but have fundamentally different rendering pipelines.

## Origin and where we are now

RKIPatch began as a splat-accumulation rendering experiment. Over time the design converged through several pivots:

1. **Splat-accumulation rendering** — original framing, retired.
2. **Per-pixel opacity-field surface march** — compute marcher descended the octree per pixel. Worked, but fragment-bound on dense scenes.
3. **Splat-rasterizer prototype** — proved 3.9× faster than the marcher on single assets but slower on real scenes. Never shipped past A/B.
4. **Surface-mesh primary visibility (current)** — assets bake to triangle meshes with cluster-DAG LOD; the renderer rasterizes triangles. Mesh runs at ~4.5 ms vs. march's ~6 ms in dense scenes, with full feature parity (glass, paint, sculpt, CSM shadows, user-shaders).

The project name still says "patch" from the original splat framing; the architecture is now a sparse octree of prefiltered surface voxels → meshed via surface-nets → cluster-DAG LOD → forward triangle rasterization.

### Key invariants of the current architecture

- **Surface defined by sparse octree leaves.** The octree-build classifier uses the 1-Lipschitz property of a signed distance function (sampled from the primitive / mesh) to decide Empty / Interior / Mixed at each octree level.
- **Prefiltered per-leaf normals.** The SDF gradient at each voxel center is computed once during voxelization, octahedrally packed (2× snorm16 in a u32, <0.05° worst-case roundtrip error), and stored in `LeafAttr`.
- **Surface-mesh raster.** Imported `.rkp` assets ship with pre-baked triangle meshes + meshlet clusters + DAG groups (v5/v6 file format). The mesh raster pass writes only the visibility-buffer triplet (position, pick, leaf_slot); a compute resolve pass fills in normal/material/glass per pixel via the octree → leaf_attr indirection. Mesh raster + glass + shadow all skin in VS against per-frame bone palettes.
- **Procedural objects → proxy mesh.** Procedural primitives flatten to an SDF opcode stream and bake to a triangle proxy mesh via GPU surface-nets-from-SDF. No voxelization for procedurals.
- **Deferred rendering.** Mesh raster + resolve → G-buffer. Shading is a separate pass with full PBR + CSM shadows + atmosphere + volumetrics + post-process.
- **Dual-material blending.** `LeafAttr` carries primary material (u16), secondary material (u16), and blend weight — per-leaf material variation.

## Architecture Goals

1. **Deferred mesh rendering** — surface-mesh raster writes G-buffer; resolve compute fills the rest. Shading reuses rkf-render's full PBR stack.
2. **Per-voxel color** — mesh textures baked into per-voxel RGB during import. Stored in companion color pool (same infrastructure as RKIField).
3. **Prefiltered octahedral normals** — surface normals come from `LeafAttr.normal_oct`, computed once at voxelization and read at shade time. No gradient reconstruction in the hot path.
4. **`.rkp` file format** — LZ4-compressed sparse octree + brick pool + per-voxel color + bone weights + baked mesh + meshlet clusters + DAG groups (v6).
5. **Mesh import pipeline** — .glb/.gltf/.obj/.fbx → BVH → SDF function → octree voxelization → surface-nets meshing → cluster-DAG bake.
6. **Shared infrastructure** — reuse RKIField's editor UI (rinch), ECS (hecs), MCP server, physics (Rapier), animation, asset streaming, material palette system.

## Tech Stack

| Component | Choice | Notes |
|-----------|--------|-------|
| Language | **Rust** | Entire codebase |
| GPU API | **wgpu** | WebGPU via wgpu crate |
| Shaders | **WGSL/WESL** | Forward triangle raster + compute resolve + post-process |
| Windowing | **winit** | |
| Math | **glam** | f32 vectors, quaternions, matrices |
| ECS | **hecs** | From RKIField (shared crate) |
| Physics | **Rapier** | From RKIField (shared crate) |
| Editor UI | **rinch** | From RKIField (shared crate) |
| Mesh Import | **rkp-import** | Own crate (`crates/rkp-import`) |
| Compression | **lz4_flex** | Section data in .rkp files |

## Shared Crates (from RKIField)

These live in `../rkifield/crates/` and are referenced as path dependencies:

| Crate | What we reuse |
|-------|---------------|
| `rkf-core` | WorldPosition, brick pool, brick maps, spatial index, Aabb, BVH, material types, constants |
| `rkf-physics` | Rapier integration, collision adapter |
| `rkf-animation` | Skeletal animation, blend shapes, `SkeletonAsset` + `save_rkskel` (used by `rkp-import` for the `.rkskel` sidecar) |
| `rkf-mcp` | MCP server, tool registry, IPC bridge |

## New Crates (RKIPatch-specific)

```
rkipatch/
  crates/
    rkp-core/        — sparse octree, brick pool, LeafAttr, mesh extract, asset_file (.rkp v6)
    rkp-render/      — surface-mesh raster + resolve + shading + post-process pipeline
    rkp-import/      — Mesh → .rkp import pipeline: mesh loaders (glTF/OBJ/FBX),
                       triangle BVH + winding number, octree voxelization,
                       surface-nets meshing + cluster-DAG bake,
                       skeleton extraction + .rkskel sidecar,
                       structured progress events (ProgressReporter / ImportEvent).
    rkp-convert/     — Thin CLI over rkp-import for headless / CI asset bakes
    rkp-runtime/     — Frame scheduling, ECS glue, streaming
    rkp-editor/      — Editor binary (reuses rinch UI from rkifield)
    rkp-engine/      — Engine-side state, command handling, import worker thread
    rkp-procedural/  — Procedural object nodes (sphere/box/union/etc.)
```

## Key Data Types

```rust
// LeafAttr — 8 bytes per surface voxel.
// word0: octahedrally-packed normal (u32; 2× snorm16, <0.05° error)
// word1: primary material_id u16 (bits 0-15) | secondary material_id u12 + blend_weight u4 (bits 16-31)
//
// Stored in the global `leaf_attr_pool` storage buffer; each octree
// leaf carries an index. Mesh raster reads it once per pixel during
// the resolve compute pass.

// MeshVertex — 32 bytes. Object-local position, packed normal,
// leaf_attr_id, bone weights/indices (for skinned assets).

// MeshletCluster — 64 bytes. Per-cluster index range, error metrics
// for Karis-Nanite admit, parent group reference for the DAG.

// Per-voxel color: stored in companion ColorBrick pool (same as RKIField)
// ColorVoxel { packed: u32 } — R|G|B|intensity, 4 bytes

// Materials: uses rkf-core's MaterialPalette system unchanged.
// 16-bit material IDs, dual-material blending per voxel, .rkmat files.
```

## Render Pipeline

Triangle mesh primary visibility, deferred shading. The build viewport
can swap the primary pass for a live procedural SDF raymarcher
(`BuildPreviewMode::Raymarch`) to preview unbaked edits.

```
1. Update transforms → flatten scene hierarchy → upload GpuObject metadata
2. BVH refit → upload GPU BVH nodes
3. Atmosphere LUTs (transmittance / multi-scatter / sky-view / aerial perspective)
4. Mesh LOD select compute — Karis-Nanite admit per cluster → DrawIndexedIndirectArgs
5. Mesh raster (or proc_raymarch in build-viewport preview mode)
   → G-buffer (position, pick, leaf_slot)
6. Mesh resolve compute → fills normal, material, glass from leaf_attr_pool
7. Proxy-mesh raster (procedural triangle meshes from surface-nets-from-SDF)
8. User-shader mesh raster (V1 grass/blade/instance shaders)
9. Mesh glass front/back raster + combine compute → gbuf_glass
10. Mesh shadow map render + blit compute → shadow_buffer (CSM, 4 cascades)
11. SSAO
12. Brush-state probe (paint cursor)
13. Deferred PBR shade (reads G-buffer + shadow_buffer + SSAO + atmosphere LUTs)
14. Volumetrics (fog, god rays, clouds)
15. Glass composite (Fresnel + Beer + screen-space refraction)
16. God rays
17. Bloom + bloom composite
18. Tone map
19. Wireframe / grid overlays
20. Present
```

## Critical Rules

1. **Mesh is the only render path.** No per-pixel ray-marching primary visibility, no splat raster, no `RKP_PRIMARY` env var.
2. **Deferred rendering.** Mesh raster writes a visibility buffer; resolve compute fills the rest; shade reads the unified G-buffer.
3. **Prefiltered octahedral normals.** No gradient reconstruction at shade time. The normal is in `LeafAttr.normal_oct`.
4. **`.rkp` v6 ships baked mesh + clusters + DAG.** Loading is a deserialize + relocate, not a re-extract. v5 files fall back to extract + DAG-build at load time.
5. **Procedurals bake to proxy mesh.** No voxelization of procedural objects — `BakeMode` is gone. If you need paintable/sculptable geometry, import a `.rkp`.
6. **G-buffer compatible with rkf-render.** Same format so all downstream passes (shading, shadows, post-process) work unchanged.
7. **WorldPosition everywhere.** Never raw Vec3 for world-space positions.
8. **Test-driven development.** Write tests first.
9. **MCP-native.** Every feature ships with MCP tools. If MCP is broken, fix it first.
10. **Ask questions, don't assume.** Stop and ask when requirements are ambiguous.
11. **We value correctness over speed of implementation.** In any choice between the "simple way" and the "correct way", the correct way always wins. We don't take shortcuts. We don't defer difficult implementations.

## Build Commands

```bash
cargo build --workspace          # build everything
cargo test --workspace           # run all tests
cargo clippy --workspace         # lint
cargo run -p rkp-editor          # editor (primary development target)
cargo run -p rkp-convert -- input.glb -o output.rkp  # asset conversion
```
