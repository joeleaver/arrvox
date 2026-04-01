# RKIPatch — Gaussian Splat Graphics Engine

## What This Is

A real-time graphics engine where **gaussian splats are the primary geometry representation**. Every object is a voxel grid of opacity data — the opacity field defines the geometry, and surface normals are derived from the field gradient at shade time. No signed distance fields, no mesh rasterization, no UV mapping.

This is a sibling project to [RKIField](../rkifield/) (SDF engine). Both share core infrastructure (editor UI, ECS, MCP, physics, animation) but have fundamentally different rendering pipelines.

## Origin

RKIPatch grew out of a proof-of-concept in RKIField that replaced SDF ray marching with gaussian splat accumulation. Key findings from the POC (reference code at `../rkipatch-poc-reference/`):

- **Opacity field as geometry.** Each voxel stores an f16 opacity value. The surface is where opacity crosses a threshold. Normals are derived from the gradient of the trilinearly-interpolated opacity field at shade time (6 taps).
- **Surface-finding march** through trilinear opacity field. Step through voxels, find where opacity crosses a threshold. Shade once at that point. No iterative SDF convergence.
- **Deferred rendering** — splat march writes to G-buffer (position, gradient normal, material). Shading is a separate pass, reusing rkf-render's full PBR shading stack.
- **VoxelSample reuse** — splat data reuses RKIField's VoxelSample format (8 bytes/voxel), reinterpreting the distance field as opacity. Same 4KB bricks, same streaming infrastructure, same dual-material system (2×u16 material IDs + u8 blend weight).
- **TAA opportunity** — splat accumulation is inherently low-frequency (weighted average). Per-frame jitter variation is much smaller than SDF ray marching. Temporal upscaling that failed for SDFs may work here.

## Architecture Goals

1. **Deferred rendering pipeline** — splat march writes G-buffer (position, gradient normal, material ID, motion vectors). Shading pass reuses rkf-render's PBR stack, shadows, GI, volumetrics — all for free.
2. **Per-voxel color** — mesh textures baked into per-voxel RGB during import. Stored in companion color pool (same infrastructure as RKIField).
3. **Gradient-derived normals** — surface normals computed from the opacity field gradient during the march pass, written to G-buffer. Conventional `dot(normal, light_dir)` lighting via rkf-render's shading pass.
4. **New .rkp file format** — splat-native storage: opacity per voxel, per-voxel color, multi-LOD, LZ4 compressed.
5. **Mesh import pipeline** — .glb/.gltf/.obj/.fbx → BVH → SDF function → splat voxelization with opacity baking + texture color sampling.
6. **Shared infrastructure** — reuse RKIField's editor UI (rinch), ECS (hecs), MCP server, physics (Rapier), animation, asset streaming, material palette system.

## Tech Stack

| Component | Choice | Notes |
|-----------|--------|-------|
| Language | **Rust** | Entire codebase |
| GPU API | **wgpu** | WebGPU via wgpu crate |
| Shaders | **WGSL** | Compute-only (forward splat + post-process) |
| Windowing | **winit** | |
| Math | **glam** | f32 vectors, quaternions, matrices |
| ECS | **hecs** | From RKIField (shared crate) |
| Physics | **Rapier** | From RKIField (shared crate) |
| Editor UI | **rinch** | From RKIField (shared crate) |
| Mesh Import | **rkf-import** | From RKIField (shared crate) |
| Compression | **lz4_flex** | Brick data in .rkp files |

## Shared Crates (from RKIField)

These live in `../rkifield/crates/` and are referenced as path dependencies:

| Crate | What we reuse |
|-------|---------------|
| `rkf-core` | WorldPosition, brick pool, brick maps, spatial index, Aabb, BVH, material types, constants |
| `rkf-import` | Mesh loading (.glb/.gltf/.obj), BVH nearest-triangle, winding number, material transfer |
| `rkf-physics` | Rapier integration, collision adapter |
| `rkf-animation` | Skeletal animation, blend shapes |
| `rkf-mcp` | MCP server, tool registry, IPC bridge |

## New Crates (RKIPatch-specific)

```
rkipatch/
  crates/
    rkp-core/        — SplatVoxel wrapper over VoxelSample, opacity accessors, splat brick format
    rkp-render/      — Splat march pass (opacity field → G-buffer), pipeline orchestration using rkf-render
    rkp-convert/     — Mesh-to-.rkp CLI (BVH + opacity baking + color transfer)
    rkp-runtime/     — Frame scheduling, ECS glue, streaming
    rkp-editor/      — Editor binary (reuses rinch UI from rkifield)
    rkp-testbed/     — (removed — all visual work done in rkp-editor)
```

## Key Data Types

```rust
// SplatVoxel — zero-cost wrapper over rkf-core's VoxelSample (8 bytes)
// Reinterprets the SDF distance field as opacity.
//
// word0: f16 opacity (bits 0-15) | blend_weight u8 (bits 16-23) | reserved (bits 24-31)
// word1: primary material_id u16 (bits 0-15) | secondary material_id u16 (bits 16-31)
//
// Provides .opacity()/.set_opacity() accessors.
// From<VoxelSample> / Into<VoxelSample> for zero-cost conversion.
// Normals derived from gradient of trilinearly-interpolated opacity field at shade time.

// Per-voxel color: stored in companion ColorBrick pool (same as RKIField)
// ColorVoxel { packed: u32 } — R|G|B|intensity, 4 bytes

// Materials: uses rkf-core's MaterialPalette system unchanged.
// 16-bit material IDs, dual-material blending per voxel, .rkmat files.
```

## Render Pipeline (deferred, maximizing rkf-render reuse)

Pure splat pipeline — no mixed SDF/splat scenes. Deferred shading reuses rkf-render's
full PBR stack.

```
1. Update transforms → flatten scene hierarchy → upload GpuObject metadata  [reuse rkf-render]
2. BVH refit → upload GPU BVH nodes                                        [reuse rkf-render]
3. Tile-based object culling → per-tile object lists                        [reuse rkf-render]
4. Splat March → per-pixel: find surface in trilinear opacity field,        [NEW — replaces ray_march]
                 compute gradient normal (6-tap), write G-buffer
                 (position, normal, material, motion vectors)
5. Shadow / AO pass                                                         [adapt from rkf-render]
6. GI — radiance injection + mip                                            [adapt from rkf-render]
7. Deferred shading — PBR, reads G-buffer + shadows + GI                    [reuse rkf-render]
8. Volumetrics (fog, god rays, clouds)                                      [reuse rkf-render]
9. Post-process (bloom, tone map, DoF, motion blur, color grade)            [reuse rkf-render]
10. TAA / temporal upscale                                                  [reuse rkf-render]
11. Present                                                                 [reuse rkf-render]
```

The only new pass is step 4 — the splat march that replaces rkf-render's `ray_march.rs`.
Everything downstream (shading, shadows, GI, post-process) reuses rkf-render code
directly, since the G-buffer format is compatible.

## POC Reference

The proof-of-concept code is at `../rkipatch-poc-reference/`. Note: the POC used L1 SH
coefficients which have been dropped in favor of gradient-derived normals. The POC is
useful as reference for the march structure and compositing, not the data format.

- `splat.rs` — (historical) SH coefficient computation, snorm10 packing — superseded by gradient normals
- `voxelize_splat.rs` — SDF→splat voxelization — opacity baking logic still relevant
- `shaders/splat_march.wgsl` — surface-finding march structure still relevant, SH eval replaced by gradient normal
- `shaders/splat_composite.wgsl` — alpha compositing over deferred background — still relevant

## Critical Rules

1. **No SDF ray marching for splat objects.** Splats use surface-finding through the opacity field, not iterative sphere tracing.
2. **Deferred rendering.** Splat march writes G-buffer (position, gradient normal, material, motion vectors). Shading is a separate pass reusing rkf-render's PBR stack.
3. **Gradient-derived normals.** No stored normals or SH coefficients. Surface normals come from the gradient of the trilinearly-interpolated opacity field (6-tap central differences), computed during the march and written to G-buffer.
4. **Trilinear interpolation of the opacity field.** Never nearest-neighbor — creates grid artifacts. The trilinear field IS the smooth representation.
5. **G-buffer compatible with rkf-render.** Same format so all downstream passes (shading, shadows, GI, post-process) work unchanged.
6. **SplatVoxel wraps VoxelSample.** Zero-cost wrapper — same 8-byte format, same brick pools. Opacity reinterprets the distance field. Materials use rkf-core's palette system unchanged.
7. **WorldPosition everywhere.** Same as RKIField — never raw Vec3 for world-space positions.
7. **Test-driven development.** Write tests first, same as RKIField.
8. **MCP-native.** Every feature ships with MCP tools. If MCP is broken, fix it first.
9. **Ask questions, don't assume.** Same as RKIField — stop and ask when requirements are ambiguous.
10. **We value correctness over speed of implementation.** In any choice between the "simple way" and the "correct way", the correct way always wins. We don't take shortcuts. We don't defer dificult implementations.

## Build Commands

```bash
cargo build --workspace          # build everything
cargo test --workspace           # run all tests
cargo clippy --workspace         # lint
cargo run -p rkp-editor          # editor (primary development target)
cargo run -p rkp-convert -- input.glb -o output.rkp  # asset conversion
```
