# Migration Plan: Octree Raymarch → Marching Cubes + Triangle Rasterization

## Why

Two days of trying to make octree raymarching scale past ~10M voxels demonstrated
a fundamental ceiling. The march bottleneck is GPU memory bandwidth + SIMT
divergence on scattered access — not something solvable with code-level tweaks
at the per-thread level. Specifically:

- Stack-based traversal failed (private-array dynamic indexing spilled to local
  memory, 2× slower).
- Single-leaf cache failed (samples at ±vs/2 always straddle leaf boundaries).
- Intermediate-ancestor cache failed (SIMT divergence ate the wins).
- DAG subtree compression and free-list reuse helped (~25-65% on data size),
  but the algorithmic cost per pixel remained.

The current marcher is doing per-pixel, per-frame **isosurface extraction**
through a scattered tree — exactly the operation triangle rasterization was
invented to make obsolete. It's also vestigial: the gradient-normal /
opacity-field-marching approach is from when this was a splat engine, and
splats themselves were abandoned for failing on quality + perf. Continuing to
raymarch the opacity field carries that history forward without serving any
remaining requirement.

Transparent voxels are a non-issue: a single material in the entire palette
(`glass.rkmat`, opacity 0.3) is non-opaque, and it's rare in real scenes. The
march's accumulation code path is essentially dead code.

## Architecture Summary

**What stays:**
- Voxels as authoritative representation
- Octree on CPU as authoring/spatial structure (also drives mesh extraction)
- Voxelization pipeline (procedural, .rkp load, primitives)
- All downstream rendering (G-buffer shade, SSAO, shadows, GI, volumetrics, post)
- G-buffer format

**What changes:**
- After voxelization (and the existing compact + DAG passes), extract a triangle
  mesh via marching cubes
- GPU consumes triangle vertex+index buffers, not octree+voxel_pool buffers
- Replace `octree_march.wgsl` compute pass with a vertex+fragment G-buffer fill
  pass

**What gets removed (Phase 4):**
- `octree_march.wgsl` and `octree_march.rs`
- GPU buffers: `octree_nodes`, `voxel_pool`, `color_pool` (CPU copies stay)
- Stats infrastructure for octree lookups

## Data Model

```rust
// Per-object mesh (lives alongside the octree handle, NOT replacing it)
pub struct ExtractedMesh {
    pub positions: Vec<[f32; 3]>,   // local-space, in octree-local coords
    pub normals: Vec<[f32; 3]>,     // unit, derived from opacity gradient
    pub colors: Vec<u32>,           // packed R8G8B8A8
    pub material_ids: Vec<u16>,     // primary material per vertex
    pub indices: Vec<u32>,          // 3 per triangle
}

// Stored per-object in the scene manager, alongside SpatialHandle
struct ObjectGeometry {
    spatial: SpatialHandle,         // existing octree handle
    mesh_handle: Option<MeshHandle>, // new — handle into mesh GPU pool
}
```

## Phase 1 — Triangles on screen, debug-colored

**Goal:** prove the pipeline end-to-end. One object renders as triangles.
Positions correct, normals fake, colors debug.

### New module

`crates/rkp-core/src/marching_cubes.rs` (collocate with octree, no new crate yet):

```rust
/// Extract a triangle mesh from a sparse octree's opacity field.
///
/// `threshold` defines the isosurface (typically 0.5).
/// Output positions are in octree-local space [0, extent).
pub fn extract_mesh(
    octree: &SparseOctree,
    pool: &VoxelPool,
    threshold: f32,
) -> ExtractedMesh;

// Internal: standard marching cubes tables
const MC_EDGE_TABLE: [u16; 256] = [...];      // which 12 edges have crossings per cube config
const MC_TRI_TABLE: [[i8; 16]; 256] = [...];  // triangle list per cube config

// Internal: process one cell (8 corner samples → triangles)
fn process_cell(corners: [f32; 8], cell_min: Vec3, cell_size: f32, threshold: f32) -> Vec<Triangle>;
```

### Extraction algorithm (Phase 1)

Naive but correct first:

```rust
pub fn extract_mesh(octree, pool, threshold) -> ExtractedMesh {
    // 1. Collect (coord, opacity) for every leaf via octree.iter_leaves().
    //    Build a HashMap<UVec3, f32> for fast neighbor lookup.
    let opacity_grid: HashMap<UVec3, f32> = octree.iter_leaves()
        .map(|(coord, slot, _depth)| (coord, pool.get(slot).opacity_f32()))
        .collect();

    // 2. Find cells to process: any (x,y,z) where at least one of the 8 corners
    //    is in opacity_grid. Use a HashSet to dedupe.
    let active_cells = find_active_cells(&opacity_grid);

    // 3. For each active cell, sample 8 corners (default to 0.0 if missing,
    //    or 1.0 if INTERIOR — query octree.lookup for full classification).
    //    Run MC on the corner values.
    let mut out = ExtractedMesh::default();
    for cell_coord in active_cells {
        let corners = sample_8_corners(octree, pool, cell_coord);
        emit_triangles(&mut out, cell_coord, corners, threshold);
    }
    out
}
```

**Don't share vertices yet.** Three vertices per triangle, even if duplicates.
Optimizing comes later (Phase 5).

### New GPU pass

`crates/rkp-render/src/triangle_gbuffer.rs`:

```rust
pub struct TriangleGBufferPass {
    pipeline: wgpu::RenderPipeline,
    // per-object vertex/index buffers managed by a mesh pool
}

impl TriangleGBufferPass {
    pub fn new(device, gbuffer_format, depth_format) -> Self;
    pub fn dispatch(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        gbuffer: &GBuffer,
        meshes: &[(GpuMesh, ObjectIndex)],
        camera: &CameraUniforms,
    );
}
```

`crates/rkp-render/src/shaders/triangle_gbuffer.wgsl`:

```wgsl
// Vertex shader
struct VertexIn {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: u32,
    @location(3) material_id: u32,
}
struct VertexOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) color: vec3<f32>,
    @location(3) @interpolate(flat) material_id: u32,
}

@vertex fn vs_main(in: VertexIn, @builtin(instance_index) inst: u32) -> VertexOut {
    let obj = objects[inst];
    let world_pos = (obj.world * vec4<f32>(in.position, 1.0)).xyz;
    let world_normal = normalize((obj.world * vec4<f32>(in.normal, 0.0)).xyz);
    var out: VertexOut;
    out.clip_pos = camera.view_proj * vec4<f32>(world_pos, 1.0);
    out.world_pos = world_pos;
    out.world_normal = world_normal;
    out.color = unpack_color(in.color);
    out.material_id = in.material_id;
    return out;
}

// Fragment shader — writes G-buffer
@fragment fn fs_main(in: VertexOut) -> GBufferOutput {
    var out: GBufferOutput;
    out.position = vec4<f32>(in.world_pos, length(in.world_pos - camera.position));
    out.normal = vec4<f32>(in.world_normal, 1.0);
    out.material = vec2<u32>(
        in.material_id,
        // Pack object_id, color in second channel as before
        ...
    );
    return out;
}
```

### Mesh GPU pool

`crates/rkp-render/src/mesh_pool.rs`:

```rust
pub struct MeshPool {
    vertex_buffer: wgpu::Buffer,  // grows on demand, packed across all meshes
    index_buffer: wgpu::Buffer,
    allocations: Vec<MeshAllocation>,  // tracks ranges per mesh
    // Free-list reuse pattern, matching VoxelPool
}

pub struct MeshHandle { vertex_range: (u32, u32), index_range: (u32, u32) }

impl MeshPool {
    pub fn upload(&mut self, mesh: &ExtractedMesh) -> MeshHandle;
    pub fn deallocate(&mut self, handle: MeshHandle);
}
```

### Wire up in engine

`crates/rkp-engine/src/engine.rs` — at the end of voxelize_opacity_fn / load_rkp / etc:

```rust
// After voxelization completes:
let mesh = rkp_core::marching_cubes::extract_mesh(&octree, &pool, 0.5);
let mesh_handle = self.scene_mgr.mesh_pool.upload(&mesh);
// Store mesh_handle on the entity's Renderable component
```

### Render flow change

`crates/rkp-render/src/rkp_renderer.rs::render`:

```rust
// OLD: self.march.dispatch(...)   ← remove (or guard behind a feature flag during transition)
// NEW:
let q = self.profiler.begin_query("triangle_gbuffer", encoder);
self.triangle_gbuffer.dispatch(encoder, &self.gbuffer, &meshes, camera);
self.profiler.end_query(encoder, q);
// downstream passes (ssao, shade, etc.) unchanged
```

**For Phase 1 specifically:** keep the march pass available for objects that
don't have a mesh yet. Render meshes for objects that do. This lets you A/B
test by switching one object at a time.

### Phase 1 acceptance

- One scene object renders as triangles
- Approximately the right shape (compared to march)
- Lit by existing shade pass with debug-color albedo
- No crashes, no missing geometry
- ~1-3ms render time for typical object

---

## Phase 2 — Real normals from opacity gradient

**Goal:** smooth shading matching the current renderer.

### Add to MC extraction

For each generated MC vertex (which lies on an edge between two voxels),
compute the opacity-field gradient and store as the vertex normal.

```rust
fn vertex_normal_from_gradient(
    pos: Vec3,
    octree: &SparseOctree,
    pool: &VoxelPool,
    voxel_size: f32,
) -> Vec3 {
    // Same math the shader was doing, but on CPU and only once per vertex.
    // Sample opacity at pos ± vs/2 along each axis, central difference.
    let h = voxel_size * 0.5;
    let gx = sample_opacity_at(pos + vec3(h,0,0), octree, pool)
           - sample_opacity_at(pos - vec3(h,0,0), octree, pool);
    let gy = sample_opacity_at(pos + vec3(0,h,0), octree, pool)
           - sample_opacity_at(pos - vec3(0,h,0), octree, pool);
    let gz = sample_opacity_at(pos + vec3(0,0,h), octree, pool)
           - sample_opacity_at(pos - vec3(0,0,h), octree, pool);
    let grad = vec3(gx, gy, gz);
    if grad.length() < 1e-8 { return Vec3::Y; }
    -grad.normalize()
}
```

### Phase 2 acceptance

- Normals interpolate smoothly across triangles
- Visual match (within reason) with the previous raymarch's shading
- Side-by-side test: render one object with march, one with mesh — should look
  nearly identical when lit

---

## Phase 3 — Real materials and colors

**Goal:** preserve voxel color and material data on the mesh.

### Per-vertex color

For each MC vertex on edge between voxel A and B:
- Vertex position is along the edge at parameter `t = (threshold - opacity_A) / (opacity_B - opacity_A)`
- Interpolate color: `color = lerp(color_A, color_B, t)` (in linear space, then pack)

### Per-vertex material

Two strategies, increasing complexity:

**Simple (Phase 3):** pick the material from the side with `opacity > threshold`.
Single mat per vertex.

**Full (Phase 3.5):** dual-material blending. Vertex stores
`(primary_mat, secondary_mat, blend_weight)`. Fragment shader looks up both,
blends. Adds 6 bytes per vertex.

### Glass / transparent rendering — DEFERRED

Proper forward transparency in a deferred pipeline is significant work
(separate pass, back-to-front sort, HDR scene-color readback, fragment
does PBR inline) for rendering a single rare material (`glass.rkmat`,
opacity 0.3 — per CLAUDE.md, "the march's accumulation code path is
essentially dead code"). The current shade pass doesn't reference
`mat.opacity` at all, so even the march renders glass as opaque today.
Revisit only when an actual glass scene surfaces.

### Phase 3 acceptance

- Voxel colors visible on triangle surface ✓
- Materials look correct (PBR shading with right roughness/metallic) ✓
- Dual-material blending between primary + secondary materials ✓
- Glass renders as semi-transparent — DEFERRED (see above)
- Visual parity with the old march in test scenes ✓

---

## Phase 4 — Remove the raymarch entirely

**Goal:** delete dead code, reclaim GPU memory.

### Files to delete

- `crates/rkp-render/src/octree_march.rs`
- `crates/rkp-render/src/shaders/octree_march.wgsl`

### Files to modify

- `rkp_renderer.rs` — remove `march` field and all references
- `rkp_scene.rs` — remove `voxel_pool_buffer`, `octree_nodes_buffer`,
  `color_pool_buffer` and their bind group entries
- `rkp_scene_manager.rs` — remove `geometry_upload()` (or change it to upload
  meshes only)
- `engine.rs` — remove the upload of voxel pool / octree to GPU. CPU-side
  voxel_pool / octree stay.

### What about voxel_pool / octree on CPU?

**Keep them.** They're authoring data:
- Voxel/material edits modify them
- `emit_faces` (if still used) reads them
- They drive mesh re-extraction
- Saving to .rkp serializes them

The **GPU** doesn't need them. CPU does.

### Phase 4 acceptance

- Engine compiles and runs with no octree GPU buffers
- All test scenes render correctly
- GPU memory usage drops by 100+ MB on heavy scenes
- `cargo test` passes

---

## Phase 5 — Quality and perf upgrades (separate project, optional)

Listed for completeness:

1. **Vertex sharing.** Marching cubes naturally produces shared vertices along
   edges. Use a `HashMap<EdgeId, VertexIdx>` during extraction to dedupe. Cuts
   vertex count ~3×.

2. **Dual contouring or extended MC.** Preserves sharp corners on procedural
   primitives (boxes, ramps). Required if you care about clean cube edges.

3. **Async extraction.** Extract on a worker thread, hand off to GPU when ready.
   Hides latency during edits.

4. **LOD.** Extract multiple resolutions, swap by distance. Requires LOD octrees
   too.

5. **Mesh shaders.** Modern GPUs (DX12 Ultimate / Vulkan mesh shaders) support
   meshlet rendering, even faster. wgpu may or may not expose this.

---

## File / Module Inventory

### New files (Phase 1-3)

```
crates/rkp-core/src/marching_cubes.rs               (~600 lines)
  - MC tables (256-entry edge + tri tables)
  - Cell processor
  - Public extract_mesh()

crates/rkp-render/src/triangle_gbuffer.rs           (~300 lines)
  - TriangleGBufferPass struct
  - Pipeline setup
  - Per-frame dispatch

crates/rkp-render/src/shaders/triangle_gbuffer.wgsl (~80 lines)
  - Vertex + fragment shader writing G-buffer

crates/rkp-render/src/mesh_pool.rs                  (~200 lines)
  - GPU vertex/index buffer pool with free-list reuse
```

### Modified files

```
crates/rkp-render/src/rkp_renderer.rs       — add triangle pass, remove march
crates/rkp-render/src/rkp_scene_manager.rs  — manage per-object meshes
crates/rkp-render/src/rkp_scene.rs          — drop voxel-related GPU buffers (Phase 4)
crates/rkp-engine/src/engine.rs             — invoke MC extraction after voxelization
crates/rkp-engine/src/components.rs         — Renderable now stores MeshHandle
```

### Eventually deleted (Phase 4)

```
crates/rkp-render/src/octree_march.rs
crates/rkp-render/src/shaders/octree_march.wgsl
```

---

## Risks and mitigations

1. **MC produces too many triangles.**
   - Mitigation: Phase 5 vertex sharing typically cuts 3×; also LOD.
   - Worst case: 10-20M triangles per heavy object. ~200-400 MB. Modern GPU
     handles fine.

2. **Sharp features get rounded by classic MC.**
   - Mitigation: Phase 5 dual contouring.
   - For Phase 1-3, accept the rounded look — it'll be obvious on procedural
     cubes but acceptable for a first version.

3. **MC extraction is slow during edits.**
   - First implementation: 100-500ms for heavy scene.
   - Mitigation: async extraction (Phase 5), or only re-extract on commit (not
     on slider drag).

4. **Variable-depth leaves in the octree.**
   - The octree has leaves at multiple depths (LOD). MC needs to handle a
     coarse leaf adjacent to fine leaves consistently.
   - Mitigation: at extraction time, "expand" coarse leaves into fine virtual
     cells (still pointing to the same opacity value). Costs no extra memory,
     just processing time.

5. **Multi-material handling at boundaries.**
   - When two voxels with different materials meet, MC vertex needs to pick or
     blend.
   - Phase 3 simple: take the dominant material.
   - Phase 3.5 full: dual-material blending preserved.

6. **Existing `emit_faces` system.**
   - There's an existing face-emission system for what looks like instanced
     cube rendering. Need to figure out if that's used or vestigial; it may be
     the path being replaced here, or it may serve a different purpose.
   - Investigate in Phase 1.

7. **Skinning / animation.**
   - The CLAUDE.md mentions "Animation needs splat deform" as future work.
     Triangle meshes need vertex skinning (standard).
   - Not blocking for Phase 1-4. Will need vertex bone weights eventually.

---

## Suggested execution order

**Day 1 — Phase 1 partial:** MC tables + naive extraction + simple Rust harness
that prints triangle counts. Verify correctness on a single sphere/cube octree
(no GPU yet). Acceptance: extracted mesh has the expected topology.

**Day 2 — Phase 1 GPU:** triangle pass, mesh pool, vertex+fragment shader,
render one object alongside the existing march. Visually compare shapes.
Acceptance: shape recognizable, position match within a voxel.

**Day 3 — Phase 2:** add gradient normals at extraction. Verify shading matches
march output. Acceptance: side-by-side visual, lit objects look the same.

**Day 4 — Phase 3:** color, materials. Glass via second pass. Acceptance: full
visual parity with the march.

**Day 5 — Phase 4:** delete octree_march.wgsl, drop voxel GPU buffers, audit
`emit_faces` and remove if unused. Acceptance: build clean, all scenes render
correctly, GPU memory dropped meaningfully.

---

## Performance expectations

| Scene | Current (raymarch) | Expected (triangles) |
|---|---|---|
| Light (~1M leaves) | 4-8ms march + ~1ms post = 5-9ms | <1ms render + post = ~1-2ms |
| Heavy (10M+ leaves) | 30-40ms march + post-cliff slowdown = 50-70ms | 2-5ms render + ~1ms post = 3-6ms |
| Extraction time (CPU) | n/a | 50-200ms per heavy object, once per voxelization |

---

## Open design decisions to make in the fresh session

1. **Threshold value for MC.** Currently 0.05 in the shader (OPACITY_THRESHOLD).
   Standard MC uses 0.5. Probably should switch to 0.5 to match industry
   conventions and have clean half-voxel surfaces.

2. **Vertex format.** Suggested: `pos (12B) + normal (12B) + color (4B) +
   material_id (4B) = 32B/vertex`. Could pack normal as 4B (snorm10x3) to save
   8B. Phase 3 question.

3. **One mesh per object vs unified mesh buffer.** Suggested: unified pool with
   per-object ranges (matches the existing octree allocator pattern).

4. **Mesh re-extraction trigger.** Suggested: piggyback on `geometry_dirty`
   flag — if dirty, re-extract.

5. **Glass rendering pass placement.** Before or after volumetrics? Probably
   after, so the glass overlay sits in front of fog.

6. **Coordinate space.** Mesh positions in object-local space (transformed by
   world matrix in vertex shader), matches current GpuObject pattern.
