// Brick (4×4×4 voxel block) — geometry and traversal constants.
//
// A brick is the leaf-level DDA structure of the octree. Each brick
// holds 64 cells in a flat `array<u32>` indexed by
// `flat_idx = z*16 + y*4 + x`. Each cell value is either the
// EMPTY/INTERIOR sentinel below, or a leaf-attr slot id used to
// fetch normal + material from the per-asset attr pool.
//
// Brick face links live in a parallel `brick_face_links` buffer
// (`brick_id * 6 + face_idx`) and are read during DDA traversal
// when a step crosses a brick face — `FACE_*` indexes that buffer,
// and `FACE_INTERIOR`/`FACE_EMPTY_LINK` are the sentinel values
// stored there.

const BRICK_DIM: u32 = 4u;
const BRICK_DIM_F: f32 = 4.0;
const BRICK_CELLS: u32 = 64u;          // BRICK_DIM³
const BRICK_CELL_EMPTY: u32 = 0xFFFFFFFFu;
const BRICK_CELL_INTERIOR: u32 = 0xFFFFFFFDu;

// Iteration cap for in-brick DDA. 4096 covers any conceivable
// 4×4×4 walk; bumps when scenes grow extents need an audit.
const BRICK_MAX_STEPS: u32 = 4096u;

// Sentinel that marks a leaf-attr slot as unallocated in the
// per-asset attr pool. Same wire value as BRICK_CELL_EMPTY because
// both encode "no slot" for the u32 storage they share with.
const INTERNAL_ATTR_NONE: u32 = 0xFFFFFFFFu;

// Brick-face neighbor link sentinels (stored per-face in
// `brick_face_links[brick_id * 6 + face_idx]`).
const FACE_INTERIOR: u32 = 0xFFFFFFFEu;
const FACE_EMPTY_LINK: u32 = 0xFFFFFFFFu;

// Face indices (the second factor in the brick_face_links layout).
const FACE_NX: u32 = 0u;
const FACE_PX: u32 = 1u;
const FACE_NY: u32 = 2u;
const FACE_PY: u32 = 3u;
const FACE_NZ: u32 = 4u;
const FACE_PZ: u32 = 5u;
