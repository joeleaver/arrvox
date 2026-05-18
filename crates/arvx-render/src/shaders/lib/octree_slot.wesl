// Octree-lookup result slot encoding.
//
// `octree_lookup` returns a u32 whose bits classify what was found:
//   bit 31 = LEAF, bit 30 = BRICK, bits 29..0 = payload.
// Two whole-word sentinels (EMPTY, INTERIOR) take precedence over the
// bit-pattern interpretation — slot helpers must reject them.
//
// Renderer-wide convention: `0xFFFFFFFFu` (= u32::MAX) is the
// "no value here" sentinel. The same numeric value is reused in
// several semantically distinct contexts (OCTREE_EMPTY,
// BRICK_CELL_EMPTY, INTERNAL_ATTR_NONE, FACE_EMPTY_LINK,
// HOST_NO_HOST_SENTINEL); the names document intent at the call
// site, the value is shared because u32::MAX is the most
// distinctive sentinel a 32-bit field can carry. Mirrored on the
// Rust side as `arvx_render::sentinels::SENTINEL_NONE`.
//
// Bit 29 used to mark BAND cells (the user-shader band-cell descent
// path); that path is gone, the bit is free for future flags.

const OCTREE_EMPTY:        u32 = 0xFFFFFFFFu;
const OCTREE_INTERIOR:     u32 = 0xFFFFFFFEu;
const OCTREE_LEAF_BIT:     u32 = 0x80000000u;
const OCTREE_BRICK_BIT:    u32 = 0x40000000u;
const OCTREE_PAYLOAD_MASK: u32 = 0x1FFFFFFFu;

/// True if a slot encodes a real brick (not EMPTY or INTERIOR sentinel).
fn slot_is_brick(slot: u32) -> bool {
    return (slot & OCTREE_BRICK_BIT) != 0u
        && slot != OCTREE_EMPTY
        && slot != OCTREE_INTERIOR;
}

/// Strip BRICK_BIT to get the brick payload index.
fn slot_brick_id(slot: u32) -> u32 {
    return slot & OCTREE_PAYLOAD_MASK;
}
