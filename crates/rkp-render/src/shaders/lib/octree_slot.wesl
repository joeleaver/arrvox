// Octree-lookup result slot encoding.
//
// `octree_lookup` returns a u32 whose bits classify what was found:
//   bit 31 = LEAF, bit 30 = BRICK, bit 29 = BAND, bits 28..0 = payload.
// Two whole-word sentinels (EMPTY, INTERIOR) take precedence over the
// bit-pattern interpretation — slot helpers must reject them.

const OCTREE_EMPTY:        u32 = 0xFFFFFFFFu;
const OCTREE_INTERIOR:     u32 = 0xFFFFFFFEu;
const OCTREE_LEAF_BIT:     u32 = 0x80000000u;
const OCTREE_BRICK_BIT:    u32 = 0x40000000u;
// Phase B-redux 3b — band-cell sentinel. A leaf with `LEAF | BAND |
// payload` is a band cell whose payload indexes `leaf_attr_pool`,
// where the slot's two u32s reinterpret as `{region_id,
// anchor_leaf_slot}` (see octree_march.wesl::handle_band_cell).
const OCTREE_BAND_BIT:     u32 = 0x20000000u;
const OCTREE_PAYLOAD_MASK: u32 = 0x1FFFFFFFu;

/// True if a slot encodes a real brick (not EMPTY or INTERIOR sentinel).
fn slot_is_brick(slot: u32) -> bool {
    return (slot & OCTREE_BRICK_BIT) != 0u
        && slot != OCTREE_EMPTY
        && slot != OCTREE_INTERIOR;
}

/// True if a slot encodes a band cell (BAND_BIT set, not a sentinel).
fn slot_is_band(slot: u32) -> bool {
    return (slot & OCTREE_BAND_BIT) != 0u
        && slot != OCTREE_EMPTY
        && slot != OCTREE_INTERIOR;
}

/// Strip BRICK_BIT to get the brick payload index.
fn slot_brick_id(slot: u32) -> u32 {
    return slot & OCTREE_PAYLOAD_MASK;
}

/// Strip BAND_BIT / BRICK_BIT to get the band-cell offset (index
/// into `leaf_attr_pool` where the 16-byte BandCell record begins;
/// the next slot at `+1` is its second half).
fn slot_band_offset(slot: u32) -> u32 {
    return slot & OCTREE_PAYLOAD_MASK;
}
