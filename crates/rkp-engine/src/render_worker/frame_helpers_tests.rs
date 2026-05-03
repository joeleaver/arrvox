use super::splice_transient_into_tile_lists;

fn u32s(v: &[u32]) -> Vec<u8> { bytemuck::cast_slice(v).to_vec() }

#[test]
fn splice_no_transient_passes_through() {
    let offsets = u32s(&[0, 2, 5]);
    let ids = u32s(&[1, 2, 3, 4, 5]);
    let (no, ni) = splice_transient_into_tile_lists(&offsets, &ids, &[]);
    assert_eq!(no, vec![0, 2, 5]);
    assert_eq!(ni, vec![1, 2, 3, 4, 5]);
}

#[test]
fn splice_appends_transient_to_each_tile() {
    // 2 tiles. Tile 0 has [1, 2], tile 1 has [3, 4, 5]. After
    // splicing transient [99, 100] into both, tile 0 → [1, 2, 99, 100],
    // tile 1 → [3, 4, 5, 99, 100].
    let offsets = u32s(&[0, 2, 5]);
    let ids = u32s(&[1, 2, 3, 4, 5]);
    let (no, ni) = splice_transient_into_tile_lists(&offsets, &ids, &[99, 100]);
    // New offsets: [0, 4, 9].
    assert_eq!(no, vec![0, 4, 9]);
    // Concatenated ids in tile order.
    assert_eq!(ni, vec![1, 2, 99, 100, 3, 4, 5, 99, 100]);
}

#[test]
fn splice_empty_tile_still_gets_transient() {
    // Tile 0 has no objects, but transient should still appear.
    let offsets = u32s(&[0, 0, 1]);
    let ids = u32s(&[42]);
    let (no, ni) = splice_transient_into_tile_lists(&offsets, &ids, &[7]);
    assert_eq!(no, vec![0, 1, 3]);
    assert_eq!(ni, vec![7, 42, 7]);
}
