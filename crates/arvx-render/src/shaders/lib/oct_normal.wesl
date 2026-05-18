// Decode a packed 2× snorm16 octahedral normal stored in `LeafAttr.normal_oct`.
// Mirror of arvx_core::unpack_oct.
//
// Hardened variant: clamps decoded u/v into [-1, 1] before octahedral
// reflection and guards against zero-length output (returns +Y as fallback)
// to avoid NaN propagation when callers feed in a degenerate / zero packing.
fn unpack_oct_normal(packed: u32) -> vec3<f32> {
    let ui_raw = i32(packed & 0xFFFFu);
    let vi_raw = i32((packed >> 16u) & 0xFFFFu);
    // snorm16: interpret as i16 (sign-extend the 16-bit value).
    let ui = select(ui_raw, ui_raw - 65536, ui_raw >= 32768);
    let vi = select(vi_raw, vi_raw - 65536, vi_raw >= 32768);
    let u = clamp(f32(ui) / 32767.0, -1.0, 1.0);
    let v = clamp(f32(vi) / 32767.0, -1.0, 1.0);
    var n = vec3<f32>(u, v, 1.0 - abs(u) - abs(v));
    if n.z < 0.0 {
        let nx0 = n.x;
        n.x = (1.0 - abs(n.y)) * select(-1.0, 1.0, nx0 >= 0.0);
        n.y = (1.0 - abs(nx0)) * select(-1.0, 1.0, n.y >= 0.0);
    }
    let len = length(n);
    if len < 1e-8 { return vec3<f32>(0.0, 1.0, 0.0); }
    return n / len;
}

// Pack a unit normal into the same 2× snorm16 octahedral u32 form
// `unpack_oct_normal` decodes. Matches `arvx_core::leaf_attr::pack_oct`.
fn pack_oct_normal(n: vec3<f32>) -> u32 {
    let l1 = abs(n.x) + abs(n.y) + abs(n.z);
    let n1 = n / max(l1, 1e-8);
    var u = n1.x;
    var v = n1.y;
    if n1.z < 0.0 {
        let ox = u;
        u = (1.0 - abs(v)) * select(-1.0, 1.0, ox >= 0.0);
        v = (1.0 - abs(ox)) * select(-1.0, 1.0, v  >= 0.0);
    }
    let ui = i32(clamp(u, -1.0, 1.0) * 32767.0);
    let vi = i32(clamp(v, -1.0, 1.0) * 32767.0);
    let ul = u32(ui & 0xFFFF);
    let vl = u32(vi & 0xFFFF);
    return ul | (vl << 16u);
}
