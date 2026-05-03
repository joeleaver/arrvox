//! CPU reference implementations of every TLAS build stage —
//! `assemble_host`, `assemble_user_shader`, Morton coding, radix sort,
//! Karras tree build, full-tree assembly. Plus low-level math
//! (`transform_aabb`, `scene_aabb_from_prims`, `expand_bits_10`,
//! `morton_30`, `karras_delta`).
//!
//! Used by the integration tests in `crates/rkp-render/tests/tlas_*`
//! as the GPU output's known-good oracle. The integration tests run
//! the GPU pipeline + the CPU reference over identical inputs and
//! compare results as multisets (atomic order across workgroups is
//! implementation-defined; the **set** of emitted entries is
//! deterministic).
//!
//! All `pub` so external test files can reach in. Internal helpers
//! (`expand_bits_10`, `morton_30`, `transform_aabb`) stay private to
//! this module.

use crate::rkp_gpu_object::{RkpGpuAsset, RkpGpuInstance};

use super::types::{InstanceTileCullEntry, TlasPrim, TLAS_LEAF_USER_SHADER};

/// CPU reference for the user-shader assembly path. Mirrors
/// `tlas_assemble_user_shader.wgsl::assemble_user_shader_main`
/// faithfully (same filter rules, same atomic-driven slot
/// assignment) so the integration test can assert the GPU output
/// matches a known-good CPU walk over the same inputs.
///
/// Returns `(prims_in_emit_order, count)`. Slot order is the order
/// threads "win" their atomic increment — for a single-workgroup
/// dispatch on most GPUs this is monotonic by `gid`, so the
/// reference walks scratch in index order. The test compares
/// outputs as **multisets** (sorted) since the GPU's atomic order
/// across multiple workgroups is implementation-defined.
pub fn cpu_reference_assemble_user_shader(
    scratch: &[InstanceTileCullEntry],
    prims_capacity: u32,
) -> (Vec<TlasPrim>, u32) {
    let mut out: Vec<TlasPrim> = Vec::new();
    let mut count: u32 = 0;
    for entry in scratch {
        if entry.live != 1 {
            continue;
        }
        let extent = [
            entry.aabb_max[0] - entry.aabb_min[0],
            entry.aabb_max[1] - entry.aabb_min[1],
            entry.aabb_max[2] - entry.aabb_min[2],
        ];
        if extent[0] <= 0.0 || extent[1] <= 0.0 || extent[2] <= 0.0 {
            continue;
        }
        let slot = count;
        count += 1;
        if slot >= prims_capacity {
            continue;
        }
        out.push(TlasPrim {
            aabb_min: entry.aabb_min,
            asset_id: entry.asset_id,
            aabb_max: entry.aabb_max,
            instance_state_offset: entry.instance_state_offset,
            material_id: entry.material_id,
            instance_index: TLAS_LEAF_USER_SHADER,
            _pad0: 0,
            _pad1: 0,
        });
    }
    (out, count)
}

/// CPU reference for the host-instance assembly path. Mirrors
/// `tlas_assemble_host.wgsl::assemble_host_main`. Same multiset
/// semantics as the user-shader reference.
pub fn cpu_reference_assemble_host(
    instances: &[RkpGpuInstance],
    assets: &[RkpGpuAsset],
    prims_capacity: u32,
) -> (Vec<TlasPrim>, u32) {
    let mut out: Vec<TlasPrim> = Vec::new();
    let mut count: u32 = 0;
    for (i, inst) in instances.iter().enumerate() {
        let asset_id = inst.asset_id as usize;
        if asset_id >= assets.len() {
            continue;
        }
        let asset = &assets[asset_id];
        if asset.shader_id != 0 {
            continue;
        }
        let (world_min, world_max) =
            transform_aabb(asset.aabb_min, asset.aabb_max, &inst.world);
        let slot = count;
        count += 1;
        if slot >= prims_capacity {
            continue;
        }
        out.push(TlasPrim {
            aabb_min: world_min,
            asset_id: inst.asset_id,
            aabb_max: world_max,
            instance_state_offset: 0,
            material_id: inst.material_id,
            instance_index: i as u32,
            _pad0: 0,
            _pad1: 0,
        });
    }
    (out, count)
}

/// Same Arvo's transform-AABB the CPU `tlas_pass.rs::transform_aabb`
/// uses; duplicated here so this module doesn't depend on
/// `tlas_pass`. Will be deleted along with the CPU `build_tlas` in
/// Session 5.
fn transform_aabb(
    local_min: [f32; 3],
    local_max: [f32; 3],
    world: &[[f32; 4]; 4],
) -> ([f32; 3], [f32; 3]) {
    let mut new_min = [world[3][0], world[3][1], world[3][2]];
    let mut new_max = [world[3][0], world[3][1], world[3][2]];
    for i in 0..3 {
        for j in 0..3 {
            let a = world[j][i] * local_min[j];
            let b = world[j][i] * local_max[j];
            new_min[i] += a.min(b);
            new_max[i] += a.max(b);
        }
    }
    (new_min, new_max)
}

/// Compute the union AABB of a list of `TlasPrim`s. Used by the
/// engine to derive the [`MortonUniform`] scene bounds before the
/// Morton-code dispatch — Morton sort just needs a stable
/// coordinate system, so a CPU-side conservative bound is fine
/// (saves a GPU reduction pass; lifts cleanly to GPU later if N
/// grows past the point where CPU iteration matters).
///
/// Empty input returns a 1-unit cube at the origin so the
/// downstream `extent.max(1e-6)` clamp in the WGSL doesn't divide
/// by zero on the (no-op) Morton dispatch.
pub fn scene_aabb_from_prims(prims: &[TlasPrim]) -> ([f32; 3], [f32; 3]) {
    if prims.is_empty() {
        return ([0.0; 3], [1.0; 3]);
    }
    let mut min = prims[0].aabb_min;
    let mut max = prims[0].aabb_max;
    for p in &prims[1..] {
        for ax in 0..3 {
            if p.aabb_min[ax] < min[ax] {
                min[ax] = p.aabb_min[ax];
            }
            if p.aabb_max[ax] > max[ax] {
                max[ax] = p.aabb_max[ax];
            }
        }
    }
    (min, max)
}

fn expand_bits_10(v_in: u32) -> u32 {
    let mut v = v_in & 0x3FF;
    v = (v | (v << 16)) & 0x030000FF;
    v = (v | (v << 8)) & 0x0300F00F;
    v = (v | (v << 4)) & 0x030C30C3;
    v = (v | (v << 2)) & 0x09249249;
    v
}

pub(super) fn morton_30(x: u32, y: u32, z: u32) -> u32 {
    (expand_bits_10(x) << 2) | (expand_bits_10(y) << 1) | expand_bits_10(z)
}

/// CPU reference for `tlas_morton.wgsl::compute_morton_main`.
/// Returns the (Morton, prim_idx) pairs that the GPU dispatch
/// would produce given the same input.
pub fn cpu_reference_morton(
    prims: &[TlasPrim],
    scene_min: [f32; 3],
    scene_max: [f32; 3],
) -> Vec<(u32, u32)> {
    let mut out = Vec::with_capacity(prims.len());
    let extent = [
        (scene_max[0] - scene_min[0]).max(1e-6),
        (scene_max[1] - scene_min[1]).max(1e-6),
        (scene_max[2] - scene_min[2]).max(1e-6),
    ];
    for (i, p) in prims.iter().enumerate() {
        let centroid = [
            0.5 * (p.aabb_min[0] + p.aabb_max[0]),
            0.5 * (p.aabb_min[1] + p.aabb_max[1]),
            0.5 * (p.aabb_min[2] + p.aabb_max[2]),
        ];
        let normalized = [
            ((centroid[0] - scene_min[0]) / extent[0]).clamp(0.0, 1.0),
            ((centroid[1] - scene_min[1]) / extent[1]).clamp(0.0, 1.0),
            ((centroid[2] - scene_min[2]) / extent[2]).clamp(0.0, 1.0),
        ];
        let q = [
            (normalized[0] * 1023.0).min(1023.0) as u32,
            (normalized[1] * 1023.0).min(1023.0) as u32,
            (normalized[2] * 1023.0).min(1023.0) as u32,
        ];
        out.push((morton_30(q[0], q[1], q[2]), i as u32));
    }
    out
}

/// CPU reference for the GPU radix sort. Standard `Vec::sort_by`
/// over the (key, val) pairs — the GPU's stability guarantees are
/// per-bucket but not within ties, so the integration test
/// compares as a multiset (sort by key first, then by val) instead
/// of asserting bit-for-bit equality.
pub fn cpu_reference_radix_sort(pairs: &[(u32, u32)]) -> Vec<(u32, u32)> {
    let mut sorted = pairs.to_vec();
    sorted.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
    sorted
}

/// Karras' `delta(i, j)` = length of common prefix of the virtual
/// keys `(morton[i] << 32) | i`. Mirrors the WGSL helper in
/// `tlas_karras.wgsl::delta`. Returns -1 for out-of-range j.
pub fn karras_delta(sorted_keys: &[u32], i: i32, j: i32) -> i32 {
    let n = sorted_keys.len() as i32;
    if j < 0 || j >= n {
        return -1;
    }
    let ki = sorted_keys[i as usize];
    let kj = sorted_keys[j as usize];
    if ki != kj {
        return (ki ^ kj).leading_zeros() as i32;
    }
    32 + ((i as u32) ^ (j as u32)).leading_zeros() as i32
}

/// CPU reference for the full Karras tree + AABB propagation. Given
/// sorted Mortons + sorted_vals + the underlying TlasPrim payloads,
/// returns the `tlas_nodes` array (length 2N-1) with both topology
/// AND AABBs filled in, exactly matching what the GPU pipeline
/// (S3 + S4) is expected to produce.
///
/// Returns `(nodes, leaves)`. `leaves[i]` is the TlasInstanceLeaf
/// payload of `prims[sorted_vals[i]]` (also matches the `tlas_leaves`
/// buffer the GPU produces).
pub fn cpu_reference_full_tree(
    sorted_keys: &[u32],
    sorted_vals: &[u32],
    prims: &[TlasPrim],
) -> (Vec<crate::tlas_pass::TlasNode>, Vec<crate::tlas_pass::TlasInstanceLeaf>) {
    use crate::tlas_pass::{TlasInstanceLeaf, TlasNode, TLAS_NODE_LEAF_BIT};
    let n = sorted_keys.len();
    if n == 0 {
        return (Vec::new(), Vec::new());
    }
    let mut nodes: Vec<TlasNode> = vec![
        TlasNode {
            aabb_min: [0.0; 3],
            left_or_leaf: 0,
            aabb_max: [0.0; 3],
            right_or_count: 0,
        };
        2 * n - 1
    ];
    let mut leaves: Vec<TlasInstanceLeaf> = Vec::with_capacity(n);

    // Leaves.
    for i in 0..n {
        let prim = &prims[sorted_vals[i] as usize];
        leaves.push(TlasInstanceLeaf {
            asset_id: prim.asset_id,
            instance_state_offset: prim.instance_state_offset,
            material_id: prim.material_id,
            instance_index: prim.instance_index,
        });
        let leaf_node = &mut nodes[n - 1 + i];
        leaf_node.aabb_min = prim.aabb_min;
        leaf_node.aabb_max = prim.aabb_max;
        leaf_node.left_or_leaf = TLAS_NODE_LEAF_BIT | (i as u32);
        leaf_node.right_or_count = 1;
    }

    // Internal nodes — topology first.
    for i in 0..n.saturating_sub(1) {
        let (l, r) = cpu_reference_karras_node(sorted_keys, i as i32);
        nodes[i].left_or_leaf = l;
        nodes[i].right_or_count = r;
    }

    // Internal AABBs — bottom-up via post-order traversal from root.
    if n >= 2 {
        fn fill_aabb(idx: u32, nodes: &mut [crate::tlas_pass::TlasNode]) {
            if (nodes[idx as usize].left_or_leaf & crate::tlas_pass::TLAS_NODE_LEAF_BIT) != 0 {
                return; // Leaf — AABB already set.
            }
            let l = nodes[idx as usize].left_or_leaf;
            let r = nodes[idx as usize].right_or_count;
            fill_aabb(l, nodes);
            fill_aabb(r, nodes);
            let lmin = nodes[l as usize].aabb_min;
            let lmax = nodes[l as usize].aabb_max;
            let rmin = nodes[r as usize].aabb_min;
            let rmax = nodes[r as usize].aabb_max;
            nodes[idx as usize].aabb_min = [
                lmin[0].min(rmin[0]),
                lmin[1].min(rmin[1]),
                lmin[2].min(rmin[2]),
            ];
            nodes[idx as usize].aabb_max = [
                lmax[0].max(rmax[0]),
                lmax[1].max(rmax[1]),
                lmax[2].max(rmax[2]),
            ];
        }
        fill_aabb(0, &mut nodes);
    }

    (nodes, leaves)
}

/// CPU reference for `tlas_karras.wgsl::build_internal_main`.
/// Returns the pair of children (left, right) for internal node
/// `idx`, in the convention the WGSL writes: `< prim_count - 1` is
/// another internal node, `≥ prim_count - 1` is a leaf-marker
/// node at index `prim_count - 1 + leaf_idx` in `tlas_nodes`.
pub fn cpu_reference_karras_node(sorted_keys: &[u32], idx: i32) -> (u32, u32) {
    let n = sorted_keys.len() as i32;
    debug_assert!(idx < n - 1, "internal node index {idx} out of range (n = {n})");

    // Direction.
    let d = (karras_delta(sorted_keys, idx, idx + 1) - karras_delta(sorted_keys, idx, idx - 1))
        .signum();
    let delta_min = karras_delta(sorted_keys, idx, idx - d);

    // Upper bound on range length.
    let mut l_max: i32 = 2;
    while karras_delta(sorted_keys, idx, idx + l_max * d) > delta_min {
        l_max *= 2;
        if l_max > n {
            break;
        }
    }

    // Binary search for length l.
    let mut l: i32 = 0;
    let mut t = l_max / 2;
    while t >= 1 {
        if karras_delta(sorted_keys, idx, idx + (l + t) * d) > delta_min {
            l += t;
        }
        t /= 2;
    }

    let j = idx + l * d;
    let delta_node = karras_delta(sorted_keys, idx, j);

    // Find split.
    let mut s: i32 = 0;
    let mut divisor: i32 = 2;
    loop {
        let t_split = (l + divisor - 1) / divisor;
        if karras_delta(sorted_keys, idx, idx + (s + t_split) * d) > delta_node {
            s += t_split;
        }
        if t_split <= 1 {
            break;
        }
        divisor *= 2;
    }
    let gamma = idx + s * d + d.min(0);

    let range_lo = idx.min(j);
    let range_hi = idx.max(j);

    let left_child = if range_lo == gamma {
        (n - 1 + gamma) as u32
    } else {
        gamma as u32
    };
    let right_child = if range_hi == gamma + 1 {
        (n - 1 + gamma + 1) as u32
    } else {
        (gamma + 1) as u32
    };
    (left_child, right_child)
}

