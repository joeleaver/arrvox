//! Top-down median-split BVH over region AABBs.
//!
//! Built once when the region set changes (rare at runtime); queried
//! by point for "which regions contain this point?". The design doc
//! calls out that V1 has "tens to low-hundreds of regions" and that
//! query cost is dominated by per-shape membership maths, not
//! traversal — so the BVH is a structural correctness foundation
//! rather than a hot-path optimisation. The interface is stable
//! enough that a future SAH-split or refit-on-move variant can land
//! without touching call sites.

use arvx_core::Aabb;
use glam::Vec3;

const LEAF_CAPACITY: usize = 4;

/// One BVH node. Either a leaf with a span of entry indices, or an
/// internal node with two children. The two states share the same
/// memory layout — `count > 0` discriminates a leaf.
///
/// Both children of an internal node carry an explicit index because
/// `build_recursive` appends each subtree's nodes at the tail of
/// `nodes`, so the right child is *not* at `left + 1`.
#[derive(Debug, Clone, Copy)]
struct Node {
    aabb: Aabb,
    /// Leaf: index of the first entry in `indices`.
    /// Internal: index of the left child node in `nodes`.
    first: u32,
    /// Leaf: unused. Internal: index of the right child node in
    /// `nodes`.
    second: u32,
    /// Leaf: number of entries (>= 1). Internal: 0 (sentinel).
    count: u32,
}

impl Node {
    #[inline]
    fn is_leaf(&self) -> bool {
        self.count > 0
    }
}

/// BVH over an external slice of AABBs.
///
/// Stores indices into the caller's flat entry array — the BVH never
/// owns the entries themselves. This keeps the BVH compact and lets
/// the [`crate::RegionIndex`] keep entries in the order callers
/// expect (typically insertion / iteration order).
#[derive(Debug, Clone, Default)]
pub struct RegionBvh {
    nodes: Vec<Node>,
    /// Permuted indices into the caller's `&[Aabb]`. Leaf nodes
    /// reference contiguous spans of this array.
    indices: Vec<u32>,
}

impl RegionBvh {
    /// Empty BVH — `query_point` returns nothing.
    pub fn new() -> Self {
        Self::default()
    }

    /// Build a BVH over `aabbs`. The indices in subsequent
    /// `query_point` callbacks are positions in the input slice.
    pub fn build(aabbs: &[Aabb]) -> Self {
        let mut indices: Vec<u32> = (0..aabbs.len() as u32).collect();
        let mut nodes: Vec<Node> = Vec::with_capacity((aabbs.len() * 2).max(1));
        if !aabbs.is_empty() {
            build_recursive(&mut nodes, &mut indices, 0, aabbs.len(), aabbs);
        }
        Self { nodes, indices }
    }

    /// Number of entries indexed.
    pub fn len(&self) -> usize {
        self.indices.len()
    }

    /// True if the BVH indexes no entries.
    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }

    /// Visit every entry whose AABB contains `point`. `visit` receives
    /// the entry index from the input slice (i.e. the position in the
    /// slice passed to `build`).
    pub fn query_point<F>(&self, aabbs: &[Aabb], point: Vec3, mut visit: F)
    where
        F: FnMut(usize),
    {
        if self.nodes.is_empty() {
            return;
        }
        // Explicit stack — recursive walks of tens-to-hundreds of
        // entries are fine but we want the same contract as the rest
        // of the query path (cheap, allocation-light).
        let mut stack: [u32; 64] = [0; 64];
        let mut top = 0usize;
        stack[top] = 0;
        top += 1;

        while top > 0 {
            top -= 1;
            let n = self.nodes[stack[top] as usize];
            if !aabb_contains(&n.aabb, point) {
                continue;
            }
            if n.is_leaf() {
                let begin = n.first as usize;
                let end = begin + n.count as usize;
                for &i in &self.indices[begin..end] {
                    if aabb_contains(&aabbs[i as usize], point) {
                        visit(i as usize);
                    }
                }
            } else {
                if top + 2 > stack.len() {
                    // Defensive: a 64-deep stack covers ~2^64 leaves;
                    // if we somehow blow it we'd rather degrade than
                    // panic. Force-flatten the remaining entries.
                    flatten_all(&self.indices, aabbs, point, &mut visit);
                    continue;
                }
                stack[top] = n.first;
                stack[top + 1] = n.second;
                top += 2;
            }
        }
    }
}

fn aabb_contains(a: &Aabb, p: Vec3) -> bool {
    p.x >= a.min.x
        && p.x <= a.max.x
        && p.y >= a.min.y
        && p.y <= a.max.y
        && p.z >= a.min.z
        && p.z <= a.max.z
}

fn build_recursive(
    nodes: &mut Vec<Node>,
    indices: &mut [u32],
    begin: usize,
    end: usize,
    aabbs: &[Aabb],
) -> u32 {
    let node_index = nodes.len() as u32;
    // Reserve the slot; fill below.
    nodes.push(Node {
        aabb: aabbs[indices[begin] as usize],
        first: 0,
        second: 0,
        count: 0,
    });

    let aabb = union_aabbs(indices, begin, end, aabbs);
    let count = end - begin;

    if count <= LEAF_CAPACITY {
        nodes[node_index as usize] = Node {
            aabb,
            first: begin as u32,
            second: 0,
            count: count as u32,
        };
        return node_index;
    }

    // Pick split axis = largest extent of the centroid bounding box.
    let mut centroid_min = Vec3::splat(f32::INFINITY);
    let mut centroid_max = Vec3::splat(f32::NEG_INFINITY);
    for &i in &indices[begin..end] {
        let c = aabbs[i as usize].center();
        centroid_min = centroid_min.min(c);
        centroid_max = centroid_max.max(c);
    }
    let extents = centroid_max - centroid_min;
    let axis = if extents.x >= extents.y && extents.x >= extents.z {
        0
    } else if extents.y >= extents.z {
        1
    } else {
        2
    };

    // If the centroids degenerate to a single point (all regions on
    // top of each other), bail to a leaf — splitting is meaningless.
    if extents[axis] <= f32::EPSILON {
        nodes[node_index as usize] = Node {
            aabb,
            first: begin as u32,
            second: 0,
            count: count as u32,
        };
        return node_index;
    }

    // Median-split on the chosen axis.
    let mid = begin + count / 2;
    indices[begin..end].select_nth_unstable_by(count / 2, |&a, &b| {
        let ca = aabbs[a as usize].center()[axis];
        let cb = aabbs[b as usize].center()[axis];
        ca.partial_cmp(&cb).unwrap_or(std::cmp::Ordering::Equal)
    });

    let left = build_recursive(nodes, indices, begin, mid, aabbs);
    let right = build_recursive(nodes, indices, mid, end, aabbs);
    nodes[node_index as usize] = Node {
        aabb,
        first: left,
        second: right,
        count: 0,
    };
    node_index
}

fn union_aabbs(indices: &[u32], begin: usize, end: usize, aabbs: &[Aabb]) -> Aabb {
    let mut acc = aabbs[indices[begin] as usize];
    for &i in &indices[begin + 1..end] {
        acc = acc.expand_aabb(&aabbs[i as usize]);
    }
    acc
}

fn flatten_all<F>(indices: &[u32], aabbs: &[Aabb], point: Vec3, visit: &mut F)
where
    F: FnMut(usize),
{
    // Last-resort fallback for the (statistically impossible) deep-
    // stack overflow path. Visit everything reachable from the entry
    // index slice — same result, just no longer logarithmic.
    for &i in indices {
        if aabb_contains(&aabbs[i as usize], point) {
            visit(i as usize);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;

    fn cube_at(center: Vec3, half: f32) -> Aabb {
        Aabb::from_center_half_extents(center, Vec3::splat(half))
    }

    #[test]
    fn empty_bvh_returns_nothing() {
        let bvh = RegionBvh::build(&[]);
        let mut hits = Vec::new();
        bvh.query_point(&[], Vec3::ZERO, |i| hits.push(i));
        assert!(hits.is_empty());
    }

    #[test]
    fn single_entry_inside_hits() {
        let aabbs = vec![cube_at(Vec3::ZERO, 1.0)];
        let bvh = RegionBvh::build(&aabbs);
        let mut hits = Vec::new();
        bvh.query_point(&aabbs, Vec3::ZERO, |i| hits.push(i));
        assert_eq!(hits, vec![0]);
    }

    #[test]
    fn single_entry_outside_misses() {
        let aabbs = vec![cube_at(Vec3::ZERO, 1.0)];
        let bvh = RegionBvh::build(&aabbs);
        let mut hits = Vec::new();
        bvh.query_point(&aabbs, Vec3::new(100.0, 0.0, 0.0), |i| hits.push(i));
        assert!(hits.is_empty());
    }

    #[test]
    fn returns_all_containing_aabbs() {
        // Three concentric AABBs — a centred point hits all three.
        let aabbs = vec![
            cube_at(Vec3::ZERO, 1.0),
            cube_at(Vec3::ZERO, 5.0),
            cube_at(Vec3::ZERO, 20.0),
        ];
        let bvh = RegionBvh::build(&aabbs);
        let mut hits = Vec::new();
        bvh.query_point(&aabbs, Vec3::ZERO, |i| hits.push(i));
        hits.sort();
        assert_eq!(hits, vec![0, 1, 2]);
    }

    #[test]
    fn point_outside_all_returns_nothing() {
        // Eight AABBs in a grid; point well outside should hit none.
        let mut aabbs = Vec::new();
        for x in [-10.0_f32, 10.0] {
            for y in [-10.0_f32, 10.0] {
                for z in [-10.0_f32, 10.0] {
                    aabbs.push(cube_at(Vec3::new(x, y, z), 2.0));
                }
            }
        }
        let bvh = RegionBvh::build(&aabbs);
        let mut hits = Vec::new();
        bvh.query_point(&aabbs, Vec3::new(50.0, 50.0, 50.0), |i| hits.push(i));
        assert!(hits.is_empty());
    }

    #[test]
    fn dense_scene_finds_overlapping() {
        // 100 cubes scattered along X; a point at x=37.5 hits exactly
        // those cubes whose AABB straddles that x.
        let mut aabbs = Vec::new();
        for i in 0..100 {
            aabbs.push(cube_at(Vec3::new(i as f32, 0.0, 0.0), 1.0));
        }
        let bvh = RegionBvh::build(&aabbs);
        let mut hits = Vec::new();
        bvh.query_point(&aabbs, Vec3::new(37.5, 0.0, 0.0), |i| hits.push(i));
        hits.sort();
        // x = 37.5 is inside cubes 37 (x ∈ [36, 38]) and 38 (x ∈ [37, 39]).
        // 36 reaches x=37, 38 reaches x=37 (max). 37.5 ∈ [36,38] for both.
        assert_eq!(hits, vec![37, 38]);
    }

    #[test]
    fn duplicates_at_same_centre_dont_panic() {
        // Stress test the centroid-degenerate path.
        let aabbs: Vec<Aabb> = (0..16).map(|_| cube_at(Vec3::ZERO, 1.0)).collect();
        let bvh = RegionBvh::build(&aabbs);
        let mut hits = Vec::new();
        bvh.query_point(&aabbs, Vec3::ZERO, |i| hits.push(i));
        hits.sort();
        assert_eq!(hits.len(), 16);
    }

    #[test]
    fn indices_are_input_positions_not_permuted() {
        // Build over 8 entries, query a point that hits exactly entry
        // #5 only — we must report 5, not the BVH's internal position.
        let mut aabbs = Vec::new();
        for i in 0..8 {
            aabbs.push(cube_at(Vec3::new(i as f32 * 100.0, 0.0, 0.0), 1.0));
        }
        let bvh = RegionBvh::build(&aabbs);
        let mut hits = Vec::new();
        bvh.query_point(&aabbs, Vec3::new(500.0, 0.0, 0.0), |i| hits.push(i));
        assert_eq!(hits, vec![5]);
    }
}
