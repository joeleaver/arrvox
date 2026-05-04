//! Arena-based tree structure for procedural objects.
//!
//! Nodes are stored in a flat `Vec` with parent pointers. This avoids
//! `Box<Node>` reference cycles and enables push-based version invalidation
//! by walking the parent chain.

use glam::Affine3A;
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;

use crate::node_kind::NodeKind;

mod structural;
mod moves;

#[cfg(test)]
mod tests;

/// Index into the node arena.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct NodeId(pub u32);

/// A single node in the procedural tree.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Node {
    pub kind: NodeKind,
    pub parent: Option<NodeId>,
    pub children: SmallVec<[NodeId; 2]>,
    /// Local transform (relative to parent).
    pub transform: Affine3A,
    /// Bumped when this node's own parameters change. Not persisted —
    /// versions are runtime cache-invalidation state, not scene data.
    /// On load we re-seed to 1 (same as a freshly constructed node);
    /// any external cache keyed on versions will re-prime naturally.
    #[serde(skip, default = "default_version")]
    pub own_version: u64,
    /// Max of own_version and all descendants' subtree_versions. Also
    /// skipped for the same reason.
    #[serde(skip, default = "default_version")]
    pub subtree_version: u64,
}

fn default_version() -> u64 {
    1
}

impl Node {
    fn new(kind: NodeKind) -> Self {
        Self {
            kind,
            parent: None,
            children: SmallVec::new(),
            transform: Affine3A::IDENTITY,
            own_version: 1,
            subtree_version: 1,
        }
    }
}

/// A procedural object: an arena of nodes forming a tree.
///
/// The tree has a single root. Nodes are addressed by [`NodeId`] which indexes
/// into the arena. Removed nodes leave tombstones (the slot is not reused) to
/// keep existing NodeIds stable.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProceduralObject {
    nodes: Vec<Option<Node>>,
    root: NodeId,
    /// Global version counter — incremented on every mutation. Not
    /// persisted; re-seeded to 2 (matches `new`) on load so any future
    /// mutation starts fresh without colliding with cached state.
    #[serde(skip, default = "default_next_version")]
    next_version: u64,
}

fn default_next_version() -> u64 {
    2
}

#[cfg(test)]
mod serde_tests {
    use super::*;
    use crate::node_kind::*;
    use glam::{Affine3A, Vec3};

    /// A tree with primitives + a combinator + transforms must round-trip
    /// through serde unchanged (structurally) — this is the guarantee
    /// `.rkproject` save/load relies on.
    #[test]
    fn procedural_object_json_roundtrip() {
        let mut obj = ProceduralObject::new(NodeKind::Union {
            material_combine: MaterialCombine::Winner,
        });
        let a = obj.add_child(obj.root(), NodeKind::Sphere(SphereParams {
            radius: 0.7, material_id: 3, color: Vec3::new(0.9, 0.2, 0.1),
        }));
        obj.set_transform(a, Affine3A::from_translation(Vec3::new(1.5, 0.0, 0.0)));
        let b = obj.add_child(obj.root(), NodeKind::Box(BoxParams {
            half_extents: Vec3::new(0.4, 0.5, 0.6), rounding: 0.05,
            material_id: 7, color: Vec3::new(0.1, 0.9, 0.3),
        }));
        obj.set_transform(b, Affine3A::from_rotation_y(0.5));

        let json = serde_json::to_string(&obj).expect("serialize");
        let back: ProceduralObject = serde_json::from_str(&json).expect("deserialize");

        // Structural equality: same node count, same root, same
        // kinds at each slot.
        assert_eq!(back.arena_len(), obj.arena_len());
        assert_eq!(back.root(), obj.root());
        for id in obj.iter_ids() {
            let orig = obj.get(id).unwrap();
            let round = back.get(id).unwrap();
            // Approximate-equal is overkill for a JSON roundtrip —
            // f32 serialization is lossless here — so strict `==` is
            // appropriate via bitwise-identical f32 literals.
            match (&orig.kind, &round.kind) {
                (NodeKind::Sphere(a), NodeKind::Sphere(b)) => {
                    assert_eq!(a.radius, b.radius);
                    assert_eq!(a.material_id, b.material_id);
                }
                (NodeKind::Box(a), NodeKind::Box(b)) => {
                    assert_eq!(a.half_extents, b.half_extents);
                    assert_eq!(a.rounding, b.rounding);
                }
                (NodeKind::Union { .. }, NodeKind::Union { .. }) => {}
                other => panic!("kind mismatch: {other:?}"),
            }
            assert_eq!(orig.parent, round.parent);
            assert_eq!(orig.children.as_slice(), round.children.as_slice());
            assert_eq!(orig.transform, round.transform);
        }
    }
}

impl ProceduralObject {
    /// Create a new procedural object with the given root node kind.
    pub fn new(root_kind: NodeKind) -> Self {
        let root = Node::new(root_kind);
        Self {
            nodes: vec![Some(root)],
            root: NodeId(0),
            next_version: 2,
        }
    }

    /// The root node ID.
    pub fn root(&self) -> NodeId {
        self.root
    }

    /// Get a node by ID.
    pub fn get(&self, id: NodeId) -> Option<&Node> {
        self.nodes.get(id.0 as usize)?.as_ref()
    }

    /// Get a mutable reference to a node by ID.
    pub fn get_mut(&mut self, id: NodeId) -> Option<&mut Node> {
        self.nodes.get_mut(id.0 as usize)?.as_mut()
    }

    /// Number of live (non-tombstone) nodes.
    pub fn node_count(&self) -> usize {
        self.nodes.iter().filter(|n| n.is_some()).count()
    }

    /// Arena capacity — highest-possible `NodeId.0 + 1`. Useful for sizing
    /// per-node side tables (e.g. bounds caches). Includes tombstones.
    pub fn arena_len(&self) -> usize {
        self.nodes.len()
    }

    /// Bump a node's version after a parameter change, propagating up to root.
    ///
    /// Call this after modifying a node's `NodeKind` parameters directly.
    pub fn bump_version(&mut self, id: NodeId) {
        if let Some(node) = &mut self.nodes[id.0 as usize] {
            node.own_version = self.next_version;
            self.next_version += 1;
            // Propagate up — start from this node, not just parent.
            node.subtree_version = node.subtree_version.max(node.own_version);
            if let Some(parent) = node.parent {
                self.propagate_version(parent);
            }
        }
    }

    /// Set the transform on a node, bumping its version.
    pub fn set_transform(&mut self, id: NodeId, transform: Affine3A) {
        if let Some(node) = &mut self.nodes[id.0 as usize] {
            node.transform = transform;
            node.own_version = self.next_version;
            self.next_version += 1;
            if let Some(parent) = node.parent {
                self.propagate_version(parent);
            } else {
                // Root node — just update its own subtree_version.
                node.subtree_version = node.subtree_version.max(node.own_version);
            }
        }
    }

    /// Iterate over all live node IDs.
    pub fn iter_ids(&self) -> impl Iterator<Item = NodeId> + '_ {
        self.nodes
            .iter()
            .enumerate()
            .filter_map(|(i, n)| n.as_ref().map(|_| NodeId(i as u32)))
    }

    /// Walk the parent chain from `start` (inclusive) to root, calling `f` at each node.
    pub fn walk_ancestors(&self, start: NodeId, mut f: impl FnMut(NodeId, &Node)) {
        let mut cursor = Some(start);
        while let Some(id) = cursor {
            if let Some(node) = &self.nodes[id.0 as usize] {
                f(id, node);
                cursor = node.parent;
            } else {
                break;
            }
        }
    }

    /// Propagate version changes up the parent chain from the given node.
    fn propagate_version(&mut self, from: NodeId) {
        let version = self.next_version;
        self.next_version += 1;

        let mut cursor = Some(from);
        while let Some(id) = cursor {
            if let Some(node) = &mut self.nodes[id.0 as usize] {
                node.subtree_version = node.subtree_version.max(version);
                cursor = node.parent;
            } else {
                break;
            }
        }
    }
}

