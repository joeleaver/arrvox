//! Arena-based tree structure for procedural objects.
//!
//! Nodes are stored in a flat `Vec` with parent pointers. This avoids
//! `Box<Node>` reference cycles and enables push-based version invalidation
//! by walking the parent chain.

use glam::Affine3A;
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;

use crate::node_kind::NodeKind;

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

    /// Add a child node under the given parent. Returns the new node's ID.
    ///
    /// Panics if `parent` doesn't exist or is a leaf shape (leaves can't have children).
    pub fn add_child(&mut self, parent: NodeId, kind: NodeKind) -> NodeId {
        let parent_node = self.nodes[parent.0 as usize]
            .as_ref()
            .expect("parent node must exist");
        assert!(
            !parent_node.kind.is_leaf(),
            "cannot add children to a leaf node"
        );

        let id = NodeId(self.nodes.len() as u32);
        let mut node = Node::new(kind);
        node.parent = Some(parent);
        node.own_version = self.next_version;
        node.subtree_version = self.next_version;
        self.next_version += 1;

        self.nodes.push(Some(node));

        // Add to parent's children list.
        self.nodes[parent.0 as usize]
            .as_mut()
            .unwrap()
            .children
            .push(id);

        // Propagate version up.
        self.propagate_version(parent);

        id
    }

    /// Insert a new `Union` parent between `id` and its current parent.
    ///
    /// - If `id` is the root, the new Union takes over the root slot and
    ///   `id` becomes its only child.
    /// - Otherwise, the new Union is inserted at `id`'s position in its
    ///   parent's children list, and `id` is reparented under the Union.
    ///
    /// Returns the new Union's `NodeId`. Versions propagate so any
    /// cached subtree_version above the insertion point invalidates.
    ///
    /// Used by the editor to "add a sibling" to a leaf — promote the
    /// leaf to a Union child, then append the requested new node as a
    /// second Union child.
    pub fn wrap_in_union(&mut self, id: NodeId) -> NodeId {
        use crate::node_kind::MaterialCombine;

        // Build the new Union node.
        let union_id = NodeId(self.nodes.len() as u32);
        let mut union_node = Node::new(NodeKind::Union {
            material_combine: MaterialCombine::Winner,
        });
        union_node.own_version = self.next_version;
        union_node.subtree_version = self.next_version;
        self.next_version += 1;

        // Find the existing node's parent (if any) and index under it.
        let old_parent = self.nodes[id.0 as usize]
            .as_ref()
            .expect("node must exist to wrap")
            .parent;

        // Place the Union node into the arena first so NodeId is stable.
        union_node.parent = old_parent;
        union_node.children.push(id);
        self.nodes.push(Some(union_node));

        // Reparent id under the new Union.
        self.nodes[id.0 as usize]
            .as_mut()
            .expect("node must exist after push")
            .parent = Some(union_id);

        match old_parent {
            None => {
                // id was the root — Union becomes the new root.
                self.root = union_id;
            }
            Some(p) => {
                // Swap id → union_id in the old parent's children vec.
                let parent_children = &mut self.nodes[p.0 as usize]
                    .as_mut()
                    .expect("old parent must exist")
                    .children;
                for c in parent_children.iter_mut() {
                    if *c == id {
                        *c = union_id;
                        break;
                    }
                }
                self.propagate_version(p);
            }
        }

        union_id
    }

    /// Remove a node and its entire subtree. Cannot remove the root.
    ///
    /// Returns `true` if the node was removed, `false` if it was the root or didn't exist.
    pub fn remove(&mut self, id: NodeId) -> bool {
        if id == self.root {
            return false;
        }

        let node = match &self.nodes[id.0 as usize] {
            Some(n) => n,
            None => return false,
        };

        let parent = node.parent;

        // Collect subtree to remove (BFS).
        let mut to_remove = vec![id];
        let mut cursor = 0;
        while cursor < to_remove.len() {
            let current = to_remove[cursor];
            if let Some(node) = &self.nodes[current.0 as usize] {
                to_remove.extend_from_slice(&node.children);
            }
            cursor += 1;
        }

        // Remove all nodes in subtree.
        for &nid in &to_remove {
            self.nodes[nid.0 as usize] = None;
        }

        // Remove from parent's children list.
        if let Some(parent_id) = parent {
            if let Some(parent_node) = &mut self.nodes[parent_id.0 as usize] {
                parent_node.children.retain(|c| *c != id);
            }
            self.propagate_version(parent_id);
        }

        true
    }

    /// Move a node to be a child of a new parent.
    ///
    /// Returns `true` on success. Fails if the move would create a cycle
    /// (new_parent is a descendant of `id`), or if `id` is the root.
    pub fn reparent(&mut self, id: NodeId, new_parent: NodeId) -> bool {
        if id == self.root {
            return false;
        }

        // Check for cycles: walk from new_parent to root, fail if we hit `id`.
        let mut cursor = Some(new_parent);
        while let Some(c) = cursor {
            if c == id {
                return false;
            }
            cursor = self.nodes[c.0 as usize].as_ref().and_then(|n| n.parent);
        }

        // Verify new parent is a combinator.
        if let Some(parent_node) = &self.nodes[new_parent.0 as usize] {
            if parent_node.kind.is_leaf() {
                return false;
            }
        } else {
            return false;
        }

        // Remove from old parent.
        let old_parent = self.nodes[id.0 as usize].as_ref().unwrap().parent;
        if let Some(old_parent_id) = old_parent {
            if let Some(old_parent_node) = &mut self.nodes[old_parent_id.0 as usize] {
                old_parent_node.children.retain(|c| *c != id);
            }
            self.propagate_version(old_parent_id);
        }

        // Add to new parent.
        self.nodes[new_parent.0 as usize]
            .as_mut()
            .unwrap()
            .children
            .push(id);
        self.nodes[id.0 as usize].as_mut().unwrap().parent = Some(new_parent);
        self.propagate_version(new_parent);

        true
    }

    /// Move a node to a new parent, inserting at a specific child index.
    ///
    /// Unifies reparent + reorder: the new parent can be the node's
    /// current parent (pure reorder) or a different combinator (reparent
    /// with ordering). `index` is clamped to `[0, children.len()]` of
    /// the new parent *after the node is removed from its old parent* —
    /// so the editor can just pass the visual insertion position it
    /// computed, without having to reason about whether the source and
    /// target are the same parent.
    ///
    /// Returns `true` on success. Fails if the move would create a
    /// cycle (new_parent is a descendant of `id`), if `id` is the root,
    /// or if the new parent is a leaf / missing.
    pub fn move_to(&mut self, id: NodeId, new_parent: NodeId, index: usize) -> bool {
        if id == self.root {
            return false;
        }

        // Cycle check: walk new_parent's ancestor chain; fail if we hit id.
        let mut cursor = Some(new_parent);
        while let Some(c) = cursor {
            if c == id {
                return false;
            }
            cursor = self.nodes[c.0 as usize].as_ref().and_then(|n| n.parent);
        }

        // New parent must exist and be a combinator.
        match &self.nodes[new_parent.0 as usize] {
            Some(n) if n.kind.is_leaf() => return false,
            None => return false,
            _ => {}
        }

        let old_parent = match self.nodes[id.0 as usize].as_ref() {
            Some(n) => match n.parent {
                Some(p) => p,
                None => return false, // root sanity (already checked above)
            },
            None => return false,
        };

        // Remove from old parent.
        if let Some(old_parent_node) = &mut self.nodes[old_parent.0 as usize] {
            old_parent_node.children.retain(|c| *c != id);
        }

        // Insert into new parent's children at the clamped index.
        let new_parent_node = self.nodes[new_parent.0 as usize].as_mut().unwrap();
        let clamped = index.min(new_parent_node.children.len());
        new_parent_node.children.insert(clamped, id);

        // Update the moved node's parent pointer.
        self.nodes[id.0 as usize].as_mut().unwrap().parent = Some(new_parent);

        // Propagate versions from both sides so any cached subtree_version
        // above either mount point is invalidated — old_parent's subtree
        // shrunk, new_parent's grew.
        self.propagate_version(old_parent);
        if old_parent != new_parent {
            self.propagate_version(new_parent);
        }

        true
    }

    /// Deep-clone a node and its entire subtree, inserting the copy as
    /// the next sibling of `id`. Returns the new subtree's root id, or
    /// `None` if `id` is the arena root (root-duplication is ambiguous
    /// — the tree has exactly one root and the copy would need to go
    /// somewhere) or if `id` is missing.
    ///
    /// Used by the "Duplicate" context-menu action; the resulting id
    /// is what the editor selects after the op.
    pub fn duplicate(&mut self, id: NodeId) -> Option<NodeId> {
        if id == self.root {
            return None;
        }

        // Capture parent + original sibling index so the clone drops in
        // right after `id` — preferred visual placement for "duplicate
        // this node."
        let parent_id = self.nodes[id.0 as usize].as_ref()?.parent?;
        let insert_at = self.nodes[parent_id.0 as usize]
            .as_ref()
            .and_then(|p| p.children.iter().position(|c| *c == id))
            .map(|i| i + 1)
            .unwrap_or_else(|| {
                self.nodes[parent_id.0 as usize]
                    .as_ref()
                    .map(|p| p.children.len())
                    .unwrap_or(0)
            });

        // BFS-clone the subtree. Walk the original ids in BFS order;
        // as each is cloned, stash its new id in `id_map` so later
        // children can rebuild their `children` vecs with the new ids.
        let mut id_map: std::collections::HashMap<NodeId, NodeId> =
            std::collections::HashMap::new();

        // Collect source ids in BFS order rooted at `id`.
        let mut bfs: Vec<NodeId> = vec![id];
        let mut cursor = 0;
        while cursor < bfs.len() {
            let current = bfs[cursor];
            if let Some(node) = &self.nodes[current.0 as usize] {
                bfs.extend_from_slice(&node.children);
            }
            cursor += 1;
        }

        // Allocate new slots and build id_map.
        for &src in &bfs {
            let new_id = NodeId(self.nodes.len() as u32);
            id_map.insert(src, new_id);
            // Push a placeholder; we'll fix up fields immediately after.
            let src_node = self.nodes[src.0 as usize].as_ref().unwrap().clone();
            self.nodes.push(Some(src_node));
        }

        // Fix up each cloned node: rewrite parent + children via id_map,
        // fresh own/subtree versions.
        for &src in &bfs {
            let new_id = *id_map.get(&src).unwrap();
            let own_version = self.next_version;
            self.next_version += 1;

            let node = self.nodes[new_id.0 as usize].as_mut().unwrap();
            node.own_version = own_version;
            node.subtree_version = own_version;
            // Clone-root's parent is the original's parent (set below);
            // descendants' parent is their mapped-to cloned parent.
            node.parent = if src == id {
                Some(parent_id)
            } else {
                node.parent.and_then(|p| id_map.get(&p).copied())
            };
            // Children always remap — every child of a cloned node was
            // itself part of the BFS so it has a mapping.
            let remapped: SmallVec<[NodeId; 2]> = node
                .children
                .iter()
                .map(|c| *id_map.get(c).unwrap_or(c))
                .collect();
            node.children = remapped;
        }

        // Insert the clone-root into the parent's children list right
        // after the original.
        let new_root = *id_map.get(&id).unwrap();
        if let Some(parent_node) = &mut self.nodes[parent_id.0 as usize] {
            let clamped = insert_at.min(parent_node.children.len());
            parent_node.children.insert(clamped, new_root);
        }
        self.propagate_version(parent_id);

        Some(new_root)
    }

    /// Move a node earlier among its siblings (swap with previous sibling).
    /// Returns `true` if the node was moved.
    pub fn move_up(&mut self, id: NodeId) -> bool {
        let parent_id = match self.nodes[id.0 as usize].as_ref() {
            Some(n) => match n.parent {
                Some(p) => p,
                None => return false, // root
            },
            None => return false,
        };

        let parent = self.nodes[parent_id.0 as usize].as_mut().unwrap();
        let pos = match parent.children.iter().position(|c| *c == id) {
            Some(p) => p,
            None => return false,
        };
        if pos == 0 {
            return false; // already first
        }
        parent.children.swap(pos, pos - 1);
        self.propagate_version(parent_id);
        true
    }

    /// Move a node later among its siblings (swap with next sibling).
    /// Returns `true` if the node was moved.
    pub fn move_down(&mut self, id: NodeId) -> bool {
        let parent_id = match self.nodes[id.0 as usize].as_ref() {
            Some(n) => match n.parent {
                Some(p) => p,
                None => return false,
            },
            None => return false,
        };

        let parent = self.nodes[parent_id.0 as usize].as_mut().unwrap();
        let pos = match parent.children.iter().position(|c| *c == id) {
            Some(p) => p,
            None => return false,
        };
        if pos + 1 >= parent.children.len() {
            return false; // already last
        }
        parent.children.swap(pos, pos + 1);
        self.propagate_version(parent_id);
        true
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::node_kind::{MaterialCombine, SphereParams};

    fn union_kind() -> NodeKind {
        NodeKind::Union {
            material_combine: MaterialCombine::Winner,
        }
    }

    fn sphere_kind() -> NodeKind {
        NodeKind::Sphere(SphereParams::default())
    }

    #[test]
    fn create_and_query_root() {
        let obj = ProceduralObject::new(union_kind());
        assert_eq!(obj.node_count(), 1);
        assert!(obj.get(obj.root()).is_some());
    }

    #[test]
    fn add_children() {
        let mut obj = ProceduralObject::new(union_kind());
        let a = obj.add_child(obj.root(), sphere_kind());
        let b = obj.add_child(obj.root(), sphere_kind());
        assert_eq!(obj.node_count(), 3);
        assert_eq!(obj.get(obj.root()).unwrap().children.len(), 2);
        assert_eq!(obj.get(a).unwrap().parent, Some(obj.root()));
        assert_eq!(obj.get(b).unwrap().parent, Some(obj.root()));
    }

    #[test]
    #[should_panic(expected = "cannot add children to a leaf")]
    fn add_child_to_leaf_panics() {
        let mut obj = ProceduralObject::new(union_kind());
        let leaf = obj.add_child(obj.root(), sphere_kind());
        obj.add_child(leaf, sphere_kind());
    }

    #[test]
    fn remove_subtree() {
        let mut obj = ProceduralObject::new(union_kind());
        let sub = obj.add_child(obj.root(), union_kind());
        let _leaf = obj.add_child(sub, sphere_kind());
        assert_eq!(obj.node_count(), 3);

        assert!(obj.remove(sub));
        assert_eq!(obj.node_count(), 1);
        assert_eq!(obj.get(obj.root()).unwrap().children.len(), 0);
    }

    #[test]
    fn cannot_remove_root() {
        let mut obj = ProceduralObject::new(union_kind());
        assert!(!obj.remove(obj.root()));
    }

    #[test]
    fn reparent_basic() {
        let mut obj = ProceduralObject::new(union_kind());
        let a = obj.add_child(obj.root(), union_kind());
        let b = obj.add_child(obj.root(), union_kind());
        let leaf = obj.add_child(a, sphere_kind());

        assert!(obj.reparent(leaf, b));
        assert_eq!(obj.get(a).unwrap().children.len(), 0);
        assert_eq!(obj.get(b).unwrap().children.len(), 1);
        assert_eq!(obj.get(leaf).unwrap().parent, Some(b));
    }

    #[test]
    fn reparent_prevents_cycle() {
        let mut obj = ProceduralObject::new(union_kind());
        let a = obj.add_child(obj.root(), union_kind());
        let b = obj.add_child(a, union_kind());

        // Moving `a` under `b` would create a cycle.
        assert!(!obj.reparent(a, b));
    }

    #[test]
    fn reparent_to_leaf_fails() {
        let mut obj = ProceduralObject::new(union_kind());
        let a = obj.add_child(obj.root(), union_kind());
        let leaf = obj.add_child(obj.root(), sphere_kind());

        assert!(!obj.reparent(a, leaf));
    }

    #[test]
    fn version_propagates_on_add() {
        let mut obj = ProceduralObject::new(union_kind());
        let v_before = obj.get(obj.root()).unwrap().subtree_version;
        let _a = obj.add_child(obj.root(), sphere_kind());
        let v_after = obj.get(obj.root()).unwrap().subtree_version;
        assert!(v_after > v_before);
    }

    #[test]
    fn set_transform_bumps_version() {
        let mut obj = ProceduralObject::new(union_kind());
        let a = obj.add_child(obj.root(), sphere_kind());
        let v_before = obj.get(obj.root()).unwrap().subtree_version;
        obj.set_transform(a, Affine3A::from_translation(glam::Vec3::X));
        let v_after = obj.get(obj.root()).unwrap().subtree_version;
        assert!(v_after > v_before);
    }

    #[test]
    fn move_to_reorders_within_same_parent() {
        let mut obj = ProceduralObject::new(union_kind());
        let a = obj.add_child(obj.root(), sphere_kind());
        let b = obj.add_child(obj.root(), sphere_kind());
        let c = obj.add_child(obj.root(), sphere_kind());
        // Move c to front: [a, b, c] → [c, a, b]
        assert!(obj.move_to(c, obj.root(), 0));
        assert_eq!(
            obj.get(obj.root()).unwrap().children.as_slice(),
            &[c, a, b]
        );
    }

    #[test]
    fn move_to_mid_position_same_parent() {
        let mut obj = ProceduralObject::new(union_kind());
        let a = obj.add_child(obj.root(), sphere_kind());
        let b = obj.add_child(obj.root(), sphere_kind());
        let c = obj.add_child(obj.root(), sphere_kind());
        // Move a between b and c: [a, b, c] → [b, a, c]
        assert!(obj.move_to(a, obj.root(), 1));
        assert_eq!(
            obj.get(obj.root()).unwrap().children.as_slice(),
            &[b, a, c]
        );
    }

    #[test]
    fn move_to_across_parents() {
        let mut obj = ProceduralObject::new(union_kind());
        let p1 = obj.add_child(obj.root(), union_kind());
        let p2 = obj.add_child(obj.root(), union_kind());
        let leaf = obj.add_child(p1, sphere_kind());
        assert!(obj.move_to(leaf, p2, 0));
        assert_eq!(obj.get(p1).unwrap().children.len(), 0);
        assert_eq!(obj.get(p2).unwrap().children.as_slice(), &[leaf]);
        assert_eq!(obj.get(leaf).unwrap().parent, Some(p2));
    }

    #[test]
    fn move_to_clamps_overflow_index() {
        let mut obj = ProceduralObject::new(union_kind());
        let _a = obj.add_child(obj.root(), sphere_kind());
        let b = obj.add_child(obj.root(), sphere_kind());
        // index=99 → clamp to end
        assert!(obj.move_to(b, obj.root(), 99));
        assert_eq!(obj.get(obj.root()).unwrap().children.last().copied(), Some(b));
    }

    #[test]
    fn move_to_rejects_root() {
        let mut obj = ProceduralObject::new(union_kind());
        let a = obj.add_child(obj.root(), union_kind());
        assert!(!obj.move_to(obj.root(), a, 0));
    }

    #[test]
    fn move_to_rejects_cycle() {
        let mut obj = ProceduralObject::new(union_kind());
        let a = obj.add_child(obj.root(), union_kind());
        let b = obj.add_child(a, union_kind());
        // Moving a into b would make b its own ancestor.
        assert!(!obj.move_to(a, b, 0));
    }

    #[test]
    fn move_to_rejects_leaf_parent() {
        let mut obj = ProceduralObject::new(union_kind());
        let a = obj.add_child(obj.root(), union_kind());
        let leaf = obj.add_child(obj.root(), sphere_kind());
        assert!(!obj.move_to(a, leaf, 0));
    }

    #[test]
    fn duplicate_leaf() {
        let mut obj = ProceduralObject::new(union_kind());
        let a = obj.add_child(obj.root(), sphere_kind());
        let dup = obj.duplicate(a).expect("leaf duplicates");
        assert_ne!(dup, a);
        let kids = obj.get(obj.root()).unwrap().children.clone();
        // Original followed by clone.
        assert_eq!(kids.as_slice(), &[a, dup]);
        assert_eq!(obj.get(dup).unwrap().parent, Some(obj.root()));
    }

    #[test]
    fn duplicate_subtree_deep_copy() {
        let mut obj = ProceduralObject::new(union_kind());
        let sub = obj.add_child(obj.root(), union_kind());
        let leaf_a = obj.add_child(sub, sphere_kind());
        let leaf_b = obj.add_child(sub, sphere_kind());

        let dup = obj.duplicate(sub).expect("subtree duplicates");
        // Dup is a new node.
        assert_ne!(dup, sub);
        // Root now has two combinator children.
        assert_eq!(obj.get(obj.root()).unwrap().children.as_slice(), &[sub, dup]);
        // Dup has two children that are NEW ids (not the original leaves).
        let dup_kids = obj.get(dup).unwrap().children.clone();
        assert_eq!(dup_kids.len(), 2);
        for &c in dup_kids.iter() {
            assert_ne!(c, leaf_a);
            assert_ne!(c, leaf_b);
            assert_eq!(obj.get(c).unwrap().parent, Some(dup));
        }
    }

    #[test]
    fn duplicate_rejects_root() {
        let mut obj = ProceduralObject::new(union_kind());
        assert!(obj.duplicate(obj.root()).is_none());
    }

    #[test]
    fn iter_ids_skips_removed() {
        let mut obj = ProceduralObject::new(union_kind());
        let a = obj.add_child(obj.root(), sphere_kind());
        let _b = obj.add_child(obj.root(), sphere_kind());
        obj.remove(a);
        let ids: Vec<_> = obj.iter_ids().collect();
        assert_eq!(ids.len(), 2); // root + b
    }
}
