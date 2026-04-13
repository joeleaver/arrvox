//! Arena-based tree structure for procedural objects.
//!
//! Nodes are stored in a flat `Vec` with parent pointers. This avoids
//! `Box<Node>` reference cycles and enables push-based version invalidation
//! by walking the parent chain.

use glam::Affine3A;
use smallvec::SmallVec;

use crate::node_kind::NodeKind;

/// Index into the node arena.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(pub u32);

/// A single node in the procedural tree.
#[derive(Debug, Clone)]
pub struct Node {
    pub kind: NodeKind,
    pub parent: Option<NodeId>,
    pub children: SmallVec<[NodeId; 2]>,
    /// Local transform (relative to parent).
    pub transform: Affine3A,
    /// Bumped when this node's own parameters change.
    pub own_version: u64,
    /// Max of own_version and all descendants' subtree_versions.
    /// Used to detect whether cached data is stale.
    pub subtree_version: u64,
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
#[derive(Debug, Clone)]
pub struct ProceduralObject {
    nodes: Vec<Option<Node>>,
    root: NodeId,
    /// Global version counter — incremented on every mutation.
    next_version: u64,
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
    fn iter_ids_skips_removed() {
        let mut obj = ProceduralObject::new(union_kind());
        let a = obj.add_child(obj.root(), sphere_kind());
        let _b = obj.add_child(obj.root(), sphere_kind());
        obj.remove(a);
        let ids: Vec<_> = obj.iter_ids().collect();
        assert_eq!(ids.len(), 2); // root + b
    }
}
