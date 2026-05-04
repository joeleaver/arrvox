//! Tree-structural mutations: add/remove subtrees and wrap nodes in new parents.

use smallvec::SmallVec;

use crate::node_kind::NodeKind;

use super::{Node, NodeId, ProceduralObject};

impl ProceduralObject {
    /// Add a child node under the given parent. Returns the new node's ID.
    ///
    /// Panics if `parent` doesn't exist or is a leaf shape (leaves can't have children).
    ///
    /// If `parent`'s kind has a child cap (e.g. a single-child effect
    /// like `NoiseDisplace` that already has one child), the existing
    /// over-cap children are evicted to `parent`'s grandparent — they
    /// become siblings of `parent` immediately after it — before the
    /// new child is appended. Falls back to no eviction if `parent` is
    /// the root (nowhere to evict to); in that case the cap is relaxed
    /// for this call and the caller is expected to clean up.
    pub fn add_child(&mut self, parent: NodeId, kind: NodeKind) -> NodeId {
        let parent_node = self.nodes[parent.0 as usize]
            .as_ref()
            .expect("parent node must exist");
        assert!(
            !parent_node.kind.is_leaf(),
            "cannot add children to a leaf node"
        );

        // Evict anything that would push the count past the cap. One
        // incoming child, so we need room for exactly one.
        self.evict_over_cap(parent, 1);

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

    /// If `parent`'s kind has a child cap, evict existing children to
    /// the grandparent so that after adding `incoming_count` more, the
    /// cap is respected. Evictees are inserted into the grandparent's
    /// children list right after `parent` (so the tree reads "effect,
    /// bumped-out thing" and the user can see what got pushed aside).
    ///
    /// `parent` being the root, missing, or having no grandparent is a
    /// no-op — at that point there's nowhere to evict to, and the
    /// caller has to live with a transient over-cap state. The UI path
    /// hides the "+" button on full effects, so this bail-out only
    /// triggers for edge cases (root-level effects via MCP etc.).
    pub(super) fn evict_over_cap(&mut self, parent: NodeId, incoming_count: usize) {
        let Some(cap) = self
            .nodes
            .get(parent.0 as usize)
            .and_then(|n| n.as_ref())
            .and_then(|n| n.kind.max_children())
        else {
            return;
        };
        // Snapshot the current children + grandparent so we can mutate
        // both without fighting the borrow checker.
        let (current, grandparent) = {
            let node = self.nodes[parent.0 as usize].as_ref().unwrap();
            (node.children.clone(), node.parent)
        };
        let total = current.len() + incoming_count;
        if total <= cap {
            return;
        }
        let Some(grandparent) = grandparent else {
            // parent is the root — no grandparent to evict into. Leave
            // the over-cap state alone; evaluator + flatten ignore the
            // extras and the next drop/move into a non-root parent
            // will sort itself out.
            return;
        };
        // Number to evict: the oldest children, so the UI's "this is
        // what got bumped" cursor lands near the top of the list.
        let evict_count = total - cap;
        let evict: SmallVec<[NodeId; 2]> = current.iter().take(evict_count).copied().collect();

        // Find `parent`'s position in the grandparent's children so
        // evictees land right after it.
        let after_parent = self.nodes[grandparent.0 as usize]
            .as_ref()
            .and_then(|gp| gp.children.iter().position(|c| *c == parent))
            .map(|i| i + 1)
            .unwrap_or_else(|| {
                self.nodes[grandparent.0 as usize]
                    .as_ref()
                    .map(|gp| gp.children.len())
                    .unwrap_or(0)
            });

        // Unhook from parent's children list (in reverse so indices stay valid).
        {
            let parent_node = self.nodes[parent.0 as usize].as_mut().unwrap();
            parent_node.children.retain(|c| !evict.contains(c));
        }

        // Insert each evictee into the grandparent's children and
        // repoint its parent pointer.
        let gp_node = self.nodes[grandparent.0 as usize].as_mut().unwrap();
        for (i, &victim) in evict.iter().enumerate() {
            gp_node.children.insert(after_parent + i, victim);
        }
        for &victim in evict.iter() {
            if let Some(v) = self.nodes[victim.0 as usize].as_mut() {
                v.parent = Some(grandparent);
            }
        }

        self.propagate_version(parent);
        self.propagate_version(grandparent);
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
    /// Migration-only: wrap a leaf root in a `Root` container so
    /// legacy saved scenes (pre-Root, where the single-primitive
    /// default had a Sphere at the root) can still accept children
    /// in the new model. Returns the new Root's id. Callers should
    /// only invoke this when `self.root()` is a leaf; calling on an
    /// already-containerized root will double-wrap.
    pub fn wrap_in_root(&mut self) -> NodeId {
        let root_id = self.root;

        let new_root_id = NodeId(self.nodes.len() as u32);
        let mut new_root_node = Node::new(NodeKind::Root);
        new_root_node.own_version = self.next_version;
        new_root_node.subtree_version = self.next_version;
        self.next_version += 1;
        new_root_node.children.push(root_id);
        self.nodes.push(Some(new_root_node));

        // Re-parent the old root leaf under the new Root.
        self.nodes[root_id.0 as usize]
            .as_mut()
            .expect("old root must exist")
            .parent = Some(new_root_id);
        self.root = new_root_id;
        new_root_id
    }

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
}
