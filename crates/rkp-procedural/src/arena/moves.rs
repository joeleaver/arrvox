//! Re-parenting and re-ordering: move nodes within or across parents,
//! deep-copy a subtree, swap with siblings.

use smallvec::SmallVec;

use super::{NodeId, ProceduralObject};

impl ProceduralObject {
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

        // Evict existing over-cap children from the new parent before
        // insertion — single-child effects swap their current child out
        // to the grandparent on any drop. Runs *after* we've detached
        // `id` from its old parent so the count math below is clean
        // (if id was already under new_parent, it no longer appears in
        // its children list and doesn't double-count against the cap).
        self.evict_over_cap(new_parent, 1);

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
}
