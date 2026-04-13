//! Version tracking and push-based invalidation.
//!
//! When a parameter changes on a node, the node's `own_version` is bumped and
//! `subtree_version` is propagated up the parent chain. Downstream consumers
//! (caching, octree re-evaluation) compare versions to detect staleness.

use crate::arena::{NodeId, ProceduralObject};

/// Bump a node's version after a parameter change, propagating up to root.
///
/// Call this after modifying a node's `NodeKind` parameters (not transform —
/// `set_transform` handles its own version bump).
pub fn bump_node_version(obj: &mut ProceduralObject, id: NodeId) {
    // We need to use the object's internal versioning. The approach:
    // 1. Bump the node's own_version to a new unique value.
    // 2. Walk parent chain, updating subtree_version at each level.
    //
    // Since ProceduralObject's next_version is private, we use set_transform
    // with the existing transform as a version-bumping mechanism... but that's
    // a hack. Instead, let's expose a proper mutation method.
    //
    // For now, we'll use a dedicated method on ProceduralObject.
    obj.bump_version(id);
}

/// Check whether a node's subtree has changed since the given version.
pub fn is_stale(obj: &ProceduralObject, id: NodeId, cached_version: u64) -> bool {
    match obj.get(id) {
        Some(node) => node.subtree_version > cached_version,
        None => false,
    }
}

/// Get the current subtree version for a node.
pub fn subtree_version(obj: &ProceduralObject, id: NodeId) -> u64 {
    obj.get(id).map_or(0, |n| n.subtree_version)
}
