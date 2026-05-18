//! Per-mutation log used by [`SparseOctree`] to capture every write to
//! `nodes[]` and `internal_attr_index[]` while a caller (typically
//! [`apply_delta`](crate::sculpt::apply_delta)) is mutating the tree.
//!
//! Sculpt's per-stamp upload path uses the log to write only the touched
//! slots into [`OctreeAllocator`](crate::OctreeAllocator)'s packed GPU
//! buffer — closing the latent CPU↔GPU sync bug where mutations to
//! `cpu_octree.nodes` never made it back to the packed buffer, and at
//! the same time enabling delta uploads on the octree.
//!
//! V1 is intentionally minimal: append-only `Vec`s of `(local_idx,
//! value)` pairs, plus the pre-mutation node count so the caller can
//! detect tree growth (which requires a re-allocation since
//! `OctreeAllocator` slots are exact-fit).

/// Records every write made to `nodes` and `internal_attr_index` during
/// a single mutation session. Indices are local (0-based within the
/// octree's own packed buffer); the caller translates to absolute
/// `root_offset + local_idx` when applying to `OctreeAllocator`.
#[derive(Debug, Default, Clone)]
pub struct OctreeMutationLog {
    /// `(local_node_idx, new_value)` pairs in write order. The same
    /// index may appear multiple times (a sibling being subdivided then
    /// re-collapsed); the last write wins.
    pub node_writes: Vec<(u32, u32)>,
    /// `(local_attr_idx, new_value)` pairs in write order.
    pub attr_writes: Vec<(u32, u32)>,
    /// Number of nodes at the time the log was started. Combined with
    /// the tree's current `node_count()` lets the caller detect growth
    /// — every mutation past `initial_node_count` lives in the appended
    /// region.
    pub initial_node_count: u32,
}

impl OctreeMutationLog {
    pub fn new(initial_node_count: u32) -> Self {
        Self {
            node_writes: Vec::new(),
            attr_writes: Vec::new(),
            initial_node_count,
        }
    }

    /// True when the tree grew past its initial node count. Growth
    /// implies the existing `OctreeAllocator` slot is too small and a
    /// re-allocation is required.
    pub fn grew(&self, current_node_count: u32) -> bool {
        current_node_count > self.initial_node_count
    }

    /// Combined byte count tracked across both writes (for telemetry).
    pub fn bytes_tracked(&self) -> u64 {
        ((self.node_writes.len() + self.attr_writes.len()) * std::mem::size_of::<u32>()) as u64
    }

    /// True when no mutations were recorded.
    pub fn is_empty(&self) -> bool {
        self.node_writes.is_empty() && self.attr_writes.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn log_starts_empty() {
        let log = OctreeMutationLog::new(8);
        assert!(log.is_empty());
        assert_eq!(log.initial_node_count, 8);
        assert!(!log.grew(8));
        assert!(log.grew(9));
    }

    #[test]
    fn writes_recorded() {
        let mut log = OctreeMutationLog::new(8);
        log.node_writes.push((0, 42));
        log.attr_writes.push((1, 99));
        assert!(!log.is_empty());
        assert_eq!(log.node_writes, vec![(0, 42)]);
        assert_eq!(log.attr_writes, vec![(1, 99)]);
        assert_eq!(log.bytes_tracked(), 8);
    }
}
