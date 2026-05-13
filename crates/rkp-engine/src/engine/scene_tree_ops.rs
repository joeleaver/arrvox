//! Scene-tree reorder (drag-drop within the outliner panel).
//!
//! Updates the per-entity `TreeOrder` side-map so the editor's sorted
//! view reflects the new arrangement. Persisted in the scene file.

use super::state::EngineState;

impl EngineState {
    /// Apply an editor-computed reorder: set `entity`'s parent and
    /// tree_order. All ordering math is editor-side (it has the visual
    /// tree context); this handler is a thin applier. Validates cycle
    /// (reject dropping inside own subtree) and self-parent.
    pub(crate) fn handle_reorder(
        &mut self,
        entity_id: uuid::Uuid,
        new_parent: Option<uuid::Uuid>,
        new_order: f64,
    ) {
        if Some(entity_id) == new_parent {
            return; // self-parent
        }
        let Some(entity) = self.uuid_to_entity.get(&entity_id).copied() else { return };

        // Cycle check: walk the proposed parent chain upward, rejecting
        // if we pass through `entity`. No-op when parent is root.
        if let Some(pid) = new_parent {
            let mut cursor = self.uuid_to_entity.get(&pid).copied();
            while let Some(c) = cursor {
                if c == entity {
                    return;
                }
                cursor = self
                    .world
                    .get::<&crate::components::Parent>(c)
                    .ok()
                    .and_then(|p| self.uuid_to_entity.get(&p.parent_id).copied());
            }
        }

        self.entity_tree_order.insert(entity, new_order);
        if new_order + 1.0 > self.next_tree_order {
            self.next_tree_order = new_order + 1.0;
        }
        match new_parent {
            Some(pid) => {
                // hecs `insert_one` replaces an existing component.
                let _ = self.world.insert_one(
                    entity,
                    crate::components::Parent { parent_id: pid },
                );
            }
            None => {
                // Only remove if present — removing an absent component
                // returns MissingComponent, which we'd discard anyway.
                if self.world.get::<&crate::components::Parent>(entity).is_ok() {
                    let _ = self.world.remove_one::<crate::components::Parent>(entity);
                }
            }
        }

        self.scene_dirty = true;
        self.gpu_objects_dirty.mark_all();
    }
}
