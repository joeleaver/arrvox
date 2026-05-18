//! Paint epoch + brush overlay + sphere paint application.
//!
//! Sibling impl block on `ArvxSceneManager`. Owns `paint_epoch` and
//! `apply_paint_sphere`. Reads `pub(super)` fields on the struct
//! directly.

use arvx_core::LeafAttr;

use super::manager::ArvxSceneManager;
use super::types::AssetInfo;

impl ArvxSceneManager {
    pub fn paint_epoch(&self) -> u64 {
        self.paint_epoch.load(std::sync::atomic::Ordering::Acquire)
    }

    /// Clone of the paint-epoch atomic for lock-free sim-side reads.
    pub fn paint_epoch_handle(&self) -> std::sync::Arc<std::sync::atomic::AtomicU64> {
        self.paint_epoch.clone()
    }

    fn bump_paint_epoch(&mut self) {
        self.paint_epoch
            .fetch_add(1, std::sync::atomic::Ordering::Release);
    }

    /// Return a byte slice of `leaf_attr_pool` covering slots
    /// `[slot_start, slot_start + slot_count)`. Used by the render
    /// thread to `queue.write_buffer` only the dirty range instead of
    /// the whole buffer.
    pub fn leaf_attr_slice_bytes(&self, slot_start: u32, slot_count: u32) -> &[u8] {
        let bytes_per = std::mem::size_of::<LeafAttr>();
        let start = slot_start as usize * bytes_per;
        let end = (slot_start as usize + slot_count as usize) * bytes_per;
        let full = self.leaf_attr_pool.as_bytes();
        if end > full.len() {
            return &[];
        }
        &full[start..end]
    }

    /// Byte slice of the color pool covering slots
    /// `[slot_start, slot_start + slot_count)`. Sibling of
    /// `leaf_attr_slice_bytes` for the color companion buffer.
    pub fn color_slice_bytes(&self, slot_start: u32, slot_count: u32) -> &[u8] {
        let bytes_per = 4; // u32 per slot
        let start = slot_start as usize * bytes_per;
        let end = (slot_start as usize + slot_count as usize) * bytes_per;
        let full = self.leaf_attr_pool.color_bytes();
        if end > full.len() {
            return &[];
        }
        &full[start..end]
    }


    // ── Paint orchestrator ───────────────────────────────────────────

    /// Apply one brush stamp to every leaf whose voxel center sits inside a
    /// world-space sphere centered at `brush_center_world`.
    ///
    /// `asset` describes the entity's geometry allocation (octree +
    /// leaf_attr range). `entity_world` is the entity's full world
    /// transform (from its Transform component); paint transforms the
    /// brush center into the entity's local frame before querying the
    /// octree, which keeps the brush aligned with rotated / scaled
    /// objects.
    ///
    /// Returns the number of leaves that received a write. The caller
    /// (editor command handler) is responsible for:
    ///
    /// * Skipping procedural / generator-owned entities (paint would be
    ///   wiped on the next rebake).
    /// * Converting its editor-side `PaintMode` into the right
    ///   [`PaintStamp`] variant.
    /// * Ensuring this is called on the sim / engine thread that owns
    ///   the scene_mgr Mutex — paint is a geometry mutation.
    /// Apply a brush stamp into a per-instance overlay.
    ///
    /// The asset's `leaf_attr_pool` is read-only here — paint mutations
    /// land in `overlay`, which is per-entity. This is the bug-fix for
    /// "paint one of N shared-asset instances paints all": before
    /// per-instance overlays, a brush write modified the shared pool
    /// and visibly affected every sibling instance pointing at the
    /// same `octree_root`.
    ///
    /// "Current" attr/color for a leaf is sourced from the overlay if
    /// already present (so multi-stroke compounding works), else from
    /// the asset's pool. The result is upserted into the overlay; the
    /// pool is never written.
    pub fn apply_paint_sphere(
        &mut self,
        asset: &AssetInfo,
        entity_world: glam::Affine3A,
        brush_center_world: glam::Vec3,
        radius: f32,
        strength: f32,
        falloff: f32,
        stamp: crate::paint::PaintStamp,
        overlay: &mut arvx_core::LeafAttrOverlay,
    ) -> usize {
        use arvx_core::scene_node::SpatialHandle;
        if radius <= 0.0 || strength <= 0.0 {
            return 0;
        }

        let SpatialHandle::Octree { root_offset, depth, base_voxel_size, .. } =
            asset.spatial
        else {
            // Non-octree spatial handles don't have leaf_attr data to paint.
            return 0;
        };

        // World → object-local. Use the affine inverse so non-uniform
        // scale still works correctly. The brush radius must be scaled
        // inversely too — otherwise scaling an object up would shrink
        // the effective paint footprint.
        let inv_world = entity_world.inverse();
        let center_local = inv_world.transform_point3(brush_center_world);
        // Radius in object-local units — average the three scale axes.
        // Exact local radius is the world radius divided by the
        // directional scale; painting is approximate enough that the
        // mean is a good default.
        let (scale, _, _) = entity_world.to_scale_rotation_translation();
        let mean_scale = (scale.x.abs() + scale.y.abs() + scale.z.abs()) / 3.0;
        let local_radius = radius / mean_scale.max(1e-6);

        let hits = crate::paint::leaves_in_sphere(
            self.octree.data(),
            root_offset,
            depth,
            base_voxel_size,
            &self.brick_pool,
            asset.grid_origin,
            center_local,
            local_radius,
        );

        if hits.is_empty() {
            return 0;
        }

        // Validate slots against this asset's allocated range — the
        // packed buffer's leaves carry scene-global slot ids, so a
        // corrupted octree could in theory produce ids outside the
        // asset's range. Clamp defensively.
        let slot_lo = asset.leaf_attr_slot_start;
        let slot_hi = slot_lo + asset.leaf_attr_slot_count;

        // Accumulate new/updated entries into a batch and commit at the
        // end via `upsert_batch`. Per-entry `upsert` is O(N) on the
        // sorted vec; a stamp touching K leaves on an overlay of size
        // N is therefore O(K · N), which on a long drag (N grows
        // each stamp) blew up to ~1 fps. Batched merge-pass is
        // O(N + K log K).
        let mut batch: Vec<arvx_core::OverlayEntry> = Vec::with_capacity(hits.len());
        for hit in &hits {
            if hit.leaf_slot < slot_lo || hit.leaf_slot >= slot_hi {
                continue;
            }
            let weight = crate::paint::brush_weight(
                hit.distance, local_radius, strength, falloff,
            );
            if weight <= 0.0 {
                continue;
            }
            let paint_slot = hit.leaf_slot;

            // Read the current (attr, color) — overlay if present, else
            // the asset's pool. Multi-stroke compounding (gradual blend,
            // erase fade-out) needs to read the previously-painted
            // value, not always the base.
            let cur_overlay = overlay.get(paint_slot);
            let (cur_attr, cur_color) = match cur_overlay {
                Some(e) => (e.attr(), e.color_packed),
                None => (
                    *self.leaf_attr_pool.get(paint_slot),
                    self.leaf_attr_pool.color(paint_slot),
                ),
            };

            match stamp {
                crate::paint::PaintStamp::Material { material_id } => {
                    let new_attr = crate::paint::compute_painted_attr(
                        cur_attr, material_id, weight,
                    );
                    if new_attr == cur_attr && cur_overlay.is_none() {
                        // No-op write that wasn't already in the overlay
                        // — skip so unpainted material-only brushes
                        // don't bloat the overlay with identity entries.
                        continue;
                    }
                    batch.push(arvx_core::OverlayEntry::from_parts(
                        paint_slot, new_attr, cur_color,
                    ));
                }
                crate::paint::PaintStamp::Color { rgb } => {
                    let new_color = crate::paint::compute_painted_color(
                        cur_color, rgb, weight,
                    );
                    if new_color == cur_color && cur_overlay.is_none() {
                        continue;
                    }
                    batch.push(arvx_core::OverlayEntry::from_parts(
                        paint_slot, cur_attr, new_color,
                    ));
                }
                crate::paint::PaintStamp::Erase => {
                    let new_color = crate::paint::compute_erased_color(
                        cur_color, weight,
                    );
                    if new_color == cur_color && cur_overlay.is_none() {
                        continue;
                    }
                    batch.push(arvx_core::OverlayEntry::from_parts(
                        paint_slot, cur_attr, new_color,
                    ));
                }
            }
        }
        let written = batch.len();
        if written > 0 {
            overlay.upsert_batch(batch);
        }

        if written > 0 {
            // Paint mutations are now per-instance and live outside
            // the leaf_attr/color pools — no slot-range dirty
            // tracking on those buffers anymore. The overlay buffer
            // is rebuilt+uploaded each frame from sim's `paint_overlays`
            // map; engine bumps `paint_epoch` after the call so the
            // next render frame picks up the new content.
            self.bump_paint_epoch();
        }
        written
    }
}
