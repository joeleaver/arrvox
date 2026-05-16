//! Click-pick readback handling and drag-drop commit.
//!
//! `process_pick_result` consumes the per-frame GPU pick payload from
//! the render thread and resolves it to a scene entity, procedural
//! node, or a queued drop commit. Also owns the sim-side CPU raycast
//! against ghost primitives that overrides the G-buffer decode when a
//! ghost silhouette is drawn at the click pixel.

use super::state::{DragPreviewKind, EngineState, PendingDrop, PendingDropAction};

impl EngineState {
    /// Apply a [`PickResult`] from the render thread. Mirrors the old
    /// `drain_pick_result` logic: ghost priority on BUILD raymarch,
    /// otherwise scene_id → entity for MAIN voxel and `selected_entity` /
    /// `selected_procedural_node` updates accordingly.
    pub(crate) fn process_pick_result(&mut self, pr: crate::render_frame::PickResult) {
        use crate::render_frame::PickKind;
        use crate::viewport::ViewportId;

        // Acknowledge: this pick request has been served, so stop
        // re-shipping it in subsequent snapshots. (See the "re-ship
        // every snapshot" rationale in `submit_render_frame`.) A
        // brand-new click later will set `pending_pick` again.
        self.pending_pick = None;

        // Paint stamp: if the pick was issued by a `PaintAtPixel`
        // command, consume it here before selection / drag-preview
        // routing. `raw_payload[0]` is the `gpu_idx` for MAIN material
        // picks (0xFFFFFFFF on sky); `position` is the world-space
        // surface hit (None on sky).
        //
        // Paint mode is strictly selection-locked: the brush only
        // affects whatever entity is selected in the scene tree. A
        // paint-click on anything else (including sky) is a no-op —
        // it never claims selection and never deselects. The only
        // way to change the locked entity during paint is via the
        // scene tree.
        if let Some(settings) = self.paint_pick_settings.take() {
            let hit_gpu_idx = pr.raw_payload[0];
            let hit_entity: Option<hecs::Entity> = if hit_gpu_idx != u32::MAX {
                let gpu_idx = hit_gpu_idx as usize;
                self.gpu_to_entity.get(gpu_idx).copied()
            } else {
                None
            };

            // Strict lock — must hit the selected entity exactly.
            // Hits on other geometry, sky, or with no selection at
            // all are misses.
            let target_entity = self
                .selected_entity
                .filter(|sel| hit_entity == Some(*sel));

            if let (Some(entity), Some(world_pos)) = (target_entity, pr.position) {
                let _ = self.apply_paint_stamp(
                    entity,
                    world_pos,
                    settings.radius,
                    settings.strength,
                    settings.falloff,
                    settings.color,
                    settings.mode,
                    settings.material_id,
                );
            }
            self.in_flight_pick_ghost = None;
            return;
        }

        // Sculpt stamp: mirrors the paint branch above. Selection-locked
        // the same way — the brush only affects the currently-selected
        // entity. Phase 0 just logs the resolved op; Phase 1 swaps the
        // stub for real octree mutation.
        if let Some(settings) = self.sculpt_pick_settings.take() {
            // Log the click-to-mutation latency. This is the total wall
            // time from SculptAtPixel arriving at the sim to the stamp
            // about to apply — spans pick round-trip + sim ticks +
            // render frames + GPU work.
            let pending_at = self.sculpt_pending_at.take();
            let hit_gpu_idx = pr.raw_payload[0];
            let hit_entity: Option<hecs::Entity> = if hit_gpu_idx != u32::MAX {
                let gpu_idx = hit_gpu_idx as usize;
                self.gpu_to_entity.get(gpu_idx).copied()
            } else {
                None
            };
            let target_entity = self
                .selected_entity
                .filter(|sel| hit_entity == Some(*sel));
            if let (Some(entity), Some(world_pos)) = (target_entity, pr.position) {
                if let Some(t0) = pending_at {
                    eprintln!(
                        "[sculpt-latency] click→pick→sculpt_start={:.2}ms",
                        t0.elapsed().as_secs_f64() * 1000.0,
                    );
                }
                let _ = self.apply_sculpt_stamp(
                    entity,
                    world_pos,
                    settings.radius,
                    settings.falloff_curve,
                    settings.strength,
                    settings.stroke_seq,
                    settings.mode,
                    settings.material_id,
                );
            }
            self.in_flight_pick_ghost = None;
            return;
        }

        // Drop-on-geometry: if a drag-drop is queued for this viewport,
        // consume it instead of running selection. The pick was issued
        // purely to get the world-space surface position at the drop
        // pixel; selection should not change from a drop.
        if let Some(drop) = self.pending_drop.as_ref() {
            if drop.viewport == pr.viewport {
                let drop = self.pending_drop.take().unwrap();
                self.handle_drop(drop, pr.position);
                self.in_flight_pick_ghost = None;
                return;
            }
        }

        // Drag-preview: move the preview to the freshest surface snap.
        // Skip the selection path entirely — picks issued while
        // dragging are purely for positioning.
        if let Some(preview) = self.drag_preview.as_ref() {
            if preview.viewport == pr.viewport {
                let kind = preview.kind.clone();
                let (cx, cy) = preview.last_cursor;
                let vp = preview.viewport;

                // Self-hit detection only matters for the model path —
                // generators have nothing spawned yet to self-hit. For
                // models, ignore picks that land on the preview entity.
                // `raw_payload[0]` is the 32-bit pick channel — gpu_idx
                // on hit, 0xFFFFFFFF on sky.
                let hit_gpu_idx = if pr.raw_payload[0] != u32::MAX {
                    Some(pr.raw_payload[0] as usize)
                } else {
                    None
                };
                let hit_self = match &kind {
                    DragPreviewKind::Model { entity, .. } => {
                        hit_gpu_idx.is_some_and(|idx| {
                            self.entity_to_gpu.get(entity).copied() == Some(idx)
                        })
                    }
                    DragPreviewKind::Generator { .. } => false,
                };

                // Resolve target world position:
                //   1. Valid surface hit (not self) → use it, cache it.
                //   2. Self-hit or sky miss with a cached pos → keep that.
                //   3. No cache yet → ground-plane ray at the cursor.
                let new_pos = if let Some(hit) = pr.position.filter(|_| !hit_self) {
                    self.drag_preview.as_mut().unwrap().last_surface_pos = Some(hit);
                    Some(hit)
                } else if let Some(cached) = self.drag_preview.as_ref()
                    .and_then(|p| p.last_surface_pos)
                {
                    Some(cached)
                } else {
                    let (ro, rd) = self.screen_to_ray_for_viewport(vp, cx as f32, cy as f32);
                    if rd.y.abs() > 1e-6 {
                        let t = -ro.y / rd.y;
                        if t > 0.0 { Some(ro + rd * t) } else { None }
                    } else { None }
                };

                if let Some(p) = new_pos {
                    match kind {
                        DragPreviewKind::Model { entity, aabb_min_y } => {
                            // Bottom-snap the asset so its feet sit on
                            // the surface under the cursor.
                            if let Ok(mut t) = self.world
                                .get::<&mut crate::components::Transform>(entity)
                            {
                                t.position = glam::Vec3::new(p.x, p.y - aabb_min_y, p.z);
                            }
                            // PERF_DEBT B1+C2: only the dragged
                            // entity's transform changed — fast path
                            // in `update_scene_gpu` patches the
                            // matrix in place.
                            self.gpu_objects_dirty.mark_entity_transform(entity);
                        }
                        DragPreviewKind::Generator { .. } => {
                            // Gizmo-only — `last_surface_pos` is the
                            // whole state. Update and redraw at frame
                            // start via `build_gizmo_wireframe`.
                            self.drag_preview.as_mut().unwrap().last_surface_pos = Some(p);
                        }
                    }
                }
                self.in_flight_pick_ghost = None;
                return;
            }
        }

        match pr.kind {
            PickKind::ProceduralNode => {
                // Ghost priority: if the CPU raycast at click time
                // found a ghost primitive on the ray, that wins —
                // matches "translucent overlay on top owns the click."
                if let Some(ghost_id) = self.in_flight_pick_ghost.take() {
                    self.selected_procedural_node = Some(ghost_id);
                    return;
                }
                // Proc raymarch writes the primitive NodeId into the
                // low 16 bits of the 32-bit pick channel, with 0xFFFF
                // as the sky sentinel.
                let node_id_16 = pr.raw_payload[0] & 0xFFFFu32;
                if node_id_16 != 0xFFFFu32 {
                    self.selected_procedural_node = Some(node_id_16);
                } else {
                    self.selected_procedural_node = None;
                }
            }
            PickKind::Material => {
                // Defense in depth: paint mode strictly locks selection
                // to whatever is selected in the scene tree. Even if a
                // viewport-level paint guard is bypassed (or a stray
                // `Pick` command arrives mid-paint), don't let a
                // pixel pick mutate selection here.
                if self.paint_mode_active {
                    self.in_flight_pick_ghost = None;
                    return;
                }
                // Voxel march writes `gpu_idx` to the 32-bit pick
                // channel, 0xFFFFFFFF on sky. Direct lookup — no
                // bit-unpacking, no 255-object cap.
                let pick = pr.raw_payload[0];
                if pick != u32::MAX {
                    let gpu_idx = pick as usize;
                    if gpu_idx < self.gpu_to_entity.len() {
                        self.selected_entity = Some(self.gpu_to_entity[gpu_idx]);
                    }
                } else {
                    self.selected_entity = None;
                }
                // Discard the ghost hint either way — Material picks
                // never hit ghost-primitive priority.
                self.in_flight_pick_ghost = None;
                let _ = ViewportId::MAIN;
            }
        }
    }

    /// Apply a queued drop. `surface_pos` is `Some(hit)` when the pick
    /// readback sampled valid geometry (hit_distance < 1e9); otherwise
    /// we cast a ray through the drop pixel and intersect the Y=0
    /// ground plane as a fallback. If that ray also misses (looking
    /// up, no floor), log and skip — no silent spawn at the origin.
    pub(crate) fn handle_drop(&mut self, drop: PendingDrop, surface_pos: Option<glam::Vec3>) {
        let pos = if let Some(p) = surface_pos {
            p
        } else {
            let (ray_o, ray_d) = self.screen_to_ray_for_viewport(
                drop.viewport, drop.x as f32, drop.y as f32,
            );
            // Ground plane at y=0. `t > 0` means "plane is ahead of
            // the camera along the ray"; negative t (looking up) is a
            // miss.
            if ray_d.y.abs() > 1e-6 {
                let t = -ray_o.y / ray_d.y;
                if t > 0.0 {
                    ray_o + ray_d * t
                } else {
                    self.console.warn(format!(
                        "Drop missed geometry and the ground plane is behind the camera — skipping."
                    ));
                    return;
                }
            } else {
                self.console.warn(format!(
                    "Drop ray parallel to ground plane — skipping."
                ));
                return;
            }
        };
        match drop.action {
            PendingDropAction::Asset { path } => {
                self.spawn_asset(&path, pos);
            }
            PendingDropAction::Generator { name } => {
                self.spawn_generator(&name, Some(pos));
            }
            PendingDropAction::GeneratorPreset { path } => {
                self.spawn_generator_preset(&path, Some(pos));
            }
        }
    }

    /// CPU raycast against tree-wide ghost primitives at a BUILD-viewport
    /// click pixel. Returns the nearest ghost hit's NodeId (or `None`
    /// if no ghost is on the ray, the viewport isn't in raymarch
    /// mode, or the click isn't on BUILD). The ghost pass renders
    /// depth-free, so "nearest ghost along the ray" matches "ghost
    /// silhouette visible at this pixel."
    pub(crate) fn compute_ghost_pick(
        &self,
        viewport_id: crate::viewport::ViewportId,
        x: u32,
        y: u32,
    ) -> Option<u32> {
        if viewport_id != crate::viewport::ViewportId::BUILD {
            return None;
        }
        let build_vp = self.viewports.get(viewport_id)?;
        if !matches!(build_vp.preview_mode, rkp_render::BuildPreviewMode::Raymarch) {
            return None;
        }

        // Resolve the procedural entity: either the viewport's focus
        // target (the build viewport pins focus to whatever procedural
        // is under edit) or fall back to the editor's global selection.
        let entity = build_vp.filter.focus_entity.or(self.selected_entity)?;

        let proc_geo = self.world
            .get::<&crate::components::ProceduralGeometry>(entity).ok()?;
        let transform = self.world
            .get::<&crate::components::Transform>(entity).ok()?;

        let entity_world = glam::Affine3A::from_scale_rotation_translation(
            transform.scale,
            glam::Quat::from_euler(
                glam::EulerRot::XYZ,
                transform.rotation.x.to_radians(),
                transform.rotation.y.to_radians(),
                transform.rotation.z.to_radians(),
            ),
            transform.position,
        );

        let (ray_o, ray_d) = self
            .screen_to_ray_for_viewport(viewport_id, x as f32, y as f32);

        nearest_ghost_hit(
            &proc_geo.tree, entity_world, ray_o, ray_d, proc_geo.voxel_size,
        )
        .map(|(id, _t)| id)
    }
}

/// Compute a safe AABB and voxel size for procedural voxelization.
///
/// Adds margin around the tight bounds and ensures the octree depth won't
/// exceed MAX_DEPTH (11). If the object is too large for the requested voxel
/// size, the voxel size is increased to fit.
/// Walk the procedural tree to find the path from `start` to `target`.
///
/// Writes the sequence of node IDs (including both endpoints) into
/// `out_path` when a match is found and returns `true`. Returns `false`
/// (and leaves `out_path` empty) if `target` isn't reachable from
/// `start` — a possible state if the snapshot a caller is holding has
/// drifted from the current tree.
/// Every leaf NodeId in the tree that plays a "ghost" role — a primitive
/// whose surface can go invisible in the final CSG output, specifically:
///   - non-primary children of a `Subtract` (the cutters) and everything
///     beneath them,
///   - every child of an `Intersect` and everything beneath it,
///   - transitively: once a subtree is in a ghost role, all its leaves
///     inherit the role regardless of further combinators below.
/// Primary children of `Subtract` and all children of `Union` are fully
/// visible in the main raymarch and aren't ghosted.
///
/// Computed tree-wide (not per-selection) so ghosts act like a constant
/// editing aid — you can see every cutter in the scene at all times
/// and click on one to pick it even when it's fully carved away.
pub(crate) fn collect_ghost_primitives(tree: &rkp_procedural::ProceduralObject) -> Vec<u32> {
    let mut out = Vec::new();
    collect_ghosts_rec(tree, tree.root(), false, &mut out);
    out
}

/// CPU sphere-trace against a single procedural primitive's SDF.
///
/// Evaluates one leaf (analytical SDF) against a world-space ray by
/// transforming the ray into the primitive's local frame via the
/// composed ancestor transform, then sphere-tracing up to `MAX_STEPS`
/// iterations. Returns the nearest positive-t hit or `None`.
///
/// Used by the click-to-pick-ghost path. The GPU raymarch shader
/// handles the common case (pixel hits a visible primitive) via
/// `gbuf_pick`; this CPU walk is only for fully-carved cutters whose
/// silhouette has no surface in the G-buffer.
pub(crate) fn raycast_leaf_primitive(
    tree: &rkp_procedural::ProceduralObject,
    leaf_id: rkp_procedural::NodeId,
    ancestor_world: glam::Affine3A,
    ray_origin: glam::Vec3,
    ray_dir: glam::Vec3,
) -> Option<f32> {
    const MAX_STEPS: u32 = 64;
    const MAX_DIST: f32 = 500.0;
    const SURFACE_EPS: f32 = 0.001;

    let node = tree.get(leaf_id)?;
    if !node.kind.is_leaf() { return None; }

    // Local-frame ray. Non-uniform scale on the transform chain would
    // make `ray_d_local` non-unit; for the current editor workflow
    // transforms are uniform-scale-only from the gizmo, so this is
    // fine. If that ever changes, normalize and scale `t` back out.
    let world = ancestor_world * node.transform;
    let inv = world.inverse();
    let ro = inv.transform_point3(ray_origin);
    let rd = inv.transform_vector3(ray_dir);

    let mut t: f32 = 0.0;
    for _ in 0..MAX_STEPS {
        let p = ro + rd * t;
        let d = rkp_procedural::eval_leaf_distance(p, &node.kind);
        if d < SURFACE_EPS { return Some(t); }
        t += d.max(SURFACE_EPS);
        if t > MAX_DIST { return None; }
    }
    None
}

/// Find the closest ghost-role primitive along a world-space ray.
/// Returns `Some((node_id, t))` for the nearest hit, or `None` if no
/// ghost is on the ray. Composed-transform walk mirrors
/// `collect_ghost_primitives` so the same inheritance rules apply.
pub(crate) fn nearest_ghost_hit(
    tree: &rkp_procedural::ProceduralObject,
    entity_world: glam::Affine3A,
    ray_origin: glam::Vec3,
    ray_dir: glam::Vec3,
    voxel_size: f32,
) -> Option<(u32, f32)> {
    let mut best: Option<(u32, f32)> = None;
    nearest_ghost_hit_rec(
        tree, tree.root(), false, entity_world,
        ray_origin, ray_dir, voxel_size, &mut best,
    );
    best
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn nearest_ghost_hit_rec(
    tree: &rkp_procedural::ProceduralObject,
    id: rkp_procedural::NodeId,
    is_ghost: bool,
    ancestor_world: glam::Affine3A,
    ray_origin: glam::Vec3,
    ray_dir: glam::Vec3,
    voxel_size: f32,
    best: &mut Option<(u32, f32)>,
) {
    use rkp_procedural::NodeKind;
    let Some(node) = tree.get(id) else { return };
    let this_world = ancestor_world * node.transform;

    if node.kind.is_leaf() {
        if is_ghost {
            // Raycast in the LEAF's frame — its own transform is
            // already composed into `this_world`, so the caller of
            // raycast_leaf_primitive passes `ancestor_world` = the
            // parent's world (i.e. without leaf.transform) and the
            // function re-composes. To avoid double-applying, use
            // the ancestor_world we were called with, not this_world.
            if let Some(t) = raycast_leaf_primitive(
                tree, id, ancestor_world, ray_origin, ray_dir,
            ) {
                match *best {
                    None => *best = Some((id.0, t)),
                    Some((_, bt)) if t < bt => *best = Some((id.0, t)),
                    _ => {}
                }
            }
        }
        return;
    }
    match &node.kind {
        NodeKind::Union { .. } => {
            for &c in &node.children {
                nearest_ghost_hit_rec(
                    tree, c, is_ghost, this_world,
                    ray_origin, ray_dir, voxel_size, best,
                );
            }
        }
        NodeKind::Intersect { .. } => {
            for &c in &node.children {
                nearest_ghost_hit_rec(
                    tree, c, true, this_world,
                    ray_origin, ray_dir, voxel_size, best,
                );
            }
        }
        NodeKind::Subtract => {
            for (i, &c) in node.children.iter().enumerate() {
                let child_ghost = is_ghost || i > 0;
                nearest_ghost_hit_rec(
                    tree, c, child_ghost, this_world,
                    ray_origin, ray_dir, voxel_size, best,
                );
            }
        }
        _ => {}
    }
}

pub(crate) fn collect_ghosts_rec(
    tree: &rkp_procedural::ProceduralObject,
    id: rkp_procedural::NodeId,
    is_ghost: bool,
    out: &mut Vec<u32>,
) {
    use rkp_procedural::NodeKind;
    let Some(node) = tree.get(id) else { return };
    if node.kind.is_leaf() {
        if is_ghost { out.push(id.0); }
        return;
    }
    match &node.kind {
        NodeKind::Union { .. } => {
            for &c in &node.children {
                collect_ghosts_rec(tree, c, is_ghost, out);
            }
        }
        NodeKind::Intersect { .. } => {
            // All children of an Intersect are "operands that can go
            // invisible where the others aren't." Flip on the ghost
            // flag for every descendant branch.
            for &c in &node.children {
                collect_ghosts_rec(tree, c, true, out);
            }
        }
        NodeKind::Subtract => {
            // First child (minuend) stays whatever its ancestor context
            // made it. Later children (cutters) become ghosts.
            for (i, &c) in node.children.iter().enumerate() {
                let child_ghost = is_ghost || i > 0;
                collect_ghosts_rec(tree, c, child_ghost, out);
            }
        }
        _ => {}
    }
}

pub(crate) fn find_path(
    tree: &rkp_procedural::ProceduralObject,
    start: rkp_procedural::NodeId,
    target: rkp_procedural::NodeId,
    out_path: &mut Vec<rkp_procedural::NodeId>,
) -> bool {
    out_path.push(start);
    if start == target {
        return true;
    }
    if let Some(node) = tree.get(start) {
        for &child in &node.children {
            if find_path(tree, child, target, out_path) {
                return true;
            }
        }
    }
    out_path.pop();
    false
}
