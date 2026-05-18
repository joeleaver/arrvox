//! BUILD-viewport procedural-preview gizmo.
//!
//! Updates the selected procedural node's per-frame gizmo drag state
//! and emits the wireframe vertices that draw it in the BUILD viewport.
//! Separate from the scene-tree entity gizmo (see `gizmo_ops`) because
//! the BUILD gizmo targets a `NodeId` inside a `ProceduralObject`
//! rather than a `hecs::Entity` transform.

use super::picking_ops::find_path;
use super::procedural_ops::decompose_affine_rotation;
use super::state::EngineState;

impl EngineState {
    /// BUILD-viewport gizmo: hover + drag for the selected procedural
    /// node's transform. Mirrors `update_gizmo` but reads BUILD mouse
    /// state, casts rays through BUILD's camera, and writes to the
    /// node's Affine3A instead of an entity Transform.
    pub(crate) fn update_procedural_gizmo(&mut self) {
        // Voxel preview mode: the gizmo would edit the tree without
        // any live visual update in the build viewport, so disable
        // it entirely. Clear any in-flight drag and reset hover so a
        // mode flip mid-interaction doesn't leave stale state.
        let raymarch = self.viewports
            .get(crate::viewport::ViewportId::BUILD)
            .map(|v| matches!(v.preview_mode, arvx_render::BuildPreviewMode::Raymarch))
            .unwrap_or(false);
        if !raymarch {
            self.proc_gizmo.hovered_axis = crate::gizmo::GizmoAxis::None;
            if self.proc_gizmo.dragging {
                self.proc_gizmo.end_drag();
            }
            return;
        }

        let (node_id, entity) = match (self.selected_procedural_node, self.selected_entity) {
            (Some(n), Some(e)) => (n, e),
            _ => {
                self.proc_gizmo.hovered_axis = crate::gizmo::GizmoAxis::None;
                if self.proc_gizmo.dragging {
                    self.proc_gizmo.end_drag();
                }
                return;
            }
        };

        // Resolve parent-world and current local transform from the tree.
        let (parent_world, current_local) = {
            let Ok(proc_geo) =
                self.world.get::<&crate::components::ProceduralGeometry>(entity)
            else {
                return;
            };
            let Ok(entity_xform) =
                self.world.get::<&crate::components::Transform>(entity)
            else {
                return;
            };
            let target = arvx_procedural::NodeId(node_id);
            let mut path = Vec::new();
            if !find_path(&proc_geo.tree, proc_geo.tree.root(), target, &mut path) {
                return;
            }
            let entity_world = glam::Affine3A::from_scale_rotation_translation(
                entity_xform.scale,
                glam::Quat::from_euler(
                    glam::EulerRot::XYZ,
                    entity_xform.rotation.x.to_radians(),
                    entity_xform.rotation.y.to_radians(),
                    entity_xform.rotation.z.to_radians(),
                ),
                entity_xform.position,
            );
            let mut parent_world = entity_world;
            for id in &path[..path.len() - 1] {
                if let Some(n) = proc_geo.tree.get(*id) {
                    parent_world = parent_world * n.transform;
                }
            }
            let current_local = proc_geo
                .tree
                .get(target)
                .map(|n| n.transform)
                .unwrap_or(glam::Affine3A::IDENTITY);
            (parent_world, current_local)
        };

        let world_transform = parent_world * current_local;
        let center = world_transform.transform_point3(glam::Vec3::ZERO);

        let cam_uniforms =
            self.build_camera_uniforms(crate::viewport::ViewportId::BUILD);
        let cam_pos = glam::Vec3::new(
            cam_uniforms.position[0],
            cam_uniforms.position[1],
            cam_uniforms.position[2],
        );
        let cam_dist = (center - cam_pos).length().max(0.1);
        let gizmo_size = cam_dist * 0.15;

        let (ray_o, ray_d) = self.screen_to_ray_for_viewport(
            crate::viewport::ViewportId::BUILD,
            self.build_mouse_pos.x,
            self.build_mouse_pos.y,
        );

        if self.proc_gizmo.dragging {
            // Apply deltas relative to the drag-start SRT, then write
            // back to the node's local transform in parent-relative
            // space. `parent_world.inverse()` handles the conversion
            // back from world deltas.
            let (init_local_t, init_local_r, init_local_s) = self.proc_gizmo_initial_local;
            let parent_inv = self.proc_gizmo_parent_world.inverse();
            let parent_rot = decompose_affine_rotation(&self.proc_gizmo_parent_world);

            let new_local = match self.gizmo.mode {
                crate::gizmo::GizmoMode::Translate => {
                    let world_delta = crate::gizmo::compute_translate_delta(
                        &self.proc_gizmo, ray_o, ray_d,
                    );
                    let new_world_pos = self.proc_gizmo.initial_position + world_delta;
                    let new_local_t = parent_inv.transform_point3(new_world_pos);
                    glam::Affine3A::from_scale_rotation_translation(
                        init_local_s, init_local_r, new_local_t,
                    )
                }
                crate::gizmo::GizmoMode::Rotate => {
                    let world_delta = crate::gizmo::compute_rotate_delta(
                        &self.proc_gizmo, ray_o, ray_d, center,
                    );
                    let new_world_rot = world_delta * self.proc_gizmo.initial_rotation;
                    let new_local_r = parent_rot.inverse() * new_world_rot;
                    glam::Affine3A::from_scale_rotation_translation(
                        init_local_s, new_local_r, init_local_t,
                    )
                }
                crate::gizmo::GizmoMode::Scale => {
                    let delta = crate::gizmo::compute_scale_delta(
                        &self.proc_gizmo, ray_o, ray_d,
                    );
                    let new_local_s = init_local_s * delta;
                    glam::Affine3A::from_scale_rotation_translation(
                        new_local_s, init_local_r, init_local_t,
                    )
                }
            };

            if let Ok(mut proc_geo) =
                self.world.get::<&mut crate::components::ProceduralGeometry>(entity)
            {
                proc_geo.tree.set_transform(
                    arvx_procedural::NodeId(node_id),
                    new_local,
                );
                proc_geo.dirty = true;
            }

            if !self.build_mouse_left {
                self.proc_gizmo.end_drag();
            }
        } else {
            self.proc_gizmo.hovered_axis = crate::gizmo::pick_gizmo_axis_for_mode(
                ray_o, ray_d, center, gizmo_size, self.gizmo.mode,
            );

            if self.build_mouse_left
                && self.proc_gizmo.hovered_axis != crate::gizmo::GizmoAxis::None
            {
                // Capture starting state. Same branching as the entity
                // gizmo — the start point depends on which handle was
                // grabbed so drag math projects from the right origin.
                let start_point = match (self.gizmo.mode, self.proc_gizmo.hovered_axis) {
                    (crate::gizmo::GizmoMode::Rotate,
                     crate::gizmo::GizmoAxis::X
                     | crate::gizmo::GizmoAxis::Y
                     | crate::gizmo::GizmoAxis::Z) => {
                        let axis_dir = self.proc_gizmo.hovered_axis.direction();
                        crate::gizmo::project_to_plane(ray_o, ray_d, center, axis_dir)
                            .unwrap_or(center)
                    }
                    (_,
                     crate::gizmo::GizmoAxis::XY
                     | crate::gizmo::GizmoAxis::XZ
                     | crate::gizmo::GizmoAxis::YZ) => {
                        let normal = self.proc_gizmo.hovered_axis.plane_normal();
                        crate::gizmo::project_to_plane(ray_o, ray_d, center, normal)
                            .unwrap_or(center)
                    }
                    (_,
                     crate::gizmo::GizmoAxis::X
                     | crate::gizmo::GizmoAxis::Y
                     | crate::gizmo::GizmoAxis::Z) => {
                        let axis_dir = self.proc_gizmo.hovered_axis.direction();
                        let t = crate::gizmo::ray_axis_closest_point(
                            ray_o, ray_d, center, axis_dir,
                        );
                        center + axis_dir * t
                    }
                    _ => crate::gizmo::project_to_plane(ray_o, ray_d, center, -ray_d)
                        .unwrap_or(center),
                };

                // Decompose current LOCAL transform once for later
                // reconstruction during drag.
                let local_t = current_local.translation.into();
                let m = current_local.matrix3;
                let sx = glam::Vec3::from(m.x_axis).length();
                let sy = glam::Vec3::from(m.y_axis).length();
                let sz = glam::Vec3::from(m.z_axis).length();
                let local_s = glam::Vec3::new(sx.max(1e-8), sy.max(1e-8), sz.max(1e-8));
                let rot_mat = glam::Mat3::from_cols(
                    (glam::Vec3::from(m.x_axis) / local_s.x).into(),
                    (glam::Vec3::from(m.y_axis) / local_s.y).into(),
                    (glam::Vec3::from(m.z_axis) / local_s.z).into(),
                );
                let local_r = glam::Quat::from_mat3(&rot_mat);

                let parent_rot = decompose_affine_rotation(&parent_world);
                let world_rot = parent_rot * local_r;
                let forward = (center - cam_pos).normalize();

                self.proc_gizmo_parent_world = parent_world;
                self.proc_gizmo_initial_local = (local_t, local_r, local_s);
                self.proc_gizmo.pivot = center;
                self.proc_gizmo.begin_drag(
                    self.proc_gizmo.hovered_axis,
                    start_point,
                    center,
                    world_rot,
                    local_s,
                    forward,
                );
            }
        }
    }

    /// Wireframe for the procedural-node gizmo drawn on the BUILD viewport.
    ///
    /// Returns an empty vec when:
    /// - no entity selected,
    /// - the selected entity has no `ProceduralGeometry`,
    /// - no procedural node is selected,
    /// - the selected node can't be reached from the root (stale snapshot).
    ///
    /// The gizmo sits at the node's origin in world space — entity world
    /// transform × accumulated parent transforms × the node's own
    /// transform, all applied to (0,0,0). Axes stay world-aligned
    /// (matches the entity gizmo's convention).
    pub(crate) fn build_procedural_gizmo_wireframe(
        &self,
        cam_pos: glam::Vec3,
    ) -> Vec<arvx_render::LineVertex> {
        let node_id = match self.selected_procedural_node {
            Some(id) => id,
            None => return Vec::new(),
        };
        let entity = match self.selected_entity {
            Some(e) => e,
            None => return Vec::new(),
        };

        let Ok(proc_geo) = self
            .world
            .get::<&crate::components::ProceduralGeometry>(entity)
        else {
            return Vec::new();
        };
        let Ok(entity_xform) = self
            .world
            .get::<&crate::components::Transform>(entity)
        else {
            return Vec::new();
        };

        // Walk root → selected node, accumulating parent transforms.
        let tree = &proc_geo.tree;
        let target = arvx_procedural::NodeId(node_id);
        let mut path: Vec<arvx_procedural::NodeId> = Vec::new();
        if !find_path(tree, tree.root(), target, &mut path) {
            return Vec::new();
        }

        // Compose entity world × each transform on the path. Path is
        // root-first and includes the target node, so the last multiply
        // pulls in the target's own local transform — which is what we
        // want: gizmo sits at the node's rotated/scaled/translated origin.
        let entity_world = glam::Affine3A::from_scale_rotation_translation(
            entity_xform.scale,
            glam::Quat::from_euler(
                glam::EulerRot::XYZ,
                entity_xform.rotation.x.to_radians(),
                entity_xform.rotation.y.to_radians(),
                entity_xform.rotation.z.to_radians(),
            ),
            entity_xform.position,
        );
        let mut accum = entity_world;
        for id in &path {
            if let Some(n) = tree.get(*id) {
                accum = accum * n.transform;
            }
        }
        let center = accum.transform_point3(glam::Vec3::ZERO);

        let cam_dist = (center - cam_pos).length().max(0.1);
        let gizmo_size = cam_dist * 0.15;
        // Use proc_gizmo's hover/drag axis so the handle highlights
        // correctly while the user is interacting with BUILD — gizmo
        // mode itself is shared with MAIN's toolbar.
        let hovered = if self.proc_gizmo.dragging {
            self.proc_gizmo.active_axis
        } else {
            self.proc_gizmo.hovered_axis
        };
        match self.gizmo.mode {
            crate::gizmo::GizmoMode::Translate => {
                crate::wireframe_builders::translate_gizmo_wireframe(
                    center, gizmo_size, hovered, cam_pos,
                )
            }
            crate::gizmo::GizmoMode::Rotate => {
                crate::wireframe_builders::rotate_gizmo_wireframe(
                    center, gizmo_size, hovered, cam_pos,
                )
            }
            crate::gizmo::GizmoMode::Scale => {
                crate::wireframe_builders::scale_gizmo_wireframe(
                    center, gizmo_size, hovered, cam_pos,
                )
            }
        }
    }
}
