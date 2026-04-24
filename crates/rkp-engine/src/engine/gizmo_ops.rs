//! Gizmo state updates and wireframe generation.
//!
//! Per-frame gizmo hit/drag state (entity-transform gizmos only; the
//! BUILD-viewport procedural gizmo lives in `procedural_ops`) and
//! wireframe vertex builders for the gizmo itself, skeleton bones, and
//! physics colliders.

use super::state::{DragPreviewKind, EngineState};

impl EngineState {
    pub(crate) fn update_gizmo(&mut self) {
        // Paint mode hides the transform gizmo entirely (see
        // `build_gizmo_wireframe`). Suppress hover/drag detection too
        // so a paint click on the selected object's centroid doesn't
        // get captured by an invisible axis handle.
        if self.paint_mode_active {
            self.gizmo.hovered_axis = crate::gizmo::GizmoAxis::None;
            if self.gizmo.dragging {
                self.gizmo.end_drag();
            }
            return;
        }
        let Some(selected) = self.selected_entity else {
            self.gizmo.hovered_axis = crate::gizmo::GizmoAxis::None;
            if self.gizmo.dragging {
                self.gizmo.end_drag();
            }
            return;
        };

        let center = match self.world.get::<&crate::components::Transform>(selected) {
            Ok(t) => t.position,
            Err(_) => return,
        };
        let cam_dist = (center - self.camera.position).length().max(0.1);
        let gizmo_size = cam_dist * 0.15;

        let (ray_o, ray_d) = self.screen_to_ray(self.mouse_pos.x, self.mouse_pos.y);

        let left_pressed = self.input_system.raw_state().is_mouse_button_pressed(rkp_runtime::input::InputMouseButton::Left);

        if self.gizmo.dragging {
            // Update drag.
            match self.gizmo.mode {
                crate::gizmo::GizmoMode::Translate => {
                    let delta = crate::gizmo::compute_translate_delta(&self.gizmo, ray_o, ray_d);
                    let new_pos = self.gizmo.initial_position + delta;
                    if let Ok(mut t) = self.world.get::<&mut crate::components::Transform>(selected) {
                        t.position = new_pos;
                        self.gpu_objects_dirty = true;
                    }
                }
                crate::gizmo::GizmoMode::Rotate => {
                    let delta = crate::gizmo::compute_rotate_delta(&self.gizmo, ray_o, ray_d, center);
                    let new_rot = delta * self.gizmo.initial_rotation;
                    // Convert quaternion back to Euler degrees for storage.
                    let (y, x, z) = new_rot.to_euler(glam::EulerRot::YXZ);
                    let euler_deg = glam::Vec3::new(x.to_degrees(), y.to_degrees(), z.to_degrees());
                    if let Ok(mut t) = self.world.get::<&mut crate::components::Transform>(selected) {
                        t.rotation = euler_deg;
                        self.gpu_objects_dirty = true;
                    }
                }
                crate::gizmo::GizmoMode::Scale => {
                    let delta = crate::gizmo::compute_scale_delta(&self.gizmo, ray_o, ray_d);
                    let new_scale = self.gizmo.initial_scale * delta;
                    if let Ok(mut t) = self.world.get::<&mut crate::components::Transform>(selected) {
                        t.scale = new_scale;
                        self.gpu_objects_dirty = true;
                    }
                    // Same path as the properties-panel scale slider:
                    // route procedural entities' scale onto Root,
                    // queue a debounced bake, and convert what we just
                    // wrote into a render-time preview multiplier.
                    self.redirect_transform_scale_to_root(selected);
                }
            }

            if !left_pressed {
                // Drag ended this tick. For Scale mode on procedural
                // entities, clear `bake_dirty_at` so the next tick's
                // `pending_settled` check fires immediately instead of
                // waiting out the 150 ms slider debounce — mouse-up is
                // an unambiguous "done" signal that a slider doesn't
                // have, no reason to sit on it.
                if matches!(self.gizmo.mode, crate::gizmo::GizmoMode::Scale) {
                    if let Ok(mut pg) = self
                        .world
                        .get::<&mut crate::components::ProceduralGeometry>(selected)
                    {
                        if pg.pending_bake {
                            pg.bake_dirty_at = None;
                        }
                    }
                }
                self.gizmo.end_drag();
            }
        } else {
            // Update hover.
            self.gizmo.hovered_axis = crate::gizmo::pick_gizmo_axis_for_mode(
                ray_o, ray_d, center, gizmo_size, self.gizmo.mode,
            );

            // Start drag if left mouse is pressed on a gizmo handle.
            if left_pressed && self.gizmo.hovered_axis != crate::gizmo::GizmoAxis::None {
                let start_point = match (self.gizmo.mode, self.gizmo.hovered_axis) {
                    // Rotation: project onto the plane perpendicular to the rotation axis.
                    (crate::gizmo::GizmoMode::Rotate, crate::gizmo::GizmoAxis::X | crate::gizmo::GizmoAxis::Y | crate::gizmo::GizmoAxis::Z) => {
                        let axis_dir = self.gizmo.hovered_axis.direction();
                        crate::gizmo::project_to_plane(ray_o, ray_d, center, axis_dir).unwrap_or(center)
                    }
                    // Plane handles (XY/XZ/YZ): project onto the constraint plane.
                    (_, crate::gizmo::GizmoAxis::XY | crate::gizmo::GizmoAxis::XZ | crate::gizmo::GizmoAxis::YZ) => {
                        let normal = self.gizmo.hovered_axis.plane_normal();
                        crate::gizmo::project_to_plane(ray_o, ray_d, center, normal).unwrap_or(center)
                    }
                    // Single-axis translate / scale: closest point on the axis line.
                    (_, crate::gizmo::GizmoAxis::X | crate::gizmo::GizmoAxis::Y | crate::gizmo::GizmoAxis::Z) => {
                        let axis_dir = self.gizmo.hovered_axis.direction();
                        let t = crate::gizmo::ray_axis_closest_point(ray_o, ray_d, center, axis_dir);
                        center + axis_dir * t
                    }
                    _ => {
                        crate::gizmo::project_to_plane(ray_o, ray_d, center, -ray_d).unwrap_or(center)
                    }
                };
                let forward = (center - self.camera.position).normalize();
                let rotation = self.world.get::<&crate::components::Transform>(selected)
                    .map(|t| {
                        let r = t.rotation;
                        glam::Quat::from_euler(
                            glam::EulerRot::YXZ,
                            r.y.to_radians(), r.x.to_radians(), r.z.to_radians(),
                        )
                    })
                    .unwrap_or(glam::Quat::IDENTITY);
                // For procedural entities the user-visible scale lives
                // on Root.transform (Transform.scale stays ~1 between
                // bakes / momentarily holds the preview multiplier
                // mid-debounce). Drag math is multiplicative against
                // `initial_scale`, so capturing Transform.scale would
                // make the first frame of a drag interpret the object
                // as scale 1 and snap it back to its baseline size.
                let scale = self.world.get::<&crate::components::ProceduralGeometry>(selected)
                    .ok()
                    .and_then(|pg| {
                        let root = pg.tree.root();
                        pg.tree
                            .get(root)
                            .map(|n| n.transform.to_scale_rotation_translation().0)
                    })
                    .or_else(|| {
                        self.world
                            .get::<&crate::components::Transform>(selected)
                            .ok()
                            .map(|t| t.scale)
                    })
                    .unwrap_or(glam::Vec3::ONE);
                self.gizmo.pivot = center;
                self.gizmo.begin_drag(
                    self.gizmo.hovered_axis,
                    start_point,
                    center,
                    rotation,
                    scale,
                    forward,
                );
            }
        }
    }

    /// Emit one line segment per parent→child bone pair for each
    /// animated entity. The skinning palette already encodes
    /// `current_world * inverse_bind`, so premultiplying by the
    /// bone's bind-pose local origin cancels the inverse-bind and
    /// gives us the animated world origin directly. Cheap (one mat4
    /// × vec3 per bone) and stateless — runs in the same pass as
    /// selection/light gizmos.
    pub(crate) fn build_bone_wireframes(&self) -> Vec<rkp_render::LineVertex> {
        use glam::{Mat4, Vec3, Vec4};
        let mut verts = Vec::new();
        let bright = [0.5, 0.9, 1.0, 1.0];
        // Bones are editor chrome for the currently-selected rig.
        // Play mode has no selection (selection is an edit-mode
        // concept), and non-selected entities clutter the viewport
        // when multiple animated characters are on screen — so we
        // draw bones for the selected entity only.
        let Some(selected) = self.selected_entity else { return verts };
        let query_result = self.world.query_one::<(&crate::components::Transform, &crate::components::Skeleton)>(selected);
        let Ok(mut query) = query_result else { return verts };
        let Some((transform, skeleton)) = query.get() else { return verts };
        {
            let color = bright;

            // Entity's root world transform (same one the renderer uses
            // for this entity). Bone origins are in object-local space;
            // multiply by this to lift them into world space.
            let root_world = Mat4::from_scale_rotation_translation(
                transform.scale,
                glam::Quat::from_euler(
                    glam::EulerRot::XYZ,
                    transform.rotation.x.to_radians(),
                    transform.rotation.y.to_radians(),
                    transform.rotation.z.to_radians(),
                ),
                transform.position,
            );

            let bones = &skeleton.asset.skeleton;
            let pose = &skeleton.current_pose;
            let bind_origins = &skeleton.bind_world_origins;
            // Defensive: if evaluate() hasn't run yet (pose is all
            // identity) or the pose is the wrong size, fall back to
            // bind-pose origins so the bones still render at rest.
            let use_pose = pose.len() == bones.bones.len();
            // `current_pose` + `bind_world_origins` are in grid frame
            // (origin at octree corner). Undo the grid offset before
            // handing the position to `root_world`, which expects
            // mesh-frame (origin at object centre).
            let grid_offset = skeleton.grid_offset;

            let animated_origin = |i: usize| -> Vec3 {
                let bind = bind_origins.get(i).copied().unwrap_or(Vec3::ZERO);
                let animated_grid = if use_pose {
                    let p = Vec4::new(bind.x, bind.y, bind.z, 1.0);
                    let v = pose[i] * p;
                    Vec3::new(v.x, v.y, v.z)
                } else {
                    bind
                };
                let animated_mesh = animated_grid - grid_offset;
                root_world.transform_point3(animated_mesh)
            };

            // Parent→child line per bone. Root bones get a tiny
            // crosshair so isolated roots (skeleton with 1 bone, or
            // detached rigs) still render something.
            for (i, &parent) in bones.hierarchy.iter().enumerate() {
                let child = animated_origin(i);
                if parent >= 0 && (parent as usize) < bones.bones.len() {
                    let parent_pos = animated_origin(parent as usize);
                    verts.push(rkp_render::LineVertex {
                        position: parent_pos.to_array(),
                        color,
                    });
                    verts.push(rkp_render::LineVertex {
                        position: child.to_array(),
                        color,
                    });
                } else {
                    // Root bone — little crosshair so it's visible even
                    // with no child.
                    verts.extend(rkp_render::wireframe::crosshair(child, 0.05, color));
                }
            }
        }
        verts
    }

    pub(crate) fn build_gizmo_wireframe(&self) -> Vec<rkp_render::LineVertex> {
        let mut verts = Vec::new();

        // Light gizmos — always visible for all light entities.
        let light_color = [1.0, 0.9, 0.5, 0.5]; // warm yellow, semi-transparent
        let selected_light_color = [1.0, 0.9, 0.5, 1.0]; // bright when selected

        for (entity, (transform, pl)) in self.world.query::<(&crate::components::Transform, &crate::components::PointLight)>().iter() {
            let selected = self.selected_entity == Some(entity);
            // Always show crosshair icon.
            let icon_color = if selected { selected_light_color } else { light_color };
            verts.extend(rkp_render::wireframe::crosshair(transform.position, 0.2, icon_color));
            // Range sphere only when selected.
            if selected {
                verts.extend(rkp_render::wireframe::point_light_wireframe(
                    transform.position, pl.range, selected_light_color,
                ));
            }
        }

        for (entity, (transform, sl)) in self.world.query::<(&crate::components::Transform, &crate::components::SpotLight)>().iter() {
            let selected = self.selected_entity == Some(entity);
            let icon_color = if selected { selected_light_color } else { light_color };
            verts.extend(rkp_render::wireframe::crosshair(transform.position, 0.2, icon_color));
            // Cone only when selected.
            if selected {
                verts.extend(rkp_render::wireframe::spot_light_wireframe(
                    transform.position, sl.direction, sl.range, sl.outer_angle.to_radians(), selected_light_color,
                ));
            }
        }

        // Physics collider wireframes.
        if self.show_colliders {
            verts.extend(self.build_collider_wireframes());
        }

        // Bone gizmo — one set of line segments per skinned entity,
        // drawn from animated bone origins. Selected entity gets a
        // brighter palette so it pops against a scene with multiple
        // animated characters.
        verts.extend(self.build_bone_wireframes());

        // Drag-preview gizmo for generators — a wireframe AABB sized
        // by the preview's `gizmo_half`, centered on the cached surface
        // hit. The generator itself only spawns on commit; this box is
        // the user's visual anchor while dragging.
        if let Some(preview) = self.drag_preview.as_ref() {
            if let DragPreviewKind::Generator { gizmo_half, .. } = &preview.kind {
                if let Some(center) = preview.last_surface_pos {
                    // Sit the box on the surface so the bottom face is
                    // flush with the drop point — matches how model
                    // previews bottom-snap.
                    let min = glam::Vec3::new(
                        center.x - gizmo_half.x,
                        center.y,
                        center.z - gizmo_half.z,
                    );
                    let max = glam::Vec3::new(
                        center.x + gizmo_half.x,
                        center.y + 2.0 * gizmo_half.y,
                        center.z + gizmo_half.z,
                    );
                    // Soft cyan, semi-transparent — the same palette
                    // the editor uses for "pending" overlays.
                    let color = [0.4, 0.9, 1.0, 0.7];
                    verts.extend(rkp_render::wireframe::aabb_wireframe(min, max, color));
                }
            }
        }

        // Paint cursor rendering lives in the shade pass (see
        // `rkp_shade.wgsl`'s brush overlay block) — the ring is
        // projected onto the shaded surface using the brush_overlay
        // storage buffer (geodesic) + the ShadeParams brush_* uniform.
        // No wireframe lines here anymore.

        // Transform gizmo — only for the selected entity, and only
        // when paint mode is off. Showing it in paint mode blocks
        // clicks on the selected object's center (gizmo axes sit
        // right on top of the surface) and crowds the cursor ring.
        if self.paint_mode_active {
            return verts;
        }
        let Some(selected) = self.selected_entity else {
            return verts;
        };

        let center = match self.world.get::<&crate::components::Transform>(selected) {
            Ok(t) => t.position,
            Err(_) => return verts,
        };

        let cam_dist = (center - self.camera.position).length().max(0.1);
        let gizmo_size = cam_dist * 0.15;

        let gizmo_verts = match self.gizmo.mode {
            crate::gizmo::GizmoMode::Translate => {
                crate::wireframe_builders::translate_gizmo_wireframe(
                    center, gizmo_size, self.gizmo.hovered_axis, self.camera.position,
                )
            }
            crate::gizmo::GizmoMode::Rotate => {
                crate::wireframe_builders::rotate_gizmo_wireframe(
                    center, gizmo_size, self.gizmo.hovered_axis, self.camera.position,
                )
            }
            crate::gizmo::GizmoMode::Scale => {
                crate::wireframe_builders::scale_gizmo_wireframe(
                    center, gizmo_size, self.gizmo.hovered_axis, self.camera.position,
                )
            }
        };
        verts.extend(gizmo_verts);
        verts
    }

    /// Rebuild collider caches for all entities with RigidBody.
    /// Called when geometry changes, RigidBody is added/modified, etc.
    pub(crate) fn rebuild_collider_caches(&mut self) {
        use crate::components::*;

        // Collect entities that need cache rebuild.
        let entities: Vec<(hecs::Entity, RigidBody, Option<SpatialData>, glam::Vec3)> = self.world
            .query::<(&RigidBody, Option<&Renderable>, &Transform)>()
            .iter()
            .map(|(e, (rb, r, t))| {
                (e, rb.clone(), r.and_then(|r| r.spatial.clone()), t.scale)
            })
            .collect();

        let sm_guard = self.scene_mgr.lock().unwrap();
        let all_nodes = sm_guard.octree.data();

        for (entity, rb, spatial, scale) in entities {
            let name = self.world.get::<&EditorMetadata>(entity)
                .map(|m| m.name.clone()).unwrap_or_default();
            let pos = self.world.get::<&Transform>(entity)
                .map(|t| t.position).unwrap_or_default();

            // Derive the fitted-shape bounds from the actually occupied
            // voxels. The padded `SpatialData.aabb` overshoots by ~14 voxels
            // per side (boundary-sampling margin) — fine for the renderer,
            // wrong for Box/Sphere/Capsule sizing.
            let tight_local = spatial.as_ref().and_then(|sp| {
                crate::play_mode::compute_tight_local_aabb(
                    all_nodes,
                    &sm_guard.brick_pool,
                    sp.root_offset as usize,
                    sp.depth,
                    sp.len,
                    sp.base_voxel_size,
                    sp.grid_origin,
                )
            });

            let (aabb_half, local_center) = match tight_local {
                Some(t) => (t.half_extents() * scale, (t.min + t.max) * 0.5 * scale),
                None => (glam::Vec3::splat(0.5), glam::Vec3::ZERO),
            };

            if let Some(ref sp) = spatial {
                eprintln!(
                    "[ColliderCache] '{name}' pos={pos:?} scale={scale:?} \
                     padded_aabb={:?}..{:?} tight_local={tight_local:?} \
                     aabb_half={aabb_half:?} local_center={local_center:?}",
                    sp.aabb.min, sp.aabb.max,
                );
            }

            let (resolved_shape, voxel_coords, voxel_size) = match rb.collider_shape {
                rkp_physics::rigid_body::ColliderShape::Auto => {
                    if let Some(ref sp) = spatial {
                        let (coords, cell_size) = crate::play_mode::build_coarse_collider(
                            all_nodes,
                            &sm_guard.brick_pool,
                            sp.root_offset as usize,
                            sp.depth,
                            sp.len,
                            sp.base_voxel_size,
                            rb.collider_cell_size,
                        );
                        if coords.is_empty() {
                            (rkp_physics::rigid_body::ColliderShape::Box, Vec::new(), 0.0)
                        } else {
                            (rkp_physics::rigid_body::ColliderShape::Auto, coords, cell_size)
                        }
                    } else {
                        (rkp_physics::rigid_body::ColliderShape::Box, Vec::new(), 0.0)
                    }
                }
                other => (other.clone(), Vec::new(), 0.0),
            };

            let (grid_origin, tree_depth) = match spatial.as_ref() {
                Some(sp) => (sp.grid_origin, sp.depth),
                None => (glam::Vec3::ZERO, 0),
            };

            let cache = ColliderCache {
                shape: resolved_shape,
                voxel_coords,
                collider_cell_size: voxel_size, // actually the coarse cell size from build_coarse_collider
                aabb_half,
                local_center,
                grid_origin,
                tree_depth,
            };

            // Insert or replace the cache component.
            if self.world.get::<&ColliderCache>(entity).is_ok() {
                let _ = self.world.remove_one::<ColliderCache>(entity);
            }
            let _ = self.world.insert_one(entity, cache);
        }
    }

    /// Build wireframe visualization for all physics colliders from cached data.
    pub(crate) fn build_collider_wireframes(&self) -> Vec<rkp_render::LineVertex> {
        use rkp_physics::rigid_body::{BodyType, ColliderShape};
        let mut verts = Vec::new();

        for (_entity, (transform, rb, cache)) in self.world.query::<(
            &crate::components::Transform,
            &crate::components::RigidBody,
            &crate::components::ColliderCache,
        )>().iter() {
            let color = match rb.body_type {
                BodyType::Dynamic => [0.2, 0.8, 0.2, 0.6],
                BodyType::Static => [0.5, 0.5, 0.8, 0.6],
                BodyType::KinematicPosition | BodyType::KinematicVelocity => [0.9, 0.6, 0.2, 0.6],
            };

            // Fitted shapes sit at `transform.position + local_center`, not
            // at `transform.position`, so they line up with off-center bakes.
            let center = transform.position + cache.local_center;
            match cache.shape {
                ColliderShape::Box => {
                    let min = center - cache.aabb_half;
                    let max = center + cache.aabb_half;
                    verts.extend(rkp_render::wireframe::aabb_wireframe(min, max, color));
                }
                ColliderShape::Sphere => {
                    let r = cache.aabb_half.max_element();
                    verts.extend(rkp_render::wireframe::sphere_wireframe(center, r, color));
                }
                ColliderShape::Capsule => {
                    let r = cache.aabb_half.x.max(cache.aabb_half.z).max(0.01);
                    let hh = (cache.aabb_half.y - r).max(0.01);
                    let top = center + glam::Vec3::new(0.0, hh, 0.0);
                    let bot = center - glam::Vec3::new(0.0, hh, 0.0);
                    verts.extend(rkp_render::wireframe::sphere_wireframe(top, r, color));
                    verts.extend(rkp_render::wireframe::sphere_wireframe(bot, r, color));
                    for angle in [0.0f32, std::f32::consts::FRAC_PI_2, std::f32::consts::PI, 3.0 * std::f32::consts::FRAC_PI_2] {
                        let offset = glam::Vec3::new(angle.cos() * r, 0.0, angle.sin() * r);
                        verts.push(rkp_render::LineVertex { position: (top + offset).to_array(), color });
                        verts.push(rkp_render::LineVertex { position: (bot + offset).to_array(), color });
                    }
                }
                ColliderShape::Auto => {
                    // 24 line vertices per coarse cell; cap so the wireframe
                    // pass never asks for a vertex buffer the GPU can't allocate.
                    // Above this, fall back to the AABB outline.
                    const MAX_WIRE_CELLS: usize = 32_768;
                    if !cache.voxel_coords.is_empty()
                        && cache.voxel_coords.len() <= MAX_WIRE_CELLS
                    {
                        let cs = cache.collider_cell_size;

                        let offset = cache.grid_origin * transform.scale;
                        for coord in &cache.voxel_coords {
                            // Match Rapier: min = coord * cell_size, max = (coord+1) * cell_size,
                            // plus grid_origin offset to align with rendered geometry.
                            let local_min = glam::Vec3::new(
                                coord.x as f32 * cs,
                                coord.y as f32 * cs,
                                coord.z as f32 * cs,
                            );
                            let local_max = glam::Vec3::new(
                                (coord.x + 1) as f32 * cs,
                                (coord.y + 1) as f32 * cs,
                                (coord.z + 1) as f32 * cs,
                            );
                            let world_min = transform.position + offset + local_min * transform.scale;
                            let world_max = transform.position + offset + local_max * transform.scale;
                            verts.extend(rkp_render::wireframe::aabb_wireframe(world_min, world_max, color));
                        }
                    } else {
                        let min = transform.position - cache.aabb_half;
                        let max = transform.position + cache.aabb_half;
                        verts.extend(rkp_render::wireframe::aabb_wireframe(min, max, color));
                    }
                }
            }
        }

        verts
    }
}
