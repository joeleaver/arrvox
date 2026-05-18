//! Per-viewport camera state, screen ↔ ray conversions, and play-mode
//! viewport entry / exit.
//!
//! Each `Viewport` carries its own editor camera; these methods keep
//! MAIN's legacy `self.camera` in sync for now (Phase 3 of the
//! viewport refactor — multi-viewport rendering removes the sync once
//! the rest of the engine reads through `Viewports`).

use super::state::EngineState;

impl EngineState {
    /// On PlayStart: hand the MAIN viewport over to the active scene
    /// camera (if one exists) and flip its layer mask so editor-only
    /// helpers vanish and HUD becomes visible.
    pub(crate) fn enter_play_mode_viewports(&mut self) {
        use crate::components::{Camera, EditorMetadata};
        use crate::viewport::{layer, CameraSource, SceneFilter, ViewportId};

        // Find the scene camera flagged active. If multiple are flagged,
        // pick the first the iteration yields — scene authoring should
        // ensure exactly one.
        let scene_cam = self.world.query::<&Camera>().iter()
            .find(|(_, c)| c.active)
            .map(|(e, _)| e);

        if let Some(main) = self.viewports.get_mut(ViewportId::MAIN) {
            if let Some(entity) = scene_cam {
                let name = self.world
                    .get::<&EditorMetadata>(entity)
                    .map(|m| m.name.clone())
                    .unwrap_or_else(|_| format!("{entity:?}"));
                self.console.info(format!("Play mode: camera → '{name}'"));
                main.runtime_override = Some(CameraSource::Entity(entity));
            } else {
                self.console.warn("Play mode: no active scene camera found, \
                                   keeping editor camera");
            }
            main.filter = SceneFilter {
                base_layers: layer::DEFAULT | layer::UI,
                focus_entity: None,
            };
        }
    }

    /// Apply a scripted viewport request (from the behavior system).
    /// Currently all requests target MAIN; per-viewport routing would use
    /// a `ViewportId` payload on the request enum.
    pub(crate) fn apply_viewport_request(&mut self, req: crate::behavior::ViewportRequest) {
        use crate::behavior::ViewportRequest;
        use crate::viewport::{CameraSource, ViewportId};
        let Some(main) = self.viewports.get_mut(ViewportId::MAIN) else { return };
        match req {
            ViewportRequest::SetActiveCamera(entity) => {
                main.runtime_override = Some(CameraSource::Entity(entity));
            }
            ViewportRequest::ClearActiveCamera => {
                main.runtime_override = None;
            }
        }
    }

    /// On PlayStop: clear the runtime override and restore the editor
    /// layer mask. The editor camera state was untouched throughout play
    /// mode, so the user lands exactly where they left off.
    pub(crate) fn exit_play_mode_viewports(&mut self) {
        use crate::viewport::{layer, SceneFilter, ViewportId};
        if let Some(main) = self.viewports.get_mut(ViewportId::MAIN) {
            main.runtime_override = None;
            main.filter = SceneFilter {
                base_layers: layer::DEFAULT | layer::EDITOR_ONLY,
                focus_entity: None,
            };
        }
    }

    /// Read the `(position, yaw, pitch, fov, near, far)` 6-tuple from a
    /// scene-camera entity. Returns `None` if the entity is missing
    /// either a `Transform` or `Camera` component, or has been despawned —
    /// callers fall back to the editor camera in that case.
    ///
    /// Yaw/pitch derive from the Transform's Euler rotation: yaw is Y in
    /// radians, pitch is X in radians (matching the editor's fly-camera
    /// convention so play-mode → edit-mode "Look Through" stays continuous).
    pub(crate) fn read_entity_camera(&self, entity: hecs::Entity)
        -> Option<(glam::Vec3, f32, f32, f32, f32, f32)>
    {
        use crate::components::{Camera, Transform};
        let transform = self.world.get::<&Transform>(entity).ok()?;
        let cam = self.world.get::<&Camera>(entity).ok()?;
        let yaw = transform.rotation.y.to_radians();
        let pitch = transform.rotation.x.to_radians();
        Some((transform.position, yaw, pitch, cam.fov, cam.near, cam.far))
    }

    /// Mirror the legacy `self.camera` state into `viewports[MAIN].editor_camera`.
    /// Phase 1 keeps the legacy field as the source of truth; the viewport copy
    /// is kept in sync so later phases can flip the dependency direction
    /// without surprises.
    pub(crate) fn sync_main_viewport_from_legacy_camera(&mut self) {
        use crate::viewport::{EditorCamera, FlyCameraState, ViewportId};
        if let Some(main) = self.viewports.get_mut(ViewportId::MAIN) {
            main.editor_camera = EditorCamera::Fly(FlyCameraState {
                position: self.camera.position,
                yaw: self.camera.yaw,
                pitch: self.camera.pitch,
                fov: self.camera.fov,
                near: self.camera.near,
                far: self.camera.far,
            });
        }
    }

    pub(crate) fn build_camera_uniforms(&self, viewport_id: crate::viewport::ViewportId)
        -> arvx_render::arvx_scene::CameraUniforms
    {
        use crate::viewport::{CameraSource, EditorCamera, ViewportId};
        let viewport = self.viewports
            .get(viewport_id)
            .expect("build_camera_uniforms: viewport must exist");

        // Camera resolution priority (Phase 5):
        //   1. runtime_override → entity's Transform + Camera components
        //   2. MAIN: legacy `self.camera` (still source of truth, synced
        //      into editor_camera by sync_main_viewport_from_legacy_camera)
        //   3. Other viewports: their own editor_camera
        let from_entity = viewport.runtime_override.and_then(|src| match src {
            CameraSource::Entity(entity) => self.read_entity_camera(entity),
        });
        let (position, yaw, pitch, fov, near, far) = if let Some(c) = from_entity {
            c
        } else if viewport_id == ViewportId::MAIN {
            (self.camera.position, self.camera.yaw, self.camera.pitch,
             self.camera.fov, self.camera.near, self.camera.far)
        } else {
            match viewport.editor_camera {
                EditorCamera::Fly(s) => (s.position, s.yaw, s.pitch, s.fov, s.near, s.far),
                EditorCamera::Turntable(t) => {
                    // Convert orbit (yaw/pitch + distance about target) to
                    // equivalent eye-position + look direction.
                    let dir = glam::Vec3::new(
                        -t.yaw.sin() * t.pitch.cos(),
                        t.pitch.sin(),
                        -t.yaw.cos() * t.pitch.cos(),
                    );
                    let position = t.target - dir * t.distance;
                    (position, t.yaw, t.pitch, t.fov, t.near, t.far)
                }
            }
        };

        let forward = glam::Vec3::new(
            -yaw.sin() * pitch.cos(),
            pitch.sin(),
            -yaw.cos() * pitch.cos(),
        ).normalize();
        let right = forward.cross(glam::Vec3::Y).normalize();
        let up = right.cross(forward).normalize();

        let fov_rad = fov.to_radians();
        let half_fov_tan = (fov_rad * 0.5).tan();
        let aspect = viewport.width as f32 / viewport.height.max(1) as f32;

        let view = glam::Mat4::look_to_rh(position, forward, glam::Vec3::Y);
        let proj = glam::Mat4::perspective_rh(fov_rad, aspect, near, far);
        let view_proj = proj * view;

        // Render-layer + focus filter from this viewport's SceneFilter.
        // u32::MAX defaults pass everything (no real object_id is u32::MAX
        // since they're sequential from 0).
        let focus_object_id = viewport.filter.focus_entity
            .and_then(|e| self.entity_to_gpu.get(&e).copied())
            .map(|idx| idx as u32)
            .unwrap_or(u32::MAX);
        let layer_mask = viewport.filter.base_layers;

        arvx_render::arvx_scene::CameraUniforms {
            position: [position.x, position.y, position.z, 1.0],
            forward: [forward.x, forward.y, forward.z, 0.0],
            right: [right.x * half_fov_tan * aspect, right.y * half_fov_tan * aspect, right.z * half_fov_tan * aspect, 0.0],
            up: [up.x * half_fov_tan, up.y * half_fov_tan, up.z * half_fov_tan, 0.0],
            resolution: [viewport.width as f32, viewport.height as f32],
            jitter: [0.0, 0.0],
            layer_mask,
            focus_object_id,
            _pad: [0; 2],
            prev_vp: viewport.prev_view_proj,
            view_proj: view_proj.to_cols_array_2d(),
        }
    }

    /// Screen-space ray from pixel coordinates.
    /// Phase 4: rays come from MAIN's camera — sculpt/paint/picking are
    /// MAIN-only operations.
    pub(crate) fn screen_to_ray(&self, px: f32, py: f32) -> (glam::Vec3, glam::Vec3) {
        self.screen_to_ray_for_viewport(crate::viewport::ViewportId::MAIN, px, py)
    }

    /// Unproject a pixel position to a world-space ray through the
    /// given viewport's camera. Each viewport has its own camera +
    /// resolution — BUILD's turntable ray lands on the procedural's
    /// gizmo handles, not on MAIN's fly-cam scene.
    pub(crate) fn screen_to_ray_for_viewport(
        &self,
        viewport_id: crate::viewport::ViewportId,
        px: f32,
        py: f32,
    ) -> (glam::Vec3, glam::Vec3) {
        let cam = self.build_camera_uniforms(viewport_id);
        let (vw, vh) = self
            .viewports
            .get(viewport_id)
            .map(|v| (v.width as f32, v.height as f32))
            .unwrap_or((self.width as f32, self.height as f32));

        let vp = glam::Mat4::from_cols_array_2d(&cam.view_proj);
        let inv_vp = vp.inverse();

        let ndc_x = (px / vw) * 2.0 - 1.0;
        let ndc_y = 1.0 - (py / vh) * 2.0;

        let near = inv_vp.project_point3(glam::Vec3::new(ndc_x, ndc_y, -1.0));
        let far = inv_vp.project_point3(glam::Vec3::new(ndc_x, ndc_y, 1.0));
        let dir = (far - near).normalize();
        let origin = glam::Vec3::new(cam.position[0], cam.position[1], cam.position[2]);
        (origin, dir)
    }
}
