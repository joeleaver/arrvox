//! Editor camera with orbit and fly modes.
//!
//! Pure math — depends only on glam and rkp_runtime::input.

use glam::Vec3;
use rkp_runtime::input::InputSystem;

/// Camera control state — owns interaction parameters, not the camera transform.
#[derive(Debug, Clone, Copy)]
pub struct CameraControlState {
    pub mode: CameraMode,
    pub target: Vec3,
    pub orbit_distance: f32,
    pub fly_speed: f32,
    pub orbit_speed: f32,
    pub zoom_speed: f32,
    pub min_orbit_distance: f32,
    pub max_orbit_distance: f32,
    pub min_pitch: f32,
    pub max_pitch: f32,
}

impl Default for CameraControlState {
    fn default() -> Self {
        Self {
            mode: CameraMode::Fly,
            target: Vec3::ZERO,
            orbit_distance: 10.0,
            fly_speed: 5.0,
            orbit_speed: 0.005,
            zoom_speed: 1.0,
            min_orbit_distance: 0.5,
            max_orbit_distance: 500.0,
            min_pitch: -1.4,
            max_pitch: 1.4,
        }
    }
}

impl CameraControlState {
    /// Per-frame update using InputSystem action values.
    pub fn update(
        &mut self,
        input: &InputSystem,
        dt: f32,
        position: &mut Vec3,
        yaw: &mut f32,
        pitch: &mut f32,
    ) {
        let look = input.axis_2d("camera.look");
        let zoom = input.axis_1d("camera.zoom");
        let orbit_held = input.pressed("camera.orbit");
        let pan_held = input.pressed("camera.pan");

        match self.mode {
            CameraMode::Orbit => {
                if orbit_held {
                    *yaw += look.x * self.orbit_speed;
                    *pitch = (*pitch + look.y * self.orbit_speed)
                        .clamp(self.min_pitch, self.max_pitch);
                    *position = self.position_from_orbit(*yaw, *pitch);
                }
                if pan_held {
                    let forward = (self.target - *position).normalize();
                    let right = forward.cross(Vec3::Y).normalize();
                    let cam_up = right.cross(forward).normalize();
                    let scale = self.orbit_distance * 0.002;
                    self.target += right * (-look.x * scale) + cam_up * (look.y * scale);
                    *position = self.position_from_orbit(*yaw, *pitch);
                }
                if zoom.abs() > f32::EPSILON {
                    self.orbit_distance = (self.orbit_distance - zoom * self.zoom_speed)
                        .clamp(self.min_orbit_distance, self.max_orbit_distance);
                    *position = self.position_from_orbit(*yaw, *pitch);
                }
            }
            CameraMode::Fly => {
                if orbit_held {
                    *yaw -= look.x * self.orbit_speed;
                    *pitch = (*pitch - look.y * self.orbit_speed)
                        .clamp(self.min_pitch, self.max_pitch);
                }
                let move_vec = input.axis_2d("camera.move");
                let elevate = input.axis_1d("camera.elevate");
                if move_vec.x != 0.0 || move_vec.y != 0.0 || elevate != 0.0 {
                    let dir = fly_direction(*yaw, *pitch);
                    let right = dir.cross(Vec3::Y).normalize();
                    let velocity = (dir * move_vec.y + right * move_vec.x + Vec3::Y * elevate)
                        * self.fly_speed * dt;
                    *position += velocity;
                }
                if zoom.abs() > f32::EPSILON {
                    let dir = fly_direction(*yaw, *pitch);
                    *position += dir * zoom * self.zoom_speed;
                }
            }
        }
    }

    pub fn position_from_orbit(&self, yaw: f32, pitch: f32) -> Vec3 {
        let offset = Vec3::new(
            pitch.cos() * yaw.sin(),
            pitch.sin(),
            pitch.cos() * yaw.cos(),
        ) * self.orbit_distance;
        self.target + offset
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CameraMode {
    Orbit,
    Fly,
}

fn fly_direction(yaw: f32, pitch: f32) -> Vec3 {
    Vec3::new(
        -yaw.sin() * pitch.cos(),
        pitch.sin(),
        -yaw.cos() * pitch.cos(),
    )
    .normalize()
}

/// Build the default editor action map with camera + edit bindings.
pub fn default_action_map() -> rkp_runtime::input::ActionMap {
    use rkp_runtime::input::*;

    ActionMap::new("editor", vec![
        // Camera movement (WASD)
        ActionDef::new("camera.move", ControlType::Axis2D, vec![
            Binding::Composite2D {
                up: PhysicalInput::Key(InputKeyCode::W),
                down: PhysicalInput::Key(InputKeyCode::S),
                left: PhysicalInput::Key(InputKeyCode::A),
                right: PhysicalInput::Key(InputKeyCode::D),
            },
        ]),
        // Camera look (mouse delta)
        ActionDef::new("camera.look", ControlType::Axis2D, vec![
            Binding::simple(PhysicalInput::MouseDelta),
        ]),
        // Camera elevation (Space/Shift)
        ActionDef::new("camera.elevate", ControlType::Axis1D, vec![
            Binding::CompositeAxis {
                positive: PhysicalInput::Key(InputKeyCode::Space),
                negative: PhysicalInput::Key(InputKeyCode::ShiftLeft),
            },
        ]),
        // Camera zoom (scroll wheel)
        ActionDef::new("camera.zoom", ControlType::Axis1D, vec![
            Binding::simple(PhysicalInput::ScrollWheel),
        ]),
        // Camera orbit (right mouse button)
        ActionDef::new("camera.orbit", ControlType::Digital, vec![
            Binding::simple(PhysicalInput::MouseButton(InputMouseButton::Right)),
        ]),
        // Camera pan (middle mouse button)
        ActionDef::new("camera.pan", ControlType::Digital, vec![
            Binding::simple(PhysicalInput::MouseButton(InputMouseButton::Middle)),
        ]),
        // Delete
        ActionDef::new("edit.delete", ControlType::Digital, vec![
            Binding::simple(PhysicalInput::Key(InputKeyCode::Delete)),
        ]),
    ])
}
