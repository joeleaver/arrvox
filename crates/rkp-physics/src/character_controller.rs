//! Custom capsule-vs-SDF character controller with iterative slide.
//!
//! Instead of relying on Rapier's built-in character controller (which expects
//! mesh colliders), this module evaluates the SDF directly at capsule sample
//! points and resolves penetrations via iterative projection + slide.

use glam::Vec3;

use crate::sdf_collision::SdfQueryable;

// ---------------------------------------------------------------------------
// SdfCharacterController
// ---------------------------------------------------------------------------

/// A capsule-based character controller that moves through the SDF world
/// using iterative slide collision resolution.
///
/// The controller samples the SDF at points on its capsule surface, detects
/// penetrations, and resolves them by projecting out along the surface normal
/// while preserving tangential movement (sliding along walls).
#[derive(Debug, Clone)]
pub struct SdfCharacterController {
    /// Radius of the capsule. Default: `0.3`.
    pub capsule_radius: f32,
    /// Half-height of the capsule shaft (excluding hemisphere caps). Default: `0.5`.
    pub capsule_half_height: f32,
    /// Maximum distance below the capsule foot to snap to ground. Default: `0.1`.
    pub ground_snap_distance: f32,
    /// Maximum walkable slope angle in radians. Default: `50` degrees.
    pub max_slope_angle: f32,
    /// Maximum step height the controller can climb. Default: `0.3`.
    pub step_height: f32,
    /// Thin shell around the capsule to prevent exact-surface jitter. Default: `0.01`.
    pub skin_width: f32,
    /// Maximum slide iterations per move. Default: `4`.
    pub max_iterations: u32,

    // -- State --
    /// Whether the controller is currently touching walkable ground.
    pub grounded: bool,
    /// Normal of the ground surface (valid only when `grounded` is true).
    pub ground_normal: Vec3,
    /// Current velocity (used for gravity accumulation and jump).
    pub velocity: Vec3,
    /// Current world-space position (center of the capsule).
    pub position: Vec3,
}

impl SdfCharacterController {
    /// Create a new character controller with sensible defaults positioned at the origin.
    pub fn new() -> Self {
        Self {
            capsule_radius: 0.3,
            capsule_half_height: 0.5,
            ground_snap_distance: 0.1,
            max_slope_angle: 50.0_f32.to_radians(),
            step_height: 0.3,
            skin_width: 0.01,
            max_iterations: 4,
            grounded: false,
            ground_normal: Vec3::Y,
            velocity: Vec3::ZERO,
            position: Vec3::ZERO,
        }
    }

    /// Return surface sample points for the capsule at the given position.
    ///
    /// The capsule is Y-aligned. Points are distributed across:
    /// - Bottom hemisphere: 5 points (pole + 4 equatorial)
    /// - Top hemisphere: 5 points (pole + 4 equatorial)
    /// - Shaft midpoints: 4 points (cardinal directions at y=0)
    ///
    /// Total: 14 points.
    pub fn capsule_sample_points(&self, center: Vec3) -> [Vec3; 14] {
        let r = self.capsule_radius;
        let hh = self.capsule_half_height;
        [
            // Bottom hemisphere (5 points)
            center + Vec3::new(0.0, -(hh + r), 0.0),           // bottom pole
            center + Vec3::new(r, -hh, 0.0),                   // +X equator
            center + Vec3::new(-r, -hh, 0.0),                  // -X equator
            center + Vec3::new(0.0, -hh, r),                   // +Z equator
            center + Vec3::new(0.0, -hh, -r),                  // -Z equator
            // Top hemisphere (5 points)
            center + Vec3::new(0.0, hh + r, 0.0),              // top pole
            center + Vec3::new(r, hh, 0.0),                    // +X equator
            center + Vec3::new(-r, hh, 0.0),                   // -X equator
            center + Vec3::new(0.0, hh, r),                    // +Z equator
            center + Vec3::new(0.0, hh, -r),                   // -Z equator
            // Shaft midpoints (4 points at y=0)
            center + Vec3::new(r, 0.0, 0.0),
            center + Vec3::new(-r, 0.0, 0.0),
            center + Vec3::new(0.0, 0.0, r),
            center + Vec3::new(0.0, 0.0, -r),
        ]
    }

    /// Find the deepest penetration among the capsule sample points.
    ///
    /// Returns `Some((penetration_depth, outward_normal))` if any point is
    /// inside the SDF (distance < 0), or `None` if all points are outside.
    fn deepest_penetration(&self, center: Vec3, sdf: &dyn SdfQueryable) -> Option<(f32, Vec3)> {
        let points = self.capsule_sample_points(center);
        let mut worst_dist = f32::MAX;
        let mut worst_point = Vec3::ZERO;

        for &p in &points {
            let d = sdf.evaluate(p);
            if d < worst_dist {
                worst_dist = d;
                worst_point = p;
            }
        }

        if worst_dist < 0.0 {
            let normal = sdf.gradient(worst_point);
            let penetration = -worst_dist; // positive depth
            Some((penetration, normal))
        } else {
            None
        }
    }

    /// Move the controller by `desired_velocity * dt`, resolving SDF
    /// collisions via iterative slide.
    ///
    /// After this call, `self.position` holds the resolved position.
    /// The controller slides along surfaces it collides with, removing
    /// the normal component of remaining displacement at each iteration.
    pub fn move_and_slide(&mut self, desired_velocity: Vec3, sdf: &dyn SdfQueryable, dt: f32) {
        let mut remaining = desired_velocity * dt;

        for _ in 0..self.max_iterations {
            if remaining.length_squared() < 1e-8 {
                break;
            }

            let tentative = self.position + remaining;

            match self.deepest_penetration(tentative, sdf) {
                Some((penetration, normal)) if normal.length_squared() > 0.01 => {
                    // Push out of the surface
                    let push = normal * (penetration + self.skin_width);
                    self.position = tentative + push;

                    // Remove the normal component from remaining displacement
                    // so the next iteration slides along the surface.
                    let into_surface = remaining.dot(normal);
                    if into_surface < 0.0 {
                        remaining -= normal * into_surface;
                    } else {
                        // Already moving away from this surface
                        break;
                    }
                }
                _ => {
                    // No penetration — accept the position
                    self.position = tentative;
                    break;
                }
            }
        }
    }

    /// Update ground state by sampling the SDF below the capsule foot.
    ///
    /// Sets `self.grounded` and `self.ground_normal`. A surface is only
    /// considered "ground" if the angle between its normal and world-up is
    /// less than `max_slope_angle`.
    pub fn update_ground_state(&mut self, sdf: &dyn SdfQueryable) {
        // Foot position: bottom of capsule
        let foot = self.position - Vec3::new(0.0, self.capsule_half_height + self.capsule_radius, 0.0);
        let distance = sdf.evaluate(foot);

        if distance < self.ground_snap_distance {
            let normal = sdf.gradient(foot);
            let angle = normal.dot(Vec3::Y).acos();

            if angle <= self.max_slope_angle {
                self.grounded = true;
                self.ground_normal = normal;
            } else {
                // Too steep — treat as wall, not ground
                self.grounded = false;
                self.ground_normal = Vec3::Y;
            }
        } else {
            self.grounded = false;
            self.ground_normal = Vec3::Y;
        }
    }

    /// Apply gravity to the controller's velocity.
    ///
    /// When grounded, vertical velocity is zeroed (with a tiny downward bias
    /// for ground adherence). When airborne, gravity accumulates.
    pub fn apply_gravity(&mut self, gravity: f32, dt: f32) {
        if self.grounded {
            // Small downward velocity to maintain ground contact
            self.velocity.y = -0.01;
        } else {
            self.velocity.y -= gravity * dt;
        }
    }

    /// Initiate a jump if currently grounded.
    ///
    /// Sets vertical velocity to `jump_speed` and clears the grounded flag.
    pub fn jump(&mut self, jump_speed: f32) {
        if self.grounded {
            self.velocity.y = jump_speed;
            self.grounded = false;
        }
    }

    /// Attempt to step up over a small obstacle.
    ///
    /// When the controller hits a wall, this method tries:
    /// 1. Move up by `step_height`
    /// 2. Move forward by the remaining displacement
    /// 3. Move back down to land on the step
    ///
    /// If the upward path is clear and the forward step lands on solid
    /// ground, the step-up position is accepted. Otherwise the original
    /// position is preserved.
    pub fn step_up(
        &mut self,
        forward_dir: Vec3,
        forward_dist: f32,
        sdf: &dyn SdfQueryable,
    ) -> bool {
        if forward_dir.length_squared() < 1e-6 || forward_dist < 1e-4 {
            return false;
        }

        let original_pos = self.position;

        // 1. Try moving up
        let up_pos = self.position + Vec3::new(0.0, self.step_height, 0.0);
        if self.deepest_penetration(up_pos, sdf).is_some() {
            // Blocked above — can't step up
            return false;
        }

        // 2. Try moving forward at the elevated position
        let forward = forward_dir.normalize() * forward_dist;
        let forward_pos = up_pos + forward;
        if self.deepest_penetration(forward_pos, sdf).is_some() {
            // Still blocked — step-up doesn't help
            return false;
        }

        // 3. Try moving back down to land on the step
        let down_pos = forward_pos - Vec3::new(0.0, self.step_height, 0.0);
        let foot_after = down_pos - Vec3::new(0.0, self.capsule_half_height + self.capsule_radius, 0.0);
        let ground_dist = sdf.evaluate(foot_after);

        if ground_dist < self.ground_snap_distance && ground_dist > -(self.step_height * 0.5) {
            // Found ground at step level — accept
            // Resolve any minor penetration
            if let Some((pen, normal)) = self.deepest_penetration(down_pos, sdf) {
                self.position = down_pos + normal * (pen + self.skin_width);
            } else {
                self.position = down_pos;
            }
            true
        } else {
            // No ground at step level — reject
            self.position = original_pos;
            false
        }
    }
}

impl Default for SdfCharacterController {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// V2 character movement (object-aware query function)
// ---------------------------------------------------------------------------

/// Perform a v2 character slide step against per-object SDFs.
///
/// This is the v2 equivalent of [`SdfCharacterController::move_and_slide`] but
/// takes a generic query closure instead of an `impl SdfQueryable`. This lets
/// callers drive the SDF evaluation through the object BVH or any other
/// v2-aware data structure without requiring access to the full
/// [`SdfCharacterController`] state.
///
/// # Arguments
///
/// * `position` — current world-space capsule centre.
/// * `velocity` — current velocity vector.
/// * `capsule_radius` — capsule radius.
/// * `capsule_half_height` — half-height of the capsule shaft (excludes caps).
/// * `dt` — frame delta time in seconds.
/// * `query_fn` — closure that evaluates `(distance, surface_normal)` at a
///   world-space point. Should return distance < 0 when the point is inside
///   solid geometry.
///
/// # Returns
///
/// A tuple `(new_pos, new_vel, grounded)`:
///
/// * `new_pos` — resolved world-space capsule centre after collision.
/// * `new_vel` — velocity with surface-normal components cancelled where
///   collision was detected.
/// * `grounded` — `true` if the capsule foot is within `0.1` world units of
///   any surface with a mostly-upward normal (angle < 50°).
pub fn slide_step_v2(
    position: Vec3,
    velocity: Vec3,
    capsule_radius: f32,
    capsule_half_height: f32,
    dt: f32,
    query_fn: impl Fn(Vec3) -> (f32, Vec3),
) -> (Vec3, Vec3, bool) {
    const MAX_ITERATIONS: u32 = 4;
    const SKIN_WIDTH: f32 = 0.01;
    const GROUND_SNAP: f32 = 0.1;
    const MAX_SLOPE_RADIANS: f32 = 0.872_664_6; // 50 degrees

    /// Generate capsule sample points for a given centre.
    fn capsule_points(center: Vec3, r: f32, hh: f32) -> [Vec3; 14] {
        [
            // Bottom hemisphere (5 points)
            center + Vec3::new(0.0, -(hh + r), 0.0),
            center + Vec3::new(r, -hh, 0.0),
            center + Vec3::new(-r, -hh, 0.0),
            center + Vec3::new(0.0, -hh, r),
            center + Vec3::new(0.0, -hh, -r),
            // Top hemisphere (5 points)
            center + Vec3::new(0.0, hh + r, 0.0),
            center + Vec3::new(r, hh, 0.0),
            center + Vec3::new(-r, hh, 0.0),
            center + Vec3::new(0.0, hh, r),
            center + Vec3::new(0.0, hh, -r),
            // Shaft midpoints (4 points at y=0)
            center + Vec3::new(r, 0.0, 0.0),
            center + Vec3::new(-r, 0.0, 0.0),
            center + Vec3::new(0.0, 0.0, r),
            center + Vec3::new(0.0, 0.0, -r),
        ]
    }

    /// Find the deepest penetration at `center` using `query_fn`.
    fn deepest_penetration(
        center: Vec3,
        r: f32,
        hh: f32,
        query_fn: &impl Fn(Vec3) -> (f32, Vec3),
    ) -> Option<(f32, Vec3)> {
        let points = capsule_points(center, r, hh);
        let mut worst_dist = f32::MAX;
        let mut worst_normal = Vec3::Y;

        for &p in &points {
            let (d, n) = query_fn(p);
            if d < worst_dist {
                worst_dist = d;
                worst_normal = n;
            }
        }

        if worst_dist < 0.0 {
            Some((-worst_dist, worst_normal))
        } else {
            None
        }
    }

    let mut pos = position;
    let mut vel = velocity;
    let mut remaining = vel * dt;

    // Iterative slide.
    for _ in 0..MAX_ITERATIONS {
        if remaining.length_squared() < 1e-8 {
            break;
        }

        let tentative = pos + remaining;

        match deepest_penetration(tentative, capsule_radius, capsule_half_height, &query_fn) {
            Some((penetration, normal)) if normal.length_squared() > 0.01 => {
                let n = normal.normalize_or_zero();
                // Push out of the surface.
                pos = tentative + n * (penetration + SKIN_WIDTH);

                // Cancel velocity and remaining displacement into the surface.
                let into_surface = remaining.dot(n);
                if into_surface < 0.0 {
                    remaining -= n * into_surface;
                } else {
                    break;
                }
                let vel_into = vel.dot(n);
                if vel_into < 0.0 {
                    vel -= n * vel_into;
                }
            }
            _ => {
                pos = tentative;
                break;
            }
        }
    }

    // Determine grounded state: sample below the foot.
    let foot = pos - Vec3::new(0.0, capsule_half_height + capsule_radius, 0.0);
    let (foot_dist, foot_normal) = query_fn(foot);
    let grounded = if foot_dist < GROUND_SNAP {
        let angle = foot_normal.normalize_or_zero().dot(Vec3::Y).acos();
        angle <= MAX_SLOPE_RADIANS
    } else {
        false
    };

    (pos, vel, grounded)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sdf_collision::{GroundPlaneSdf, SphereSdf};

    #[test]
    fn test_controller_defaults() {
        let cc = SdfCharacterController::new();
        assert!((cc.capsule_radius - 0.3).abs() < 1e-6);
        assert!((cc.capsule_half_height - 0.5).abs() < 1e-6);
        assert!((cc.ground_snap_distance - 0.1).abs() < 1e-6);
        assert!((cc.max_slope_angle - 50.0_f32.to_radians()).abs() < 1e-4);
        assert!((cc.step_height - 0.3).abs() < 1e-6);
        assert!((cc.skin_width - 0.01).abs() < 1e-6);
        assert_eq!(cc.max_iterations, 4);
        assert!(!cc.grounded);
        assert_eq!(cc.velocity, Vec3::ZERO);
        assert_eq!(cc.position, Vec3::ZERO);
    }

    #[test]
    fn test_move_on_flat_ground() {
        let ground = GroundPlaneSdf { height: 0.0 };
        let mut cc = SdfCharacterController::new();
        // Place the controller so its foot is exactly on the ground
        // foot = pos.y - (hh + r) = pos.y - 0.8
        // For foot at y=0: pos.y = 0.8
        cc.position = Vec3::new(0.0, 0.8, 0.0);

        // Move horizontally
        let velocity = Vec3::new(5.0, 0.0, 0.0);
        let dt = 0.016;
        cc.move_and_slide(velocity, &ground, dt);

        // Should have moved in X
        assert!(
            cc.position.x > 0.0,
            "should move in X: pos={:?}",
            cc.position
        );
        // Should stay at approximately the same height (not fall through)
        assert!(
            (cc.position.y - 0.8).abs() < 0.2,
            "should stay at ground level: y={}",
            cc.position.y
        );
    }

    #[test]
    fn test_gravity_when_airborne() {
        let mut cc = SdfCharacterController::new();
        cc.position = Vec3::new(0.0, 10.0, 0.0);
        cc.grounded = false;

        let initial_vy = cc.velocity.y;
        cc.apply_gravity(9.81, 0.016);

        assert!(
            cc.velocity.y < initial_vy,
            "velocity.y should decrease: was {}, now {}",
            initial_vy,
            cc.velocity.y
        );
    }

    #[test]
    fn test_grounded_detection() {
        let ground = GroundPlaneSdf { height: 0.0 };
        let mut cc = SdfCharacterController::new();

        // On ground: foot at y=0, pos.y = 0.8
        cc.position = Vec3::new(0.0, 0.8, 0.0);
        cc.update_ground_state(&ground);
        assert!(cc.grounded, "should be grounded when foot is on ground");

        // In air: pos.y = 5.0, foot at y=4.2 — well above ground
        cc.position = Vec3::new(0.0, 5.0, 0.0);
        cc.update_ground_state(&ground);
        assert!(!cc.grounded, "should not be grounded when in air");
    }

    #[test]
    fn test_slide_along_wall() {
        // Place a sphere as a wall obstacle at x=2
        let wall = SphereSdf {
            center: Vec3::new(2.0, 1.0, 0.0),
            radius: 1.0,
        };
        let mut cc = SdfCharacterController::new();
        cc.position = Vec3::new(0.0, 1.0, 0.0);

        // Move directly toward the wall with a large velocity
        let velocity = Vec3::new(20.0, 0.0, 1.0);
        cc.move_and_slide(velocity, &wall, 0.1);

        // The controller should not end up inside the sphere
        let dist_to_center = (cc.position - Vec3::new(2.0, 1.0, 0.0)).length();
        assert!(
            dist_to_center >= 0.9, // at least roughly outside (sphere r=1, capsule r=0.3)
            "should not penetrate wall: dist_to_center={}",
            dist_to_center
        );
    }

    #[test]
    fn test_slope_limit() {
        // A steep slope: sphere centered below ground creates a steep surface
        // Use a tilted plane approximated by a sphere with large radius
        // For simplicity, just use a steep SphereSdf and check that the
        // controller is NOT grounded when on its steep side.

        // Ground plane at y=0 for baseline
        let ground = GroundPlaneSdf { height: 0.0 };
        let mut cc = SdfCharacterController::new();
        cc.max_slope_angle = 10.0_f32.to_radians(); // Very strict slope limit

        // On flat ground — should be grounded
        cc.position = Vec3::new(0.0, 0.8, 0.0);
        cc.update_ground_state(&ground);
        assert!(cc.grounded, "flat ground should be walkable");

        // On the side of a sphere (steep) — the gradient is nearly horizontal
        let steep_sphere = SphereSdf {
            center: Vec3::new(0.0, 0.0, 0.0),
            radius: 1.0,
        };
        // Position on the side: the gradient at (1.0, 0.0, 0.0) is (1,0,0) — horizontal
        // foot at (1.0 + 0.8_offset, ...) — steep surface
        cc.position = Vec3::new(1.8, 0.0, 0.0);
        cc.update_ground_state(&steep_sphere);
        assert!(
            !cc.grounded,
            "steep surface should not be walkable with strict slope limit"
        );
    }

    #[test]
    fn v2_slide_step_basic() {
        use super::slide_step_v2;

        // Query function: open space — always returns large positive distance
        // with an upward normal (mimics standing on flat ground far below).
        let open_space = |_p: Vec3| -> (f32, Vec3) { (100.0_f32, Vec3::Y) };

        let start = Vec3::new(0.0, 5.0, 0.0);
        let vel = Vec3::new(3.0, 0.0, 0.0);
        let dt = 0.1_f32;

        let (new_pos, new_vel, grounded) =
            slide_step_v2(start, vel, 0.3, 0.5, dt, open_space);

        // Should have moved roughly in +X (no collision).
        assert!(
            new_pos.x > 0.0,
            "should have moved in X: pos={:?}",
            new_pos
        );
        // Velocity should be unchanged (no collision).
        assert!(
            (new_vel - vel).length() < 1e-4,
            "velocity should be unchanged in open space: vel={:?}",
            new_vel
        );
        // Not grounded (100 units above any surface).
        assert!(!grounded, "should not be grounded in open space");
    }

    #[test]
    fn v2_slide_step_collision_stops_penetration() {
        use super::slide_step_v2;

        // Query function: floor at y=0 (character standing on it should not go below).
        // Returns negative distance below y=0, positive above.
        let floor = |p: Vec3| -> (f32, Vec3) { (p.y, Vec3::Y) };

        // Start just above the floor, capsule foot at y = 0.8 - 0.8 = 0.0
        let start = Vec3::new(0.0, 0.8, 0.0);
        // Velocity downward (into the floor)
        let vel = Vec3::new(0.0, -5.0, 0.0);
        let dt = 0.1_f32;

        let (new_pos, _new_vel, grounded) =
            slide_step_v2(start, vel, 0.3, 0.5, dt, floor);

        // Should not have gone below the floor significantly.
        // The capsule foot is at new_pos.y - 0.8; it should not be deeply below 0.
        let foot_y = new_pos.y - 0.8;
        assert!(
            foot_y >= -0.1,
            "capsule foot should not penetrate floor deeply: foot_y={}",
            foot_y
        );
        // Should be grounded.
        assert!(grounded, "should be grounded after landing on floor");
    }

    #[test]
    fn test_jump() {
        let ground = GroundPlaneSdf { height: 0.0 };
        let mut cc = SdfCharacterController::new();
        cc.position = Vec3::new(0.0, 0.8, 0.0);
        cc.update_ground_state(&ground);
        assert!(cc.grounded);

        cc.jump(5.0);
        assert!((cc.velocity.y - 5.0).abs() < 1e-6, "jump should set vy=5.0");
        assert!(!cc.grounded, "should no longer be grounded after jump");

        // Jumping when not grounded should do nothing
        cc.velocity.y = 2.0;
        cc.grounded = false;
        cc.jump(5.0);
        assert!(
            (cc.velocity.y - 2.0).abs() < 1e-6,
            "should not jump when airborne"
        );
    }
}
