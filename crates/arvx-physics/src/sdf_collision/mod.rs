//! SDF collision adapter for generating contacts between rigid bodies
//! and the SDF world.
//!
//! Instead of mesh colliders, this module samples the SDF at points on a
//! body's surface and generates contact normals/depths from the distance
//! field gradient. The resulting contacts can be applied as position
//! corrections and velocity adjustments.

use glam::{Quat, Vec3};
use serde::{Deserialize, Serialize};

use crate::rapier_world::{from_rapier_vec3, to_rapier_vec3, PhysicsWorld};
use rapier3d::prelude::*;

// ---------------------------------------------------------------------------
// ContactPoint
// ---------------------------------------------------------------------------

/// A contact between a rigid body surface point and the SDF world.
#[derive(Debug, Clone)]
pub struct ContactPoint {
    /// World-space position of the contact.
    pub position: Vec3,
    /// Outward-pointing surface normal from the SDF (points away from solid).
    pub normal: Vec3,
    /// Penetration depth. Positive means the sample point is inside the SDF.
    pub penetration: f32,
}

// ---------------------------------------------------------------------------
// SdfQueryable trait
// ---------------------------------------------------------------------------

/// Abstraction for evaluating a signed distance field at arbitrary positions.
///
/// Negative distance means inside the surface. Implementors must provide both
/// [`evaluate`](SdfQueryable::evaluate) and [`gradient`](SdfQueryable::gradient).
/// Use [`gradient_central_diff`] for a finite-difference gradient if no analytic form exists.
pub trait SdfQueryable {
    /// Evaluate SDF distance at a world position. Negative = inside.
    fn evaluate(&self, pos: Vec3) -> f32;

    /// Compute the outward surface normal at `pos`.
    fn gradient(&self, pos: Vec3) -> Vec3;
}

/// Compute the SDF gradient at `pos` using central finite differences.
///
/// The epsilon is 0.01 world units — fine enough for physics contacts but
/// coarse enough to smooth over single-voxel noise.
pub fn gradient_central_diff(sdf: &dyn SdfQueryable, pos: Vec3) -> Vec3 {
    let eps = 0.01;
    let dx = sdf.evaluate(pos + Vec3::X * eps) - sdf.evaluate(pos - Vec3::X * eps);
    let dy = sdf.evaluate(pos + Vec3::Y * eps) - sdf.evaluate(pos - Vec3::Y * eps);
    let dz = sdf.evaluate(pos + Vec3::Z * eps) - sdf.evaluate(pos - Vec3::Z * eps);
    Vec3::new(dx, dy, dz).normalize_or_zero()
}

// ---------------------------------------------------------------------------
// CollisionShape
// ---------------------------------------------------------------------------

/// Collision shape used for sample-point generation.
///
/// These are simpler than the full SDF shapes — they define surface sample
/// points for the SDF contact test, not the SDF itself.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CollisionShape {
    /// Sphere with the given radius.
    Sphere {
        /// Sphere radius.
        radius: f32,
    },
    /// Axis-aligned box with the given half-extents.
    Box {
        /// Half-extents along each axis.
        half_extents: Vec3,
    },
    /// Capsule aligned along the Y axis.
    Capsule {
        /// Half the height of the cylindrical shaft (not including hemisphere caps).
        half_height: f32,
        /// Radius of the capsule.
        radius: f32,
    },
    /// Pre-cached surface sample points in local space.
    /// Used for SDF collision mode — points are generated at play start
    /// by sampling the entity's own SDF zero-crossing.
    Points {
        /// Local-space surface sample points.
        points: Vec<Vec3>,
    },
}

impl CollisionShape {
    /// Generate surface sample points in local space.
    ///
    /// These points are transformed to world space by the caller before
    /// SDF evaluation. The number of samples is fixed per shape type to
    /// keep the contact generation cost predictable.
    pub fn sample_points(&self) -> Vec<Vec3> {
        match self {
            CollisionShape::Sphere { radius } => {
                let r = *radius;
                let d3 = r / 3.0_f32.sqrt();
                vec![
                    // 6 axis-aligned points
                    Vec3::new(r, 0.0, 0.0),
                    Vec3::new(-r, 0.0, 0.0),
                    Vec3::new(0.0, r, 0.0),
                    Vec3::new(0.0, -r, 0.0),
                    Vec3::new(0.0, 0.0, r),
                    Vec3::new(0.0, 0.0, -r),
                    // 8 diagonal points (corners of inscribed cube)
                    Vec3::new(d3, d3, d3),
                    Vec3::new(d3, d3, -d3),
                    Vec3::new(d3, -d3, d3),
                    Vec3::new(d3, -d3, -d3),
                    Vec3::new(-d3, d3, d3),
                    Vec3::new(-d3, d3, -d3),
                    Vec3::new(-d3, -d3, d3),
                    Vec3::new(-d3, -d3, -d3),
                ]
            }
            CollisionShape::Box { half_extents } => {
                let h = *half_extents;
                let mut points = Vec::with_capacity(14);
                // 8 corners
                for sx in [-1.0_f32, 1.0] {
                    for sy in [-1.0_f32, 1.0] {
                        for sz in [-1.0_f32, 1.0] {
                            points.push(Vec3::new(h.x * sx, h.y * sy, h.z * sz));
                        }
                    }
                }
                // 6 face centers
                points.push(Vec3::new(h.x, 0.0, 0.0));
                points.push(Vec3::new(-h.x, 0.0, 0.0));
                points.push(Vec3::new(0.0, h.y, 0.0));
                points.push(Vec3::new(0.0, -h.y, 0.0));
                points.push(Vec3::new(0.0, 0.0, h.z));
                points.push(Vec3::new(0.0, 0.0, -h.z));
                points
            }
            CollisionShape::Capsule {
                half_height,
                radius,
            } => {
                let hh = *half_height;
                let r = *radius;
                vec![
                    // Top hemisphere: pole + 4 equatorial
                    Vec3::new(0.0, hh + r, 0.0),
                    Vec3::new(r, hh, 0.0),
                    Vec3::new(-r, hh, 0.0),
                    Vec3::new(0.0, hh, r),
                    Vec3::new(0.0, hh, -r),
                    // Bottom hemisphere: pole + 4 equatorial
                    Vec3::new(0.0, -(hh + r), 0.0),
                    Vec3::new(r, -hh, 0.0),
                    Vec3::new(-r, -hh, 0.0),
                    Vec3::new(0.0, -hh, r),
                    Vec3::new(0.0, -hh, -r),
                    // 2 shaft midpoints
                    Vec3::new(r, 0.0, 0.0),
                    Vec3::new(-r, 0.0, 0.0),
                ]
            }
            CollisionShape::Points { points } => points.clone(),
        }
    }

    /// Build a Rapier [`Collider`] matching this shape.
    ///
    /// For [`Points`](CollisionShape::Points), builds a bounding sphere
    /// from the maximum point distance to origin.
    pub fn to_rapier_collider(&self) -> Collider {
        match self {
            CollisionShape::Sphere { radius } => ColliderBuilder::ball(*radius).build(),
            CollisionShape::Box { half_extents } => {
                ColliderBuilder::cuboid(half_extents.x, half_extents.y, half_extents.z).build()
            }
            CollisionShape::Capsule {
                half_height,
                radius,
            } => ColliderBuilder::capsule_y(*half_height, *radius).build(),
            CollisionShape::Points { points } => {
                let max_r = points
                    .iter()
                    .map(|p| p.length())
                    .fold(0.0_f32, f32::max)
                    .max(0.01);
                ColliderBuilder::ball(max_r).build()
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Contact generation
// ---------------------------------------------------------------------------

/// Generate SDF contacts for a shape at a given position and rotation.
///
/// For each sample point on the shape's surface, evaluates the SDF.
/// Points with SDF distance below `contact_threshold` generate a
/// [`ContactPoint`]. Results are sorted by penetration depth (deepest first).
pub fn generate_sdf_contacts(
    shape: &CollisionShape,
    position: Vec3,
    rotation: Quat,
    sdf: &dyn SdfQueryable,
    contact_threshold: f32,
) -> Vec<ContactPoint> {
    let local_points = shape.sample_points();
    let mut contacts = Vec::new();

    for lp in &local_points {
        // Transform local sample point to world space
        let world_point = position + rotation * *lp;

        let distance = sdf.evaluate(world_point);

        // Contact fires when sample point is within threshold of the surface.
        // Penetration = -distance (how far past the surface). For points
        // still outside but within threshold, penetration is negative —
        // the correction nudges toward the surface without overshooting.
        if distance < contact_threshold {
            let normal = sdf.gradient(world_point);
            contacts.push(ContactPoint {
                position: world_point,
                normal,
                penetration: -distance,
            });
        }
    }

    // Sort by penetration depth, deepest first
    contacts.sort_by(|a, b| {
        b.penetration
            .partial_cmp(&a.penetration)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    contacts
}

// ---------------------------------------------------------------------------
// Contact application
// ---------------------------------------------------------------------------

/// Apply SDF contacts using hybrid position correction + impulse response.
///
/// 1. **Position correction** (immediate): pushes the body out of the surface
///    to prevent tunneling. This is non-negotiable — without it, fast objects
///    pass through thin geometry.
/// 2. **Velocity impulse** (via Rapier): cancels velocity into the surface and
///    applies at the contact point so Rapier computes correct torque. Rapier
///    then handles angular response, friction, damping, and sleeping.
pub fn apply_sdf_contacts(
    world: &mut PhysicsWorld,
    handle: RigidBodyHandle,
    contacts: &[ContactPoint],
    friction: f32,
) {
    if contacts.is_empty() {
        return;
    }

    let body = match world.rigid_body_set.get_mut(handle) {
        Some(b) => b,
        None => return,
    };

    if !body.is_dynamic() {
        return;
    }

    let mass = body.mass();
    if mass < 1e-6 {
        return;
    }

    // --- Phase 1: Position correction (immediate, prevents tunneling) ---
    // Find the deepest penetration and push the body out.
    let deepest = contacts.iter()
        .filter(|c| c.penetration > 0.0 && c.normal.length_squared() > 0.001)
        .max_by(|a, b| a.penetration.partial_cmp(&b.penetration).unwrap_or(std::cmp::Ordering::Equal));

    if let Some(contact) = deepest {
        let correction = contact.normal * contact.penetration;
        let current = from_rapier_vec3(body.translation());
        body.set_translation(to_rapier_vec3(current + correction), true);
    }

    // --- Phase 2: Velocity response (via Rapier impulse API) ---
    // Distribute the velocity cancellation impulse across all active contacts
    // so that the total impulse equals exactly one velocity cancellation,
    // regardless of contact count. Each contact's share creates torque from
    // its offset from the body center.
    let vel = from_rapier_vec3(body.linvel());
    let active_contacts: Vec<_> = contacts.iter()
        .filter(|c| c.normal.length_squared() > 0.001)
        .collect();
    let n_contacts = active_contacts.len().max(1) as f32;

    for contact in &active_contacts {
        let n = contact.normal;
        let vel_normal = vel.dot(n);
        if vel_normal < 0.0 {
            // Each contact gets 1/N of the total velocity-cancel impulse.
            let normal_impulse = n * (-vel_normal) * mass / n_contacts;
            body.apply_impulse_at_point(
                to_rapier_vec3(normal_impulse),
                to_rapier_vec3(contact.position),
                true,
            );

            // Friction: also divided by contact count.
            let vel_tangent = vel - n * vel_normal;
            if vel_tangent.length() > 0.001 {
                let friction_impulse = -vel_tangent.normalize()
                    * vel_tangent.length() * mass * friction.clamp(0.0, 1.0)
                    / n_contacts;
                body.apply_impulse_at_point(
                    to_rapier_vec3(friction_impulse),
                    to_rapier_vec3(contact.position),
                    true,
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Test SDF implementations
// ---------------------------------------------------------------------------

/// An infinite ground plane at a given Y height.
///
/// Useful for testing and simple scenes. The SDF is simply `pos.y - height`.
pub struct GroundPlaneSdf {
    /// Y coordinate of the ground plane.
    pub height: f32,
}

impl SdfQueryable for GroundPlaneSdf {
    fn evaluate(&self, pos: Vec3) -> f32 {
        pos.y - self.height
    }

    fn gradient(&self, _pos: Vec3) -> Vec3 {
        Vec3::Y
    }
}

/// A sphere SDF for testing.
pub struct SphereSdf {
    /// Center of the sphere.
    pub center: Vec3,
    /// Radius of the sphere.
    pub radius: f32,
}

impl SdfQueryable for SphereSdf {
    fn evaluate(&self, pos: Vec3) -> f32 {
        (pos - self.center).length() - self.radius
    }

    fn gradient(&self, pos: Vec3) -> Vec3 {
        (pos - self.center).normalize_or_zero()
    }
}

// ---------------------------------------------------------------------------
// V2 Object-Centric Collision
// ---------------------------------------------------------------------------

/// Result of a v2 object-aware SDF collision query.
#[derive(Debug, Clone)]
pub struct V2CollisionResult {
    /// Signed distance at the query point.
    pub distance: f32,
    /// Material ID at the surface.
    pub material_id: u16,
    /// Object ID that was hit (0 = terrain, u32::MAX = no hit).
    pub object_id: u32,
    /// Surface normal estimate.
    pub normal: Vec3,
}

/// Evaluate a single v2 object's SDF at `world_pos`.
///
/// Each object is a sphere in object-local space. The tuple fields are:
/// - `id`: object identity
/// - `position`: object world-space centre
/// - `scale`: uniform scale (radius of the bounding sphere / SDF extent)
/// - `rotation`: object orientation
/// - `sdf_radius`: SDF sphere radius in local space
///
/// Returns `(distance_in_world_space, id)`.
fn eval_object_sdf(
    world_pos: Vec3,
    id: u32,
    obj_pos: Vec3,
    scale: f32,
    rotation: glam::Quat,
    sdf_radius: f32,
) -> (f32, u32) {
    // Transform world_pos into object-local space.
    let local = rotation.inverse() * ((world_pos - obj_pos) / scale.max(1e-6));
    // SDF in local space: sphere of radius sdf_radius.
    let local_dist = local.length() - sdf_radius;
    // Scale distance back to world space.
    (local_dist * scale, id)
}

/// Query the v2 scene for SDF collision at a world position.
///
/// Evaluates all objects (sphere SDF in object-local space) and optional
/// terrain. Returns the hit with the smallest (most-negative / closest)
/// signed distance.
///
/// # Arguments
///
/// * `world_pos` — world-space query point.
/// * `objects` — slice of `(id, position, scale, rotation, sdf_radius)` tuples.
/// * `terrain_height` — optional flat terrain at this Y height.
///
/// # Returns
///
/// [`V2CollisionResult`] for the nearest surface. If nothing is within
/// 1 000 world units the result has `object_id == u32::MAX`.
pub fn query_v2_scene(
    world_pos: Vec3,
    objects: &[(u32, Vec3, f32, glam::Quat, f32)],
    terrain_height: Option<f32>,
) -> V2CollisionResult {
    let mut best_dist = f32::MAX;
    let mut best_id: u32 = u32::MAX;
    let mut best_material: u16 = 0;

    // Evaluate each object SDF.
    for &(id, pos, scale, rot, sdf_radius) in objects {
        let (dist, _) = eval_object_sdf(world_pos, id, pos, scale, rot, sdf_radius);
        if dist < best_dist {
            best_dist = dist;
            best_id = id;
            // Material ID: low 16 bits of object id as a simple default.
            best_material = (id & 0xFFFF) as u16;
        }
    }

    // Evaluate terrain (infinite plane).
    if let Some(th) = terrain_height {
        let terrain_dist = world_pos.y - th;
        if terrain_dist < best_dist {
            best_dist = terrain_dist;
            best_id = 0; // 0 == terrain
            best_material = 0;
        }
    }

    // Estimate normal at the winning surface.
    let normal = estimate_normal_v2(world_pos, objects, terrain_height, 0.01);

    V2CollisionResult {
        distance: best_dist,
        material_id: best_material,
        object_id: best_id,
        normal,
    }
}

/// Estimate the surface normal at `world_pos` via central finite differences
/// over the v2 scene SDF.
///
/// `epsilon` controls the step size for the finite-difference stencil (0.01
/// gives good results for typical scene scales).
pub fn estimate_normal_v2(
    world_pos: Vec3,
    objects: &[(u32, Vec3, f32, glam::Quat, f32)],
    terrain_height: Option<f32>,
    epsilon: f32,
) -> Vec3 {
    /// Evaluate the minimum SDF across all objects + terrain.
    fn scene_dist(
        p: Vec3,
        objects: &[(u32, Vec3, f32, glam::Quat, f32)],
        terrain_height: Option<f32>,
    ) -> f32 {
        let mut d = f32::MAX;
        for &(id, pos, scale, rot, sdf_radius) in objects {
            let (od, _) = eval_object_sdf(p, id, pos, scale, rot, sdf_radius);
            if od < d {
                d = od;
            }
        }
        if let Some(th) = terrain_height {
            let td = p.y - th;
            if td < d {
                d = td;
            }
        }
        d
    }

    let e = epsilon;
    let dx = scene_dist(world_pos + Vec3::X * e, objects, terrain_height)
        - scene_dist(world_pos - Vec3::X * e, objects, terrain_height);
    let dy = scene_dist(world_pos + Vec3::Y * e, objects, terrain_height)
        - scene_dist(world_pos - Vec3::Y * e, objects, terrain_height);
    let dz = scene_dist(world_pos + Vec3::Z * e, objects, terrain_height)
        - scene_dist(world_pos - Vec3::Z * e, objects, terrain_height);

    Vec3::new(dx, dy, dz).normalize_or_zero()
}

#[cfg(test)]
mod tests;
