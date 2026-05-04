use super::*;
use glam::{Quat, Vec3};
use crate::rapier_world::PhysicsWorld;
use rapier3d::prelude::*;


#[test]
fn test_ground_plane_contact() {
    let ground = GroundPlaneSdf { height: 0.0 };
    let shape = CollisionShape::Sphere { radius: 0.5 };

    // Sphere centered at y=2.0, well above ground → no contacts
    let contacts = generate_sdf_contacts(
        &shape,
        Vec3::new(0.0, 2.0, 0.0),
        Quat::IDENTITY,
        &ground,
        0.02,
    );
    assert!(
        contacts.is_empty(),
        "sphere above ground should have no contacts, got {}",
        contacts.len()
    );

    // Sphere centered at y=0.3, bottom sample at y=-0.2 → penetrating
    let contacts = generate_sdf_contacts(
        &shape,
        Vec3::new(0.0, 0.3, 0.0),
        Quat::IDENTITY,
        &ground,
        0.02,
    );
    assert!(
        !contacts.is_empty(),
        "sphere intersecting ground should have contacts"
    );

    // All contacts should have upward normal (Y)
    for c in &contacts {
        assert!(
            c.normal.y > 0.9,
            "ground contact normal should point up, got {:?}",
            c.normal
        );
    }
}

#[test]
fn test_contact_penetration_depth() {
    let ground = GroundPlaneSdf { height: 0.0 };
    let shape = CollisionShape::Sphere { radius: 0.5 };

    // Sphere at y=0.0 → bottom point at y=-0.5, penetration = 0.5 + threshold
    let threshold = 0.02;
    let contacts = generate_sdf_contacts(
        &shape,
        Vec3::ZERO,
        Quat::IDENTITY,
        &ground,
        threshold,
    );
    assert!(!contacts.is_empty());

    // The deepest contact should be the bottom pole at y=-0.5
    let deepest = &contacts[0];
    // SDF at y=-0.5 is -0.5, penetration = threshold - (-0.5) = 0.52
    assert!(
        (deepest.penetration - 0.52).abs() < 0.05,
        "expected penetration ~0.52, got {}",
        deepest.penetration
    );
}

#[test]
fn test_gradient_central_diff() {
    let sphere = SphereSdf {
        center: Vec3::ZERO,
        radius: 1.0,
    };

    // Gradient at (2, 0, 0) should point in +X direction
    let grad = gradient_central_diff(&sphere, Vec3::new(2.0, 0.0, 0.0));
    assert!(
        grad.x > 0.95,
        "gradient at +X should point in +X, got {:?}",
        grad
    );
    assert!(grad.y.abs() < 0.1);
    assert!(grad.z.abs() < 0.1);

    // Gradient at (0, -2, 0) should point in -Y direction
    let grad = gradient_central_diff(&sphere, Vec3::new(0.0, -2.0, 0.0));
    assert!(
        grad.y < -0.95,
        "gradient at -Y should point in -Y, got {:?}",
        grad
    );
}

#[test]
fn test_sample_points_sphere() {
    let shape = CollisionShape::Sphere { radius: 1.0 };
    let points = shape.sample_points();
    assert_eq!(points.len(), 14, "sphere should have 14 sample points");

    // All points should be at distance 1.0 from origin
    for (i, p) in points.iter().enumerate() {
        let dist = p.length();
        assert!(
            (dist - 1.0).abs() < 0.01,
            "point {i} at {:?} has distance {dist}, expected 1.0",
            p
        );
    }
}

#[test]
fn test_sample_points_box() {
    let shape = CollisionShape::Box {
        half_extents: Vec3::new(1.0, 2.0, 3.0),
    };
    let points = shape.sample_points();
    assert_eq!(points.len(), 14, "box should have 14 sample points");

    // First 8 should be corners
    for p in &points[0..8] {
        assert!(
            (p.x.abs() - 1.0).abs() < 1e-6
                && (p.y.abs() - 2.0).abs() < 1e-6
                && (p.z.abs() - 3.0).abs() < 1e-6,
            "corner {:?} doesn't match half_extents",
            p
        );
    }

    // Last 6 should be face centers
    let face_centers = &points[8..14];
    // +X face center
    assert!((face_centers[0] - Vec3::new(1.0, 0.0, 0.0)).length() < 1e-6);
    // -X face center
    assert!((face_centers[1] - Vec3::new(-1.0, 0.0, 0.0)).length() < 1e-6);
}

#[test]
fn test_sample_points_capsule() {
    let shape = CollisionShape::Capsule {
        half_height: 1.0,
        radius: 0.5,
    };
    let points = shape.sample_points();
    assert_eq!(points.len(), 12, "capsule should have 12 sample points");

    // Top pole should be at (0, half_height + radius, 0)
    assert!(
        (points[0] - Vec3::new(0.0, 1.5, 0.0)).length() < 1e-6,
        "top pole: {:?}",
        points[0]
    );

    // Bottom pole should be at (0, -(half_height + radius), 0)
    assert!(
        (points[5] - Vec3::new(0.0, -1.5, 0.0)).length() < 1e-6,
        "bottom pole: {:?}",
        points[5]
    );
}

#[test]
fn test_contacts_sorted_by_depth() {
    let ground = GroundPlaneSdf { height: 0.0 };
    let shape = CollisionShape::Sphere { radius: 1.0 };

    // Sphere centered at y=0 → multiple contact points at different depths
    let contacts = generate_sdf_contacts(
        &shape,
        Vec3::ZERO,
        Quat::IDENTITY,
        &ground,
        0.02,
    );
    assert!(contacts.len() >= 2, "should have multiple contacts");

    // Verify sorted by penetration (deepest first)
    for i in 1..contacts.len() {
        assert!(
            contacts[i - 1].penetration >= contacts[i].penetration,
            "contacts not sorted: [{}].pen={} > [{}].pen={}",
            i - 1,
            contacts[i - 1].penetration,
            i,
            contacts[i].penetration
        );
    }
}

// -----------------------------------------------------------------------
// V2 collision tests
// -----------------------------------------------------------------------

#[test]
fn v2_collision_no_objects() {
    // Empty object list, no terrain → distance should be f32::MAX and
    // object_id should be u32::MAX (sentinel "no hit").
    let result = query_v2_scene(Vec3::ZERO, &[], None);
    assert_eq!(result.object_id, u32::MAX, "empty scene should return no-hit sentinel");
    assert_eq!(result.distance, f32::MAX);
}

#[test]
fn v2_collision_with_sphere_object() {
    // Single sphere object at the origin with radius 1.0 and unit scale.
    // A query point just outside the surface should return a small positive distance.
    let objects = [(
        42_u32,          // id
        Vec3::new(0.0, 0.0, 0.0), // position
        1.0_f32,         // scale
        glam::Quat::IDENTITY, // rotation
        1.0_f32,         // sdf_radius
    )];

    // Point on the surface (distance ~ 0)
    let on_surface = query_v2_scene(Vec3::new(1.0, 0.0, 0.0), &objects, None);
    assert!(
        on_surface.distance.abs() < 0.05,
        "point on sphere surface should have near-zero distance, got {}",
        on_surface.distance
    );
    assert_eq!(on_surface.object_id, 42, "should identify the sphere object");

    // Point well outside — positive distance
    let outside = query_v2_scene(Vec3::new(5.0, 0.0, 0.0), &objects, None);
    assert!(
        outside.distance > 0.0,
        "point outside sphere should have positive distance, got {}",
        outside.distance
    );

    // Point inside — negative distance
    let inside = query_v2_scene(Vec3::ZERO, &objects, None);
    assert!(
        inside.distance < 0.0,
        "point at sphere center should have negative distance, got {}",
        inside.distance
    );
}

#[test]
fn v2_collision_terrain_floor() {
    // Terrain at y=0, query below → negative distance (inside terrain).
    let result_below = query_v2_scene(Vec3::new(0.0, -1.0, 0.0), &[], Some(0.0));
    assert!(
        result_below.distance < 0.0,
        "point below terrain should be negative distance, got {}",
        result_below.distance
    );
    assert_eq!(result_below.object_id, 0, "terrain id should be 0");

    // Query above → positive distance.
    let result_above = query_v2_scene(Vec3::new(0.0, 2.0, 0.0), &[], Some(0.0));
    assert!(
        result_above.distance > 0.0,
        "point above terrain should have positive distance, got {}",
        result_above.distance
    );
    assert_eq!(result_above.object_id, 0, "terrain id should be 0");
}

#[test]
fn v2_collision_object_identity() {
    // Two objects at different positions. The closer one's id must be returned.
    let objects = [
        (10_u32, Vec3::new(-5.0, 0.0, 0.0), 1.0_f32, glam::Quat::IDENTITY, 1.0_f32),
        (20_u32, Vec3::new(1.0, 0.0, 0.0), 1.0_f32, glam::Quat::IDENTITY, 1.0_f32),
    ];

    // Query near object 20 (id=20)
    let result = query_v2_scene(Vec3::new(1.5, 0.0, 0.0), &objects, None);
    assert_eq!(
        result.object_id, 20,
        "should return id of closest object (20), got {}",
        result.object_id
    );

    // Query near object 10 (id=10)
    let result2 = query_v2_scene(Vec3::new(-4.5, 0.0, 0.0), &objects, None);
    assert_eq!(
        result2.object_id, 10,
        "should return id of closest object (10), got {}",
        result2.object_id
    );
}

#[test]
fn v2_normal_estimation() {
    // Sphere at origin, radius 1.0. Normal at (2,0,0) should point in +X.
    let objects = [(
        1_u32,
        Vec3::ZERO,
        1.0_f32,
        glam::Quat::IDENTITY,
        1.0_f32,
    )];

    let normal = estimate_normal_v2(Vec3::new(2.0, 0.0, 0.0), &objects, None, 0.01);
    assert!(
        normal.x > 0.9,
        "normal at +X side of sphere should point in +X, got {:?}",
        normal
    );
    assert!(normal.y.abs() < 0.1);
    assert!(normal.z.abs() < 0.1);

    // Terrain floor at y=0, normal should point up.
    let terrain_normal = estimate_normal_v2(Vec3::new(0.0, 0.5, 0.0), &[], Some(0.0), 0.01);
    assert!(
        terrain_normal.y > 0.9,
        "normal above terrain should point up, got {:?}",
        terrain_normal
    );
}

#[test]
fn test_apply_contacts_resolves_penetration() {
    let config = crate::rapier_world::PhysicsConfig {
        gravity: Vec3::ZERO, // no gravity for this test
        ..Default::default()
    };
    let mut world = PhysicsWorld::new(config);

    // Create a dynamic body at the origin (sphere radius 0.5)
    let body = RigidBodyBuilder::dynamic()
        .translation(to_rapier_vec3(Vec3::new(0.0, 0.0, 0.0)))
        .build();
    let handle = world.add_rigid_body(body);
    let collider = ColliderBuilder::ball(0.5).sensor(true).build();
    world.add_collider(collider, handle);

    // Simulate a contact pushing the body upward
    let contacts = vec![ContactPoint {
        position: Vec3::new(0.0, -0.5, 0.0),
        normal: Vec3::Y,
        penetration: 0.5,
    }];

    apply_sdf_contacts(&mut world, handle, &contacts, 0.5);

    // Impulse was applied — step to let Rapier integrate it.
    world.step(1.0 / 60.0);

    // Body should have moved up (impulse pushed it along +Y).
    let body = world.get_body(handle).unwrap();
    assert!(
        body.translation().y > 0.0,
        "body should have been pushed up: y={}",
        body.translation().y
    );
}
