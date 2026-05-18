use super::*;
use glam::{Mat4, Quat, Vec3};
use std::f32::consts::FRAC_PI_2;


#[test]
fn transform_default_is_identity() {
    let t = Transform::default();
    assert_eq!(t.position, Vec3::ZERO);
    assert_eq!(t.rotation, Quat::IDENTITY);
    assert_eq!(t.scale, Vec3::ONE);
}

#[test]
fn transform_to_matrix_identity() {
    let t = Transform::default();
    let m = t.to_matrix();
    let diff = (m - Mat4::IDENTITY).abs_diff_eq(Mat4::ZERO, 1e-6);
    assert!(diff, "identity transform should produce identity matrix");
}

#[test]
fn transform_to_matrix_translation() {
    let t = Transform::new(Vec3::new(1.0, 2.0, 3.0), Quat::IDENTITY, Vec3::ONE);
    let m = t.to_matrix();
    let p = m.transform_point3(Vec3::ZERO);
    assert!((p - Vec3::new(1.0, 2.0, 3.0)).length() < 1e-5);
}

#[test]
fn transform_to_matrix_scale() {
    let t = Transform::new(Vec3::ZERO, Quat::IDENTITY, Vec3::splat(2.0));
    let m = t.to_matrix();
    let p = m.transform_point3(Vec3::ONE);
    assert!((p - Vec3::splat(2.0)).length() < 1e-5);
}

#[test]
fn transform_to_matrix_rotation() {
    let t = Transform::new(Vec3::ZERO, Quat::from_rotation_z(FRAC_PI_2), Vec3::ONE);
    let m = t.to_matrix();
    let p = m.transform_point3(Vec3::X);
    // 90° Z rotation: X -> Y
    assert!((p - Vec3::Y).length() < 1e-4);
}

#[test]
fn scene_node_new_defaults() {
    let node = SceneNode::new("test");
    assert_eq!(node.name, "test");
    assert!(matches!(node.sdf_source, SdfSource::None));
    assert!(matches!(node.blend_mode, BlendMode::SmoothUnion(_)));
    assert!(node.children.is_empty());
    assert!(node.metadata.visible);
    assert!(!node.metadata.locked);
}

#[test]
fn scene_node_analytical() {
    let node = SceneNode::analytical(
        "sphere",
        SdfPrimitive::Sphere { radius: 0.5 },
        1,
    );
    assert_eq!(node.name, "sphere");
    match &node.sdf_source {
        SdfSource::Analytical {
            primitive,
            material_id,
        } => {
            assert!(matches!(primitive, SdfPrimitive::Sphere { radius } if (*radius - 0.5).abs() < 1e-6));
            assert_eq!(*material_id, 1);
        }
        _ => panic!("expected Analytical"),
    }
}

#[test]
fn scene_node_builder_pattern() {
    let node = SceneNode::analytical(
        "box",
        SdfPrimitive::Box {
            half_extents: Vec3::ONE,
        },
        2,
    )
    .with_transform(Transform::new(Vec3::new(1.0, 0.0, 0.0), Quat::IDENTITY, Vec3::splat(0.5)))
    .with_blend_mode(BlendMode::Subtract);

    assert_eq!(node.local_transform.position, Vec3::new(1.0, 0.0, 0.0));
    assert_eq!(node.local_transform.scale, Vec3::splat(0.5));
    assert!(matches!(node.blend_mode, BlendMode::Subtract));
}

#[test]
fn scene_node_tree_building() {
    let mut root = SceneNode::new("root");
    let child_a = SceneNode::analytical(
        "arm_left",
        SdfPrimitive::Capsule {
            radius: 0.1,
            half_height: 0.3,
        },
        1,
    );
    let child_b = SceneNode::analytical(
        "arm_right",
        SdfPrimitive::Capsule {
            radius: 0.1,
            half_height: 0.3,
        },
        1,
    );
    root.add_child(child_a);
    root.add_child(child_b);

    assert_eq!(root.children.len(), 2);
    assert_eq!(root.node_count(), 3);
}

#[test]
fn scene_node_deep_tree() {
    let mut root = SceneNode::new("hips");
    let mut spine = SceneNode::new("spine");
    let mut chest = SceneNode::new("chest");
    let head = SceneNode::analytical(
        "head",
        SdfPrimitive::Sphere { radius: 0.15 },
        1,
    );
    chest.add_child(head);
    spine.add_child(chest);
    root.add_child(spine);

    assert_eq!(root.node_count(), 4);
}

#[test]
fn scene_node_find_by_name() {
    let mut root = SceneNode::new("root");
    let mut child = SceneNode::new("child");
    let grandchild = SceneNode::new("grandchild");
    child.add_child(grandchild);
    root.add_child(child);

    assert!(root.find_by_name("root").is_some());
    assert!(root.find_by_name("child").is_some());
    assert!(root.find_by_name("grandchild").is_some());
    assert!(root.find_by_name("missing").is_none());
}

#[test]
fn scene_node_find_by_name_mut() {
    let mut root = SceneNode::new("root");
    let child = SceneNode::new("child");
    root.add_child(child);

    if let Some(node) = root.find_by_name_mut("child") {
        node.metadata.locked = true;
    }
    assert!(root.children[0].metadata.locked);
}

#[test]
fn scene_node_display() {
    let node = SceneNode::analytical(
        "sphere",
        SdfPrimitive::Sphere { radius: 1.0 },
        0,
    );
    let s = format!("{node}");
    assert!(s.contains("sphere"));
    assert!(s.contains("Sphere"));
}

#[test]
fn blend_mode_default_is_smooth_union() {
    let mode = BlendMode::default();
    assert!(matches!(mode, BlendMode::SmoothUnion(r) if r > 0.0));
}

#[test]
fn node_metadata_default_is_visible_unlocked() {
    let m = NodeMetadata::default();
    assert!(m.visible);
    assert!(!m.locked);
    assert!(!m.selected);
    assert!(m.expand_in_tree);
}

#[test]
fn all_sdf_primitives_constructable() {
    let _s = SdfPrimitive::Sphere { radius: 1.0 };
    let _b = SdfPrimitive::Box {
        half_extents: Vec3::ONE,
    };
    let _c = SdfPrimitive::Capsule {
        radius: 0.1,
        half_height: 0.5,
    };
    let _t = SdfPrimitive::Torus {
        major_radius: 1.0,
        minor_radius: 0.2,
    };
    let _cy = SdfPrimitive::Cylinder {
        radius: 0.5,
        half_height: 1.0,
    };
    let _p = SdfPrimitive::Plane {
        normal: Vec3::Y,
        distance: 0.0,
    };
}

// ── Child access (A.1) ───────────────────────────────────────────────

#[test]
fn child_count_empty() {
    let node = SceneNode::new("leaf");
    assert_eq!(node.child_count(), 0);
}

#[test]
fn child_count_with_children() {
    let mut node = SceneNode::new("root");
    node.add_child(SceneNode::new("a"));
    node.add_child(SceneNode::new("b"));
    node.add_child(SceneNode::new("c"));
    assert_eq!(node.child_count(), 3);
}

#[test]
fn child_by_index() {
    let mut node = SceneNode::new("root");
    node.add_child(SceneNode::new("first"));
    node.add_child(SceneNode::new("second"));
    assert_eq!(node.child(0).unwrap().name, "first");
    assert_eq!(node.child(1).unwrap().name, "second");
}

#[test]
fn child_out_of_bounds_returns_none() {
    let node = SceneNode::new("root");
    assert!(node.child(0).is_none());
    assert!(node.child(99).is_none());
}

#[test]
fn child_mut_by_index() {
    let mut node = SceneNode::new("root");
    node.add_child(SceneNode::new("child"));
    node.child_mut(0).unwrap().name = "renamed".to_string();
    assert_eq!(node.child(0).unwrap().name, "renamed");
}

#[test]
fn remove_child_by_index() {
    let mut node = SceneNode::new("root");
    node.add_child(SceneNode::new("a"));
    node.add_child(SceneNode::new("b"));
    node.add_child(SceneNode::new("c"));
    let removed = node.remove_child(1);
    assert_eq!(removed.name, "b");
    assert_eq!(node.child_count(), 2);
    assert_eq!(node.child(0).unwrap().name, "a");
    assert_eq!(node.child(1).unwrap().name, "c");
}

#[test]
fn remove_child_by_name_found() {
    let mut node = SceneNode::new("root");
    node.add_child(SceneNode::new("keep"));
    node.add_child(SceneNode::new("remove_me"));
    let removed = node.remove_child_by_name("remove_me");
    assert!(removed.is_some());
    assert_eq!(removed.unwrap().name, "remove_me");
    assert_eq!(node.child_count(), 1);
}

#[test]
fn remove_child_by_name_not_found() {
    let mut node = SceneNode::new("root");
    node.add_child(SceneNode::new("child"));
    assert!(node.remove_child_by_name("nope").is_none());
    assert_eq!(node.child_count(), 1);
}

#[test]
fn insert_child_at_beginning() {
    let mut node = SceneNode::new("root");
    node.add_child(SceneNode::new("b"));
    node.insert_child_at(0, SceneNode::new("a"));
    assert_eq!(node.child(0).unwrap().name, "a");
    assert_eq!(node.child(1).unwrap().name, "b");
}

#[test]
fn insert_child_at_middle() {
    let mut node = SceneNode::new("root");
    node.add_child(SceneNode::new("a"));
    node.add_child(SceneNode::new("c"));
    node.insert_child_at(1, SceneNode::new("b"));
    assert_eq!(node.child(0).unwrap().name, "a");
    assert_eq!(node.child(1).unwrap().name, "b");
    assert_eq!(node.child(2).unwrap().name, "c");
}

#[test]
fn insert_child_at_end() {
    let mut node = SceneNode::new("root");
    node.add_child(SceneNode::new("a"));
    node.insert_child_at(1, SceneNode::new("b"));
    assert_eq!(node.child(1).unwrap().name, "b");
    assert_eq!(node.child_count(), 2);
}

#[test]
fn iter_children_count() {
    let mut node = SceneNode::new("root");
    node.add_child(SceneNode::new("a"));
    node.add_child(SceneNode::new("b"));
    assert_eq!(node.iter_children().count(), node.child_count());
}

#[test]
fn iter_children_mut_modify() {
    let mut node = SceneNode::new("root");
    node.add_child(SceneNode::new("a"));
    node.add_child(SceneNode::new("b"));
    for child in node.iter_children_mut() {
        child.metadata.locked = true;
    }
    assert!(node.child(0).unwrap().metadata.locked);
    assert!(node.child(1).unwrap().metadata.locked);
}

// ── Path-based access (A.2) ────────────────────────────────────────

#[test]
fn find_by_path_single_segment() {
    let mut root = SceneNode::new("root");
    root.add_child(SceneNode::new("child"));
    assert_eq!(root.find_by_path("child").unwrap().name, "child");
}

#[test]
fn find_by_path_multi_segment() {
    let mut root = SceneNode::new("root");
    let mut spine = SceneNode::new("spine");
    let mut chest = SceneNode::new("chest");
    chest.add_child(SceneNode::new("head"));
    spine.add_child(chest);
    root.add_child(spine);
    assert_eq!(
        root.find_by_path("spine/chest/head").unwrap().name,
        "head"
    );
}

#[test]
fn find_by_path_not_found() {
    let mut root = SceneNode::new("root");
    root.add_child(SceneNode::new("child"));
    assert!(root.find_by_path("nope").is_none());
    assert!(root.find_by_path("child/deep").is_none());
}

#[test]
fn find_by_path_empty_returns_none() {
    let root = SceneNode::new("root");
    assert!(root.find_by_path("").is_none());
}

#[test]
fn find_by_path_mut_modifies() {
    let mut root = SceneNode::new("root");
    let mut spine = SceneNode::new("spine");
    spine.add_child(SceneNode::new("chest"));
    root.add_child(spine);
    root.find_by_path_mut("spine/chest").unwrap().metadata.locked = true;
    assert!(root.find_by_path("spine/chest").unwrap().metadata.locked);
}

#[test]
fn walk_yields_correct_order() {
    let mut root = SceneNode::new("root");
    let mut a = SceneNode::new("a");
    a.add_child(SceneNode::new("a1"));
    root.add_child(a);
    root.add_child(SceneNode::new("b"));
    let names: Vec<&str> = root.walk().iter().map(|(_, n)| n.name.as_str()).collect();
    assert_eq!(names, vec!["root", "a", "a1", "b"]);
}

#[test]
fn walk_yields_correct_depths() {
    let mut root = SceneNode::new("root");
    let mut a = SceneNode::new("a");
    a.add_child(SceneNode::new("a1"));
    root.add_child(a);
    let depths: Vec<usize> = root.walk().iter().map(|(d, _)| *d).collect();
    assert_eq!(depths, vec![0, 1, 2]);
}

#[test]
fn all_names_collects_tree() {
    let mut root = SceneNode::new("root");
    root.add_child(SceneNode::new("x"));
    root.add_child(SceneNode::new("y"));
    assert_eq!(root.all_names(), vec!["root", "x", "y"]);
}

#[test]
fn all_blend_modes_constructable() {
    let _su = BlendMode::SmoothUnion(0.1);
    let _u = BlendMode::Union;
    let _s = BlendMode::Subtract;
    let _i = BlendMode::Intersect;
}

#[test]
fn all_sdf_sources_constructable() {
    let _none = SdfSource::None;
    let _analytical = SdfSource::Analytical {
        primitive: SdfPrimitive::Sphere { radius: 1.0 },
        material_id: 0,
    };
    let _voxelized = SdfSource::Voxelized {
        spatial_handle: SpatialHandle::BrickMap(BrickMapHandle {
            offset: 0,
            dims: glam::UVec3::new(4, 4, 4),
        }),
        voxel_size: 0.02,
        aabb: Aabb::new(Vec3::splat(-1.0), Vec3::splat(1.0)),
    };
}

#[test]
fn transform_combined_trs() {
    // Translation + rotation + scale combined
    let t = Transform::new(
        Vec3::new(5.0, 0.0, 0.0),
        Quat::from_rotation_z(FRAC_PI_2),
        Vec3::splat(2.0),
    );
    let m = t.to_matrix();
    // Point at (1, 0, 0) should be: scale(2) → (2,0,0), rotate 90°Z → (0,2,0), translate → (5,2,0)
    let p = m.transform_point3(Vec3::X);
    assert!((p.x - 5.0).abs() < 1e-4, "x: {}", p.x);
    assert!((p.y - 2.0).abs() < 1e-4, "y: {}", p.y);
    assert!(p.z.abs() < 1e-4, "z: {}", p.z);
}
