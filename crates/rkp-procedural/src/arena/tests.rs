use super::*;
use crate::node_kind::{MaterialCombine, NodeKind, SphereParams};
use glam::Affine3A;


fn union_kind() -> NodeKind {
    NodeKind::Union {
        material_combine: MaterialCombine::Winner,
    }
}

fn sphere_kind() -> NodeKind {
    NodeKind::Sphere(SphereParams::default())
}

#[test]
fn create_and_query_root() {
    let obj = ProceduralObject::new(union_kind());
    assert_eq!(obj.node_count(), 1);
    assert!(obj.get(obj.root()).is_some());
}

#[test]
fn add_children() {
    let mut obj = ProceduralObject::new(union_kind());
    let a = obj.add_child(obj.root(), sphere_kind());
    let b = obj.add_child(obj.root(), sphere_kind());
    assert_eq!(obj.node_count(), 3);
    assert_eq!(obj.get(obj.root()).unwrap().children.len(), 2);
    assert_eq!(obj.get(a).unwrap().parent, Some(obj.root()));
    assert_eq!(obj.get(b).unwrap().parent, Some(obj.root()));
}

#[test]
#[should_panic(expected = "cannot add children to a leaf")]
fn add_child_to_leaf_panics() {
    let mut obj = ProceduralObject::new(union_kind());
    let leaf = obj.add_child(obj.root(), sphere_kind());
    obj.add_child(leaf, sphere_kind());
}

#[test]
fn remove_subtree() {
    let mut obj = ProceduralObject::new(union_kind());
    let sub = obj.add_child(obj.root(), union_kind());
    let _leaf = obj.add_child(sub, sphere_kind());
    assert_eq!(obj.node_count(), 3);

    assert!(obj.remove(sub));
    assert_eq!(obj.node_count(), 1);
    assert_eq!(obj.get(obj.root()).unwrap().children.len(), 0);
}

#[test]
fn cannot_remove_root() {
    let mut obj = ProceduralObject::new(union_kind());
    assert!(!obj.remove(obj.root()));
}

#[test]
fn reparent_basic() {
    let mut obj = ProceduralObject::new(union_kind());
    let a = obj.add_child(obj.root(), union_kind());
    let b = obj.add_child(obj.root(), union_kind());
    let leaf = obj.add_child(a, sphere_kind());

    assert!(obj.reparent(leaf, b));
    assert_eq!(obj.get(a).unwrap().children.len(), 0);
    assert_eq!(obj.get(b).unwrap().children.len(), 1);
    assert_eq!(obj.get(leaf).unwrap().parent, Some(b));
}

#[test]
fn reparent_prevents_cycle() {
    let mut obj = ProceduralObject::new(union_kind());
    let a = obj.add_child(obj.root(), union_kind());
    let b = obj.add_child(a, union_kind());

    // Moving `a` under `b` would create a cycle.
    assert!(!obj.reparent(a, b));
}

#[test]
fn reparent_to_leaf_fails() {
    let mut obj = ProceduralObject::new(union_kind());
    let a = obj.add_child(obj.root(), union_kind());
    let leaf = obj.add_child(obj.root(), sphere_kind());

    assert!(!obj.reparent(a, leaf));
}

#[test]
fn version_propagates_on_add() {
    let mut obj = ProceduralObject::new(union_kind());
    let v_before = obj.get(obj.root()).unwrap().subtree_version;
    let _a = obj.add_child(obj.root(), sphere_kind());
    let v_after = obj.get(obj.root()).unwrap().subtree_version;
    assert!(v_after > v_before);
}

#[test]
fn set_transform_bumps_version() {
    let mut obj = ProceduralObject::new(union_kind());
    let a = obj.add_child(obj.root(), sphere_kind());
    let v_before = obj.get(obj.root()).unwrap().subtree_version;
    obj.set_transform(a, Affine3A::from_translation(glam::Vec3::X));
    let v_after = obj.get(obj.root()).unwrap().subtree_version;
    assert!(v_after > v_before);
}

#[test]
fn move_to_reorders_within_same_parent() {
    let mut obj = ProceduralObject::new(union_kind());
    let a = obj.add_child(obj.root(), sphere_kind());
    let b = obj.add_child(obj.root(), sphere_kind());
    let c = obj.add_child(obj.root(), sphere_kind());
    // Move c to front: [a, b, c] → [c, a, b]
    assert!(obj.move_to(c, obj.root(), 0));
    assert_eq!(
        obj.get(obj.root()).unwrap().children.as_slice(),
        &[c, a, b]
    );
}

#[test]
fn move_to_mid_position_same_parent() {
    let mut obj = ProceduralObject::new(union_kind());
    let a = obj.add_child(obj.root(), sphere_kind());
    let b = obj.add_child(obj.root(), sphere_kind());
    let c = obj.add_child(obj.root(), sphere_kind());
    // Move a between b and c: [a, b, c] → [b, a, c]
    assert!(obj.move_to(a, obj.root(), 1));
    assert_eq!(
        obj.get(obj.root()).unwrap().children.as_slice(),
        &[b, a, c]
    );
}

#[test]
fn move_to_across_parents() {
    let mut obj = ProceduralObject::new(union_kind());
    let p1 = obj.add_child(obj.root(), union_kind());
    let p2 = obj.add_child(obj.root(), union_kind());
    let leaf = obj.add_child(p1, sphere_kind());
    assert!(obj.move_to(leaf, p2, 0));
    assert_eq!(obj.get(p1).unwrap().children.len(), 0);
    assert_eq!(obj.get(p2).unwrap().children.as_slice(), &[leaf]);
    assert_eq!(obj.get(leaf).unwrap().parent, Some(p2));
}

#[test]
fn move_to_clamps_overflow_index() {
    let mut obj = ProceduralObject::new(union_kind());
    let _a = obj.add_child(obj.root(), sphere_kind());
    let b = obj.add_child(obj.root(), sphere_kind());
    // index=99 → clamp to end
    assert!(obj.move_to(b, obj.root(), 99));
    assert_eq!(obj.get(obj.root()).unwrap().children.last().copied(), Some(b));
}

#[test]
fn move_to_rejects_root() {
    let mut obj = ProceduralObject::new(union_kind());
    let a = obj.add_child(obj.root(), union_kind());
    assert!(!obj.move_to(obj.root(), a, 0));
}

#[test]
fn move_to_rejects_cycle() {
    let mut obj = ProceduralObject::new(union_kind());
    let a = obj.add_child(obj.root(), union_kind());
    let b = obj.add_child(a, union_kind());
    // Moving a into b would make b its own ancestor.
    assert!(!obj.move_to(a, b, 0));
}

#[test]
fn move_to_rejects_leaf_parent() {
    let mut obj = ProceduralObject::new(union_kind());
    let a = obj.add_child(obj.root(), union_kind());
    let leaf = obj.add_child(obj.root(), sphere_kind());
    assert!(!obj.move_to(a, leaf, 0));
}

#[test]
fn duplicate_leaf() {
    let mut obj = ProceduralObject::new(union_kind());
    let a = obj.add_child(obj.root(), sphere_kind());
    let dup = obj.duplicate(a).expect("leaf duplicates");
    assert_ne!(dup, a);
    let kids = obj.get(obj.root()).unwrap().children.clone();
    // Original followed by clone.
    assert_eq!(kids.as_slice(), &[a, dup]);
    assert_eq!(obj.get(dup).unwrap().parent, Some(obj.root()));
}

#[test]
fn duplicate_subtree_deep_copy() {
    let mut obj = ProceduralObject::new(union_kind());
    let sub = obj.add_child(obj.root(), union_kind());
    let leaf_a = obj.add_child(sub, sphere_kind());
    let leaf_b = obj.add_child(sub, sphere_kind());

    let dup = obj.duplicate(sub).expect("subtree duplicates");
    // Dup is a new node.
    assert_ne!(dup, sub);
    // Root now has two combinator children.
    assert_eq!(obj.get(obj.root()).unwrap().children.as_slice(), &[sub, dup]);
    // Dup has two children that are NEW ids (not the original leaves).
    let dup_kids = obj.get(dup).unwrap().children.clone();
    assert_eq!(dup_kids.len(), 2);
    for &c in dup_kids.iter() {
        assert_ne!(c, leaf_a);
        assert_ne!(c, leaf_b);
        assert_eq!(obj.get(c).unwrap().parent, Some(dup));
    }
}

#[test]
fn duplicate_rejects_root() {
    let mut obj = ProceduralObject::new(union_kind());
    assert!(obj.duplicate(obj.root()).is_none());
}

/// Effects were single-child in the early prototype and
/// evicted extras to the grandparent. That capability was
/// dropped when effects went multi-child with an implicit
/// Union at flatten time — drop N shapes under one NoiseDisplace
/// and they all stay, combined into one logical sample before
/// the warp applies.
#[test]
fn effects_accept_multiple_children() {
    // Effects flipped from `max_children = Some(1)` to `None` so
    // users can drop several shapes under one NoiseDisplace and
    // have them implicitly unioned before the warp applies. Both
    // children must stick around (no eviction).
    use crate::node_kind::NoiseDisplaceParams;
    let mut obj = ProceduralObject::new(union_kind());
    let effect = obj.add_child(
        obj.root(),
        NodeKind::NoiseDisplace(NoiseDisplaceParams::default()),
    );
    let first = obj.add_child(effect, sphere_kind());
    let second = obj.add_child(effect, sphere_kind());
    let kids = obj.get(effect).unwrap().children.clone();
    assert_eq!(kids.as_slice(), &[first, second]);
}

/// Over-cap at the root has nowhere to evict to — behavior there is
/// "leave the extra alone" so evaluator's ignore-anything-past-[0]
/// rule still holds and nothing panics.
#[test]
fn over_cap_at_root_is_no_op() {
    use crate::node_kind::NoiseDisplaceParams;
    // Start with a NoiseDisplace directly as root (unusual but legal).
    let mut obj =
        ProceduralObject::new(NodeKind::NoiseDisplace(NoiseDisplaceParams::default()));
    let a = obj.add_child(obj.root(), sphere_kind());
    let b = obj.add_child(obj.root(), sphere_kind());

    // Both hangs off the root because there's no grandparent to evict to.
    let kids = obj.get(obj.root()).unwrap().children.clone();
    assert_eq!(kids.as_slice(), &[a, b]);
}

#[test]
fn iter_ids_skips_removed() {
    let mut obj = ProceduralObject::new(union_kind());
    let a = obj.add_child(obj.root(), sphere_kind());
    let _b = obj.add_child(obj.root(), sphere_kind());
    obj.remove(a);
    let ids: Vec<_> = obj.iter_ids().collect();
    assert_eq!(ids.len(), 2); // root + b
}
