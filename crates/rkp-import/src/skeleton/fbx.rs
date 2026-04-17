//! FBX skeleton + skinning + animation extraction (ufbx).
//!
//! ## Multi-skin handling
//!
//! An FBX scene can have more than one skin deformer — for example,
//! Mixamo characters ship a visible `Beta_Surface` mesh plus a hidden
//! `Beta_Joints` joint-visualisation mesh, each with its own
//! `ufbx::SkinDeformer`. The two skins often share most bones but
//! differ by a cluster or two (Mixamo's joint mesh adds
//! `HeadTop_End`), which shifts every cluster index from that point on.
//!
//! The extractor therefore treats cluster indices as *per-skin* and
//! translates them to a *unified* bone index via the cluster's
//! `bone_node.typed_id`. Using positional cluster indices as bone
//! indices — the original implementation — silently mis-skinned
//! everything past the first divergent cluster (the visible symptom
//! was "upper leg follows lower leg bone" because a Mixamo
//! `RightUpLeg` cluster index in the joint mesh resolved to
//! `RightLeg` in the surface-mesh bone table).
//!
//! Bones that only appear in later skins are appended to the unified
//! table so their weights still have a home. `inverse_bind` for each
//! bone comes from the first cluster we see targeting that bone —
//! this is fine as long as every mesh shares the same post-
//! `ModifyGeometry` space, which is true for Mixamo and any rig whose
//! meshes sit under a common armature with identity locals.

use std::collections::HashMap;

use glam::{Mat4, Quat, Vec3};

use rkp_animation::clip::{AnimationClip, BoneChannel, Keyframe};
use rkp_animation::skeleton::{Bone, Skeleton};

use crate::mesh::fbx::fbx_load_opts;

use super::{SkeletonExtraction, VertexSkinning};

/// Extract skeleton + skinning + clips from an FBX file.
pub fn extract(path: &str) -> Result<Option<SkeletonExtraction>, String> {
    let scene = ufbx::load_file(path, fbx_load_opts())
        .map_err(|e| format!("Failed to load FBX '{path}': {}", e.description))?;

    if scene.skin_deformers.is_empty() {
        return Ok(None);
    }

    // Unified bone table across all skin deformers. We add each unique
    // `bone_node.typed_id` once, in the order first seen — this makes
    // the first (usually visible) mesh's skin define the canonical bone
    // ordering, with any extras from later skins appended.
    let mut node_to_bone: HashMap<u32, usize> = HashMap::new();
    let mut bone_nodes: Vec<&ufbx::Node> = Vec::new();
    // `inverse_bind` per bone — populated from the first cluster we see
    // targeting that bone. Same bone across skins should have the same
    // `geometry_to_bone` under ModifyGeometry for meshes sharing a common
    // space; if they ever diverge, a per-mesh fix-up would be needed,
    // but that's not the case for Mixamo or any standard armature rig.
    let mut inverse_binds: Vec<Mat4> = Vec::new();

    for skin in scene.skin_deformers.iter() {
        for cluster in skin.clusters.iter() {
            let Some(ref bone_node) = cluster.bone_node else {
                eprintln!("[rkp-import] warn: FBX skin cluster has no bone node, skipping");
                continue;
            };
            let id = bone_node.element.typed_id;
            if node_to_bone.contains_key(&id) {
                continue;
            }
            node_to_bone.insert(id, bone_nodes.len());
            bone_nodes.push(bone_node);
            inverse_binds.push(ufbx_matrix_to_mat4(&cluster.geometry_to_bone));
        }
    }
    if bone_nodes.is_empty() {
        return Ok(None);
    }

    let bone_count = bone_nodes.len();
    let mut bones = Vec::with_capacity(bone_count);
    let mut hierarchy = Vec::with_capacity(bone_count);

    for (bone_idx, bone_node) in bone_nodes.iter().enumerate() {
        let lt = &bone_node.local_transform;
        let bind_transform = Mat4::from_scale_rotation_translation(
            Vec3::new(lt.scale.x as f32, lt.scale.y as f32, lt.scale.z as f32),
            Quat::from_xyzw(
                lt.rotation.x as f32,
                lt.rotation.y as f32,
                lt.rotation.z as f32,
                lt.rotation.w as f32,
            ),
            Vec3::new(
                lt.translation.x as f32,
                lt.translation.y as f32,
                lt.translation.z as f32,
            ),
        );

        bones.push(Bone {
            name: bone_node.element.name.to_string(),
            bind_transform,
            inverse_bind: inverse_binds[bone_idx],
        });

        hierarchy.push(find_parent_bone(bone_node, &node_to_bone));
    }

    let skeleton = Skeleton::new(bones, hierarchy)
        .map_err(|e| format!("Failed to construct skeleton from FBX joints: {e}"))?;

    let skinning = extract_skinning(&scene, &node_to_bone, bone_count);
    let clips = extract_animations(&scene, &node_to_bone);

    Ok(Some(SkeletonExtraction { skeleton, skinning, clips }))
}

fn ufbx_matrix_to_mat4(m: &ufbx::Matrix) -> Mat4 {
    Mat4::from_cols_array(&[
        m.m00 as f32, m.m10 as f32, m.m20 as f32, 0.0,
        m.m01 as f32, m.m11 as f32, m.m21 as f32, 0.0,
        m.m02 as f32, m.m12 as f32, m.m22 as f32, 0.0,
        m.m03 as f32, m.m13 as f32, m.m23 as f32, 1.0,
    ])
}

fn find_parent_bone(node: &ufbx::Node, node_to_bone: &HashMap<u32, usize>) -> i32 {
    let mut current = node.parent.as_ref();
    while let Some(parent) = current {
        if let Some(&bi) = node_to_bone.get(&parent.element.typed_id) {
            return bi as i32;
        }
        current = parent.parent.as_ref();
    }
    -1
}

/// Per-corner skinning extraction.
///
/// Iterates the scene meshes in the same order as `mesh::fbx::load`
/// and emits one skin-weight record per triangle corner. Uses
/// [`CornerIter`] to encapsulate the iteration so the two paths
/// stay in lock-step.
///
/// For each mesh we build a local `cluster_to_bone` table that maps
/// *this mesh's skin*'s cluster indices into the unified bone table
/// shared across all skins. That table is the whole reason the FBX
/// path survives files with more than one skin deformer whose cluster
/// orderings disagree (see module-level docs).
fn extract_skinning(
    scene: &ufbx::Scene,
    node_to_bone: &HashMap<u32, usize>,
    bone_count: usize,
) -> VertexSkinning {
    let max_face_tris = scene
        .meshes
        .iter()
        .map(|m| m.max_face_triangles)
        .max()
        .unwrap_or(1);
    let mut tri_indices = vec![0u32; max_face_tris * 3];

    let mut joints = Vec::new();
    let mut weights = Vec::new();

    for mesh in &scene.meshes {
        let mesh_skin = mesh.skin_deformers.iter().next();
        let cluster_to_bone: Vec<i32> = mesh_skin
            .map(|s| {
                s.clusters
                    .iter()
                    .map(|c| {
                        c.bone_node
                            .as_ref()
                            .and_then(|n| node_to_bone.get(&n.element.typed_id).copied())
                            .map(|i| i as i32)
                            .unwrap_or(-1)
                    })
                    .collect()
            })
            .unwrap_or_default();

        for corner in CornerIter::new(mesh, &mut tri_indices) {
            let (j, w) = if let Some(skin) = mesh_skin {
                get_vertex_skin_weights(skin, corner.vertex_idx, &cluster_to_bone, bone_count, mesh)
            } else {
                ([-1; 4], [0.0; 4])
            };
            joints.push(j);
            weights.push(w);
        }
    }

    VertexSkinning { joints, weights }
}

/// Top-4 skin weights for a vertex, with validation.
///
/// Returns zeros on invalid input *and* emits a warning — never
/// silent. `cluster_to_bone[cluster_index]` resolves each cluster in
/// *this skin* to the unified bone index; a `-1` entry means the
/// cluster has no bone_node (malformed skin) and the influence is
/// dropped. Cluster indices past the map's length are treated the same
/// way — ufbx occasionally emits garbage cluster_index values on
/// broken files.
fn get_vertex_skin_weights(
    skin: &ufbx::SkinDeformer,
    vertex_idx: usize,
    cluster_to_bone: &[i32],
    bone_count: usize,
    mesh: &ufbx::Mesh,
) -> ([i32; 4], [f32; 4]) {
    if vertex_idx >= skin.vertices.len() {
        eprintln!(
            "[rkp-import] warn: FBX vertex_idx {vertex_idx} out of skin.vertices ({}) for mesh '{}'",
            skin.vertices.len(), mesh.element.name,
        );
        return ([-1; 4], [0.0; 4]);
    }

    let sv = &skin.vertices[vertex_idx];
    let begin = sv.weight_begin as usize;
    let n = sv.num_weights as usize;

    if begin + n > skin.weights.len() {
        eprintln!(
            "[rkp-import] warn: FBX vertex {vertex_idx} weight range [{begin}, {}) exceeds skin.weights ({})",
            begin + n, skin.weights.len(),
        );
        return ([-1; 4], [0.0; 4]);
    }

    // Single pass: collect validated influences (cluster resolves to a
    // real bone, finite weight), then sort by magnitude and take top 4.
    let mut all: Vec<(i32, f32)> = Vec::with_capacity(n);
    let mut dropped_clusters = 0u32;
    let mut nan_weights = 0u32;
    for i in 0..n {
        let sw = &skin.weights[begin + i];
        let cluster = sw.cluster_index as usize;
        let w = sw.weight as f32;
        if !w.is_finite() {
            nan_weights += 1;
            continue;
        }
        let bone = cluster_to_bone.get(cluster).copied().unwrap_or(-1);
        if bone < 0 || (bone as usize) >= bone_count {
            dropped_clusters += 1;
            continue;
        }
        if w <= 0.0 {
            continue;
        }
        all.push((bone, w));
    }
    if dropped_clusters > 0 {
        eprintln!(
            "[rkp-import] warn: FBX vertex {vertex_idx} had {dropped_clusters} cluster indices with no bone (bone_count = {bone_count})"
        );
    }
    if nan_weights > 0 {
        eprintln!(
            "[rkp-import] warn: FBX vertex {vertex_idx} had {nan_weights} non-finite weights"
        );
    }

    if all.is_empty() {
        return ([-1; 4], [0.0; 4]);
    }

    // Sort by weight descending, NaN-safe (we already filtered NaNs
    // above, but keep `total_cmp` as belt-and-braces in case a future
    // change reintroduces them).
    all.sort_by(|a, b| b.1.total_cmp(&a.1));
    all.truncate(4);

    let mut joints = [-1i32; 4];
    let mut weights = [0.0f32; 4];
    for (i, (j, w)) in all.iter().enumerate() {
        joints[i] = *j;
        weights[i] = *w;
    }

    let sum: f32 = weights.iter().sum();
    if sum > 1e-6 {
        for w in &mut weights {
            *w /= sum;
        }
    }

    (joints, weights)
}

/// Encapsulates the per-corner iteration shared with [`crate::mesh::fbx`].
/// Keeping the iteration rule in one place makes the invariant
/// "mesh corners and skin weights line up one-to-one" explicit
/// rather than a fragile by-convention match between two files.
struct CornerIter<'a> {
    mesh: &'a ufbx::Mesh,
    tri_indices: &'a mut [u32],
    face_idx: usize,
    tri_in_face: usize,
    num_tris: u32,
    corner_in_tri: usize,
}

struct Corner {
    vertex_idx: usize,
}

impl<'a> CornerIter<'a> {
    fn new(mesh: &'a ufbx::Mesh, tri_indices: &'a mut [u32]) -> Self {
        Self {
            mesh,
            tri_indices,
            face_idx: 0,
            tri_in_face: 0,
            num_tris: 0,
            corner_in_tri: 0,
        }
    }
}

impl<'a> Iterator for CornerIter<'a> {
    type Item = Corner;

    fn next(&mut self) -> Option<Corner> {
        loop {
            if self.face_idx >= self.mesh.faces.len() {
                return None;
            }
            // Triangulate the current face on first touch.
            if self.tri_in_face == 0 && self.corner_in_tri == 0 {
                let face = self.mesh.faces[self.face_idx];
                self.num_tris =
                    ufbx::triangulate_face(self.tri_indices, self.mesh, face);
                if self.num_tris == 0 {
                    self.face_idx += 1;
                    continue;
                }
            }
            let corner_local = self.tri_in_face * 3 + self.corner_in_tri;
            let corner_idx = self.tri_indices[corner_local] as usize;
            let vertex_idx = self.mesh.vertex_indices[corner_idx] as usize;

            self.corner_in_tri += 1;
            if self.corner_in_tri == 3 {
                self.corner_in_tri = 0;
                self.tri_in_face += 1;
                if self.tri_in_face as u32 >= self.num_tris {
                    self.tri_in_face = 0;
                    self.face_idx += 1;
                }
            }
            return Some(Corner { vertex_idx });
        }
    }
}

fn extract_animations(
    scene: &ufbx::Scene,
    node_to_bone: &HashMap<u32, usize>,
) -> Vec<AnimationClip> {
    let mut clips = Vec::new();

    for stack in &scene.anim_stacks {
        let raw_name = stack.element.name.to_string();
        let name = if raw_name.is_empty() { "unnamed".to_string() } else { raw_name };

        let bake_opts = ufbx::BakeOpts {
            resample_rate: 30.0,
            key_reduction_enabled: true,
            key_reduction_threshold: 0.001,
            ..Default::default()
        };

        let baked = match ufbx::bake_anim(scene, &stack.anim, bake_opts) {
            Ok(b) => b,
            Err(e) => {
                eprintln!(
                    "[rkp-import] warn: failed to bake FBX animation '{name}': {}",
                    e.description
                );
                continue;
            }
        };

        let mut channels = Vec::new();

        for baked_node in &baked.nodes {
            let Some(&bi) = node_to_bone.get(&baked_node.typed_id) else {
                continue;
            };
            let bone_index = bi as u32;

            if baked_node.constant_translation
                && baked_node.constant_rotation
                && baked_node.constant_scale
            {
                continue;
            }

            let mut times: Vec<f64> = Vec::new();
            for k in baked_node.translation_keys.iter() { times.push(k.time); }
            for k in baked_node.rotation_keys.iter()    { times.push(k.time); }
            for k in baked_node.scale_keys.iter()       { times.push(k.time); }
            times.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            times.dedup_by(|a, b| (*a - *b).abs() < 1e-6);

            let keyframes: Vec<Keyframe> = times
                .iter()
                .map(|&t| {
                    let pos = ufbx::evaluate_baked_vec3(&baked_node.translation_keys, t);
                    let rot = ufbx::evaluate_baked_quat(&baked_node.rotation_keys, t);
                    let scl = ufbx::evaluate_baked_vec3(&baked_node.scale_keys, t);
                    Keyframe {
                        time: t as f32,
                        position: Vec3::new(pos.x as f32, pos.y as f32, pos.z as f32),
                        rotation: Quat::from_xyzw(
                            rot.x as f32, rot.y as f32, rot.z as f32, rot.w as f32,
                        ),
                        scale: Vec3::new(scl.x as f32, scl.y as f32, scl.z as f32),
                    }
                })
                .collect();

            if !keyframes.is_empty() {
                channels.push(BoneChannel { bone_index, keyframes });
            }
        }

        clips.push(AnimationClip::new(name, baked.playback_duration as f32, channels));
    }

    clips
}
