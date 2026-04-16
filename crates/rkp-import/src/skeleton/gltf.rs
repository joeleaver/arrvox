//! glTF skeleton + skinning + animation extraction.
//!
//! Uses the first skin in the document as the source of truth for
//! the bone list. Hierarchy is recovered by walking the scene-graph
//! parent chain until we hit another joint node. Inverse bind
//! matrices fall back to identity if the file omits them.
//!
//! Animation extraction collapses the three per-bone channel types
//! (translation / rotation / scale) at matching timestamps into
//! single [`Keyframe`]s with unspecified channels defaulting to
//! bind-pose values.

use std::collections::HashMap;

use glam::{Mat4, Quat, Vec3};

use rkf_animation::clip::{AnimationClip, BoneChannel, Keyframe};
use rkf_animation::skeleton::{Bone, Skeleton};

use super::{SkeletonExtraction, VertexSkinning};

/// Extract skeleton + skinning + clips from a glTF / GLB file.
pub fn extract(path: &str) -> Result<Option<SkeletonExtraction>, String> {
    let (document, buffers, _images) =
        gltf::import(path).map_err(|e| format!("Failed to load glTF '{path}': {e}"))?;

    let Some(skin) = document.skins().next() else {
        return Ok(None);
    };

    let joint_nodes: Vec<_> = skin.joints().collect();
    let joint_count = joint_nodes.len();
    if joint_count == 0 {
        return Ok(None);
    }

    let mut node_to_bone: HashMap<usize, usize> = HashMap::new();
    for (bone_idx, node) in joint_nodes.iter().enumerate() {
        node_to_bone.insert(node.index(), bone_idx);
    }

    let reader = skin.reader(|buf| Some(&buffers[buf.index()]));
    let inverse_binds: Vec<Mat4> = reader
        .read_inverse_bind_matrices()
        .map(|it| it.map(|m| Mat4::from_cols_array_2d(&m)).collect())
        .unwrap_or_else(|| vec![Mat4::IDENTITY; joint_count]);

    let mut node_parent: HashMap<usize, usize> = HashMap::new();
    for node in document.nodes() {
        for child in node.children() {
            node_parent.insert(child.index(), node.index());
        }
    }

    let mut bones = Vec::with_capacity(joint_count);
    let mut hierarchy = Vec::with_capacity(joint_count);

    for (bone_idx, node) in joint_nodes.iter().enumerate() {
        let (t, r, s) = node.transform().decomposed();
        let bind_transform =
            Mat4::from_scale_rotation_translation(Vec3::from(s), Quat::from_array(r), Vec3::from(t));

        bones.push(Bone {
            name: node.name().unwrap_or("unnamed").to_string(),
            bind_transform,
            inverse_bind: inverse_binds
                .get(bone_idx)
                .copied()
                .unwrap_or(Mat4::IDENTITY),
        });

        hierarchy.push(nearest_joint_ancestor(node.index(), &node_parent, &node_to_bone));
    }

    let skeleton = Skeleton::new(bones, hierarchy)
        .map_err(|e| format!("Failed to construct skeleton from glTF joints: {e}"))?;

    let skinning = extract_skinning(&document, &buffers);
    let clips = extract_animations(&document, &buffers, &node_to_bone);

    Ok(Some(SkeletonExtraction { skeleton, skinning, clips }))
}

fn nearest_joint_ancestor(
    node: usize,
    parents: &HashMap<usize, usize>,
    node_to_bone: &HashMap<usize, usize>,
) -> i32 {
    let mut current = node;
    while let Some(&parent_node) = parents.get(&current) {
        if let Some(&bi) = node_to_bone.get(&parent_node) {
            return bi as i32;
        }
        current = parent_node;
    }
    -1
}

fn extract_skinning(
    document: &gltf::Document,
    buffers: &[gltf::buffer::Data],
) -> VertexSkinning {
    let mut joints = Vec::new();
    let mut weights = Vec::new();

    for mesh in document.meshes() {
        for primitive in mesh.primitives() {
            let reader = primitive.reader(|buf| Some(&buffers[buf.index()]));

            if let Some(joints_reader) = reader.read_joints(0) {
                for j in joints_reader.into_u16() {
                    joints.push([j[0] as i32, j[1] as i32, j[2] as i32, j[3] as i32]);
                }
            }
            if let Some(weights_reader) = reader.read_weights(0) {
                for w in weights_reader.into_f32() {
                    weights.push(w);
                }
            }
        }
    }

    // Pad to common length if one channel is shorter (shouldn't happen in valid glTF).
    let len = joints.len().max(weights.len());
    joints.resize(len, [-1, -1, -1, -1]);
    weights.resize(len, [0.0; 4]);

    VertexSkinning { joints, weights }
}

/// One row of the per-bone timeline accumulator: (time, pos?, rot?, scale?).
type ChannelEntry = (f32, Option<Vec3>, Option<Quat>, Option<Vec3>);

fn extract_animations(
    document: &gltf::Document,
    buffers: &[gltf::buffer::Data],
    node_to_bone: &HashMap<usize, usize>,
) -> Vec<AnimationClip> {
    let mut clips = Vec::new();

    for anim in document.animations() {
        let name = anim.name().unwrap_or("unnamed").to_string();

        // Per-bone accumulator: timestamp → (pos, rot, scale) (each Option).
        let mut bone_channels: HashMap<u32, Vec<ChannelEntry>> = HashMap::new();

        for channel in anim.channels() {
            let target_node = channel.target().node().index();
            let Some(&bone_idx) = node_to_bone.get(&target_node) else {
                continue;
            };
            let bone_idx = bone_idx as u32;

            let reader = channel.reader(|buf| Some(&buffers[buf.index()]));
            let timestamps: Vec<f32> = reader
                .read_inputs()
                .map(|it| it.collect())
                .unwrap_or_default();

            let entries = bone_channels.entry(bone_idx).or_default();

            match reader.read_outputs() {
                Some(gltf::animation::util::ReadOutputs::Translations(values)) => {
                    merge_channel(entries, &timestamps, values.map(Vec3::from), ChannelKind::T);
                }
                Some(gltf::animation::util::ReadOutputs::Rotations(values)) => {
                    merge_channel(
                        entries,
                        &timestamps,
                        values.into_f32().map(Quat::from_array),
                        ChannelKind::R,
                    );
                }
                Some(gltf::animation::util::ReadOutputs::Scales(values)) => {
                    merge_channel(entries, &timestamps, values.map(Vec3::from), ChannelKind::S);
                }
                _ => {}
            }
        }

        let mut channels = Vec::new();
        for (bone_index, mut entries) in bone_channels {
            entries.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
            let keyframes: Vec<Keyframe> = entries
                .iter()
                .map(|(time, pos, rot, scale)| Keyframe {
                    time: *time,
                    position: pos.unwrap_or(Vec3::ZERO),
                    rotation: rot.unwrap_or(Quat::IDENTITY),
                    scale: scale.unwrap_or(Vec3::ONE),
                })
                .collect();
            channels.push(BoneChannel { bone_index, keyframes });
        }

        let duration = channels
            .iter()
            .flat_map(|c| c.keyframes.last())
            .map(|kf| kf.time)
            .fold(0.0f32, f32::max);

        clips.push(AnimationClip::new(name, duration, channels));
    }

    clips
}

enum ChannelKind {
    T,
    R,
    S,
}

// Consumers: Vec3 for T/S, Quat for R — wrap with an enum so one
// helper handles all three. Takes any iterator over the typed value
// and deposits it into the correct Option slot at the right time.
fn merge_channel<I, V>(
    entries: &mut Vec<ChannelEntry>,
    timestamps: &[f32],
    values: I,
    kind: ChannelKind,
) where
    I: IntoIterator<Item = V>,
    V: Into<ChannelValue>,
{
    for (t, v) in timestamps.iter().zip(values) {
        let v = v.into();
        let entry = match entries.iter_mut().find(|e| (e.0 - t).abs() < 1e-6) {
            Some(e) => e,
            None => {
                entries.push((*t, None, None, None));
                // Vec is non-empty right after `push`, so the
                // `expect` is unreachable; kept so that a future
                // refactor can't silently introduce a panic path.
                entries.last_mut().expect("entry just pushed")
            }
        };
        match (v, &kind) {
            (ChannelValue::Vec(p), ChannelKind::T) => entry.1 = Some(p),
            (ChannelValue::Quat(q), ChannelKind::R) => entry.2 = Some(q),
            (ChannelValue::Vec(s), ChannelKind::S) => entry.3 = Some(s),
            _ => {}
        }
    }
}

enum ChannelValue {
    Vec(Vec3),
    Quat(Quat),
}

impl From<Vec3> for ChannelValue {
    fn from(v: Vec3) -> Self { Self::Vec(v) }
}
impl From<Quat> for ChannelValue {
    fn from(q: Quat) -> Self { Self::Quat(q) }
}
