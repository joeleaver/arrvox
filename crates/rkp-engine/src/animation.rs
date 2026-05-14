//! Skeletal-animation engine integration.
//!
//! Owns two responsibilities:
//!
//! 1. [`AnimationAssetCache`] — path-keyed cache of loaded `.rkskel`
//!    skeleton assets, shared across entity instances. Loading is
//!    synchronous (RON + small file); happens on the engine thread
//!    during `.rkp` load.
//!
//! 2. [`tick`] — per-frame system. For each `(Skeleton, AnimationPlayer)`
//!    tuple in the world it advances the player's time and re-evaluates
//!    the skinning palette into `Skeleton.current_pose` in place.
//!
//! No GPU upload happens here — that's scene-sync's job.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use glam::Mat4;
use rkp_animation::player::LoopMode;
use rkp_animation::skeleton_asset::{load_rkskel, SkeletonAsset};

use crate::components::{AnimationPlayer, Skeleton};

/// Path-keyed cache of loaded [`SkeletonAsset`]s.
///
/// `.rkskel` files are small RON; instances of the same asset share an
/// `Arc` so switching between character instances costs nothing beyond
/// a pointer bump.
#[derive(Default)]
pub struct AnimationAssetCache {
    assets: HashMap<PathBuf, Arc<SkeletonAsset>>,
}

impl AnimationAssetCache {
    pub fn new() -> Self { Self::default() }

    /// Load the `.rkskel` at `path` (if not already cached). Returns the
    /// cached `Arc` on success. Errors propagate from `load_rkskel`.
    pub fn get_or_load(&mut self, path: &Path) -> Result<Arc<SkeletonAsset>, String> {
        if let Some(existing) = self.assets.get(path) {
            return Ok(existing.clone());
        }
        let asset = load_rkskel(path).map_err(|e| format!("load_rkskel {}: {e}", path.display()))?;
        let arc = Arc::new(asset);
        self.assets.insert(path.to_path_buf(), arc.clone());
        Ok(arc)
    }

    /// Drop a cached entry. The underlying `Arc` survives until all
    /// entity references release it.
    pub fn evict(&mut self, path: &Path) {
        self.assets.remove(path);
    }
}

/// Build a [`Skeleton`] component for an entity that's just loaded `path.rkskel`.
///
/// `grid_offset` translates mesh-frame (centered at 0) into grid-frame
/// (octree corner at 0). Folded into `normalize` so `current_pose` ends
/// up operating on grid-frame positions — matches the frame that
/// `rest_bone_aabbs` and the scatter shader's `rest_pos` already use.
/// For non-skinned or procedural entities with no spatial data, pass
/// `Vec3::ZERO` and the transform falls back to plain mesh-frame.
pub fn skeleton_component(
    asset: Arc<SkeletonAsset>,
    path: PathBuf,
    grid_offset: glam::Vec3,
) -> Skeleton {
    let bone_count = asset.skeleton.bones.len();
    let normalize_mesh = Skeleton::compute_normalization(&asset);
    let normalize = Mat4::from_translation(grid_offset) * normalize_mesh;
    let normalize_inverse = normalize.inverse();
    // Bind-world origins come out of the glTF-space hierarchy walk;
    // pre-multiply by `normalize` so they land in the same grid frame
    // as the pose. Gizmo rendering undoes the grid offset before
    // handing positions to the world transform.
    let bind_world_origins: Vec<_> = Skeleton::compute_bind_world_origins(&asset)
        .into_iter()
        .map(|p| normalize.transform_point3(p))
        .collect();
    Skeleton {
        asset,
        path,
        current_pose: vec![Mat4::IDENTITY; bone_count],
        inverse_pose: vec![Mat4::IDENTITY; bone_count],
        bind_world_origins,
        normalize,
        normalize_inverse,
        grid_offset,
    }
}

/// Default `AnimationPlayer` for a freshly-attached skeleton — plays the
/// first bundled clip on Loop, paused. Leaves `clip_name` empty when the
/// asset has no clips (entity still gets a player so the UI can attach
/// later imports / procedural clips).
pub fn default_player(asset: &SkeletonAsset) -> AnimationPlayer {
    let clip_name = asset.clips.first().map(|c| c.name.clone()).unwrap_or_default();
    AnimationPlayer {
        clip_name,
        time: 0.0,
        speed: 1.0,
        playing: false,
        loop_mode: LoopMode::Loop,
        forward: true,
    }
}

/// Advance time on one [`AnimationPlayer`] against a clip of the given
/// duration. Duplicated from `rkp_animation::AnimationPlayer::advance`
/// because our ECS player carries `clip_name` rather than owning a
/// clip; porting the same semantics (Once clamp, Loop wrap, PingPong
/// fold) verbatim keeps behaviour identical for tests and muscle memory.
fn advance_player(p: &mut AnimationPlayer, duration: f32, dt: f32) {
    if !p.playing || duration <= 0.0 {
        return;
    }
    let delta = dt * p.speed;
    match p.loop_mode {
        LoopMode::Once => {
            p.time = (p.time + delta).clamp(0.0, duration);
        }
        LoopMode::Loop => {
            p.time = (p.time + delta).rem_euclid(duration);
        }
        LoopMode::PingPong => {
            if p.forward {
                p.time += delta;
                if p.time >= duration {
                    p.time = 2.0 * duration - p.time;
                    p.forward = false;
                }
            } else {
                p.time -= delta;
                if p.time <= 0.0 {
                    p.time = -p.time;
                    p.forward = true;
                }
            }
            p.time = p.time.clamp(0.0, duration);
        }
    }
}

/// Per-frame animation tick.
///
/// For every `(Skeleton, AnimationPlayer)` entity: resolves the player's
/// active clip against the skeleton asset, advances time, and re-evaluates
/// the skinning matrices into `Skeleton.current_pose` in place. When the
/// player has no active clip (empty `clip_name` or the name doesn't match
/// any clip in the asset) the pose is left as the bind-pose identity.
///
/// Returns `true` if any pose changed — callers can use that to flag
/// `gpu_objects_dirty` when a frame is otherwise static. Always-true
/// when a playing animation advanced; false when every player is paused
/// or missing a clip.
pub fn tick(world: &mut hecs::World, dt: f32) -> bool {
    let mut any_changed = false;
    // Borrowing `&mut Skeleton` and `&mut AnimationPlayer` from hecs in
    // one query is fine — different components, no aliasing.
    for (_entity, (skel, player)) in world.query_mut::<(&mut Skeleton, &mut AnimationPlayer)>() {
        // Resolve the active clip by name. Missing clip → leave pose as
        // identity; the render path still draws the mesh rigidly.
        let clip = if player.clip_name.is_empty() {
            None
        } else {
            skel.asset.clips.iter().find(|c| c.name == player.clip_name)
        };

        let bone_count = skel.asset.skeleton.bones.len();
        let Some(clip) = clip else {
            // No active clip — ensure pose is identity-sized.
            if skel.current_pose.len() != bone_count {
                skel.current_pose = vec![Mat4::IDENTITY; bone_count];
            }
            if skel.inverse_pose.len() != bone_count {
                skel.inverse_pose = vec![Mat4::IDENTITY; bone_count];
            }
            continue;
        };

        // Paused player: pose was set the last time we advanced time
        // (or by the initial component build). Re-evaluating produces
        // bit-identical matrices, so it's pure waste — and worse, the
        // unconditional `any_changed = true` below makes the caller
        // call `gpu_objects_dirty.mark_all()` every tick, defeating
        // PERF_DEBT.md's per-entity dirty work (B1/C2/C2-narrow). A
        // paused player should be invisible to the dirty system.
        if !player.playing {
            continue;
        }

        advance_player(player, clip.duration, dt);
        let mut matrices = skel.asset.skeleton.evaluate(clip, player.time);

        // Skeleton::evaluate produces matrices in glTF space; voxel
        // data lives in rkp-import's normalised frame. Conjugate by
        // the normalization transform so the palette operates in
        // voxel space (same frame as the scatter's rest positions
        // and the march's inverse-skin samples).
        //
        // `pose_voxel = N · pose_glTF · N⁻¹`
        let n = skel.normalize;
        let n_inv = skel.normalize_inverse;
        for m in matrices.iter_mut() {
            *m = n * *m * n_inv;
        }

        // Evaluate returns exactly `bones.len()` matrices; overwrite in place.
        if skel.current_pose.len() == matrices.len() {
            skel.current_pose.copy_from_slice(&matrices);
        } else {
            skel.current_pose = matrices;
        }
        // Inverse skinning palette. A fresh `Mat4::inverse` per bone per
        // frame is ~200 ns each — negligible at typical bone counts
        // (50-200 bones, ~10-40 μs total). Packed after the forward
        // matrices in the GPU bone buffer so the shader can index both
        // from `bone_buffer_offset`.
        if skel.inverse_pose.len() != bone_count {
            skel.inverse_pose.resize(bone_count, Mat4::IDENTITY);
        }
        for i in 0..bone_count {
            skel.inverse_pose[i] = skel.current_pose[i].inverse();
        }
        any_changed = true;
    }
    any_changed
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::{Quat, Vec3};
    use rkp_animation::clip::{AnimationClip, BoneChannel, Keyframe};
    use rkp_animation::skeleton::{Bone, Skeleton as SkelData};
    use rkp_animation::skeleton_asset::SkeletonAsset;

    fn tiny_asset() -> Arc<SkeletonAsset> {
        let bones = vec![
            Bone {
                name: "root".into(),
                bind_transform: Mat4::IDENTITY,
                inverse_bind: Mat4::IDENTITY,
            },
        ];
        let skeleton = SkelData::new(bones, vec![-1]).expect("valid skeleton");
        let clip = AnimationClip::new(
            "wave".into(),
            1.0,
            vec![BoneChannel {
                bone_index: 0,
                keyframes: vec![
                    Keyframe { time: 0.0, position: Vec3::ZERO, rotation: Quat::IDENTITY, scale: Vec3::ONE },
                    Keyframe { time: 1.0, position: Vec3::new(10.0, 0.0, 0.0), rotation: Quat::IDENTITY, scale: Vec3::ONE },
                ],
            }],
        );
        Arc::new(SkeletonAsset::new(skeleton, vec![clip]))
    }

    #[test]
    fn tick_updates_current_pose_for_active_clip() {
        let asset = tiny_asset();
        let mut world = hecs::World::new();
        let entity = world.spawn((
            skeleton_component(asset.clone(), PathBuf::from("test.rkskel"), glam::Vec3::ZERO),
            AnimationPlayer {
                clip_name: "wave".into(),
                time: 0.0,
                speed: 1.0,
                playing: true,
                loop_mode: LoopMode::Loop,
                forward: true,
            },
        ));

        let changed = tick(&mut world, 0.5);
        assert!(changed, "a playing animation should report change");

        let skel = world.get::<&Skeleton>(entity).unwrap();
        assert_eq!(skel.current_pose.len(), 1);
        let m = skel.current_pose[0];
        // Root has identity inverse_bind, so pose ≈ world ≈ translation(5, 0, 0).
        let translation = m.to_cols_array()[12];
        assert!((translation - 5.0).abs() < 1e-4, "expected x≈5, got {translation}");
        // Every value must be finite.
        for v in m.to_cols_array() {
            assert!(v.is_finite(), "non-finite matrix entry: {v}");
        }
    }

    #[test]
    fn tick_skips_when_clip_missing() {
        let asset = tiny_asset();
        let mut world = hecs::World::new();
        let entity = world.spawn((
            skeleton_component(asset.clone(), PathBuf::from("test.rkskel"), glam::Vec3::ZERO),
            AnimationPlayer {
                clip_name: "does-not-exist".into(),
                time: 0.0,
                speed: 1.0,
                playing: true,
                loop_mode: LoopMode::Loop,
                forward: true,
            },
        ));

        let changed = tick(&mut world, 0.5);
        assert!(!changed, "a missing clip should report no change");

        // Pose still identity.
        let skel = world.get::<&Skeleton>(entity).unwrap();
        assert_eq!(skel.current_pose.len(), 1);
        assert_eq!(skel.current_pose[0], Mat4::IDENTITY);
    }

    #[test]
    fn tick_paused_does_not_advance() {
        let asset = tiny_asset();
        let mut world = hecs::World::new();
        let entity = world.spawn((
            skeleton_component(asset.clone(), PathBuf::from("test.rkskel"), glam::Vec3::ZERO),
            AnimationPlayer {
                clip_name: "wave".into(),
                time: 0.5,
                speed: 1.0,
                playing: false,  // paused
                loop_mode: LoopMode::Loop,
                forward: true,
            },
        ));

        tick(&mut world, 100.0);  // try to advance a lot

        let player = world.get::<&AnimationPlayer>(entity).unwrap();
        assert_eq!(player.time, 0.5, "paused player must not advance");
    }
}
