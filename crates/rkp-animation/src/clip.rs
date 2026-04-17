//! Animation clips: keyframes, bone channels, and sampling.
//!
//! An [`AnimationClip`] contains one or more [`BoneChannel`]s, each targeting
//! a specific bone by index. A channel holds [`Keyframe`]s sorted by time.
//! Sampling interpolates between surrounding keyframes using lerp for
//! position/scale and slerp for rotation.

use glam::{Quat, Vec3};
use serde::{Deserialize, Serialize};

/// A single keyframe sample at a specific point in time.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Keyframe {
    /// Time in seconds from clip start.
    pub time: f32,
    /// Local position relative to parent bone.
    pub position: Vec3,
    /// Local rotation relative to parent bone.
    pub rotation: Quat,
    /// Local scale (uniform recommended for SDF engine).
    pub scale: Vec3,
}

/// Animation channel for one bone, containing an ordered sequence of keyframes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoneChannel {
    /// Index of the bone this channel animates.
    pub bone_index: u32,
    /// Keyframes sorted by ascending time.
    pub keyframes: Vec<Keyframe>,
}

impl BoneChannel {
    /// Sample the channel at time `t`, interpolating between keyframes.
    ///
    /// - Position and scale use linear interpolation (lerp).
    /// - Rotation uses spherical linear interpolation (slerp).
    ///
    /// Edge cases:
    /// - Before first keyframe: clamps to first keyframe values.
    /// - After last keyframe: clamps to last keyframe values.
    /// - Single keyframe: returns that keyframe regardless of `t`.
    /// - Empty keyframes: returns identity transform (Vec3::ZERO, Quat::IDENTITY, Vec3::ONE).
    pub fn sample(&self, t: f32) -> (Vec3, Quat, Vec3) {
        match self.keyframes.len() {
            0 => (Vec3::ZERO, Quat::IDENTITY, Vec3::ONE),
            1 => {
                let kf = &self.keyframes[0];
                (kf.position, kf.rotation, kf.scale)
            }
            _ => {
                let first = &self.keyframes[0];
                let last = self.keyframes.last().unwrap();

                // Clamp to bounds.
                if t <= first.time {
                    return (first.position, first.rotation, first.scale);
                }
                if t >= last.time {
                    return (last.position, last.rotation, last.scale);
                }

                // Binary search for the keyframe pair surrounding `t`.
                // We want the largest index where keyframes[index].time <= t.
                let idx = self
                    .keyframes
                    .partition_point(|kf| kf.time <= t)
                    .saturating_sub(1);

                let kf_a = &self.keyframes[idx];
                let kf_b = &self.keyframes[idx + 1];

                // Interpolation factor [0, 1].
                let duration = kf_b.time - kf_a.time;
                let alpha = if duration > 0.0 {
                    (t - kf_a.time) / duration
                } else {
                    0.0
                };

                let position = kf_a.position.lerp(kf_b.position, alpha);
                let rotation = kf_a.rotation.slerp(kf_b.rotation, alpha);
                let scale = kf_a.scale.lerp(kf_b.scale, alpha);

                (position, rotation, scale)
            }
        }
    }
}

/// An animation clip containing channels for one or more bones.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnimationClip {
    /// Clip name (e.g. "walk", "idle", "attack").
    pub name: String,
    /// Total duration in seconds.
    pub duration: f32,
    /// Animation channels, one per animated bone.
    pub channels: Vec<BoneChannel>,
}

impl AnimationClip {
    /// Create a new animation clip.
    pub fn new(name: String, duration: f32, channels: Vec<BoneChannel>) -> Self {
        Self {
            name,
            duration,
            channels,
        }
    }

    /// Find the channel animating the given bone index, if any.
    pub fn channel_for_bone(&self, bone_index: u32) -> Option<&BoneChannel> {
        self.channels.iter().find(|ch| ch.bone_index == bone_index)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::FRAC_PI_2;

    fn kf(time: f32, pos: Vec3, rot: Quat, scale: Vec3) -> Keyframe {
        Keyframe {
            time,
            position: pos,
            rotation: rot,
            scale,
        }
    }

    #[test]
    fn test_sample_empty_keyframes() {
        let ch = BoneChannel {
            bone_index: 0,
            keyframes: vec![],
        };
        let (pos, rot, scl) = ch.sample(0.5);
        assert_eq!(pos, Vec3::ZERO);
        assert_eq!(rot, Quat::IDENTITY);
        assert_eq!(scl, Vec3::ONE);
    }

    #[test]
    fn test_sample_single_keyframe() {
        let position = Vec3::new(1.0, 2.0, 3.0);
        let rotation = Quat::from_rotation_y(1.0);
        let scale = Vec3::splat(2.0);
        let ch = BoneChannel {
            bone_index: 0,
            keyframes: vec![kf(0.5, position, rotation, scale)],
        };

        // Should return same values regardless of time.
        for t in [0.0, 0.25, 0.5, 0.75, 1.0] {
            let (p, r, s) = ch.sample(t);
            assert_eq!(p, position, "position at t={t}");
            assert!((r - rotation).length() < 1e-6, "rotation at t={t}");
            assert_eq!(s, scale, "scale at t={t}");
        }
    }

    #[test]
    fn test_sample_before_first_keyframe_clamps() {
        let ch = BoneChannel {
            bone_index: 0,
            keyframes: vec![
                kf(1.0, Vec3::new(1.0, 0.0, 0.0), Quat::IDENTITY, Vec3::ONE),
                kf(2.0, Vec3::new(2.0, 0.0, 0.0), Quat::IDENTITY, Vec3::ONE),
            ],
        };
        let (pos, _, _) = ch.sample(0.0);
        assert_eq!(pos, Vec3::new(1.0, 0.0, 0.0));
    }

    #[test]
    fn test_sample_after_last_keyframe_clamps() {
        let ch = BoneChannel {
            bone_index: 0,
            keyframes: vec![
                kf(0.0, Vec3::new(1.0, 0.0, 0.0), Quat::IDENTITY, Vec3::ONE),
                kf(1.0, Vec3::new(2.0, 0.0, 0.0), Quat::IDENTITY, Vec3::ONE),
            ],
        };
        let (pos, _, _) = ch.sample(5.0);
        assert_eq!(pos, Vec3::new(2.0, 0.0, 0.0));
    }

    #[test]
    fn test_sample_lerp_position_midpoint() {
        let ch = BoneChannel {
            bone_index: 0,
            keyframes: vec![
                kf(0.0, Vec3::new(0.0, 0.0, 0.0), Quat::IDENTITY, Vec3::ONE),
                kf(1.0, Vec3::new(10.0, 0.0, 0.0), Quat::IDENTITY, Vec3::ONE),
            ],
        };
        let (pos, _, _) = ch.sample(0.5);
        assert!(
            (pos - Vec3::new(5.0, 0.0, 0.0)).length() < 1e-5,
            "expected (5,0,0), got {pos}"
        );
    }

    #[test]
    fn test_sample_lerp_position_quarter() {
        let ch = BoneChannel {
            bone_index: 0,
            keyframes: vec![
                kf(0.0, Vec3::ZERO, Quat::IDENTITY, Vec3::ONE),
                kf(2.0, Vec3::new(8.0, 4.0, 0.0), Quat::IDENTITY, Vec3::ONE),
            ],
        };
        let (pos, _, _) = ch.sample(0.5); // 25% of [0, 2]
        assert!(
            (pos - Vec3::new(2.0, 1.0, 0.0)).length() < 1e-5,
            "expected (2,1,0), got {pos}"
        );
    }

    #[test]
    fn test_sample_slerp_rotation_midpoint() {
        let rot_a = Quat::IDENTITY;
        let rot_b = Quat::from_rotation_z(FRAC_PI_2); // 90 degrees
        let ch = BoneChannel {
            bone_index: 0,
            keyframes: vec![
                kf(0.0, Vec3::ZERO, rot_a, Vec3::ONE),
                kf(1.0, Vec3::ZERO, rot_b, Vec3::ONE),
            ],
        };
        let (_, rot, _) = ch.sample(0.5);
        let expected = Quat::from_rotation_z(FRAC_PI_2 / 2.0); // 45 degrees
        assert!(
            (rot - expected).length() < 1e-5 || (rot + expected).length() < 1e-5,
            "expected ~45deg Z rotation, got {rot:?}"
        );
    }

    #[test]
    fn test_sample_lerp_scale() {
        let ch = BoneChannel {
            bone_index: 0,
            keyframes: vec![
                kf(0.0, Vec3::ZERO, Quat::IDENTITY, Vec3::ONE),
                kf(1.0, Vec3::ZERO, Quat::IDENTITY, Vec3::splat(3.0)),
            ],
        };
        let (_, _, scl) = ch.sample(0.5);
        assert!(
            (scl - Vec3::splat(2.0)).length() < 1e-5,
            "expected uniform scale 2.0, got {scl}"
        );
    }

    #[test]
    fn test_sample_three_keyframes() {
        // Three keyframes: 0.0, 1.0, 2.0
        // Sample at 1.5 should interpolate between keyframes 1 and 2.
        let ch = BoneChannel {
            bone_index: 0,
            keyframes: vec![
                kf(0.0, Vec3::new(0.0, 0.0, 0.0), Quat::IDENTITY, Vec3::ONE),
                kf(1.0, Vec3::new(10.0, 0.0, 0.0), Quat::IDENTITY, Vec3::ONE),
                kf(2.0, Vec3::new(10.0, 10.0, 0.0), Quat::IDENTITY, Vec3::ONE),
            ],
        };
        let (pos, _, _) = ch.sample(1.5);
        // Halfway between kf1 (10,0,0) and kf2 (10,10,0) = (10, 5, 0)
        assert!(
            (pos - Vec3::new(10.0, 5.0, 0.0)).length() < 1e-5,
            "expected (10,5,0), got {pos}"
        );
    }

    // ── serde roundtrip ────────────────────────────────────────────────────

    #[test]
    fn test_keyframe_serde_roundtrip() {
        let kf_orig = kf(
            0.5,
            Vec3::new(1.0, 2.0, 3.0),
            Quat::from_rotation_y(1.0),
            Vec3::splat(2.0),
        );
        let ron_str = ron::to_string(&kf_orig).expect("serialize Keyframe");
        let kf2: Keyframe = ron::from_str(&ron_str).expect("deserialize Keyframe");
        assert!((kf_orig.time - kf2.time).abs() < 1e-6);
        assert!((kf_orig.position - kf2.position).length() < 1e-6);
        assert!((kf_orig.scale - kf2.scale).length() < 1e-6);
    }

    #[test]
    fn test_bone_channel_serde_roundtrip() {
        let ch = BoneChannel {
            bone_index: 5,
            keyframes: vec![
                kf(0.0, Vec3::ZERO, Quat::IDENTITY, Vec3::ONE),
                kf(0.5, Vec3::new(1.0, 0.0, 0.0), Quat::IDENTITY, Vec3::ONE),
                kf(1.0, Vec3::new(2.0, 0.0, 0.0), Quat::IDENTITY, Vec3::ONE),
            ],
        };
        let ron_str = ron::to_string(&ch).expect("serialize BoneChannel");
        let ch2: BoneChannel = ron::from_str(&ron_str).expect("deserialize BoneChannel");
        assert_eq!(ch.bone_index, ch2.bone_index);
        assert_eq!(ch.keyframes.len(), ch2.keyframes.len());
    }

    #[test]
    fn test_animation_clip_serde_roundtrip() {
        let clip = AnimationClip::new(
            "walk".to_string(),
            2.0,
            vec![
                BoneChannel {
                    bone_index: 0,
                    keyframes: vec![kf(0.0, Vec3::ZERO, Quat::IDENTITY, Vec3::ONE)],
                },
                BoneChannel {
                    bone_index: 3,
                    keyframes: vec![
                        kf(0.0, Vec3::ZERO, Quat::IDENTITY, Vec3::ONE),
                        kf(1.0, Vec3::X, Quat::IDENTITY, Vec3::ONE),
                    ],
                },
            ],
        );
        let ron_str = ron::to_string(&clip).expect("serialize AnimationClip");
        let clip2: AnimationClip = ron::from_str(&ron_str).expect("deserialize AnimationClip");
        assert_eq!(clip.name, clip2.name);
        assert_eq!(clip.duration, clip2.duration);
        assert_eq!(clip.channels.len(), clip2.channels.len());
    }

    #[test]
    fn test_clip_channel_for_bone() {
        let clip = AnimationClip::new(
            "test".to_string(),
            1.0,
            vec![
                BoneChannel {
                    bone_index: 2,
                    keyframes: vec![],
                },
                BoneChannel {
                    bone_index: 5,
                    keyframes: vec![],
                },
            ],
        );
        assert!(clip.channel_for_bone(2).is_some());
        assert!(clip.channel_for_bone(5).is_some());
        assert!(clip.channel_for_bone(0).is_none());
        assert!(clip.channel_for_bone(3).is_none());
    }
}
