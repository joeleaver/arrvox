//! Animation playback system for skeletal and blend shape animations.
//!
//! [`AnimationPlayer`] drives an [`AnimationClip`] forward in time, handling
//! looping modes (Once, Loop, PingPong), speed scaling, and pause/stop/seek.
//!
//! [`BlendShapeAnimation`] provides per-blend-shape weight tracks with
//! keyframe interpolation, applied to a [`BlendShapeSet`].

use arvx_core::scene_node::Transform;
use serde::{Deserialize, Serialize};

use crate::blend_shape::BlendShapeSet;
use crate::clip::AnimationClip;
use crate::skeleton::Skeleton;

// ─── LoopMode ────────────────────────────────────────────────────────────────

/// Looping mode for animation playback.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LoopMode {
    /// Play once and stop at the end.
    Once,
    /// Loop continuously.
    Loop,
    /// Play forward then backward (ping-pong).
    PingPong,
}

// ─── AnimationPlayer ─────────────────────────────────────────────────────────

/// Animation player component.
///
/// Tracks the current clip, time, speed, and looping mode.
/// Call `advance(dt)` each frame, then use `current_time()` to evaluate.
#[derive(Debug, Clone)]
pub struct AnimationPlayer {
    /// The current animation clip being played.
    clip: AnimationClip,
    /// Current playback time in seconds.
    current_time: f32,
    /// Playback speed multiplier (1.0 = normal, 0.5 = half speed, 2.0 = double).
    speed: f32,
    /// Whether the player is currently playing.
    playing: bool,
    /// Loop mode.
    loop_mode: LoopMode,
    /// For PingPong: true = forward, false = backward.
    forward: bool,
}

impl AnimationPlayer {
    /// Create a new player with the given clip.
    pub fn new(clip: AnimationClip) -> Self {
        Self {
            clip,
            current_time: 0.0,
            speed: 1.0,
            playing: true,
            loop_mode: LoopMode::Loop,
            forward: true,
        }
    }

    /// Get the current clip.
    pub fn clip(&self) -> &AnimationClip {
        &self.clip
    }

    /// Set a new clip, resetting time to 0.
    pub fn set_clip(&mut self, clip: AnimationClip) {
        self.clip = clip;
        self.current_time = 0.0;
        self.forward = true;
    }

    /// Get current playback time.
    pub fn current_time(&self) -> f32 {
        self.current_time
    }

    /// Set current time (clamped to [0, duration]).
    pub fn seek(&mut self, time: f32) {
        self.current_time = time.clamp(0.0, self.clip.duration);
    }

    /// Get playback speed.
    pub fn speed(&self) -> f32 {
        self.speed
    }

    /// Set playback speed.
    pub fn set_speed(&mut self, speed: f32) {
        self.speed = speed;
    }

    /// Whether the player is playing.
    pub fn is_playing(&self) -> bool {
        self.playing
    }

    /// Start playback.
    pub fn play(&mut self) {
        self.playing = true;
    }

    /// Pause playback.
    pub fn pause(&mut self) {
        self.playing = false;
    }

    /// Stop playback and reset to beginning.
    pub fn stop(&mut self) {
        self.playing = false;
        self.current_time = 0.0;
        self.forward = true;
    }

    /// Get loop mode.
    pub fn loop_mode(&self) -> LoopMode {
        self.loop_mode
    }

    /// Set loop mode.
    pub fn set_loop_mode(&mut self, mode: LoopMode) {
        self.loop_mode = mode;
    }

    /// Whether the animation has finished (only relevant for LoopMode::Once).
    pub fn is_finished(&self) -> bool {
        self.loop_mode == LoopMode::Once && self.current_time >= self.clip.duration
    }

    /// Evaluate per-bone local transforms at the current time.
    ///
    /// Returns a `Transform` per bone suitable for writing directly to
    /// SceneNode `local_transform` fields.
    pub fn evaluate_local(&self, skeleton: &Skeleton) -> Vec<Transform> {
        skeleton.evaluate_local(&self.clip, self.current_time)
    }

    /// Advance the animation by `dt` seconds.
    pub fn advance(&mut self, dt: f32) {
        if !self.playing {
            return;
        }

        let delta = dt * self.speed;

        match self.loop_mode {
            LoopMode::Once => {
                self.current_time = (self.current_time + delta).clamp(0.0, self.clip.duration);
            }
            LoopMode::Loop => {
                self.current_time += delta;
                if self.clip.duration > 0.0 {
                    self.current_time = self.current_time.rem_euclid(self.clip.duration);
                }
            }
            LoopMode::PingPong => {
                if self.forward {
                    self.current_time += delta;
                    if self.current_time >= self.clip.duration {
                        self.current_time = 2.0 * self.clip.duration - self.current_time;
                        self.forward = false;
                    }
                } else {
                    self.current_time -= delta;
                    if self.current_time <= 0.0 {
                        self.current_time = -self.current_time;
                        self.forward = true;
                    }
                }
                self.current_time = self.current_time.clamp(0.0, self.clip.duration);
            }
        }
    }
}

// ─── BlendShapeAnimation ─────────────────────────────────────────────────────

/// Blend shape animation: a set of weight keyframes for blend shapes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlendShapeAnimation {
    /// Channel per blend shape: (blend_shape_name, keyframes of (time, weight)).
    pub channels: Vec<BlendShapeChannel>,
    /// Duration of the animation.
    pub duration: f32,
}

/// A channel of blend shape weight keyframes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlendShapeChannel {
    /// Name of the target blend shape.
    pub shape_name: String,
    /// Keyframes: (time, weight) pairs, sorted by time.
    pub keyframes: Vec<(f32, f32)>,
}

impl BlendShapeChannel {
    /// Sample the weight at time t.
    pub fn sample(&self, t: f32) -> f32 {
        if self.keyframes.is_empty() {
            return 0.0;
        }
        if self.keyframes.len() == 1 || t <= self.keyframes[0].0 {
            return self.keyframes[0].1;
        }
        if t >= self.keyframes.last().unwrap().0 {
            return self.keyframes.last().unwrap().1;
        }
        // Binary search for the surrounding keyframes.
        let idx = self.keyframes.partition_point(|k| k.0 < t);
        let (t0, w0) = self.keyframes[idx - 1];
        let (t1, w1) = self.keyframes[idx];
        let frac = (t - t0) / (t1 - t0).max(1e-10);
        w0 + (w1 - w0) * frac
    }
}

impl BlendShapeAnimation {
    /// Apply blend shape weights at time t to a BlendShapeSet.
    pub fn apply(&self, t: f32, shapes: &mut BlendShapeSet) {
        for ch in &self.channels {
            let w = ch.sample(t);
            shapes.set_weight(&ch.shape_name, w);
        }
    }
}

// ─── tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::blend_shape::BlendShape;
    use glam::Vec3;
    use arvx_core::aabb::Aabb;

    fn make_clip(duration: f32) -> AnimationClip {
        AnimationClip::new("test".to_string(), duration, vec![])
    }

    fn default_aabb() -> Aabb {
        Aabb::new(Vec3::ZERO, Vec3::ONE)
    }

    fn make_blend_shape_set() -> BlendShapeSet {
        BlendShapeSet::new(vec![
            BlendShape::new("smile", "Head", 0, 1, default_aabb()),
            BlendShape::new("blink", "Head", 1, 1, default_aabb()),
        ])
    }

    // ── serde roundtrip ────────────────────────────────────────────────────

    #[test]
    fn loop_mode_serde_roundtrip() {
        for mode in [LoopMode::Once, LoopMode::Loop, LoopMode::PingPong] {
            let ron_str = ron::to_string(&mode).expect("serialize LoopMode");
            let mode2: LoopMode = ron::from_str(&ron_str).expect("deserialize LoopMode");
            assert_eq!(mode, mode2, "roundtrip failed for {mode:?}");
        }
    }

    // ── AnimationPlayer defaults ──────────────────────────────────────────────

    #[test]
    fn player_new_defaults() {
        let player = AnimationPlayer::new(make_clip(2.0));
        assert!(player.is_playing(), "should start playing");
        assert_eq!(player.speed(), 1.0, "default speed is 1.0");
        assert_eq!(player.loop_mode(), LoopMode::Loop, "default loop mode is Loop");
        assert_eq!(player.current_time(), 0.0, "starts at time 0");
        assert!(!player.is_finished(), "not finished at start");
    }

    // ── Loop wraps ────────────────────────────────────────────────────────────

    #[test]
    fn advance_loop_wraps_at_duration() {
        let mut player = AnimationPlayer::new(make_clip(1.0));
        player.set_loop_mode(LoopMode::Loop);
        // Advance 1.25 seconds on a 1s clip — should wrap to 0.25.
        player.advance(1.25);
        let t = player.current_time();
        assert!(
            (t - 0.25).abs() < 1e-5,
            "expected 0.25 after wrap, got {t}"
        );
    }

    #[test]
    fn advance_loop_wraps_exactly_at_duration() {
        let mut player = AnimationPlayer::new(make_clip(2.0));
        player.set_loop_mode(LoopMode::Loop);
        // 2.0 rem_euclid 2.0 = 0.0
        player.advance(2.0);
        assert_eq!(player.current_time(), 0.0);
    }

    // ── Once clamps ───────────────────────────────────────────────────────────

    #[test]
    fn advance_once_clamps_at_duration() {
        let mut player = AnimationPlayer::new(make_clip(1.0));
        player.set_loop_mode(LoopMode::Once);
        player.advance(5.0);
        assert_eq!(
            player.current_time(),
            1.0,
            "Once mode should clamp at duration"
        );
    }

    #[test]
    fn advance_once_partial_does_not_clamp() {
        let mut player = AnimationPlayer::new(make_clip(2.0));
        player.set_loop_mode(LoopMode::Once);
        player.advance(0.5);
        assert!((player.current_time() - 0.5).abs() < 1e-5);
    }

    // ── PingPong reverses ─────────────────────────────────────────────────────

    #[test]
    fn advance_pingpong_reverses_direction() {
        let mut player = AnimationPlayer::new(make_clip(1.0));
        player.set_loop_mode(LoopMode::PingPong);
        // Advance to just past the end — should fold back.
        player.advance(1.2);
        let t = player.current_time();
        // After folding: 2*1.0 - 1.2 = 0.8, then clamped.
        assert!(
            (t - 0.8).abs() < 1e-5,
            "expected ~0.8 after ping, got {t}"
        );
        // Now going backward — advance again should decrease time.
        player.advance(0.2);
        let t2 = player.current_time();
        assert!(
            (t2 - 0.6).abs() < 1e-5,
            "expected ~0.6 going backward, got {t2}"
        );
    }

    #[test]
    fn advance_pingpong_bounces_at_zero() {
        let mut player = AnimationPlayer::new(make_clip(1.0));
        player.set_loop_mode(LoopMode::PingPong);
        // Go to end, then reverse back past zero.
        player.advance(1.5); // forward: fold at 1.0 → 0.5, going backward
        player.advance(0.8); // backward: 0.5 - 0.8 = -0.3 → bounce → 0.3, forward
        let t = player.current_time();
        assert!(
            (t - 0.3).abs() < 1e-5,
            "expected ~0.3 after bounce at zero, got {t}"
        );
    }

    // ── is_finished ───────────────────────────────────────────────────────────

    #[test]
    fn is_finished_only_true_for_once_at_end() {
        let mut player = AnimationPlayer::new(make_clip(1.0));
        player.set_loop_mode(LoopMode::Once);
        assert!(!player.is_finished());
        player.advance(10.0);
        assert!(player.is_finished(), "Once mode at end should be finished");
    }

    #[test]
    fn is_finished_false_for_loop_at_end() {
        let mut player = AnimationPlayer::new(make_clip(1.0));
        player.set_loop_mode(LoopMode::Loop);
        player.advance(10.0);
        assert!(!player.is_finished(), "Loop mode is never finished");
    }

    #[test]
    fn is_finished_false_for_pingpong_at_end() {
        let mut player = AnimationPlayer::new(make_clip(1.0));
        player.set_loop_mode(LoopMode::PingPong);
        player.advance(10.0);
        assert!(!player.is_finished(), "PingPong mode is never finished");
    }

    // ── pause ─────────────────────────────────────────────────────────────────

    #[test]
    fn pause_stops_advancement() {
        let mut player = AnimationPlayer::new(make_clip(2.0));
        player.advance(0.5);
        player.pause();
        assert!(!player.is_playing());
        let t_before = player.current_time();
        player.advance(1.0); // should have no effect
        assert_eq!(
            player.current_time(),
            t_before,
            "time must not change while paused"
        );
    }

    #[test]
    fn play_resumes_after_pause() {
        let mut player = AnimationPlayer::new(make_clip(2.0));
        player.pause();
        player.play();
        assert!(player.is_playing());
        let t_before = player.current_time();
        player.advance(0.3);
        assert!(
            player.current_time() > t_before,
            "time should advance after play()"
        );
    }

    // ── stop ──────────────────────────────────────────────────────────────────

    #[test]
    fn stop_resets_to_beginning() {
        let mut player = AnimationPlayer::new(make_clip(2.0));
        player.advance(1.0);
        player.stop();
        assert!(!player.is_playing(), "stop should pause playback");
        assert_eq!(player.current_time(), 0.0, "stop should reset time to 0");
    }

    // ── seek ──────────────────────────────────────────────────────────────────

    #[test]
    fn seek_clamps_to_valid_range() {
        let mut player = AnimationPlayer::new(make_clip(2.0));
        player.seek(-1.0);
        assert_eq!(player.current_time(), 0.0, "seek below 0 clamps to 0");
        player.seek(100.0);
        assert_eq!(
            player.current_time(),
            2.0,
            "seek above duration clamps to duration"
        );
        player.seek(1.5);
        assert!((player.current_time() - 1.5).abs() < 1e-6);
    }

    // ── set_speed ─────────────────────────────────────────────────────────────

    #[test]
    fn set_speed_affects_advance_rate() {
        let mut player_normal = AnimationPlayer::new(make_clip(10.0));
        let mut player_double = AnimationPlayer::new(make_clip(10.0));
        player_double.set_speed(2.0);

        player_normal.advance(1.0);
        player_double.advance(1.0);

        assert!(
            (player_double.current_time() - 2.0 * player_normal.current_time()).abs() < 1e-5,
            "double speed should advance twice as fast"
        );
    }

    // ── set_clip resets time ──────────────────────────────────────────────────

    #[test]
    fn set_clip_resets_time_to_zero() {
        let mut player = AnimationPlayer::new(make_clip(2.0));
        player.advance(1.5);
        assert!(player.current_time() > 0.0);
        let new_clip = make_clip(3.0);
        player.set_clip(new_clip);
        assert_eq!(player.current_time(), 0.0, "set_clip must reset time");
        assert_eq!(
            player.clip().duration,
            3.0,
            "new clip should be active"
        );
    }

    // ── evaluate_local ─────────────────────────────────────────────────────────

    #[test]
    fn player_evaluate_local_returns_transforms() {
        use crate::clip::{AnimationClip, BoneChannel, Keyframe};
        use crate::skeleton::{Bone, Skeleton};
        use glam::{Mat4, Quat, Vec3};

        let bones = vec![
            Bone {
                name: "root".into(),
                bind_transform: Mat4::IDENTITY,
                inverse_bind: Mat4::IDENTITY,
            },
        ];
        let skel = Skeleton::new(bones, vec![-1]).unwrap();

        let clip = AnimationClip::new(
            "test".to_string(),
            1.0,
            vec![BoneChannel {
                bone_index: 0,
                keyframes: vec![
                    Keyframe { time: 0.0, position: Vec3::ZERO, rotation: Quat::IDENTITY, scale: Vec3::ONE },
                    Keyframe { time: 1.0, position: Vec3::new(10.0, 0.0, 0.0), rotation: Quat::IDENTITY, scale: Vec3::ONE },
                ],
            }],
        );

        let mut player = AnimationPlayer::new(clip);
        player.advance(0.5);

        let locals = player.evaluate_local(&skel);
        assert_eq!(locals.len(), 1);
        assert!((locals[0].position.x - 5.0).abs() < 1e-4, "expected x=5.0, got {}", locals[0].position.x);
    }

    // ── BlendShapeChannel ─────────────────────────────────────────────────────

    #[test]
    fn blend_shape_channel_sample_single_keyframe() {
        let ch = BlendShapeChannel {
            shape_name: "smile".to_string(),
            keyframes: vec![(0.5, 0.8)],
        };
        // Any time should return 0.8.
        assert!((ch.sample(0.0) - 0.8).abs() < 1e-6);
        assert!((ch.sample(0.5) - 0.8).abs() < 1e-6);
        assert!((ch.sample(2.0) - 0.8).abs() < 1e-6);
    }

    #[test]
    fn blend_shape_channel_sample_interpolates_between_two() {
        let ch = BlendShapeChannel {
            shape_name: "blink".to_string(),
            keyframes: vec![(0.0, 0.0), (1.0, 1.0)],
        };
        assert!((ch.sample(0.5) - 0.5).abs() < 1e-5);
        assert!((ch.sample(0.25) - 0.25).abs() < 1e-5);
        assert!((ch.sample(0.75) - 0.75).abs() < 1e-5);
    }

    #[test]
    fn blend_shape_channel_sample_clamps_before_first_and_after_last() {
        let ch = BlendShapeChannel {
            shape_name: "jaw".to_string(),
            keyframes: vec![(1.0, 0.2), (2.0, 0.9)],
        };
        // Before first keyframe — clamps to first weight.
        assert!((ch.sample(0.0) - 0.2).abs() < 1e-6, "before first");
        // After last keyframe — clamps to last weight.
        assert!((ch.sample(5.0) - 0.9).abs() < 1e-6, "after last");
    }

    #[test]
    fn blend_shape_channel_sample_empty_returns_zero() {
        let ch = BlendShapeChannel {
            shape_name: "empty".to_string(),
            keyframes: vec![],
        };
        assert_eq!(ch.sample(0.5), 0.0);
    }

    // ── BlendShapeAnimation::apply ────────────────────────────────────────────

    #[test]
    fn blend_shape_animation_apply_sets_weights_on_set() {
        let anim = BlendShapeAnimation {
            channels: vec![
                BlendShapeChannel {
                    shape_name: "smile".to_string(),
                    keyframes: vec![(0.0, 0.0), (1.0, 1.0)],
                },
                BlendShapeChannel {
                    shape_name: "blink".to_string(),
                    keyframes: vec![(0.0, 0.5), (1.0, 0.5)],
                },
            ],
            duration: 1.0,
        };

        let mut shapes = make_blend_shape_set();
        anim.apply(0.5, &mut shapes);

        let smile_w = shapes.find("smile").unwrap().weight;
        let blink_w = shapes.find("blink").unwrap().weight;

        assert!((smile_w - 0.5).abs() < 1e-5, "smile at t=0.5 should be 0.5, got {smile_w}");
        assert!((blink_w - 0.5).abs() < 1e-5, "blink should stay 0.5, got {blink_w}");
    }

    #[test]
    fn blend_shape_animation_apply_ignores_unknown_shapes() {
        let anim = BlendShapeAnimation {
            channels: vec![BlendShapeChannel {
                shape_name: "nonexistent".to_string(),
                keyframes: vec![(0.0, 1.0)],
            }],
            duration: 1.0,
        };
        // Should not panic even when shape name doesn't exist in the set.
        let mut shapes = make_blend_shape_set();
        anim.apply(0.0, &mut shapes);
        // All shapes should remain at weight 0.
        for s in &shapes.shapes {
            assert_eq!(s.weight, 0.0);
        }
    }
}
