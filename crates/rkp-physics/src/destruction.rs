//! Destruction events and debris particle generation.
//!
//! When voxels are destroyed (CSG subtraction, explosion, etc.), this module
//! converts the destruction into a set of [`DebrisSpawnRequest`]s that the
//! particle system can pick up. Debris velocities radiate outward from the
//! destruction center, scaled by the event force.
//!
//! Uses a deterministic hash-based pseudo-random generator to avoid a `rand`
//! dependency while keeping debris distribution repeatable for a given seed.

use glam::Vec3;

// ---------------------------------------------------------------------------
// Deterministic hash RNG
// ---------------------------------------------------------------------------

/// Deterministic hash producing a float in `[0, 1]` from a seed and index.
///
/// Uses integer hashing (Knuth-style multiplicative + finalizer) to generate
/// repeatable pseudo-random values without pulling in `rand`.
#[inline]
fn hash_f32(seed: u32, index: u32) -> f32 {
    let h = seed
        .wrapping_mul(0x9E37_79B9)
        .wrapping_add(index.wrapping_mul(0x517C_C1B7));
    let h = (h ^ (h >> 16)).wrapping_mul(0x045D_9F3B);
    let h = h ^ (h >> 16);
    (h & 0xFFFF) as f32 / 65535.0
}

// ---------------------------------------------------------------------------
// DestructionEvent
// ---------------------------------------------------------------------------

/// Describes a destruction that occurred in the world.
///
/// Typically created by the edit system when a CSG subtraction removes voxels,
/// or by gameplay code for explosions.
#[derive(Debug, Clone)]
pub struct DestructionEvent {
    /// World-space center of the destruction.
    pub center: Vec3,
    /// Approximate radius of the destroyed region.
    pub radius: f32,
    /// Material ID of the destroyed voxels (used to tint debris).
    pub material_id: u16,
    /// Explosion force — scales outward debris velocity.
    pub force: f32,
}

// ---------------------------------------------------------------------------
// DebrisConfig
// ---------------------------------------------------------------------------

/// Tuning parameters for debris generation.
#[derive(Debug, Clone)]
pub struct DebrisConfig {
    /// Maximum number of debris particles per event. Default: `50`.
    pub max_debris_count: u32,
    /// Lifetime in seconds before debris fades out. Default: `3.0`.
    pub debris_lifetime: f32,
    /// Minimum debris particle radius. Default: `0.02`.
    pub debris_size_min: f32,
    /// Maximum debris particle radius. Default: `0.08`.
    pub debris_size_max: f32,
    /// Multiplier on outward velocity. Default: `1.0`.
    pub velocity_scale: f32,
    /// Gravity scale for debris particles. Default: `1.0`.
    pub gravity_scale: f32,
}

impl Default for DebrisConfig {
    fn default() -> Self {
        Self {
            max_debris_count: 50,
            debris_lifetime: 3.0,
            debris_size_min: 0.02,
            debris_size_max: 0.08,
            velocity_scale: 1.0,
            gravity_scale: 1.0,
        }
    }
}

// ---------------------------------------------------------------------------
// DebrisSpawnRequest
// ---------------------------------------------------------------------------

/// A single debris particle to be handed off to the particle system.
#[derive(Debug, Clone)]
pub struct DebrisSpawnRequest {
    /// World-space spawn position.
    pub position: Vec3,
    /// Initial velocity (outward from destruction center).
    pub velocity: Vec3,
    /// Particle radius.
    pub size: f32,
    /// Material ID for visual tinting.
    pub material_id: u16,
    /// Time in seconds before this particle expires.
    pub lifetime: f32,
}

// ---------------------------------------------------------------------------
// generate_debris
// ---------------------------------------------------------------------------

/// Generate debris spawn requests for a destruction event.
///
/// Produces up to `config.max_debris_count` particles distributed in a sphere
/// around `event.center`. Each particle gets a deterministic pseudo-random
/// position offset, outward velocity proportional to `event.force`, and a
/// random size within the configured range.
pub fn generate_debris(
    event: &DestructionEvent,
    config: &DebrisConfig,
    rng_seed: u32,
) -> Vec<DebrisSpawnRequest> {
    let count = config.max_debris_count;
    let mut result = Vec::with_capacity(count as usize);

    for i in 0..count {
        // 3 hash calls per particle for x, y, z offsets
        let base = i * 7;
        let rx = hash_f32(rng_seed, base) * 2.0 - 1.0;
        let ry = hash_f32(rng_seed, base + 1) * 2.0 - 1.0;
        let rz = hash_f32(rng_seed, base + 2) * 2.0 - 1.0;

        // Random direction, scaled to within the destruction radius
        let offset_dir = Vec3::new(rx, ry, rz);
        let offset_len = offset_dir.length();
        let offset_dir = if offset_len > 1e-6 {
            offset_dir / offset_len
        } else {
            Vec3::Y
        };

        // Position: within the destruction radius
        let radius_frac = hash_f32(rng_seed, base + 3);
        let position = event.center + offset_dir * (event.radius * radius_frac);

        // Velocity: outward, scaled by force and distance from center
        // Closer to center = slightly faster (inverse falloff)
        let dist_factor = 1.0 - radius_frac * 0.5; // range [0.5, 1.0]
        let speed = event.force * dist_factor * config.velocity_scale;
        let velocity = offset_dir * speed;

        // Size: random between min and max
        let size_t = hash_f32(rng_seed, base + 4);
        let size = config.debris_size_min + size_t * (config.debris_size_max - config.debris_size_min);

        // Lifetime: slight variation around configured value
        let life_t = hash_f32(rng_seed, base + 5);
        let lifetime = config.debris_lifetime * (0.5 + life_t * 0.5);

        result.push(DebrisSpawnRequest {
            position,
            velocity,
            size,
            material_id: event.material_id,
            lifetime,
        });
    }

    result
}

// ---------------------------------------------------------------------------
// DebrisEventQueue
// ---------------------------------------------------------------------------

/// Simple queue of pending destruction events.
///
/// Game code or the edit system pushes events; the physics/particle tick
/// drains them to generate debris.
#[derive(Debug, Default)]
pub struct DebrisEventQueue {
    events: Vec<DestructionEvent>,
}

impl DebrisEventQueue {
    /// Create an empty event queue.
    pub fn new() -> Self {
        Self { events: Vec::new() }
    }

    /// Push a destruction event into the queue.
    pub fn push(&mut self, event: DestructionEvent) {
        self.events.push(event);
    }

    /// Drain all pending events, returning them as a `Vec`.
    pub fn drain(&mut self) -> Vec<DestructionEvent> {
        std::mem::take(&mut self.events)
    }

    /// Number of pending events.
    pub fn len(&self) -> usize {
        self.events.len()
    }

    /// Whether the queue is empty.
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn test_event() -> DestructionEvent {
        DestructionEvent {
            center: Vec3::new(5.0, 3.0, 1.0),
            radius: 2.0,
            material_id: 7,
            force: 10.0,
        }
    }

    #[test]
    fn test_generate_debris_count() {
        let event = test_event();
        let config = DebrisConfig {
            max_debris_count: 30,
            ..Default::default()
        };
        let debris = generate_debris(&event, &config, 42);
        assert_eq!(debris.len(), 30, "should produce exactly max_debris_count");
    }

    #[test]
    fn test_debris_outward_velocity() {
        let event = test_event();
        let config = DebrisConfig::default();
        let debris = generate_debris(&event, &config, 123);

        // Each debris particle's velocity should point roughly away from center
        for (i, d) in debris.iter().enumerate() {
            let to_particle = d.position - event.center;
            if to_particle.length() < 1e-4 {
                continue; // at center, direction undefined
            }
            let dot = d.velocity.normalize_or_zero().dot(to_particle.normalize_or_zero());
            assert!(
                dot > 0.5,
                "debris[{i}] velocity should point outward: dot={dot}, vel={:?}, offset={:?}",
                d.velocity,
                to_particle
            );
        }
    }

    #[test]
    fn test_debris_size_range() {
        let event = test_event();
        let config = DebrisConfig {
            debris_size_min: 0.05,
            debris_size_max: 0.10,
            ..Default::default()
        };
        let debris = generate_debris(&event, &config, 99);

        for (i, d) in debris.iter().enumerate() {
            assert!(
                d.size >= config.debris_size_min && d.size <= config.debris_size_max,
                "debris[{i}] size {} out of range [{}, {}]",
                d.size,
                config.debris_size_min,
                config.debris_size_max
            );
        }
    }

    #[test]
    fn test_debris_material_preserved() {
        let event = test_event();
        let config = DebrisConfig::default();
        let debris = generate_debris(&event, &config, 0);

        for (i, d) in debris.iter().enumerate() {
            assert_eq!(
                d.material_id, event.material_id,
                "debris[{i}] material_id should match event"
            );
        }
    }

    #[test]
    fn test_event_queue() {
        let mut queue = DebrisEventQueue::new();
        assert!(queue.is_empty());
        assert_eq!(queue.len(), 0);

        queue.push(test_event());
        queue.push(DestructionEvent {
            center: Vec3::ZERO,
            radius: 1.0,
            material_id: 2,
            force: 5.0,
        });

        assert_eq!(queue.len(), 2);
        assert!(!queue.is_empty());

        let drained = queue.drain();
        assert_eq!(drained.len(), 2);
        assert!(queue.is_empty());
        assert_eq!(queue.len(), 0);

        // Draining again yields nothing
        let drained2 = queue.drain();
        assert!(drained2.is_empty());
    }

    #[test]
    fn test_debris_force_scaling() {
        let low_force_event = DestructionEvent {
            center: Vec3::ZERO,
            radius: 1.0,
            material_id: 0,
            force: 1.0,
        };
        let high_force_event = DestructionEvent {
            center: Vec3::ZERO,
            radius: 1.0,
            material_id: 0,
            force: 100.0,
        };
        let config = DebrisConfig {
            max_debris_count: 20,
            ..Default::default()
        };

        let low_debris = generate_debris(&low_force_event, &config, 77);
        let high_debris = generate_debris(&high_force_event, &config, 77);

        // Average speed should be much higher for the high-force event
        let avg_low: f32 = low_debris.iter().map(|d| d.velocity.length()).sum::<f32>()
            / low_debris.len() as f32;
        let avg_high: f32 = high_debris.iter().map(|d| d.velocity.length()).sum::<f32>()
            / high_debris.len() as f32;

        assert!(
            avg_high > avg_low * 10.0,
            "high force should produce faster debris: avg_low={avg_low}, avg_high={avg_high}"
        );
    }
}
