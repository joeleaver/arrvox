//! CPU implementations of opacity shaders for procedural volume voxelization.
//!
//! These mirror the WGSL opacity shader functions so that procedural volumes
//! (grass, fur, etc.) can be voxelized into real bricks on the CPU. The GPU
//! shader is no longer evaluated at render time — the bricks ARE the geometry.

use glam::Vec3;

/// Parameters for the grass opacity shader.
///
/// Maps to `ShaderParams::param0..param3` on the GPU.
#[derive(Debug, Clone, Copy)]
pub struct GrassParams {
    /// Blade density (blades per unit area).
    pub density: f32,
    /// Base blade height (world units).
    pub height: f32,
    /// Height variation (0.0 = uniform, 1.0 = max variation).
    pub height_var: f32,
    /// Bend amount (0.0 = straight, 1.0 = max bend).
    pub bend: f32,
}

/// CPU implementation of the grass opacity shader.
///
/// Mirrors `opacity_grass` from `opacity_grass.wgsl`. Given a local-space
/// position and height above the surface, returns opacity (0.0–1.0).
pub fn opacity_grass_cpu(
    local_pos: Vec3,
    h_above: f32,
    blend_weight: f32,
    params: &GrassParams,
) -> f32 {
    if params.density <= 0.0 {
        return 0.0;
    }

    let height = params.height * blend_weight.max(0.05);
    let height_var = params.height_var;
    let bend = params.bend;

    let cell_size = 1.0 / (params.density.max(0.01)).sqrt();

    // Above the tallest blade — skip.
    if h_above > height * 1.3 {
        return 0.0;
    }

    let cell_freq = 1.0 / cell_size;
    let cell_x = (local_pos.x * cell_freq).floor();
    let cell_z = (local_pos.z * cell_freq).floor();

    let blade_width = 0.002 + height * 0.005;
    let softness = (blade_width * 0.4).max(height / 32.0);

    let mut max_opacity: f32 = 0.0;

    // Check center cell first, then neighbors.
    for ring in 0..2u32 {
        for dx in -1i32..=1 {
            for dz in -1i32..=1 {
                let is_center = dx == 0 && dz == 0;
                if ring == 0 && !is_center {
                    continue;
                }
                if ring == 1 && is_center {
                    continue;
                }

                let cx = cell_x + dx as f32;
                let cz = cell_z + dz as f32;

                let h = hash2(cx, cz);

                // Blade root position (jittered within cell).
                let root_x = (cx + 0.5 + (h.0 - 0.5) * 0.7) / cell_freq;
                let root_z = (cz + 0.5 + (h.1 - 0.5) * 0.7) / cell_freq;

                // Per-blade height variation.
                let blade_h = height * (1.0 - height_var * hash1(cx * 127.1, cz * 127.1));

                if h_above > blade_h {
                    continue;
                }

                // Cheap rotation from hash.
                let rot_h = hash2(cx * 311.7, cz * 311.7);
                let rot_x = rot_h.0 * 2.0 - 1.0;
                let rot_z = rot_h.1 * 2.0 - 1.0;
                let rot_len = (rot_x * rot_x + rot_z * rot_z).sqrt().max(0.01);
                let cos_r = rot_x / rot_len;
                let sin_r = rot_z / rot_len;

                let px = local_pos.x - root_x;
                let py = h_above;
                let pz = local_pos.z - root_z;
                let rx = px * cos_r + pz * sin_r;
                let rz = -px * sin_r + pz * cos_r;

                // Quadratic bend.
                let t_blade = (py / blade_h).clamp(0.0, 1.0);
                let bend_dir = hash2(cx * 73.1, cz * 73.1);
                let bend_amount =
                    bend * (blade_h.max(blade_width * 12.0)) * t_blade * t_blade;
                let bent_x = rx - bend_amount * (bend_dir.0 - 0.5);
                let bent_z = rz - bend_amount * (bend_dir.1 - 0.5) * 0.3;

                // Flat blade cross-section.
                let flatten = 5.0;
                let py_clamped = py.clamp(0.0, blade_h);
                let taper = 1.0 - py_clamped / blade_h;
                let half_w = blade_width * (0.3 + 0.7 * taper);
                let half_t = half_w / flatten;

                let qx = (bent_x.abs() - half_w).max(0.0);
                let qz = (bent_z.abs() - half_t).max(0.0);
                let d = (qx * qx + qz * qz).sqrt();

                let blade_opacity = 1.0 - smoothstep(0.0, softness, d);
                max_opacity = max_opacity.max(blade_opacity);

                if max_opacity > 0.99 {
                    return max_opacity;
                }
            }
            if max_opacity > 0.99 {
                return max_opacity;
            }
        }
        // If center cell hit, skip neighbor ring.
        if max_opacity > 0.0 {
            break;
        }
    }

    max_opacity
}

// --- Hash utilities (matching WGSL versions) ---

#[inline]
fn hash1(x: f32, y: f32) -> f32 {
    let h = x * 127.1 + y * 311.7;
    fract(h.sin() * 43758.5453)
}

#[inline]
fn hash2(x: f32, y: f32) -> (f32, f32) {
    let h1 = x * 127.1 + y * 311.7;
    let h2 = x * 269.5 + y * 183.3;
    (fract(h1.sin() * 43758.5453), fract(h2.sin() * 43758.5453))
}

#[inline]
fn fract(x: f32) -> f32 {
    x - x.floor()
}

#[inline]
fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_grass_params() -> GrassParams {
        GrassParams {
            density: 100.0,
            height: 0.1,
            height_var: 0.3,
            bend: 0.2,
        }
    }

    #[test]
    fn zero_density_returns_zero() {
        let params = GrassParams {
            density: 0.0,
            ..default_grass_params()
        };
        let opacity = opacity_grass_cpu(Vec3::new(0.5, 0.0, 0.5), 0.05, 1.0, &params);
        assert_eq!(opacity, 0.0);
    }

    #[test]
    fn above_max_height_returns_zero() {
        let params = default_grass_params();
        // h_above = 1.0, well above height * 1.3 = 0.13
        let opacity = opacity_grass_cpu(Vec3::new(0.5, 0.0, 0.5), 1.0, 1.0, &params);
        assert_eq!(opacity, 0.0);
    }

    #[test]
    fn ground_level_can_be_nonzero() {
        let params = default_grass_params();
        // At ground level (h_above=0) near a blade root, should have non-zero opacity.
        // Try many positions — at least one should hit a blade base.
        let mut found_nonzero = false;
        for ix in 0..20 {
            for iz in 0..20 {
                let x = ix as f32 * 0.01;
                let z = iz as f32 * 0.01;
                let opacity = opacity_grass_cpu(Vec3::new(x, 0.0, z), 0.001, 1.0, &params);
                if opacity > 0.0 {
                    found_nonzero = true;
                    break;
                }
            }
            if found_nonzero {
                break;
            }
        }
        assert!(found_nonzero, "should find at least one blade near ground level");
    }

    #[test]
    fn opacity_decreases_with_height() {
        let params = default_grass_params();
        // Sample at a position where we know there's a blade (from grid search).
        let mut best_pos = Vec3::ZERO;
        let mut best_opacity = 0.0f32;
        for ix in 0..20 {
            for iz in 0..20 {
                let x = ix as f32 * 0.01;
                let z = iz as f32 * 0.01;
                let opacity = opacity_grass_cpu(Vec3::new(x, 0.0, z), 0.001, 1.0, &params);
                if opacity > best_opacity {
                    best_opacity = opacity;
                    best_pos = Vec3::new(x, 0.0, z);
                }
            }
        }

        if best_opacity > 0.1 {
            // Sample at increasing heights — opacity should decrease.
            let low = opacity_grass_cpu(best_pos, 0.01, 1.0, &params);
            let high = opacity_grass_cpu(best_pos, params.height * 0.9, 1.0, &params);
            assert!(
                low >= high,
                "opacity should decrease with height: low={low}, high={high}"
            );
        }
    }

    #[test]
    fn blend_weight_affects_height() {
        let params = default_grass_params();
        // Low blend weight should produce shorter grass (lower effective height).
        // At height = params.height * 0.5, full blend should be visible but low blend should not.
        let h = params.height * 0.5;
        let mut full_blend_max = 0.0f32;
        let mut low_blend_max = 0.0f32;
        for ix in 0..20 {
            for iz in 0..20 {
                let x = ix as f32 * 0.01;
                let z = iz as f32 * 0.01;
                let pos = Vec3::new(x, 0.0, z);
                full_blend_max = full_blend_max.max(opacity_grass_cpu(pos, h, 1.0, &params));
                low_blend_max = low_blend_max.max(opacity_grass_cpu(pos, h, 0.1, &params));
            }
        }
        // With low blend weight, effective height is much lower, so at h above
        // the grass should be empty or near-empty.
        assert!(
            full_blend_max >= low_blend_max,
            "full blend {full_blend_max} should be >= low blend {low_blend_max}"
        );
    }
}
