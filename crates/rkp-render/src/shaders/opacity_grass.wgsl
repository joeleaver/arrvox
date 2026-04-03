// Opacity grass shader — procedural grass blades via domain repetition.
//
// Returns vec2(opacity, skip_hint) at a point in object-local space.
// opacity: 0.0 = empty, 1.0 = solid.
// skip_hint: when opacity is 0, how far along the ray the march can safely skip.
//
// Injected into splat_march.wgsl by ShaderComposer. Paired with shade_grass.wgsl.

// --- Hash utilities (self-contained, no external dependencies) ---

fn grass_hash1(p: vec2<f32>) -> f32 {
    var h = dot(p, vec2<f32>(127.1, 311.7));
    return fract(sin(h) * 43758.5453);
}

fn grass_hash2(p: vec2<f32>) -> vec2<f32> {
    let h = vec2<f32>(
        dot(p, vec2<f32>(127.1, 311.7)),
        dot(p, vec2<f32>(269.5, 183.3)),
    );
    return fract(sin(h) * 43758.5453);
}

// --- Grass blade opacity ---

fn opacity_grass(local_pos: vec3<f32>, h_above: f32, blend_weight: f32, obj: GpuObject, mat_id: u32) -> f32 {

    // Only grow grass above the surface
    if h_above < 0.0 {
        return 0.0; // skip to surface
    }

    // Read shader-specific params
    let sp = shader_params[mat_id];
    let density = sp.param0;
    if density <= 0.0 {
        return 0.0;
    }

    let height = sp.param1 * max(blend_weight, 0.05);
    let height_var = sp.param2;
    let bend = sp.param3;

    let cell_size = 1.0 / sqrt(max(density, 0.01));

    // Above the tallest blade — skip to exit
    if h_above > height * 1.3 {
        return 0.0;
    }

    // Domain repetition in XZ
    let cell_freq = 1.0 / cell_size;
    let cell = floor(local_pos.xz * cell_freq);

    // Realistic blade width
    let blade_width = 0.002 + height * 0.005;
    let softness = max(blade_width * 0.4, height / 32.0);

    var max_opacity = 0.0;

    // Check center cell first, then neighbors only if needed.
    // This avoids 8 unnecessary blade evaluations for most positions.
    for (var ring = 0u; ring < 2u; ring++) {
    for (var dx = -1i; dx <= 1i; dx++) {
        for (var dz = -1i; dz <= 1i; dz++) {
            // Ring 0: center only. Ring 1: neighbors only.
            let is_center = (dx == 0 && dz == 0);
            if ring == 0u && !is_center { continue; }
            if ring == 1u && is_center { continue; }
            let c = cell + vec2<f32>(f32(dx), f32(dz));
            let h = grass_hash2(c);

            // Blade root position (jittered within cell)
            let root_xz = (c + 0.5 + (h - 0.5) * 0.7) / cell_freq;

            // Per-blade height variation
            let blade_h = height * (1.0 - height_var * grass_hash1(c * 127.1));

            // Skip if above this blade
            if h_above > blade_h {
                continue;
            }

            // Cheap rotation: hash-derived direction vector (avoids sin/cos)
            let rot_h = grass_hash2(c * 311.7) * 2.0 - 1.0;
            let rot_len = max(length(rot_h), 0.01);
            let cos_r = rot_h.x / rot_len;
            let sin_r = rot_h.y / rot_len;

            var p = vec3<f32>(local_pos.x - root_xz.x, h_above, local_pos.z - root_xz.y);
            let rx = p.x * cos_r + p.z * sin_r;
            let rz = -p.x * sin_r + p.z * cos_r;
            p = vec3<f32>(rx, p.y, rz);

            // Quadratic bend
            let t_blade = saturate(p.y / blade_h);
            let bend_dir = grass_hash2(c * 73.1) - 0.5;
            let bend_amount = bend * max(blade_h, blade_width * 12.0) * t_blade * t_blade;
            p.x -= bend_amount * bend_dir.x;
            p.z -= bend_amount * bend_dir.y * 0.3;

            // Flat blade cross-section
            let flatten = 5.0;
            let py = clamp(p.y, 0.0, blade_h);
            let taper = 1.0 - py / blade_h;
            let half_w = blade_width * (0.3 + 0.7 * taper);
            let half_t = half_w / flatten;

            let qx = max(abs(p.x) - half_w, 0.0);
            let qz = max(abs(p.z) - half_t, 0.0);
            let d = sqrt(qx * qx + qz * qz);

            let blade_opacity = 1.0 - smoothstep(0.0, softness, d);
            max_opacity = max(max_opacity, blade_opacity);
            // Early-out: if we found a solid hit, no need to check more blades.
            if max_opacity > 0.99 { break; }
        }
        if max_opacity > 0.99 { break; }
    }
    // If center cell hit, skip neighbor ring entirely.
    if max_opacity > 0.0 { break; }
    }

    return max_opacity;
}
