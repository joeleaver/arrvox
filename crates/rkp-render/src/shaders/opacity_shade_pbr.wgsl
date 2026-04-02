// PBR shading model — full physically-based rendering with Cook-Torrance GGX.
//
// Evaluates all lights (directional, point, spot) with SDF soft shadows,
// ambient occlusion, subsurface scattering, GI via voxel cone tracing,
// atmospheric ambient, and contact shadows.

// Shadow density is now shade_uniforms.shadow_params.y

fn shade_pbr(ctx: ShadingContext) -> vec3<f32> {
    var total_diffuse = vec3<f32>(0.0);
    var total_specular = vec3<f32>(0.0);
    var sss_total = vec3<f32>(0.0);
    var shadow_count = 0u;
    let shadow_budget = shade_uniforms.shadow_budget_k;

    // Read precomputed shadow + AO from half-res texture
    let shadow_ao = textureLoad(shadow_ao_tex, vec2<i32>(ctx.pixel), 0);
    let precomputed_shadow = shadow_ao.r;
    let precomputed_ao = shadow_ao.g;
    var used_precomputed_shadow = false;

    // Pre-compute SSS thickness once (saves one sample_sdf per additional light)
    let has_sss = ctx.subsurface > 0.0;
    var thickness = 0.0;
    if has_sss {
        thickness = sss_thickness(ctx.world_pos, ctx.normal);
    }

    // Iterate all lights
    let num_lights = shade_uniforms.num_lights;
    for (var li = 0u; li < num_lights; li++) {
        let light = lights[li];
        let light_color = vec3<f32>(light.color_r, light.color_g, light.color_b);
        let radiance = light_color * light.intensity;

        // Compute light direction and attenuation based on light type
        var light_dir: vec3<f32>;
        var atten = 1.0;
        var shadow_max = SHADOW_MAX_DIST;

        if light.light_type == LIGHT_TYPE_DIRECTIONAL {
            light_dir = normalize(vec3<f32>(light.dir_x, light.dir_y, light.dir_z));
        } else {
            // Point and spot lights — camera-relative positions
            let light_pos = vec3<f32>(light.pos_x, light.pos_y, light.pos_z) + shade_uniforms.camera_pos.xyz;
            let to_light = light_pos - ctx.world_pos;
            let dist = length(to_light);
            light_dir = to_light / max(dist, 0.0001);
            shadow_max = min(dist, SHADOW_MAX_DIST);

            atten = distance_attenuation(dist, light.range);

            // Spot cone falloff
            if light.light_type == LIGHT_TYPE_SPOT {
                let spot_dir = normalize(vec3<f32>(light.dir_x, light.dir_y, light.dir_z));
                let cos_angle = dot(-light_dir, spot_dir);
                let cos_outer = cos(light.outer_angle);
                let cos_inner = cos(light.inner_angle);
                let spot = clamp((cos_angle - cos_outer) / max(cos_inner - cos_outer, 0.0001), 0.0, 1.0);
                atten *= spot;
            }
        }

        if atten < 0.001 {
            continue;
        }

        let n_dot_l = max(dot(ctx.normal, light_dir), 0.0);

        // Skip if light is behind surface and no SSS
        if n_dot_l <= 0.0 && !has_sss {
            continue;
        }

        // Evaluate BRDF
        let brdf = evaluate_brdf(light_dir, ctx);

        // Shadow
        var shadow = 1.0;
        if n_dot_l > 0.0 && light.shadow_caster == 1u && (shadow_budget == 0u || shadow_count < shadow_budget) {
            if !used_precomputed_shadow {
                shadow = precomputed_shadow;
                used_precomputed_shadow = true;
            } else {
                let shadow_origin = ctx.world_pos + ctx.normal * SHADOW_BIAS + light_dir * SHADOW_BIAS * 0.5;
                shadow = compute_shadow(shadow_origin, light_dir, shadow_max);
            }
            shadow_count += 1u;
        }
        shadow = max(shadow, shade_uniforms.shadow_params.y);
        shadow = mix(shadow, 1.0, ctx.atmo_shadow_fill);

        let attenuated_radiance = radiance * atten;
        total_diffuse += brdf.diffuse * attenuated_radiance * n_dot_l * shadow;
        total_specular += brdf.specular * attenuated_radiance * n_dot_l * shadow;

        // SSS with cached thickness
        if has_sss {
            sss_total += sss_from_thickness(thickness, ctx.normal, light_dir, ctx.subsurface, ctx.sss_color)
                         * attenuated_radiance * shadow;
        }
    }

    // Read AO from precomputed half-res texture
    let ao = precomputed_ao;

    // GI via voxel cone tracing (4 diffuse + 1 specular cone).
    let gi_origin = ctx.world_pos + ctx.normal * SHADOW_BIAS * 2.0;
    let gi_diffuse_raw = cone_trace_diffuse(gi_origin, ctx.normal, ctx.jitter);
    let kd_gi = (1.0 - ctx.metallic);
    let gi_diffuse = gi_diffuse_raw * ctx.albedo * kd_gi * ao * shade_uniforms.sun_color.w;

    let gi_specular_raw = cone_trace_specular(gi_origin, ctx.reflect_dir, ctx.roughness, ctx.jitter);
    let gi_fresnel = fresnel_schlick(ctx.n_dot_v, ctx.f0);
    let gi_specular = gi_specular_raw * gi_fresnel * ao * shade_uniforms.sun_color.w;

    // Ambient sky illumination — hemisphere of sky light filling shadowed areas.
    // The hemisphere integral of radiance introduces a factor of pi which cancels
    // the Lambertian BRDF's 1/pi, so the final reflected ambient is simply: albedo x L_avg.
    var ambient_radiance: vec3<f32>;
    var ambient_reflect_color: vec3<f32>;
    if shade_uniforms.sky_params.z > 0.5 {
        // Use precomputed hemisphere average from CPU (saves 3x atmosphere_sky per pixel).
        ambient_radiance = shade_uniforms.ambient_sky.xyz;
        // Per-pixel reflection still needs the per-pixel direction.
        let sun_d = normalize(shade_uniforms.sun_dir.xyz);
        let reflect_env = reflect(-ctx.view_dir, ctx.normal);
        ambient_reflect_color = atmosphere_sky(reflect_env, sun_d);
    } else {
        // Fallback: sky-colored ambient for non-atmosphere scenes
        ambient_radiance = mix(SKY_HORIZON, SKY_ZENITH, 0.5) * 0.15;
        let reflect_env = reflect(-ctx.view_dir, ctx.normal);
        let sky_up_frac = clamp(reflect_env.y * 0.5 + 0.5, 0.0, 1.0);
        ambient_reflect_color = mix(SKY_HORIZON, SKY_ZENITH, sky_up_frac);
    }
    let kd_ambient = 1.0 - ctx.metallic;
    // Sky ambient uses softened AO — distant sky light is less affected by local
    // geometry occlusion than nearby GI bounces. This prevents shadows from going
    // fully black even in occluded areas.
    let sky_ao = mix(1.0, ao, 0.5);
    let ambient_diffuse = ambient_radiance * ctx.albedo * sky_ao * kd_ambient;
    let ambient_fresnel = fresnel_schlick(ctx.n_dot_v, ctx.f0);
    let ambient_specular = ambient_reflect_color * ambient_fresnel * ao * SKY_REFLECT_STRENGTH;
    let ambient = ambient_diffuse + ambient_specular;

    // Final color = direct + SSS + GI + ambient + emission
    // Contact shadow only darkens direct lighting — ambient and GI provide fill
    // light at surface junctions regardless of contact occlusion.
    let emission = ctx.emission * ctx.emission_strength;
    let direct = (total_diffuse + total_specular) * ctx.contact;
    let indirect = gi_diffuse + gi_specular + ambient;
    return direct + sss_total + indirect + emission;
}
