// Procedural RPN evaluator — shared across the raymarch preview and
// (Phase 2 on) the GPU voxel-bake compute shader. Contains every bit
// of math that turns a flattened node stream + a point in space into a
// `TreeSample`.
//
// Types / constants live in `proc_eval_types.wgsl` and are concatenated
// before this file at pipeline creation. The `instructions` storage
// buffer is declared by the caller shader (with whatever @group /
// @binding its bind-group layout uses); this module references it by
// name as a module-scope global.

// ── Primitive SDFs ─────────────────────────────────────────────────────
// Each mirrors its CPU counterpart in `rkp_procedural::leaves`.

fn sdf_sphere(p: vec3<f32>, radius: f32) -> f32 {
    return length(p) - radius;
}

fn sdf_box(p: vec3<f32>, half_extents: vec3<f32>, rounding: f32) -> f32 {
    let q = abs(p) - half_extents + vec3<f32>(rounding);
    let outside = length(max(q, vec3<f32>(0.0)));
    let inside = min(max(q.x, max(q.y, q.z)), 0.0);
    return outside + inside - rounding;
}

fn sdf_capsule(p: vec3<f32>, radius: f32, half_height: f32) -> f32 {
    let t = clamp(p.y, -half_height, half_height);
    let closest = vec3<f32>(0.0, t, 0.0);
    return length(p - closest) - radius;
}

fn sdf_cylinder(p: vec3<f32>, radius: f32, half_height: f32) -> f32 {
    let radial = length(vec3<f32>(p.x, 0.0, p.z)) - radius;
    let axial = abs(p.y) - half_height;
    // Keep the CPU branching exactly (`leaves::eval_cylinder`): when
    // the point is outside both cylinder and caps, distance is the
    // diagonal length; otherwise it's `max` of the signed axes.
    if (radial > 0.0 && axial > 0.0) {
        return sqrt(radial * radial + axial * axial);
    }
    return max(radial, axial);
}

fn sdf_torus(p: vec3<f32>, major_radius: f32, minor_radius: f32) -> f32 {
    let xz_len = length(vec3<f32>(p.x, 0.0, p.z));
    let q = vec3<f32>(xz_len - major_radius, p.y, 0.0);
    return length(q) - minor_radius;
}

fn sdf_plane(p: vec3<f32>) -> f32 {
    return p.y;
}

fn sdf_ramp(p: vec3<f32>, half_length: f32, half_height: f32, half_width: f32) -> f32 {
    let q = abs(p) - vec3<f32>(half_length, half_height, half_width);
    let outside = length(max(q, vec3<f32>(0.0)));
    let inside = min(max(q.x, max(q.y, q.z)), 0.0);
    let box_dist = outside + inside;
    let hyp = max(sqrt(half_length * half_length + half_height * half_height), 1e-6);
    let plane_dist = (half_length * p.y - half_height * p.x) / hyp;
    return max(box_dist, plane_dist);
}

fn eval_primitive(ins: ProcInstruction, world_pos: vec3<f32>) -> TreeSample {
    let local4 = ins.inverse_world * vec4<f32>(world_pos, 1.0);
    let local = local4.xyz;
    var d: f32 = 1e30;
    switch ins.op {
        case 0u: { d = sdf_sphere(local, ins.params_lo.x); }
        case 1u: { d = sdf_box(local, ins.params_lo.xyz, ins.params_lo.w); }
        case 2u: { d = sdf_capsule(local, ins.params_lo.x, ins.params_lo.y); }
        case 3u: { d = sdf_cylinder(local, ins.params_lo.x, ins.params_lo.y); }
        case 4u: { d = sdf_torus(local, ins.params_lo.x, ins.params_lo.y); }
        case 5u: { d = sdf_plane(local); }
        case 6u: { d = sdf_ramp(local, ins.params_lo.x, ins.params_lo.y, ins.params_lo.z); }
        default: { d = 1e30; }
    }
    // Convert the primitive's LOCAL distance to WORLD distance. See
    // `distance_scale` docs in `proc_eval_types.wgsl` /
    // `ProcInstruction`. All downstream combinators + the classifier
    // expect world-space distances.
    d = d * ins.distance_scale;
    var s: TreeSample;
    s.distance = d;
    s.material_id = ins.material_id;
    // Match the CPU `Sample::with_color` convention: leaves initialize
    // secondary = 0 with blend = 0. Post-op effects (by-height, by-noise)
    // overwrite both fields when they fire. Live rendering doesn't
    // notice either way (rkp_shade guards on `blend_weight > 0`), but
    // the bake's LeafAttr dedup hashes on the full packed secondary,
    // so CPU and GPU must agree bit-for-bit.
    s.secondary_material_id = 0u;
    s.blend_weight = 0.0;
    s.color = ins.color.xyz;
    s.node_id = ins.node_id;
    return s;
}

// ── Combinators ────────────────────────────────────────────────────────

// Blend mode: smooth the material/color transition across a band of
// width `radius` where the two samples have equal distance. Outside
// the band falls back to Winner. Geometry stays sharp min/max.
fn blended_union_sample(a: TreeSample, b: TreeSample, radius: f32) -> TreeSample {
    let distance = min(a.distance, b.distance);
    let diff = abs(a.distance - b.distance);
    let r = max(radius, 1e-6);
    if (diff >= r) {
        if (a.distance <= b.distance) { return a; }
        return b;
    }
    // t=0 → fully b, t=1 → fully a (matches combine.rs's convention).
    let t = 0.5 + 0.5 * (b.distance - a.distance) / r;
    let winner_is_a = a.distance <= b.distance;
    var s: TreeSample;
    s.distance = distance;
    s.material_id = select(b.material_id, a.material_id, winner_is_a);
    s.secondary_material_id = select(b.secondary_material_id, a.secondary_material_id, winner_is_a);
    s.blend_weight = select(b.blend_weight, a.blend_weight, winner_is_a);
    s.color = mix(b.color, a.color, t);
    s.node_id = select(b.node_id, a.node_id, winner_is_a);
    return s;
}

fn combine_union(a: TreeSample, b: TreeSample, mat_mode: u32, radius: f32) -> TreeSample {
    if (mat_mode == MAT_COMBINE_BLEND) {
        return blended_union_sample(a, b, radius);
    }
    if (a.distance <= b.distance) { return a; }
    return b;
}

fn combine_intersect(a: TreeSample, b: TreeSample) -> TreeSample {
    // Max of distances; material from the one with the larger distance —
    // that's the boundary that defines the intersect surface.
    if (a.distance >= b.distance) { return a; }
    return b;
}

fn combine_subtract(a: TreeSample, b: TreeSample) -> TreeSample {
    // Subtract: max(a, -b). Material always from `a` — cutters don't
    // contribute geometry you can click on.
    let neg_b = -b.distance;
    if (a.distance >= neg_b) {
        return a;
    }
    var r: TreeSample;
    r.distance = neg_b;
    r.material_id = a.material_id;
    r.secondary_material_id = a.secondary_material_id;
    r.blend_weight = a.blend_weight;
    r.color = a.color;
    r.node_id = a.node_id;
    return r;
}

// ── Height-band classifier ─────────────────────────────────────────────
// Mirror of `rkp_procedural::node_kind::classify_bands`. Returns two
// adjacent band indices and a smoothstep blend alpha between them
// (alpha=0 → fully `lower`; alpha=1 → fully `upper`).

fn classify_bands(
    y: f32,
    low_to_mid: f32,
    mid_to_high: f32,
    transition_width: f32,
) -> HeightClassify {
    let w_half = max(transition_width * 0.5, 1e-6);
    if (y < low_to_mid - w_half) {
        return HeightClassify(0u, 0u, 0.0);
    }
    if (y < low_to_mid + w_half) {
        let t = clamp((y - (low_to_mid - w_half)) / (2.0 * w_half), 0.0, 1.0);
        return HeightClassify(0u, 1u, t * t * (3.0 - 2.0 * t));
    }
    if (y < mid_to_high - w_half) {
        return HeightClassify(1u, 1u, 0.0);
    }
    if (y < mid_to_high + w_half) {
        let t = clamp((y - (mid_to_high - w_half)) / (2.0 * w_half), 0.0, 1.0);
        return HeightClassify(1u, 2u, t * t * (3.0 - 2.0 * t));
    }
    return HeightClassify(2u, 2u, 0.0);
}

// ── Noise (port of `crates/rkp-procedural/src/noise.rs`) ──────────────
// Keep byte-for-byte equivalent to the CPU side (until Phase 4 removes
// the CPU copy) so a bake run and a live preview produce identical
// displaced geometry. WGSL u32 ops wrap by default — same semantics
// as Rust's `wrapping_*`.

fn rkp_hash_f32(x: u32) -> f32 {
    var n = x;
    n = (n ^ 61u) ^ (n >> 16u);
    n = n * 9u;
    n = n ^ (n >> 4u);
    n = n * 0x27d4eb2du;
    n = n ^ (n >> 15u);
    return f32(n & 0x00ffffffu) * (1.0 / 16777216.0) * 2.0 - 1.0;
}

fn rkp_hash_3i(ix: i32, iy: i32, iz: i32, seed: u32) -> f32 {
    let k = u32(ix) * 0x9e3779b9u
          + u32(iy) * 0x7ed55d16u
          + u32(iz) * 0xa3a52d49u
          + seed;
    return rkp_hash_f32(k);
}

fn rkp_smootherstep(t: f32) -> f32 {
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}

fn rkp_noise_3d(pos: vec3<f32>, seed: u32) -> f32 {
    let xf = floor(pos.x);
    let yf = floor(pos.y);
    let zf = floor(pos.z);
    let ix = i32(xf);
    let iy = i32(yf);
    let iz = i32(zf);
    let tx = rkp_smootherstep(pos.x - xf);
    let ty = rkp_smootherstep(pos.y - yf);
    let tz = rkp_smootherstep(pos.z - zf);
    let c000 = rkp_hash_3i(ix,       iy,       iz,       seed);
    let c100 = rkp_hash_3i(ix + 1,   iy,       iz,       seed);
    let c010 = rkp_hash_3i(ix,       iy + 1,   iz,       seed);
    let c110 = rkp_hash_3i(ix + 1,   iy + 1,   iz,       seed);
    let c001 = rkp_hash_3i(ix,       iy,       iz + 1,   seed);
    let c101 = rkp_hash_3i(ix + 1,   iy,       iz + 1,   seed);
    let c011 = rkp_hash_3i(ix,       iy + 1,   iz + 1,   seed);
    let c111 = rkp_hash_3i(ix + 1,   iy + 1,   iz + 1,   seed);
    let x00 = c000 + (c100 - c000) * tx;
    let x10 = c010 + (c110 - c010) * tx;
    let x01 = c001 + (c101 - c001) * tx;
    let x11 = c011 + (c111 - c011) * tx;
    let y0 = x00 + (x10 - x00) * ty;
    let y1 = x01 + (x11 - x01) * ty;
    return y0 + (y1 - y0) * tz;
}

fn rkp_noise_3d_vec(pos: vec3<f32>, seed: u32) -> vec3<f32> {
    return vec3<f32>(
        rkp_noise_3d(pos, seed),
        rkp_noise_3d(pos, seed + 0x9e3779b1u),
        rkp_noise_3d(pos, seed + 0xb74684abu),
    );
}

fn rkp_fbm_3d_vec(pos: vec3<f32>, frequency: f32, seed: u32, octaves_in: u32) -> vec3<f32> {
    let octaves = clamp(octaves_in, 1u, 8u);
    var sum = vec3<f32>(0.0);
    var amp = 1.0;
    var freq = max(frequency, 1e-6);
    var total_amp = 0.0;
    for (var k: u32 = 0u; k < octaves; k = k + 1u) {
        sum = sum + rkp_noise_3d_vec(pos * freq, seed + k * 131u) * amp;
        total_amp = total_amp + amp;
        amp = amp * 0.5;
        freq = freq * 2.0;
    }
    return sum / max(total_amp, 1e-6);
}

// Scalar FBM — port of `rkp_procedural::noise::fbm_3d_scalar`.
fn rkp_fbm_3d_scalar(pos: vec3<f32>, frequency: f32, seed: u32, octaves_in: u32) -> f32 {
    let octaves = clamp(octaves_in, 1u, 8u);
    var sum = 0.0;
    var amp = 1.0;
    var freq = max(frequency, 1e-6);
    var total_amp = 0.0;
    for (var k: u32 = 0u; k < octaves; k = k + 1u) {
        sum = sum + rkp_noise_3d(pos * freq, seed + k * 131u) * amp;
        total_amp = total_amp + amp;
        amp = amp * 0.5;
        freq = freq * 2.0;
    }
    return sum / max(total_amp, 1e-6);
}

// Unpack an RGB color from a u24-bit-laid f32 — mirror of the CPU
// `unpack_rgb_u24` in `flatten.rs`.
fn unpack_rgb_u24(packed: f32) -> vec3<f32> {
    let bits = bitcast<u32>(packed);
    let r = f32(bits & 0xFFu) / 255.0;
    let g = f32((bits >> 8u) & 0xFFu) / 255.0;
    let b = f32((bits >> 16u) & 0xFFu) / 255.0;
    return vec3<f32>(r, g, b);
}

// ── RPN execution ──────────────────────────────────────────────────────
// Evaluates the flattened tree at `world_pos`. Reads the caller's
// module-scope `instructions: array<ProcInstruction>` storage buffer.
// The count is passed explicitly so the same code runs from a uniform
// (raymarch) or a push-constant / different uniform (bake compute).

fn eval_tree(world_pos: vec3<f32>, count: u32) -> TreeSample {
    var stack: array<TreeSample, STACK_CAP>;
    var sp: u32 = 0u;

    // Position stack. `pos_top` indexes the current sample position;
    // `pos_stack[0]` is the outer world_pos. PUSH increments pos_top
    // and writes the warped position; POP decrements it. Primitives
    // evaluate at `pos_stack[pos_top]`.
    var pos_stack: array<vec3<f32>, POS_STACK_CAP>;
    var pos_top: u32 = 0u;
    pos_stack[0u] = world_pos;

    for (var i: u32 = 0u; i < count; i = i + 1u) {
        let ins = instructions[i];
        let op = ins.op;

        // ── Position-warp effects ──────────────────────────────────
        if (op == OP_PUSH_NOISE_DISPLACE) {
            let cur = pos_stack[pos_top];
            let amp  = ins.params_lo.x;
            let freq = ins.params_lo.y;
            let seed = u32(ins.params_lo.z);
            let oct  = u32(ins.params_lo.w);
            let warped = cur + rkp_fbm_3d_vec(cur, freq, seed, oct) * amp;
            if (pos_top + 1u < POS_STACK_CAP) {
                pos_top = pos_top + 1u;
                pos_stack[pos_top] = warped;
            }
            continue;
        }
        if (op == OP_POP_NOISE_DISPLACE) {
            if (pos_top > 0u) {
                pos_top = pos_top - 1u;
            }
            // Shrink top distance by the conservative envelope so the
            // sphere tracer stays safe — mirror of the CPU evaluator's
            // `child.distance - amp * sqrt(3)`.
            if (sp > 0u) {
                let amp = ins.params_lo.x;
                stack[sp - 1u].distance = stack[sp - 1u].distance - amp * 1.7320508;
            }
            continue;
        }
        if (op == OP_PUSH_MIRROR) {
            let cur = pos_stack[pos_top];
            let origin = ins.params_lo.xyz;
            let normal = ins.params_hi.xyz;
            // Reflect across the plane if `cur` is on the normal's
            // negative side; otherwise leave as-is.
            let d = dot(cur - origin, normal);
            let folded = cur - 2.0 * min(d, 0.0) * normal;
            if (pos_top + 1u < POS_STACK_CAP) {
                pos_top = pos_top + 1u;
                pos_stack[pos_top] = folded;
            }
            continue;
        }
        if (op == OP_POP_MIRROR) {
            if (pos_top > 0u) {
                pos_top = pos_top - 1u;
            }
            // No distance adjustment — reflection is an isometry.
            continue;
        }
        if (op == OP_PUSH_ARRAY) {
            let cur = pos_stack[pos_top];
            let x_axis = vec3<f32>(
                ins.inverse_world[0][0], ins.inverse_world[0][1], ins.inverse_world[0][2]);
            let y_axis = vec3<f32>(
                ins.inverse_world[1][0], ins.inverse_world[1][1], ins.inverse_world[1][2]);
            let z_axis = vec3<f32>(
                ins.inverse_world[2][0], ins.inverse_world[2][1], ins.inverse_world[2][2]);
            let origin = vec3<f32>(
                ins.inverse_world[3][0], ins.inverse_world[3][1], ins.inverse_world[3][2]);
            let spacing = ins.params_lo.xyz;
            let counts  = ins.params_hi.xyz;
            // `opRepLim` with per-axis centering. Cell centers along
            // each axis sit at `(i - (N-1)/2) * spacing` for i in
            // {0..N-1}; for odd N these are integer multiples of
            // spacing, for even N they're half-integer. Rounding `t`
            // to the nearest integer and clamping to `±half` works for
            // odd N but would yield non-uniform cells for even N
            // (interior round-to-int + edge clamped-to-half-int).
            // Shift-round-unshift gives the correct centers for any N
            // in constant time.
            let rel = cur - origin;
            var delta = vec3<f32>(0.0);
            let half_x = (counts.x - 1.0) * 0.5;
            let half_y = (counts.y - 1.0) * 0.5;
            let half_z = (counts.z - 1.0) * 0.5;
            if (spacing.x > 1e-6 && counts.x > 1.0) {
                let t = dot(rel, x_axis) / spacing.x;
                let k = clamp(round(t + half_x) - half_x, -half_x, half_x);
                delta = delta + k * spacing.x * x_axis;
            }
            if (spacing.y > 1e-6 && counts.y > 1.0) {
                let t = dot(rel, y_axis) / spacing.y;
                let k = clamp(round(t + half_y) - half_y, -half_y, half_y);
                delta = delta + k * spacing.y * y_axis;
            }
            if (spacing.z > 1e-6 && counts.z > 1.0) {
                let t = dot(rel, z_axis) / spacing.z;
                let k = clamp(round(t + half_z) - half_z, -half_z, half_z);
                delta = delta + k * spacing.z * z_axis;
            }
            let folded = cur - delta;
            if (pos_top + 1u < POS_STACK_CAP) {
                pos_top = pos_top + 1u;
                pos_stack[pos_top] = folded;
            }
            continue;
        }
        if (op == OP_POP_ARRAY) {
            if (pos_top > 0u) {
                pos_top = pos_top - 1u;
            }
            // No distance adjustment — the fold is a translation, so
            // distances are preserved. Matches Mirror's POP.
            continue;
        }

        // ── Attribute-rewrite post-ops ──────────────────────────────
        if (op == OP_APPLY_MATERIAL_BY_HEIGHT) {
            if (sp > 0u) {
                let cur = pos_stack[pos_top];
                let local = (ins.inverse_world * vec4<f32>(cur, 1.0)).xyz;
                let c = classify_bands(
                    local.y, ins.params_lo.x, ins.params_lo.y, ins.params_lo.z,
                );
                var mats: array<u32, 3>;
                mats[0] = u32(ins.params_lo.w);
                mats[1] = u32(ins.params_hi.x);
                mats[2] = u32(ins.params_hi.y);
                stack[sp - 1u].material_id = mats[c.lower];
                stack[sp - 1u].secondary_material_id = mats[c.upper];
                stack[sp - 1u].blend_weight = c.alpha;
            }
            continue;
        }
        if (op == OP_APPLY_COLOR_BY_HEIGHT) {
            if (sp > 0u) {
                let cur = pos_stack[pos_top];
                let local = (ins.inverse_world * vec4<f32>(cur, 1.0)).xyz;
                let low_to_mid     = ins.params_lo.x;
                let low_color      = ins.params_lo.yzw;
                let mid_to_high    = ins.params_hi.x;
                let mid_color      = ins.params_hi.yzw;
                let high_color     = ins.color.xyz;
                let transition_w   = ins.color.w;
                let c = classify_bands(
                    local.y, low_to_mid, mid_to_high, transition_w,
                );
                var colors: array<vec3<f32>, 3>;
                colors[0] = low_color;
                colors[1] = mid_color;
                colors[2] = high_color;
                stack[sp - 1u].color = mix(colors[c.lower], colors[c.upper], c.alpha);
            }
            continue;
        }
        if (op == OP_APPLY_MATERIAL_BY_NOISE) {
            if (sp > 0u) {
                let cur = pos_stack[pos_top];
                let local = (ins.inverse_world * vec4<f32>(cur, 1.0)).xyz;
                // Layout: params_lo = [t1, t2, width, freq],
                //         params_hi = [seed, oct, low_mat, mid_mat],
                //         color.x = high_mat.
                let freq = ins.params_lo.w;
                let seed = u32(ins.params_hi.x);
                let oct  = u32(ins.params_hi.y);
                let n = rkp_fbm_3d_scalar(local, freq, seed, oct);
                let c = classify_bands(
                    n, ins.params_lo.x, ins.params_lo.y, ins.params_lo.z,
                );
                var mats: array<u32, 3>;
                mats[0] = u32(ins.params_hi.z);
                mats[1] = u32(ins.params_hi.w);
                mats[2] = u32(ins.color.x);
                stack[sp - 1u].material_id = mats[c.lower];
                stack[sp - 1u].secondary_material_id = mats[c.upper];
                stack[sp - 1u].blend_weight = c.alpha;
            }
            continue;
        }
        if (op == OP_APPLY_COLOR_BY_NOISE) {
            if (sp > 0u) {
                let cur = pos_stack[pos_top];
                let local = (ins.inverse_world * vec4<f32>(cur, 1.0)).xyz;
                // Layout: params_lo = [t1, t2, width, freq],
                //         params_hi = [seed, oct, low_rgb_u24, mid_rgb_u24],
                //         color.x = high_rgb_u24.
                let freq = ins.params_lo.w;
                let seed = u32(ins.params_hi.x);
                let oct  = u32(ins.params_hi.y);
                let n = rkp_fbm_3d_scalar(local, freq, seed, oct);
                let c = classify_bands(
                    n, ins.params_lo.x, ins.params_lo.y, ins.params_lo.z,
                );
                var colors: array<vec3<f32>, 3>;
                colors[0] = unpack_rgb_u24(ins.params_hi.z);
                colors[1] = unpack_rgb_u24(ins.params_hi.w);
                colors[2] = unpack_rgb_u24(ins.color.x);
                stack[sp - 1u].color = mix(
                    colors[c.lower], colors[c.upper], c.alpha,
                );
            }
            continue;
        }

        if (op < 100u) {
            // Primitive — evaluates at the top of the position stack.
            let s = eval_primitive(ins, pos_stack[pos_top]);
            if (sp < STACK_CAP) {
                stack[sp] = s;
                sp = sp + 1u;
            }
        } else {
            // Combinator. Pop `arity`, combine, push one.
            let arity = ins.arity;
            if (arity == 0u || arity > sp) {
                continue;
            }
            let base = sp - arity;
            var acc = stack[base];
            let blend_radius = ins.params_lo.x;
            for (var k: u32 = 1u; k < arity; k = k + 1u) {
                let rhs = stack[base + k];
                switch op {
                    case 100u: { acc = combine_union(acc, rhs, ins.material_combine, blend_radius); }
                    case 101u: { acc = combine_intersect(acc, rhs); }
                    case 102u: { acc = combine_subtract(acc, rhs); }
                    default: {}
                }
            }
            stack[base] = acc;
            sp = base + 1u;
        }
    }

    if (sp == 0u) {
        var miss: TreeSample;
        miss.distance = 1e30;
        miss.material_id = 0u;
        miss.secondary_material_id = 0u;
        miss.blend_weight = 0.0;
        miss.color = vec3<f32>(0.0);
        miss.node_id = 0xFFFFFFFFu;
        return miss;
    }
    return stack[sp - 1u];
}

// ── Distance-only fast path ────────────────────────────────────────────
// Sphere-trace loops only need `distance`; they don't read material /
// color / node_id during the march. Evaluating the full `TreeSample`
// per step forces a 32-byte-per-slot `array<TreeSample, STACK_CAP>`
// into per-thread private memory, which GPUs spill into global memory
// and hit on every step of every pixel — the dominant cost on a
// raymarched scene. The distance-only path uses a 4-byte-per-slot
// `array<f32, STACK_CAP>` (64 bytes total), small enough to stay in
// registers, and skips post-op attribute rewrites (by-height /
// by-noise) entirely since they don't affect distance.
//
// The sphere-trace loop in `proc_raymarch.wgsl` uses this path for
// stepping + for the 6-tap gradient; it calls full `eval_tree` once
// at the hit point for the G-buffer sample (materials + color).
// `proc_sample.wgsl` (GPU bake) still uses full `eval_tree` because
// it needs attributes at every voxel center.

fn eval_primitive_distance(ins: ProcInstruction, world_pos: vec3<f32>) -> f32 {
    let local4 = ins.inverse_world * vec4<f32>(world_pos, 1.0);
    let local = local4.xyz;
    switch ins.op {
        case 0u: { return sdf_sphere(local, ins.params_lo.x); }
        case 1u: { return sdf_box(local, ins.params_lo.xyz, ins.params_lo.w); }
        case 2u: { return sdf_capsule(local, ins.params_lo.x, ins.params_lo.y); }
        case 3u: { return sdf_cylinder(local, ins.params_lo.x, ins.params_lo.y); }
        case 4u: { return sdf_torus(local, ins.params_lo.x, ins.params_lo.y); }
        case 5u: { return sdf_plane(local); }
        case 6u: { return sdf_ramp(local, ins.params_lo.x, ins.params_lo.y, ins.params_lo.z); }
        default: { return 1e30; }
    }
}

fn eval_tree_distance(world_pos: vec3<f32>, count: u32) -> f32 {
    // f32 stack — 64 bytes total at STACK_CAP=16. Dynamic-indexed, so
    // GPUs may demote it to local memory anyway; still far cheaper
    // than the full TreeSample stack (8× smaller) and critical for
    // chains of combinators. The pos_stack below is gated on
    // `HAS_POS_WARPS` because most BUILD-preview trees have no
    // position-warp effects and shouldn't pay that cost.
    var stack: array<f32, STACK_CAP>;
    var sp: u32 = 0u;

    // Only allocated when the tree actually has PUSH/POP effects.
    // When `HAS_POS_WARPS=false` the compiler dead-strips the entire
    // stack + all PUSH/POP branches below.
    var pos_stack: array<vec3<f32>, POS_STACK_CAP>;
    var pos_top: u32 = 0u;
    if (HAS_POS_WARPS) {
        pos_stack[0u] = world_pos;
    }

    for (var i: u32 = 0u; i < count; i = i + 1u) {
        let ins = instructions[i];
        let op = ins.op;

        // Position-warp effects — only reachable when HAS_POS_WARPS.
        if (HAS_POS_WARPS) {
            if (op == OP_PUSH_NOISE_DISPLACE) {
                let cur = pos_stack[pos_top];
                let amp  = ins.params_lo.x;
                let freq = ins.params_lo.y;
                let seed = u32(ins.params_lo.z);
                let oct  = u32(ins.params_lo.w);
                let warped = cur + rkp_fbm_3d_vec(cur, freq, seed, oct) * amp;
                if (pos_top + 1u < POS_STACK_CAP) {
                    pos_top = pos_top + 1u;
                    pos_stack[pos_top] = warped;
                }
                continue;
            }
            if (op == OP_POP_NOISE_DISPLACE) {
                if (pos_top > 0u) { pos_top = pos_top - 1u; }
                if (sp > 0u) {
                    let amp = ins.params_lo.x;
                    stack[sp - 1u] = stack[sp - 1u] - amp * 1.7320508;
                }
                continue;
            }
            if (op == OP_PUSH_MIRROR) {
                let cur = pos_stack[pos_top];
                let origin = ins.params_lo.xyz;
                let normal = ins.params_hi.xyz;
                let d = dot(cur - origin, normal);
                let folded = cur - 2.0 * min(d, 0.0) * normal;
                if (pos_top + 1u < POS_STACK_CAP) {
                    pos_top = pos_top + 1u;
                    pos_stack[pos_top] = folded;
                }
                continue;
            }
            if (op == OP_POP_MIRROR) {
                if (pos_top > 0u) { pos_top = pos_top - 1u; }
                continue;
            }
            if (op == OP_PUSH_ARRAY) {
                let cur = pos_stack[pos_top];
                let x_axis = vec3<f32>(
                    ins.inverse_world[0][0], ins.inverse_world[0][1], ins.inverse_world[0][2]);
                let y_axis = vec3<f32>(
                    ins.inverse_world[1][0], ins.inverse_world[1][1], ins.inverse_world[1][2]);
                let z_axis = vec3<f32>(
                    ins.inverse_world[2][0], ins.inverse_world[2][1], ins.inverse_world[2][2]);
                let origin = vec3<f32>(
                    ins.inverse_world[3][0], ins.inverse_world[3][1], ins.inverse_world[3][2]);
                let spacing = ins.params_lo.xyz;
                let counts  = ins.params_hi.xyz;
                let rel = cur - origin;
                var delta = vec3<f32>(0.0);
                let half_x = (counts.x - 1.0) * 0.5;
                let half_y = (counts.y - 1.0) * 0.5;
                let half_z = (counts.z - 1.0) * 0.5;
                if (spacing.x > 1e-6 && counts.x > 1.0) {
                    let t = dot(rel, x_axis) / spacing.x;
                    let k = clamp(round(t + half_x) - half_x, -half_x, half_x);
                    delta = delta + k * spacing.x * x_axis;
                }
                if (spacing.y > 1e-6 && counts.y > 1.0) {
                    let t = dot(rel, y_axis) / spacing.y;
                    let k = clamp(round(t + half_y) - half_y, -half_y, half_y);
                    delta = delta + k * spacing.y * y_axis;
                }
                if (spacing.z > 1e-6 && counts.z > 1.0) {
                    let t = dot(rel, z_axis) / spacing.z;
                    let k = clamp(round(t + half_z) - half_z, -half_z, half_z);
                    delta = delta + k * spacing.z * z_axis;
                }
                let folded = cur - delta;
                if (pos_top + 1u < POS_STACK_CAP) {
                    pos_top = pos_top + 1u;
                    pos_stack[pos_top] = folded;
                }
                continue;
            }
            if (op == OP_POP_ARRAY) {
                if (pos_top > 0u) { pos_top = pos_top - 1u; }
                continue;
            }
        }

        // Post-op attribute rewrites don't affect distance — skip.
        if (op == OP_APPLY_MATERIAL_BY_HEIGHT
         || op == OP_APPLY_COLOR_BY_HEIGHT
         || op == OP_APPLY_MATERIAL_BY_NOISE
         || op == OP_APPLY_COLOR_BY_NOISE) {
            continue;
        }

        if (op < 100u) {
            // Primitive. When HAS_POS_WARPS=false, `world_pos` is
            // the only position the primitive ever needs; the
            // pos_stack lookup is dead-stripped.
            var p: vec3<f32>;
            if (HAS_POS_WARPS) {
                p = pos_stack[pos_top];
            } else {
                p = world_pos;
            }
            let d = eval_primitive_distance(ins, p);
            if (sp < STACK_CAP) {
                stack[sp] = d;
                sp = sp + 1u;
            }
        } else {
            // Combinator — min/max on distances. Blend radius is
            // irrelevant for distance (color-only blend in the full
            // path), so MAT_COMBINE_BLEND collapses to sharp min.
            let arity = ins.arity;
            if (arity == 0u || arity > sp) { continue; }
            let base = sp - arity;
            var acc = stack[base];
            for (var k: u32 = 1u; k < arity; k = k + 1u) {
                let rhs = stack[base + k];
                switch op {
                    case 100u: { acc = min(acc, rhs); }              // Union
                    case 101u: { acc = max(acc, rhs); }              // Intersect
                    case 102u: { acc = max(acc, -rhs); }             // Subtract
                    default: {}
                }
            }
            stack[base] = acc;
            sp = base + 1u;
        }
    }

    if (sp == 0u) {
        return 1e30;
    }
    return stack[sp - 1u];
}

// ── Gradient normal ────────────────────────────────────────────────────
// 6-tap central-difference gradient of the distance field. `h` should
// be wider than the raymarch's surface epsilon so the normal stays
// stable on slightly-over-marched hits. Uses the distance-only path —
// material/color are irrelevant for finite-difference gradients.

fn gradient_normal(p: vec3<f32>, count: u32, h: f32) -> vec3<f32> {
    let dx = eval_tree_distance(p + vec3<f32>(h, 0.0, 0.0), count)
           - eval_tree_distance(p - vec3<f32>(h, 0.0, 0.0), count);
    let dy = eval_tree_distance(p + vec3<f32>(0.0, h, 0.0), count)
           - eval_tree_distance(p - vec3<f32>(0.0, h, 0.0), count);
    let dz = eval_tree_distance(p + vec3<f32>(0.0, 0.0, h), count)
           - eval_tree_distance(p - vec3<f32>(0.0, 0.0, h), count);
    let g = vec3<f32>(dx, dy, dz);
    let len = length(g);
    if (len < 1e-8) { return vec3<f32>(0.0, 1.0, 0.0); }
    return g / len;
}
