//! Building generator — modular 1920s brick office building (splat engine).
//!
//! Produces a `Subtree` where each architectural element is its own object:
//!
//! ```text
//! Building
//!   Ground Floor          — walls + slab (window holes cut out)
//!   Floor 1               — walls + slab
//!   ...
//!   Window F0-W0          — glass pane + lintel + sill
//!   Window F0-W1
//!   Door F0 (front)       — lintel only (no glass, no sill)
//!   ...
//!   Cornice               — decorative top band
//! ```
//!
//! Uses `voxelize_splat` for opacity-field output (binary 1.0/0.0 opacity)
//! instead of SDF distances.

use glam::Vec3;
use rkf_core::Aabb;
use rkf_macros::{component, generator};
use rkf_runtime::behavior::{Ranged, RangedInt};
use rkf_runtime::generator::helpers::{voxelize_splat, voxelize_box_splat, VoxelQuery, in_box};
use rkf_runtime::generator::{
    GeneratedObject, GeneratorContext, GeneratorError, GeneratorOutput,
};

#[component(no_default)]
pub struct BuildingParams {
    /// Random seed. Set to -1 for a new random result each regeneration.
    pub seed: i32,
    pub floors: RangedInt,
    pub width: Ranged,
    pub depth: Ranged,
    pub floor_height: Ranged,
    pub ground_floor_height: Ranged,
    pub wall_thickness: f32,
    pub window_width: Ranged,
    pub window_height: Ranged,
    pub door_width: f32,
    pub door_height: f32,
    pub voxel_size: f32,
    #[material_ref]
    pub mat_brick: u16,
    #[material_ref]
    pub mat_stone: u16,
    #[material_ref]
    pub mat_floor: u16,
    #[material_ref]
    pub mat_glass: u16,
}

impl Default for BuildingParams {
    fn default() -> Self {
        Self {
            seed: 42,
            floors: RangedInt::new(2, 5),
            width: Ranged::new(6.0, 14.0),
            depth: Ranged::new(6.0, 10.0),
            floor_height: Ranged::new(3.0, 4.0),
            ground_floor_height: Ranged::new(4.0, 5.0),
            wall_thickness: 0.4,
            window_width: Ranged::new(1.0, 1.4),
            window_height: Ranged::new(1.6, 2.0),
            door_width: 1.4,
            door_height: 2.8,
            voxel_size: 0.15,
            mat_brick: 15,
            mat_stone: 1,
            mat_floor: 0,
            mat_glass: 14,
        }
    }
}

#[generator(name = "building", params = BuildingParams)]
fn generate_building(
    params: &BuildingParams,
    ctx: &GeneratorContext,
) -> Result<GeneratorOutput, GeneratorError> {
    if params.voxel_size <= 0.0 {
        return Err(GeneratorError::InvalidParams(
            "voxel_size must be positive".into(),
        ));
    }

    let p = params;
    let vs = p.voxel_size;
    let seed = if p.seed < 0 { ctx.generation } else { p.seed as u64 };
    let snap = |v: f32| -> f32 { (v / vs).round() * vs };

    let floors = (p.floors.sample_seeded(seed).max(1)) as u32;
    let width = p.width.sample_seeded(seed + 100).max(2.0);
    let depth = p.depth.sample_seeded(seed + 200).max(2.0);
    let floor_height = p.floor_height.sample_seeded(seed + 300).max(2.0);
    let ground_floor_height = p.ground_floor_height.sample_seeded(seed + 400).max(2.0);

    let upper_floors = if floors > 1 { floors - 1 } else { 0 };
    let total_height = snap(ground_floor_height + upper_floors as f32 * floor_height + 0.3);
    let half_w = snap(width / 2.0);
    let half_d = snap(depth / 2.0);
    let wt = snap(p.wall_thickness).max(vs * 3.0);
    let slab_thickness = snap((0.25_f32).max(vs * 3.0));
    let pane_thick = vs * 3.0;
    let lintel_h = snap((0.12_f32).max(vs * 3.0));
    let sill_h = snap((0.08_f32).max(vs * 3.0));
    let overhang = snap(0.06_f32).max(vs);
    let protrusion = snap(0.04_f32).max(vs);

    let sp = SampledParams {
        seed, floors, width, depth, floor_height, ground_floor_height,
        wall_thickness: p.wall_thickness, voxel_size: vs,
        door_width: p.door_width, door_height: p.door_height,
        mat_brick: p.mat_brick, mat_stone: p.mat_stone,
        mat_floor: p.mat_floor, mat_glass: p.mat_glass,
        window_width: p.window_width, window_height: p.window_height,
    };

    let windows = compute_window_layout(&sp);

    let mut children: Vec<GeneratedObject> = Vec::new();

    // ── Floors (walls + slab, with cutout holes for windows/doors) ──────
    for floor_idx in 0..floors {
        let is_ground = floor_idx == 0;
        let floor_base = if is_ground {
            0.0
        } else {
            snap(ground_floor_height + (floor_idx - 1) as f32 * floor_height)
        };
        let fh = if is_ground { snap(ground_floor_height) } else { snap(floor_height) };
        let half_h = fh / 2.0;

        let floor_aabb = Aabb::new(
            Vec3::new(-half_w, -half_h, -half_d),
            Vec3::new(half_w, half_h, half_d),
        );

        // Precompute window cutout boxes for this floor (floor-local coords).
        let cutouts: Vec<(Vec3, Vec3)> = windows.iter()
            .filter(|w| w.floor == floor_idx)
            .map(|w| {
                let hw = snap(w.half_width);
                let hh = snap(w.half_height);
                let cx = snap(w.center_along_wall);
                let cy = snap(w.local_center_y);
                wall_box(cx, cy, hw, hh, w.wall, half_w, half_d, wt)
            })
            .collect();

        let f_half_w = half_w;
        let f_half_d = half_d;
        let f_wt = wt;
        let f_slab = slab_thickness;
        let f_height = fh;
        let f_mat_brick = sp.mat_brick;
        let f_mat_floor = sp.mat_floor;

        let output = voxelize_splat(floor_aabb, vs, Some(ctx), |pos| {
            let lp = pos + Vec3::new(0.0, half_h, 0.0); // to floor-local

            // Outside envelope.
            if lp.x < -f_half_w || lp.x > f_half_w
                || lp.y < 0.0 || lp.y > f_height
                || lp.z < -f_half_d || lp.z > f_half_d
            {
                return VoxelQuery { solid: false, material: 0 };
            }

            // Interior: slab or air.
            let inner_x = lp.x.abs() < f_half_w - f_wt;
            let inner_z = lp.z.abs() < f_half_d - f_wt;
            if inner_x && inner_z {
                return if lp.y <= f_slab {
                    VoxelQuery { solid: true, material: f_mat_floor }
                } else {
                    VoxelQuery { solid: false, material: 0 }
                };
            }

            // Wall — cut holes for windows/doors.
            for (cmin, cmax) in &cutouts {
                if in_box(lp, *cmin, *cmax) {
                    return VoxelQuery { solid: false, material: 0 };
                }
            }

            VoxelQuery { solid: true, material: f_mat_brick }
        })?;

        let floor_name = if is_ground {
            "Ground Floor".to_string()
        } else {
            format!("Floor {}", floor_idx)
        };

        children.push(GeneratedObject::with_geometry(
            floor_name,
            rkf_core::Transform {
                position: Vec3::new(0.0, floor_base + half_h, 0.0),
                rotation: glam::Quat::IDENTITY,
                scale: Vec3::ONE,
            },
            rkf_core::SceneNode::new("floor"),
            output,
        ));
    }

    // ── Windows and doors (each is a group of sub-pieces) ───────────────
    let wall_names = ["front", "back", "left", "right"];

    for (wi, win) in windows.iter().enumerate() {
        let hw = snap(win.half_width);
        let hh = snap(win.half_height);
        let cx = snap(win.center_along_wall);
        let cy = snap(win.local_center_y);

        // Floor base in building-space Y.
        let floor_base = if win.floor == 0 {
            0.0
        } else {
            snap(ground_floor_height + (win.floor - 1) as f32 * floor_height)
        };

        let wall_name = wall_names[win.wall as usize % 4];

        if win.is_door {
            // Door: just a lintel above the opening.
            let lw = hw + overhang;
            let lintel_y = cy + hh; // floor-local
            let (lmin, lmax) = wall_accent(cx, lintel_y, lintel_h, lw, win.wall, half_w, half_d, wt, protrusion);
            let (child_center, child_output) = voxelize_box_splat(lmin, lmax, sp.mat_stone, vs, Some(ctx))?;

            children.push(GeneratedObject::with_geometry(
                format!("Door F{} ({})", win.floor, wall_name),
                rkf_core::Transform {
                    position: child_center + Vec3::new(0.0, floor_base, 0.0),
                    rotation: glam::Quat::IDENTITY,
                    scale: Vec3::ONE,
                },
                rkf_core::SceneNode::new("door-lintel"),
                child_output,
            ));
        } else {
            // Window: glass pane + lintel + sill — each as separate children.

            // Glass pane.
            let (pmin, pmax) = wall_pane(cx, cy, hw, hh, win.wall, half_w, half_d, pane_thick);
            let (child_center, child_output) = voxelize_box_splat(pmin, pmax, sp.mat_glass, vs, Some(ctx))?;

            children.push(GeneratedObject::with_geometry(
                format!("Window F{}-{} glass ({})", win.floor, wi, wall_name),
                rkf_core::Transform {
                    position: child_center + Vec3::new(0.0, floor_base, 0.0),
                    rotation: glam::Quat::IDENTITY,
                    scale: Vec3::ONE,
                },
                rkf_core::SceneNode::new("glass"),
                child_output,
            ));

            // Lintel (above opening).
            let lw = hw + overhang;
            let lintel_y = cy + hh;
            let (lmin, lmax) = wall_accent(cx, lintel_y, lintel_h, lw, win.wall, half_w, half_d, wt, protrusion);
            let (child_center, child_output) = voxelize_box_splat(lmin, lmax, sp.mat_stone, vs, Some(ctx))?;

            children.push(GeneratedObject::with_geometry(
                format!("Window F{}-{} lintel ({})", win.floor, wi, wall_name),
                rkf_core::Transform {
                    position: child_center + Vec3::new(0.0, floor_base, 0.0),
                    rotation: glam::Quat::IDENTITY,
                    scale: Vec3::ONE,
                },
                rkf_core::SceneNode::new("lintel"),
                child_output,
            ));

            // Sill (below opening).
            let sill_y = cy - hh - sill_h;
            let (smin, smax) = wall_accent(cx, sill_y, sill_h, lw, win.wall, half_w, half_d, wt, protrusion);
            let (child_center, child_output) = voxelize_box_splat(smin, smax, sp.mat_stone, vs, Some(ctx))?;

            children.push(GeneratedObject::with_geometry(
                format!("Window F{}-{} sill ({})", win.floor, wi, wall_name),
                rkf_core::Transform {
                    position: child_center + Vec3::new(0.0, floor_base, 0.0),
                    rotation: glam::Quat::IDENTITY,
                    scale: Vec3::ONE,
                },
                rkf_core::SceneNode::new("sill"),
                child_output,
            ));
        }
    }

    // ── Cornice ─────────────────────────────────────────────────────────
    let cornice_h = snap((0.3_f32).max(vs * 3.0));
    let cornice_overhang = snap(0.08_f32).max(vs);
    let half_ch = cornice_h / 2.0;
    let co = cornice_overhang;

    let cornice_aabb = Aabb::new(
        Vec3::new(-half_w - co, -half_ch, -half_d - co),
        Vec3::new(half_w + co, half_ch, half_d + co),
    );

    let c_hw = half_w;
    let c_hd = half_d;
    let c_h = cornice_h;
    let mat_stone = sp.mat_stone;

    let cornice_output = voxelize_splat(cornice_aabb, vs, Some(ctx), |pos| {
        let lp = pos + Vec3::new(0.0, half_ch, 0.0);
        let solid = lp.y >= 0.0 && lp.y <= c_h
            && lp.x >= -c_hw - co && lp.x <= c_hw + co
            && lp.z >= -c_hd - co && lp.z <= c_hd + co;
        VoxelQuery { solid, material: mat_stone }
    })?;

    children.push(GeneratedObject::with_geometry(
        "Cornice",
        rkf_core::Transform {
            position: Vec3::new(0.0, total_height + half_ch, 0.0),
            rotation: glam::Quat::IDENTITY,
            scale: Vec3::ONE,
        },
        rkf_core::SceneNode::new("cornice"),
        cornice_output,
    ));

    Ok(GeneratorOutput::Subtree(children))
}

// ── Sampled params ───────────────────────────────────────────────────────

struct SampledParams {
    seed: u64,
    floors: u32,
    width: f32,
    depth: f32,
    floor_height: f32,
    ground_floor_height: f32,
    wall_thickness: f32,
    voxel_size: f32,
    door_width: f32,
    door_height: f32,
    mat_brick: u16,
    mat_stone: u16,
    mat_floor: u16,
    mat_glass: u16,
    window_width: Ranged,
    window_height: Ranged,
}

// ── Wall geometry helpers ────────────────────────────────────────────────

fn wall_box(cx: f32, cy: f32, hw: f32, hh: f32, wall: u16, half_w: f32, half_d: f32, wt: f32) -> (Vec3, Vec3) {
    match wall {
        0 => (Vec3::new(cx - hw, cy - hh, -half_d), Vec3::new(cx + hw, cy + hh, -half_d + wt)),
        1 => (Vec3::new(cx - hw, cy - hh, half_d - wt), Vec3::new(cx + hw, cy + hh, half_d)),
        2 => (Vec3::new(-half_w, cy - hh, cx - hw), Vec3::new(-half_w + wt, cy + hh, cx + hw)),
        _ => (Vec3::new(half_w - wt, cy - hh, cx - hw), Vec3::new(half_w, cy + hh, cx + hw)),
    }
}

fn wall_pane(cx: f32, cy: f32, hw: f32, hh: f32, wall: u16, half_w: f32, half_d: f32, pane_thick: f32) -> (Vec3, Vec3) {
    match wall {
        0 => (Vec3::new(cx - hw, cy - hh, -half_d), Vec3::new(cx + hw, cy + hh, -half_d + pane_thick)),
        1 => (Vec3::new(cx - hw, cy - hh, half_d - pane_thick), Vec3::new(cx + hw, cy + hh, half_d)),
        2 => (Vec3::new(-half_w, cy - hh, cx - hw), Vec3::new(-half_w + pane_thick, cy + hh, cx + hw)),
        _ => (Vec3::new(half_w - pane_thick, cy - hh, cx - hw), Vec3::new(half_w, cy + hh, cx + hw)),
    }
}

fn wall_accent(cx: f32, y: f32, h: f32, lw: f32, wall: u16, half_w: f32, half_d: f32, wt: f32, protrusion: f32) -> (Vec3, Vec3) {
    match wall {
        0 => (Vec3::new(cx - lw, y, -half_d - protrusion), Vec3::new(cx + lw, y + h, -half_d + wt + protrusion)),
        1 => (Vec3::new(cx - lw, y, half_d - wt - protrusion), Vec3::new(cx + lw, y + h, half_d + protrusion)),
        2 => (Vec3::new(-half_w - protrusion, y, cx - lw), Vec3::new(-half_w + wt + protrusion, y + h, cx + lw)),
        _ => (Vec3::new(half_w - wt - protrusion, y, cx - lw), Vec3::new(half_w + protrusion, y + h, cx + lw)),
    }
}

// ── Window layout ────────────────────────────────────────────────────────

struct WindowOpening {
    center_along_wall: f32,
    local_center_y: f32,
    half_width: f32,
    half_height: f32,
    wall: u16,
    floor: u32,
    is_door: bool,
}

fn compute_window_layout(p: &SampledParams) -> Vec<WindowOpening> {
    let mut windows = Vec::new();
    let seed_base = p.seed as u64;
    let sill_height = 0.9;
    let dhw = p.door_width / 2.0;
    let dhh = p.door_height / 2.0;

    let avg_width = p.window_width.midpoint();
    let n_windows_fb = ((p.width - 1.5) / (avg_width + 1.0)).floor() as u32;
    let spacing_fb = p.width / (n_windows_fb + 1) as f32;

    let n_windows_side = ((p.depth - 1.5) / (avg_width + 1.0)).floor() as u32;
    let spacing_side = p.depth / (n_windows_side + 1) as f32;

    let center_fb = n_windows_fb / 2;
    let mut win_index = 0u64;

    for floor in 0..p.floors {
        let is_ground = floor == 0;

        for i in 0..n_windows_fb {
            let cx = -p.width / 2.0 + spacing_fb * (i + 1) as f32;

            if is_ground && i == center_fb {
                let cy = dhh;
                for wall in [0u16, 1] {
                    windows.push(WindowOpening {
                        center_along_wall: cx, local_center_y: cy,
                        half_width: dhw, half_height: dhh, wall, floor, is_door: true,
                    });
                    win_index += 1;
                }
            } else {
                let hw = p.window_width.sample_seeded(seed_base + win_index) / 2.0;
                let hh = p.window_height.sample_seeded(seed_base + win_index + 1000) / 2.0;
                let cy = sill_height + hh;
                for wall in [0u16, 1] {
                    windows.push(WindowOpening {
                        center_along_wall: cx, local_center_y: cy,
                        half_width: hw, half_height: hh, wall, floor, is_door: false,
                    });
                }
                win_index += 1;
            }
        }

        for i in 0..n_windows_side {
            let cz = -p.depth / 2.0 + spacing_side * (i + 1) as f32;
            let hw = p.window_width.sample_seeded(seed_base + win_index) / 2.0;
            let hh = p.window_height.sample_seeded(seed_base + win_index + 1000) / 2.0;
            let cy = sill_height + hh;
            for wall in [2u16, 3] {
                windows.push(WindowOpening {
                    center_along_wall: cz, local_center_y: cy,
                    half_width: hw, half_height: hh, wall, floor, is_door: false,
                });
            }
            win_index += 1;
        }
    }

    windows
}
