//! Integration tests for `GpuEvaluator` and the GPU voxel-bake path.
//!
//! After Phase 4, the procedural evaluator lives entirely on the GPU.
//! These tests pin correctness against analytic ground truth — for
//! every known shape, we can compute the expected signed distance by
//! hand and compare the shader's answer within float tolerance. No
//! CPU reference evaluator exists to compare against, which is the
//! whole point of Phase 4.
//!
//! Tests require a real GPU. On CI without an adapter the tests
//! silently skip — `request_adapter` returns `None` and the test
//! prints a skip notice.

use glam::{Affine3A, Vec3};
use rkp_procedural::{
    flatten_tree,
    node_kind::{
        ArrayParams, BoxParams, MaterialCombine, NodeKind, NoiseDisplaceParams,
        SphereParams,
    },
    ProceduralObject,
};
use rkp_render::proc_sample::GpuEvaluator;

/// Create a headless wgpu device. Returns `None` if no adapter is
/// available (CI without GPU, headless CI images, etc.) — callers
/// `eprintln` and skip in that case so the suite stays green.
fn create_device() -> Option<(wgpu::Device, wgpu::Queue)> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::VULKAN | wgpu::Backends::METAL | wgpu::Backends::DX12,
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    });
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::default(),
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .ok()?;
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("gpu_evaluator test device"),
        required_features: wgpu::Features::empty(),
        required_limits: wgpu::Limits::default(),
        memory_hints: wgpu::MemoryHints::Performance,
        trace: wgpu::Trace::Off,
        experimental_features: wgpu::ExperimentalFeatures::default(),
    }))
    .ok()?;
    Some((device, queue))
}

/// Tolerance used across the distance-value asserts. The GPU
/// interpreter is single-precision float throughout, and a sphere's
/// `length(p) - r` accumulates ~1 ulp per operation. 1e-4 absolute is
/// well above any f32 noise we'd see at these distances.
const DIST_TOL: f32 = 1e-4;

/// A sphere of radius `r` at the origin has `d = length(p) - r`. Test
/// the three canonical distance regimes (inside, surface, outside) so
/// a ULP-level drift shows up clearly.
#[test]
fn sphere_distance_matches_analytic() {
    let Some((device, queue)) = create_device() else {
        eprintln!("[sphere] no wgpu adapter — skipping");
        return;
    };

    let obj = ProceduralObject::new(NodeKind::Sphere(SphereParams {
        radius: 0.7,
        material_id: 3,
        ..Default::default()
    }));
    let instructions = flatten_tree(&obj);
    let mut evaluator = GpuEvaluator::new(&device);

    let points = vec![
        Vec3::ZERO,                     // center → -r
        Vec3::new(0.7, 0.0, 0.0),        // on surface → 0
        Vec3::new(1.4, 0.0, 0.0),        // 2r out on +X → r
        Vec3::new(0.0, -0.35, 0.0),      // halfway to surface → -r/2
    ];
    let expected_distances = [-0.7, 0.0, 0.7, -0.35];

    let results = evaluator.evaluate(&device, &queue, &points, &instructions);
    for (i, r) in results.iter().enumerate() {
        assert!(
            (r.distance - expected_distances[i]).abs() < DIST_TOL,
            "sample {i}: got {}, expected {}",
            r.distance, expected_distances[i],
        );
        assert_eq!(r.primary, 3, "primary material should be 3");
        assert_eq!(r.secondary, 0, "secondary defaults to 0 (no post-op)");
        assert_eq!(r.blend_u4, 0, "blend defaults to 0 (no post-op)");
    }
}

/// Union of two offset primitives. The result at every point is
/// `min(d_left, d_right)` — checked at points where each branch
/// clearly wins.
#[test]
fn union_is_min_of_children() {
    let Some((device, queue)) = create_device() else {
        eprintln!("[union] no wgpu adapter — skipping");
        return;
    };

    let mut obj = ProceduralObject::new(NodeKind::Union {
        material_combine: MaterialCombine::Winner,
    });
    let left = obj.add_child(
        obj.root(),
        NodeKind::Sphere(SphereParams {
            radius: 0.3,
            material_id: 5,
            ..Default::default()
        }),
    );
    obj.set_transform(left, Affine3A::from_translation(Vec3::new(-1.0, 0.0, 0.0)));
    let right = obj.add_child(
        obj.root(),
        NodeKind::Sphere(SphereParams {
            radius: 0.3,
            material_id: 11,
            ..Default::default()
        }),
    );
    obj.set_transform(right, Affine3A::from_translation(Vec3::new(1.0, 0.0, 0.0)));

    let instructions = flatten_tree(&obj);
    let mut evaluator = GpuEvaluator::new(&device);

    // Center of left sphere: distance -0.3, material 5.
    // Center of right sphere: distance -0.3, material 11.
    // Origin (midway): each sphere reports d = length(offset) - 0.3
    //   = 1.0 - 0.3 = 0.7; min is 0.7, winner is arbitrary (tie).
    let points = vec![
        Vec3::new(-1.0, 0.0, 0.0),
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::ZERO,
    ];
    let results = evaluator.evaluate(&device, &queue, &points, &instructions);
    assert!((results[0].distance - -0.3).abs() < DIST_TOL);
    assert_eq!(results[0].primary, 5);
    assert!((results[1].distance - -0.3).abs() < DIST_TOL);
    assert_eq!(results[1].primary, 11);
    assert!((results[2].distance - 0.7).abs() < DIST_TOL);
}

/// NoiseDisplace wraps a child sphere: the bounded distance envelope
/// is `child - amp * sqrt(3) ≤ d ≤ child + amp * sqrt(3)` (the POP
/// shrinks by the conservative envelope). Check a point that should
/// stay inside and one that should stay outside regardless of the
/// warp, so the test is robust across noise phases.
#[test]
fn noise_displace_stays_within_envelope() {
    let Some((device, queue)) = create_device() else {
        eprintln!("[noise_displace] no wgpu adapter — skipping");
        return;
    };

    let amp = 0.08_f32;
    let mut obj = ProceduralObject::new(NodeKind::NoiseDisplace(NoiseDisplaceParams {
        amplitude: amp,
        frequency: 2.5,
        octaves: 3,
        seed: 1234,
    }));
    obj.add_child(
        obj.root(),
        NodeKind::Sphere(SphereParams {
            radius: 0.5,
            material_id: 2,
            ..Default::default()
        }),
    );
    let instructions = flatten_tree(&obj);
    let mut evaluator = GpuEvaluator::new(&device);

    let envelope = amp * 3.0_f32.sqrt();

    // Center: base sphere reports -0.5; POP shrinks by `amp * sqrt(3)`,
    // so evaluator returns -0.5 - envelope. Inside regardless of warp.
    let center_expected = -0.5 - envelope;

    // Far point (|p| = 2 + envelope + padding): base is well above 0;
    // subtracting envelope still leaves it positive (outside).
    let far_expected_floor = 2.0 - 0.5 - envelope;

    let points = vec![Vec3::ZERO, Vec3::new(2.0, 0.0, 0.0)];
    let results = evaluator.evaluate(&device, &queue, &points, &instructions);

    // Center: allow +/- envelope around -0.5 (noise moves surface) then
    // also -envelope from POP's conservative shrink.
    assert!(
        results[0].distance <= center_expected + envelope + DIST_TOL,
        "noise-displaced center should read well inside: got {}",
        results[0].distance,
    );
    // Far: the reading should be positive (outside) and at least the
    // pessimistic envelope-shrunk floor.
    assert!(
        results[1].distance >= far_expected_floor - DIST_TOL,
        "noise-displaced far point should read outside: got {}",
        results[1].distance,
    );
    assert!(
        results[1].distance > 0.0,
        "noise-displaced far point distance should be positive: got {}",
        results[1].distance,
    );
}

/// Array fold: a sphere of radius r at origin, Array with counts=[N,1,1]
/// and spacing s, should place identical spheres at
/// {-(N-1)/2 * s, ..., (N-1)/2 * s} along X. Checked by querying the
/// distance at each nominal sphere center (expect ~-r, i.e. inside) and
/// midway between two centers (expect ~s/2 - r outside if s > 2r).
#[test]
fn array_places_repeats_along_axis() {
    let Some((device, queue)) = create_device() else {
        eprintln!("[array] no wgpu adapter — skipping");
        return;
    };

    // Sphere of radius 0.3, linear array of 5 along X with spacing 1.0.
    // Centers at x = -2, -1, 0, 1, 2.
    let r = 0.3_f32;
    let n = 5;
    let s = 1.0_f32;
    let mut obj = ProceduralObject::new(NodeKind::Array(ArrayParams {
        counts: [n, 1, 1],
        spacings: [s, 1.0, 1.0],
    }));
    obj.add_child(
        obj.root(),
        NodeKind::Sphere(SphereParams {
            radius: r,
            material_id: 7,
            ..Default::default()
        }),
    );
    let instructions = flatten_tree(&obj);
    let mut evaluator = GpuEvaluator::new(&device);

    // Nominal centers: d(center_i) should equal -r.
    let centers: Vec<Vec3> = (0..n as i32)
        .map(|i| {
            let x = (i as f32 - (n as f32 - 1.0) * 0.5) * s;
            Vec3::new(x, 0.0, 0.0)
        })
        .collect();
    // Midpoints between centers: exactly s/2 from each neighbor, so
    // distance = s/2 - r outside the spheres (positive when s > 2r).
    let midpoints: Vec<Vec3> = (0..n as i32 - 1)
        .map(|i| {
            let x = (i as f32 + 0.5 - (n as f32 - 1.0) * 0.5) * s;
            Vec3::new(x, 0.0, 0.0)
        })
        .collect();
    // Well past the last center: distance should grow without bound
    // (fold clamps to the outermost cell, then it's a normal sphere).
    let far = Vec3::new((n as f32 - 1.0) * 0.5 * s + 3.0, 0.0, 0.0);
    let expect_far = (far.x - (n as f32 - 1.0) * 0.5 * s) - r;

    let mut points = centers.clone();
    points.extend(midpoints.iter().copied());
    points.push(far);
    let results = evaluator.evaluate(&device, &queue, &points, &instructions);

    for (i, _) in centers.iter().enumerate() {
        let got = results[i].distance;
        assert!(
            (got - -r).abs() < 1e-4,
            "cell {i} center: expected {}, got {got}",
            -r,
        );
        assert_eq!(results[i].primary, 7, "cell {i} material should propagate");
    }
    let mid_base = centers.len();
    for (i, _) in midpoints.iter().enumerate() {
        let got = results[mid_base + i].distance;
        let expect = s * 0.5 - r;
        assert!(
            (got - expect).abs() < 1e-4,
            "midpoint {i}: expected {expect}, got {got}",
        );
    }
    let far_got = results[points.len() - 1].distance;
    assert!(
        (far_got - expect_far).abs() < 1e-4,
        "far point: expected {expect_far}, got {far_got}",
    );
}

/// Even-count Array: cell centers should be at half-integer multiples
/// of spacing, not integer. For N=4 spacing=1 the cells sit at
/// z ∈ {-1.5, -0.5, 0.5, 1.5}, and the origin should be *outside* any
/// sphere (it's exactly between cells -0.5 and 0.5, each reporting
/// distance 0.5 - r). This is the regression the initial implementation
/// had — `round(t)` + clamp-to-`±1.5` produced cells at integer
/// positions interior + half-integer at the clamp edges.
#[test]
fn array_even_count_cells_are_half_integer() {
    let Some((device, queue)) = create_device() else {
        eprintln!("[array_even] no wgpu adapter — skipping");
        return;
    };

    let r = 0.3_f32;
    let n = 4;
    let s = 1.0_f32;
    let mut obj = ProceduralObject::new(NodeKind::Array(ArrayParams {
        counts: [1, 1, n],
        spacings: [1.0, 1.0, s],
    }));
    obj.add_child(
        obj.root(),
        NodeKind::Sphere(SphereParams {
            radius: r,
            material_id: 3,
            ..Default::default()
        }),
    );
    let instructions = flatten_tree(&obj);
    let mut evaluator = GpuEvaluator::new(&device);

    // 4 cells, centered: z = {-1.5, -0.5, 0.5, 1.5}. Origin is midway
    // between the inner two cells — expect distance s/2 - r.
    let centers: Vec<Vec3> = [-1.5, -0.5, 0.5, 1.5]
        .iter()
        .map(|&z| Vec3::new(0.0, 0.0, z))
        .collect();
    let origin = Vec3::ZERO;
    let mut points = centers.clone();
    points.push(origin);
    let results = evaluator.evaluate(&device, &queue, &points, &instructions);

    for (i, _) in centers.iter().enumerate() {
        let got = results[i].distance;
        assert!(
            (got - -r).abs() < 1e-4,
            "even-count cell {i} center: expected {}, got {got}",
            -r,
        );
    }
    let origin_got = results[centers.len()].distance;
    let expect = s * 0.5 - r;
    assert!(
        (origin_got - expect).abs() < 1e-4,
        "origin between inner cells: expected {expect}, got {origin_got}",
    );
}

/// End-to-end: drive `voxelize_octree` with a GPU-backed callback and
/// assert the bake produces the expected structural signature for a
/// sphere union at typical editor-use voxel sizes. Previously this
/// compared GPU against a CPU reference; with no CPU reference to
/// compare against, we lean on the CPU-side voxelizer's structural
/// guarantees: a solid-enough shape produces bricks, those bricks
/// contain non-empty cells, and the voxel count is stable from run
/// to run at a fixed voxel_size.
#[test]
fn voxelize_octree_gpu_runs_end_to_end() {
    use rkp_core::brick_pool::BrickPool;
    use rkp_core::leaf_attr_pool::LeafAttrPool;
    use rkp_core::voxelize_octree::voxelize_octree;

    let Some((device, queue)) = create_device() else {
        eprintln!("[voxelize_gpu] no wgpu adapter — skipping");
        return;
    };

    let mut obj = ProceduralObject::new(NodeKind::Union {
        material_combine: MaterialCombine::Winner,
    });
    let a = obj.add_child(
        obj.root(),
        NodeKind::Sphere(SphereParams {
            radius: 0.35,
            material_id: 5,
            ..Default::default()
        }),
    );
    obj.set_transform(a, Affine3A::from_translation(Vec3::new(0.25, 0.0, 0.0)));
    let b = obj.add_child(
        obj.root(),
        NodeKind::Box(BoxParams {
            half_extents: Vec3::new(0.22, 0.22, 0.22),
            rounding: 0.05,
            material_id: 11,
            color: Vec3::new(0.2, 0.8, 0.4),
        }),
    );
    obj.set_transform(b, Affine3A::from_translation(Vec3::new(-0.25, 0.0, 0.0)));

    let aabb = rkp_core::Aabb {
        min: Vec3::splat(-0.8),
        max: Vec3::splat(0.8),
    };
    let voxel_size = 0.04;

    let mut evaluator = GpuEvaluator::new(&device);
    let instructions = flatten_tree(&obj);
    let mut attrs = LeafAttrPool::new(1_000_000);
    let mut bricks = BrickPool::new(10_000);
    let gpu_sdf = |positions: &[Vec3]| -> Vec<(f32, u16, u16, u8, u32)> {
        evaluator
            .evaluate(&device, &queue, positions, &instructions)
            .into_iter()
            .map(|r| r.into_tuple())
            .collect()
    };
    let result = voxelize_octree(gpu_sdf, &aabb, voxel_size, &mut attrs, &mut bricks)
        .expect("gpu bake");

    assert!(
        result.voxel_count > 0,
        "sphere ∪ box should produce surface voxels",
    );
    assert!(
        !result.brick_ids.is_empty(),
        "shape is deep enough to emit bricks",
    );
    assert!(
        result.leaf_attr_unique_count > 0,
        "should have at least one unique LeafAttr",
    );

    // Spot-check materials: every allocated LeafAttr must carry
    // primary 5 or 11 — the two source spheres' materials are the
    // only options (combinator is Winner, no post-op effects).
    for i in result.leaf_attr_slot_start
        ..(result.leaf_attr_slot_start + result.leaf_attr_unique_count)
    {
        let mat = attrs.get(i).material_primary;
        assert!(
            mat == 5 || mat == 11,
            "leaf_attr {i} has unexpected material {mat}",
        );
    }
}

/// Validates that `voxelize_to_artifact` (worker-path) + scene-side
/// `integrate_artifact` produce the same voxel / leaf_attr / brick
/// counts as direct `voxelize_sdf_fn` (sync path). The remap math is
/// where all the bugs in the async pipeline would live.
#[test]
fn artifact_roundtrip_matches_direct_voxelize() {
    use rkp_core::voxelize_to_artifact;
    use rkp_render::rkp_scene_manager::RkpSceneManager;

    let Some((device, queue)) = create_device() else {
        eprintln!("[artifact_roundtrip] no wgpu adapter — skipping");
        return;
    };

    let mut obj = ProceduralObject::new(NodeKind::Union {
        material_combine: MaterialCombine::Winner,
    });
    obj.add_child(
        obj.root(),
        NodeKind::Sphere(SphereParams {
            radius: 0.35,
            material_id: 5,
            ..Default::default()
        }),
    );

    let aabb = rkp_core::Aabb {
        min: Vec3::splat(-0.6),
        max: Vec3::splat(0.6),
    };
    let voxel_size = 0.04;

    let mut evaluator = GpuEvaluator::new(&device);
    let instructions = flatten_tree(&obj);

    // ── Path A: artifact → integrate ──
    let sdf_a = |positions: &[Vec3]| -> Vec<(f32, u16, u16, u8, u32)> {
        evaluator
            .evaluate(&device, &queue, positions, &instructions)
            .into_iter()
            .map(|r| r.into_tuple())
            .collect()
    };
    let artifact = voxelize_to_artifact(sdf_a, &aabb, voxel_size)
        .expect("artifact voxelize");
    let artifact_voxels = artifact.voxel_count;
    let artifact_attrs = artifact.leaf_attrs.len();
    let artifact_bricks = artifact.brick_cells.len();

    let mut scene_a = RkpSceneManager::new(1_000_000);
    let r_a = scene_a
        .integrate_artifact(artifact, &aabb, voxel_size)
        .expect("integrate");

    // ── Path B: direct voxelize_sdf_fn on the same scene ──
    let sdf_b = |positions: &[Vec3]| -> Vec<(f32, u16, u16, u8, u32)> {
        evaluator
            .evaluate(&device, &queue, positions, &instructions)
            .into_iter()
            .map(|r| r.into_tuple())
            .collect()
    };
    let mut scene_b = RkpSceneManager::new(1_000_000);
    let r_b = scene_b
        .voxelize_sdf_fn(sdf_b, &aabb, voxel_size, 42)
        .expect("direct voxelize");

    assert_eq!(r_a.voxel_count, r_b.voxel_count,
        "voxel count mismatch: async={}, sync={}", r_a.voxel_count, r_b.voxel_count);
    assert_eq!(r_a.leaf_attr_slot_count, r_b.leaf_attr_slot_count,
        "leaf_attr unique count mismatch");
    assert_eq!(r_a.brick_ids.len(), r_b.brick_ids.len(),
        "brick count mismatch");
    assert_eq!(artifact_voxels, r_b.voxel_count,
        "artifact voxel_count should match before remap");
    assert_eq!(artifact_attrs as u32, r_b.leaf_attr_slot_count,
        "artifact leaf_attrs count should match");
    assert_eq!(artifact_bricks, r_b.brick_ids.len(),
        "artifact brick count should match");

    eprintln!(
        "[artifact_roundtrip] voxels={} attrs={} bricks={} — all paths agree",
        r_a.voxel_count, r_a.leaf_attr_slot_count, r_a.brick_ids.len()
    );
}

/// Deliberate perf harness: run a typical-size bake (sphere, 0.5m, no
/// combinators) at 0.08 / 0.02 / 0.005 voxel sizes — the four standard
/// tiers with 0.005 = finest the editor exposes. `#[ignore]` because
/// timing output is informational, not a regression gate.
#[test]
#[ignore]
fn bake_perf_sweep() {
    use rkp_core::brick_pool::BrickPool;
    use rkp_core::leaf_attr_pool::LeafAttrPool;
    use rkp_core::voxelize_octree::voxelize_octree;

    let Some((device, queue)) = create_device() else {
        eprintln!("[perf_sweep] no wgpu adapter — skipping");
        return;
    };

    let mut obj = ProceduralObject::new(NodeKind::Root);
    obj.add_child(
        obj.root(),
        NodeKind::Sphere(SphereParams {
            radius: 0.5,
            material_id: 1,
            ..Default::default()
        }),
    );

    let aabb = rkp_core::Aabb {
        min: Vec3::splat(-0.7),
        max: Vec3::splat(0.7),
    };

    let mut evaluator = GpuEvaluator::new(&device);
    let instructions = flatten_tree(&obj);

    for &voxel_size in &[0.08f32, 0.02, 0.005] {
        eprintln!("\n[perf_sweep] ===== voxel_size = {} =====", voxel_size);
        let mut attrs = LeafAttrPool::new(10_000_000);
        let mut bricks = BrickPool::new(1_000_000);
        let gpu_sdf = |positions: &[Vec3]| -> Vec<(f32, u16, u16, u8, u32)> {
            evaluator
                .evaluate(&device, &queue, positions, &instructions)
                .into_iter()
                .map(|r| r.into_tuple())
                .collect()
        };
        let t = std::time::Instant::now();
        let r = voxelize_octree(gpu_sdf, &aabb, voxel_size, &mut attrs, &mut bricks)
            .expect("bake");
        eprintln!(
            "[perf_sweep] voxel_size={} voxels={} wall={:.2}ms",
            voxel_size,
            r.voxel_count,
            t.elapsed().as_secs_f32() * 1000.0,
        );
    }
}
