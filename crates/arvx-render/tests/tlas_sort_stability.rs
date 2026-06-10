//! In-vivo probe for the TLAS radix-sort stability bug (Xid-109 hunt).
//!
//! The production chain is: Morton compute → 4× LSD radix passes →
//! Karras `build_internal_main` → `propagate_atomic_main`. The radix
//! scatter claims output slots with a per-(workgroup, bucket)
//! `atomicAdd`, which is NOT stable — but LSD radix sort derives
//! global sortedness FROM per-pass stability, so distinct keys that
//! share their higher digits can come out of the final pass unsorted.
//! Karras requires sorted keys; on unsorted input two internal nodes
//! can claim the same child (`parents[child] = idx` races,
//! last-writer-wins), the parents graph can contain cycles, and
//! `propagate_atomic_main`'s uncapped walk-up loop then spins forever
//! on the GPU — the observed whole-desktop NVRM Xid 109 hang.
//!
//! This test runs the REAL shader chain on the REAL adapter — but
//! deliberately never dispatches `propagate_atomic_main`, so it
//! cannot hang. Instead it reads back the sorted keys and the
//! `parents[]` buffer and checks on the CPU:
//!   1. keys are globally sorted ascending;
//!   2. every leaf's parent walk reaches the root — no cycles, no
//!      orphans (a cycle here = a GPU hang in production).
//!
//! Prim layouts mimic the editor scene that froze: grid-clustered
//! terrain tiles (64 m / 128 m, Morton keys sharing high bits and
//! differing in low bits — the exact recipe for LSD instability to
//! become visible) plus jittered scattered assets, swept across prim
//! counts that bracket the observed ~15–20-resident-tile threshold
//! and the 32-lane warp boundary.
//!
//! EXPECTED today: FAILS (demonstrates the bug in vivo). After the
//! scatter is made stable this goes green and pins the regression.
//! Skips silently when no wgpu adapter is available.

use arvx_render::tlas_build_pass::{
    scene_aabb_from_prims, MortonUniform, RadixUniform, TlasBuildPass, TlasPrim, TlasState,
    RADIX_BUCKETS, RADIX_PASSES, RADIX_WG_SIZE,
};
use arvx_render::tlas_pass::{TlasInstanceLeaf, TlasNode};

const PARENT_SENTINEL: u32 = 0xFFFF_FFFF;

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
        label: Some("tlas_sort_stability test device"),
        required_features: wgpu::Features::empty(),
        required_limits: wgpu::Limits::default(),
        memory_hints: wgpu::MemoryHints::Performance,
        trace: wgpu::Trace::Off,
        experimental_features: wgpu::ExperimentalFeatures::default(),
    }))
    .ok()?;
    Some((device, queue))
}

/// Deterministic LCG so trials are reproducible without a rand dep.
struct Lcg(u64);
impl Lcg {
    fn next_f32(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 33) as u32) as f32 / (u32::MAX as f32)
    }
}

fn prim(min: [f32; 3], size: f32, id: u32) -> TlasPrim {
    TlasPrim {
        aabb_min: min,
        asset_id: id,
        aabb_max: [min[0] + size, min[1] + size, min[2] + size],
        instance_state_offset: 0,
        material_id: 0,
        instance_index: id,
        _pad0: 0,
        _pad1: 0,
    }
}

/// Editor-like scene: `tiles` 64 m grid tiles (row-major over a square
/// grid, like the streamer's residency set), a ring of 128 m LOD-1
/// tiles, and `scattered` jittered props. Mirrors the 192 m-terrain
/// freeze scene shape.
fn make_scene(tiles: usize, lod1_tiles: usize, scattered: usize, rng: &mut Lcg) -> Vec<TlasPrim> {
    let mut prims = Vec::new();
    let mut id = 0u32;
    let side = (tiles as f32).sqrt().ceil() as usize;
    for t in 0..tiles {
        let gx = (t % side) as f32;
        let gz = (t / side) as f32;
        prims.push(prim([gx * 64.0, 0.0, gz * 64.0], 64.0, id));
        id += 1;
    }
    for t in 0..lod1_tiles {
        let gx = (t % 4) as f32;
        let gz = (t / 4) as f32;
        prims.push(prim([-512.0 + gx * 128.0, 0.0, -512.0 + gz * 128.0], 128.0, id));
        id += 1;
    }
    for _ in 0..scattered {
        let x = (rng.next_f32() - 0.5) * 1000.0;
        let y = rng.next_f32() * 50.0;
        let z = (rng.next_f32() - 0.5) * 1000.0;
        prims.push(prim([x, y, z], 0.5 + rng.next_f32() * 4.0, id));
        id += 1;
    }
    prims
}

struct TrialResult {
    sorted: bool,
    cycles: usize,
    orphans: usize,
}

/// Run the production dispatch chain (Morton → 4× radix → Karras
/// leaves+internal) once on the GPU and classify the outcome on the
/// CPU. `propagate_atomic_main` is intentionally NOT dispatched.
fn run_trial(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pass: &TlasBuildPass,
    nodes_buffer: &wgpu::Buffer,
    leaves_buffer: &wgpu::Buffer,
    prims: &[TlasPrim],
) -> TrialResult {
    let n = prims.len() as u32;
    let total_nodes = 2 * n - 1;
    let num_workgroups = ((n + RADIX_WG_SIZE - 1) / RADIX_WG_SIZE).max(1);

    queue.write_buffer(&pass.tlas_prims_buffer, 0, bytemuck::cast_slice(prims));
    queue.write_buffer(
        &pass.tlas_state_buffer,
        0,
        bytemuck::bytes_of(&TlasState {
            prim_count: n,
            radix_workgroups: num_workgroups,
            internal_wgs: (((n - 1) + 63) / 64).max(1),
            total_node_wgs: ((total_nodes + 63) / 64).max(1),
        }),
    );
    let (scene_min, scene_max) = scene_aabb_from_prims(prims);
    queue.write_buffer(
        &pass.morton_uniform_buffer,
        0,
        bytemuck::bytes_of(&MortonUniform { scene_min, _pad0: 0, scene_max, _pad1: 0 }),
    );
    // Per-pass digit shifts at 256-byte dynamic-offset stride —
    // without this every radix pass sorts by digit_shift 0 (the
    // buffer is zero-initialized) and the output is unsorted by
    // construction, which would make this test prove nothing.
    let radix_uniform_stride: u64 = 256;
    let mut radix_uniform_bytes: Vec<u8> =
        vec![0u8; (RADIX_PASSES as u64 * radix_uniform_stride) as usize];
    for p in 0..RADIX_PASSES {
        let u = RadixUniform { digit_shift: p * 8, _pad0: 0, _pad1: 0, _pad2: 0 };
        let off = (p as u64 * radix_uniform_stride) as usize;
        radix_uniform_bytes[off..off + std::mem::size_of::<RadixUniform>()]
            .copy_from_slice(bytemuck::bytes_of(&u));
    }
    queue.write_buffer(&pass.radix_uniform_buffer, 0, &radix_uniform_bytes);
    // Re-seed parents to the sentinel every trial — production's
    // `init_atomic_aabb_main` does this on GPU; stale parents from a
    // previous trial would contaminate the cycle check.
    let parents_init: Vec<u32> = vec![PARENT_SENTINEL; total_nodes as usize];
    queue.write_buffer(&pass.parents_buffer, 0, bytemuck::cast_slice(&parents_init));

    let morton_g0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pass.morton_g0_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: pass.tlas_prims_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: pass.keys_a_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: pass.vals_a_buffer.as_entire_binding() },
        ],
    });
    let morton_g1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pass.morton_g1_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: pass.morton_uniform_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: pass.tlas_state_buffer.as_entire_binding() },
        ],
    });
    let radix_g0_a_to_b = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pass.radix_g0_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: pass.keys_a_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: pass.vals_a_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: pass.keys_b_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: pass.vals_b_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: pass.histogram_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: pass.scan_offsets_buffer.as_entire_binding() },
        ],
    });
    let radix_g0_b_to_a = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pass.radix_g0_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: pass.keys_b_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: pass.vals_b_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: pass.keys_a_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: pass.vals_a_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: pass.histogram_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: pass.scan_offsets_buffer.as_entire_binding() },
        ],
    });
    let radix_g1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pass.radix_g1_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &pass.radix_uniform_buffer,
                    offset: 0,
                    size: std::num::NonZeroU64::new(std::mem::size_of::<RadixUniform>() as u64),
                }),
            },
            wgpu::BindGroupEntry { binding: 1, resource: pass.tlas_state_buffer.as_entire_binding() },
        ],
    });
    let karras_g0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pass.karras_g0_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: pass.keys_a_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: pass.vals_a_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: pass.tlas_prims_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: nodes_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: leaves_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: pass.parents_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 6, resource: pass.aabb_min_atomic_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 7, resource: pass.aabb_max_atomic_buffer.as_entire_binding() },
        ],
    });
    let karras_g1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pass.karras_g1_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: pass.tlas_state_buffer.as_entire_binding(),
        }],
    });

    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut cpass = encoder
            .begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
        cpass.set_pipeline(&pass.morton_pipeline);
        cpass.set_bind_group(0, &morton_g0, &[]);
        cpass.set_bind_group(1, &morton_g1, &[]);
        cpass.dispatch_workgroups(num_workgroups, 1, 1);
    }
    let histogram_bytes = (num_workgroups as u64) * (RADIX_BUCKETS as u64) * 4;
    for p in 0..RADIX_PASSES {
        let radix_g0 = if p % 2 == 0 { &radix_g0_a_to_b } else { &radix_g0_b_to_a };
        let dyn_off = (p as u64 * radix_uniform_stride) as u32;
        encoder.clear_buffer(&pass.histogram_buffer, 0, Some(histogram_bytes));
        {
            let mut cpass = encoder
                .begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
            cpass.set_pipeline(&pass.radix_count_pipeline);
            cpass.set_bind_group(0, radix_g0, &[]);
            cpass.set_bind_group(1, &radix_g1, &[dyn_off]);
            cpass.dispatch_workgroups(num_workgroups, 1, 1);
        }
        {
            let mut cpass = encoder
                .begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
            cpass.set_pipeline(&pass.radix_scan_pipeline);
            cpass.set_bind_group(0, radix_g0, &[]);
            cpass.set_bind_group(1, &radix_g1, &[dyn_off]);
            cpass.dispatch_workgroups(1, 1, 1);
        }
        {
            let mut cpass = encoder
                .begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
            cpass.set_pipeline(&pass.radix_scatter_pipeline);
            cpass.set_bind_group(0, radix_g0, &[]);
            cpass.set_bind_group(1, &radix_g1, &[dyn_off]);
            cpass.dispatch_workgroups(num_workgroups, 1, 1);
        }
    }
    // Karras leaves + internal — mirrors build_gpu_tlas. The
    // propagate pass is intentionally absent (it is the hang).
    {
        let mut cpass = encoder
            .begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
        cpass.set_pipeline(&pass.karras_leaves_pipeline);
        cpass.set_bind_group(0, &karras_g0, &[]);
        cpass.set_bind_group(1, &karras_g1, &[]);
        cpass.dispatch_workgroups(num_workgroups, 1, 1);
    }
    {
        let mut cpass = encoder
            .begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
        cpass.set_pipeline(&pass.karras_internal_pipeline);
        cpass.set_bind_group(0, &karras_g0, &[]);
        cpass.set_bind_group(1, &karras_g1, &[]);
        cpass.dispatch_workgroups((((n - 1) + 63) / 64).max(1), 1, 1);
    }

    let keys_bytes = (n as u64) * 4;
    let parents_bytes = (total_nodes as u64) * 4;
    let keys_readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: keys_bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let parents_readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: parents_bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    encoder.copy_buffer_to_buffer(&pass.keys_a_buffer, 0, &keys_readback, 0, keys_bytes);
    encoder.copy_buffer_to_buffer(&pass.parents_buffer, 0, &parents_readback, 0, parents_bytes);
    queue.submit(std::iter::once(encoder.finish()));

    let ks = keys_readback.slice(..);
    ks.map_async(wgpu::MapMode::Read, |_| {});
    device.poll(wgpu::PollType::wait_indefinitely()).expect("device poll");
    let kv = ks.get_mapped_range();
    let gpu_keys: Vec<u32> = bytemuck::cast_slice::<u8, u32>(&kv).to_vec();
    drop(kv);
    keys_readback.unmap();

    let ps = parents_readback.slice(..);
    ps.map_async(wgpu::MapMode::Read, |_| {});
    device.poll(wgpu::PollType::wait_indefinitely()).expect("device poll");
    let pv = ps.get_mapped_range();
    let parents: Vec<u32> = bytemuck::cast_slice::<u8, u32>(&pv).to_vec();
    drop(pv);
    parents_readback.unmap();

    let sorted = gpu_keys.windows(2).all(|w| w[0] <= w[1]);

    // CPU replica of propagate_atomic_main's walk, hop-capped: any
    // leaf whose walk exceeds total_nodes hops without reaching the
    // sentinel is in a cycle — in production that thread never
    // terminates and the GPU channel hangs (Xid 109). A walk that
    // hits the sentinel at a non-root node is an orphan: a WRONG tree
    // (silently tolerated in production — the TLAS has no consumers)
    // but not a hang.
    let mut cycles = 0usize;
    let mut orphans = 0usize;
    for i in 0..n {
        let mut cur = n - 1 + i;
        let mut hops = 0u32;
        loop {
            let p = parents[cur as usize];
            if p == PARENT_SENTINEL {
                if cur != 0 && cur != n - 1 + i {
                    // Walked somewhere and stalled before the root.
                    orphans += 1;
                }
                break;
            }
            cur = p;
            hops += 1;
            if hops > total_nodes {
                cycles += 1;
                break;
            }
        }
    }

    TrialResult { sorted, cycles, orphans }
}

#[test]
fn radix_sort_is_stable_and_karras_parents_acyclic_in_vivo() {
    let Some((device, queue)) = create_device() else {
        eprintln!("[tlas_sort_stability] no wgpu adapter — skipping");
        return;
    };

    // (tiles, lod1_tiles, scattered) sweeps bracketing the observed
    // freeze threshold (~15-20 resident tiles + ~a dozen other assets)
    // and the 32-lane warp boundary.
    let configs: &[(usize, usize, usize)] = &[
        (4, 0, 8),    // n=12  — size of the existing green sort test
        (9, 0, 8),    // n=17
        (9, 4, 8),    // n=21
        (16, 4, 8),   // n=28
        (16, 8, 12),  // n=36  — crosses the warp boundary
        (25, 8, 12),  // n=45
        (25, 16, 23), // n=64  — exactly one full workgroup
        (36, 16, 28), // n=80  — two workgroups
    ];
    const TRIALS: usize = 100;

    let max_n: u32 = configs.iter().map(|(t, l, s)| (t + l + s) as u32).max().unwrap();
    let mut pass = TlasBuildPass::new(&device);
    pass.ensure_prims_capacity(&device, max_n);
    pass.ensure_keys_capacity(&device, max_n);
    pass.ensure_histogram_capacity(&device, ((max_n + RADIX_WG_SIZE - 1) / RADIX_WG_SIZE).max(1));
    pass.ensure_parents_capacity(&device, 2 * max_n - 1);
    pass.ensure_aabb_atomic_capacity(&device, (2 * max_n - 1) * 3);

    let nodes_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("test tlas_nodes"),
        size: ((2 * max_n - 1) as u64) * (std::mem::size_of::<TlasNode>() as u64),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let leaves_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("test tlas_leaves"),
        size: (max_n as u64) * (std::mem::size_of::<TlasInstanceLeaf>() as u64),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let mut any_failure = false;
    eprintln!("config                     n   unsorted  cyclic-trials  orphan-trials  (of {TRIALS})");
    for &(tiles, lod1, scattered) in configs {
        let n = (tiles + lod1 + scattered) as u32;
        let mut unsorted = 0usize;
        let mut cyclic_trials = 0usize;
        let mut orphan_trials = 0usize;
        for trial in 0..TRIALS {
            let mut rng = Lcg(0x9E3779B97F4A7C15u64.wrapping_mul(trial as u64 + 1) ^ n as u64);
            let prims = make_scene(tiles, lod1, scattered, &mut rng);
            let r = run_trial(&device, &queue, &pass, &nodes_buffer, &leaves_buffer, &prims);
            if !r.sorted {
                unsorted += 1;
            }
            if r.cycles > 0 {
                cyclic_trials += 1;
            }
            if r.orphans > 0 {
                orphan_trials += 1;
            }
        }
        eprintln!(
            "tiles={tiles:2} lod1={lod1:2} scat={scattered:2}   {n:3}   {unsorted:8}  {cyclic_trials:13}  {orphan_trials:13}"
        );
        if unsorted > 0 || cyclic_trials > 0 {
            any_failure = true;
        }
    }

    assert!(
        !any_failure,
        "GPU radix sort produced unsorted output and/or Karras parents[] cycles — \
         a cyclic trial means propagate_atomic_main would spin forever in production \
         (the Xid 109 desktop freeze). Fix: make the radix scatter stable."
    );
}
