//! GPU procedural evaluator — "sample N positions" compute pipeline.
//!
//! Dispatches a compute shader that reads a batch of query positions,
//! runs the same RPN interpreter the live raymarch uses (`proc_eval.wgsl`),
//! and writes a packed result per position. Phase 3 swaps the voxel bake
//! from a per-sample CPU callback to batched calls through this
//! evaluator, so the procedural math lives in one place (WGSL) and the
//! CPU-side mirror can be deleted in Phase 4.
//!
//! Scope of this module is deliberately narrow:
//!   * `evaluate` runs one dispatch — caller chunks large inputs.
//!   * Buffers grow to the high-water mark and are reused across calls
//!     (bakes dispatch many times with increasing/decreasing counts).
//!   * Readback is blocking — caller owns whatever async state-machine
//!     the bake needs (Phase 3 scope, not ours).

use bytemuck::{Pod, Zeroable};
use glam::Vec3;
use rkp_procedural::flatten::ProcInstruction;

use crate::validate_wgsl;

/// Per-dispatch parameters (uniform).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default)]
struct SampleParams {
    instruction_count: u32,
    position_count: u32,
    _pad0: u32,
    _pad1: u32,
}

/// One result per input position. Matches `SampleResult` in `proc_sample.wgsl`
/// byte-for-byte, including the three trailing `_pad` words that round the
/// struct to a 32-byte stride.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default, Debug)]
pub struct SampleResult {
    pub distance: f32,
    /// Primary material id (low 16 bits used; upper 16 zero).
    pub primary: u32,
    /// Secondary material id (low 16 bits used; upper 16 zero).
    pub secondary: u32,
    /// Quantized blend weight, 0..15 (low 4 bits used; upper bits zero).
    pub blend_u4: u32,
    /// Per-voxel color packed as `R | (G << 8) | (B << 16) | (0xFF << 24)`.
    /// The high byte is a non-zero sentinel — the voxel-path shader
    /// treats `color_pool[id] == 0` as "no override" and falls back to
    /// the material's base color, so user-written black (0,0,0) would
    /// otherwise be indistinguishable from "unset".
    pub color: u32,
    pub _pad0: u32,
    pub _pad1: u32,
    pub _pad2: u32,
}

impl SampleResult {
    /// Narrow to the tuple shape `voxelize_octree`'s callback expects.
    /// Order matches the `sdf_fn` signature: distance, primary material,
    /// secondary material, quantized blend weight, packed color.
    #[inline]
    pub fn into_tuple(self) -> (f32, u16, u16, u8, u32) {
        (
            self.distance,
            self.primary as u16,
            self.secondary as u16,
            self.blend_u4 as u8,
            self.color,
        )
    }
}

/// Wall-clock breakdown accumulated across every chunk in one `evaluate`
/// call. `submit_to_map_ready` is the interesting one for the async-bake
/// decision — it's the portion where the CPU main thread has no work to
/// do except wait for the GPU to finish.
#[derive(Default)]
struct EvalStats {
    chunks: u32,
    upload: std::time::Duration,
    encode: std::time::Duration,
    submit_to_map_ready: std::time::Duration,
    readback: std::time::Duration,
}

pub struct GpuEvaluator {
    /// Pipeline variants keyed on the `HAS_POS_WARPS` shader
    /// override. `evaluate` picks the simple pipeline when the
    /// flattened tree contains no NoiseDisplace/Mirror PUSH opcodes,
    /// letting the compiler dead-strip the `pos_stack`. See
    /// `proc_eval.wgsl` + `proc_eval_types.wgsl` for the WGSL side.
    pipeline_simple: wgpu::ComputePipeline,
    pipeline_warps: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,

    params_buf: wgpu::Buffer,
    // Storage buffers that grow on demand. `None` until the first
    // evaluate call sizes them.
    instructions_buf: Option<wgpu::Buffer>,
    instructions_cap: usize,
    positions_buf: Option<wgpu::Buffer>,
    positions_cap: usize,
    results_buf: Option<wgpu::Buffer>,
    staging_buf: Option<wgpu::Buffer>,
    results_cap: usize,

    /// Reusable scratch buffer for padding `Vec3` → `[f32; 4]` before
    /// `write_buffer`. Grows once to the high-water chunk size (up to
    /// 8 M entries = 128 MiB) and stays allocated across bakes —
    /// orders of magnitude cheaper than re-allocating per chunk.
    padding_scratch: Vec<[f32; 4]>,

    /// Cap for one dispatch's `positions.len()`, computed at
    /// construction from the device's storage-buffer binding limit and
    /// the workgroup-count ceiling. The hosted device may report
    /// anywhere from 128 MiB (wgpu default) to the adapter's max, so
    /// this is per-instance, not a const.
    max_positions_per_dispatch: usize,
}

impl GpuEvaluator {
    pub fn new(device: &wgpu::Device) -> Self {
        // Concat order matches proc_raymarch: types → this shader's
        // bindings + entry → shared function bodies.
        let types_src = include_str!("shaders/proc_eval_types.wgsl");
        let sample_src = include_str!("shaders/proc_sample.wgsl");
        let eval_src = include_str!("shaders/proc_eval.wgsl");
        let shader_src = format!("{types_src}\n{sample_src}\n{eval_src}");
        validate_wgsl(&shader_src, "proc_sample");

        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("proc_sample"),
            source: wgpu::ShaderSource::Wgsl(shader_src.into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("proc_sample bind group layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("proc_sample pipeline layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let make_pipeline = |label: &str, has_warps: bool| -> wgpu::ComputePipeline {
            let overrides: &[(&str, f64)] = if has_warps {
                &[("HAS_POS_WARPS", 1.0)]
            } else {
                &[("HAS_POS_WARPS", 0.0)]
            };
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: Some("main"),
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: overrides,
                    zero_initialize_workgroup_memory: false,
                },
                cache: None,
            })
        };

        let pipeline_simple = make_pipeline("proc_sample (simple)", false);
        let pipeline_warps = make_pipeline("proc_sample (warps)", true);

        let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("proc_sample params"),
            size: std::mem::size_of::<SampleParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Two ceilings on a single dispatch:
        //   1. Workgroup-count limit: wgpu caps per-dimension dispatch
        //      at 65 535. With `@workgroup_size(64)` that's 4 194 240
        //      invocations (rounded down to a multiple of 64).
        //   2. Storage-buffer binding limit: the results buffer holds
        //      `N * sizeof(SampleResult)` bytes and binds in one go.
        //      `Limits::default()` allows 128 MiB (rinch creates the
        //      device with defaults today). Cap N to fit.
        // Round down to a multiple of 64 so the dispatch boundary lines
        // up with the workgroup edge.
        let workgroup_cap = 65_535usize * 64;
        let result_size = std::mem::size_of::<SampleResult>();
        let binding_cap = (device.limits().max_storage_buffer_binding_size as usize)
            / result_size;
        let max_positions_per_dispatch = workgroup_cap.min(binding_cap) & !63;

        Self {
            pipeline_simple,
            pipeline_warps,
            bind_group_layout,
            params_buf,
            instructions_buf: None,
            instructions_cap: 0,
            positions_buf: None,
            positions_cap: 0,
            results_buf: None,
            staging_buf: None,
            results_cap: 0,
            padding_scratch: Vec::new(),
            max_positions_per_dispatch,
        }
    }

    /// Evaluate the tree at every position. If the batch exceeds the
    /// per-dispatch cap, splits into chunks transparently — caller
    /// sees one `Vec<SampleResult>` that matches the input length.
    ///
    /// Blocks until results are readable. Instruction upload is hoisted
    /// out of the chunk loop: the opcode stream is identical across all
    /// chunks so uploading once saves N-1 `write_buffer`s per bake.
    pub fn evaluate(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        positions: &[Vec3],
        instructions: &[ProcInstruction],
    ) -> Vec<SampleResult> {
        if positions.is_empty() {
            return Vec::new();
        }

        let t_start = std::time::Instant::now();
        // Upload instructions once, reuse across every chunk.
        let instructions_pad: Vec<ProcInstruction>;
        let instructions_slice: &[ProcInstruction] = if instructions.is_empty() {
            instructions_pad = vec![ProcInstruction::zeroed()];
            &instructions_pad
        } else {
            instructions
        };
        self.ensure_instructions_capacity(device, instructions_slice.len());
        queue.write_buffer(
            self.instructions_buf.as_ref().unwrap(),
            0,
            bytemuck::cast_slice(instructions_slice),
        );

        let has_warps = instructions.iter().any(|ins| {
            ins.op == rkp_procedural::OpKind::PushNoiseDisplace as u32
                || ins.op == rkp_procedural::OpKind::PushMirror as u32
                || ins.op == rkp_procedural::OpKind::PushArray as u32
        });
        let instruction_count = instructions.len() as u32;

        let mut stats = EvalStats::default();
        let chunk_cap = self.max_positions_per_dispatch;
        let out = if positions.len() <= chunk_cap {
            self.evaluate_chunk(
                device, queue, positions, instruction_count, has_warps, &mut stats,
            )
        } else {
            let mut out: Vec<SampleResult> = Vec::with_capacity(positions.len());
            for chunk in positions.chunks(chunk_cap) {
                out.extend(self.evaluate_chunk(
                    device, queue, chunk, instruction_count, has_warps, &mut stats,
                ));
            }
            out
        };
        let ms = |d: std::time::Duration| d.as_secs_f32() * 1000.0;
        eprintln!(
            "[gpu_eval] N={} ins={} chunks={} upload={:.2}ms encode={:.2}ms \
             submit_to_map_ready={:.2}ms readback={:.2}ms total={:.2}ms",
            positions.len(),
            instructions.len(),
            stats.chunks,
            ms(stats.upload),
            ms(stats.encode),
            ms(stats.submit_to_map_ready),
            ms(stats.readback),
            ms(t_start.elapsed()),
        );
        out
    }

    /// Single-dispatch evaluation. Caller guarantees
    /// `positions.len() <= MAX_POSITIONS_PER_DISPATCH` and has already
    /// uploaded `instructions` + computed `has_warps`.
    fn evaluate_chunk(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        positions: &[Vec3],
        instruction_count: u32,
        has_warps: bool,
        stats: &mut EvalStats,
    ) -> Vec<SampleResult> {
        debug_assert!(!positions.is_empty());
        debug_assert!(positions.len() <= self.max_positions_per_dispatch);
        stats.chunks += 1;

        let t_upload_start = std::time::Instant::now();
        self.ensure_positions_capacity(device, positions.len());
        self.ensure_results_capacity(device, positions.len());

        // Pad positions to vec4 (16 B stride, matching the shader's
        // `array<vec4<f32>>`). Reuses `padding_scratch` to avoid a
        // fresh 128 MiB allocation on every chunk.
        self.padding_scratch.clear();
        self.padding_scratch.reserve(positions.len());
        for p in positions {
            self.padding_scratch.push([p.x, p.y, p.z, 0.0]);
        }
        queue.write_buffer(
            self.positions_buf.as_ref().unwrap(),
            0,
            bytemuck::cast_slice(&self.padding_scratch),
        );

        let params = SampleParams {
            instruction_count,
            position_count: positions.len() as u32,
            _pad0: 0,
            _pad1: 0,
        };
        queue.write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&params));
        stats.upload += t_upload_start.elapsed();

        let t_encode_start = std::time::Instant::now();
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("proc_sample bind group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.instructions_buf.as_ref().unwrap().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.positions_buf.as_ref().unwrap().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.results_buf.as_ref().unwrap().as_entire_binding(),
                },
            ],
        });

        let results_size_bytes =
            (positions.len() * std::mem::size_of::<SampleResult>()) as u64;

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("proc_sample encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("proc_sample"),
                timestamp_writes: None,
            });
            let pipeline = if has_warps {
                &self.pipeline_warps
            } else {
                &self.pipeline_simple
            };
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // 64 threads / workgroup → ceil(N / 64) workgroups.
            let groups = positions.len().div_ceil(64) as u32;
            pass.dispatch_workgroups(groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(
            self.results_buf.as_ref().unwrap(),
            0,
            self.staging_buf.as_ref().unwrap(),
            0,
            results_size_bytes,
        );

        queue.submit(Some(encoder.finish()));
        stats.encode += t_encode_start.elapsed();

        // Map staging and read back. `device.poll` drives the async
        // map completion on native; we block until it reports Success.
        let t_map_start = std::time::Instant::now();
        let staging = self.staging_buf.as_ref().unwrap();
        let slice = staging.slice(0..results_size_bytes);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("device poll");
        rx.recv().expect("map_async channel").expect("map_async result");
        stats.submit_to_map_ready += t_map_start.elapsed();

        let t_readback_start = std::time::Instant::now();
        let results: Vec<SampleResult> = {
            let data = slice.get_mapped_range();
            bytemuck::cast_slice::<u8, SampleResult>(&data).to_vec()
        };
        staging.unmap();
        stats.readback += t_readback_start.elapsed();
        results
    }

    fn ensure_instructions_capacity(&mut self, device: &wgpu::Device, needed: usize) {
        if self.instructions_cap < needed {
            // 1.5× headroom to amortize reallocs across growing bakes.
            let new_cap = (needed.max(64) * 3) / 2;
            let buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("proc_sample instructions"),
                size: (new_cap * std::mem::size_of::<ProcInstruction>()) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.instructions_buf = Some(buf);
            self.instructions_cap = new_cap;
        }
    }

    fn ensure_positions_capacity(&mut self, device: &wgpu::Device, needed: usize) {
        if self.positions_cap < needed {
            // 1.5× growth amortizes reallocations across the levels of a
            // single bake (small classify dispatches → big brick batch),
            // but never grow past the per-dispatch cap — past it the
            // buffer would exceed the device's storage-binding limit
            // and `create_bind_group` would refuse it.
            let new_cap = ((needed.max(1024) * 3) / 2)
                .min(self.max_positions_per_dispatch);
            let buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("proc_sample positions"),
                // 16 bytes per padded vec4 position.
                size: (new_cap * 16) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.positions_buf = Some(buf);
            self.positions_cap = new_cap;
        }
    }

    fn ensure_results_capacity(&mut self, device: &wgpu::Device, needed: usize) {
        if self.results_cap < needed {
            let new_cap = ((needed.max(1024) * 3) / 2)
                .min(self.max_positions_per_dispatch);
            let size = (new_cap * std::mem::size_of::<SampleResult>()) as u64;
            let storage = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("proc_sample results"),
                size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let staging = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("proc_sample results staging"),
                size,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.results_buf = Some(storage);
            self.staging_buf = Some(staging);
            self.results_cap = new_cap;
        }
    }
}
