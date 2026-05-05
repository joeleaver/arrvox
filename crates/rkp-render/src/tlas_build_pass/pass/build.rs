//! `TlasBuildPass::build_gpu_tlas` — per-frame end-to-end dispatch chain.
//!
//! Pure-GPU pipeline: assembly → compute_dispatch_args → Morton →
//! 4× radix → atomic-AABB init → Karras leaves + internal →
//! propagate → decode. Every dispatch downstream of assembly fires
//! via `dispatch_workgroups_indirect`, so the host never reads the
//! per-frame primitive count back to CPU. Single submit, no
//! `device.poll`.

use super::super::types::{
    AssembleHostUniform, MortonUniform, RadixUniform, RADIX_BUCKETS, RADIX_PASSES,
    TLAS_DISPATCH_ARG_STRIDE, TLAS_DISPATCH_SLOT_DECODE, TLAS_DISPATCH_SLOT_INIT_ATOMIC,
    TLAS_DISPATCH_SLOT_KARRAS_INTERNAL, TLAS_DISPATCH_SLOT_KARRAS_LEAVES,
    TLAS_DISPATCH_SLOT_MORTON, TLAS_DISPATCH_SLOT_PROPAGATE, TLAS_DISPATCH_SLOT_RADIX,
};

use super::{GpuTlasBuildInputs, TlasBuildPass};

const RADIX_WG_SIZE: u32 = 64;

/// Bytes of a `TlasNode` whose AABB is guaranteed to miss every ray
/// (`min = +∞`, `max = −∞`, so the slab test always yields
/// `t_range.x > t_range.y`). Pre-filled into `tlas_nodes_buffer[0]`
/// before each chain dispatch — the chain overwrites it whenever
/// the actual prim count is ≥ 1; for the empty-after-assembly edge
/// case (`instance_count > 0` but every host instance gets filtered
/// by `assemble_host_main`'s asset-id / shader-id gates) the safe
/// value remains, and the shadow trace's traversal skips node 0
/// without infinite-recursing on stale topology.
const SAFE_TLAS_NODE_BYTES: [u8; 32] = {
    let pos_inf = f32::INFINITY.to_le_bytes();
    let neg_inf = f32::NEG_INFINITY.to_le_bytes();
    let zero = [0u8; 4];
    let mut out = [0u8; 32];
    let mut i = 0;
    while i < 4 {
        out[i] = pos_inf[i];
        out[4 + i] = pos_inf[i];
        out[8 + i] = pos_inf[i];
        out[12 + i] = zero[i]; // left_or_leaf = 0
        out[16 + i] = neg_inf[i];
        out[20 + i] = neg_inf[i];
        out[24 + i] = neg_inf[i];
        out[28 + i] = zero[i]; // right_or_count = 0
        i += 1;
    }
    out
};

impl TlasBuildPass {
    /// Drive the full GPU TLAS build end to end. Encodes assembly →
    /// `compute_dispatch_args` → Morton → 4× radix → init_atomic_aabb →
    /// Karras leaves + internal → propagate → decode, writing the
    /// final `tlas_nodes` + `tlas_leaves` into the supplied
    /// [`crate::tlas_pass::TlasPass`] buffers.
    ///
    /// Returns `inputs.instance_count` — the upper-bound primitive
    /// count, used by the caller to stamp `tlas_pass.last_*_count`.
    /// The actual count is GPU-resident in `tlas_state.prim_count`;
    /// the shadow trace's empty-scene skip uses the dispatch_args
    /// 0-workgroup gating + each shader's own early-out (so a
    /// stamped upper-bound here is harmless).
    pub fn build_gpu_tlas(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        inputs: &GpuTlasBuildInputs,
        tlas_pass: &mut crate::tlas_pass::TlasPass,
        profiler: Option<&wgpu_profiler::GpuProfiler>,
    ) -> u32 {
        let upper_bound = inputs.instance_count;
        if upper_bound == 0 {
            tlas_pass.last_node_count = 0;
            tlas_pass.last_leaf_count = 0;
            return 0;
        }

        // Pre-size every chain buffer to the instance-count upper
        // bound. The actual prim_count is GPU-resident and may be
        // smaller, but indirect dispatch is driven by GPU-written
        // workgroup counts — the buffers just need to fit the
        // upper bound. The CPU never sees the per-frame count.
        self.ensure_prims_capacity(device, upper_bound);
        self.ensure_keys_capacity(device, upper_bound);
        let radix_workgroups_upper = ((upper_bound + RADIX_WG_SIZE - 1) / RADIX_WG_SIZE).max(1);
        self.ensure_histogram_capacity(device, radix_workgroups_upper);
        let total_nodes_upper = (2 * upper_bound).saturating_sub(1).max(1);
        self.ensure_parents_capacity(device, total_nodes_upper);
        let aabb_atomic_entries = total_nodes_upper.saturating_mul(3).max(3);
        self.ensure_aabb_atomic_capacity(device, aabb_atomic_entries);
        tlas_pass.ensure_capacity(device, total_nodes_upper, upper_bound);

        // Pre-fill `tlas_nodes[0]` with a "miss-everything" sentinel
        // so the shadow trace terminates cleanly when assembly
        // produces zero prims (every host instance filtered out).
        // The chain dispatches' first writes to node 0 — by
        // `karras_internal` (N ≥ 2) or `karras_leaves` (N == 1) —
        // overwrite this; for N == 0 they have 0 workgroups and
        // node 0 stays at the safe value.
        queue.write_buffer(&tlas_pass.nodes_buffer, 0, &SAFE_TLAS_NODE_BYTES);

        // Upload all uniforms up-front. The dispatch_args pass needs
        // `prims_capacity` from `host_uniform_buffer`; downstream
        // shaders need scene_min/max (morton) and digit_shift (radix).
        queue.write_buffer(
            &self.host_uniform_buffer,
            0,
            bytemuck::bytes_of(&AssembleHostUniform {
                instance_count: inputs.instance_count,
                asset_count: inputs.asset_count,
                prims_capacity: self.tlas_prims_capacity,
                _pad: 0,
            }),
        );
        queue.write_buffer(
            &self.morton_uniform_buffer,
            0,
            bytemuck::bytes_of(&MortonUniform {
                scene_min: inputs.scene_min,
                _pad0: 0,
                scene_max: inputs.scene_max,
                _pad1: 0,
            }),
        );
        let radix_stride: u64 = 256;
        let mut radix_bytes: Vec<u8> = vec![0u8; (RADIX_PASSES as u64 * radix_stride) as usize];
        for p in 0..RADIX_PASSES {
            let u = RadixUniform {
                digit_shift: p * 8,
                _pad0: 0,
                _pad1: 0,
                _pad2: 0,
            };
            let off = (p as u64 * radix_stride) as usize;
            radix_bytes[off..off + std::mem::size_of::<RadixUniform>()]
                .copy_from_slice(bytemuck::bytes_of(&u));
        }
        queue.write_buffer(&self.radix_uniform_buffer, 0, &radix_bytes);

        // Single command encoder for the entire chain.
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("tlas_build chain"),
        });
        enc.clear_buffer(&self.tlas_prim_count_buffer, 0, Some(4));

        // Per-region profiler scopes. Each call surrounds a single
        // dispatch (or a tight block of them) so the wgpu-profiler
        // results break the TLAS build down into measurable phases.
        // `Option<&GpuProfiler>` keeps the path tests-friendly —
        // tests pass `None` and skip query allocation.
        let begin = |label: &'static str, enc: &mut wgpu::CommandEncoder| {
            profiler.map(|p| p.begin_query(label, enc))
        };
        let end = |enc: &mut wgpu::CommandEncoder,
                   q: Option<wgpu_profiler::GpuProfilerQuery>| {
            if let (Some(p), Some(q)) = (profiler, q) {
                p.end_query(enc, q);
            }
        };

        // ── Assembly ─────────────────────────────────────────────
        let assemble_g0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("tlas_assemble_host g0"),
            layout: &self.host_g0_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: inputs.instances_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: inputs.assets_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: self.tlas_prims_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: self.tlas_prim_count_buffer.as_entire_binding() },
            ],
        });
        let assemble_g1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("tlas_assemble_host g1"),
            layout: &self.host_g1_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: self.host_uniform_buffer.as_entire_binding(),
            }],
        });
        let q = begin("tlas.assemble", &mut enc);
        {
            let mut cpass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("assemble_host_main"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.host_pipeline);
            cpass.set_bind_group(0, &assemble_g0, &[]);
            cpass.set_bind_group(1, &assemble_g1, &[]);
            let wgs = ((inputs.instance_count + 63) / 64).max(1);
            cpass.dispatch_workgroups(wgs, 1, 1);
        }
        end(&mut enc, q);

        // ── Compute dispatch args ────────────────────────────────
        let dispatch_args_g0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("tlas_compute_dispatch_args g0"),
            layout: &self.dispatch_args_g0_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.tlas_prim_count_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.tlas_state_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: self.tlas_dispatch_args_buffer.as_entire_binding() },
            ],
        });
        let dispatch_args_g1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("tlas_compute_dispatch_args g1"),
            layout: &self.dispatch_args_g1_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: self.host_uniform_buffer.as_entire_binding(),
            }],
        });
        let q = begin("tlas.dispatch_args", &mut enc);
        {
            let mut cpass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("compute_dispatch_args_main"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.dispatch_args_pipeline);
            cpass.set_bind_group(0, &dispatch_args_g0, &[]);
            cpass.set_bind_group(1, &dispatch_args_g1, &[]);
            cpass.dispatch_workgroups(1, 1, 1);
        }
        end(&mut enc, q);

        // ── Morton compute ───────────────────────────────────────
        let morton_g0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("morton g0"),
            layout: &self.morton_g0_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.tlas_prims_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.keys_a_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: self.vals_a_buffer.as_entire_binding() },
            ],
        });
        let morton_g1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("morton g1"),
            layout: &self.morton_g1_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.morton_uniform_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.tlas_state_buffer.as_entire_binding() },
            ],
        });
        let q = begin("tlas.morton", &mut enc);
        {
            let mut cpass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("compute_morton_main"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.morton_pipeline);
            cpass.set_bind_group(0, &morton_g0, &[]);
            cpass.set_bind_group(1, &morton_g1, &[]);
            cpass.dispatch_workgroups_indirect(
                &self.tlas_dispatch_args_buffer,
                TLAS_DISPATCH_SLOT_MORTON as u64 * TLAS_DISPATCH_ARG_STRIDE,
            );
        }
        end(&mut enc, q);

        // ── Radix sort — 4 passes ping-ponging a→b→a→b ──────────
        let radix_g0_a_to_b = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("radix g0 a→b"),
            layout: &self.radix_g0_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.keys_a_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.vals_a_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: self.keys_b_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: self.vals_b_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: self.histogram_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: self.scan_offsets_buffer.as_entire_binding() },
            ],
        });
        let radix_g0_b_to_a = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("radix g0 b→a"),
            layout: &self.radix_g0_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.keys_b_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.vals_b_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: self.keys_a_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: self.vals_a_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: self.histogram_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: self.scan_offsets_buffer.as_entire_binding() },
            ],
        });
        let radix_g1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("radix g1"),
            layout: &self.radix_g1_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &self.radix_uniform_buffer,
                        offset: 0,
                        size: std::num::NonZeroU64::new(std::mem::size_of::<RadixUniform>() as u64),
                    }),
                },
                wgpu::BindGroupEntry { binding: 1, resource: self.tlas_state_buffer.as_entire_binding() },
            ],
        });
        let histogram_bytes = (radix_workgroups_upper as u64) * (RADIX_BUCKETS as u64) * 4;
        let q = begin("tlas.radix", &mut enc);
        for p in 0..RADIX_PASSES {
            let g0 = if p % 2 == 0 { &radix_g0_a_to_b } else { &radix_g0_b_to_a };
            let dyn_off = (p as u64 * radix_stride) as u32;
            enc.clear_buffer(&self.histogram_buffer, 0, Some(histogram_bytes));
            {
                let mut cpass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("radix count_main"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&self.radix_count_pipeline);
                cpass.set_bind_group(0, g0, &[]);
                cpass.set_bind_group(1, &radix_g1, &[dyn_off]);
                cpass.dispatch_workgroups_indirect(
                    &self.tlas_dispatch_args_buffer,
                    TLAS_DISPATCH_SLOT_RADIX as u64 * TLAS_DISPATCH_ARG_STRIDE,
                );
            }
            {
                // Scan stays direct — always 1 workgroup of 256 threads.
                // It walks `tlas_state.radix_workgroups` internally.
                let mut cpass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("radix scan_main"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&self.radix_scan_pipeline);
                cpass.set_bind_group(0, g0, &[]);
                cpass.set_bind_group(1, &radix_g1, &[dyn_off]);
                cpass.dispatch_workgroups(1, 1, 1);
            }
            {
                let mut cpass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("radix scatter_main"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&self.radix_scatter_pipeline);
                cpass.set_bind_group(0, g0, &[]);
                cpass.set_bind_group(1, &radix_g1, &[dyn_off]);
                cpass.dispatch_workgroups_indirect(
                    &self.tlas_dispatch_args_buffer,
                    TLAS_DISPATCH_SLOT_RADIX as u64 * TLAS_DISPATCH_ARG_STRIDE,
                );
            }
        }
        end(&mut enc, q);

        // ── Karras tree + atomic AABB propagation ───────────────
        // Order matters: `init_atomic_aabb_main` writes both the
        // ±∞ accumulator sentinels AND `parents[i] = PARENT_SENTINEL`,
        // and must run BEFORE `build_internal_main` (which overwrites
        // `parents[non-root]` with real values). The root keeps the
        // sentinel so propagate's walk-up loop terminates.
        let karras_g0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("karras g0"),
            layout: &self.karras_g0_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.keys_a_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.vals_a_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: self.tlas_prims_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: tlas_pass.nodes_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: tlas_pass.leaves_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: self.parents_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: self.aabb_min_atomic_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: self.aabb_max_atomic_buffer.as_entire_binding() },
            ],
        });
        let karras_g1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("karras g1"),
            layout: &self.karras_g1_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: self.tlas_state_buffer.as_entire_binding(),
            }],
        });
        // 1. Init — clears ±∞ sentinels into both atomic AABB
        //    buffers and seeds `parents[i] = PARENT_SENTINEL`.
        let q = begin("tlas.init_atomic", &mut enc);
        {
            let mut cpass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("init_atomic_aabb_main"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.init_atomic_aabb_pipeline);
            cpass.set_bind_group(0, &karras_g0, &[]);
            cpass.set_bind_group(1, &karras_g1, &[]);
            cpass.dispatch_workgroups_indirect(
                &self.tlas_dispatch_args_buffer,
                TLAS_DISPATCH_SLOT_INIT_ATOMIC as u64 * TLAS_DISPATCH_ARG_STRIDE,
            );
        }
        end(&mut enc, q);
        // 2. Karras leaves — packs `tlas_leaves` payload + leaf-marker
        //    `tlas_nodes` entries.
        let q = begin("tlas.karras_leaves", &mut enc);
        {
            let mut cpass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("build_leaves_main"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.karras_leaves_pipeline);
            cpass.set_bind_group(0, &karras_g0, &[]);
            cpass.set_bind_group(1, &karras_g1, &[]);
            cpass.dispatch_workgroups_indirect(
                &self.tlas_dispatch_args_buffer,
                TLAS_DISPATCH_SLOT_KARRAS_LEAVES as u64 * TLAS_DISPATCH_ARG_STRIDE,
            );
        }
        end(&mut enc, q);
        // 3. Karras internal — assigns child indices and writes
        //    real parent pointers (overwriting init's sentinels for
        //    non-root nodes).
        let q = begin("tlas.karras_internal", &mut enc);
        {
            let mut cpass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("build_internal_main"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.karras_internal_pipeline);
            cpass.set_bind_group(0, &karras_g0, &[]);
            cpass.set_bind_group(1, &karras_g1, &[]);
            cpass.dispatch_workgroups_indirect(
                &self.tlas_dispatch_args_buffer,
                TLAS_DISPATCH_SLOT_KARRAS_INTERNAL as u64 * TLAS_DISPATCH_ARG_STRIDE,
            );
        }
        end(&mut enc, q);
        // 4. Propagate — atomicMin/Max walks each leaf up to root.
        let q = begin("tlas.propagate", &mut enc);
        {
            let mut cpass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("propagate_atomic_main"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.propagate_atomic_pipeline);
            cpass.set_bind_group(0, &karras_g0, &[]);
            cpass.set_bind_group(1, &karras_g1, &[]);
            cpass.dispatch_workgroups_indirect(
                &self.tlas_dispatch_args_buffer,
                TLAS_DISPATCH_SLOT_PROPAGATE as u64 * TLAS_DISPATCH_ARG_STRIDE,
            );
        }
        end(&mut enc, q);
        // 5. Decode — read accumulators, write `tlas_nodes[i].aabb_*`.
        let q = begin("tlas.decode", &mut enc);
        {
            let mut cpass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("decode_aabb_main"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.decode_aabb_pipeline);
            cpass.set_bind_group(0, &karras_g0, &[]);
            cpass.set_bind_group(1, &karras_g1, &[]);
            cpass.dispatch_workgroups_indirect(
                &self.tlas_dispatch_args_buffer,
                TLAS_DISPATCH_SLOT_DECODE as u64 * TLAS_DISPATCH_ARG_STRIDE,
            );
        }
        end(&mut enc, q);

        queue.submit(std::iter::once(enc.finish()));

        // Stamp the upper bound. The shadow trace's empty-scene skip
        // checks `tlas_node_count == 0`; an upper-bound stamp here
        // is harmless when the actual GPU tree is smaller because
        // the unused leaf nodes have aabb=(0,0,0) and won't be hit
        // by any ray, while internal nodes beyond the real root
        // chain are unreachable from index 0.
        tlas_pass.last_node_count = total_nodes_upper;
        tlas_pass.last_leaf_count = upper_bound;
        upper_bound
    }
}
