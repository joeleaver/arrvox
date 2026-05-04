//! `TlasBuildPass::build_gpu_tlas` — per-frame end-to-end dispatch chain:
//! assembly → readback → Morton → 4× radix → Karras leaves+internal → AABB propagation.

use super::super::types::{
    AssembleHostUniform, KarrasUniform, MortonUniform, RadixUniform, RADIX_BUCKETS,
    RADIX_PASSES,
};

use super::{GpuTlasBuildInputs, TlasBuildPass};

impl TlasBuildPass {
    /// Drive the full GPU TLAS build (Sessions 1-4) end to end.
    /// Encodes assembly → readback → Morton → 4× radix → Karras
    /// leaves + internal → AABB propagation, writing the final
    /// `tlas_nodes` + `tlas_leaves` into the supplied
    /// [`crate::tlas_pass::TlasPass`] buffers (which the shadow
    /// trace already binds).
    ///
    /// Returns the actual primitive count after assembly (= number
    /// of leaves in the built TLAS). Caller stamps this into
    /// `tlas_pass.last_node_count = 2N-1` and
    /// `tlas_pass.last_leaf_count = N` so the shadow trace's empty-
    /// scene skip works (the WGSL early-outs when `tlas_node_count
    /// == 0`).
    ///
    /// V1 uses a synchronous readback between assembly and the
    /// downstream chain — `device.poll(wait_indefinitely)` blocks
    /// the calling thread for ~1 ms per frame. Acceptable for V1
    /// (we're trading 1 ms here to save 30+ ms of shadow trace);
    /// future refactor to indirect dispatch would remove the stall.
    pub fn build_gpu_tlas(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        inputs: &GpuTlasBuildInputs,
        tlas_pass: &mut crate::tlas_pass::TlasPass,
    ) -> u32 {
        let upper_bound = inputs.instance_count;
        if upper_bound == 0 {
            tlas_pass.last_node_count = 0;
            tlas_pass.last_leaf_count = 0;
            return 0;
        }

        // Capacities for the assembly stage.
        self.ensure_prims_capacity(device, upper_bound);

        // Assemble + count readback. Single submit + map_async +
        // device.poll. The blocking poll is the 1 ms stall.
        let mut enc1 = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("tlas_build assemble"),
        });
        enc1.clear_buffer(&self.tlas_prim_count_buffer, 0, Some(4));

        // Upload assembly uniform.
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

        // Host assembly bind groups + dispatch.
        if inputs.instance_count > 0 {
            let g0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("tlas_assemble_host g0"),
                layout: &self.host_g0_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: inputs.instances_buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: inputs.assets_buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: self.tlas_prims_buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: self.tlas_prim_count_buffer.as_entire_binding() },
                ],
            });
            let g1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("tlas_assemble_host g1"),
                layout: &self.host_g1_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.host_uniform_buffer.as_entire_binding(),
                }],
            });
            let mut cpass = enc1.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("assemble_host_main"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.host_pipeline);
            cpass.set_bind_group(0, &g0, &[]);
            cpass.set_bind_group(1, &g1, &[]);
            let wgs = ((inputs.instance_count + 63) / 64).max(1);
            cpass.dispatch_workgroups(wgs, 1, 1);
        }

        // Copy count to staging for readback.
        enc1.copy_buffer_to_buffer(&self.tlas_prim_count_buffer, 0, &self.count_staging_buffer, 0, 4);
        queue.submit(std::iter::once(enc1.finish()));

        // Synchronous readback. Stalls the engine thread ~1 ms.
        let slice = self.count_staging_buffer.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("device poll for tlas count readback");
        let raw_count = {
            let view = slice.get_mapped_range();
            let c = u32::from_le_bytes(view[0..4].try_into().unwrap());
            drop(view);
            self.count_staging_buffer.unmap();
            c
        };
        // Clamp the readback to actual capacity. If the assembly
        // atomic recorded more attempted writes than the buffer
        // holds (overflow), the writes themselves were gated by
        // `if (slot >= u.prims_capacity) return;`; the counter
        // just kept incrementing for telemetry.
        let actual_count = raw_count.min(self.tlas_prims_capacity);
        let upper_bound = inputs.instance_count;
        if raw_count > upper_bound || (raw_count == 0 && upper_bound > 0) {
            eprintln!(
                "[tlas_build] suspect raw={raw_count} upper={upper_bound} host={}",
                inputs.instance_count,
            );
        }

        if actual_count == 0 {
            tlas_pass.last_node_count = 0;
            tlas_pass.last_leaf_count = 0;
            return 0;
        }

        // Capacities for the downstream chain.
        self.ensure_keys_capacity(device, actual_count);
        let radix_workgroups = ((actual_count + 63) / 64).max(1);
        self.ensure_histogram_capacity(device, radix_workgroups);
        self.ensure_parents_capacity(device, 2 * actual_count - 1);
        // Phase 7c.6 — atomic AABB accumulators sized 3 × (2N-1)
        // u32s each (one per axis per node). Init pass clears
        // them to ±∞ sentinels at frame start; `propagate_atomic_main`
        // walks each leaf up the parent chain applying atomicMin/Max.
        let total_nodes = (2 * actual_count).saturating_sub(1).max(1);
        let aabb_atomic_entries = total_nodes.saturating_mul(3).max(3);
        self.ensure_aabb_atomic_capacity(device, aabb_atomic_entries);
        tlas_pass.ensure_capacity(device, 2 * actual_count - 1, actual_count);

        // Upload uniforms for the chain.
        queue.write_buffer(
            &self.morton_uniform_buffer,
            0,
            bytemuck::bytes_of(&MortonUniform {
                scene_min: inputs.scene_min,
                _pad0: 0,
                scene_max: inputs.scene_max,
                prim_count: actual_count,
            }),
        );
        let radix_stride: u64 = 256;
        let mut radix_bytes: Vec<u8> = vec![0u8; (RADIX_PASSES as u64 * radix_stride) as usize];
        for p in 0..RADIX_PASSES {
            let u = RadixUniform {
                prim_count: actual_count,
                digit_shift: p * 8,
                num_workgroups: radix_workgroups,
                _pad: 0,
            };
            let off = (p as u64 * radix_stride) as usize;
            radix_bytes[off..off + std::mem::size_of::<RadixUniform>()]
                .copy_from_slice(bytemuck::bytes_of(&u));
        }
        queue.write_buffer(&self.radix_uniform_buffer, 0, &radix_bytes);
        queue.write_buffer(
            &self.karras_uniform_buffer,
            0,
            bytemuck::bytes_of(&KarrasUniform {
                prim_count: actual_count,
                _pad0: 0,
                _pad1: 0,
                _pad2: 0,
            }),
        );

        // Init parents (sentinel) + visit_counter (zero).
        let parents_init: Vec<u32> = vec![0xFFFFFFFFu32; (2 * actual_count - 1) as usize];
        queue.write_buffer(&self.parents_buffer, 0, bytemuck::cast_slice(&parents_init));
        // Phase 7c.6 — `init_atomic_aabb_main` (in the dispatch
        // chain below) writes ±∞ sentinels to both atomic
        // buffers; no CPU pre-fill needed. Just declare we're
        // about to use them at the new size.
        let _ = total_nodes;

        // Encode the chain.
        let mut enc2 = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("tlas_build chain"),
        });

        // Morton compute.
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
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: self.morton_uniform_buffer.as_entire_binding(),
            }],
        });
        {
            let mut cpass = enc2.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("compute_morton_main"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.morton_pipeline);
            cpass.set_bind_group(0, &morton_g0, &[]);
            cpass.set_bind_group(1, &morton_g1, &[]);
            cpass.dispatch_workgroups(radix_workgroups, 1, 1);
        }

        // Radix sort — 4 passes ping-ponging a→b→a→b.
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
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &self.radix_uniform_buffer,
                    offset: 0,
                    size: std::num::NonZeroU64::new(std::mem::size_of::<RadixUniform>() as u64),
                }),
            }],
        });
        let histogram_bytes = (radix_workgroups as u64) * (RADIX_BUCKETS as u64) * 4;
        for p in 0..RADIX_PASSES {
            let g0 = if p % 2 == 0 { &radix_g0_a_to_b } else { &radix_g0_b_to_a };
            let dyn_off = (p as u64 * radix_stride) as u32;
            enc2.clear_buffer(&self.histogram_buffer, 0, Some(histogram_bytes));
            {
                let mut cpass = enc2.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("radix count_main"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&self.radix_count_pipeline);
                cpass.set_bind_group(0, g0, &[]);
                cpass.set_bind_group(1, &radix_g1, &[dyn_off]);
                cpass.dispatch_workgroups(radix_workgroups, 1, 1);
            }
            {
                let mut cpass = enc2.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("radix scan_main"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&self.radix_scan_pipeline);
                cpass.set_bind_group(0, g0, &[]);
                cpass.set_bind_group(1, &radix_g1, &[dyn_off]);
                cpass.dispatch_workgroups(1, 1, 1);
            }
            {
                let mut cpass = enc2.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("radix scatter_main"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&self.radix_scatter_pipeline);
                cpass.set_bind_group(0, g0, &[]);
                cpass.set_bind_group(1, &radix_g1, &[dyn_off]);
                cpass.dispatch_workgroups(radix_workgroups, 1, 1);
            }
        }

        // Karras tree + AABB propagation. Output goes into the
        // shadow-trace consumer buffers (`tlas_pass.{nodes,leaves}_buffer`).
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
                resource: self.karras_uniform_buffer.as_entire_binding(),
            }],
        });
        let leaf_wgs = ((actual_count + 63) / 64).max(1);
        {
            let mut cpass = enc2.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("build_leaves_main"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.karras_leaves_pipeline);
            cpass.set_bind_group(0, &karras_g0, &[]);
            cpass.set_bind_group(1, &karras_g1, &[]);
            cpass.dispatch_workgroups(leaf_wgs, 1, 1);
        }
        if actual_count >= 2 {
            let internal_wgs = (((actual_count - 1) + 63) / 64).max(1);
            {
                let mut cpass = enc2.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("build_internal_main"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&self.karras_internal_pipeline);
                cpass.set_bind_group(0, &karras_g0, &[]);
                cpass.set_bind_group(1, &karras_g1, &[]);
                cpass.dispatch_workgroups(internal_wgs, 1, 1);
            }
        }
        // Phase 7c.6 — atomic AABB propagation. Three passes:
        //   1. init: clear accumulators to ±∞ sentinels.
        //   2. propagate: each leaf walks up to root, atomic-min/max
        //      into ancestors. Commutative — no thread ordering
        //      issues, no cross-buffer memory visibility needed.
        //   3. decode: read accumulators, write tlas_nodes AABBs.
        let total_node_wgs = (((2 * actual_count - 1) + 63) / 64).max(1);
        let internal_wgs = if actual_count >= 2 {
            (((actual_count - 1) + 63) / 64).max(1)
        } else {
            1
        };
        if actual_count >= 2 {
            {
                let mut cpass = enc2.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("init_atomic_aabb_main"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&self.init_atomic_aabb_pipeline);
                cpass.set_bind_group(0, &karras_g0, &[]);
                cpass.set_bind_group(1, &karras_g1, &[]);
                cpass.dispatch_workgroups(total_node_wgs, 1, 1);
            }
            {
                let mut cpass = enc2.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("propagate_atomic_main"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&self.propagate_atomic_pipeline);
                cpass.set_bind_group(0, &karras_g0, &[]);
                cpass.set_bind_group(1, &karras_g1, &[]);
                cpass.dispatch_workgroups(leaf_wgs, 1, 1);
            }
            {
                let mut cpass = enc2.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("decode_aabb_main"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&self.decode_aabb_pipeline);
                cpass.set_bind_group(0, &karras_g0, &[]);
                cpass.set_bind_group(1, &karras_g1, &[]);
                cpass.dispatch_workgroups(internal_wgs, 1, 1);
            }
        }

        queue.submit(std::iter::once(enc2.finish()));

        tlas_pass.last_node_count = 2 * actual_count - 1;
        tlas_pass.last_leaf_count = actual_count;
        actual_count
    }
}
