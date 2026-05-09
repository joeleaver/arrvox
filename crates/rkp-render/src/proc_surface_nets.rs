//! GPU surface-nets-from-SDF spike.
//!
//! Takes a flattened procedural tree and emits a triangle mesh on a
//! dense `N³` cell grid in three compute passes (classify cells →
//! emit vertex per active SN cube → emit triangle quads per active
//! sample-edge). Skips the octree / brick / DAG / meshlet stack —
//! this is the proxy-mesh path for "render a procedural without baking
//! it into voxels".
//!
//! Shape mirrors `proc_sample::GpuEvaluator`: own pipelines + bind
//! group layout, growable storage buffers reused across calls,
//! readback is blocking. Caller-friendly: one `extract` call per
//! procedural per re-mesh.
//!
//! Vertex layout matches `rkp_core::mesh_extract::MeshVertex` (32 B)
//! so feeding the result into the existing mesh raster pipeline is a
//! later plumbing step, not another data conversion.

use bytemuck::{Pod, Zeroable};
use glam::Vec3;
use rkp_procedural::flatten::ProcInstruction;

use crate::compile_pass_shader;

/// Per-dispatch parameters (uniform).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default)]
struct SnParams {
    aabb_min: [f32; 3],
    cell_size: f32,
    grid_n: u32,
    instruction_count: u32,
    vertex_cap: u32,
    index_cap: u32,
}

/// 32 B — same layout as `rkp_core::mesh_extract::MeshVertex`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default, Debug)]
pub struct MeshVertex {
    pub local_pos: [f32; 3],
    pub normal_oct: u32,
    pub leaf_attr_id: u32,
    pub bone_indices: u32,
    pub bone_weights: u32,
    pub _pad: u32,
}

const _: () = assert!(std::mem::size_of::<MeshVertex>() == 32);

/// Output of one extraction call.
pub struct SurfaceMesh {
    pub vertices: Vec<MeshVertex>,
    pub indices: Vec<u32>,
}

/// Per-pass timings, accumulated across one `extract` call.
#[derive(Default, Debug, Clone, Copy)]
pub struct ExtractStats {
    pub vertex_count: u32,
    pub index_count: u32,
    pub upload: std::time::Duration,
    pub classify: std::time::Duration,
    pub vertex_emit: std::time::Duration,
    pub index_emit: std::time::Duration,
    pub readback: std::time::Duration,
    pub total: std::time::Duration,
}

pub struct GpuSurfaceNets {
    pipeline_classify_simple: wgpu::ComputePipeline,
    pipeline_classify_warps: wgpu::ComputePipeline,
    pipeline_vertex_simple: wgpu::ComputePipeline,
    pipeline_vertex_warps: wgpu::ComputePipeline,
    pipeline_index: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,

    params_buf: wgpu::Buffer,

    instructions_buf: Option<wgpu::Buffer>,
    instructions_cap: usize,
    cell_solid_buf: Option<wgpu::Buffer>,
    cell_solid_cap: u64,
    cube_map_buf: Option<wgpu::Buffer>,
    cube_map_cap: u64,
    /// Two atomic counters laid out as `[vertex_count, index_count]`.
    counters_buf: wgpu::Buffer,
    counters_staging: wgpu::Buffer,
    vertices_buf: Option<wgpu::Buffer>,
    vertices_cap: u64,
    vertices_staging: Option<wgpu::Buffer>,
    indices_buf: Option<wgpu::Buffer>,
    indices_cap: u64,
    indices_staging: Option<wgpu::Buffer>,
}

impl GpuSurfaceNets {
    pub fn new(device: &wgpu::Device) -> Self {
        let module = compile_pass_shader(
            device,
            wesl::include_wesl!("proc_surface_nets"),
            "proc_surface_nets",
        );

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("proc_surface_nets bind group layout"),
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
                // bindings 2..7 are all read_write storage
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
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
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
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
            label: Some("proc_surface_nets pipeline layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let make_pipeline =
            |label: &str, entry: &str, has_warps: bool| -> wgpu::ComputePipeline {
                let overrides: &[(&str, f64)] = if has_warps {
                    &[("HAS_POS_WARPS", 1.0)]
                } else {
                    &[("HAS_POS_WARPS", 0.0)]
                };
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(label),
                    layout: Some(&pipeline_layout),
                    module: &module,
                    entry_point: Some(entry),
                    compilation_options: wgpu::PipelineCompilationOptions {
                        constants: overrides,
                        zero_initialize_workgroup_memory: false,
                    },
                    cache: None,
                })
            };

        let pipeline_classify_simple = make_pipeline("sn classify (simple)", "classify", false);
        let pipeline_classify_warps = make_pipeline("sn classify (warps)", "classify", true);
        let pipeline_vertex_simple = make_pipeline("sn vertex (simple)", "vertex_emit", false);
        let pipeline_vertex_warps = make_pipeline("sn vertex (warps)", "vertex_emit", true);
        // `index_emit` doesn't call into the procedural evaluator, so it
        // doesn't care about HAS_POS_WARPS — we only need one variant.
        // Pick the simple form so the dead-stripped pos_stack stays out
        // of the artifact.
        let pipeline_index = make_pipeline("sn index", "index_emit", false);

        let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sn params"),
            size: std::mem::size_of::<SnParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let counters_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sn counters"),
            size: 8,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let counters_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sn counters staging"),
            size: 8,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            pipeline_classify_simple,
            pipeline_classify_warps,
            pipeline_vertex_simple,
            pipeline_vertex_warps,
            pipeline_index,
            bind_group_layout,
            params_buf,
            instructions_buf: None,
            instructions_cap: 0,
            cell_solid_buf: None,
            cell_solid_cap: 0,
            cube_map_buf: None,
            cube_map_cap: 0,
            counters_buf,
            counters_staging,
            vertices_buf: None,
            vertices_cap: 0,
            vertices_staging: None,
            indices_buf: None,
            indices_cap: 0,
            indices_staging: None,
        }
    }

    /// Mesh the procedural at `grid_n³` cell resolution over the AABB.
    ///
    /// `vertex_cap` / `index_cap` size the output buffers. A safe rule
    /// of thumb is `grid_n² * 4` for vertices and `grid_n² * 24` for
    /// indices on a typical surfacy procedural — surface area scales
    /// as `O(N²)`, not `O(N³)`. The shader silently truncates if you
    /// undersize them; the returned `ExtractStats` reports actual
    /// counts so you can detect truncation.
    ///
    /// `read_geometry = false` skips the vertex/index readback
    /// (counters always read back). Useful when you only want timing
    /// + counts for measurement runs.
    pub fn extract(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        instructions: &[ProcInstruction],
        aabb_min: Vec3,
        aabb_max: Vec3,
        grid_n: u32,
        vertex_cap: u32,
        index_cap: u32,
        read_geometry: bool,
    ) -> (Option<SurfaceMesh>, ExtractStats) {
        assert!(grid_n >= 2);
        let t_start = std::time::Instant::now();

        let cell_size = (aabb_max - aabb_min).max_element() / grid_n as f32;
        // Recompute the actual axis sizes so the AABB fits exactly even
        // on non-cubic boxes (the unused tail along shorter axes stays
        // empty cells, which classify happily zeros).
        let cube_n = grid_n + 1;
        let cells = (grid_n as u64).pow(3);
        let cubes = (cube_n as u64).pow(3);

        let instructions_pad: Vec<ProcInstruction>;
        let ins_slice: &[ProcInstruction] = if instructions.is_empty() {
            instructions_pad = vec![ProcInstruction::zeroed()];
            &instructions_pad
        } else {
            instructions
        };
        let has_warps = instructions.iter().any(|ins| {
            ins.op == rkp_procedural::OpKind::PushNoiseDisplace as u32
                || ins.op == rkp_procedural::OpKind::PushMirror as u32
                || ins.op == rkp_procedural::OpKind::PushArray as u32
        });

        let t_upload = std::time::Instant::now();
        self.ensure_instructions_capacity(device, ins_slice.len());
        self.ensure_cell_capacity(device, cells);
        self.ensure_cube_capacity(device, cubes);
        self.ensure_vertex_capacity(device, vertex_cap as u64);
        self.ensure_index_capacity(device, index_cap as u64);

        queue.write_buffer(
            self.instructions_buf.as_ref().unwrap(),
            0,
            bytemuck::cast_slice(ins_slice),
        );
        let params = SnParams {
            aabb_min: aabb_min.to_array(),
            cell_size,
            grid_n,
            instruction_count: instructions.len() as u32,
            vertex_cap,
            index_cap,
        };
        queue.write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&params));
        // Reset both atomic counters.
        queue.write_buffer(&self.counters_buf, 0, bytemuck::bytes_of(&[0u32, 0u32]));

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sn bind group"),
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
                    resource: self.cell_solid_buf.as_ref().unwrap().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.cube_map_buf.as_ref().unwrap().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.counters_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: self.vertices_buf.as_ref().unwrap().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: self.indices_buf.as_ref().unwrap().as_entire_binding(),
                },
            ],
        });
        let upload_dt = t_upload.elapsed();

        let pipeline_classify = if has_warps {
            &self.pipeline_classify_warps
        } else {
            &self.pipeline_classify_simple
        };
        let pipeline_vertex = if has_warps {
            &self.pipeline_vertex_warps
        } else {
            &self.pipeline_vertex_simple
        };

        let dispatch_classify = (grid_n + 3) / 4;
        let dispatch_vertex = (cube_n + 3) / 4;
        let dispatch_index = (grid_n + 3) / 4;

        // Each pass goes in its own encoder + submit so we get a clean
        // GPU timing per pass (queue.submit returns after work is queued
        // but device.poll(wait) blocks until it's done, giving us
        // per-pass wall-clock without timestamp queries).
        let classify_dt;
        let vertex_dt;
        let index_dt;

        // Pass 1: classify
        {
            let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("sn classify encode"),
            });
            {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("sn classify"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(pipeline_classify);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(dispatch_classify, dispatch_classify, dispatch_classify);
            }
            let t = std::time::Instant::now();
            queue.submit(Some(enc.finish()));
            device
                .poll(wgpu::PollType::wait_indefinitely())
                .expect("device poll");
            classify_dt = t.elapsed();
        }

        // Pass 2: vertex emit
        {
            let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("sn vertex_emit encode"),
            });
            {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("sn vertex_emit"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(pipeline_vertex);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(dispatch_vertex, dispatch_vertex, dispatch_vertex);
            }
            let t = std::time::Instant::now();
            queue.submit(Some(enc.finish()));
            device
                .poll(wgpu::PollType::wait_indefinitely())
                .expect("device poll");
            vertex_dt = t.elapsed();
        }

        // Pass 3: index emit
        {
            let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("sn index_emit encode"),
            });
            {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("sn index_emit"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipeline_index);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(dispatch_index, dispatch_index, dispatch_index);
            }
            let t = std::time::Instant::now();
            queue.submit(Some(enc.finish()));
            device
                .poll(wgpu::PollType::wait_indefinitely())
                .expect("device poll");
            index_dt = t.elapsed();
        }

        // Readback counters (always) + geometry (optional).
        let t_readback = std::time::Instant::now();
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("sn readback encode"),
        });
        enc.copy_buffer_to_buffer(&self.counters_buf, 0, &self.counters_staging, 0, 8);
        if read_geometry {
            enc.copy_buffer_to_buffer(
                self.vertices_buf.as_ref().unwrap(),
                0,
                self.vertices_staging.as_ref().unwrap(),
                0,
                self.vertices_cap * std::mem::size_of::<MeshVertex>() as u64,
            );
            enc.copy_buffer_to_buffer(
                self.indices_buf.as_ref().unwrap(),
                0,
                self.indices_staging.as_ref().unwrap(),
                0,
                self.indices_cap * 4,
            );
        }
        queue.submit(Some(enc.finish()));

        let counters = map_and_read::<u32>(device, &self.counters_staging, 8);
        let vertex_count = counters[0];
        let index_count = counters[1];

        let mesh = if read_geometry {
            let v_bytes =
                self.vertices_cap * std::mem::size_of::<MeshVertex>() as u64;
            let i_bytes = self.indices_cap * 4;
            let all_v = map_and_read::<MeshVertex>(
                device,
                self.vertices_staging.as_ref().unwrap(),
                v_bytes,
            );
            let all_i =
                map_and_read::<u32>(device, self.indices_staging.as_ref().unwrap(), i_bytes);
            let v_keep = (vertex_count.min(vertex_cap)) as usize;
            let i_keep = (index_count.min(index_cap)) as usize;
            Some(SurfaceMesh {
                vertices: all_v[..v_keep].to_vec(),
                indices: all_i[..i_keep].to_vec(),
            })
        } else {
            None
        };
        let readback_dt = t_readback.elapsed();

        let stats = ExtractStats {
            vertex_count,
            index_count,
            upload: upload_dt,
            classify: classify_dt,
            vertex_emit: vertex_dt,
            index_emit: index_dt,
            readback: readback_dt,
            total: t_start.elapsed(),
        };
        (mesh, stats)
    }

    fn ensure_instructions_capacity(&mut self, device: &wgpu::Device, needed: usize) {
        if self.instructions_cap < needed {
            let new_cap = (needed.max(64) * 3) / 2;
            self.instructions_buf = Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("sn instructions"),
                size: (new_cap * std::mem::size_of::<ProcInstruction>()) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            self.instructions_cap = new_cap;
        }
    }

    fn ensure_cell_capacity(&mut self, device: &wgpu::Device, needed: u64) {
        if self.cell_solid_cap < needed {
            let new_cap = (needed * 3) / 2;
            self.cell_solid_buf = Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("sn cell_solid"),
                size: new_cap * 4,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            }));
            self.cell_solid_cap = new_cap;
        }
    }

    fn ensure_cube_capacity(&mut self, device: &wgpu::Device, needed: u64) {
        if self.cube_map_cap < needed {
            let new_cap = (needed * 3) / 2;
            self.cube_map_buf = Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("sn cube_vertex_map"),
                size: new_cap * 4,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            }));
            self.cube_map_cap = new_cap;
        }
    }

    fn ensure_vertex_capacity(&mut self, device: &wgpu::Device, needed: u64) {
        if self.vertices_cap < needed {
            let new_cap = (needed * 3) / 2;
            let bytes = new_cap * std::mem::size_of::<MeshVertex>() as u64;
            self.vertices_buf = Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("sn vertices"),
                size: bytes,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }));
            self.vertices_staging = Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("sn vertices staging"),
                size: bytes,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            self.vertices_cap = new_cap;
        }
    }

    fn ensure_index_capacity(&mut self, device: &wgpu::Device, needed: u64) {
        if self.indices_cap < needed {
            let new_cap = (needed * 3) / 2;
            let bytes = new_cap * 4;
            self.indices_buf = Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("sn indices"),
                size: bytes,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }));
            self.indices_staging = Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("sn indices staging"),
                size: bytes,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            self.indices_cap = new_cap;
        }
    }
}

fn map_and_read<T: Pod>(device: &wgpu::Device, buf: &wgpu::Buffer, size: u64) -> Vec<T> {
    let slice = buf.slice(0..size);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("device poll");
    rx.recv().expect("map_async channel").expect("map_async result");
    let out = bytemuck::cast_slice::<u8, T>(&slice.get_mapped_range()).to_vec();
    buf.unmap();
    out
}
