//! GPU surface shell buffer — per-brick occupancy bitmasks for the emit pass.
//!
//! Each allocated brick slot has a 512-bit occupancy bitmask (`[u64; 8]`)
//! stored in this buffer. The emit compute shader reads these to determine
//! which voxels are surface voxels and which faces are exposed.
//!
//! Indexed by brick pool slot: `shell_data[slot * 16 .. slot * 16 + 16]`
//! (16 u32s = 8 u64s = 512 bits per brick).

/// GPU storage buffer for surface shell occupancy bitmasks.
pub struct SurfaceShellGpu {
    /// Storage buffer: 16 u32s per brick slot (= 8 u64s = 512 bits).
    pub buffer: wgpu::Buffer,
    /// Bind group layout with one storage buffer binding.
    pub bind_group_layout: wgpu::BindGroupLayout,
    /// Bind group.
    pub bind_group: wgpu::BindGroup,
    /// Current capacity in brick slots.
    capacity_slots: u32,
}

/// Number of u32s per brick slot in the shell buffer (512 bits = 16 u32s).
const U32S_PER_SLOT: u32 = 16;

impl SurfaceShellGpu {
    /// Create with initial capacity for `slot_count` bricks.
    pub fn new(device: &wgpu::Device, slot_count: u32) -> Self {
        let cap = slot_count.max(64);
        let buffer = Self::create_buffer(device, cap);
        let bind_group_layout = Self::create_layout(device);
        let bind_group = Self::create_bind_group(device, &bind_group_layout, &buffer);

        Self {
            buffer,
            bind_group_layout,
            bind_group,
            capacity_slots: cap,
        }
    }

    /// Upload occupancy data for a single brick slot.
    ///
    /// `occupancy`: the 512-bit bitmask as `[u64; 8]`.
    pub fn upload_slot(&self, queue: &wgpu::Queue, slot: u32, occupancy: &[u64; 8]) {
        let offset = (slot as u64) * (U32S_PER_SLOT as u64) * 4;
        queue.write_buffer(&self.buffer, offset, bytemuck::cast_slice(occupancy));
    }

    /// Ensure capacity for at least `needed_slots` bricks. Rebuilds buffer
    /// and bind group if growth is needed.
    pub fn ensure_capacity(&mut self, device: &wgpu::Device, needed_slots: u32) {
        if needed_slots <= self.capacity_slots {
            return;
        }
        let new_cap = (needed_slots).max(self.capacity_slots * 2);
        self.buffer = Self::create_buffer(device, new_cap);
        self.bind_group =
            Self::create_bind_group(device, &self.bind_group_layout, &self.buffer);
        self.capacity_slots = new_cap;
    }

    /// Current capacity in brick slots.
    pub fn capacity(&self) -> u32 {
        self.capacity_slots
    }

    fn create_buffer(device: &wgpu::Device, slot_count: u32) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("surface_shell"),
            size: (slot_count as u64) * (U32S_PER_SLOT as u64) * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    fn create_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("surface_shell layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        })
    }

    fn create_bind_group(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        buffer: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("surface_shell bind group"),
            layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer.as_entire_binding(),
            }],
        })
    }
}
