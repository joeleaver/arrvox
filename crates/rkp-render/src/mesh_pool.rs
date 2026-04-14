//! GPU vertex + index buffer pool for extracted marching-cubes meshes.
//!
//! A single vertex buffer and a single index buffer hold the geometry for
//! every mesh-backed object. Per-object allocations are tracked in a
//! [`HashMap`] keyed by `object_id`. Allocation mirrors
//! [`VoxelPool`](rkp_core::VoxelPool): bump pointer + first-fit free list to
//! keep re-voxelization from leaking ranges.
//!
//! Phase 1 intentionally keeps the CPU-side mirror — `Vec<MeshVertex>` and
//! `Vec<u32>` — grown alongside the GPU buffers so we can re-upload a single
//! contiguous range when a mesh changes. Later phases can switch to
//! incremental uploads (queue.write_buffer at the allocation offset) without
//! changing the public API.

use bytemuck::{Pod, Zeroable};
use rkp_core::ExtractedMesh;
use std::collections::HashMap;

/// Vertex layout for the triangle G-buffer pass. 36 bytes.
///
/// Layout mirrored in `shaders/triangle_gbuffer.wgsl`:
/// ```wgsl
/// struct VertexIn {
///   @location(0) position:      vec3<f32>,
///   @location(1) normal:        vec3<f32>,
///   @location(2) color:         u32,
///   @location(3) material_pack: u32,  // primary(lo16) | secondary(hi16)
///   @location(4) blend_weight:  u32,  // 0..=255 in low byte (0=primary only)
/// }
/// ```
#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable)]
pub struct MeshVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    /// Packed R8G8B8 | intensity (same format as `VoxelPool::color`).
    pub color: u32,
    /// Primary material id in low 16 bits, secondary in high 16 bits.
    pub material_pack: u32,
    /// Blend weight in low 8 bits (0 = primary only, 255 = pure secondary).
    pub blend_weight: u32,
}

impl MeshVertex {
    /// `wgpu::VertexBufferLayout` for creating the render pipeline.
    pub fn vertex_layout() -> wgpu::VertexBufferLayout<'static> {
        static ATTRS: [wgpu::VertexAttribute; 5] = wgpu::vertex_attr_array![
            0 => Float32x3,  // position
            1 => Float32x3,  // normal
            2 => Uint32,     // color
            3 => Uint32,     // material_pack (primary | secondary << 16)
            4 => Uint32,     // blend_weight (0..255 in low byte)
        ];
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<MeshVertex>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &ATTRS,
        }
    }
}

/// Handle to a contiguous range of vertices + indices belonging to one object.
///
/// `Copy` so the engine can cheaply build per-frame draw lists.
#[derive(Debug, Copy, Clone)]
pub struct MeshAllocation {
    pub vertex_start: u32,
    pub vertex_count: u32,
    pub index_start: u32,
    pub index_count: u32,
}

impl MeshAllocation {
    pub fn is_empty(&self) -> bool {
        self.index_count == 0
    }
}

/// GPU vertex + index buffer pool. See module docs.
pub struct MeshPool {
    /// CPU-side vertex mirror — length matches allocated_count() worth of verts.
    vertices: Vec<MeshVertex>,
    /// CPU-side index mirror.
    indices: Vec<u32>,

    /// GPU vertex buffer (grown as needed).
    vertex_buffer: wgpu::Buffer,
    /// GPU index buffer (grown as needed).
    index_buffer: wgpu::Buffer,

    /// Per-object allocations. Key is the object_id used at upload time.
    allocations: HashMap<u32, MeshAllocation>,

    /// Bump-pointer high water marks.
    next_vertex: u32,
    next_index: u32,

    /// First-fit free lists — `(start, count)` entries.
    free_vertices: Vec<(u32, u32)>,
    free_indices: Vec<(u32, u32)>,

    /// Pending re-upload range; when set, the next [`flush`] uploads
    /// `vertices[0..next_vertex]` and `indices[0..next_index]`. We use the
    /// simplest possible strategy for Phase 1: any edit triggers a full
    /// re-upload of the used prefix. Phase 5 can switch to per-allocation
    /// partial uploads.
    dirty: bool,
}

impl MeshPool {
    /// Create an empty pool with minimal GPU allocations. Both buffers grow on
    /// demand; the initial sizes exist only so wgpu accepts the binding.
    pub fn new(device: &wgpu::Device) -> Self {
        let vertex_buffer = Self::create_vertex_buffer(device, 256);
        let index_buffer = Self::create_index_buffer(device, 256);
        Self {
            vertices: Vec::new(),
            indices: Vec::new(),
            vertex_buffer,
            index_buffer,
            allocations: HashMap::new(),
            next_vertex: 0,
            next_index: 0,
            free_vertices: Vec::new(),
            free_indices: Vec::new(),
            dirty: false,
        }
    }

    /// Upload (or re-upload) the mesh for `object_id`. Any previous allocation
    /// for this id is released first.
    ///
    /// Returns the new allocation, or `None` for an empty mesh (no GPU state
    /// is created and any existing allocation is removed).
    pub fn upload_mesh(&mut self, object_id: u32, mesh: &ExtractedMesh) -> Option<MeshAllocation> {
        self.deallocate(object_id);

        if mesh.is_empty() {
            return None;
        }

        let vertex_count = mesh.positions.len() as u32;
        let index_count = mesh.indices.len() as u32;

        let vertex_start = Self::alloc_range(
            &mut self.free_vertices,
            &mut self.next_vertex,
            &mut self.vertices,
            vertex_count,
            MeshVertex::zeroed(),
        );
        let index_start = Self::alloc_range(
            &mut self.free_indices,
            &mut self.next_index,
            &mut self.indices,
            index_count,
            0u32,
        );

        // Copy vertex data into the allocated range.
        let vbase = vertex_start as usize;
        for i in 0..vertex_count as usize {
            let pos = mesh.positions[i];
            let nrm = mesh.normals[i];
            let col = mesh.colors[i];
            let primary = mesh.material_ids[i] as u32 & 0xFFFF;
            let secondary = mesh.secondary_material_ids[i] as u32 & 0xFFFF;
            let blend = mesh.blend_weights[i] as u32;
            self.vertices[vbase + i] = MeshVertex {
                position: pos.into(),
                normal: nrm.into(),
                color: col,
                material_pack: primary | (secondary << 16),
                blend_weight: blend,
            };
        }

        // Indices are relative to this allocation's vertex base — rebase them
        // to the pool-global vertex index so the GPU can draw with a single
        // bound vertex buffer.
        let ibase = index_start as usize;
        for (i, &idx) in mesh.indices.iter().enumerate() {
            self.indices[ibase + i] = idx + vertex_start;
        }

        let alloc = MeshAllocation {
            vertex_start,
            vertex_count,
            index_start,
            index_count,
        };
        self.allocations.insert(object_id, alloc);
        self.dirty = true;
        Some(alloc)
    }

    /// Release the allocation for `object_id`, if any.
    pub fn deallocate(&mut self, object_id: u32) {
        let Some(alloc) = self.allocations.remove(&object_id) else {
            return;
        };
        Self::free_range(
            &mut self.free_vertices,
            &mut self.next_vertex,
            &mut self.vertices,
            alloc.vertex_start,
            alloc.vertex_count,
            MeshVertex::zeroed(),
        );
        Self::free_range(
            &mut self.free_indices,
            &mut self.next_index,
            &mut self.indices,
            alloc.index_start,
            alloc.index_count,
            0u32,
        );
        self.dirty = true;
    }

    /// Look up the allocation for `object_id`.
    pub fn get(&self, object_id: u32) -> Option<MeshAllocation> {
        self.allocations.get(&object_id).copied()
    }

    /// Number of objects currently holding mesh allocations.
    pub fn len(&self) -> usize {
        self.allocations.len()
    }

    pub fn is_empty(&self) -> bool {
        self.allocations.is_empty()
    }

    /// Upload dirty CPU data to the GPU. Grows buffers as needed. Returns
    /// `true` if the GPU buffers were replaced (the caller may need to rebind).
    pub fn flush(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) -> bool {
        if !self.dirty {
            return false;
        }
        self.dirty = false;

        let mut replaced = false;

        // Vertices.
        let vbytes: &[u8] = bytemuck::cast_slice(&self.vertices[..self.next_vertex as usize]);
        if !vbytes.is_empty() {
            let needed = vbytes.len() as u64;
            if needed > self.vertex_buffer.size() {
                self.vertex_buffer = Self::create_vertex_buffer(device, needed);
                replaced = true;
            }
            queue.write_buffer(&self.vertex_buffer, 0, vbytes);
        }

        // Indices.
        let ibytes: &[u8] = bytemuck::cast_slice(&self.indices[..self.next_index as usize]);
        if !ibytes.is_empty() {
            let needed = ibytes.len() as u64;
            if needed > self.index_buffer.size() {
                self.index_buffer = Self::create_index_buffer(device, needed);
                replaced = true;
            }
            queue.write_buffer(&self.index_buffer, 0, ibytes);
        }

        replaced
    }

    #[inline]
    pub fn vertex_buffer(&self) -> &wgpu::Buffer {
        &self.vertex_buffer
    }

    #[inline]
    pub fn index_buffer(&self) -> &wgpu::Buffer {
        &self.index_buffer
    }

    // ----- internal helpers -----

    fn create_vertex_buffer(device: &wgpu::Device, capacity_bytes: u64) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rkp_mesh_vertices"),
            size: capacity_bytes.max(std::mem::size_of::<MeshVertex>() as u64),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    fn create_index_buffer(device: &wgpu::Device, capacity_bytes: u64) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rkp_mesh_indices"),
            size: capacity_bytes.max(4),
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    /// Generic range allocator — first-fit free-list, else bump, else grow.
    fn alloc_range<T: Copy>(
        free_list: &mut Vec<(u32, u32)>,
        next: &mut u32,
        mirror: &mut Vec<T>,
        count: u32,
        fill: T,
    ) -> u32 {
        if let Some(idx) = free_list.iter().position(|(_, c)| *c >= count) {
            let (start, free_count) = free_list[idx];
            if free_count == count {
                free_list.swap_remove(idx);
            } else {
                free_list[idx] = (start + count, free_count - count);
            }
            return start;
        }
        let start = *next;
        *next += count;
        if mirror.len() < *next as usize {
            mirror.resize(*next as usize, fill);
        }
        start
    }

    /// Generic range free — coalesces at the tail, otherwise pushes to list.
    fn free_range<T: Copy>(
        free_list: &mut Vec<(u32, u32)>,
        next: &mut u32,
        mirror: &mut Vec<T>,
        start: u32,
        count: u32,
        fill: T,
    ) {
        if count == 0 {
            return;
        }
        // Zero out so stale data doesn't bleed into re-allocations (matches
        // VoxelPool behavior).
        for s in start as usize..(start + count) as usize {
            if s < mirror.len() {
                mirror[s] = fill;
            }
        }
        if start + count == *next {
            *next = start;
            loop {
                let idx = free_list.iter().position(|(s, c)| s + c == *next);
                match idx {
                    Some(i) => {
                        let (s, _) = free_list.swap_remove(i);
                        *next = s;
                    }
                    None => break,
                }
            }
        } else {
            free_list.push((start, count));
        }
    }
}

#[cfg(test)]
mod tests {
    //! Note: these are API-shape tests only. We don't spin up a wgpu device in
    //! unit tests — that's covered by the editor smoke test.
    use super::*;

    #[test]
    fn mesh_vertex_is_36_bytes() {
        assert_eq!(std::mem::size_of::<MeshVertex>(), 36);
    }

    #[test]
    fn vertex_layout_has_five_attrs() {
        let layout = MeshVertex::vertex_layout();
        assert_eq!(layout.array_stride, 36);
        assert_eq!(layout.attributes.len(), 5);
    }
}
