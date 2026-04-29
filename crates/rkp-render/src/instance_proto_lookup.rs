//! Stage 5b — GPU-side prototype lookup for the instance march.
//!
//! Stage 2's [`crate::user_shader_proto_pass::PrototypeCache`] caches
//! one [`crate::user_shader_proto_pass::PrototypeEntry`] per shader,
//! keyed by `shader_id`. Stage 5b's instance march needs to look up
//! `(octree_root, max_depth, instance_stride_u32, pos_offset_u32,
//! scale_offset_u32, scale_kind)` per shader_id from the GPU side.
//!
//! ## Layout
//!
//! Flat `array<GpuPrototypeEntry>`, sorted by `shader_id`. The march
//! either binary-searches by id or — V1's choice — linear-scans.
//! Linear scan is fine because instance shaders count in the dozens at
//! most; the per-pixel cost is `O(num_instance_shaders)`, not
//! `O(num_instances)`.
//!
//! Entries combine prototype state (octree_root, max_depth — from the
//! [`crate::user_shader_proto_pass::PrototypeCache`]) with instance-
//! struct layout (stride, pos/scale offsets — from the
//! [`crate::instance_proto::InstanceLayout`] parsed at composer scan
//! time). The march composes these to decode an instance record's pos
//! and scale, then transforms its ray into prototype space.

use crate::instance_proto::ScaleKind;
use crate::shader_composer::UserShaderEntry;
use crate::user_shader_proto_pass::{PrototypeCache, PrototypeEntry};

/// Sentinel for "no scale field" — the instance pipeline reads this and
/// treats the prototype as fixed-size (uniform scale = 1.0). Stored in
/// `scale_offset_u32` so callers don't have to switch on `scale_kind`.
pub const NO_SCALE_OFFSET: u32 = u32::MAX;

/// Encoded `scale_kind` values for the GPU. Mirror of [`ScaleKind`].
/// Stored as `u32` so the GPU struct stays 4-byte alignment friendly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuScaleKind {
    None = 0,
    Uniform = 1,
    PerAxis = 2,
}

impl GpuScaleKind {
    pub fn from_cpu(k: ScaleKind) -> Self {
        match k {
            ScaleKind::None => Self::None,
            ScaleKind::Uniform => Self::Uniform,
            ScaleKind::PerAxis => Self::PerAxis,
        }
    }

    pub fn as_u32(self) -> u32 {
        self as u32
    }
}

/// One per-shader entry the GPU march reads to translate a `shader_id`
/// into "where is this prototype's octree, and how do I decode an
/// instance's pos/scale?"
///
/// 32 bytes — 4-byte alignment, no `vec3` gotchas.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuPrototypeEntry {
    pub shader_id: u32,
    /// Absolute pool index of the prototype's level-0 root in
    /// `octree_nodes`. Equivalent to
    /// `PrototypeEntry::octree_root(pool_octree_base)`.
    pub octree_root: u32,
    pub max_depth: u32,
    /// Stride between consecutive instance records in `instance_pool`,
    /// in u32s. Same value the emit codegen uses.
    pub instance_stride_u32: u32,
    /// u32 offset of the `pos: vec3<f32>` field within an instance
    /// record. Always present — `@instance_proto` requires it.
    pub pos_offset_u32: u32,
    /// u32 offset of the `scale` field. [`NO_SCALE_OFFSET`] when the
    /// shader's instance struct has no `scale` field; otherwise this
    /// is the offset of either the `f32` (uniform) or `vec3<f32>`
    /// (per-axis) scale.
    pub scale_offset_u32: u32,
    /// Encoded [`GpuScaleKind`].
    pub scale_kind: u32,
    pub _pad0: u32,
}

const _: () = assert!(std::mem::size_of::<GpuPrototypeEntry>() == 32);

/// Errors that can arise while flattening shader entries to GPU records.
#[derive(Debug, Clone, thiserror::Error)]
pub enum ProtoLookupError {
    #[error("shader `{shader_name}` (id {shader_id}) is registered as an instance shader but has no PrototypeCache entry")]
    MissingPrototype {
        shader_name: String,
        shader_id: u32,
    },

    #[error("shader `{shader_name}` (id {shader_id}) has no parsed InstanceLayout — should not be in the instance shader set")]
    MissingLayout {
        shader_name: String,
        shader_id: u32,
    },
}

/// Build one GPU entry from a CPU [`PrototypeEntry`] + instance layout.
/// Used internally by [`flatten_prototype_lookup`] but also exported so
/// tests / Stage 6 wiring can construct entries individually without
/// going through the registry.
pub fn build_gpu_entry(
    entry: &UserShaderEntry,
    proto: &PrototypeEntry,
    pool_octree_base: u32,
) -> Result<GpuPrototypeEntry, ProtoLookupError> {
    let layout = entry.instance_layout.as_ref().ok_or_else(|| {
        ProtoLookupError::MissingLayout {
            shader_name: entry.name.clone(),
            shader_id: entry.id,
        }
    })?;
    let stride_u32 = layout.total_size.div_ceil(4);
    let pos_offset_u32 = layout.pos_offset / 4;
    let (scale_offset_u32, scale_kind) = match layout.scale_offset {
        Some(byte_off) => (byte_off / 4, GpuScaleKind::from_cpu(layout.scale_kind)),
        None => (NO_SCALE_OFFSET, GpuScaleKind::None),
    };
    Ok(GpuPrototypeEntry {
        shader_id: entry.id,
        octree_root: proto.octree_root(pool_octree_base),
        max_depth: proto.max_depth,
        instance_stride_u32: stride_u32,
        pos_offset_u32,
        scale_offset_u32,
        scale_kind: scale_kind.as_u32(),
        _pad0: 0,
    })
}

/// Walk every instance-pipeline shader in the registry, look up its
/// cached prototype, and emit a sorted-by-shader-id `Vec` of GPU
/// entries the march can bind directly. Skips shaders missing a baked
/// prototype (those have an open bake task — caller should re-flatten
/// once the bake completes).
///
/// Returns the entries plus the `shader_id`s that were skipped because
/// they had no `PrototypeCache` entry — useful for diagnostics and for
/// the engine layer to know what's still pending.
pub struct PrototypeLookupBuild {
    pub entries: Vec<GpuPrototypeEntry>,
    /// Shader ids the registry knows are instance shaders but the
    /// prototype cache hasn't baked yet. The march simply won't find
    /// these shaders if a region's `shader_id` matches one of them
    /// (instance traversal is skipped for unknown ids).
    pub skipped_unbaked: Vec<u32>,
}

pub fn flatten_prototype_lookup(
    registry_entries: &[UserShaderEntry],
    cache: &PrototypeCache,
) -> Result<PrototypeLookupBuild, ProtoLookupError> {
    let pool_base = cache.pool_octree_base();
    let mut entries: Vec<GpuPrototypeEntry> = Vec::new();
    let mut skipped: Vec<u32> = Vec::new();
    for entry in registry_entries {
        if !entry.is_instance_pipeline() {
            continue;
        }
        match cache.get(entry.id) {
            Some(proto) => {
                entries.push(build_gpu_entry(entry, proto, pool_base)?);
            }
            None => {
                skipped.push(entry.id);
            }
        }
    }
    entries.sort_by_key(|e| e.shader_id);
    Ok(PrototypeLookupBuild {
        entries,
        skipped_unbaked: skipped,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::instance_proto::{InstanceField, InstanceLayout, WgslType};
    use crate::user_shader_proto_pass::PrototypeEntry;

    fn make_layout(scale_kind: ScaleKind, scale_offset: Option<u32>) -> InstanceLayout {
        // Realistic Blade-style 32-byte struct: pos (12) + yaw (4) +
        // sway_phase (4) + height_scale/scale (4) + tint (4) + maybe pad.
        InstanceLayout {
            struct_name: "Blade".into(),
            struct_text: "struct Blade { … }".into(),
            fields: vec![
                InstanceField {
                    name: "pos".into(),
                    ty: WgslType::Vec3F32,
                    byte_offset: 0,
                },
                InstanceField {
                    name: "yaw".into(),
                    ty: WgslType::F32,
                    byte_offset: 12,
                },
            ],
            total_size: 32,
            alignment: 16,
            pos_offset: 0,
            rot_offset: None,
            scale_kind,
            scale_offset,
            warnings: vec![],
        }
    }

    fn make_entry(id: u32, layout: Option<InstanceLayout>) -> UserShaderEntry {
        UserShaderEntry {
            name: format!("shader_{id}"),
            file_path: std::path::PathBuf::new(),
            id,
            metadata: Default::default(),
            shade_text: None,
            generate_text: None,
            helpers: vec![],
            proto_text: layout.as_ref().map(|_| "fn proto() {}".into()),
            emit_text: layout.as_ref().map(|_| "fn emit() {}".into()),
            inst_aabb_text: None,
            inst_to_local_text: None,
            struct_decls: vec![],
            instance_layout: layout,
        }
    }

    fn make_proto(id: u32, max_depth: u32, octree_offset: u32) -> PrototypeEntry {
        PrototypeEntry {
            shader_id: id,
            source_hash: 0,
            max_depth,
            octree_extent: (octree_offset, 64),
            brick_extent: (0, 64),
            leaf_attr_extent: (0, 64),
            touched_this_frame: true,
        }
    }

    #[test]
    fn entry_is_pod_and_32_bytes() {
        assert_eq!(std::mem::size_of::<GpuPrototypeEntry>(), 32);
        let e = GpuPrototypeEntry {
            shader_id: 7,
            octree_root: 100,
            max_depth: 2,
            instance_stride_u32: 8,
            pos_offset_u32: 0,
            scale_offset_u32: 4,
            scale_kind: GpuScaleKind::Uniform.as_u32(),
            _pad0: 0,
        };
        let bytes: &[u8] = bytemuck::bytes_of(&e);
        assert_eq!(bytes.len(), 32);
        let round: GpuPrototypeEntry = *bytemuck::from_bytes(bytes);
        assert_eq!(round, e);
    }

    #[test]
    fn build_gpu_entry_no_scale_yields_sentinel() {
        let layout = make_layout(ScaleKind::None, None);
        let entry = make_entry(3, Some(layout));
        let proto = make_proto(3, 2, 0);
        let gpu = build_gpu_entry(&entry, &proto, 0).unwrap();
        assert_eq!(gpu.shader_id, 3);
        assert_eq!(gpu.octree_root, 0);
        assert_eq!(gpu.max_depth, 2);
        assert_eq!(gpu.instance_stride_u32, 8); // 32 / 4
        assert_eq!(gpu.pos_offset_u32, 0);
        assert_eq!(gpu.scale_offset_u32, NO_SCALE_OFFSET);
        assert_eq!(gpu.scale_kind, GpuScaleKind::None.as_u32());
    }

    #[test]
    fn build_gpu_entry_uniform_scale_offsets_match_byte_layout() {
        // pos at byte 0, scale: f32 at byte 16 → u32 offsets 0 and 4.
        let layout = make_layout(ScaleKind::Uniform, Some(16));
        let entry = make_entry(5, Some(layout));
        let proto = make_proto(5, 1, 0);
        let gpu = build_gpu_entry(&entry, &proto, 0).unwrap();
        assert_eq!(gpu.pos_offset_u32, 0);
        assert_eq!(gpu.scale_offset_u32, 4);
        assert_eq!(gpu.scale_kind, GpuScaleKind::Uniform.as_u32());
    }

    #[test]
    fn build_gpu_entry_per_axis_scale_round_trip() {
        let layout = make_layout(ScaleKind::PerAxis, Some(16));
        let entry = make_entry(8, Some(layout));
        let proto = make_proto(8, 3, 0);
        let gpu = build_gpu_entry(&entry, &proto, 0).unwrap();
        assert_eq!(gpu.scale_kind, GpuScaleKind::PerAxis.as_u32());
    }

    #[test]
    fn build_gpu_entry_pool_base_offsets_root() {
        let layout = make_layout(ScaleKind::None, None);
        let entry = make_entry(1, Some(layout));
        let proto = make_proto(1, 2, 100); // octree_extent.0 = 100
        // pool_base = 1000 → octree_root = 1000 + 100 = 1100.
        let gpu = build_gpu_entry(&entry, &proto, 1000).unwrap();
        assert_eq!(gpu.octree_root, 1100);
    }

    #[test]
    fn build_gpu_entry_missing_layout_errors() {
        let entry = make_entry(2, None);
        let proto = make_proto(2, 1, 0);
        let err = build_gpu_entry(&entry, &proto, 0).unwrap_err();
        assert!(matches!(err, ProtoLookupError::MissingLayout { .. }));
    }

    #[test]
    fn flatten_skips_non_instance_shaders() {
        // One instance shader (id=1, baked) + one shade-only shader (id=2).
        let inst = make_entry(1, Some(make_layout(ScaleKind::None, None)));
        let mut shade_only = make_entry(2, None);
        shade_only.shade_text = Some("fn shade() {}".into());
        let entries = vec![inst.clone(), shade_only];

        let mut cache = PrototypeCache::with_capacities(1024, 1024, 8192);
        cache.set_pool_bases(0, 0, 0);
        let _ = cache.lookup_or_allocate(1, 0xAA, 2).unwrap();

        let build = flatten_prototype_lookup(&entries, &cache).unwrap();
        assert_eq!(build.entries.len(), 1);
        assert_eq!(build.entries[0].shader_id, 1);
        assert!(build.skipped_unbaked.is_empty());
    }

    #[test]
    fn flatten_collects_unbaked_shaders() {
        // Two instance shaders, only one baked.
        let a = make_entry(1, Some(make_layout(ScaleKind::None, None)));
        let b = make_entry(7, Some(make_layout(ScaleKind::None, None)));
        let entries = vec![a, b];

        let mut cache = PrototypeCache::with_capacities(1024, 1024, 8192);
        cache.set_pool_bases(0, 0, 0);
        let _ = cache.lookup_or_allocate(1, 0xAA, 2).unwrap();

        let build = flatten_prototype_lookup(&entries, &cache).unwrap();
        assert_eq!(build.entries.len(), 1);
        assert_eq!(build.entries[0].shader_id, 1);
        assert_eq!(build.skipped_unbaked, vec![7]);
    }

    #[test]
    fn flatten_sorts_by_shader_id() {
        let entries = vec![
            make_entry(7, Some(make_layout(ScaleKind::None, None))),
            make_entry(2, Some(make_layout(ScaleKind::None, None))),
            make_entry(5, Some(make_layout(ScaleKind::None, None))),
        ];
        let mut cache = PrototypeCache::with_capacities(8 * 1024, 8 * 1024, 64 * 1024);
        cache.set_pool_bases(0, 0, 0);
        for id in [7u32, 2, 5] {
            let _ = cache.lookup_or_allocate(id, 0xAA, 2).unwrap();
        }
        let build = flatten_prototype_lookup(&entries, &cache).unwrap();
        assert_eq!(
            build.entries.iter().map(|e| e.shader_id).collect::<Vec<_>>(),
            vec![2, 5, 7],
        );
    }
}
