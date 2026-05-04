//! Rust ↔ WGSL struct-layout contract tests.
//!
//! Every Rust `#[repr(C)]` mirror of a WGSL struct is uploaded via a
//! `bytemuck` cast — if the two layouts ever drift the GPU reads
//! garbage with no static error. These tests parse the emitted WESL
//! artifact (which is the same flat WGSL the runtime feeds wgpu) and
//! compare each WGSL struct's byte size against the Rust mirror's
//! `std::mem::size_of`.
//!
//! Field-level offsets are validated implicitly: Rust `#[repr(C)]`
//! lays fields out in declaration order, and WGSL alignment rules
//! produce the same layout because every mirror is hand-tuned to
//! match (vec3 fields are flattened to per-component scalars on the
//! Rust side, padding is explicit). If a future struct adds a vec3
//! that gets WGSL-padded but matched by Rust as `[f32; 3]`, the size
//! check catches the drift before any data corruption can happen.
//!
//! Add a new mirror by appending to the `MIRRORS` table below.

use rkp_render::octree_march::MarchParams;
use rkp_render::rkp_scene::CameraUniforms;
use rkp_render::rkp_gpu_object::{RkpGpuAsset, RkpGpuInstance};
use rkp_render::rkp_shade::{GpuLight, GpuMaterial, ShadeParams};
use rkp_render::rkp_volumetric::VolumetricParams;

/// (WGSL struct name, expected size from the Rust mirror, mirror
/// name for diagnostics).
fn mirrors() -> Vec<(&'static str, usize, &'static str)> {
    vec![
        ("RkpAsset", std::mem::size_of::<RkpGpuAsset>(), "RkpGpuAsset"),
        ("RkpInstance", std::mem::size_of::<RkpGpuInstance>(), "RkpGpuInstance"),
        ("GpuMaterial", std::mem::size_of::<GpuMaterial>(), "GpuMaterial"),
        ("GpuLight", std::mem::size_of::<GpuLight>(), "GpuLight"),
        ("CameraUniforms", std::mem::size_of::<CameraUniforms>(), "CameraUniforms"),
        ("MarchParams", std::mem::size_of::<MarchParams>(), "MarchParams"),
        ("ShadeParams", std::mem::size_of::<ShadeParams>(), "ShadeParams"),
        ("VolumetricParams", std::mem::size_of::<VolumetricParams>(), "VolumetricParams"),
    ]
}

/// Parse a WGSL source and return the byte size of a top-level
/// struct named `name`, or `None` if the struct isn't declared in
/// this artifact.
fn struct_size(source: &str, name: &str) -> Option<u32> {
    let module = naga::front::wgsl::parse_str(source)
        .unwrap_or_else(|e| panic!("naga parse failed:\n{}", e.emit_to_string(source)));
    for (_handle, ty) in module.types.iter() {
        let Some(struct_name) = ty.name.as_deref() else { continue };
        if struct_name != name {
            continue;
        }
        if !matches!(ty.inner, naga::TypeInner::Struct { .. }) {
            continue;
        }
        // `Type::Struct.span` is the struct's total byte size after
        // WGSL alignment is applied; this is the value naga uses to
        // size buffer bindings.
        if let naga::TypeInner::Struct { span, .. } = ty.inner {
            return Some(span);
        }
    }
    None
}

/// Walk the table against the host march artifact (which imports
/// every shared lib type via the import block at the top of the
/// file). Each struct present in the artifact must size-match its
/// Rust mirror.
#[test]
fn host_march_struct_sizes_match_rust_mirrors() {
    let source: &str = wesl::include_wesl!("octree_march");
    for (wgsl_name, rust_size, rust_name) in mirrors() {
        let Some(span) = struct_size(source, wgsl_name) else {
            // The host march doesn't import every struct in lib/types;
            // skipping is the correct behaviour for those.
            continue;
        };
        assert_eq!(
            span as usize, rust_size,
            "size mismatch — WGSL `{wgsl_name}` is {span} bytes but Rust `{rust_name}` is {rust_size} bytes",
        );
    }
}

/// VolumetricParams and CloudParams only appear in the volumetric
/// pipelines — verify them against their own artifact.
#[test]
fn volumetric_struct_sizes_match_rust_mirrors() {
    let source: &str = wesl::include_wesl!("rkp_cloud_march");
    let span = struct_size(source, "VolumetricParams")
        .expect("rkp_cloud_march must import VolumetricParams");
    assert_eq!(
        span as usize,
        std::mem::size_of::<VolumetricParams>(),
        "size mismatch — WGSL `VolumetricParams` is {span} bytes but Rust mirror is {} bytes",
        std::mem::size_of::<VolumetricParams>(),
    );
}

/// Sanity: every entry in `mirrors()` must turn up in at least one
/// of the artifacts we scan. Catches the case where a Rust mirror
/// is added but the WGSL counterpart is silently missing — the
/// per-artifact tests above would otherwise no-op for it.
#[test]
fn every_mirror_appears_in_some_artifact() {
    let host_src: &str = wesl::include_wesl!("octree_march");
    let cloud_src: &str = wesl::include_wesl!("rkp_cloud_march");
    for (wgsl_name, _, rust_name) in mirrors() {
        let in_host = struct_size(host_src, wgsl_name).is_some();
        let in_cloud = struct_size(cloud_src, wgsl_name).is_some();
        assert!(
            in_host || in_cloud,
            "WGSL struct `{wgsl_name}` (mirror of Rust `{rust_name}`) was not found in any \
             scanned artifact — either the import was dropped or this test needs another \
             artifact added to the scan set",
        );
    }
}
