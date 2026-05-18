//! Top-level import orchestration: mesh → `.arvx` opacity-octree
//! asset (with optional `.arvxskel` sidecar).
//!
//! High-level stages (emitted as [`ImportEvent::StageStart`] /
//! `StageEnd` pairs through the caller's [`ProgressReporter`]):
//!
//! 1. `load_mesh` — parse source file, extract positions / UVs /
//!    materials.
//! 2. `prepare_mesh` — apply rotation offset, normalize to target
//!    size, optional uniform scale. Captures
//!    [`NormalizationParams`] for the skeleton sidecar.
//! 3. `build_bvh` — triangle BVH (see [`crate::bvh`]).
//! 4. `extract_skeleton` — optional, skipped for static meshes.
//! 5. `classify_bricks` — brick-level narrow-band + inside/outside.
//! 6. `voxelize_surface` — rayon-parallel per-voxel SDF sampling.
//! 7. `emit_shell_leaves` — 1-voxel outer shell + INTERIOR fill +
//!    SDF-gradient normals.
//! 8. `write_rkp` — atomic serialize + rename.
//! 9. `write_rkskel` — atomic skeleton sidecar (if a skeleton was
//!    found).

use std::collections::HashSet;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

use glam::Vec3;
use rayon::prelude::*;

use arvx_core::Aabb;
use arvx_core::constants::MESH_BRICK_DIM as BRICK_DIM;

use crate::bvh::TriangleBvh;
use crate::config::{ImportConfig, ImportResult};
use crate::error::ImportError;
use crate::event::{ImportEvent, NullReporter, ProgressReporter};
use crate::mesh::load_mesh;
use crate::normalize::prepare_mesh;
use crate::skeleton;

pub mod classify;
pub mod shell;
pub mod write;

use classify::{auto_voxel_size, classify_bricks, process_brick};
use shell::emit_shell_leaves;

/// Convenience wrapper for callers that don't care about progress
/// events. Delegates to [`import_mesh_to_opacity_rkp_with`] with a
/// null reporter.
pub fn import_mesh_to_opacity_rkp(
    input_path: &Path,
    output_path: &Path,
    config: &ImportConfig,
) -> Result<ImportResult, ImportError> {
    import_mesh_to_opacity_rkp_with(input_path, output_path, config, &NullReporter)
}

/// Full importer. Streams progress via `reporter` so a UI can render
/// a live status bar. Returns the assembled [`ImportResult`] on
/// success; errors are both returned *and* reported via
/// [`ImportEvent::Error`].
pub fn import_mesh_to_opacity_rkp_with(
    input_path: &Path,
    output_path: &Path,
    config: &ImportConfig,
    reporter: &dyn ProgressReporter,
) -> Result<ImportResult, ImportError> {
    // ── Validate config preconditions before any expensive work ─────
    if let Err(e) = config.validate() {
        return Err(emit_err(reporter, ImportError::InvalidConfig(e)));
    }

    // Abort at the next stage boundary if the caller has fired
    // cancellation on the reporter.
    macro_rules! check_cancel {
        () => {
            if reporter.is_cancelled() {
                return Err(emit_err(reporter, ImportError::Cancelled));
            }
        };
    }

    // ── Stage: load_mesh ────────────────────────────────────────────
    stage_start(
        reporter,
        "load_mesh",
        format!("Loading mesh: {}", input_path.display()),
    );
    let input_str = input_path.to_string_lossy();
    let mut mesh = load_mesh(&input_str)
        .map_err(|e| emit_err(reporter, ImportError::mesh_load(input_path, e)))?;
    stage_end(reporter, "load_mesh");
    check_cancel!();

    // ── Stage: prepare_mesh ─────────────────────────────────────────
    stage_start(reporter, "prepare_mesh", "Normalizing mesh".into());
    let norm = prepare_mesh(&mut mesh, config);
    let aabb = Aabb::new(mesh.bounds_min, mesh.bounds_max);
    let voxel_size = config.voxel_size.unwrap_or_else(|| auto_voxel_size(&aabb));
    stage_end(reporter, "prepare_mesh");

    // ── Stage: build_bvh ────────────────────────────────────────────
    stage_start(
        reporter,
        "build_bvh",
        format!("Building BVH ({} triangles)", mesh.triangle_count()),
    );
    let bvh = TriangleBvh::build(&mesh);
    stage_end(reporter, "build_bvh");
    check_cancel!();

    // ── Stage: extract_skeleton (optional) ──────────────────────────
    stage_start(reporter, "extract_skeleton", "Scanning for skeleton".into());
    let skinning = match skeleton::extract_skeleton(&input_str) {
        Ok(Some(ex)) => {
            reporter.report(ImportEvent::StageProgress {
                stage: "extract_skeleton",
                done: ex.skeleton.bones.len() as u64,
                total: ex.skeleton.bones.len() as u64,
            });
            Some(ex)
        }
        Ok(None) => None,
        // Skeleton failure is non-fatal: mesh import still proceeds
        // without bones, same behaviour as a static mesh.
        Err(e) => {
            reporter.report(ImportEvent::Warn {
                message: format!("skeleton extract failed: {e}"),
            });
            None
        }
    };
    stage_end(reporter, "extract_skeleton");

    // ── Compute grid dimensions ─────────────────────────────────────
    let brick_world_size = voxel_size * BRICK_DIM as f32;
    let padding = voxel_size * 4.0;
    let padded_aabb = Aabb::new(
        aabb.min - Vec3::splat(padding),
        aabb.max + Vec3::splat(padding),
    );
    let aabb_size = padded_aabb.max - padded_aabb.min;
    let max_dim = aabb_size.x.max(aabb_size.y).max(aabb_size.z);
    let bricks_needed = (max_dim / brick_world_size).ceil().max(1.0) as u32;
    let brick_depth = if bricks_needed <= 1 {
        1u8
    } else {
        (32 - (bricks_needed - 1).leading_zeros()) as u8
    };
    let depth = brick_depth + 3; // +3 for per-voxel (8 voxels per brick axis)
    let octree_bricks = 1u32 << brick_depth;
    let extent = octree_bricks as f32 * brick_world_size;
    let aabb_center = (padded_aabb.min + padded_aabb.max) * 0.5;
    let grid_origin = aabb_center - Vec3::splat(extent * 0.5);

    // ── Stage: classify_bricks ──────────────────────────────────────
    stage_start(
        reporter,
        "classify_bricks",
        format!("Classifying {0}×{0}×{0} bricks", octree_bricks),
    );
    let narrow_band = brick_world_size * 1.8;
    let (surface_work, interior_bricks) =
        classify_bricks(&bvh, grid_origin, brick_world_size, octree_bricks, narrow_band);
    reporter.report(ImportEvent::StageProgress {
        stage: "classify_bricks",
        done: (surface_work.len() + interior_bricks.len()) as u64,
        total: (octree_bricks as u64).pow(3),
    });
    stage_end(reporter, "classify_bricks");
    check_cancel!();

    // ── Stage: voxelize_surface (parallel) ──────────────────────────
    let total_surface = surface_work.len() as u64;
    stage_start(
        reporter,
        "voxelize_surface",
        format!(
            "Sampling {} surface bricks ({} interior)",
            total_surface, interior_bricks.len()
        ),
    );
    // Emit ~200 progress events across the stage — enough to feel
    // live in the UI without flooding the event channel on huge
    // meshes. A hot per-brick `report()` call on every brick would
    // be fine too (mpsc is cheap) but throttling keeps Warn/Error
    // events from getting buried under progress spam.
    let progress_step = (total_surface / 200).max(1);
    let counter = AtomicU64::new(0);
    let vertex_skinning = skinning.as_ref().map(|ex| &ex.skinning);
    let results: Vec<_> = surface_work
        .into_par_iter()
        .map(|w| {
            let result = process_brick(
                &mesh, &bvh, w.brick_min, voxel_size, config, vertex_skinning,
            );
            let done = counter.fetch_add(1, Ordering::Relaxed) + 1;
            if done % progress_step == 0 || done == total_surface {
                reporter.report(ImportEvent::StageProgress {
                    stage: "voxelize_surface",
                    done,
                    total: total_surface,
                });
            }
            (w, result)
        })
        .collect();
    stage_end(reporter, "voxelize_surface");
    check_cancel!();

    // ── Stage: emit_shell_leaves ────────────────────────────────────
    stage_start(reporter, "emit_shell_leaves", "Emitting surface shell".into());
    let mut octree = arvx_core::SparseOctree::new(depth, voxel_size);
    let interior_set: HashSet<(u32, u32, u32)> = interior_bricks.iter().copied().collect();
    let shell_output = emit_shell_leaves(
        &mut octree,
        results,
        interior_set,
        brick_depth,
        octree_bricks,
        depth,
        voxel_size,
    );
    reporter.report(ImportEvent::StageProgress {
        stage: "emit_shell_leaves",
        done: shell_output.voxel_count as u64,
        total: shell_output.voxel_count as u64,
    });
    eprintln!(
        "[arvx-import] emit_shell_leaves: {} shell voxels (from {} interior bricks, depth {})",
        shell_output.voxel_count,
        interior_bricks.len(),
        depth,
    );
    stage_end(reporter, "emit_shell_leaves");
    check_cancel!();

    // ── Stage: smooth_normals ───────────────────────────────────────
    // 3 iterations of face-kernel Laplacian relaxation, operating
    // directly on the flat buffers we're about to write. Previously
    // ran at asset-load time in `arvx-render::arvx_scene_manager`;
    // moved here so each asset pays the cost once, not on every
    // load.
    let mut shell_output = shell_output;
    if shell_output.voxel_count > 0 {
        stage_start(
            reporter,
            "smooth_normals",
            format!("Smoothing {} shell normals", shell_output.voxel_count),
        );
        let smoothed = arvx_core::laplacian_smooth::smooth_shell_normals_raw(
            &octree,
            &shell_output.file_bricks,
            &mut shell_output.normals_packed,
            3,
        );
        reporter.report(ImportEvent::StageProgress {
            stage: "smooth_normals",
            done: smoothed as u64,
            total: smoothed as u64,
        });
        stage_end(reporter, "smooth_normals");
        check_cancel!();
    }

    // ── Stage: write_rkp ────────────────────────────────────────────
    stage_start(reporter, "write_rkp", format!("Writing {}", output_path.display()));
    // The `.arvx` header carries a 16-entry material palette. Voxel
    // material IDs are `u16` so they *can* reference up to 65,535
    // materials — but only the first 16 fit in the header's palette
    // field, and the runtime material table is indexed off the
    // project's MaterialPalette rather than the header. This
    // truncation is therefore harmless today, but warn loudly so a
    // future change to the file format doesn't silently drop data.
    let material_ids: Vec<u16> = if let Some(id) = config.material_id_override {
        vec![id]
    } else {
        (0..mesh.materials.len().min(65536) as u16).collect()
    };
    if mesh.materials.len() > 16 {
        reporter.report(ImportEvent::Warn {
            message: format!(
                "mesh has {} materials; only the first 16 fit in the .arvx palette header (voxel material IDs are unaffected — they index the project MaterialPalette at runtime)",
                mesh.materials.len()
            ),
        });
    }
    let file_size = write::write_rkp(
        output_path,
        &shell_output,
        octree.as_slice(),
        depth,
        voxel_size,
        &aabb,
        &material_ids,
        reporter,
    )
    .map_err(|e| emit_err(reporter, ImportError::Write(e)))?;
    stage_end(reporter, "write_rkp");

    // ── Stage: write_rkskel (optional) ──────────────────────────────
    let skeleton_path = if let Some(ref extraction) = skinning {
        stage_start(
            reporter,
            "write_rkskel",
            format!("Writing skeleton ({} bones)", extraction.skeleton.bones.len()),
        );
        let path = write::write_rkskel(output_path, extraction, &norm)
            .map_err(|e| emit_err(reporter, ImportError::Write(e)))?;
        stage_end(reporter, "write_rkskel");
        path
    } else {
        None
    };

    Ok(ImportResult {
        aabb,
        shell_voxels: shell_output.voxel_count,
        finest_voxel_size: voxel_size,
        file_size,
        skeleton_path,
    })
}

fn stage_start(reporter: &dyn ProgressReporter, stage: &'static str, message: String) {
    reporter.report(ImportEvent::StageStart { stage, message });
}

fn stage_end(reporter: &dyn ProgressReporter, stage: &'static str) {
    reporter.report(ImportEvent::StageEnd { stage });
}

fn emit_err(reporter: &dyn ProgressReporter, err: ImportError) -> ImportError {
    reporter.report(ImportEvent::Error { message: err.to_string() });
    err
}
