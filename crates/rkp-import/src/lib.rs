//! # rkp-import
//!
//! Mesh-to-`.rkp` import pipeline for RKIPatch. Owns the full path from a
//! source mesh file (`.glb`, `.gltf`, `.obj`, `.fbx`) to an opacity-octree
//! asset on disk, including triangle BVH, per-voxel signed-distance
//! sampling, thin-shell classification, baked SDF-gradient normals, and
//! `.rkskel` sidecar writing.
//!
//! Rewrite of the original `rkf-import` crate, scoped to RKIPatch's
//! opacity-octree format with a structured progress-event API so the
//! editor UI can stream import status instead of watching stdout.

#![warn(missing_docs)]

pub mod bvh;
pub mod config;
pub mod error;
pub mod event;
pub mod mesh;
pub mod normalize;
pub mod sample;
pub mod skeleton;
pub mod voxelize;

pub use config::{ImportConfig, ImportResult};
pub use error::ImportError;
pub use event::{
    CancelHandle, CancelToken, ImportEvent, NullReporter, ProgressReporter, StderrReporter,
};
pub use voxelize::{import_mesh_to_opacity_rkp, import_mesh_to_opacity_rkp_with};
