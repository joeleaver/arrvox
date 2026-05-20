//! # arvx-regions
//!
//! Cross-cutting region primitive for the Arrvox engine.
//!
//! A *region* is a named volume in the world that consumers (biomes,
//! audio zones, fog volumes, gameplay triggers, AI behaviour areas)
//! query for soft membership. Building the primitive as a separate
//! crate lets each consumer attach its own ECS data component while
//! sharing one spatial index and one membership function.
//!
//! See `docs/REGIONS.md` for the full design. V1 ships analytical
//! shapes (Sphere / Box / OBB), three falloff curves (Hard / Linear /
//! Smoothstep), `membership(...)`, and a BVH-backed `RegionIndex`.
//! `BiomeRegion` (the V1 consumer's data component) lives in
//! `arvx-terrain` because it carries a `TerrainFn` override; the
//! pattern there is the template for future consumers.

#![warn(missing_docs)]

pub mod bvh;
pub mod falloff;
pub mod index;
pub mod region;
pub mod shape;

pub use bvh::RegionBvh;
pub use falloff::Falloff;
pub use index::{RegionEntry, RegionIndex};
pub use region::{membership, Region};
pub use shape::RegionShape;
