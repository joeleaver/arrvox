//! # arvx-animation
//!
//! Skeletal animation system for the Arrvox sparse voxel engine.
//!
//! In v2, skeletal animation is pure transform hierarchy animation — bones are
//! nodes in the SceneNode tree and animation = updating local transforms from
//! keyframes. No joint rebaking, no segment management.
//!
//! This crate provides:
//! - Skeleton definition and bone hierarchy types
//! - Animation clip and keyframe data
//! - Playback and blending
//! - Blend shape support via delta-SDF application
//! - Character assembly

#![warn(missing_docs)]

pub mod blend_shape;
pub mod character;
pub mod clip;
pub mod player;
pub mod skeleton;
pub mod skeleton_asset;
