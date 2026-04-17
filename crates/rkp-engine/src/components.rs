//! Built-in ECS components for the RKIPatch engine.
//!
//! These are the standard components that the engine knows about.
//! Additional components can be registered via the ComponentRegistry.

use serde::{Deserialize, Serialize};

/// Spatial transform — position, rotation, scale.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Transform {
    pub position: glam::Vec3,
    /// Euler rotation in degrees (XYZ order).
    pub rotation: glam::Vec3,
    pub scale: glam::Vec3,
}

impl Default for Transform {
    fn default() -> Self {
        Self {
            position: glam::Vec3::ZERO,
            rotation: glam::Vec3::ZERO,
            scale: glam::Vec3::ONE,
        }
    }
}

/// Editor-only metadata (name, locked status, etc.).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EditorMetadata {
    pub name: String,
}

/// Renderable geometry — references a voxelized asset or analytical primitive.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Renderable {
    /// Path to the .rkp asset file (relative to project assets/).
    pub asset_path: Option<String>,
    /// Primitive type if this is an analytical object ("box", "sphere", etc.).
    pub primitive: Option<String>,
    /// Material slot index.
    pub material_id: u16,
    /// Number of voxels (populated after voxelization).
    #[serde(default)]
    pub voxel_count: u32,
    /// Octree spatial reference (populated after voxelization/loading).
    #[serde(skip)]
    pub spatial: Option<SpatialData>,
    /// Handle into the scene manager's asset cache for .rkp-backed
    /// entities. Present when this entity shares an octree with other
    /// instances; the cache refcounts these and frees the underlying
    /// pool ranges when the last instance releases. Procedural geometry
    /// doesn't use this (it owns its octree exclusively).
    #[serde(skip)]
    pub asset_handle: Option<rkp_render::AssetHandle>,
}

/// Octree spatial data for a renderable entity. Not serialized — rebuilt on load.
#[derive(Debug, Clone)]
pub struct SpatialData {
    pub root_offset: u32,
    pub len: u32,
    pub depth: u8,
    pub base_voxel_size: f32,
    pub aabb: rkp_core::Aabb,
    pub voxel_size: f32,
    /// First voxel pool slot used by this allocation, and the count.
    /// Used to free the allocation when geometry is replaced.
    pub voxel_slot_start: u32,
    pub voxel_slot_count: u32,
    /// Brick ids owned by this procedural allocation (empty for .rkp-backed
    /// entities — those are managed by the asset cache). Used to free bricks
    /// on re-voxelize or entity delete.
    pub brick_ids: Vec<u32>,
}

/// Point light source.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct PointLight {
    pub color: [f32; 3],
    pub intensity: f32,
    pub range: f32,
    pub cast_shadow: bool,
}

impl Default for PointLight {
    fn default() -> Self {
        Self {
            color: [1.0, 1.0, 1.0],
            intensity: 5000.0,
            range: 10.0,
            cast_shadow: true,
        }
    }
}

/// Spot light source — like a point light but with a cone direction.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct SpotLight {
    pub color: [f32; 3],
    pub intensity: f32,
    pub range: f32,
    /// Direction the spot light points (normalized).
    pub direction: glam::Vec3,
    /// Outer cone angle in degrees.
    pub outer_angle: f32,
    /// Inner cone angle in degrees (full intensity within this cone).
    pub inner_angle: f32,
    pub cast_shadow: bool,
}

impl Default for SpotLight {
    fn default() -> Self {
        Self {
            color: [1.0, 1.0, 1.0],
            intensity: 10000.0,
            range: 15.0,
            direction: glam::Vec3::new(0.0, -1.0, 0.0),
            outer_angle: 45.0,
            inner_angle: 30.0,
            cast_shadow: true,
        }
    }
}

/// Parent-child relationship — references a parent entity by UUID.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Parent {
    /// UUID of the parent entity (matches the UUID in entity_uuids map).
    pub parent_id: uuid::Uuid,
}

/// Rigid body physics component — configures physics behavior for play mode.
///
/// The physics system reads this at play start to create a Rapier rigid body
/// with the appropriate collider. Not a runtime component — `RigidBodyRuntime`
/// (containing the Rapier handle) is added transiently during play.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct RigidBody {
    /// Body type: Dynamic (gravity+forces), Static (immovable), Kinematic.
    pub body_type: rkp_physics::rigid_body::BodyType,
    /// Collision shape. Auto = voxel collider for voxelized, box for analytical.
    pub collider_shape: rkp_physics::rigid_body::ColliderShape,
    /// Mass in kg (dynamic bodies only).
    pub mass: f32,
    /// Friction coefficient.
    pub friction: f32,
    /// Restitution (bounciness).
    pub restitution: f32,
    /// Voxel size for the physics collider grid (Auto mode only).
    /// Larger = coarser but faster. Default 0.1m.
    pub collider_cell_size: f32,
}

impl Default for RigidBody {
    fn default() -> Self {
        Self {
            body_type: rkp_physics::rigid_body::BodyType::Dynamic,
            collider_shape: rkp_physics::rigid_body::ColliderShape::Auto,
            mass: 1.0,
            friction: 0.5,
            restitution: 0.3,
            collider_cell_size: 0.1,
        }
    }
}

/// Precomputed collider data — cached on the entity, rebuilt when RigidBody
/// settings or geometry change. PlayStart reads this instead of computing on the fly.
#[derive(Debug, Clone)]
pub struct ColliderCache {
    /// The resolved collider shape type.
    pub shape: rkp_physics::rigid_body::ColliderShape,
    /// For voxel colliders: the coarse grid cell coordinates.
    pub voxel_coords: Vec<glam::IVec3>,
    /// Coarse cell size for the collider grid.
    pub collider_cell_size: f32,
    /// AABB half-extents (used for box/sphere/capsule).
    pub aabb_half: glam::Vec3,
    /// Grid origin offset in local space (aabb_center - extent/2).
    /// Used to convert voxel coords to world positions.
    pub grid_origin: glam::Vec3,
    /// Octree depth (for computing grid extent).
    pub tree_depth: u8,
}

/// Runtime Rapier handle — transient, exists only during play mode.
#[derive(Debug, Clone)]
pub struct RigidBodyRuntime {
    pub handle: rapier3d::prelude::RigidBodyHandle,
}

/// Procedural geometry — an entity whose voxels are generated from a node tree.
///
/// The tree is the source of truth. When dirty, the engine re-evaluates the tree
/// into the voxel pool via octree voxelization. The resulting SpatialData is
/// stored on the sibling `Renderable` component.
#[derive(Debug, Clone)]
pub struct ProceduralGeometry {
    /// The procedural node tree (arena-based).
    pub tree: rkp_procedural::ProceduralObject,
    /// Voxel size tier for rendering. Smaller = more detail, more voxels.
    /// Must be one of the standard tiers: 0.005, 0.02, 0.08, 0.32.
    pub voxel_size: f32,

    /// Voxel size for the physics collider grid.
    pub collider_resolution: f32,
    /// Whether the tree needs re-evaluation.
    pub dirty: bool,
    /// Scale at last evaluation — re-evaluate if scale changes.
    pub last_evaluated_scale: glam::Vec3,
}

impl ProceduralGeometry {
    /// Create a default procedural object: a union root with one sphere child.
    pub fn default_sphere() -> Self {
        use rkp_procedural::*;
        let mut tree = ProceduralObject::new(NodeKind::Union {
            material_combine: MaterialCombine::Winner,
        });
        tree.add_child(
            tree.root(),
            NodeKind::Sphere(rkp_procedural::node_kind::SphereParams::default()),
        );
        Self {
            tree,
            voxel_size: 0.02,
            collider_resolution: 0.1,
            dirty: true,
            last_evaluated_scale: glam::Vec3::ONE,
        }
    }
}

/// Skeletal skeleton attached to a voxelized asset.
///
/// Attached automatically by the engine when a `.rkp` load finds a sibling
/// `.rkskel`. Not serialized through the scene file — rebuilt on load from
/// the asset cache, same as `SpatialData`.
///
/// `current_pose` holds the evaluated skinning palette (`world × inverse_bind`
/// per bone), rewritten in place by the animation system each frame.
///
/// `bind_world_origins[i]` is the bone's rest-pose origin in object-local
/// space (= translation component of the accumulated bind transform).
/// Multiplying by `current_pose[i]` cancels the inverse-bind and lands
/// us at the bone's *animated* origin — cheap way to draw a bone gizmo
/// without recomputing world transforms from the hierarchy each frame.
///
/// `inverse_pose[i] = current_pose[i].inverse()` — the matrix that takes
/// a deformed object-local position back to the rest-pose position of
/// the bone's influence region. Phase-3 skin-deform uses it as the
/// inverse LBS basis: the march shader blends `inverse_pose` by the
/// weights in the scattered bone field to invert the deformation.
///
/// `normalize` / `normalize_inverse` encode the mesh-normalisation
/// transform (rotation offset, centre translation, uniform scale)
/// that rkp-import applied to the voxel data but *not* to the
/// skeleton. Bone matrices from `Skeleton::evaluate` are in glTF
/// space; voxel data is in normalised "rkp-local" space. The pose
/// written to `current_pose` / `inverse_pose` is conjugated
/// `N * M * N⁻¹` so both palettes land in the same frame as the
/// voxel data. Computed once at load from the asset's
/// `SkeletonAsset` fields.
#[derive(Clone)]
pub struct Skeleton {
    pub asset: std::sync::Arc<rkp_animation::skeleton_asset::SkeletonAsset>,
    pub path: std::path::PathBuf,
    pub current_pose: Vec<glam::Mat4>,
    pub inverse_pose: Vec<glam::Mat4>,
    pub bind_world_origins: Vec<glam::Vec3>,
    pub normalize: glam::Mat4,
    pub normalize_inverse: glam::Mat4,
    /// Translation that takes a mesh-frame position (origin at object
    /// centre) to a grid-frame position (origin at octree corner).
    /// Equals `half_extent * Vec3::ONE` for a cubic octree. The
    /// gizmo overlay unwinds this to put bones back in mesh frame
    /// before applying the entity's world transform.
    pub grid_offset: glam::Vec3,
}

impl std::fmt::Debug for Skeleton {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Skeleton")
            .field("path", &self.path)
            .field("bones", &self.asset.skeleton.bones.len())
            .field("clips", &self.asset.clips.len())
            .field("pose_len", &self.current_pose.len())
            .finish()
    }
}

impl Skeleton {
    /// Walk the skeleton hierarchy once to accumulate each bone's
    /// bind-pose origin in object-local space. Called at component
    /// construction and again if the skeleton asset is ever hot-swapped.
    pub fn compute_bind_world_origins(
        asset: &rkp_animation::skeleton_asset::SkeletonAsset,
    ) -> Vec<glam::Vec3> {
        let skel = &asset.skeleton;
        let n = skel.bones.len();
        let mut bind_world = vec![glam::Mat4::IDENTITY; n];
        for i in 0..n {
            let local = skel.bones[i].bind_transform;
            let parent = skel.hierarchy.get(i).copied().unwrap_or(-1);
            bind_world[i] = if parent >= 0 && (parent as usize) < n {
                bind_world[parent as usize] * local
            } else {
                local
            };
        }
        bind_world.iter().map(|m| m.transform_point3(glam::Vec3::ZERO)).collect()
    }

    /// Build the glTF→rkp-voxel-space transform from the asset's
    /// normalization parameters. Mirrors the mesh preparation that
    /// `rkp-import::normalize::prepare_mesh` applies:
    /// `v = mesh_scale × (R·(g − rot_center) + rot_center − mesh_center)`
    /// expressed as one `Mat4`:
    /// `Scale(s) · T(rc − mc) · R · T(−rc)`.
    pub fn compute_normalization(
        asset: &rkp_animation::skeleton_asset::SkeletonAsset,
    ) -> glam::Mat4 {
        use glam::{EulerRot, Mat4, Quat, Vec3};
        let mesh_center = Vec3::from_array(asset.mesh_center);
        let mesh_scale = asset.mesh_scale;
        let rotation_center = Vec3::from_array(asset.rotation_center);
        let [rx, ry, rz] = asset.rotation_offset;
        let rot = Quat::from_euler(
            EulerRot::XYZ,
            rx.to_radians(),
            ry.to_radians(),
            rz.to_radians(),
        );
        let t_neg_rc = Mat4::from_translation(-rotation_center);
        let r = Mat4::from_quat(rot);
        let t_rc_minus_mc = Mat4::from_translation(rotation_center - mesh_center);
        let s = Mat4::from_scale(Vec3::splat(mesh_scale));
        s * t_rc_minus_mc * r * t_neg_rc
    }
}

/// Animation playback state for a skinned entity.
///
/// References a clip in the sibling [`Skeleton`]'s asset by name — the clip
/// itself stays on the asset, so many instances can share clip data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnimationPlayer {
    /// Active clip name (empty = no clip selected).
    pub clip_name: String,
    /// Current playback time in seconds.
    pub time: f32,
    /// Playback speed multiplier.
    pub speed: f32,
    /// Whether the player is advancing.
    pub playing: bool,
    /// Loop mode (Once / Loop / PingPong).
    pub loop_mode: rkp_animation::player::LoopMode,
    /// PingPong direction (true = forward). Not user-facing; kept as state
    /// so a scrub-backwards PingPong keeps its direction across frames.
    #[serde(default = "player_default_forward")]
    pub forward: bool,
}

fn player_default_forward() -> bool { true }

impl Default for AnimationPlayer {
    fn default() -> Self {
        Self {
            clip_name: String::new(),
            time: 0.0,
            speed: 1.0,
            playing: false,
            loop_mode: rkp_animation::player::LoopMode::Loop,
            forward: true,
        }
    }
}

/// Camera entity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Camera {
    pub fov: f32,
    pub near: f32,
    pub far: f32,
    pub active: bool,
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            fov: 60.0,
            near: 0.01,
            far: 1000.0,
            active: false,
        }
    }
}
