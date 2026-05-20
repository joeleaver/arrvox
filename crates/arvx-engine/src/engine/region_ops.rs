//! Region lifecycle on the engine side — spawn.
//!
//! Regions are simpler than stamps because nothing consumes them yet
//! (Phase 7 wires terrain biomes; other consumers — audio, fog,
//! triggers — land in their own phases). For Phase 6 spawn just adds
//! an entity with `Transform + EditorMetadata + Region`. The
//! component-registry "Add Component" affordance attaches a
//! `BiomeRegion` (or future consumer data) on demand.

use arvx_regions::{Falloff, Region, RegionShape};
use glam::{Quat, Vec3};

use crate::command::RegionShapeSpec;
use crate::components::{EditorMetadata, Transform};

impl super::state::EngineState {
    /// Spawn a Region entity with the given shape at the supplied
    /// world position. Selects the new entity (consistent with every
    /// other Spawn handler) so the gizmo and Inspector light up
    /// immediately.
    pub(crate) fn handle_spawn_region(&mut self, shape_spec: RegionShapeSpec, position: Vec3) {
        let (shape, label) = build_default_shape(shape_spec);
        let region = Region {
            shape,
            falloff: Falloff::Smoothstep { transition_m: 5.0 },
            priority: 0,
        };

        let name = self.unique_name(&format!("{label} Region"));
        let mut transform = Transform::default();
        transform.position = position;
        let entity = self.world.spawn((
            transform,
            EditorMetadata { name: name.clone() },
            region,
        ));
        self.assign_entity_uuid(entity);
        self.scene_dirty.mark_entity(entity);
        self.selected_entity = Some(entity);
        self.console.info(format!("Spawned '{name}'"));
    }
}

fn build_default_shape(spec: RegionShapeSpec) -> (RegionShape, &'static str) {
    match spec {
        RegionShapeSpec::Sphere => (RegionShape::Sphere { radius: 25.0 }, "Sphere"),
        RegionShapeSpec::Box => (
            RegionShape::Box {
                half_extents: Vec3::new(15.0, 15.0, 15.0),
            },
            "Box",
        ),
        RegionShapeSpec::Obb => (
            RegionShape::Obb {
                half_extents: Vec3::new(15.0, 15.0, 15.0),
                rotation: Quat::IDENTITY,
            },
            "OBB",
        ),
    }
}
