//! Layout data model — containers, zones, tabs, panel IDs.

pub mod container;
pub mod layout_root;
pub mod panel_registry;
pub mod persist;
pub mod splitter;
pub mod tab_bar;
pub mod zone;

use serde::{Deserialize, Serialize};

/// Identifies a panel type. Add new panels by adding a variant here
/// and a match arm in `panel_registry::render_panel()`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub enum PanelId {
    #[default]
    SceneTree,
    SceneView,
    ObjectProperties,
    AssetProperties,
    Environment,
    Materials,
    Console,
    Profiling,
    Models,
    Shaders,
    Build,
}

/// Which container a panel lives in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum ContainerKind {
    #[default]
    Left,
    Center,
    Right,
    Bottom,
}

/// A zone holds tabs and an active tab index.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Zone {
    pub tabs: Vec<PanelId>,
    pub active_tab: usize,
    /// Fractional size within the container (normalized, sums to ~1.0).
    pub fraction: f32,
}

/// A container holds one or more zones.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Container {
    pub kind: ContainerKind,
    pub zones: Vec<Zone>,
    pub visible: bool,
}

/// A floating (detached) panel.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FloatingPanel {
    pub panel: PanelId,
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

/// Complete layout configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LayoutConfig {
    pub left: Container,
    pub center: Container,
    pub right: Container,
    pub bottom: Container,
    pub floating: Vec<FloatingPanel>,
}

impl LayoutConfig {
    /// Get a container by kind.
    pub fn container(&self, kind: ContainerKind) -> &Container {
        match kind {
            ContainerKind::Left => &self.left,
            ContainerKind::Center => &self.center,
            ContainerKind::Right => &self.right,
            ContainerKind::Bottom => &self.bottom,
        }
    }

    /// Get a mutable container by kind.
    pub fn container_mut(&mut self, kind: ContainerKind) -> &mut Container {
        match kind {
            ContainerKind::Left => &mut self.left,
            ContainerKind::Center => &mut self.center,
            ContainerKind::Right => &mut self.right,
            ContainerKind::Bottom => &mut self.bottom,
        }
    }

    /// Append any panel ids that exist in the registry but aren't
    /// referenced anywhere in the loaded layout. Runs at load time so
    /// upgrading the editor doesn't strand new panels behind a saved
    /// project's frozen tab list. New panels land alongside their
    /// closest sibling (Shaders next to Models, etc.); fallback is
    /// the bottom container's first zone.
    pub fn migrate_panels(&mut self) {
        // All panel ids known to this build. Order matters only for
        // the "what's missing" diff — placement uses the per-id
        // sibling map below.
        let known: &[PanelId] = &[
            PanelId::SceneTree,
            PanelId::SceneView,
            PanelId::ObjectProperties,
            PanelId::AssetProperties,
            PanelId::Environment,
            PanelId::Materials,
            PanelId::Console,
            PanelId::Profiling,
            PanelId::Models,
            PanelId::Shaders,
            PanelId::Build,
        ];

        // Collect every id currently referenced in any zone.
        let mut present: std::collections::HashSet<PanelId> =
            std::collections::HashSet::new();
        for c in [&self.left, &self.center, &self.right, &self.bottom] {
            for z in &c.zones {
                for &id in &z.tabs {
                    present.insert(id);
                }
            }
        }
        for f in &self.floating {
            present.insert(f.panel);
        }

        // For each missing panel, find a sibling already in the layout
        // and append next to it. The fallback zone is the bottom
        // container's first zone (which the default layout pre-creates).
        let sibling_for = |id: PanelId| -> &'static [PanelId] {
            match id {
                PanelId::Shaders => &[PanelId::Models, PanelId::Materials],
                _ => &[],
            }
        };

        for &id in known {
            if present.contains(&id) {
                continue;
            }
            // Try siblings first, fall back to bottom[0].
            let mut placed = false;
            'outer: for &sibling in sibling_for(id) {
                for c in [
                    ContainerKind::Bottom,
                    ContainerKind::Left,
                    ContainerKind::Right,
                    ContainerKind::Center,
                ] {
                    let container = self.container_mut(c);
                    for z in &mut container.zones {
                        if z.tabs.contains(&sibling) {
                            z.tabs.push(id);
                            placed = true;
                            break 'outer;
                        }
                    }
                }
            }
            if !placed {
                if let Some(z) = self.bottom.zones.first_mut() {
                    z.tabs.push(id);
                } else {
                    // No bottom zone at all (very unusual loaded state) —
                    // create one so the panel isn't dropped.
                    self.bottom.zones.push(Zone {
                        tabs: vec![id],
                        active_tab: 0,
                        fraction: 1.0,
                    });
                    self.bottom.visible = true;
                }
            }
        }
    }

    /// Switch the active tab in a zone.
    pub fn set_active_tab(&mut self, kind: ContainerKind, zone_idx: usize, tab_idx: usize) {
        let container = self.container_mut(kind);
        if let Some(zone) = container.zones.get_mut(zone_idx) {
            if tab_idx < zone.tabs.len() {
                zone.active_tab = tab_idx;
            }
        }
    }

    /// Move a tab from one zone to another.
    pub fn move_tab(
        &mut self,
        from_container: ContainerKind,
        from_zone: usize,
        from_tab: usize,
        to_container: ContainerKind,
        to_zone: usize,
    ) {
        // Extract the panel.
        let panel = {
            let src = self.container_mut(from_container);
            if let Some(zone) = src.zones.get_mut(from_zone) {
                if from_tab < zone.tabs.len() {
                    let p = zone.tabs.remove(from_tab);
                    if zone.active_tab >= zone.tabs.len() && zone.active_tab > 0 {
                        zone.active_tab -= 1;
                    }
                    Some(p)
                } else {
                    None
                }
            } else {
                None
            }
        };

        // Insert into target zone.
        if let Some(panel) = panel {
            let dst = self.container_mut(to_container);
            if let Some(zone) = dst.zones.get_mut(to_zone) {
                zone.tabs.push(panel);
                zone.active_tab = zone.tabs.len() - 1;
            }
        }

        self.cleanup_empty_zones();
    }

    /// Split a zone by inserting a new zone with the given panel before or after it.
    pub fn split_zone(
        &mut self,
        panel: PanelId,
        container: ContainerKind,
        zone_idx: usize,
        before: bool,
    ) {
        let c = self.container_mut(container);
        if zone_idx < c.zones.len() {
            let new_zone = Zone {
                tabs: vec![panel],
                active_tab: 0,
                fraction: 0.5,
            };
            // Halve the existing zone's fraction.
            c.zones[zone_idx].fraction *= 0.5;
            let insert_at = if before { zone_idx } else { zone_idx + 1 };
            c.zones.insert(insert_at, new_zone);
        }
    }

    /// Remove empty zones and hide containers with no content.
    pub fn cleanup_empty_zones(&mut self) {
        for container in [&mut self.left, &mut self.center, &mut self.right, &mut self.bottom] {
            container.zones.retain(|z| !z.tabs.is_empty());
            if container.zones.is_empty() {
                container.visible = false;
            }
            // Normalize fractions.
            let total: f32 = container.zones.iter().map(|z| z.fraction).sum();
            if total > 0.0 {
                for zone in &mut container.zones {
                    zone.fraction /= total;
                }
            }
        }
    }

}

/// Default editor layout.
pub fn default_layout() -> LayoutConfig {
    LayoutConfig {
        left: Container {
            kind: ContainerKind::Left,
            zones: vec![Zone {
                tabs: vec![PanelId::SceneTree],
                active_tab: 0,
                fraction: 1.0,
            }],
            visible: true,
        },
        center: Container {
            kind: ContainerKind::Center,
            zones: vec![Zone {
                tabs: vec![PanelId::SceneView],
                active_tab: 0,
                fraction: 1.0,
            }],
            visible: true,
        },
        right: Container {
            kind: ContainerKind::Right,
            zones: vec![
                Zone {
                    tabs: vec![PanelId::ObjectProperties, PanelId::Build],
                    active_tab: 0,
                    fraction: 0.5,
                },
                Zone {
                    tabs: vec![PanelId::AssetProperties, PanelId::Environment],
                    active_tab: 0,
                    fraction: 0.5,
                },
            ],
            visible: true,
        },
        bottom: Container {
            kind: ContainerKind::Bottom,
            zones: vec![Zone {
                tabs: vec![PanelId::Materials, PanelId::Models, PanelId::Shaders, PanelId::Console, PanelId::Profiling],
                active_tab: 0,
                fraction: 1.0,
            }],
            visible: true,
        },
        floating: Vec::new(),
    }
}
