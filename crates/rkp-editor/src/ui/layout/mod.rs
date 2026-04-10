//! Layout data model — containers, zones, tabs, panel IDs.

pub mod container;
pub mod layout_root;
pub mod panel_registry;
pub mod splitter;
pub mod tab_bar;
pub mod zone;

/// Identifies a panel type. Add new panels by adding a variant here
/// and a match arm in `panel_registry::render_panel()`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
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
}

/// Which container a panel lives in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ContainerKind {
    #[default]
    Left,
    Center,
    Right,
    Bottom,
}

/// A zone holds tabs and an active tab index.
#[derive(Debug, Clone, PartialEq)]
pub struct Zone {
    pub tabs: Vec<PanelId>,
    pub active_tab: usize,
    /// Fractional size within the container (normalized, sums to ~1.0).
    pub fraction: f32,
}

/// A container holds one or more zones.
#[derive(Debug, Clone, PartialEq)]
pub struct Container {
    pub kind: ContainerKind,
    pub zones: Vec<Zone>,
    pub visible: bool,
}

/// A floating (detached) panel.
#[derive(Debug, Clone, PartialEq)]
pub struct FloatingPanel {
    pub panel: PanelId,
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

/// Complete layout configuration.
#[derive(Debug, Clone, PartialEq)]
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

    /// Float a panel (remove from container, add to floating list).
    pub fn float_panel(&mut self, container: ContainerKind, zone_idx: usize, tab_idx: usize) {
        let panel = {
            let c = self.container_mut(container);
            if let Some(zone) = c.zones.get_mut(zone_idx) {
                if tab_idx < zone.tabs.len() {
                    let p = zone.tabs.remove(tab_idx);
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
        if let Some(panel) = panel {
            self.floating.push(FloatingPanel {
                panel,
                x: 200.0,
                y: 200.0,
                width: 400.0,
                height: 300.0,
            });
            self.cleanup_empty_zones();
        }
    }

    /// Dock a floating panel back into a container.
    pub fn dock_panel(&mut self, floating_idx: usize, target: ContainerKind, zone_idx: usize) {
        if floating_idx < self.floating.len() {
            let fp = self.floating.remove(floating_idx);
            let container = self.container_mut(target);
            if let Some(zone) = container.zones.get_mut(zone_idx) {
                zone.tabs.push(fp.panel);
                zone.active_tab = zone.tabs.len() - 1;
            }
            container.visible = true;
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
                    tabs: vec![PanelId::ObjectProperties],
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
                tabs: vec![PanelId::Materials, PanelId::Models, PanelId::Console, PanelId::Profiling],
                active_tab: 0,
                fraction: 1.0,
            }],
            visible: true,
        },
        floating: Vec::new(),
    }
}
