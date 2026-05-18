//! Splitter — draggable divider between containers.
//!
//! Uses `Drag::absolute()` with delta from start position for smooth resizing.

use rinch::prelude::*;

use crate::ui::store::EditorStore;

use super::ContainerKind;

/// Which container boundary this splitter controls.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SplitTarget {
    #[default]
    LeftWidth,
    RightWidth,
    BottomHeight,
}

/// Direction of the splitter.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SplitDirection {
    #[default]
    Vertical,
    Horizontal,
}

#[component]
pub fn ContainerSplitter(direction: SplitDirection, target: SplitTarget) -> NodeHandle {
    let store = use_context::<EditorStore>();

    let style = match direction {
        SplitDirection::Vertical => {
            "width:4px;min-width:4px;flex-shrink:0;cursor:col-resize;\
             background:transparent;position:relative;z-index:10;"
        }
        SplitDirection::Horizontal => {
            "height:4px;min-height:4px;flex-shrink:0;cursor:row-resize;\
             background:transparent;position:relative;z-index:10;"
        }
    };

    rsx! {
        div {
            style: style,
            onmousedown: move || {
                let ctx = get_click_context();
                let is_horizontal = matches!(direction, SplitDirection::Horizontal);
                let start_mouse = if is_horizontal { ctx.mouse_y } else { ctx.mouse_x };
                let start_size = match target {
                    SplitTarget::LeftWidth => store.left_width_px.get(),
                    SplitTarget::RightWidth => store.right_width_px.get(),
                    SplitTarget::BottomHeight => store.bottom_height_px.get(),
                };
                Drag::absolute()
                    .on_move(move |mx, my| {
                        let current = if is_horizontal { my } else { mx };
                        let delta = current - start_mouse;
                        let min_size = 80.0;
                        let new_size = match target {
                            SplitTarget::LeftWidth => (start_size + delta).max(min_size),
                            SplitTarget::RightWidth => (start_size - delta).max(min_size),
                            SplitTarget::BottomHeight => (start_size - delta).max(min_size),
                        };
                        match target {
                            SplitTarget::LeftWidth => store.left_width_px.set(new_size),
                            SplitTarget::RightWidth => store.right_width_px.set(new_size),
                            SplitTarget::BottomHeight => store.bottom_height_px.set(new_size),
                        }
                    })
                    .start();
            },
        }
    }
}

/// Draggable divider between zones within a container.
///
/// Adjusts the `fraction` of the zone above (`zone_idx`) and below (`zone_idx + 1`).
#[component]
pub fn ZoneSplitter(container: ContainerKind, zone_idx: usize) -> NodeHandle {
    let store = use_context::<EditorStore>();

    rsx! {
        div {
            style: "height:4px;min-height:4px;flex-shrink:0;cursor:row-resize;\
                    background:transparent;position:relative;z-index:10;",
            onmousedown: move || {
                let ctx = get_click_context();
                let start_mouse = ctx.mouse_y;

                let layout = store.layout.get();
                let zones = &layout.container(container).zones;
                let frac_total: f32 = zones.iter().map(|z| z.fraction).sum();
                let frac_above = zones.get(zone_idx).map(|z| z.fraction).unwrap_or(0.5);
                let frac_below = zones.get(zone_idx + 1).map(|z| z.fraction).unwrap_or(0.5);
                let frac_sum = frac_above + frac_below;
                let num_splitters = (zones.len() - 1) as f32;

                // Compute the container's available height for zones (excluding splitter chrome).
                //
                // For the bottom panel we know its pixel height directly. For left/right/center,
                // we derive it from the splitter's absolute Y position: the splitter sits at
                //   container_top + (frac_above_cumulative / frac_total) * available + zone_idx * 4
                // so:
                //   available = (element_y - container_top - zone_idx * 4) * frac_total / frac_above_cumulative
                //
                // container_top ≈ 36px (BorderlessWindow title bar height).
                let available_h = if container == ContainerKind::Bottom {
                    (store.bottom_height_px.get() - num_splitters * 4.0).max(100.0)
                } else {
                    let title_bar = 36.0_f32;
                    let frac_above_cumulative: f32 = zones[..=zone_idx].iter().map(|z| z.fraction).sum();
                    let above_px = ctx.element_y - title_bar - zone_idx as f32 * 4.0;
                    (above_px * frac_total / frac_above_cumulative).max(100.0)
                };

                Drag::absolute()
                    .on_move(move |_mx, my| {
                        let delta = my - start_mouse;
                        let frac_delta = delta * frac_total / available_h;
                        let min_frac = 0.05;
                        let new_above = (frac_above + frac_delta).clamp(min_frac, frac_sum - min_frac);
                        let new_below = frac_sum - new_above;
                        store.update_layout(|layout| {
                            let zones = &mut layout.container_mut(container).zones;
                            if let Some(z) = zones.get_mut(zone_idx) {
                                z.fraction = new_above;
                            }
                            if let Some(z) = zones.get_mut(zone_idx + 1) {
                                z.fraction = new_below;
                            }
                        });
                    })
                    .start();
            },
        }
    }
}
