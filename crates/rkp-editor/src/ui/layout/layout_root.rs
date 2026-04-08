//! Layout root — assembles the 4 containers + splitters.

use rinch::prelude::*;

use super::ContainerKind;
use super::container::ContainerComponent;
use super::floating::FloatingPanelHost;
use super::splitter::{ContainerSplitter, SplitDirection, SplitTarget};
use crate::ui::store::EditorStore;
use crate::ui::panels::StatusBar;

#[component]
pub fn LayoutRoot() -> NodeHandle {
    let store = use_context::<EditorStore>();

    rsx! {
        div {
            style: "display:flex;flex-direction:column;width:100%;height:100%;background:#1e1e1e;\
                    position:relative;",

            // Titlebar
            div {
                style: "height:36px;display:flex;align-items:center;padding:0 16px;\
                        background:#323233;color:#ccc;font-size:13px;font-weight:500;\
                        flex-shrink:0;border-bottom:1px solid #3c3c3c;",
                "RKIPatch Editor"
            }

            // Main area. ondragleave clears drop target when cursor exits
            // entirely. Child zone ondragenter immediately re-sets it if
            // the cursor is still over a zone.
            div {
                style: "display:flex;flex:1;min-height:0;",
                ondragleave: move || {
                    store.drop_target.set(None);
                },

                // Left container
                div {
                    style: {
                        move || format!(
                            "width:{:.0}px;flex-shrink:0;display:flex;flex-direction:column;\
                             min-height:0;",
                            store.left_width_px.get()
                        )
                    },
                    ContainerComponent { kind: ContainerKind::Left }
                }

                // Splitter: left ↔ center
                ContainerSplitter {
                    direction: SplitDirection::Vertical,
                    target: SplitTarget::LeftWidth,
                }

                // Center area (center + bottom stacked)
                div {
                    style: "display:flex;flex-direction:column;flex:1;min-height:0;min-width:0;",

                    // Center container
                    div {
                        style: "display:flex;flex:1;min-height:0;",
                        ContainerComponent { kind: ContainerKind::Center }
                    }

                    // Splitter: center ↔ bottom
                    ContainerSplitter {
                        direction: SplitDirection::Horizontal,
                        target: SplitTarget::BottomHeight,
                    }

                    // Bottom container
                    div {
                        style: {
                            move || format!(
                                "height:{:.0}px;flex-shrink:0;display:flex;",
                                store.bottom_height_px.get()
                            )
                        },
                        ContainerComponent { kind: ContainerKind::Bottom }
                    }
                }

                // Splitter: center ↔ right
                ContainerSplitter {
                    direction: SplitDirection::Vertical,
                    target: SplitTarget::RightWidth,
                }

                // Right container
                div {
                    style: {
                        move || format!(
                            "width:{:.0}px;flex-shrink:0;display:flex;flex-direction:column;\
                             min-height:0;",
                            store.right_width_px.get()
                        )
                    },
                    ContainerComponent { kind: ContainerKind::Right }
                }
            }

            // Floating panels (absolutely positioned over everything)
            FloatingPanelHost {}

            // Status bar
            StatusBar {}
        }
    }
}
