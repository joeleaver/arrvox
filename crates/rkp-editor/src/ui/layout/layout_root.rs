//! Layout root — assembles the 4 containers + splitters inside a BorderlessWindow.

use rinch::prelude::*;

use super::ContainerKind;
use super::container::ContainerComponent;
use super::splitter::{ContainerSplitter, SplitDirection, SplitTarget};
use crate::ui::store::EditorStore;
use crate::ui::panels::{StatusBar, WelcomeScreen};

#[component]
pub fn LayoutRoot() -> NodeHandle {
    let store = use_context::<EditorStore>();

    // Title section for the borderless window titlebar.
    let title_section: std::rc::Rc<dyn Fn(&mut rinch::core::dom::RenderScope) -> rinch::core::dom::NodeHandle> =
        std::rc::Rc::new(|__scope| {
            rsx! {
                div {
                    style: "padding:0 8px;display:flex;align-items:center;\
                            font-size:12px;font-weight:600;color:var(--rinch-color-text);\
                            letter-spacing:0.5px;white-space:nowrap;",
                    "RKIPatch"
                }
            }
        });

    rsx! {
        BorderlessWindow {
            radius: "none",
            on_minimize: minimize_current_window,
            on_maximize: toggle_maximize_current_window,
            on_close: close_current_window,
            left_section: Some(title_section),

            // Dark theme CSS overrides for rinch components.
            style {
                {"
                    :root {
                        --rinch-color-default-border: #3c3c3c;
                        --rinch-color-surface: #2d2d2d;
                        --rinch-color-text: #ccc;
                        --rinch-color-dimmed: #888;
                    }
                    .rinch-color-input__input {
                        color: #ccc !important;
                        font-size: 11px !important;
                        height: 26px !important;
                    }
                    .rinch-color-input__input-group {
                        border-color: #3c3c3c !important;
                        background-color: #1e1e1e !important;
                    }
                    .rinch-color-input__dropdown {
                        background-color: #2d2d2d !important;
                        border-color: #3c3c3c !important;
                    }
                    .rinch-dropdown-menu__dropdown-inner {
                        background-color: #2d2d2d !important;
                        border-color: #3c3c3c !important;
                    }
                    .rinch-dropdown-menu__item {
                        color: #ccc !important;
                    }
                    .rinch-dropdown-menu__item:hover {
                        background-color: #37373d !important;
                    }
                    .rinch-context-menu__dropdown-inner {
                        background-color: #2d2d2d !important;
                        border-color: #3c3c3c !important;
                    }
                "}
            }
            div {
                style: "display:flex;flex-direction:column;width:100%;height:100%;\
                        position:relative;overflow:hidden;\
                        background:var(--rinch-color-dark-9);color:var(--rinch-color-text);",

                // Main area
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

                // Status bar
                StatusBar {}

                // Welcome screen overlay
                if !store.project_loaded.get() {
                    WelcomeScreen {}
                }
            }
        }
    }
}
