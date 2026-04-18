//! Layout root — assembles the 4 containers + splitters inside a BorderlessWindow.

use rinch::prelude::*;

use super::ContainerKind;
use super::container::ContainerComponent;
use super::splitter::{ContainerSplitter, SplitDirection, SplitTarget};
use crate::CommandSender;
use crate::ui::store::EditorStore;
use crate::ui::panels::{StatusBar, WelcomeScreen};

#[component]
pub fn LayoutRoot() -> NodeHandle {
    let store = use_context::<EditorStore>();
    let cmd_tx = Signal::new(use_context::<CommandSender>().0);

    // "Convert to Voxel Object" confirmation. Mounted here (full-
    // window bounds) rather than inside `SceneTree` because rinch's
    // hit-test skips descendants of `overflow` containers when the
    // click lies outside the parent's bounds — a centered modal
    // inside a narrow panel would never catch its own button
    // clicks.
    let convert_target = store.convert_procedural_target;
    let convert_name = Memo::new(move || {
        let Some(id) = convert_target.get() else { return String::new() };
        store.objects.get()
            .into_iter()
            .find(|o| o.id == id)
            .map(|o| o.name)
            .unwrap_or_default()
    });
    let on_confirm = move || {
        if let Some(id) = convert_target.get() {
            let _ = cmd_tx.get().send(
                rkp_engine::EngineCommand::ConvertProceduralToVoxel { entity_id: id },
            );
        }
        convert_target.set(None);
    };

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

                Modal {
                    opened_fn: move || convert_target.get().is_some(),
                    onclose: move || { convert_target.set(None); },
                    title: "Convert to voxel object?",
                    size: "sm",
                    centered: true,
                    with_overlay: true,
                    close_on_click_outside: true,
                    close_on_escape: true,
                    with_close_button: true,
                    Stack { gap: "md",
                        Text { size: "sm",
                            {move || format!(
                                "Drop the procedural tree on \"{}\" and keep the \
                                 currently-baked voxels as a plain voxel object. The \
                                 tree will be lost — you won't be able to tweak its \
                                 parameters afterwards.",
                                convert_name.get()
                            )}
                        }
                        Text { size: "xs", color: "dimmed",
                            "Tip: \"Copy to New Voxel Object\" does the same thing \
                             without destroying the original."
                        }
                        Group { justify: "flex-end", gap: "sm",
                            Button {
                                variant: "subtle",
                                onclick: move || { convert_target.set(None); },
                                "Cancel"
                            }
                            Button {
                                color: "red",
                                onclick: on_confirm,
                                "Convert"
                            }
                        }
                    }
                }
            }
        }
    }
}
