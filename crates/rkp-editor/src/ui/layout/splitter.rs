//! Splitter — draggable divider between containers.
//!
//! Uses `Drag::absolute()` with delta from start position for smooth resizing.

use rinch::prelude::*;

use crate::ui::store::EditorStore;

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
