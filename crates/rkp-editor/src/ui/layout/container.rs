//! Container component — renders zones within a container (left, center, right, bottom).

use rinch::prelude::*;

use super::ContainerKind;
use super::zone::ZoneComponent;
use crate::ui::store::EditorStore;

#[component]
pub fn ContainerComponent(kind: ContainerKind) -> NodeHandle {
    let store = use_context::<EditorStore>();

    // Direction: left/right stack zones vertically, bottom stacks horizontally.
    let flex_dir = match kind {
        ContainerKind::Left | ContainerKind::Right => "column",
        ContainerKind::Center | ContainerKind::Bottom => "column",
    };

    let zone_count = Memo::new(move || {
        store.layout.get().container(kind).zones.len()
    });

    rsx! {
        div {
            style: {|| format!(
                "display:flex;flex-direction:{};flex:1;min-height:0;min-width:0;\
                 background:#1e1e1e;overflow:hidden;",
                flex_dir
            )},
            // For now, render zones with simple flex sizing.
            // TODO: ZoneSplitters between zones when zone_count > 1.
            for i in 0..zone_count.get() {
                ZoneComponent {
                    key: i.to_string(),
                    container: kind,
                    zone_idx: i,
                }
            }
        }
    }
}
