//! Models panel — asset browser.

use rinch::prelude::*;

#[component]
pub fn ModelsPanel() -> NodeHandle {
    rsx! {
        div {
            style: "padding:8px 12px;color:#888;font-size:12px;font-style:italic;",
            "Models browser (coming soon)"
        }
    }
}
