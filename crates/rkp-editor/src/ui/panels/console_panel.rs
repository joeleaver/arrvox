//! Console panel — log output.

use rinch::prelude::*;

#[component]
pub fn ConsolePanel() -> NodeHandle {
    rsx! {
        div {
            style: "padding:8px 12px;color:#888;font-size:12px;font-style:italic;",
            "Console (coming soon)"
        }
    }
}
