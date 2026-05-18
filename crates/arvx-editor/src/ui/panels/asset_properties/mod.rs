//! Asset Properties panel — shows editable properties for the selected asset.
//!
//! Dispatches based on what kind of asset is currently selected:
//! - Material selected → material PBR properties
//! - Model selected → import profile settings
//!
//! When nothing is selected, shows a placeholder message.

// `info` bound inside `if let Some(info) = ...` expands through the rsx
// macro into generated closures that rustc's usage analysis can't see
// past. The binding looks unused at the pattern site even though the
// expanded code uses it. Local `let _ = &info;` silencers are stripped
// by the macro, so we allow at module scope.
#![allow(unused_variables)]

use rinch::prelude::*;

use crate::ui::store::EditorStore;

mod material;
mod model;

use material::MaterialPropertiesSection;
use model::ModelPropertiesSection;

type CmdSignal = Signal<crossbeam::channel::Sender<arvx_engine::EngineCommand>>;

#[component]
pub fn AssetProperties() -> NodeHandle {
    let store = use_context::<EditorStore>();

    let has_material = Memo::new(move || store.selected_material.get().is_some());
    let has_model = Memo::new(move || {
        store.selected_model.get().is_some() && store.selected_material.get().is_none()
    });
    let has_nothing = Memo::new(move || {
        store.selected_material.get().is_none() && store.selected_model.get().is_none()
    });

    rsx! {
        div {
            style: "display:flex;flex-direction:column;height:100%;overflow-y:auto;",

            if has_material.get() {
                MaterialPropertiesSection {}
            }

            if has_model.get() {
                ModelPropertiesSection {}
            }

            if has_nothing.get() {
                div {
                    style: "padding:12px;color:#555;font-size:12px;font-style:italic;",
                    {"Select an asset to edit its properties"}
                }
            }
        }
    }
}

/// Snap a stored voxel size to the nearest standard tier string so the
/// dropdown always shows a value it offers. Out-of-tier values (legacy
/// profiles, hand-edited sidecars) get the nearest match.
fn format_voxel_tier(v: f32) -> String {
    const TIERS: [f32; 4] = [0.005, 0.02, 0.08, 0.32];
    let mut best = TIERS[0];
    let mut best_err = (v - best).abs();
    for &t in &TIERS[1..] {
        let err = (v - t).abs();
        if err < best_err {
            best = t;
            best_err = err;
        }
    }
    format!("{best}")
}

fn format_size(bytes: u64) -> String {
    if bytes < 1024 {
        format!("{bytes} B")
    } else if bytes < 1024 * 1024 {
        format!("{:.1} KB", bytes as f64 / 1024.0)
    } else {
        format!("{:.1} MB", bytes as f64 / (1024.0 * 1024.0))
    }
}

/// Pretty-print a shell voxel count for the model header. Zero
/// renders as a blank string so legacy `.arvx` files whose header
/// couldn't be read don't display "0 voxels".
fn format_voxel_count(count: u32) -> String {
    if count == 0 {
        return String::new();
    }
    if count < 1_000 {
        format!("{count} voxels")
    } else if count < 1_000_000 {
        format!("{:.1}K voxels", count as f64 / 1_000.0)
    } else {
        format!("{:.2}M voxels", count as f64 / 1_000_000.0)
    }
}
