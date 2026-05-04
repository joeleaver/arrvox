//! Model asset panel: import-profile fields (voxel size tier, padding,
//! center origin) + live import-progress UI.

#![allow(unused_variables)]

use std::rc::Rc;

use rinch::prelude::*;
use rinch_tabler_icons::{TablerIcon, TablerIconStyle, render_tabler_icon};

use crate::CommandSender;
use crate::ui::store::EditorStore;
use crate::ui::panels::prop_controls::*;

use super::{CmdSignal, format_size, format_voxel_count, format_voxel_tier};

// ── Model import properties ───────────────────────────────────────────────

#[component]
pub(super) fn ModelPropertiesSection() -> NodeHandle {
    let store = use_context::<EditorStore>();

    let model_info = Memo::new(move || {
        let sel_path = store.selected_model.get()?;
        store.available_models.get().iter()
            .find(|m| m.source_path == sel_path)
            .cloned()
    });

    rsx! {
        div {
            style: "display:flex;flex-direction:column;height:100%;",

            // Remount the form on selection change by keying a
            // single-element `for` loop on the source path. Without
            // this, `if let Some(info)` keeps the form's inner
            // Signals alive across selection switches and the form
            // stays pinned to the first-selected model's values —
            // same pattern used by `ComponentSection` in
            // `object_properties.rs` to remount component editors
            // when the selected entity changes.
            for info in model_info.get().into_iter() {
                ModelPropertiesForm {
                    key: info.source_path.clone(),
                    info: info,
                }
            }
        }
    }
}

#[component]
fn ModelPropertiesForm(info: rkp_engine::ModelInfo) -> NodeHandle {
    let cmd_tx: CmdSignal = Signal::new(use_context::<CommandSender>().0);
    let store = use_context::<EditorStore>();
    // Project-relative source path for the header subtitle. Kept as
    // a Memo so it updates when the project root signal changes on
    // project load/switch.
    let display_source_path = Memo::new({
        let sp = info.source_path.clone();
        move || crate::ui::path_display::display_rel_path(&sp, &store.project_dir.get())
    });

    rsx! {
        div {
            style: "display:flex;flex-direction:column;gap:0;height:100%;",

            // Header — fields are the initial values from `info`;
            // they never stale out because the parent remounts this
            // whole component on selection change.
            div {
                style: "display:flex;align-items:center;padding:6px 8px;\
                        border-bottom:1px solid #333;gap:6px;flex-shrink:0;",
                span {
                    style: "width:20px;height:20px;display:inline-flex;\
                            align-items:center;justify-content:center;\
                            flex-shrink:0;color:#999;",
                    {render_tabler_icon(__scope, TablerIcon::Cube, TablerIconStyle::Outline)}
                }
                div {
                    style: "flex:1;min-width:0;",
                    div {
                        style: "font-size:12px;font-weight:600;color:#ddd;\
                                overflow:hidden;text-overflow:ellipsis;white-space:nowrap;",
                        {info.name.clone()}
                    }
                    div {
                        style: "font-size:10px;color:#666;overflow:hidden;\
                                text-overflow:ellipsis;white-space:nowrap;",
                        {|| display_source_path.get()}
                    }
                }
                div {
                    style: "font-size:10px;color:#666;flex-shrink:0;\
                            display:flex;flex-direction:column;align-items:flex-end;",
                    div { {format_size(info.size)} }
                    div { {format_voxel_count(info.voxel_count)} }
                }
            }

            // Import fields
            {model_import_fields(__scope, info.source_path.clone(), info.import_profile.clone(), cmd_tx)}
        }
    }
}

fn model_import_fields(
    __scope: &mut rinch::core::dom::RenderScope,
    source_path: String,
    profile: Option<rkp_engine::import_profile::ImportProfile>,
    cmd_tx: CmdSignal,
) -> rinch::core::dom::NodeHandle {
    let store = use_context::<EditorStore>();
    let profile = profile.unwrap_or_default();
    let src = Signal::new(source_path);

    // Track whether this source is currently being re-imported on the
    // engine thread. Used below to swap the Re-import button for a
    // progress indicator — voxelizing a high-poly mesh at a fine tier
    // can take minutes, and the button gave no feedback otherwise.
    let is_importing = Memo::new(move || {
        let s = src.get();
        store.importing_models.get().iter().any(|p| *p == s)
    });
    // Look up the matching live-progress record. Returns None when
    // no import is in flight for this source (button mode) or when
    // the import just started and no events have landed yet.
    let progress = Memo::new(move || {
        let s = src.get();
        store
            .import_progress
            .get()
            .iter()
            .find(|p| p.source_path == s)
            .cloned()
    });

    let display_name = Signal::new(profile.display_name.unwrap_or_default());
    // Voxel size is chosen from the four standard tiers (matches the
    // procedural build panel + mesh-import auto-detect tiers in
    // voxelize_opacity::auto_voxel_size). "auto" means pick the coarsest
    // tier that still gives ≥8 bricks on the longest mesh axis.
    let voxel_size_str = Signal::new(
        profile.voxel_size
            .map(format_voxel_tier)
            .unwrap_or_else(|| "auto".into()),
    );
    let target_size = Signal::new(profile.target_size);
    let no_normalize = Signal::new(profile.no_normalize);
    let import_colors = Signal::new(profile.import_colors);
    let rot_x = Signal::new(profile.rotation_offset[0]);
    let rot_y = Signal::new(profile.rotation_offset[1]);
    let rot_z = Signal::new(profile.rotation_offset[2]);

    // Helper to create an import field update callback.
    let import_cb = move |field: &'static str| -> Rc<dyn Fn(String)> {
        Rc::new(move |v: String| {
            let _ = cmd_tx.get().send(rkp_engine::EngineCommand::UpdateImportField {
                source_path: src.get(),
                field: field.into(),
                value: v,
            });
        })
    };
    let import_cb_f32 = move |field: &'static str| -> Rc<dyn Fn(f32)> {
        Rc::new(move |v: f32| {
            let _ = cmd_tx.get().send(rkp_engine::EngineCommand::UpdateImportField {
                source_path: src.get(),
                field: field.into(),
                value: v.to_string(),
            });
        })
    };
    let import_cb_bool = move |field: &'static str| -> Rc<dyn Fn(bool)> {
        Rc::new(move |v: bool| {
            let _ = cmd_tx.get().send(rkp_engine::EngineCommand::UpdateImportField {
                source_path: src.get(),
                field: field.into(),
                value: v.to_string(),
            });
        })
    };

    rsx! {
        div {
            style: "padding:8px;display:flex;flex-direction:column;gap:6px;\
                    overflow-y:auto;flex:1;",

            {prop_text(__scope, "Name", display_name, import_cb("display_name"))}
            {prop_select(
                __scope,
                "Voxel Size",
                Memo::new(move || voxel_size_str.get()),
                &[
                    ("auto", "Auto"),
                    ("0.005", "5mm (finest)"),
                    ("0.02", "2cm"),
                    ("0.08", "8cm"),
                    ("0.32", "32cm (coarsest)"),
                ],
                import_cb("voxel_size"),
            )}
            {prop_slider(__scope, "Target Size", target_size, 0.1, 10.0, 0.1, import_cb_f32("target_size"))}
            {prop_slider(__scope, "Rotate X", rot_x, -180.0, 180.0, 1.0, import_cb_f32("rotation_x"))}
            {prop_slider(__scope, "Rotate Y", rot_y, -180.0, 180.0, 1.0, import_cb_f32("rotation_y"))}
            {prop_slider(__scope, "Rotate Z", rot_z, -180.0, 180.0, 1.0, import_cb_f32("rotation_z"))}
            {prop_checkbox(__scope, "Import Colors", import_colors, import_cb_bool("import_colors"))}
            {prop_checkbox(__scope, "Keep Original Scale", no_normalize, import_cb_bool("no_normalize"))}

            // Re-import button — replaced by a live stage + progress
            // bar while the engine is voxelizing this source.
            div {
                style: "margin-top:4px;",
                if is_importing.get() {
                    {render_import_progress(__scope, progress)}
                } else {
                    {prop_button(__scope, "Re-import", "#2d5a2d", Rc::new(move || {
                        let _ = cmd_tx.get().send(rkp_engine::EngineCommand::ReimportModel {
                            source_path: src.get(),
                        });
                    }))}
                }
            }
        }
    }
}

/// Render the live import progress indicator: spinner + stage
/// message + a filled bar when the stage reports a `done/total`.
/// Falls back to an indeterminate spinner when the stage is running
/// but total isn't known (e.g. during `load_mesh`).
///
/// Every dynamic value is read inside a `{|| ...}` closure so rinch
/// registers a reactive dependency on `progress` — without that wrap
/// the node's text/width freeze on the value present at first render
/// (which for the common "Starting…" case means the user sees the
/// placeholder for the entire import). Rule 2 in the rinch guide.
fn render_import_progress(
    __scope: &mut rinch::core::dom::RenderScope,
    progress: Memo<Option<rkp_engine::snapshot::ImportProgressInfo>>,
) -> rinch::core::dom::NodeHandle {
    let store = use_context::<EditorStore>();
    rsx! {
        div {
            style: "display:flex;flex-direction:column;gap:4px;padding:6px 12px;\
                    background:#2d2d2d;border:1px solid #3c3c3c;border-radius:4px;\
                    color:#bbb;font-size:11px;",
            div {
                style: "display:flex;align-items:center;gap:8px;",
                Loader { r#type: "oval", size: "xs", color: "blue" }
                {|| {
                    // Strip the project-root prefix from any absolute
                    // path the `rkp-import` stage message embeds
                    // (e.g. `Loading mesh: /.../assets/bunny.obj`).
                    // Done at render time because project_dir is a
                    // main-thread signal.
                    let root = store.project_dir.get();
                    progress
                        .get()
                        .as_ref()
                        .map(|p| crate::ui::path_display::relativize_paths_in_text(&p.message, &root))
                        .filter(|m| !m.is_empty())
                        .unwrap_or_else(|| "Starting…".into())
                }}
            }
            div {
                style: {|| {
                    let total = progress.get().as_ref().map(|p| p.total).unwrap_or(0);
                    if total > 0 {
                        "width:100%;height:4px;background:#1e1e1e;\
                         border-radius:2px;overflow:hidden;".to_string()
                    } else {
                        "display:none;".to_string()
                    }
                }},
                div {
                    style: {|| {
                        let (done, total) = progress
                            .get()
                            .as_ref()
                            .map(|p| (p.done, p.total))
                            .unwrap_or((0, 0));
                        let pct = if total > 0 {
                            (done as f64 / total as f64 * 100.0).clamp(0.0, 100.0)
                        } else {
                            0.0
                        };
                        format!(
                            "height:100%;width:{pct:.1}%;background:#4a90e2;\
                             transition:width 120ms ease-out;"
                        )
                    }}
                }
            }
            div {
                style: {|| {
                    let warn_count = progress
                        .get()
                        .as_ref()
                        .map(|p| p.warnings.len())
                        .unwrap_or(0);
                    if warn_count > 0 {
                        "color:#e2a04a;font-size:10px;".to_string()
                    } else {
                        "display:none;".to_string()
                    }
                }},
                {|| {
                    let warn_count = progress
                        .get()
                        .as_ref()
                        .map(|p| p.warnings.len())
                        .unwrap_or(0);
                    if warn_count > 0 {
                        format!(
                            "{warn_count} warning{}",
                            if warn_count == 1 { "" } else { "s" }
                        )
                    } else {
                        String::new()
                    }
                }}
            }
        }
    }
}
