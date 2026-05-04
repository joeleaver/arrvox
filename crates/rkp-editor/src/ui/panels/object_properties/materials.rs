//! Material-usage section: lists materials used on the entity with voxel counts.

use rinch::prelude::*;

use crate::CommandSender;
use crate::ui::store::EditorStore;
use crate::ui::panels::prop_controls::*;

use super::CmdSignal;

/// Material usage section — which materials are used and how many voxels of each.
#[component]
pub(super) fn MaterialUsageSection() -> NodeHandle {
    let store = use_context::<EditorStore>();
    let cmd_tx: CmdSignal = Signal::new(use_context::<CommandSender>().0);
    let collapsed = Signal::new(false);

    let entity_id = Memo::new(move || {
        store.inspector.get().map(|s| s.entity_id.clone()).unwrap_or_default()
    });

    let usage = Memo::new(move || {
        store.inspector.get()
            .map(|snap| snap.material_usage.clone())
            .unwrap_or_default()
    });

    rsx! {
        if !usage.get().is_empty() {
            div {
                {prop_section_header(__scope, "Materials", collapsed, None)}

                if !collapsed.get() {
                    div {
                        style: "padding:6px 12px;display:flex;flex-direction:column;gap:2px;",
                        for mu in usage.get() {
                            div {
                                key: format!("{}-{}-{}", entity_id.get(), mu.material_id, mu.is_fallback as u8),
                                {material_usage_row(
                                    __scope,
                                    mu.material_id,
                                    mu.voxel_count,
                                    mu.is_fallback,
                                    store,
                                    cmd_tx,
                                )}
                            }
                        }
                    }
                }
            }
        }
    }
}

/// A single material usage row with swatch, name, count, and drag-drop remap.
///
/// When `is_fallback` is true, the entity has no voxel geometry yet —
/// the drop path sends `AssignMaterial` (writes Renderable.material_id)
/// instead of `RemapMaterial` (no-op without voxels), and the voxel
/// count is suppressed.
fn material_usage_row(
    __scope: &mut rinch::core::dom::RenderScope,
    material_id: u16,
    voxel_count: u32,
    is_fallback: bool,
    store: EditorStore,
    cmd_tx: CmdSignal,
) -> rinch::core::dom::NodeHandle {
    let mat_name = Signal::new({
        store.materials.get()
            .iter()
            .find(|m| m.id == material_id)
            .map(|m| m.name.clone())
            .unwrap_or_else(|| format!("Material {material_id}"))
    });
    let mat_color = Signal::new({
        store.materials.get()
            .iter()
            .find(|m| m.id == material_id)
            .map(|m| [m.albedo[0], m.albedo[1], m.albedo[2], 1.0])
            .unwrap_or([0.5, 0.5, 0.5, 1.0])
    });

    let is_drop_target = Signal::new(false);
    let count_str = format_voxel_count(voxel_count);

    rsx! {
        div {
            style: {move || {
                if is_drop_target.get() {
                    "display:flex;align-items:center;gap:6px;padding:3px 4px;\
                     border-radius:3px;border:1px dashed #4fc3f7;background:#1a2a3a;"
                } else {
                    "display:flex;align-items:center;gap:6px;padding:3px 4px;\
                     border-radius:3px;border:1px solid transparent;"
                }
            }},
            ondragenter: move || {
                if store.material_drag.get().is_some() {
                    is_drop_target.set(true);
                }
            },
            ondragleave: move || {
                is_drop_target.set(false);
            },
            ondrop: move || {
                is_drop_target.set(false);
                if let Some(new_mat_id) = store.material_drag.get() {
                    if let Some(snap) = store.inspector.get() {
                        if let Ok(eid) = uuid::Uuid::parse_str(&snap.entity_id) {
                            let cmd = if is_fallback {
                                rkp_engine::EngineCommand::AssignMaterial {
                                    entity_id: eid,
                                    material_id: new_mat_id,
                                }
                            } else {
                                rkp_engine::EngineCommand::RemapMaterial {
                                    object_id: eid,
                                    from_material: material_id,
                                    to_material: new_mat_id,
                                }
                            };
                            let _ = cmd_tx.get().send(cmd);
                        }
                    }
                    store.material_drag.set(None);
                }
            },

            // Color swatch
            div {
                style: {move || {
                    let [r, g, b, _] = mat_color.get();
                    format!(
                        "width:14px;height:14px;border-radius:3px;flex-shrink:0;\
                         border:1px solid #3c3c3c;\
                         background:rgb({},{},{});",
                        (r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8,
                    )
                }},
            }

            // Material name
            div {
                style: "flex:1;font-size:11px;color:#ccc;\
                        overflow:hidden;text-overflow:ellipsis;white-space:nowrap;",
                {move || mat_name.get()}
            }

            // Voxel count (suppressed for fallback rows — no voxels to count).
            if !is_fallback {
                div {
                    style: "font-size:10px;color:#666;flex-shrink:0;font-family:monospace;",
                    {count_str.clone()}
                }
            }
        }
    }
}

fn format_voxel_count(count: u32) -> String {
    if count >= 1_000_000 {
        format!("{:.1}M", count as f64 / 1_000_000.0)
    } else if count >= 1_000 {
        format!("{:.1}K", count as f64 / 1_000.0)
    } else {
        format!("{count}")
    }
}
