//! Console panel — displays engine log messages with severity filtering.

use rinch::prelude::*;

use crate::CommandSender;
use crate::ui::store::EditorStore;
use rkp_engine::console::LogLevel;

#[component]
pub fn ConsolePanel() -> NodeHandle {
    let store = use_context::<EditorStore>();
    let cmd = use_context::<CommandSender>();

    let show_info = Signal::new(true);
    let show_warn = Signal::new(true);
    let show_error = Signal::new(true);

    let entry_count = Memo::new(move || store.console_entries.get().len());

    rsx! {
        div {
            style: "display:flex;flex-direction:column;height:100%;",

            // Toolbar: filter toggles + count + clear
            div {
                style: "display:flex;align-items:center;padding:4px 8px;\
                        border-bottom:1px solid #333;gap:6px;flex-shrink:0;",

                // Info filter
                {filter_toggle(__scope, "I", "#4fc3f7", show_info)}
                // Warn filter
                {filter_toggle(__scope, "W", "#ffa726", show_warn)}
                // Error filter
                {filter_toggle(__scope, "E", "#ef5350", show_error)}

                // Entry count
                div {
                    style: "flex:1;font-size:10px;color:#666;text-align:right;",
                    {move || format!("{} entries", entry_count.get())}
                }

                // Clear button
                div {
                    style: "font-size:10px;color:#888;cursor:pointer;padding:2px 6px;\
                            border-radius:3px;background:#2d2d2d;border:1px solid #3c3c3c;",
                    onclick: {
                        let cmd = cmd.clone();
                        move || {
                            store.console_entries.set(Vec::new());
                            let _ = cmd.0.send(rkp_engine::EngineCommand::ClearConsole);
                        }
                    },
                    {"Clear"}
                }
            }

            // Log entries — user-select: text enables cross-message text selection + copy.
            div {
                style: "flex:1;overflow-y:auto;font-family:monospace;font-size:11px;\
                        user-select:text;cursor:text;",
                for entry in store.console_entries.get() {
                    {log_entry_row(
                        __scope,
                        &entry,
                        show_info,
                        show_warn,
                        show_error,
                    )}
                }
            }
        }
    }
}

fn filter_toggle(
    __scope: &mut rinch::core::dom::RenderScope,
    label: &'static str,
    color: &'static str,
    active: Signal<bool>,
) -> rinch::core::dom::NodeHandle {
    rsx! {
        div {
            style: {move || {
                if active.get() {
                    format!(
                        "width:20px;height:18px;border-radius:3px;cursor:pointer;\
                         background:{color};color:#fff;font-size:10px;font-weight:700;\
                         display:flex;align-items:center;justify-content:center;"
                    )
                } else {
                    "width:20px;height:18px;border-radius:3px;cursor:pointer;\
                     background:#2d2d2d;color:#666;font-size:10px;font-weight:700;\
                     display:flex;align-items:center;justify-content:center;\
                     border:1px solid #3c3c3c;".into()
                }
            }},
            onclick: move || active.update(|v| *v = !*v),
            {label}
        }
    }
}

fn log_entry_row(
    __scope: &mut rinch::core::dom::RenderScope,
    entry: &rkp_engine::console::LogEntry,
    show_info: Signal<bool>,
    show_warn: Signal<bool>,
    show_error: Signal<bool>,
) -> rinch::core::dom::NodeHandle {
    let store = use_context::<crate::ui::store::EditorStore>();
    // Filter check — use initial values since entry is static once created.
    let level = entry.level;
    let visible = Signal::new(true);
    let timestamp = entry.timestamp;
    let message = entry.message.clone();
    // Stripped-for-display version — recomputes when `project_dir`
    // changes (e.g. opening a different project) without re-running
    // the whole row.
    let display_message = Memo::new({
        let message = message.clone();
        move || crate::ui::path_display::relativize_paths_in_text(
            &message,
            &store.project_dir.get(),
        )
    });

    let (dot_color, text_color) = match level {
        LogLevel::Info => ("#4fc3f7", "#bbb"),
        LogLevel::Warn => ("#ffa726", "#e0c080"),
        LogLevel::Error => ("#ef5350", "#f08080"),
    };

    let ts_min = (timestamp / 60.0) as u32;
    let ts_sec = timestamp % 60.0;
    let ts_str = format!("{ts_min:02}:{ts_sec:05.2}");

    rsx! {
        div {
            style: {move || {
                let show = match level {
                    LogLevel::Info => show_info.get(),
                    LogLevel::Warn => show_warn.get(),
                    LogLevel::Error => show_error.get(),
                };
                if show {
                    format!(
                        "display:flex;align-items:flex-start;gap:6px;padding:2px 8px;\
                         border-bottom:1px solid #2a2a2a;color:{text_color};"
                    )
                } else {
                    "display:none;".into()
                }
            }},

            // Timestamp
            span {
                style: "color:#555;flex-shrink:0;width:56px;",
                {ts_str}
            }

            // Severity dot
            span {
                style: {|| format!(
                    "width:6px;height:6px;border-radius:50%;flex-shrink:0;\
                     margin-top:4px;background:{dot_color};"
                )},
            }

            // Message — strip the project-root prefix from any
            // absolute paths embedded in the log line so users see
            // `assets/bunny.obj` not `/home/joe/.../assets/bunny.obj`.
            span {
                style: "flex:1;white-space:pre-wrap;word-break:break-word;",
                {|| display_message.get()}
            }
        }
    }
}
