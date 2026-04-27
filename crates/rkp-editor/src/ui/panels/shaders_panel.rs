//! Shaders panel — lists registered user shaders from
//! `<project>/assets/shaders/`.
//!
//! Read-only V1: name, file path, hook coverage, declared param
//! count, and `@animated` flag. Future polish: "open in editor"
//! button, live error surface for parse failures.

use rinch::prelude::*;
use rinch_tabler_icons::{TablerIcon, TablerIconStyle, render_tabler_icon};

use crate::ui::store::EditorStore;

#[component]
pub fn ShadersPanel() -> NodeHandle {
    let store = use_context::<EditorStore>();

    rsx! {
        div {
            style: "display:flex;flex-direction:column;height:100%;overflow-y:auto;",
            for shader in store.user_shaders.get() {
                ShaderItem {
                    key: shader.name.clone(),
                    shader: shader.clone(),
                }
            }
            if store.user_shaders.get().is_empty() {
                div {
                    style: "padding:12px;color:#666;font-size:12px;font-style:italic;",
                    {"No user shaders. Add `assets/shaders/<name>.wgsl` to the project."}
                }
            }
        }
    }
}

#[component]
fn ShaderItem(shader: rkp_render::shader_composer::UserShaderInfo) -> NodeHandle {
    // Visual flags. `has_shade` and `has_generate` come from the parsed
    // hooks; `animated` from the `// @animated` directive. A shader
    // with neither hook is legal but shows a muted "no hooks" tag so
    // the user knows nothing's actually wired up yet.
    let mut hook_tags: Vec<&'static str> = Vec::new();
    if shader.has_shade {
        hook_tags.push("shade");
    }
    if shader.has_generate {
        hook_tags.push("generate");
    }
    if shader.animated {
        hook_tags.push("animated");
    }

    let hooks_label = if hook_tags.is_empty() {
        "no hooks".to_string()
    } else {
        hook_tags.join(" · ")
    };

    let param_label = format!(
        "{} param{}",
        shader.params.len(),
        if shader.params.len() == 1 { "" } else { "s" }
    );

    let path_str = shader.file_path.to_string_lossy().into_owned();

    rsx! {
        div {
            style: "padding:8px 10px;border-bottom:1px solid #2d2d30;\
                    display:flex;flex-direction:column;gap:3px;",
            div {
                style: "display:flex;align-items:center;gap:6px;\
                        font-size:13px;color:#ddd;",
                {render_tabler_icon(__scope, TablerIcon::Code, TablerIconStyle::default())}
                {shader.name.clone()}
            }
            div {
                style: "font-size:10px;color:#888;",
                {hooks_label}
            }
            div {
                style: "font-size:10px;color:#666;",
                {param_label}
            }
            div {
                style: "font-size:10px;color:#555;font-family:monospace;\
                        white-space:nowrap;overflow:hidden;text-overflow:ellipsis;",
                {path_str}
            }
        }
    }
}
