//! Viewport component — renders the engine's output via RenderSurface.

use rinch::prelude::*;
use rinch::render_surface::RenderSurface;

#[component]
pub fn Viewport() -> NodeHandle {
    let surface = use_context::<RenderSurfaceHandle>();
    rsx! {
        RenderSurface { surface: Some(surface.clone()) }
    }
}
