//! Viewport — owns one camera, one render target, and one slice of the scene.
//!
//! **Status: Phase 1 scaffolding.** Types are defined and `EngineState` carries a
//! `viewports: HashMap<ViewportId, Viewport>` with a single "main" entry, but
//! nothing reads from it yet. The renderer still consults the legacy
//! `EngineState.camera` field. Subsequent phases route resize / input commands
//! and the GBuffer through `Viewport`, then enable per-viewport rendering.
//!
//! See `notes/viewport-camera-refactor.md` for the full migration plan.

use std::collections::HashMap;

use glam::Vec3;

// ── Identity ────────────────────────────────────────────────────────────

/// Stable identifier for a `Viewport`. Two well-known IDs (`MAIN`, `BUILD`)
/// cover the editor's primary surfaces; user-defined IDs (PiP, minimap,
/// reflection capture) get higher numbers via `user(n)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ViewportId(pub u32);

impl ViewportId {
    /// The main editing/play viewport.
    pub const MAIN: Self = Self(0);
    /// The procedural-build viewport (turntable preview).
    pub const BUILD: Self = Self(1);

    /// Allocate a stable ID for a user-defined viewport. `n` is an arbitrary
    /// project-chosen index — only equality matters, not ordering.
    pub const fn user(n: u32) -> Self {
        Self(2 + n)
    }
}

// ── Camera ──────────────────────────────────────────────────────────────

/// Persistent editor-controlled camera for a viewport. Survives play mode
/// and runtime overrides — when those clear, the viewport falls back to
/// this state so the user lands exactly where they left off.
#[derive(Debug, Clone, Copy)]
pub enum EditorCamera {
    /// WASD + mouse-look fly camera. The default for the main viewport.
    Fly(FlyCameraState),
    /// Orbit camera locked to a target. The default for the build viewport.
    Turntable(TurntableCameraState),
}

#[derive(Debug, Clone, Copy)]
pub struct FlyCameraState {
    pub position: Vec3,
    pub yaw: f32,
    pub pitch: f32,
    pub fov: f32,
    pub near: f32,
    pub far: f32,
}

impl Default for FlyCameraState {
    fn default() -> Self {
        Self {
            position: Vec3::new(0.0, 2.0, 5.0),
            yaw: 0.0,
            pitch: 0.0,
            fov: 60.0,
            near: 0.01,
            far: 1000.0,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct TurntableCameraState {
    pub target: Vec3,
    pub yaw: f32,
    pub pitch: f32,
    pub distance: f32,
    pub fov: f32,
    pub near: f32,
    pub far: f32,
}

impl Default for TurntableCameraState {
    fn default() -> Self {
        Self {
            target: Vec3::ZERO,
            yaw: 0.0,
            pitch: -0.3,
            distance: 4.0,
            fov: 50.0,
            near: 0.01,
            far: 1000.0,
        }
    }
}

/// Where a non-editor camera comes from. Used by `Viewport.runtime_override`
/// to swap the active view (play mode, scripted cutscenes, "look through"
/// previews). When `None`, the viewport renders from `editor_camera`.
#[derive(Debug, Clone, Copy)]
pub enum CameraSource {
    /// A scene entity carrying a `Camera` component (Phase 5+).
    Entity(hecs::Entity),
}

// ── Scene filtering ─────────────────────────────────────────────────────

/// Determines which entities a viewport renders. Filtering is **additive**:
/// an entity is visible iff
/// `(entity_layers & base_layers) != 0  OR  entity == focus_entity`.
///
/// This lets the build viewport, for example, restrict to BUILD_PREVIEW-
/// layer helpers while still pulling in whatever procedural the user
/// currently has selected, regardless of that procedural's layer.
#[derive(Debug, Clone, Copy)]
pub struct SceneFilter {
    pub base_layers: u32,
    pub focus_entity: Option<hecs::Entity>,
}

impl Default for SceneFilter {
    fn default() -> Self {
        Self {
            base_layers: layer::DEFAULT | layer::EDITOR_ONLY,
            focus_entity: None,
        }
    }
}

/// Background fill mode for the viewport. `Environment` uses the scene sky
/// stack; `Neutral` is a flat studio gray for the build viewport;
/// `SolidColor` is for tooling.
#[derive(Debug, Clone, Copy)]
pub enum BackgroundMode {
    Environment,
    Neutral,
    SolidColor([f32; 3]),
}

// ── Render layers ───────────────────────────────────────────────────────

/// 32-bit render-layer mask attached to renderable entities. Defaults to
/// `layer::DEFAULT`. Tile-cull and march shaders gate visibility by
/// `(entity.mask & viewport.base_layers) != 0`.
///
/// Components without this attached are treated as `DEFAULT`.
#[derive(Debug, Clone, Copy)]
pub struct RenderLayer {
    pub mask: u32,
}

impl Default for RenderLayer {
    fn default() -> Self {
        Self { mask: layer::DEFAULT }
    }
}

/// Reserved render layer bits. Bits 0–5 are system-reserved; user layers
/// (named in the project config) live in bits 6–31.
pub mod layer {
    /// Standard renderable content.
    pub const DEFAULT: u32 = 1 << 0;
    /// Editor-only gizmos, light icons, debug markers — visible only when
    /// not in play mode.
    pub const EDITOR_ONLY: u32 = 1 << 1;
    /// Game UI / HUD — visible only in play mode.
    pub const UI: u32 = 1 << 2;
    /// Helpers visible only in the build viewport (grid, reference scale).
    pub const BUILD_PREVIEW: u32 = 1 << 3;
    /// Invisible to the camera but still casts shadows.
    pub const SHADOW_ONLY: u32 = 1 << 4;
    /// Hidden from reflection captures.
    pub const NO_REFLECTION: u32 = 1 << 5;
}

// ── Viewport ────────────────────────────────────────────────────────────

/// One displayable view of the scene. Owns a persistent editor camera, an
/// optional runtime-camera override, the filter that decides which entities
/// it sees, and the dimensions of its render target.
///
/// The renderer's per-viewport GBuffer + post-process state lives elsewhere
/// (Phase 2's `ViewportRenderer`); this struct is pure scene-side
/// configuration that the engine owns.
#[derive(Debug, Clone)]
pub struct Viewport {
    pub id: ViewportId,
    /// Human-readable label, useful for telemetry and UI.
    pub name: String,

    /// Persistent editor camera. Untouched by play mode or scripted swaps —
    /// when overrides clear, this is what the viewport falls back to.
    pub editor_camera: EditorCamera,

    /// When `Some`, takes precedence over `editor_camera` for rendering.
    /// Set by play-mode entry, scripted camera swaps, and editor "look
    /// through" previews. WASD continues to drive `editor_camera` in the
    /// background so that clearing the override returns the user to the
    /// same fly position.
    pub runtime_override: Option<CameraSource>,

    pub width: u32,
    pub height: u32,
    /// Hidden viewports skip rendering entirely.
    pub visible: bool,

    pub filter: SceneFilter,
    pub background: BackgroundMode,

    /// Render-pipeline shape — `InSitu` (full PBR stack matching the
    /// main edit viewport) or `Isolation` (neutral background + grid,
    /// no atmosphere/clouds/volumetrics/god-rays/shadow/bloom). Drives
    /// pass gating in `RkpRenderer::render_to`.
    pub mode: rkp_render::RenderMode,

    /// Primary-visibility source for this viewport — voxel octree march
    /// (default, what every viewport except the build one uses) or the
    /// procedural CSG raymarcher (live preview of the tree without a
    /// voxel bake). Only the build viewport should ever flip this to
    /// `Raymarch`; main and any play-mode viewport stay on `Voxel`.
    /// The procedural entity being previewed is pulled from the engine's
    /// current selection at render time — nothing to track here.
    pub preview_mode: rkp_render::BuildPreviewMode,

    /// Whether to overlay editor gizmos. Gated to "no runtime override
    /// active" in higher layers.
    pub show_gizmos: bool,
}

impl Viewport {
    /// Create the default main viewport — fly camera, full visibility,
    /// editor gizmos on, environment background.
    pub fn new_main(width: u32, height: u32) -> Self {
        Self {
            id: ViewportId::MAIN,
            name: "main".to_string(),
            editor_camera: EditorCamera::Fly(FlyCameraState::default()),
            runtime_override: None,
            width,
            height,
            visible: true,
            filter: SceneFilter {
                base_layers: layer::DEFAULT | layer::EDITOR_ONLY,
                focus_entity: None,
            },
            background: BackgroundMode::Environment,
            mode: rkp_render::RenderMode::InSitu,
            preview_mode: rkp_render::BuildPreviewMode::Voxel,
            show_gizmos: true,
        }
    }

    /// Create the default build viewport — turntable camera focused on the
    /// selected procedural, neutral studio background, build-layer helpers
    /// only. Phase 6 wires this up to the editor.
    pub fn new_build(width: u32, height: u32) -> Self {
        Self {
            id: ViewportId::BUILD,
            name: "build".to_string(),
            editor_camera: EditorCamera::Turntable(TurntableCameraState::default()),
            runtime_override: None,
            width,
            height,
            visible: false,
            filter: SceneFilter {
                base_layers: layer::BUILD_PREVIEW,
                focus_entity: None,
            },
            background: BackgroundMode::Neutral,
            mode: rkp_render::RenderMode::Isolation,
            preview_mode: rkp_render::BuildPreviewMode::Voxel,
            show_gizmos: false,
        }
    }
}

// ── Container ───────────────────────────────────────────────────────────

/// Engine-side viewport storage. Wraps `HashMap<ViewportId, Viewport>` so
/// that lookups by well-known ID are explicit and panicky callers fail
/// loudly. Using a HashMap (vs a Vec) keeps IDs stable across insertion /
/// removal — important once user-defined viewports exist.
#[derive(Debug, Clone, Default)]
pub struct Viewports {
    map: HashMap<ViewportId, Viewport>,
}

impl Viewports {
    pub fn new() -> Self {
        Self { map: HashMap::new() }
    }

    /// Insert or replace a viewport. Returns the previous entry, if any.
    pub fn insert(&mut self, viewport: Viewport) -> Option<Viewport> {
        self.map.insert(viewport.id, viewport)
    }

    pub fn remove(&mut self, id: ViewportId) -> Option<Viewport> {
        self.map.remove(&id)
    }

    pub fn get(&self, id: ViewportId) -> Option<&Viewport> {
        self.map.get(&id)
    }

    pub fn get_mut(&mut self, id: ViewportId) -> Option<&mut Viewport> {
        self.map.get_mut(&id)
    }

    /// Convenience for the engine's hot path: the main viewport must always
    /// exist, so panicking is preferable to threading `Option` everywhere.
    pub fn main(&self) -> &Viewport {
        self.map
            .get(&ViewportId::MAIN)
            .expect("main viewport must exist on EngineState")
    }

    pub fn main_mut(&mut self) -> &mut Viewport {
        self.map
            .get_mut(&ViewportId::MAIN)
            .expect("main viewport must exist on EngineState")
    }

    pub fn iter(&self) -> impl Iterator<Item = (&ViewportId, &Viewport)> {
        self.map.iter()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = (&ViewportId, &mut Viewport)> {
        self.map.iter_mut()
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn well_known_ids_are_distinct() {
        assert_ne!(ViewportId::MAIN, ViewportId::BUILD);
        assert_ne!(ViewportId::MAIN, ViewportId::user(0));
        assert_ne!(ViewportId::BUILD, ViewportId::user(0));
    }

    #[test]
    fn user_ids_are_distinct_from_each_other() {
        assert_ne!(ViewportId::user(0), ViewportId::user(1));
    }

    #[test]
    fn main_viewport_lookup_after_insert() {
        let mut viewports = Viewports::new();
        viewports.insert(Viewport::new_main(1920, 1080));
        assert_eq!(viewports.main().id, ViewportId::MAIN);
        assert_eq!(viewports.main().width, 1920);
        assert!(viewports.main().visible);
    }

    #[test]
    #[should_panic(expected = "main viewport must exist")]
    fn main_lookup_panics_when_missing() {
        let viewports = Viewports::new();
        let _ = viewports.main();
    }

    #[test]
    fn build_viewport_defaults_to_hidden() {
        let v = Viewport::new_build(800, 600);
        assert!(!v.visible);
        assert_eq!(v.filter.base_layers, layer::BUILD_PREVIEW);
        assert!(matches!(v.background, BackgroundMode::Neutral));
    }

    #[test]
    fn render_layer_defaults_to_default_bit() {
        let l = RenderLayer::default();
        assert_eq!(l.mask, layer::DEFAULT);
        assert_ne!(l.mask & layer::DEFAULT, 0);
    }

    #[test]
    fn set_viewport_visible_toggles_field() {
        // Mirror what the SetViewportVisible command handler does, on a
        // bare Viewports — protects against the field being renamed and
        // the handler silently no-oping.
        let mut viewports = Viewports::new();
        viewports.insert(Viewport::new_main(800, 600));
        assert!(viewports.main().visible);
        viewports.main_mut().visible = false;
        assert!(!viewports.main().visible);
    }

    #[test]
    fn play_mode_mask_includes_ui_excludes_editor_only() {
        // Locks the convention used by EngineState's
        // enter_play_mode_viewports: play-mode shows HUD, hides editor
        // gizmos. If someone reshuffles the layer bits this fires.
        let play = layer::DEFAULT | layer::UI;
        let edit = layer::DEFAULT | layer::EDITOR_ONLY;
        assert_ne!(play & layer::UI, 0);
        assert_eq!(play & layer::EDITOR_ONLY, 0);
        assert_ne!(edit & layer::EDITOR_ONLY, 0);
        assert_eq!(edit & layer::UI, 0);
    }

    #[test]
    fn runtime_override_clears_to_none() {
        let mut viewports = Viewports::new();
        viewports.insert(Viewport::new_main(800, 600));
        let mut world = hecs::World::new();
        let entity = world.spawn(());
        viewports.main_mut().runtime_override = Some(CameraSource::Entity(entity));
        assert!(viewports.main().runtime_override.is_some());
        viewports.main_mut().runtime_override = None;
        assert!(viewports.main().runtime_override.is_none());
    }

    #[test]
    fn additive_filter_passes_focus_entity_outside_layer_mask() {
        // The contract: `(layers & base) != 0 || entity == focus`.
        // Verify the boolean arithmetic holds for an entity outside the
        // base mask but matching focus_entity.
        let mut world = hecs::World::new();
        let entity = world.spawn(());
        let filter = SceneFilter {
            base_layers: layer::BUILD_PREVIEW,
            focus_entity: Some(entity),
        };
        let entity_layers = layer::DEFAULT;
        let visible = (entity_layers & filter.base_layers) != 0
            || filter.focus_entity == Some(entity);
        assert!(visible);
    }
}
