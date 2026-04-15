# Viewport + Camera Refactor

**Status:** DESIGNED, NOT STARTED. Deferred behind the current rendering pipeline rewrite.

**Why:** The engine currently has a single ad-hoc `self.camera: CameraState` field. Adding a build viewport would mean a second ad-hoc field. The right shape is a proper `Viewport` abstraction that also cleanly handles: play-mode camera switching, script-driven camera swaps, editor "look through scene camera", multi-viewport rendering, and render layers.

## Goals

- One abstraction that handles main viewport, build viewport, future minimap / PiP / split-screen
- Camera switching at runtime via scripts (`set_active_camera(entity)`)
- Play mode automatically switches to the active scene camera
- Editor camera state persists across play mode / camera overrides
- Render layers (Unity-style 32-bit mask) for editor-only gizmos, HUD, shadow-only proxies, etc.
- Per-viewport visibility gating — hidden viewports don't render

## Core types

```rust
pub struct Viewport {
    pub id: ViewportId,
    pub name: String,                 // "main", "build"

    // Persistent editor camera — survives play mode + overrides.
    pub editor_camera: EditorCamera,

    // Runtime override — takes precedence over editor_camera when set.
    // Play mode, scripted swaps, editor "look through" previews.
    pub runtime_override: Option<CameraSource>,

    pub width: u32,
    pub height: u32,
    pub visible: bool,

    pub filter: SceneFilter,
    pub background: BackgroundMode,   // Scene | Neutral | SolidColor

    pub show_gizmos: bool,            // Gizmos shown when no runtime override

    pub frame_callback: FrameCallback,
}

pub enum EditorCamera {
    Fly(FlyCameraState),              // Main viewport — WASD + mouse
    Turntable(TurntableCameraState),  // Build viewport — orbit around target
}

pub enum CameraSource {
    Entity(hecs::Entity),             // Transform + Camera component
    // Future: Direct(view, proj), NamedCamera(Uuid)
}

pub struct FlyCameraState {
    pub position: Vec3,
    pub yaw: f32, pub pitch: f32,
    pub fov: f32, pub near: f32, pub far: f32,
}

pub struct TurntableCameraState {
    pub target: Vec3,
    pub yaw: f32, pub pitch: f32,
    pub distance: f32,
    pub fov: f32, pub near: f32, pub far: f32,
}

/// Additive filter: entity visible iff
///   (entity.layers & filter.base_layers) != 0  OR  entity == filter.focus_entity
pub struct SceneFilter {
    pub base_layers: u32,
    pub focus_entity: Option<hecs::Entity>,
}

pub enum BackgroundMode {
    Environment,          // Scene sky
    Neutral,              // Studio gray
    SolidColor([f32; 3]),
}
```

## Render layers (32-bit mask)

```rust
// Component
pub struct RenderLayer { pub mask: u32 }  // Default: DEFAULT (bit 0)

// System layers (bits 0–5 reserved)
pub mod layer {
    pub const DEFAULT: u32       = 1 << 0;
    pub const EDITOR_ONLY: u32   = 1 << 1;  // Gizmos, light icons
    pub const UI: u32            = 1 << 2;  // HUD
    pub const BUILD_PREVIEW: u32 = 1 << 3;  // Build viewport helpers
    pub const SHADOW_ONLY: u32   = 1 << 4;  // Invisible, casts shadow
    pub const NO_REFLECTION: u32 = 1 << 5;  // Hidden in reflection captures
}
```

User layers (bits 6–31) named in project config:
```toml
[layers]
Player = 6
Enemies = 7
Particles = 8
```

### Default viewport filters

| Viewport / mode | base_layers | focus_entity |
|---|---|---|
| Main, edit mode | DEFAULT \| EDITOR_ONLY | None |
| Main, play mode | DEFAULT \| UI | None |
| Build viewport | BUILD_PREVIEW | Some(selected_procedural) |

Play mode flips main's mask from `EDITOR_ONLY` → `UI`: editor gizmos vanish, HUD appears.

### GPU implementation

Add `layer_mask: u32` to `RkpGpuObject` and to camera uniforms. Tile-cull and march shaders gate on `(obj.layer_mask & cam.layer_mask) != 0`. Focus-entity is a second check: `obj.object_id == cam.focus_object_id`. Either one passing = visible. One gpu_objects array shared across all viewports.

## Renderer split

- `RkpRenderer` — shared GPU pipelines, scene data, pipeline objects
- `ViewportRenderer` (new) — per-viewport: GBuffer at viewport size, bloom/tonemap state, camera uniform buffer

Each tick, for each visible viewport, `RkpRenderer` runs its passes against the viewport's `ViewportRenderer`.

Memory cost: ~90 MB for a 1920×1080 main + 500×500 build. Acceptable.

## Runtime camera switching (game-engine concerns)

### Play mode
1. On PlayStart: engine finds the scene camera with `Camera.active = true`, sets `viewports[main].runtime_override = Some(Entity(that_camera))`, flips layer mask to `DEFAULT | UI`.
2. On PlayStop: clear override, restore layer mask.
3. Editor camera state is untouched — returning to edit mode lands you where you left off.

### Script-driven camera swaps
```rust
// Host function callable from behaviors
api.set_active_camera(entity);
// → engine sets viewports[main].runtime_override = Some(Entity(entity))
```

Cutscenes: scripts can sequence camera changes or animate between cameras.

Smooth camera transitions (lerp between A and B over T seconds) are a layer on top, not a primitive. Implement as a behavior that updates `runtime_override` or animates a dedicated "transition camera" entity between two targets.

### Editor "Look through camera" preview
Right-click a Camera entity in the outliner → "Look Through". Sets main viewport's `runtime_override` to that entity. "Back to editor camera" button clears it.

Editor-camera WASD continues to move the fly state in the background even while an override is active, so it's always ready when you return.

### Input routing rule
- `play_mode == false`: input → editor camera of the focused viewport
- `play_mode == true`: input → gameplay (scripts), editor cameras frozen

## Commands

All input commands carry a `ViewportId`:

```rust
enum EngineCommand {
    ResizeViewport { id: ViewportId, width: u32, height: u32 },
    SetViewportVisible { id: ViewportId, visible: bool },
    SetViewportCamera { id: ViewportId, source: CameraSource },
    ClearViewportOverride { id: ViewportId },
    SetViewportFilter { id: ViewportId, filter: SceneFilter },

    MouseMoveViewport { id: ViewportId, x: f32, y: f32, dx: f32, dy: f32 },
    MouseButtonViewport { id: ViewportId, button, pressed },
    ScrollViewport { id: ViewportId, delta: f32 },
    PickViewport { id: ViewportId, x: u32, y: u32 },

    // Global
    KeyDown { key }, KeyUp { key },

    // Gameplay — script-driven
    SetActiveSceneCamera { entity: Uuid },
}
```

## Storage

`viewports: HashMap<ViewportId, Viewport>` on `EngineState`. Stable IDs regardless of insertion order. `ViewportId::main()` and `ViewportId::build()` for well-known IDs; user-defined ViewportIds possible for future PiP / minimaps.

## Phase plan

### Phase 1 — Types + scaffolding (no behavior change)
- Define `Viewport`, `ViewportId`, `EditorCamera`, `CameraSource`, `SceneFilter`, `BackgroundMode`
- Add `RenderLayer` component, layer constants module
- `viewports: HashMap<ViewportId, Viewport>` on engine, one entry = current camera
- `self.camera` stays as a view into `viewports[main].editor_camera` temporarily
- All existing behavior preserved

### Phase 2 — Renderer split
- `ViewportRenderer` owns GBuffer + post-process chain
- `RkpRenderer::render_to(viewport_renderer, camera, filter)` entry point
- Add `layer_mask` to `RkpGpuObject` + camera uniforms
- Shaders gate on layer mask (focus_entity too)
- Still one viewport — mask defaults to ALL

### Phase 3 — Per-viewport commands
- Route Resize, SetCamera, mouse events through `ViewportId`
- Editor tags events with `ViewportId::main()`
- `SetViewportFilter`, `SetViewportLayerMask`

### Phase 4 — Multi-viewport rendering + visibility gating
- Iterate visible viewports each tick
- Second `RenderSurface` + frame callback in editor
- Visibility derived from panel layout (`Memo` in store)
- `SetViewportVisible` commands on layout change

### Phase 5 — Runtime overrides + play mode + layers in practice
- `runtime_override` wired up
- Play mode sets main's override + layer mask
- Clear on play stop
- Host function for scripts
- Tag editor gizmos with `EDITOR_ONLY` layer
- Tag HUD entities with `UI` layer

### Phase 6 — Build viewport
- Build viewport with `EditorCamera::Turntable`
- `focus_entity` = selected procedural
- `BUILD_PREVIEW` layer for studio helpers (grid, reference scale)
- `BackgroundMode::Neutral` studio lighting
- Semi-transparent overlay UI for tree, params, add button

### Phase 7 — Editor polish
- Layer panel in project settings (rename user layers)
- Per-entity visibility toggles (shortcuts for layer membership)
- "Look Through" context menu for camera entities
- Auto-frame build viewport on focus change

## Memory / performance notes

- Two viewports ≈ 1920×1080 + 500×500 GBuffer ≈ 90 MB extra
- Visibility gating prevents wasted work when a viewport is tabbed away
- Shared `gpu_objects` upload across viewports — no duplication
- Layer mask is one u32 per object — negligible
- March/tile-cull shader branches on mask — predicted-branch, cheap

## Open questions (deferred)

- Orthographic projection support (`ProjectionMode { Perspective, Ortho }`) — just leave room in the data model
- Smooth camera transitions (lerp) — implement as a behavior on top later
- Camera dollying along splines, cinematic cameras — future behavior layer
- Per-viewport post-process settings (different tonemap per viewport) — likely needed for the build viewport to have studio lighting; add to `ViewportRenderer` when Phase 6 arrives

## When to resume

When the current rendering pipeline rewrite is done. This refactor assumes the renderer has a reasonably stable pass structure; doing it mid-rewrite would just mean doing it twice.
