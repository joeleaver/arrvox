//! Public data types for the user-shader registry.
//!
//! No logic — just the structs and enums that move between the parser,
//! the composer, the engine, and the editor:
//!
//! - [`ParamDef`] — one `@param` schema entry.
//! - [`ShaderMetadata`] — header `@`-directives, attached to each entry.
//! - [`UserShaderEntry`] — a single shader's parsed hooks + helpers +
//!   structs + parsed instance layout.
//! - [`UserShaderRegistry`] — the scanned-from-disk collection.
//! - [`UserShaderInfo`] — editor-facing snapshot (no captured WGSL bodies).
//! - [`ShaderComposerError`] — io / parse error sum.
//! - [`ComposedChunks`] — the per-pipeline output of `compose`.

use std::path::PathBuf;

use crate::instance_proto::InstanceLayout;

/// One user-declared parameter: name, default, optional UI range. Built
/// from `// @param <name>: <type> = <default>, range = [<lo>, <hi>]`
/// header comments in the shader source.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ParamDef {
    pub name: String,
    pub default: f32,
    pub range: Option<(f32, f32)>,
}

/// Per-shader metadata extracted from `// @<key> ...` comments at the top
/// of the source file (anything before the first `fn`). All fields are
/// optional with sensible defaults; missing headers don't error.
///
/// Editor controls are built from `params`; the `animated`,
/// `region_thickness`, and `cell_size` flags steer the geometry pass's
/// region-collection and cache logic in Phase C.
#[derive(Debug, Clone, Default)]
pub struct ShaderMetadata {
    /// Named parameter schema — order is the order they appear in the
    /// source. Materials store values keyed by name; the GPU param
    /// buffer packs them in this order.
    pub params: Vec<ParamDef>,
    /// How far from the host surface the geometry hook may emit voxels,
    /// in world units. Drives the bounding region used to size the
    /// per-(object, material) generation dispatch in Phase C.
    /// Default 0.0 — pure shade-pass shaders don't need it.
    pub region_thickness: f32,
    /// Opt-in: regenerate every frame instead of caching. For waving
    /// grass / fluttering hair / etc. Default false (cache by source +
    /// param hash).
    pub animated: bool,
    /// Preferred voxel resolution for the geometry pass. `None` falls
    /// back to a per-object default (e.g. host's voxel size). The
    /// region's max octree depth is derived from this — sim solves
    /// `extent / (cell_size * 4) = 2^depth` and clamps against
    /// `max_depth` below.
    pub cell_size: Option<f32>,
    /// Cap on the V9 sparse-BFS octree depth. `None` falls back to
    /// the engine default. Hard ceiling is `MAX_DEPTH = 8` (mirrors
    /// the WGSL constant + queue buffer sizing); requests above that
    /// silently clamp.
    pub max_depth: Option<u32>,
    /// V10 multi-region tiling. When set, the engine's auto-scan
    /// splits the painted area into `tile_size`-edge cubes (in
    /// host-local space) and emits one region per tile that contains
    /// any painted leaves. Each tile's cube is fixed-extent, so cell
    /// size is independent of paint area — `cell_size = tile_size /
    /// (4 × 2^max_depth)` regardless of how big a patch the user
    /// paints. None falls back to V9 single-region behaviour
    /// (one region per (object, material) covering the painted-leaf
    /// AABB; cell size grows with paint extent).
    pub tile_size: Option<f32>,
    /// `@instance_proto <StructName>` — opt-in for the per-instance
    /// pipeline (Phase B-redux band-cell descent). When `Some`, the
    /// file MUST also contain the named struct declaration plus the
    /// `user_<stem>_proto` hook; the parsed struct layout lives on the
    /// [`UserShaderEntry`]. None means the shader uses the existing
    /// per-cell `generate` pipeline.
    pub instance_proto_struct: Option<String>,
    /// `@max_emits_per_thread <u32>` — per-host-position cap on how
    /// many instances `instance_at` may return for a single host hit
    /// before the dispatcher gives up. Uses 1 when absent. Hard
    /// ceiling: 16.
    pub max_emits_per_thread: Option<u32>,
}

/// One user shader's parsed hook bodies + header metadata. Each `*_text`
/// field, when `Some`, is the full `fn ... { ... }` declaration as it
/// appeared in the source file (the function name is rewritten to the
/// dispatch form at emit time, not at capture).
///
/// Two hooks are recognized:
///   * `shade(ctx)` — Phase B; called per-pixel from the deferred shade
///     pass to override or augment PBR.
///   * `generate(cell_world_pos, host, ctx)` — Phase C; called from the
///     GPU geometry pipeline to emit voxels into the sidecar pool.
///
/// A shader may declare either or both hooks (or neither — empty is
/// legal, just contributes nothing). Helper functions (anything not
/// matching `user_<stem>_<hook>`) are captured into `helpers` and
/// emitted alongside the hook bodies so user code can call them.
#[derive(Debug, Clone)]
pub struct UserShaderEntry {
    /// File stem, used both as the on-disk shader name (what materials
    /// reference via `MaterialDef.shader`) and as the prefix the
    /// parser scans for in the source.
    pub name: String,
    /// Path the entry was loaded from. Stored so the editor can offer
    /// "open in external editor" and error messages include source
    /// locations.
    pub file_path: PathBuf,
    /// Numeric dispatch id, 1-based. Registry assigns these in scan
    /// order; resolved by `MaterialDef::to_gpu` into
    /// `GpuMaterial.shader_id`.
    pub id: u32,
    /// Header-comment metadata.
    pub metadata: ShaderMetadata,
    /// Captured fn declarations.
    pub shade_text: Option<String>,
    pub generate_text: Option<String>,
    /// User-defined helper functions (not hooks). Captured verbatim
    /// so hook bodies can call them. Identifier collisions across
    /// shaders are user-managed — pick unique helper names if
    /// loading multiple shaders together.
    pub helpers: Vec<String>,
    /// `fn user_<stem>_proto(uvw: vec3<f32>) -> VoxelEmit` — the
    /// prototype shape descended at march time from band-cell hits.
    /// Required when `metadata.instance_proto_struct` is `Some`.
    pub proto_text: Option<String>,
    /// `fn user_<stem>_inst_aabb(inst: <Struct>) -> Aabb` — instance
    /// world-space AABB. Required alongside `instance_at`.
    pub inst_aabb_text: Option<String>,
    /// `fn user_<stem>_inst_to_local(world_pos: vec3<f32>, inst: <Struct>) -> vec3<f32>`
    /// — world→prototype-local mapping. Required alongside `instance_at`.
    pub inst_to_local_text: Option<String>,
    /// Phase B-redux march-time derivation hook. Signature:
    /// `fn user_<stem>_instance_at(host_pos: vec3<f32>, host: HostSample,
    /// ctx: UserCtx, k: u32, out_instance: ptr<function, <Struct>>) -> bool`.
    /// Returns the k-th instance for this host position, or `false` to
    /// signal "no instance at index k." Called from the host march on
    /// band-cell hits; allows zero per-frame state writes.
    pub instance_at_text: Option<String>,
    /// Verbatim `struct ... { ... }` declarations captured from the
    /// file's top level, in source order. Shader code can declare its
    /// own helper structs; the engine splices them all back into the
    /// generated WGSL alongside the hooks. The instance struct (if any)
    /// is one of these.
    pub struct_decls: Vec<String>,
    /// Parsed layout of the per-instance state struct named by
    /// `metadata.instance_proto_struct`. Populated alongside the entry
    /// when the shader opts into Option B.
    pub instance_layout: Option<InstanceLayout>,
}

impl UserShaderEntry {
    /// Whether this shader contributes any dispatchable hook. Shaders
    /// with neither hook are legal (the file might just be header-only
    /// for now) but the dispatcher won't call into them.
    #[allow(dead_code)] // sanity check called by the parser; may grow callers
    pub(super) fn has_any_hook(&self) -> bool {
        self.shade_text.is_some()
            || self.generate_text.is_some()
            || self.proto_text.is_some()
            || self.instance_at_text.is_some()
    }
}

/// Registry of all user shaders discovered in the project's
/// `assets/shaders/` directory. Built once per scan; consumers hold
/// references for the lifetime of the bake worker's current shader
/// generation, then a new registry replaces it on filesystem change.
#[derive(Debug, Clone, Default)]
pub struct UserShaderRegistry {
    pub(super) entries: Vec<UserShaderEntry>,
    /// Stable hash of the concatenation of every entry's source text
    /// in deterministic (alphabetical) order. Bake outputs use this
    /// in their cache key so editing a `.wgsl` invalidates only
    /// dependent caches; callers compare hashes to skip no-op reloads.
    pub(super) source_hash: u64,
}

impl UserShaderRegistry {
    /// An empty registry — equivalent to "no user shaders." Bake/dispatch
    /// behave as identity for every `shader_id`.
    pub fn empty() -> Self {
        Self::default()
    }

    pub fn entries(&self) -> &[UserShaderEntry] {
        &self.entries
    }

    pub fn source_hash(&self) -> u64 {
        self.source_hash
    }

    /// Names of all registered shaders, in id order.
    pub fn names(&self) -> Vec<String> {
        self.entries.iter().map(|e| e.name.clone()).collect()
    }

    /// Build a snapshot-friendly view: just the metadata the editor
    /// needs (shader name, file path, param schema, flags). Excludes
    /// the captured fn bodies, which the editor never needs and would
    /// just inflate every snapshot.
    pub fn shader_infos(&self) -> Vec<UserShaderInfo> {
        self.entries
            .iter()
            .map(|e| UserShaderInfo {
                name: e.name.clone(),
                file_path: e.file_path.clone(),
                params: e.metadata.params.clone(),
                region_thickness: e.metadata.region_thickness,
                animated: e.metadata.animated,
                cell_size: e.metadata.cell_size,
                max_depth: e.metadata.max_depth,
                tile_size: e.metadata.tile_size,
                has_shade: e.shade_text.is_some(),
                has_generate: e.generate_text.is_some(),
                has_instance_at: e.instance_at_text.is_some(),
                instance_struct_name: e
                    .metadata
                    .instance_proto_struct
                    .clone(),
                instance_struct_size: e
                    .instance_layout
                    .as_ref()
                    .map(|l| l.total_size),
                max_emits_per_thread: e.metadata.max_emits_per_thread,
            })
            .collect()
    }

    /// Resolve a `shader_name` (as stored on `MaterialDef.shader`) to
    /// the numeric dispatch id. `None` means "not registered" — material
    /// falls back to id=0 (identity), which is the default-arm of every
    /// dispatch switch.
    pub fn resolve(&self, name: &str) -> Option<u32> {
        if name.is_empty() {
            return None;
        }
        self.entries
            .iter()
            .find(|e| e.name == name)
            .map(|e| e.id)
    }
}

/// Editor-facing snapshot of one registered shader. The shader_infos
/// endpoint produces these so the editor's material panel can build a
/// shader dropdown + dynamic param controls without hauling around the
/// captured WGSL bodies.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct UserShaderInfo {
    pub name: String,
    pub file_path: PathBuf,
    pub params: Vec<ParamDef>,
    pub region_thickness: f32,
    pub animated: bool,
    pub cell_size: Option<f32>,
    pub max_depth: Option<u32>,
    pub tile_size: Option<f32>,
    pub has_shade: bool,
    pub has_generate: bool,
    /// Phase B-redux — true if the shader provides a `instance_at`
    /// hook. Such shaders take the march-time derivation path
    /// (Phase 3a host-leaf dispatch + Phase 3b band-cell dispatch)
    /// instead of Option B's emit-into-instance-pool flow.
    pub has_instance_at: bool,
    /// Name of the per-instance struct (from `@instance_proto`) if any.
    pub instance_struct_name: Option<String>,
    /// Byte size of the per-instance struct, if parsed. Helpful for
    /// editor visibility into "am I close to the soft/hard limit?"
    pub instance_struct_size: Option<u32>,
    /// Phase 7b — per-thread emit cap. `None` falls back to 1.
    pub max_emits_per_thread: Option<u32>,
}

/// Errors that can arise while scanning / parsing user shaders.
#[derive(Debug, Clone, thiserror::Error)]
pub enum ShaderComposerError {
    #[error("io error reading {path}: {msg}")]
    Io { path: PathBuf, msg: String },

    #[error("parse error in {path}:{line}: {msg}")]
    Parse {
        path: PathBuf,
        line: usize,
        msg: String,
    },
}

/// Output of [`crate::shader_composer::compose`] — one chunk per pipeline
/// that consumes user shaders. Each chunk is self-contained: rewritten
/// user fn bodies for that pipeline's hook, followed by a
/// `dispatch_user_<hook>` switch statement with an identity default arm.
/// Pipelines splice the matching chunk into their own WGSL between
/// begin/end markers.
///
/// Both chunks share the same user-shader names + ids, so a single
/// material's `shader_id` correctly routes through both pipelines.
#[derive(Debug, Clone, Default)]
pub struct ComposedChunks {
    /// Spliced into `rkp_shade.wgsl` by the deferred shade pass.
    /// Defines `dispatch_user_shade(shader_id, ctx) -> ShadeResult`.
    pub shade: String,
    /// Spliced into the geometry-build compute shader. Defines
    /// `dispatch_user_generate(shader_id, cell_world_pos, host, ctx)
    /// -> VoxelEmit`.
    pub generate: String,
    /// Spliced into the prototype-bake compute shader. Defines
    /// `dispatch_user_proto(shader_id, uvw) -> VoxelEmit`. Routes
    /// only shaders with `@instance_proto` directives; identity
    /// default arm returns a skip emit.
    pub proto: String,
    /// Phase B-redux — march-time derivation chunk. Spliced into the
    /// host march / shadow-trace templates between the
    /// `USER_INSTANCE_AT_DISPATCH_*` markers. Defines per-shader
    /// `rkp_user_<id>_instance_at(host_pos, host, ctx, k, &instance)
    /// -> bool` (verbatim user body, fn name rewritten). The march
    /// splices per-shader switch cases that call into these directly
    /// with the user's instance-struct-typed local var; no unified
    /// dispatcher because each shader's instance struct differs.
    /// Empty when no instance shaders register an `instance_at` hook.
    pub instance_at: String,
}
