//! Instance prototype struct layout for Option B (voxel sprite instancing).
//!
//! User shaders that opt into the instance pipeline declare a per-instance
//! state struct in WGSL alongside their `proto`/`emit` hooks. This module
//! parses that struct, computes its byte layout under WGSL storage-buffer
//! rules, and detects engine-recognized tagged fields (`pos`, `rot`,
//! `scale`).
//!
//! The parser is intentionally narrow — it accepts only the scalar / vector
//! types the GPU instance pipeline knows how to read. Anything richer
//! (matrices, nested structs, arrays) rejects with a clear error rather
//! than silently producing a layout the dispatch code can't honor.

use std::path::{Path, PathBuf};

/// Soft warning threshold — instances larger than this start eroding the
/// "10× memory win" Option B promises over per-cell storage. Not an
/// error, just emitted into `InstanceLayout::warnings` so the editor /
/// CLI can surface it.
pub const INSTANCE_SOFT_LIMIT_BYTES: u32 = 32;

/// Hard cap — bigger than this and the buffer math (per-region instance
/// reservations, atomic counters) gets uncomfortable. Reject so the shader
/// author sees the wall before runtime.
pub const INSTANCE_HARD_LIMIT_BYTES: u32 = 64;

/// Scalar / vector types the instance struct may contain. WGSL has more,
/// but the GPU dispatch path only knows how to read these — adding a new
/// type means teaching codegen + the march to handle it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WgslType {
    F32,
    U32,
    I32,
    Vec2F32,
    Vec3F32,
    Vec4F32,
}

impl WgslType {
    /// Bytes the value occupies (NOT including any trailing padding the
    /// struct alignment rules require — that's separate, see
    /// [`Self::alignment`]).
    pub fn size(self) -> u32 {
        match self {
            Self::F32 | Self::U32 | Self::I32 => 4,
            Self::Vec2F32 => 8,
            Self::Vec3F32 => 12, // size 12, alignment 16 — the WGSL gotcha
            Self::Vec4F32 => 16,
        }
    }

    /// Required alignment for this type when placed in a storage buffer.
    /// `vec3<f32>` is the load-bearing one — size 12, alignment 16, so
    /// any field after a `vec3<f32>` lands at offset 16 unless it's
    /// also 4-byte-aligned and packs into the trailing 4 bytes.
    pub fn alignment(self) -> u32 {
        match self {
            Self::F32 | Self::U32 | Self::I32 => 4,
            Self::Vec2F32 => 8,
            Self::Vec3F32 | Self::Vec4F32 => 16,
        }
    }

    fn parse(s: &str) -> Option<Self> {
        // Tolerate whitespace inside angle brackets: `vec3 < f32 >`.
        let s: String = s.chars().filter(|c| !c.is_whitespace()).collect();
        match s.as_str() {
            "f32" => Some(Self::F32),
            "u32" => Some(Self::U32),
            "i32" => Some(Self::I32),
            "vec2<f32>" => Some(Self::Vec2F32),
            "vec3<f32>" => Some(Self::Vec3F32),
            "vec4<f32>" => Some(Self::Vec4F32),
            _ => None,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::U32 => "u32",
            Self::I32 => "i32",
            Self::Vec2F32 => "vec2<f32>",
            Self::Vec3F32 => "vec3<f32>",
            Self::Vec4F32 => "vec4<f32>",
        }
    }
}

/// A single member of the instance struct. `byte_offset` is the
/// post-alignment offset within the struct, in bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InstanceField {
    pub name: String,
    pub ty: WgslType,
    pub byte_offset: u32,
}

/// Whether the instance struct declared a scale field, and which form.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScaleKind {
    /// No `scale` field — engine uses identity scale.
    None,
    /// `scale: f32` — uniform across all axes.
    Uniform,
    /// `scale: vec3<f32>` — per-axis (non-uniform) scale.
    PerAxis,
}

/// Parsed layout of a user-declared per-instance struct, ready for the
/// GPU dispatch + march to consume.
#[derive(Debug, Clone)]
pub struct InstanceLayout {
    /// User-supplied struct name (the identifier after the `struct`
    /// keyword). Echoed back verbatim so codegen can reference it.
    pub struct_name: String,
    /// Captured `struct ... { ... }` text from the user's shader source,
    /// to be spliced into the generated WGSL so the GPU side sees the
    /// same struct definition the user wrote.
    pub struct_text: String,
    /// Members in source order, with computed offsets.
    pub fields: Vec<InstanceField>,
    /// Total struct size in bytes, rounded up to the struct's alignment.
    pub total_size: u32,
    /// Struct alignment (max of any member's alignment).
    pub alignment: u32,
    /// Byte offset of the required `pos: vec3<f32>` field.
    pub pos_offset: u32,
    /// Byte offset of the optional `rot: vec4<f32>` field, if present.
    /// Treated as a quaternion (xyzw) by the dispatch code.
    pub rot_offset: Option<u32>,
    /// Whether the optional `scale` field is present and which form
    /// it took. Offset stored separately as [`Self::scale_offset`].
    pub scale_kind: ScaleKind,
    /// Byte offset of `scale`, if present.
    pub scale_offset: Option<u32>,
    /// Non-fatal advisories — currently just "soft size limit exceeded."
    /// Surface to the user via the editor / CLI.
    pub warnings: Vec<String>,
}

impl InstanceLayout {
    /// True if any non-identity scaling is in play (uniform OR per-axis).
    /// The march uses this to decide whether to apply a scale transform.
    pub fn has_scale(&self) -> bool {
        !matches!(self.scale_kind, ScaleKind::None)
    }
}

/// Errors that can arise while parsing the per-instance struct. Mirrors
/// `shader_composer::ShaderComposerError::Parse` shape so callers can
/// fold them together.
#[derive(Debug, Clone, thiserror::Error)]
#[error("parse error in {path:?}: {msg}")]
pub struct InstanceParseError {
    pub path: PathBuf,
    pub msg: String,
}

/// Parse a captured `struct <Name> { … }` block into an [`InstanceLayout`].
///
/// `path` is only used for error context. `expected_name` is the struct
/// name that the `@instance_proto` directive declared; the function
/// confirms the captured text actually defines that struct (catches
/// "directive declares Foo, file defines Bar" typos).
pub fn parse_instance_layout(
    path: &Path,
    expected_name: &str,
    struct_text: &str,
) -> Result<InstanceLayout, InstanceParseError> {
    let err = |msg: &str| InstanceParseError {
        path: path.to_path_buf(),
        msg: msg.to_string(),
    };

    // 1. Confirm `struct <expected_name>` heads the captured text.
    let trimmed = struct_text.trim_start();
    let after_struct = trimmed.strip_prefix("struct").ok_or_else(|| {
        err(&format!(
            "@instance_proto target `{expected_name}` does not start with `struct`"
        ))
    })?;
    let after_struct = after_struct.trim_start();
    let name_end = after_struct
        .find(|c: char| !c.is_ascii_alphanumeric() && c != '_')
        .unwrap_or(after_struct.len());
    let actual_name = &after_struct[..name_end];
    if actual_name != expected_name {
        return Err(err(&format!(
            "@instance_proto declared struct `{expected_name}` but the captured definition is `{actual_name}`"
        )));
    }

    // 2. Extract the body between `{` and the matching `}`.
    let body_open = struct_text
        .find('{')
        .ok_or_else(|| err(&format!("struct `{expected_name}` has no body")))?;
    let body_close = struct_text
        .rfind('}')
        .ok_or_else(|| err(&format!("struct `{expected_name}` has no closing brace")))?;
    if body_close <= body_open {
        return Err(err(&format!("struct `{expected_name}` has malformed body")));
    }
    let body = &struct_text[body_open + 1..body_close];

    // 3. Split body on commas (WGSL allows trailing comma) and parse each
    //    `<name>: <type>` pair. Comments would already have been stripped
    //    by the upstream brace matcher in the composer's main parser, but
    //    be defensive — strip line comments here too.
    let body_clean = strip_line_comments(body);

    let mut fields: Vec<InstanceField> = Vec::new();
    let mut offset: u32 = 0;
    let mut struct_align: u32 = 1;
    let mut pos_offset: Option<u32> = None;
    let mut rot_offset: Option<u32> = None;
    let mut scale_kind = ScaleKind::None;
    let mut scale_offset: Option<u32> = None;

    for raw_part in body_clean.split(',') {
        let part = raw_part.trim();
        if part.is_empty() {
            continue;
        }
        // Reject WGSL field attributes (`@align(16) pad: f32`) — we
        // re-derive layout ourselves and user-supplied attributes would
        // skew the offsets the GPU dispatch assumes. The attribute lives
        // BEFORE the name, so check the whole part, not just ty_text.
        if part.starts_with('@') {
            return Err(err(&format!(
                "field `{part}` carries a WGSL attribute; @instance_proto structs must use plain types so engine layout matches"
            )));
        }
        let (name, ty_text) = part.split_once(':').ok_or_else(|| {
            err(&format!(
                "field `{part}` in `{expected_name}` is missing `:` between name and type"
            ))
        })?;
        let name = name.trim().to_string();
        if name.is_empty() {
            return Err(err(&format!(
                "empty field name in struct `{expected_name}`"
            )));
        }
        let ty_text = ty_text.trim();
        let ty = WgslType::parse(ty_text).ok_or_else(|| {
            err(&format!(
                "field `{name}` in `{expected_name}` has unsupported type `{ty_text}` (allowed: f32, u32, i32, vec2<f32>, vec3<f32>, vec4<f32>)"
            ))
        })?;

        let align = ty.alignment();
        offset = round_up(offset, align);
        let field_offset = offset;
        offset += ty.size();
        struct_align = struct_align.max(align);

        // Tagged-field detection by name + type.
        match (name.as_str(), ty) {
            ("pos", WgslType::Vec3F32) => {
                if pos_offset.is_some() {
                    return Err(err(&format!(
                        "struct `{expected_name}` has duplicate `pos` field"
                    )));
                }
                pos_offset = Some(field_offset);
            }
            ("pos", other) => {
                return Err(err(&format!(
                    "field `pos` must be vec3<f32>, got {}",
                    other.name()
                )));
            }
            ("rot", WgslType::Vec4F32) => {
                rot_offset = Some(field_offset);
            }
            ("rot", other) => {
                return Err(err(&format!(
                    "field `rot` must be vec4<f32> (xyzw quaternion), got {}",
                    other.name()
                )));
            }
            ("scale", WgslType::F32) => {
                scale_kind = ScaleKind::Uniform;
                scale_offset = Some(field_offset);
            }
            ("scale", WgslType::Vec3F32) => {
                scale_kind = ScaleKind::PerAxis;
                scale_offset = Some(field_offset);
            }
            ("scale", other) => {
                return Err(err(&format!(
                    "field `scale` must be f32 (uniform) or vec3<f32> (per-axis), got {}",
                    other.name()
                )));
            }
            _ => {}
        }

        fields.push(InstanceField {
            name,
            ty,
            byte_offset: field_offset,
        });
    }

    let pos_offset = pos_offset.ok_or_else(|| {
        err(&format!(
            "struct `{expected_name}` is missing required field `pos: vec3<f32>`"
        ))
    })?;

    let total_size = round_up(offset, struct_align);

    if total_size > INSTANCE_HARD_LIMIT_BYTES {
        return Err(err(&format!(
            "instance struct `{expected_name}` is {total_size} bytes — exceeds hard cap of {INSTANCE_HARD_LIMIT_BYTES} bytes"
        )));
    }

    let mut warnings = Vec::new();
    if total_size > INSTANCE_SOFT_LIMIT_BYTES {
        warnings.push(format!(
            "instance struct `{expected_name}` is {total_size} bytes — above soft limit of {INSTANCE_SOFT_LIMIT_BYTES} bytes; memory wins of Option B start to erode"
        ));
    }

    Ok(InstanceLayout {
        struct_name: expected_name.to_string(),
        struct_text: struct_text.to_string(),
        fields,
        total_size,
        alignment: struct_align,
        pos_offset,
        rot_offset,
        scale_kind,
        scale_offset,
        warnings,
    })
}

fn round_up(value: u32, align: u32) -> u32 {
    debug_assert!(align.is_power_of_two());
    (value + align - 1) & !(align - 1)
}

fn strip_line_comments(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for line in s.lines() {
        let cut = line.find("//").unwrap_or(line.len());
        out.push_str(&line[..cut]);
        out.push('\n');
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    fn p() -> &'static Path {
        Path::new("test.wgsl")
    }

    #[test]
    fn parses_minimal_struct_with_pos_only() {
        let src = "struct Pt { pos: vec3<f32> }";
        let layout = parse_instance_layout(p(), "Pt", src).unwrap();
        assert_eq!(layout.struct_name, "Pt");
        assert_eq!(layout.fields.len(), 1);
        assert_eq!(layout.pos_offset, 0);
        assert_eq!(layout.alignment, 16);
        assert_eq!(layout.total_size, 16); // rounded up to alignment
        assert_eq!(layout.rot_offset, None);
        assert_eq!(layout.scale_kind, ScaleKind::None);
        assert!(layout.warnings.is_empty());
    }

    #[test]
    fn detects_uniform_vs_per_axis_scale() {
        let uni = "struct A { pos: vec3<f32>, scale: f32 }";
        let l = parse_instance_layout(p(), "A", uni).unwrap();
        assert_eq!(l.scale_kind, ScaleKind::Uniform);
        assert!(l.has_scale());

        let per = "struct B { pos: vec3<f32>, scale: vec3<f32> }";
        let l = parse_instance_layout(p(), "B", per).unwrap();
        assert_eq!(l.scale_kind, ScaleKind::PerAxis);
        assert!(l.has_scale());
    }

    #[test]
    fn detects_rot_quaternion() {
        let src = "struct R { pos: vec3<f32>, rot: vec4<f32> }";
        let l = parse_instance_layout(p(), "R", src).unwrap();
        assert_eq!(l.rot_offset, Some(16)); // pos at 0..12, padded to 16 by vec4 alignment
    }

    #[test]
    fn rejects_pos_with_wrong_type() {
        let src = "struct X { pos: f32 }";
        let err = parse_instance_layout(p(), "X", src).unwrap_err();
        assert!(err.msg.contains("`pos` must be vec3<f32>"), "got: {}", err.msg);
    }

    #[test]
    fn rejects_missing_pos() {
        let src = "struct X { foo: f32 }";
        let err = parse_instance_layout(p(), "X", src).unwrap_err();
        assert!(err.msg.contains("missing required field"), "got: {}", err.msg);
        assert!(err.msg.contains("pos"), "got: {}", err.msg);
    }

    #[test]
    fn rejects_rot_with_wrong_type() {
        let src = "struct X { pos: vec3<f32>, rot: vec3<f32> }";
        let err = parse_instance_layout(p(), "X", src).unwrap_err();
        assert!(err.msg.contains("`rot` must be vec4<f32>"), "got: {}", err.msg);
    }

    #[test]
    fn rejects_scale_with_wrong_type() {
        let src = "struct X { pos: vec3<f32>, scale: vec4<f32> }";
        let err = parse_instance_layout(p(), "X", src).unwrap_err();
        assert!(err.msg.contains("`scale` must be"), "got: {}", err.msg);
    }

    #[test]
    fn rejects_unsupported_type() {
        let src = "struct X { pos: vec3<f32>, mat: mat4x4<f32> }";
        let err = parse_instance_layout(p(), "X", src).unwrap_err();
        assert!(err.msg.contains("unsupported type"), "got: {}", err.msg);
    }

    #[test]
    fn rejects_user_attribute_on_field() {
        // User-supplied @align would lie to the engine about the offset
        // it computed — reject so the GPU layout matches our math.
        let src = "struct X { pos: vec3<f32>, @align(16) pad: f32 }";
        let err = parse_instance_layout(p(), "X", src).unwrap_err();
        assert!(err.msg.contains("attribute"), "got: {}", err.msg);
    }

    #[test]
    fn rejects_struct_name_mismatch() {
        let src = "struct Foo { pos: vec3<f32> }";
        let err = parse_instance_layout(p(), "Bar", src).unwrap_err();
        assert!(err.msg.contains("declared struct"), "got: {}", err.msg);
    }

    #[test]
    fn rejects_oversize_struct() {
        // Eight vec4<f32>s = 128 bytes, comfortably over the 64 B cap.
        let src = "struct Big {
            pos: vec3<f32>,
            a: vec4<f32>,
            b: vec4<f32>,
            c: vec4<f32>,
            d: vec4<f32>,
            e: vec4<f32>,
            f: vec4<f32>,
            g: vec4<f32>
        }";
        let err = parse_instance_layout(p(), "Big", src).unwrap_err();
        assert!(err.msg.contains("hard cap"), "got: {}", err.msg);
    }

    #[test]
    fn warns_on_soft_overflow_but_succeeds() {
        // pos (vec3<f32> = 16 with align) + rot (vec4 = 16) + 8 bytes = 40,
        // > 32 (soft) but < 64 (hard).
        let src = "struct M {
            pos: vec3<f32>,
            rot: vec4<f32>,
            a: u32,
            b: u32
        }";
        let l = parse_instance_layout(p(), "M", src).unwrap();
        assert_eq!(l.total_size, 48); // 16 (pos→pad) + 16 (rot) + 4+4 + pad to align 16
        assert!(!l.warnings.is_empty(), "expected soft-limit warning");
        assert!(l.warnings[0].contains("soft limit"));
    }

    #[test]
    fn vec3_alignment_pads_following_field() {
        // pos (size 12, align 16) followed by f32 — f32 lands at offset 12
        // because f32 alignment is 4, not 16. Confirms we don't accidentally
        // pad to 16 between vec3 and f32.
        let src = "struct V { pos: vec3<f32>, yaw: f32 }";
        let l = parse_instance_layout(p(), "V", src).unwrap();
        assert_eq!(l.fields[0].byte_offset, 0); // pos
        assert_eq!(l.fields[1].byte_offset, 12); // yaw — no pad needed
        assert_eq!(l.total_size, 16); // round to struct alignment 16
    }

    #[test]
    fn dense_grass_blade_layout() {
        // Realistic instance: pos(16) + yaw(4) + sway(4) + height(4) + tint(4) = 32.
        let src = "struct Blade {
            pos: vec3<f32>,
            yaw: f32,
            sway_phase: f32,
            height_scale: f32,
            tint: u32
        }";
        let l = parse_instance_layout(p(), "Blade", src).unwrap();
        assert_eq!(l.total_size, 32);
        assert_eq!(l.alignment, 16);
        assert!(l.warnings.is_empty(), "32 bytes is at the soft limit, not over");
        assert_eq!(l.fields.len(), 5);
        assert_eq!(l.fields[0].byte_offset, 0);
        assert_eq!(l.fields[1].byte_offset, 12);
        assert_eq!(l.fields[2].byte_offset, 16);
        assert_eq!(l.fields[3].byte_offset, 20);
        assert_eq!(l.fields[4].byte_offset, 24);
    }

    #[test]
    fn trailing_comma_is_tolerated() {
        let src = "struct X { pos: vec3<f32>, yaw: f32, }";
        let l = parse_instance_layout(p(), "X", src).unwrap();
        assert_eq!(l.fields.len(), 2);
    }

    #[test]
    fn whitespace_in_type_tolerated() {
        let src = "struct X { pos : vec3 < f32 >  , yaw : f32 }";
        let l = parse_instance_layout(p(), "X", src).unwrap();
        assert_eq!(l.fields[0].ty, WgslType::Vec3F32);
        assert_eq!(l.fields[1].ty, WgslType::F32);
    }

    #[test]
    fn rejects_duplicate_pos() {
        let src = "struct X { pos: vec3<f32>, pos: vec3<f32> }";
        let err = parse_instance_layout(p(), "X", src).unwrap_err();
        assert!(err.msg.contains("duplicate `pos`"), "got: {}", err.msg);
    }

    #[test]
    fn line_comments_in_body_are_stripped() {
        let src = "struct X {
            pos: vec3<f32>, // primary placement
            yaw: f32 // around up-axis
        }";
        let l = parse_instance_layout(p(), "X", src).unwrap();
        assert_eq!(l.fields.len(), 2);
    }
}
