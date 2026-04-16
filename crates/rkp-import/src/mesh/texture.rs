//! FBX texture-data lookup: embedded-content first, then a disk fallback
//! chain (absolute path → path relative to the FBX file's directory).
//! Shared between mesh and skeleton FBX loaders.

use std::path::Path;

use super::TextureData;

/// Try to decode an FBX texture's embedded content, or failing that,
/// locate it on disk. Walks the candidate paths in order:
/// 1. Embedded `tex.content` bytes
/// 2. `tex.relative_filename` as an absolute path
/// 3. `tex.relative_filename`'s basename, joined onto `fbx_dir`
/// 4. `tex.absolute_filename` as an absolute path
/// 5. `tex.absolute_filename`'s basename, joined onto `fbx_dir`
///
/// Returns `None` (with a stderr warning) if no candidate decodes.
pub fn load_fbx_texture(tex: &ufbx::Texture, fbx_dir: &Path) -> Option<TextureData> {
    if !tex.content.is_empty() {
        if let Some(decoded) = decode_bytes(&tex.content) {
            return Some(decoded);
        }
    }

    for filename in [&tex.relative_filename, &tex.absolute_filename] {
        if filename.is_empty() {
            continue;
        }
        if let Some(decoded) = try_load_path(filename.as_ref(), fbx_dir) {
            return Some(decoded);
        }
    }

    let name = if !tex.relative_filename.is_empty() {
        &tex.relative_filename
    } else {
        &tex.absolute_filename
    };
    eprintln!("[rkp-import] warn: failed to load FBX texture '{name}'");
    None
}

fn try_load_path(filename: &str, fbx_dir: &Path) -> Option<TextureData> {
    let path = Path::new(filename);
    if path.is_absolute() {
        if let Ok(decoded) = decode_file(path) {
            return Some(decoded);
        }
    }
    let rel = fbx_dir.join(path.file_name().unwrap_or_default());
    decode_file(&rel).ok()
}

fn decode_file(path: &Path) -> Result<TextureData, image::ImageError> {
    let img = image::open(path)?;
    let rgba = img.to_rgba8();
    Ok(TextureData {
        width: rgba.width(),
        height: rgba.height(),
        data: rgba.into_raw(),
    })
}

fn decode_bytes(bytes: &[u8]) -> Option<TextureData> {
    let img = image::load_from_memory(bytes).ok()?;
    let rgba = img.to_rgba8();
    Some(TextureData {
        width: rgba.width(),
        height: rgba.height(),
        data: rgba.into_raw(),
    })
}
