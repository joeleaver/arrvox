//! `.arvxsculpt` save / load — per-tile sculpt diff sidecar.
//!
//! Companion to `.arvxtile`. Where `.arvxtile` saves a full baked
//! `BakeArtifact` (procedural + stamps + sculpt baked into the
//! octree), `.arvxsculpt` saves **only the sculpt edits as a
//! [`SculptDiff`]**, so:
//!
//! 1. **Coarse-LOD ancestors can replay it.** A coarse tile's bake
//!    consults the fine-tile diffs via the engine's
//!    `gather_replay_edits` hook; that hook reads
//!    `TerrainRuntime::diffs`, which is populated from this sidecar
//!    at scene load. Without `.arvxsculpt`, the diff map would be
//!    empty across sessions and coarse-LOD propagation would only
//!    cover this-session edits.
//! 2. **Re-bakes after eviction stay correct.** If a level-0 tile is
//!    evicted between an edit and the next save, the live state is
//!    lost — but the diff in `runtime.diffs` survives, and a
//!    subsequent save still writes the `.arvxsculpt`. On reload the
//!    diff is brought back even without a matching `.arvxtile`.
//!
//! ## File format (v1)
//!
//! Little-endian throughout. **No compression** — diffs are typically
//! a few hundred edits per tile (~kB), and the existing scene-save
//! footprint dwarfs any LZ4 savings.
//!
//! ```text
//! offset  bytes  field
//! 0       4      magic = b"AVXS"
//! 4       2      version = 1
//! 6       2      _reserved (always 0; lets v2 stash flag bits without
//!                growing the header)
//! 8       4      edit_count : u32
//! 12      ...    edits: edit_count records
//! ```
//!
//! Each edit record:
//! ```text
//! offset  bytes  field
//! 0       12     coord : u32 x 3 (UVec3)
//! 12      1      op_tag : u8
//! 13      ...    op_payload (variable per tag)
//! ```
//!
//! `op_tag` values:
//! * `0` → `LeafEditOp::Remove`, no payload
//! * `1` → `LeafEditOp::Add { material: u16, normal: Vec3 }`, 14 bytes
//! * `2` → `LeafEditOp::Empty`, no payload
//! * `3` → `LeafEditOp::SetInterior`, no payload
//!
//! `LeafEditOp::SetNormal` is never written — [`SculptDiff::append_delta`]
//! filters it because the variant carries a per-octree slot id that
//! can't be replayed onto a fresh bake.

use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use arvx_core::sculpt::{LeafEdit, LeafEditOp};
use glam::{UVec3, Vec3};

use crate::sculpt_diff::SculptDiff;
use crate::tile_key::TileKey;

/// File magic — "AVXS" = "arrvox sculpt".
const MAGIC: [u8; 4] = *b"AVXS";
/// Current on-disk format version. Bump when the layout changes.
const VERSION: u16 = 1;

/// Subdirectory under the scene root that holds saved `.arvxsculpt`
/// sidecars. Parallel to [`crate::persist::TILES_SUBDIR`].
pub const SCULPT_SUBDIR: &str = "sculpt";

/// On-disk file extension (without the leading dot).
const FILE_EXT: &str = "arvxsculpt";

/// Resolve the `.arvxsculpt` path for a tile inside a scene directory.
///
/// Layout: `<scene_dir>/sculpt/{level}_{x}_{y}_{z}.arvxsculpt`. The
/// scheme mirrors [`crate::persist::tile_path`] so the two sidecar
/// families round-trip via the same key.
pub fn sculpt_path(scene_dir: &Path, key: TileKey) -> PathBuf {
    scene_dir.join(SCULPT_SUBDIR).join(format!(
        "{}_{}_{}_{}.{}",
        key.level, key.x, key.y, key.z, FILE_EXT,
    ))
}

/// Persist a per-tile sculpt diff next to the scene. Atomic write via
/// `<path>.inprogress` + rename. Creates the `sculpt/` subdir on first
/// save.
///
/// Empty diffs are written verbatim (header + zero edits) — callers
/// that want to skip empties should check `diff.is_empty()` first.
/// The engine's flush path does exactly that to keep noise off disk.
pub fn save_sculpt_diff(
    scene_dir: &Path,
    key: TileKey,
    diff: &SculptDiff,
) -> Result<PathBuf, String> {
    let path = sculpt_path(scene_dir, key);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("create_dir_all {}: {e}", parent.display()))?;
    }
    let tmp = path.with_extension(format!("{FILE_EXT}.inprogress"));
    {
        let file = std::fs::File::create(&tmp)
            .map_err(|e| format!("create {}: {e}", tmp.display()))?;
        let mut w = std::io::BufWriter::new(file);
        write_header(&mut w, diff.edits.len() as u32)
            .map_err(|e| format!("write header: {e}"))?;
        for edit in &diff.edits {
            write_edit(&mut w, edit)
                .map_err(|e| format!("write edit: {e}"))?;
        }
        w.flush().map_err(|e| format!("flush: {e}"))?;
    }
    std::fs::rename(&tmp, &path)
        .map_err(|e| format!("rename {} → {}: {e}", tmp.display(), path.display()))?;
    Ok(path)
}

/// Read a `.arvxsculpt` back into a `SculptDiff`. Returns the file's
/// tile key — the caller is expected to have derived `path` from a
/// scene scan and uses the key to populate `TerrainRuntime::diffs`.
/// The key is parsed from the filename via [`parse_sculpt_filename`].
pub fn load_sculpt_diff(path: &Path) -> Result<(TileKey, SculptDiff), String> {
    let key = parse_sculpt_filename(path)
        .ok_or_else(|| format!("bad .arvxsculpt filename: {}", path.display()))?;
    let file = std::fs::File::open(path)
        .map_err(|e| format!("open {}: {e}", path.display()))?;
    let mut r = std::io::BufReader::new(file);

    let edit_count = read_header(&mut r).map_err(|e| format!("read header: {e}"))?;
    let mut edits = Vec::with_capacity(edit_count as usize);
    for i in 0..edit_count {
        let edit = read_edit(&mut r)
            .map_err(|e| format!("read edit {i}/{edit_count}: {e}"))?;
        edits.push(edit);
    }
    Ok((key, SculptDiff { edits }))
}

/// Scan `<scene_dir>/sculpt/` and load every `.arvxsculpt` into a
/// `HashMap<TileKey, SculptDiff>`. The engine calls this once at
/// scene load and populates `TerrainRuntime::diffs` so the post-
/// integrate replay path sees the per-tile edits.
///
/// Missing directory is a no-op (returns an empty map). Per-file
/// errors are logged via `eprintln!` and the file is skipped —
/// partial recovery is better than refusing to load the scene.
pub fn load_all_sculpt_diffs(
    scene_dir: &Path,
) -> std::collections::HashMap<TileKey, SculptDiff> {
    let mut out = std::collections::HashMap::new();
    let dir = scene_dir.join(SCULPT_SUBDIR);
    let entries = match std::fs::read_dir(&dir) {
        Ok(e) => e,
        Err(_) => return out, // missing dir → empty map
    };
    for entry in entries.flatten() {
        let p = entry.path();
        if p.extension().and_then(|s| s.to_str()) != Some(FILE_EXT) {
            continue;
        }
        match load_sculpt_diff(&p) {
            Ok((key, diff)) => {
                out.insert(key, diff);
            }
            Err(e) => {
                eprintln!(
                    "[terrain] failed to load sculpt diff {}: {e} (skipping)",
                    p.display(),
                );
            }
        }
    }
    out
}

/// Parse the level/x/y/z key out of a `.arvxsculpt` filename. Returns
/// `None` for any malformed name. Used by [`load_sculpt_diff`] and the
/// directory walker so callers don't have to re-implement the format.
fn parse_sculpt_filename(path: &Path) -> Option<TileKey> {
    let stem = path.file_stem()?.to_str()?;
    let parts: Vec<&str> = stem.split('_').collect();
    if parts.len() != 4 {
        return None;
    }
    Some(TileKey {
        level: parts[0].parse().ok()?,
        x: parts[1].parse().ok()?,
        y: parts[2].parse().ok()?,
        z: parts[3].parse().ok()?,
    })
}

fn write_header<W: Write>(w: &mut W, edit_count: u32) -> std::io::Result<()> {
    w.write_all(&MAGIC)?;
    w.write_all(&VERSION.to_le_bytes())?;
    w.write_all(&0u16.to_le_bytes())?; // _reserved
    w.write_all(&edit_count.to_le_bytes())?;
    Ok(())
}

fn read_header<R: Read>(r: &mut R) -> std::io::Result<u32> {
    let mut magic = [0u8; 4];
    r.read_exact(&mut magic)?;
    if magic != MAGIC {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("bad magic: got {magic:?}, expected {MAGIC:?}"),
        ));
    }
    let mut version_bytes = [0u8; 2];
    r.read_exact(&mut version_bytes)?;
    let version = u16::from_le_bytes(version_bytes);
    if version != VERSION {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("unsupported version: {version} (this build reads {VERSION})"),
        ));
    }
    let mut _reserved = [0u8; 2];
    r.read_exact(&mut _reserved)?;
    let mut count_bytes = [0u8; 4];
    r.read_exact(&mut count_bytes)?;
    Ok(u32::from_le_bytes(count_bytes))
}

fn write_edit<W: Write>(w: &mut W, edit: &LeafEdit) -> std::io::Result<()> {
    w.write_all(&edit.coord.x.to_le_bytes())?;
    w.write_all(&edit.coord.y.to_le_bytes())?;
    w.write_all(&edit.coord.z.to_le_bytes())?;
    match edit.op {
        LeafEditOp::Remove => w.write_all(&[0u8])?,
        LeafEditOp::Add { material, normal, .. } => {
            w.write_all(&[1u8])?;
            w.write_all(&material.to_le_bytes())?;
            w.write_all(&normal.x.to_le_bytes())?;
            w.write_all(&normal.y.to_le_bytes())?;
            w.write_all(&normal.z.to_le_bytes())?;
        }
        LeafEditOp::Empty => w.write_all(&[2u8])?,
        LeafEditOp::SetInterior => w.write_all(&[3u8])?,
        LeafEditOp::SetNormal { .. } | LeafEditOp::SetDist { .. } => {
            // The SculptDiff::append_delta filter (AggOp::from_leaf_op → None)
            // drops SetNormal / SetDist before they ever reach the diff;
            // reaching here means a caller bypassed the filter, which would
            // produce a non-replayable diff (both carry a per-octree slot id).
            // Surface it loudly.
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "SetNormal / SetDist ops cannot be persisted — their slot id is per-octree",
            ));
        }
    }
    Ok(())
}

fn read_edit<R: Read>(r: &mut R) -> std::io::Result<LeafEdit> {
    let mut buf12 = [0u8; 12];
    r.read_exact(&mut buf12)?;
    let coord = UVec3::new(
        u32::from_le_bytes(buf12[0..4].try_into().unwrap()),
        u32::from_le_bytes(buf12[4..8].try_into().unwrap()),
        u32::from_le_bytes(buf12[8..12].try_into().unwrap()),
    );
    let mut tag_buf = [0u8; 1];
    r.read_exact(&mut tag_buf)?;
    let op = match tag_buf[0] {
        0 => LeafEditOp::Remove,
        1 => {
            let mut mat_buf = [0u8; 2];
            r.read_exact(&mut mat_buf)?;
            let material = u16::from_le_bytes(mat_buf);
            let mut nrm_buf = [0u8; 12];
            r.read_exact(&mut nrm_buf)?;
            let normal = Vec3::new(
                f32::from_le_bytes(nrm_buf[0..4].try_into().unwrap()),
                f32::from_le_bytes(nrm_buf[4..8].try_into().unwrap()),
                f32::from_le_bytes(nrm_buf[8..12].try_into().unwrap()),
            );
            LeafEditOp::Add { material, normal, dist: 0.0 }
        }
        2 => LeafEditOp::Empty,
        3 => LeafEditOp::SetInterior,
        other => {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("unknown op tag {other}"),
            ));
        }
    };
    Ok(LeafEdit { coord, op })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn diff_with_edits() -> SculptDiff {
        SculptDiff {
            edits: vec![
                LeafEdit {
                    coord: UVec3::new(1, 2, 3),
                    op: LeafEditOp::Add {
                        material: 42,
                        normal: Vec3::new(0.0, 1.0, 0.0),
                        dist: 0.0,
                    },
                },
                LeafEdit {
                    coord: UVec3::new(4, 5, 6),
                    op: LeafEditOp::Remove,
                },
                LeafEdit {
                    coord: UVec3::new(7, 8, 9),
                    op: LeafEditOp::SetInterior,
                },
                LeafEdit {
                    coord: UVec3::new(10, 11, 12),
                    op: LeafEditOp::Empty,
                },
            ],
        }
    }

    #[test]
    fn sculpt_path_layout_matches_scheme() {
        let dir = PathBuf::from("/tmp/scene");
        let p = sculpt_path(&dir, TileKey::level0(3, -2, 5));
        assert_eq!(p, PathBuf::from("/tmp/scene/sculpt/0_3_-2_5.arvxsculpt"));
    }

    #[test]
    fn parse_filename_round_trips() {
        let key = TileKey {
            level: 2,
            x: -7,
            y: 0,
            z: 12,
        };
        let p = sculpt_path(Path::new("/scene"), key);
        assert_eq!(parse_sculpt_filename(&p), Some(key));
    }

    #[test]
    fn parse_filename_rejects_garbage() {
        assert_eq!(parse_sculpt_filename(Path::new("/x/garbage.arvxsculpt")), None);
        assert_eq!(parse_sculpt_filename(Path::new("/x/a_b_c.arvxsculpt")), None);
    }

    #[test]
    fn save_then_load_preserves_diff() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let key = TileKey::level0(1, 2, 3);
        let diff = diff_with_edits();
        let saved = save_sculpt_diff(tmp.path(), key, &diff).expect("save");
        assert!(saved.exists());

        let (loaded_key, loaded_diff) =
            load_sculpt_diff(&saved).expect("load");
        assert_eq!(loaded_key, key);
        assert_eq!(loaded_diff.len(), diff.len());
        for (a, b) in loaded_diff.edits.iter().zip(diff.edits.iter()) {
            assert_eq!(a.coord, b.coord);
            // Compare ops via discriminant + payload — LeafEditOp has
            // no PartialEq.
            match (a.op, b.op) {
                (LeafEditOp::Remove, LeafEditOp::Remove) => {}
                (
                    LeafEditOp::Add { material: m1, normal: n1, .. },
                    LeafEditOp::Add { material: m2, normal: n2, .. },
                ) => {
                    assert_eq!(m1, m2);
                    assert!((n1 - n2).length() < 1e-6);
                }
                (LeafEditOp::Empty, LeafEditOp::Empty) => {}
                (LeafEditOp::SetInterior, LeafEditOp::SetInterior) => {}
                (a_op, b_op) => panic!("op mismatch: {a_op:?} vs {b_op:?}"),
            }
        }
    }

    #[test]
    fn empty_diff_round_trips() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let key = TileKey::level0(0, 0, 0);
        let diff = SculptDiff::new();
        let saved = save_sculpt_diff(tmp.path(), key, &diff).expect("save");
        let (k, d) = load_sculpt_diff(&saved).expect("load");
        assert_eq!(k, key);
        assert!(d.is_empty());
    }

    #[test]
    fn load_all_picks_up_every_sculpt_file() {
        let tmp = tempfile::tempdir().expect("tempdir");
        save_sculpt_diff(tmp.path(), TileKey::level0(0, 0, 0), &diff_with_edits())
            .expect("save 1");
        save_sculpt_diff(tmp.path(), TileKey::level0(1, 0, 0), &diff_with_edits())
            .expect("save 2");
        // Drop a non-arvxsculpt file in the dir to verify the filter.
        std::fs::write(
            tmp.path().join(SCULPT_SUBDIR).join("noise.txt"),
            "ignored",
        )
        .expect("write noise");

        let all = load_all_sculpt_diffs(tmp.path());
        assert_eq!(all.len(), 2);
        assert!(all.contains_key(&TileKey::level0(0, 0, 0)));
        assert!(all.contains_key(&TileKey::level0(1, 0, 0)));
    }

    #[test]
    fn load_missing_dir_returns_empty_map() {
        let tmp = tempfile::tempdir().expect("tempdir");
        // No sculpt/ subdir created.
        let all = load_all_sculpt_diffs(tmp.path());
        assert!(all.is_empty());
    }

    /// Bad magic surfaces as a clean error, not a panic.
    #[test]
    fn bad_magic_is_an_error() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let key = TileKey::level0(0, 0, 0);
        let p = sculpt_path(tmp.path(), key);
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        std::fs::write(&p, b"BADMAGIC_BADMAGIC").expect("write");
        let err = load_sculpt_diff(&p).unwrap_err();
        assert!(err.contains("magic"), "unexpected error: {err}");
    }
}
