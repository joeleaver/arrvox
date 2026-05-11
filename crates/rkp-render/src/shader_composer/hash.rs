//! Deterministic FNV-1a 64 hash of a registry's source content.
//!
//! Why FNV: std's `DefaultHasher` uses a per-process random seed, which
//! would invalidate every cache on every restart. FNV is keyless and
//! stable across runs, so cache keys built from this hash survive
//! editor restarts. The bake worker uses the registry's `source_hash`
//! as its cache key; consumers compare hashes to skip no-op reloads.

use super::types::{GeometryDecl, SpawnCountCache, UserShaderEntry};

/// Deterministic FNV-1a 64. Public so tests in the integration layer
/// can reproduce the empty-registry hash for sanity checks.
pub fn fnv1a_64(bytes: &[u8]) -> u64 {
    const OFFSET: u64 = 0xcbf29ce484222325;
    const PRIME: u64 = 0x100000001b3;
    let mut h = OFFSET;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(PRIME);
    }
    h
}

/// Hash every entry's hooks + helpers + struct decls + metadata, in
/// alphabetical order by name. The buffer layout is deliberately
/// stable: edits to one shader move the hash without depending on
/// scan order.
pub(super) fn compute_registry_hash(entries: &[UserShaderEntry]) -> u64 {
    let mut sorted: Vec<&UserShaderEntry> = entries.iter().collect();
    sorted.sort_by(|a, b| a.name.cmp(&b.name));
    let mut buf = Vec::new();
    for e in sorted {
        buf.extend_from_slice(e.name.as_bytes());
        buf.push(0);
        for hook in [
            &e.shade_text,
            &e.generate_text,
            &e.spawn_count_text,
            &e.spawn_alive_text,
            &e.vs_text,
            &e.fs_text,
        ] {
            if let Some(t) = hook {
                buf.extend_from_slice(t.as_bytes());
            }
            buf.push(0);
        }
        for helper in &e.helpers {
            buf.extend_from_slice(helper.as_bytes());
            buf.push(0);
        }
        for sd in &e.struct_decls {
            buf.extend_from_slice(sd.as_bytes());
            buf.push(0);
        }
        // Metadata also contributes to the hash so a change to default
        // values / range / @animated invalidates dependent caches.
        for p in &e.metadata.params {
            buf.extend_from_slice(p.name.as_bytes());
            buf.push(0);
            buf.extend_from_slice(&p.default.to_le_bytes());
            if let Some((lo, hi)) = p.range {
                buf.extend_from_slice(&lo.to_le_bytes());
                buf.extend_from_slice(&hi.to_le_bytes());
            }
            buf.push(0);
        }
        buf.extend_from_slice(&e.metadata.region_thickness.to_le_bytes());
        buf.push(if e.metadata.animated { 1 } else { 0 });
        if let Some(s) = e.metadata.cell_size {
            buf.extend_from_slice(&s.to_le_bytes());
        }
        buf.push(0);
        if let Some(d) = e.metadata.max_depth {
            buf.extend_from_slice(&d.to_le_bytes());
        }
        buf.push(0);
        if let Some(s) = e.metadata.tile_size {
            buf.extend_from_slice(&s.to_le_bytes());
        }
        buf.push(0);
        // V1 mesh-path manifest. Appending keeps the hash stable for
        // older shaders (None → zero byte → no change to bytes for
        // non-mesh entries) while still propagating directive edits.
        match &e.metadata.mesh_geometry {
            None => buf.push(0),
            Some(GeometryDecl::Procedural { vertex_count }) => {
                buf.push(1);
                buf.extend_from_slice(&vertex_count.to_le_bytes());
            }
            Some(GeometryDecl::Mesh { asset }) => {
                buf.push(2);
                buf.extend_from_slice(asset.as_bytes());
            }
        }
        buf.push(0);
        buf.push(match e.metadata.spawn_count_cache {
            SpawnCountCache::Static => 0,
            SpawnCountCache::PerFrame => 1,
        });
    }
    fnv1a_64(&buf)
}
