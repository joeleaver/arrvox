use super::*;
use std::io::{Cursor, Seek, SeekFrom, Write};


#[test]
fn header_size_is_144_bytes() {
    // v5: grew from 128 → 144 with the addition of three mesh
    // section size slots + the lod0_index_count prefix (4 × u32).
    assert_eq!(std::mem::size_of::<RkpHeader>(), 144);
}

#[test]
fn write_and_read_header_roundtrip() {
    let mut buf = Vec::new();
    let mut cursor = Cursor::new(&mut buf);

    let octree_nodes: Vec<u32> = vec![0xFFFF_FFFF]; // single EMPTY root
    let voxel_data: Vec<u8> = Vec::new();

    write_rkp(
        &mut cursor,
        &octree_nodes,
        1,
        0.1,
        0,
        [-1.0, -1.0, -1.0],
        [1.0, 1.0, 1.0],
        &[0, 1],
        &voxel_data,
        None,
        None,
        None,
        None, // skin_meta
        None, // mesh_sections
    )
    .unwrap();

    cursor.seek(SeekFrom::Start(0)).unwrap();
    let header = read_rkp_header(&mut cursor).unwrap();

    assert_eq!(header.magic, RKP_MAGIC);
    assert_eq!(header.version, RKP_VERSION);
    assert_eq!(header.octree_node_count, 1);
    assert_eq!(header.octree_depth, 1);
    assert!((header.base_voxel_size - 0.1).abs() < 1e-6);
    assert_eq!(header.voxel_count, 0);
    assert_eq!(header.material_ids[0], 0);
    assert_eq!(header.material_ids[1], 1);
    assert_eq!(header.flags & FLAG_HAS_COLOR, 0);
    assert_eq!(header.flags & FLAG_HAS_BONES, 0);
}

#[test]
fn write_and_read_skin_meta_roundtrip() {
    // Three voxels, two bricks, two bones — exercises every part
    // of the structured skin-meta payload: weights, origins, and
    // rest AABBs all survive the LZ4 + length-prefix round trip.
    use crate::companion::BoneVoxel;

    let bones: Vec<BoneVoxel> = vec![
        BoneVoxel::new([0, 1, 2, 3], [64, 64, 64, 63]),
        BoneVoxel::new([4, 0, 0, 0], [255, 0, 0, 0]),
        BoneVoxel::new([7, 3, 0, 0], [200, 55, 0, 0]),
    ];
    let bone_bytes: &[u8] = bytemuck::cast_slice(&bones);
    let brick_origins: Vec<[u32; 3]> = vec![[0, 0, 0], [8, 0, 0]];
    let rest_aabbs: Vec<[f32; 6]> = vec![
        [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        [-1.0, -2.0, -3.0, 2.0, 3.0, 4.0],
    ];
    let voxel_bytes = vec![0u8; 3 * std::mem::size_of::<crate::voxel::VoxelSample>()];

    let mut buf = Vec::new();
    let mut cursor = Cursor::new(&mut buf);
    write_rkp(
        &mut cursor,
        &[0xFFFF_FFFF],   // single EMPTY root octree
        1,
        0.1,
        3,                // voxel_count
        [-1.0; 3], [1.0; 3],
        &[0],
        &voxel_bytes,
        None, None, None,
        Some(SkinMetaIn {
            bone_voxels: bone_bytes,
            brick_origins: &brick_origins,
            rest_bone_aabbs: &rest_aabbs,
        }),
        None, // mesh_sections
    )
    .unwrap();

    cursor.seek(SeekFrom::Start(0)).unwrap();
    let header = read_rkp_header(&mut cursor).unwrap();
    assert!(header.flags & FLAG_HAS_BONES != 0, "FLAG_HAS_BONES must be set");
    assert!(header.bone_compressed_size > 0, "skin-meta section must be non-empty");

    let _ = read_rkp_octree(&mut cursor, &header).unwrap();
    let _ = read_rkp_voxels(&mut cursor, &header).unwrap();
    let back = read_rkp_skin_meta(&mut cursor, &header).unwrap();

    assert_eq!(back.bone_voxels, bone_bytes, "bone-voxel bytes must roundtrip");
    assert_eq!(back.brick_origins, brick_origins, "brick origins must roundtrip");
    assert_eq!(back.rest_bone_aabbs, rest_aabbs, "rest bone AABBs must roundtrip");

    // Decode bone voxels + weight-sum invariant.
    let decoded: &[BoneVoxel] = bytemuck::cast_slice(&back.bone_voxels);
    for (i, (bv_in, bv_out)) in bones.iter().zip(decoded).enumerate() {
        for slot in 0..4 {
            assert_eq!(bv_in.bone_index(slot), bv_out.bone_index(slot), "bone_index mismatch at voxel {i} slot {slot}");
            assert_eq!(bv_in.bone_weight(slot), bv_out.bone_weight(slot), "bone_weight mismatch at voxel {i} slot {slot}");
        }
        let sum: u16 = (0..4).map(|s| bv_out.bone_weight(s) as u16).sum();
        assert_eq!(sum, 255, "voxel {i} weights must sum to 255");
    }
}

#[test]
fn write_artifact_rkp_roundtrip() {
    // Bake a tiny sphere into a BakeArtifact via the canonical
    // voxelize path, persist through write_artifact_rkp, then read
    // the sections back and check material/normal/brick/color
    // round-trip. This is the procedural bake-cache pipeline end
    // to end minus the scene integration.
    use crate::voxel::VoxelSample;
    use glam::Vec3;

    let voxel_size = 0.05;
    let half = Vec3::splat(0.3);
    let aabb = crate::Aabb::new(-half, half);
    let radius: f32 = 0.25;
    let sdf = |positions: &[Vec3]| -> Vec<(f32, u16, u16, u8, u32)> {
        positions
            .iter()
            .map(|p| (p.length() - radius, 7u16, 0u16, 0u8, 0u32))
            .collect()
    };

    let mut artifact = crate::voxelize_to_artifact(sdf, &aabb, voxel_size)
        .expect("voxelize sphere");
    assert!(artifact.voxel_count > 0, "sphere must produce voxels");
    // Spike a non-zero color on the first leaf so the color
    // section is emitted — verifies `has_color` detection works.
    artifact.leaf_attr_colors[0] = 0xFFAABBCC;

    let tmp = std::env::temp_dir().join(format!(
        "rkp_artifact_roundtrip_{}.rkp",
        std::process::id()
    ));
    let _ = std::fs::remove_file(&tmp);
    write_artifact_rkp(
        &tmp,
        &artifact,
        aabb.min.to_array(),
        aabb.max.to_array(),
        voxel_size,
    )
    .expect("write_artifact_rkp");

    let mut file = std::fs::File::open(&tmp).expect("open");
    let mut reader = std::io::BufReader::new(&mut file);
    let header = read_rkp_header(&mut reader).expect("read header");
    // Header stores leaf_attr-slot count, not cell count — the
    // per-slot voxel_data length is what the loader cares about.
    // `voxelize_to_artifact` already ran prefilter, so this
    // includes internal-node attrs in addition to the surface
    // leaves. On load, a fresh prefilter appends again; the
    // unreferenced "old" prefilter attrs linger harmlessly.
    assert_eq!(header.voxel_count, artifact.leaf_attrs.len() as u32);
    assert_eq!(header.octree_depth as u8, artifact.octree.depth());
    assert!((header.base_voxel_size - voxel_size).abs() < 1e-6);
    assert!(header.flags & FLAG_HAS_BRICKS != 0);
    assert!(header.flags & FLAG_HAS_NORMALS != 0);
    assert!(header.flags & FLAG_HAS_COLOR != 0);

    let octree_nodes = read_rkp_octree(&mut reader, &header).expect("octree");
    assert_eq!(octree_nodes, artifact.octree.as_slice());

    let voxel_bytes = read_rkp_voxels(&mut reader, &header).expect("voxels");
    let voxels: &[VoxelSample] = bytemuck::cast_slice(&voxel_bytes);
    assert_eq!(voxels.len(), artifact.leaf_attrs.len());
    for (v, a) in voxels.iter().zip(artifact.leaf_attrs.iter()) {
        assert_eq!(v.material_id(), a.material_primary);
        assert_eq!(v.secondary_material_id(), a.material_secondary());
        assert_eq!(v.blend_weight(), a.blend_weight());
    }

    let normals_bytes = read_rkp_normals(&mut reader, &header).expect("normals");
    let normals: &[u32] = bytemuck::cast_slice(&normals_bytes);
    assert_eq!(normals.len(), artifact.leaf_attrs.len());
    for (n, a) in normals.iter().zip(artifact.leaf_attrs.iter()) {
        assert_eq!(*n, a.normal_oct);
    }

    let bricks_bytes = read_rkp_bricks(&mut reader, &header).expect("bricks");
    let bricks: &[u32] = bytemuck::cast_slice(&bricks_bytes);
    let expected_brick_u32s = artifact.brick_cells.len() * crate::BRICK_CELLS as usize;
    assert_eq!(bricks.len(), expected_brick_u32s);

    let color_bytes = read_rkp_color(&mut reader, &header).expect("colors");
    let colors: &[u32] = bytemuck::cast_slice(&color_bytes);
    assert_eq!(colors.len(), artifact.leaf_attr_colors.len());
    assert_eq!(colors[0], 0xFFAABBCC);

    let _ = std::fs::remove_file(&tmp);
}

#[test]
fn write_and_read_octree_roundtrip() {
    let mut buf = Vec::new();
    let mut cursor = Cursor::new(&mut buf);

    let octree_nodes: Vec<u32> = vec![1, 0xFFFF_FFFF, 0x8000_002A, 0xFFFF_FFFF,
                                      0xFFFF_FFFF, 0xFFFF_FFFF, 0xFFFF_FFFF, 0xFFFF_FFFF,
                                      0xFFFF_FFFF];

    write_rkp(
        &mut cursor,
        &octree_nodes,
        1,
        0.1,
        1,
        [-1.0; 3],
        [1.0; 3],
        &[],
        &[0u8; 8], // one voxel = 1 VoxelSample * 8 bytes
        None,
        None,
        None,
        None, // skin_meta
        None, // mesh_sections
    )
    .unwrap();

    cursor.seek(SeekFrom::Start(0)).unwrap();
    let header = read_rkp_header(&mut cursor).unwrap();
    let nodes = read_rkp_octree(&mut cursor, &header).unwrap();

    assert_eq!(nodes, octree_nodes);
}
