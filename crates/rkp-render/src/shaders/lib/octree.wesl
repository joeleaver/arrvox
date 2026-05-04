// Octree-traversal helpers shared across the host march and shadow
// passes. Pure math — no bindings, no struct dependencies.
//
// CAVEAT: most octree helpers (`octree_lookup`, `descend_proto_octree`,
// `dispatch_user_instance_descend`) DON'T live here:
//
//   - `octree_lookup` has signature divergence between consumers
//     (5-arg with-stats variant in `octree_march`/`rkp_shadow_trace`,
//     4-arg no-stats variant in `shadow_scatter`); a unified
//     signature would force unused arguments on most callers.
//   - `descend_proto_octree` is called from the user-shader
//     composer's spliced `instance_at` chunk — splice-helpers must
//     keep their unmangled name in the consumer file. See
//     `feedback_wesl_splice_chunk_helpers` memory.
//   - `dispatch_user_instance_descend` IS the splice target — its
//     declaration is what `splice_const_marker` replaces.

// Compute the t-distance to the next octree node boundary along a
// ray. `vs` is the voxel size (smallest cell dimension); the
// `+ vs * 0.01` epsilon kicks the ray just past the boundary so the
// next iteration steps into the new node rather than re-hitting the
// boundary.
fn skip_node(pos: vec3<f32>, dir: vec3<f32>, inv_dir: vec3<f32>, node_depth: u32, extent: f32, vs: f32) -> f32 {
    let node_size = extent / f32(1u << node_depth);
    let node_min = floor(pos / node_size) * node_size;
    let node_max = node_min + node_size;
    // Exit planes: for positive dir, exit through max; for negative, through min.
    let t_exit = select((node_min - pos) * inv_dir, (node_max - pos) * inv_dir, dir > vec3<f32>(0.0));
    // Smallest positive exit = nearest boundary crossing.
    let t_pos = max(t_exit, vec3<f32>(1e-6));
    return min(min(t_pos.x, t_pos.y), t_pos.z) + vs * 0.01;
}
