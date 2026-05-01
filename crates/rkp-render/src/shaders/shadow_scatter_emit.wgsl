// Phase 8 V2 — shadow scatter emit pass.
//
// One workgroup per `tlas_prim` (`workgroup_size(64)`). Each
// workgroup fills its instance's slot range in `work_list` with
// packed `(instance_idx, tile_x_local, tile_y_local)` tuples —
// one per 8×8 tile of the instance's projected rect.
//
// Threads cooperate within the workgroup: thread `lid.x` writes
// every 64th tile (k = lid.x, 64+lid.x, 128+lid.x, ...).
// Big instances (e.g., the studio floor's 256×256 = 65 536 tiles)
// have all 64 threads loop ~1024 iterations each. Small
// instances (grass blades' ~9 tiles) have most threads idle —
// no work, fast.
//
// Pack format (32 bits / entry):
//   bits  0..15 — instance_idx
//   bits 16..23 — tile_x_local  (0..255, fits a 2K shadow map / 8)
//   bits 24..31 — tile_y_local
//
// MAX 65 535 instances and 256 tiles per axis is plenty for V1.

struct ScatterInstance {
    tx0: u32, ty0: u32,
    tile_w: u32, tile_h: u32,
    asset_id: u32,
    instance_state_offset: u32,
    instance_index: u32,
    work_offset: u32,
}

@group(0) @binding(0) var<storage, read> scatter_instances: array<ScatterInstance>;
@group(0) @binding(1) var<storage, read_write> work_list: array<u32>;

fn pack_work(inst_idx: u32, tile_x_local: u32, tile_y_local: u32) -> u32 {
    return (inst_idx & 0xFFFFu)
        | ((tile_x_local & 0xFFu) << 16u)
        | ((tile_y_local & 0xFFu) << 24u);
}

@compute @workgroup_size(64, 1, 1)
fn emit_main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let i = wid.x;
    if i >= arrayLength(&scatter_instances) { return; }
    let inst = scatter_instances[i];
    let tile_count = inst.tile_w * inst.tile_h;
    if tile_count == 0u { return; }

    var k = lid.x;
    loop {
        if k >= tile_count { break; }
        let tile_x_local = k % inst.tile_w;
        let tile_y_local = k / inst.tile_w;
        work_list[inst.work_offset + k] = pack_work(i, tile_x_local, tile_y_local);
        k = k + 64u;
    }
}
