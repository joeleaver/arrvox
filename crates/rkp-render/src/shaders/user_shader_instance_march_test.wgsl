// user_shader_instance_march_test.wgsl — Stage 5a test harness.
//
// Three standalone compute entries that exercise the helpers from
// `user_shader_instance_march_helpers.wgsl` against deterministic
// inputs. The Rust composer concatenates that helpers file ahead of
// this one so all helper fns + pool bindings are in scope.
//
// Each entry takes a small uniform of inputs and writes a fixed-layout
// `result` buffer at @binding(2). One workgroup, one invocation per
// dispatch — the helpers are pure, so threading adds no signal.

struct AabbTestInputs {
    ro: vec3<f32>,
    _pad0: f32,
    rd: vec3<f32>,
    _pad1: f32,
    inv_dir: vec3<f32>,
    _pad2: f32,
    aabb_min: vec3<f32>,
    _pad3: f32,
    aabb_max: vec3<f32>,
    _pad4: f32,
}

struct WorldToLocalTestInputs {
    world_pos: vec3<f32>,
    instance_scale: f32,
    instance_pos: vec3<f32>,
    _pad0: f32,
}

struct ProtoDescendTestInputs {
    local_origin: vec3<f32>,
    octree_root: u32,
    local_dir: vec3<f32>,
    max_depth: u32,
    max_steps_outer: u32,
    max_steps_brick: u32,
    _pad0: u32,
    _pad1: u32,
}

struct AabbTestResult {
    t_near: f32,
    t_far: f32,
    _pad0: f32,
    _pad1: f32,
}

struct WorldToLocalTestResult {
    local: vec3<f32>,
    _pad0: f32,
}

struct ProtoDescendTestResult {
    hit: u32,
    leaf_attr_slot: u32,
    material_local: u32,
    _pad0: u32,
    t: f32,
    _pad1: f32,
    _pad2: f32,
    _pad3: f32,
    normal: vec3<f32>,
    _pad4: f32,
}

@group(1) @binding(0) var<uniform> aabb_inputs: AabbTestInputs;
@group(1) @binding(1) var<storage, read_write> aabb_result: AabbTestResult;

@group(2) @binding(0) var<uniform> w2l_inputs: WorldToLocalTestInputs;
@group(2) @binding(1) var<storage, read_write> w2l_result: WorldToLocalTestResult;

@group(3) @binding(0) var<uniform> proto_inputs: ProtoDescendTestInputs;
@group(3) @binding(1) var<storage, read_write> proto_result: ProtoDescendTestResult;

@compute @workgroup_size(1)
fn aabb_test_main() {
    let r = inst_ray_aabb_intersect(
        aabb_inputs.ro, aabb_inputs.inv_dir,
        aabb_inputs.aabb_min, aabb_inputs.aabb_max,
    );
    aabb_result.t_near = r.x;
    aabb_result.t_far = r.y;
}

@compute @workgroup_size(1)
fn world_to_local_test_main() {
    w2l_result.local = inst_world_to_local(
        w2l_inputs.world_pos, w2l_inputs.instance_pos, w2l_inputs.instance_scale,
    );
}

@compute @workgroup_size(1)
fn proto_descend_test_main() {
    let r = inst_proto_descend(
        proto_inputs.local_origin, proto_inputs.local_dir,
        proto_inputs.octree_root, proto_inputs.max_depth,
        proto_inputs.max_steps_outer, proto_inputs.max_steps_brick,
    );
    proto_result.hit = r.hit;
    proto_result.t = r.t;
    proto_result.normal = r.normal;
    proto_result.material_local = r.material_local;
    proto_result.leaf_attr_slot = r.leaf_attr_slot;
}
