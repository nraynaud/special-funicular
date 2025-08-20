override workgroup_size = 64;
override workgroupxy_size = 8;

struct Params {
    from_mip: u32,
    extrema_threshold: f32,
    extrema_border: i32
};

struct CountType {
    count: atomic<u32>,
    dispatch_x: atomic<u32>,
    dispatch_y: atomic<u32>,
    dispatch_z: atomic<u32>
}

@group(0) @binding(0) var<uniform> parameters: Params;
@group(0) @binding(1) var diff_stack: texture_2d_array<f32>;
// 4D position (x, y, scale, mip) of found extrema
@group(0) @binding(2) var<storage, read_write> extrema_storage: array<vec4f>;
@group(0) @binding(3) var<storage, read_write> extrema_count: CountType;

const NEIGHBOR_COUNT = 26;
const neighbors:array<vec3i, NEIGHBOR_COUNT>  = array(
    vec3i(-1, -1, -1), vec3i(0, -1, -1), vec3i(1, -1, -1),
    vec3i(-1, 0, -1), vec3i(0, 0, -1), vec3i(1, 0, -1),
    vec3i(-1, 1, -1), vec3i(0, 1, -1), vec3i(1, 1, -1),

    vec3i(-1, -1, 0), vec3i(0, -1, 0), vec3i(1, -1, 0),
    vec3i(-1, 0, 0), vec3i(1, 0, 0),
    vec3i(-1, 1, 0), vec3i(0, 1, 0), vec3i(1, 1, 0),

    vec3i(-1, -1, 1), vec3i(0, -1, 1), vec3i(1, -1, 1),
    vec3i(-1, 0, 1), vec3i(0, 0, 1), vec3i(1, 0, 1),
    vec3i(-1, 1, 1), vec3i(0, 1, 1), vec3i(1, 1, 1)
);

fn combine_min_max(mm1: vec2f, mm2: vec2f) -> vec2f {
    let new_min = min(mm1.x, mm2.x);
    let new_max = max(mm1.y, mm2.y);
    return vec2f(new_min, new_max);
}

@compute @workgroup_size(workgroupxy_size, workgroupxy_size, 1)
// one invocation per pixel
fn extrema(@builtin(global_invocation_id) global_id: vec3<u32>,
         @builtin(workgroup_id) workgroup_id: vec3<u32>,
         @builtin(local_invocation_id) local_id: vec3<u32>,
         @builtin(num_workgroups) num_workgroups: vec3<u32>) {
    let pixel_pos = vec2i(global_id.xy);
    let texture_size = vec2i(textureDimensions(diff_stack, parameters.from_mip));
    let v_border = vec2i(parameters.extrema_border);
    let array_index = i32(global_id.z) + 1;
    let current_texel = textureLoad(diff_stack, pixel_pos, array_index, parameters.from_mip).r;
    let wg_linearization = vec3u(1u, num_workgroups.x, num_workgroups.x * num_workgroups.y);
    let wg_linear_index = dot(workgroup_id.xyz, wg_linearization);
    if all(pixel_pos >= v_border) && all(pixel_pos < texture_size - v_border) {
        var current_min_max = vec2f(0);
        for (var i = 0; i < NEIGHBOR_COUNT; i++) {
            let neighbor = neighbors[i];
            let texel = vec2f(textureLoad(diff_stack, pixel_pos + neighbor.xy, array_index + neighbor.z, parameters.from_mip).r);
            current_min_max = select(combine_min_max(current_min_max, texel), texel, i == 0);
        }
        if abs(current_texel) > parameters.extrema_threshold && (current_texel < current_min_max.x || current_texel > current_min_max.y)  {
            let extremum_index = atomicAdd(&extrema_count.count, 1u);
            let invocation_count = extremum_index / u32(workgroup_size) + select(0u, 1u, extremum_index % u32(workgroup_size) > 0);
            atomicMax(&extrema_count.dispatch_x, invocation_count);
            atomicStore(&extrema_count.dispatch_y, 1);
            atomicStore(&extrema_count.dispatch_z, 1);
            if extremum_index < arrayLength(&extrema_storage) {
                extrema_storage[extremum_index] = vec4f(vec3f(global_id.xyz), f32(parameters.from_mip));
            }
        }
    }
}
