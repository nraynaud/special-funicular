override workgroup_size = 64;
override workgroupxy_size = 8;

struct Params {
    from_mip: u32,
    extrema_threshold: f32,
    extrema_border: i32
};

@group(0) @binding(2) var<uniform> parameters: Params;
@group(0) @binding(5) var diff_stack: texture_storage_2d_array<r32sint, read>;
// 4D position (x, y, scale, mip) of found extrema
@group(0) @binding(10) var<storage, read_write> extrema_storage: array<vec4f>;
@group(0) @binding(11) var<storage, read_write> extrema_count: atomic<u32>;

const MAX_2_31 = pow(2, 31) - 1.0;

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

fn combine_min_max(mm1: vec2i, mm2: vec2i) -> vec2i {
    let new_min = min(mm1.x, mm2.x);
    let new_max = max(mm1.y, mm2.y);
    return vec2i(new_min, new_max);
}

@compute @workgroup_size(workgroupxy_size, workgroupxy_size, 1)
fn extrema(@builtin(global_invocation_id) global_id: vec3<u32>,
         @builtin(workgroup_id) workgroup_id: vec3<u32>,
         @builtin(local_invocation_id) local_id: vec3<u32>,
         @builtin(num_workgroups) num_workgroups: vec3<u32>) {
    let i_threshold = i32(parameters.extrema_threshold * MAX_2_31);
    let pixel_pos = vec2i(global_id.xy);
    let texture_size = vec2i(textureDimensions(diff_stack));
    let v_border = vec2i(parameters.extrema_border);
    let array_index = i32(global_id.z) + 1;
    let current_texel = textureLoad(diff_stack, pixel_pos, array_index).r;
    let wg_linearization = vec3u(1u, num_workgroups.x, num_workgroups.x * num_workgroups.y);
    let wg_linear_index = dot(workgroup_id.xyz, wg_linearization);
    if all(pixel_pos >= v_border) && all(pixel_pos < texture_size - v_border) {
        var current_min_max = vec2i(0);
        for (var i = 0; i < NEIGHBOR_COUNT; i++) {
            let neighbor = neighbors[i];
            let texel = vec2i(textureLoad(diff_stack, pixel_pos + neighbor.xy, array_index + neighbor.z).r);
            current_min_max = select(combine_min_max(current_min_max, texel), texel, i == 0);
        }
        if abs(current_texel) > i_threshold && (current_texel < current_min_max.x || current_texel > current_min_max.y)  {
            let extremum_index = atomicAdd(&extrema_count, 1u);
            if extremum_index < arrayLength(&extrema_storage) {
                extrema_storage[extremum_index] = vec4f(vec3f(global_id.xyz), f32(parameters.from_mip));
            }
        }
    }
}


@compute @workgroup_size(workgroup_size, 1, 1)
fn refine_extrema(@builtin(global_invocation_id) global_id: vec3<u32>) {

}
