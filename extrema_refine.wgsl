override workgroup_size = 64;
override workgroupxy_size = 8;

struct Params {
    from_mip: u32,
    extrema_threshold: f32,
    extrema_border: i32
};

@group(0) @binding(1) var<uniform> parameters: Params;
@group(0) @binding(2) var diff_stack: texture_storage_2d_array<r32sint, read_write>;
// 4D position (x, y, scale, mip) of found extrema
@group(0) @binding(10) var<storage, read_write> extrema_storage: array<vec4f>;
@group(0) @binding(11) var<storage, read_write> extrema_count: atomic<u32>;

const MAX_2_31 = pow(2, 31) - 1.0;

@compute @workgroup_size(workgroup_size, 1, 1)
// one invocation per extremum
fn extrema_refine(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let extremum = extrema_storage[global_id.x];
    let xy = vec2i(round(extremum.xy));
    let mip = i32(round(extremum.w));
    let texel = textureLoad(diff_stack, xy, i32(round(extremum.z))).r;
}
