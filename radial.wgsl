override workgroup_size = 64;
override workgroupxy_size = 8;
const use_workgroup_mem = 1;
//maxComputeWorkgroupStorageSize is 16384, reserving a bit of leeway
override workgroup_pixel_count = 16300/4;
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
struct Params {
    horizontal: u32,
    from_mip: u32,
    to_mip: u32,
    convert_to_gray: u32,
    diff_index: u32,
    from_gray_negative: u32,
    extrema_threshold: f32,
    extrema_border: i32,
    max_extrema_per_wg: u32,
    // used to mimic OpenCV
    border_reflect_101: u32
};

@group(0) @binding(0) var inputTexture: texture_2d<i32>;
@group(0) @binding(1) var gaussian_textures: texture_storage_2d<r32sint, read_write>;
@group(0) @binding(2) var<uniform> parameters: Params;
@group(0) @binding(3) var<storage> kernel: array<f32>;
@group(0) @binding(4) var diff_input_stack: texture_2d_array<i32>;
@group(0) @binding(5) var diff_output_stack: texture_storage_2d_array<r32sint, read_write>;
@group(0) @binding(6) var max_input_stack: texture_2d_array<f32>;
@group(0) @binding(7) var input_rgba: texture_2d<f32>;
@group(0) @binding(8) var output_gray: texture_storage_2d<r32sint, write>;
@group(0) @binding(9) var output_rgba: texture_storage_2d<rgba8unorm, write>;
// 4D position (x, y, scale, mip) of found extrema
@group(0) @binding(10) var<storage, read_write> extrema_storage: array<vec4f>;
@group(0) @binding(11) var<storage, read_write> extrema_count: atomic<u32>;

var<workgroup> workgroupPixels: array<u32, workgroup_pixel_count>;

fn read_input(pix_pos: vec2i) -> vec4i {
    return textureLoad(inputTexture, pix_pos, parameters.from_mip);
}

fn handle_border(pix_pos: vec2i, img_size: vec2i) -> vec2i {
    if parameters.border_reflect_101 != 0 {
        // https://github.com/opencv/opencv/blob/4.x/modules/imgproc/src/opencl/filterSep_singlePass.cl#L67
        var new_pos = max(pix_pos, -pix_pos);
        new_pos = min(new_pos, (img_size - 1) * 2 - pix_pos);
        return new_pos;
    }
    return pix_pos;
}

fn fill_workgroup_mem(direction: vec2i, workgroup_pos: vec2i, local_pos: vec2i, pixel_pos: vec2i, kernel_radius: i32, io_ratio: vec2i) {
    let otherDirection = direction.yx;
  // -1 for the first pixel of the kernel (the center), who is not repeated
    let workgroupReadPixelsCount = workgroup_size + (kernel_radius - 1) * 2;
    let workgroupPosInDirection = dot(workgroup_pos, direction);
    let workgroupFirstReadPixel = workgroupPosInDirection - kernel_radius + 1;
  // how many pixels will each thread read from the input texture (sampler allows reading outside the texture)
    let memberPixelCount = i32(ceil(f32(workgroupReadPixelsCount) / f32(workgroup_size)));
    let localPosInDirection = dot(local_pos, direction);
    let myFirstWritePixel = localPosInDirection * memberPixelCount;
    let myFirstInputPixel = workgroupFirstReadPixel + localPosInDirection * memberPixelCount;
    let myFirstInputPosition = myFirstInputPixel * direction + pixel_pos * otherDirection;
    let inputSize = vec2i(textureDimensions(inputTexture, parameters.from_mip));
    for (var i = 0; i < memberPixelCount; i++) {
        let samplePos = (myFirstInputPosition + direction * i) * io_ratio;
        if all(samplePos >= vec2i(0, 0)) && all(samplePos < inputSize) {
            let texel = read_input(samplePos);
            workgroupPixels[myFirstWritePixel + i] = u32(texel.r);
        }
    }
    workgroupBarrier();
}

fn read_workgroup_mem(pix_pos: vec2i, workgroup_pos: vec2i, direction: vec2i, kernel_radius: i32) -> u32 {
    let sample_pos = dot((pix_pos - workgroup_pos), direction) + kernel_radius - 1;
    return workgroupPixels[sample_pos];
}

fn to_gray(texel: vec4f) -> vec4f {
    let gray = dot(vec3f(0.299, 0.587, 0.114), texel.rgb);
    return vec4f(vec3f(gray), texel.a);
}

@compute @workgroup_size(workgroup_size, 1)
fn single_pass_radial(@builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>) {
    let kernel_radius = i32(arrayLength(&kernel));
    var pixel_pos = vec2i(global_id.xy);
    var workgroup_pos = vec2i(workgroup_id.xy) * workgroup_size;
    var local_pos = vec2i(local_id.xy);
    var direction = vec2i(1, 0);
    if parameters.horizontal == 0 {
        pixel_pos = pixel_pos.yx;
        workgroup_pos = workgroup_pos.yx;
        local_pos = local_pos.yx;
        direction = direction.yx;
    }
    let output_size = vec2i(textureDimensions(gaussian_textures));
    let inputSize = vec2i(textureDimensions(inputTexture, parameters.from_mip));
    // sift only jumps by a ratio of 2, and samples the bigger input texture at every other textel
    // we will compute everything in the output sampling space, and just scale the input coords by io_ratio
    let io_ratio = inputSize / output_size;
    let otherDirection = direction.yx;

  // *** 1) put some input pixels in workgroup memory
    if use_workgroup_mem != 0 {
        fill_workgroup_mem(direction, workgroup_pos, local_pos, pixel_pos, kernel_radius, io_ratio);
    }

  // *** 2) compute the 1D kernel
    if any(pixel_pos >= output_size) {
        return;
    }

    let localPosInDirection = dot(local_pos, direction);
    let myWorkgroupPixelOffset = localPosInDirection + kernel_radius - 1;
    var sum = 0.0;
    // used for the border pixels whose mask is not the full kernel
    var weightSum = 0.0;
    for (var i = -kernel_radius + 1; i < kernel_radius; i++) {
        let pix_read_pos = handle_border(pixel_pos + direction * i, output_size);
        // using outputsize because the workgroup mem is at outputSize sampling rate
        if all(pix_read_pos >= vec2i(0, 0)) && all(pix_read_pos < output_size) {
            let weight = kernel[abs(i)];
            var texel: u32;
            if use_workgroup_mem != 0 {
                texel = read_workgroup_mem(pix_read_pos, workgroup_pos, direction, kernel_radius);
            } else {
                texel = u32(read_input(pix_read_pos * io_ratio).r);
            }
            sum += f32(texel) * weight;
            weightSum += weight;
        }
    }
    if weightSum == 0.0 {
        textureStore(gaussian_textures, pixel_pos, vec4i(0, 0, 0, 0));
    } else {
        var result: f32;
        result = sum / weightSum;
        textureStore(gaussian_textures, pixel_pos, vec4i(i32(result)));
    }
}

// copy or resize
@compute @workgroup_size(workgroupxy_size, workgroupxy_size, 1)
fn copy(@builtin(global_invocation_id) global_id: vec3<u32>) {
    var pixel_pos = vec2i(global_id.xy);
    let output_size = vec2i(textureDimensions(gaussian_textures));
    let input_size = vec2i(textureDimensions(inputTexture, parameters.from_mip));
    if all(pixel_pos >= vec2i(0, 0)) && all(pixel_pos < output_size) {
        let texel = read_input(pixel_pos * input_size / output_size);
        textureStore(gaussian_textures, pixel_pos, texel);
    }
}

@compute @workgroup_size(workgroupxy_size, workgroupxy_size, 1)
fn subtract(@builtin(global_invocation_id) global_id: vec3<u32>,
        @builtin(workgroup_id) workgroup_id: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>) {
    let pixel_pos = vec2i(global_id.xy);
    let array_index = global_id.z;
    if all(pixel_pos >= vec2i(0)) && all(pixel_pos < vec2i(textureDimensions(diff_input_stack, parameters.from_mip)) ) {
        let pix1 = textureLoad(diff_input_stack, pixel_pos, array_index, parameters.from_mip);
        let pix2 = textureLoad(diff_input_stack, pixel_pos, array_index + 1, parameters.from_mip);
        textureStore(diff_output_stack, pixel_pos, array_index, vec4i(pix2.r - pix1.r));
    }
}

@compute @workgroup_size(workgroupxy_size, workgroupxy_size, 1)
fn convert_to_gray(@builtin(global_invocation_id) global_id: vec3<u32>,
        @builtin(workgroup_id) workgroup_id: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>) {
    let pixel_pos = vec2i(global_id.xy);
    if all(pixel_pos >= vec2i(0)) && all(pixel_pos < vec2i(textureDimensions(input_rgba, parameters.from_mip))) {
        let pix = textureLoad(input_rgba, pixel_pos, parameters.from_mip);
        let gray = to_gray(pix);
        textureStore(output_gray, pixel_pos, vec4i(i32(gray.r * MAX_2_31)));
    }
}

@compute @workgroup_size(workgroupxy_size, workgroupxy_size, 1)
fn convert_from_gray(@builtin(global_invocation_id) global_id: vec3<u32>,
        @builtin(workgroup_id) workgroup_id: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>) {
    let pixel_pos = vec2i(global_id.xy);
    if all(pixel_pos >= vec2i(0)) && all(pixel_pos < vec2i(textureDimensions(inputTexture, parameters.from_mip))) {
        let pix = textureLoad(inputTexture, pixel_pos, parameters.from_mip);
        var luminance = f32(pix.r) / MAX_2_31;
        if parameters.from_gray_negative != 0 {
            luminance = luminance / 2.0 + 0.5;
        }
        textureStore(output_rgba, pixel_pos, vec4f(luminance, luminance, luminance, 1.0));
    }
}

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
    let texture_size = vec2i(textureDimensions(diff_output_stack));
    let v_border = vec2i(parameters.extrema_border);
    let array_index = i32(global_id.z) + 1;
    let current_texel = textureLoad(diff_output_stack, pixel_pos, array_index).r;
    let wg_linearization = vec3u(1u, num_workgroups.x, num_workgroups.x * num_workgroups.y);
    let wg_linear_index = dot(workgroup_id.xyz, wg_linearization);
    if all(pixel_pos >= v_border) && all(pixel_pos < texture_size - v_border) {
        var current_min_max = vec2i(0);
        for (var i = 0; i < NEIGHBOR_COUNT; i++) {
            let neighbor = neighbors[i];
            let texel = vec2i(textureLoad(diff_output_stack, pixel_pos + neighbor.xy, array_index + neighbor.z).r);
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
