override workgroup_size = 64;
const use_workgroup_mem = 1;
override workgroup_pixel_count = 16300/4;

struct Params {
    horizontal: u32,
    from_mip: u32,
    convert_to_gray: u32,
    diff_index: u32
};

@group(0) @binding(0) var inputTexture: texture_2d<f32>;
@group(0) @binding(1) var outputTexture: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(2) var<uniform> parameters: Params;
@group(0) @binding(3) var<storage> kernel: array<f32>;
@group(0) @binding(4) var diff_input_stack: texture_2d_array<f32>;
@group(0) @binding(5) var diff_output_stack: texture_storage_2d_array<rgba8unorm, write>;

var<workgroup> workgroupPixels: array<u32, workgroup_pixel_count>;
var<workgroup> current_kernel_radius: u32;

fn read_input(pix_pos: vec2i) -> vec4f {
    return textureLoad(inputTexture, pix_pos, parameters.from_mip);
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
            workgroupPixels[myFirstWritePixel + i] = pack4x8unorm(texel);
        }
    }
    workgroupBarrier();
}

fn read_workgroup_mem(pix_pos: vec2i, workgroup_pos: vec2i, direction: vec2i, kernel_radius: i32) -> vec4f {
    let sample_pos = dot((pix_pos - workgroup_pos), direction) + kernel_radius - 1;
    return unpack4x8unorm(workgroupPixels[sample_pos]);
}

fn to_gray(texel: vec4f) -> vec4f {
    let gray = dot(vec3f(0.299, 0.587, 0.114), texel.rgb);
    return vec4f(vec3f(gray), texel.a);
}

@compute @workgroup_size(workgroup_size, 1)
fn single_pass_radial(@builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>) {
    current_kernel_radius = arrayLength(&kernel);
    let kernel_radius = i32(workgroupUniformLoad(&current_kernel_radius));
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
    let outputSize = vec2i(textureDimensions(outputTexture));
    let inputSize = vec2i(textureDimensions(inputTexture, parameters.from_mip));
    // sift only jumps by a ratio of 2, and samples the bigger input texture at every other textel
    // we will compute everything in the output sampling space, and just scale the input coords by io_ratio
    let io_ratio = inputSize / outputSize;
    let otherDirection = direction.yx;

  // *** 1) put some input pixels in workgroup memory
    if use_workgroup_mem != 0 {
        fill_workgroup_mem(direction, workgroup_pos, local_pos, pixel_pos, kernel_radius, io_ratio);
    }

  // *** 2) compute the 1D kernel
    if any(pixel_pos >= outputSize) {
        return;
    }

    let localPosInDirection = dot(local_pos, direction);
    let myWorkgroupPixelOffset = localPosInDirection + kernel_radius - 1;
    var sum = vec4<f32>(0.0);
    var weightSum = 0.0;
    for (var i = -kernel_radius + 1; i < kernel_radius; i++) {
        let pixReadPos = pixel_pos + direction * i;
        // using outputsize because the workgroup mem is at outputSize sampling rate
        if all(pixReadPos >= vec2i(0, 0)) && all(pixReadPos < outputSize) {
            let weight = kernel[abs(i)];
            var texel: vec4f;
            if use_workgroup_mem != 0 {
                texel = read_workgroup_mem(pixReadPos, workgroup_pos, direction, kernel_radius);
            } else {
                texel = read_input(pixReadPos * io_ratio);
            }
            if parameters.convert_to_gray != 0 {
                texel = to_gray(texel);
            }
            sum += texel * weight;
            weightSum += weight;
        }
    }
    if weightSum == 0.0 {
        textureStore(outputTexture, pixel_pos, vec4f(1.0, 0.0, 0.0, 1.0));
    } else {
        var result: vec4<f32>;
        result = sum / weightSum;
        result.a = 1.0;
        textureStore(outputTexture, pixel_pos, result);
    }
}

@compute @workgroup_size(8, 8, 1)
fn subtract(@builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>) {
    let pixel_pos = vec2i(global_id.xy);
    if all(pixel_pos >= vec2i(0)) && all(pixel_pos < vec2i(textureDimensions(diff_input_stack, parameters.from_mip)) ) {
        let pix1 = textureLoad(diff_input_stack, pixel_pos, parameters.diff_index, parameters.from_mip);
        let pix2 = textureLoad(diff_input_stack, pixel_pos, parameters.diff_index+1, parameters.from_mip);
        var diff = (pix1 - pix2);
        diff.a = 1.0;
        textureStore(diff_output_stack, pixel_pos, parameters.diff_index, diff);
    }
}
