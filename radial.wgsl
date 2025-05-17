override workgroup_size = 64;
override kernel_radius = 20;
const use_workgroup_mem = 1;
override workgroup_pixel_count = workgroup_size + 2 * (kernel_radius - 1);
@group(0) @binding(0) var outputTexture: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(1) var diffuseSampler: sampler;
@group(0) @binding(2) var inputTexture: texture_2d<f32>;
@group(0) @binding(3) var<uniform> horizontal: i32;
@group(0) @binding(4) var<storage, read> kernel: array<f32>;
var<workgroup> workgroupPixels: array<u32, workgroup_pixel_count>;

fn fill_workgroup_mem(direction: vec2i, workgroup_pos: vec2i, local_pos: vec2i, pixel_pos: vec2i) {
  let otherDirection = direction.yx;
  let inputSize = vec2i(textureDimensions(inputTexture));
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
  for (var i = 0; i < memberPixelCount; i++) {
    let samplePos = myFirstInputPosition + direction * i;
    if (all(samplePos >= vec2i(0, 0)) && all(samplePos < inputSize)) {
      // +0.5 to get to textel center
      let samplePosf = (vec2f(samplePos)+0.5) / vec2f(inputSize);
      let texel = textureSampleLevel(inputTexture, diffuseSampler, samplePosf, 0);
      workgroupPixels[myFirstWritePixel+i] = pack4x8unorm(texel);
    }
  }
  workgroupBarrier();
}

fn read_workgroup_mem(pixReadPos: vec2i, workgroup_pos: vec2i, direction: vec2i) -> vec4f {
  let samplePos = dot((pixReadPos - workgroup_pos) , direction) + kernel_radius - 1;
  return unpack4x8unorm(workgroupPixels[samplePos]);
}

@compute @workgroup_size(workgroup_size, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>) {
  var pixel_pos = vec2i(global_id.xy);
  var workgroup_pos = vec2i(workgroup_id.xy) * workgroup_size;
  var local_pos = vec2i(local_id.xy);
  var direction = vec2i(1, 0);
  if (horizontal == 0) {
    pixel_pos = pixel_pos.yx;
    workgroup_pos = workgroup_pos.yx;
    local_pos = local_pos.yx;
    direction = direction.yx;
  }
  let outputSize = vec2i(textureDimensions(outputTexture));
  let inputSize = vec2i(textureDimensions(inputTexture));
  let otherDirection = direction.yx;

  // *** 1) put some input pixels in workgroup memory
  if (use_workgroup_mem != 0) {
    fill_workgroup_mem(direction, workgroup_pos, local_pos, pixel_pos);
  }

  // *** 2) compute the 1D kernel
  if (any(pixel_pos >= outputSize)) {
    return;
  }

  let localPosInDirection = dot(local_pos, direction);
  let myWorkgroupPixelOffset = localPosInDirection + kernel_radius - 1;
  var sum = vec4<f32>(0.0);
  var weightSum = 0.0;
  for (var i = -kernel_radius + 1; i < kernel_radius; i++) {
    let pixReadPos = pixel_pos + direction * i;
    if (all(pixReadPos >= vec2i(0, 0)) && all(pixReadPos < inputSize)) {
      let weight = kernel[abs(i)];
      var texel: vec4f;
      if (use_workgroup_mem != 0) {
        texel = read_workgroup_mem(pixReadPos, workgroup_pos, direction);
      } else {
        texel = textureSampleLevel(inputTexture, diffuseSampler, (vec2f(pixReadPos) + vec2f(0.5)) / vec2f(inputSize), 0);
      }
      sum += texel * weight;
      weightSum += weight;
    }
  }
  if (weightSum == 0.0) {
    textureStore(outputTexture, pixel_pos, vec4f(1.0, 0.0, 0.0, 1.0));
  } else {
    var result: vec4<f32>;
    result = sum / weightSum;
    result.a = 1.0;
    textureStore(outputTexture, pixel_pos, result);
  }
}
