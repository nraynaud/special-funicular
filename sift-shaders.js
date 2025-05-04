// SIFT WebGPU Shader Definitions
// This module contains shared shader code used by both the worker and test files

// SIFT implementation constants
export const NUM_OCTAVES = 4;
export const SCALES_PER_OCTAVE = 5;
export const SIGMA_INITIAL = 1.6;
export const SIGMA_MULTIPLIER = Math.sqrt(2);
export const CONTRAST_THRESHOLD = 0.001; // Reduced threshold to detect more features
export const EDGE_THRESHOLD = 5.0;
export const MAX_KEYPOINTS = 10000;

// language=WGSL
export const radialKernelShader = `
override workgroup_size = 64;
override kernel_radius = 20;
override workgroup_pixel_count = workgroup_size + 2 * (kernel_radius - 1);
@group(0) @binding(0) var outputTexture: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(2) var inputTexture: texture_2d<f32>;
@group(0) @binding(3) var<uniform> horizontal: i32;
@group(0) @binding(4) var<storage, read> kernel: array<f32>;
var<workgroup> workgroupPixels: array<u32, workgroup_pixel_count>;

@compute @workgroup_size(workgroup_size, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>, 
    @builtin(workgroup_id) workgroup_id: vec3<u32>, 
    @builtin(local_invocation_id) local_id: vec3<u32>) {
  var pixel_pos = vec2i(global_id.xy);
  var workgroup_pos = vec2i(workgroup_id.xy);
  var local_pos = vec2i(local_id.xy);
  var direction = vec2i(1, 0);
  if (horizontal == 0) {
    pixel_pos = pixel_pos.yx;
    workgroup_pos = workgroup_pos.yx;
    local_pos = local_pos.yx;
    direction = direction.yx;
  }
  let outputSize = vec2i(textureDimensions(outputTexture));
  let otherDirection = direction.yx;

  // *** 1) put some input pixels in workgroup memory
  // -1 for the first pixel of the kernel (the center), who is not repeated
  let workgroupReadPixelsCount = workgroup_size + (kernel_radius - 1) * 2;
  let workgroupPosInDirection = dot(workgroup_pos, direction);
  let workgroupFirstReadPixel = workgroup_size * workgroupPosInDirection - kernel_radius + 1;
  // how many pixels will each thread read from the input texture (sampler allows reading outside the texture)
  let memberPixelCount = i32(ceil(f32(workgroupReadPixelsCount) / f32(workgroup_size)));
  let localPosInDirection = dot(local_pos, direction);
  let myFirstWritePixel = localPosInDirection * memberPixelCount;
  let myFirstInputPixel = workgroupFirstReadPixel + localPosInDirection * memberPixelCount;
  let myFirstInputPosition = myFirstInputPixel * direction + pixel_pos * otherDirection;
  for (var i = 0; i < memberPixelCount; i++) {
    let samplePos = myFirstInputPosition + direction * i;
    if (all(samplePos >= vec2i(0, 0)) && all(samplePos < outputSize)) {
      let samplePosf = vec2f(samplePos) / vec2f(outputSize);
      let texel = textureLoad(inputTexture, samplePos, 0);
      workgroupPixels[myFirstWritePixel+i] = pack4x8unorm(texel);
    }
  }
  workgroupBarrier();

  // *** 2) compute the 1D kernel
  if (pixel_pos.x >= outputSize.x || pixel_pos.y >= outputSize.y) {
    return;
  }
  let myWorkgroupPixelOffset = localPosInDirection + kernel_radius - 1;
  var sum = vec4<f32>(0.0);
  var weightSum = 0.0;
  for (var i = -kernel_radius + 1; i < kernel_radius; i++) {
    let pixReadPos = pixel_pos + direction * i;
    if (all(pixReadPos >= vec2i(0, 0)) && all(pixReadPos < outputSize)) {
      let weight = kernel[abs(i)];
      let samplePos = myWorkgroupPixelOffset + i;
      let texel = unpack4x8unorm(workgroupPixels[samplePos]);
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
`

export const gaussianBlurShader = `
@group(0) @binding(0) var outputTexture: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(1) var diffuseSampler: sampler;
@group(0) @binding(2) var inputTexture: texture_2d<f32>;
@group(0) @binding(3) var<uniform> params: GaussianParams;

struct GaussianParams {
  sigma: f32,
  direction: vec2<f32>,
  imageSize: vec2<f32>,
}

// Compute Gaussian weight for a given distance and sigma
fn gaussian(x: f32, sigma: f32) -> f32 {
  let sigmaSq = sigma * sigma;
  // Ensure we're getting a non-zero weight
  return max(0.000001, (1.0 / sqrt(2.0 * 3.14159 * sigmaSq)) * exp(-(x * x) / (2.0 * sigmaSq)));
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let imageSize = vec2<i32>(params.imageSize.xy);
  let pixel_pos = vec2<i32>(global_id.xy);
  if (pixel_pos.x >= imageSize.x || pixel_pos.y >= imageSize.y) {
    return;
  }

  // First, read the center pixel to ensure we have a valid starting point
  let centerTexel = textureLoad(inputTexture, pixel_pos, 0);
  // Determine kernel radius based on sigma (3*sigma covers >99% of Gaussian)
  let kernelRadius = i32(ceil(3.0 * params.sigma));
  var sum = vec4<f32>(0.0);
  var weightSum = 0.0;
  for (var i = -kernelRadius; i <= kernelRadius; i++) {
    let weight = gaussian(f32(i), params.sigma);
    let samplePos = (vec2<f32>(pixel_pos) + params.direction*f32(i)) / params.imageSize;
    let texel = textureSampleLevel(inputTexture, diffuseSampler, samplePos , 0);
    sum += texel * weight;
    weightSum += weight;
  }
  // Normalize by weight sum and ensure we don't divide by zero
  var result: vec4<f32>;
  if (weightSum > 0.0) {
    result = sum / weightSum;
  } else {
    result = centerTexel;
  }
  result.a = 1.0;
  textureStore(outputTexture, pixel_pos, result);
}
`;

// WebGPU shader for DoG (Difference of Gaussians)
export const dogShader = `
@group(0) @binding(0) var texture1: texture_2d<f32>;
@group(0) @binding(1) var texture2: texture_2d<f32>;
@group(0) @binding(2) var outputTexture: texture_storage_2d<rgba8unorm, write>;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let pixel_pos = vec2<i32>(global_id.xy);
  let dimensions = vec2<i32>(textureDimensions(texture1));

  // Check if within bounds
  if (pixel_pos.x >= dimensions.x || pixel_pos.y >= dimensions.y) {
    return;
  }

  let texel1 = textureLoad(texture1, pixel_pos, 0);
  let texel2 = textureLoad(texture2, pixel_pos, 0);

  // Calculate difference of Gaussians
  let diff = texel1 - texel2;

  // Store the difference in the R channel (used by keypoint detection)
  // Calculate the luminance of the difference to get a single value
  let luminance = dot(diff.rgb, vec3<f32>(0.299, 0.587, 0.114));
  textureStore(outputTexture, pixel_pos, vec4<f32>(luminance, 0.0, 0.0, 1.0));
}
`;


// WebGPU shader for visualizing keypoints
export const visualizeKeypointsShader = `
@group(0) @binding(0) var inputTexture: texture_2d<f32>;
@group(0) @binding(1) var outputTexture: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(2) var<storage, read> keypoints: array<Keypoint>;
@group(0) @binding(3) var<uniform> params: VisualizeParams;

struct Keypoint {
  position: vec2<f32>,
  scale: f32,
  orientation: f32,
  response: f32,
  octave: i32,
}

struct VisualizeParams {
  keypointCount: u32,
  circleColor: vec4<f32>,
  lineWidth: f32,
}

// Draw a circle at the keypoint position with radius proportional to scale
fn drawKeypoint(pos: vec2<i32>, kp: Keypoint) -> vec4<f32> {
  let kpPos = vec2<i32>(kp.position);
  let dist = distance(vec2<f32>(pos), kp.position);
  let radius = kp.scale * 2.0;

  // Draw circle outline
  if (abs(dist - radius) < params.lineWidth) {
    return params.circleColor;
  }

  // Draw orientation line
  let angle = kp.orientation;
  let lineEnd = kp.position + vec2<f32>(cos(angle), sin(angle)) * radius;

  // Simple line drawing - could be improved
  let lineStart = kp.position;
  let lineDir = normalize(lineEnd - lineStart);
  let perpDir = vec2<f32>(-lineDir.y, lineDir.x);
  let posFloat = vec2<f32>(pos);

  let projection = dot(posFloat - lineStart, lineDir);
  let perpDistance = abs(dot(posFloat - lineStart, perpDir));

  if (projection >= 0.0 && projection <= length(lineEnd - lineStart) && perpDistance < params.lineWidth) {
    return params.circleColor;
  }

  return vec4<f32>(0.0);
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let pixel_pos = vec2<i32>(global_id.xy);
  let dimensions = vec2<i32>(textureDimensions(inputTexture));

  // Check if within bounds
  if (pixel_pos.x >= dimensions.x || pixel_pos.y >= dimensions.y) {
    return;
  }

  // Get the original image color
  let originalColor = textureLoad(inputTexture, pixel_pos, 0);

  // Ensure the image is not transparent by setting a white background
  // if the original image is too transparent
  var baseColor = originalColor;
  if (originalColor.a < 0.1) {
    baseColor = vec4<f32>(1.0, 1.0, 1.0, 1.0); // White background
  }

  var finalColor = baseColor;

  // Check if this pixel is part of any keypoint visualization
  for (var i = 0u; i < params.keypointCount; i++) {
    let kp = keypoints[i];
    let kpColor = drawKeypoint(pixel_pos, kp);

    // Alpha blending
    if (kpColor.a > 0.0) {
      finalColor = kpColor * kpColor.a + baseColor * (1.0 - kpColor.a);
    }
  }

  // Ensure the final color has full opacity
  finalColor.a = 1.0;

  textureStore(outputTexture, pixel_pos, finalColor);
}
`;
