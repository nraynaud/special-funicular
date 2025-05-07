// SIFT WebGPU Shader Definitions
// This module contains shared shader code used by both the worker and test files

// SIFT implementation constants
export const NUM_OCTAVES = 4
export const SCALES_PER_OCTAVE = 5
export const SIGMA_INITIAL = 1.6
export const SIGMA_MULTIPLIER = Math.sqrt(2)
export const CONTRAST_THRESHOLD = 0.001 // Reduced threshold to detect more features
export const EDGE_THRESHOLD = 5.0
export const MAX_KEYPOINTS = 10000

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
`

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
`

