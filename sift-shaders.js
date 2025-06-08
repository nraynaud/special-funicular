// SIFT WebGPU Shader Definitions
// This module contains shared shader code used by both the worker and test files

// SIFT implementation constants
import {
  makeBindGroupLayoutDescriptors,
  makeShaderDataDefinitions,
  makeStructuredView
} from './lib/webgpu-utils.module.js'

export const NUM_OCTAVES = 4
export const SCALES_PER_OCTAVE = 5
export const SIGMA_INITIAL = 1.6
export const SIGMA_MULTIPLIER = Math.sqrt(2)
export const CONTRAST_THRESHOLD = 0.001 // Reduced threshold to detect more features
export const EDGE_THRESHOLD = 5.0
export const MAX_KEYPOINTS = 10000

class AllocatedRadialShader {
  constructor (shader) {
    this.shader = shader
  }

  static async createGPUResources (shader, pipelineDesc, inputImage, outputWidth, outputHeight, kernelBuffer) {
    const resources = new AllocatedRadialShader(shader)
    resources.pipeline = shader.device.createComputePipeline(pipelineDesc)
    resources.pipelineDesc = pipelineDesc
    resources.kernelBuffer = kernelBuffer
    resources.outputWidth = outputWidth
    resources.outputHeight = outputHeight
    resources.uniformsView = makeStructuredView(shader.defs.uniforms.parameters)
    resources.horizontalParam = shader.device.createBuffer({
      size: resources.uniformsView.arrayBuffer.byteLength,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })
    resources.uniformsView.set({
      horizontal: 1
    })
    shader.device.queue.writeBuffer(resources.horizontalParam, 0, resources.uniformsView.arrayBuffer)
    resources.verticalParam = shader.device.createBuffer({
      size: resources.uniformsView.arrayBuffer.byteLength,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })
    resources.uniformsView.set({
      horizontal: 0
    })
    shader.device.queue.writeBuffer(resources.verticalParam, 0, resources.uniformsView.arrayBuffer)
    resources.tempTexture = shader.device.createTexture({
      label: 'temp',
      size: [outputWidth, outputHeight],
      mipLevelCount: 1,
      format: 'rgba8unorm',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC | GPUTextureUsage.COPY_DST | GPUTextureUsage.TEXTURE_BINDING
    })
    resources.outputTexture = shader.device.createTexture({
      label: 'output',
      size: [outputWidth, outputHeight, 5],
      mipLevelCount: 1,
      format: 'rgba8unorm',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC | GPUTextureUsage.COPY_DST | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT
    })
    const copySize = [inputImage.width, inputImage.height, 1]
    // should handle both ImageBitmap and ImageData
    shader.device.queue.copyExternalImageToTexture({source: inputImage}, {
      texture: resources.outputTexture,
      origin: [0, 0, 0]
    }, copySize)
    resources.tempView = resources.tempTexture.createView({dimension: '2d'})
    resources.outViews = []
    for (let i = 0; i < resources.outputTexture.depthOrArrayLayers; i++) {
      resources.outViews.push(resources.outputTexture.createView({dimension: '2d', baseArrayLayer: i}))
    }
    return resources
  }

  async runShader (callHorizontal = true, callVertical = true) {
    console.time('runShader')
    // Calculate bytesPerRow, which must be a multiple of 256 for WebGPU
    const bytesPerRow = Math.ceil((this.outputWidth * 4) / 256) * 256
    // Create a buffer to copy the texture data to
    const device = this.shader.device
    const outputBuffer = device.createBuffer({
      size: bytesPerRow * this.outputHeight,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
    })
    const querySet = device.createQuerySet({
      type: 'timestamp',
      count: 2,
    })
    const resolveBuffer = device.createBuffer({
      size: querySet.count * 8,
      usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
    })
    const resultBuffer = device.createBuffer({
      size: resolveBuffer.size,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    })

    const commandEncoder = device.createCommandEncoder()
    const workgroupSize = this.pipelineDesc.compute.constants.workgroup_size
    const horizontalWorkGroups = [Math.ceil(this.outputWidth / workgroupSize), this.outputHeight]
    const verticalWorkGroups = [Math.ceil(this.outputHeight / workgroupSize), this.outputWidth]
    console.log('verticalWorkGroups', verticalWorkGroups)
    const computePass = commandEncoder.beginComputePass({
      label: 'Gaussian compute pass',
      timestampWrites: {
        querySet,
        beginningOfPassWriteIndex: 0,
        endOfPassWriteIndex: 1,
      },
    })
    computePass.setPipeline(this.pipeline)
    let vInput
    let vOutput = this.outViews[1]
    let outCopyOrig
    if (callHorizontal) {
      computePass.setBindGroup(0, device.createBindGroup({
        label: 'Gauss bind group 0',
        layout: this.pipeline.getBindGroupLayout(0),
        entries: [
          {binding: 0, resource: this.outViews[0]},
          {binding: 1, resource: this.tempView},
          {binding: 2, resource: {buffer: this.horizontalParam}},
          {binding: 3, resource: {buffer: this.kernelBuffer}},
        ]
      }))
      computePass.dispatchWorkgroups(...horizontalWorkGroups)
      outCopyOrig = {texture: this.tempTexture, origin: [0, 0, 0]}
      vInput = this.tempView
    } else {
      vInput = this.outViews[0]
      outCopyOrig = {texture: this.outputTexture, origin: [0, 0, 0]}
    }
    if (callVertical) {
      computePass.setBindGroup(0, device.createBindGroup({
        label: 'Gauss bind group 0',
        layout: this.pipeline.getBindGroupLayout(0),
        entries: [
          {binding: 0, resource: vInput},
          {binding: 1, resource: vOutput},
          {binding: 2, resource: {buffer: this.verticalParam}},
          {binding: 3, resource: {buffer: this.kernelBuffer}},
        ]
      }))

      computePass.dispatchWorkgroups(...verticalWorkGroups)
      outCopyOrig = {texture: this.outputTexture, origin: [0, 0, 1]}
    }
    computePass.end()
    commandEncoder.copyTextureToBuffer(
      outCopyOrig,
      {buffer: outputBuffer, bytesPerRow},
      [this.outputWidth, this.outputHeight, 1]
    )
    commandEncoder.resolveQuerySet(querySet, 0, querySet.count, resolveBuffer, 0)
    commandEncoder.copyBufferToBuffer(resolveBuffer, 0, resultBuffer, 0, resultBuffer.size)
    console.log('Submitting command buffer to GPU queue')
    device.queue.submit([commandEncoder.finish()])
    await device.queue.onSubmittedWorkDone()
    await resultBuffer.mapAsync(GPUMapMode.READ)
    const times = new BigInt64Array(resultBuffer.getMappedRange())
    let gpuTime = (Number(times[1] - times[0]) / 1000).toFixed(1) + 'µs'
    console.log('GPU time', gpuTime)
    this.gpuTime = gpuTime
    resultBuffer.unmap()
    await outputBuffer.mapAsync(GPUMapMode.READ)
    try {
      console.time('result copy')
      const outputData = new Uint8ClampedArray(outputBuffer.getMappedRange()).slice()
      let outputImage = new ImageData(outputData, bytesPerRow / 4, this.outputHeight)
      console.timeEnd('result copy')
      return outputImage
    } finally {
      outputBuffer.unmap()
      outputBuffer.destroy()
      console.timeEnd('runShader')
    }
  }
}

export class RadialShader {

  constructor (device, pipelineLayout, pipelineDesc, defs, module) {
    this.device = device
    this.pipelineLayout = pipelineLayout
    this.pipelineDesc = pipelineDesc
    this.defs = defs
    this.module = module
  }

  static async createShader (device) {
    const shaderCode = await (await fetch('radial.wgsl')).text()
    const defs = makeShaderDataDefinitions(shaderCode)
    // partial pipeline to generate layout
    const pipelineDesc = {
      label: 'Gaussian Blur Pipeline Test',
      layout: 'auto',
      compute: {
        entryPoint: 'single_pass_radial',
        constants: {},
      }
    }
    const descriptors = makeBindGroupLayoutDescriptors(defs, pipelineDesc)
    const bindGroupLayouts = descriptors.map(d => device.createBindGroupLayout(d))
    const pipelineLayout = device.createPipelineLayout({bindGroupLayouts: bindGroupLayouts})
    const module = device.createShaderModule({
      label: 'Gaussian Blur Shader Test',
      code: shaderCode,
    })
    return new RadialShader(device, pipelineLayout, pipelineDesc, defs, module)
  }

  async createGPUResources (workgroupSize, inputImage, outputWidth, outputHeight, kernelBuffer) {
    this.pipelineDesc.compute.constants.workgroup_size = workgroupSize
    let pipelineDesc = structuredClone(this.pipelineDesc)
    console.log('pipelineDesc', JSON.stringify(pipelineDesc))
    pipelineDesc.layout = this.pipelineLayout
    pipelineDesc.compute.module = this.module
    return AllocatedRadialShader.createGPUResources(this, pipelineDesc, inputImage, outputWidth, outputHeight, kernelBuffer)
  }

}

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

