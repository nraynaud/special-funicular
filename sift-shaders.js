// SIFT WebGPU Shader Definitions
// This module contains shared shader code used by both the worker and test files

// SIFT implementation constants
import {
  makeShaderDataDefinitions, makeStructuredView
} from './lib/webgpu-utils.module.js'

export const NUM_OCTAVES = 4
export const SCALES_PER_OCTAVE = 5
export const SIGMA_INITIAL = 1.6
export const SIGMA_MULTIPLIER = Math.sqrt(2)
export const CONTRAST_THRESHOLD = 0.001 // Reduced threshold to detect more features
export const EDGE_THRESHOLD = 5.0
export const MAX_KEYPOINTS = 10000

export const HORIZONTAL = 1
export const VERTICAL = 2

class AllocatedRadialShader {
  constructor (shader) {
    this.shader = shader
  }

  static async createGPUResources (shader, pipelineDesc, inputImage, outputWidth, outputHeight, kernels, oneDirection = null) {
    console.assert(oneDirection == null || kernels.length === 1, `can use oneDirection parameter only when there is only one kernel, found ${kernels.length} kernels`)
    const resources = new AllocatedRadialShader(shader)
    resources.device = shader.device
    resources.pipeline = shader.device.createComputePipeline(pipelineDesc)
    resources.pipelineDesc = pipelineDesc
    resources.kernelBuffers = kernels.map(k => {
      const kernelBuffer = shader.device.createBuffer({
        size: k.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      })
      shader.device.queue.writeBuffer(kernelBuffer, 0, k)
      return kernelBuffer
    })
    resources.outputWidth = outputWidth
    resources.outputHeight = outputHeight
    resources.uniformsView = shader.uniformsView
    const mipLevels = Math.floor(Math.log2(Math.min(outputWidth, outputHeight)))
    console.log('mipLevels', mipLevels)
    resources.outputTexture = shader.device.createTexture({
      label: 'output',
      size: [outputWidth, outputHeight, kernels.length + 1],
      mipLevelCount: mipLevels,
      format: 'rgba8unorm',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC | GPUTextureUsage.COPY_DST | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT
    })
    resources.tempTexture = shader.device.createTexture({
      label: 'temp',
      size: [outputWidth, outputHeight],
      mipLevelCount: mipLevels,
      format: 'rgba8unorm',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC | GPUTextureUsage.COPY_DST | GPUTextureUsage.TEXTURE_BINDING
    })
    const copySize = [inputImage.width, inputImage.height, 1]
    // should handle both ImageBitmap and ImageData
    shader.device.queue.copyExternalImageToTexture({source: inputImage}, {
      texture: resources.outputTexture, origin: [0, 0, 0]
    }, copySize)
    resources.tempView = resources.tempTexture.createView({dimension: '2d'})
    resources.tempViewStorage = []
    for (let mip = 0; mip < resources.tempTexture.mipLevelCount; mip++) {
      resources.tempViewStorage.push(resources.tempTexture.createView({
        dimension: '2d',
        mipLevelCount: 1,
        baseMipLevel: mip
      }))
    }
    resources.outViewsStorage = []
    resources.outViewMipmap = []
    for (let i = 0; i < resources.outputTexture.depthOrArrayLayers; i++) {
      resources.outViewsStorage.push([])
      resources.outViewMipmap.push(resources.outputTexture.createView({
        dimension: '2d', baseArrayLayer: i
      }))
      for (let j = 0; j < resources.outputTexture.mipLevelCount; j++) {
        resources.outViewsStorage[i].push(resources.outputTexture.createView({
          dimension: '2d', baseArrayLayer: i, mipLevelCount: 1, baseMipLevel: j
        }))
      }
    }
    const bytesPerRow = Math.ceil((outputWidth * 4) / 256) * 256
    resources.outputBuffer = shader.device.createBuffer({
      size: bytesPerRow * outputHeight, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
    })
    const workgroupSize = resources.pipelineDesc.compute.constants.workgroup_size
    const horizontalWorkGroups = [Math.ceil(resources.outputWidth / workgroupSize), resources.outputHeight]
    const verticalWorkGroups = [Math.ceil(resources.outputHeight / workgroupSize), resources.outputWidth]
    if (kernels.length === 1) {
      const callHorizontal = oneDirection == null || oneDirection === HORIZONTAL
      const callVertical = oneDirection == null || oneDirection === VERTICAL
      resources.encodedCommands = await resources.encodeSinglePass(callHorizontal, callVertical, horizontalWorkGroups, verticalWorkGroups)
    } else {
      resources.encodedCommands = await resources.encodeRepeatedPasses(workgroupSize)
    }
    return resources
  }

  createUniformBuffer (values) {
    const buff = this.device.createBuffer({
      size: this.uniformsView.arrayBuffer.byteLength, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })

    this.uniformsView.set(values)
    this.device.queue.writeBuffer(buff, 0, this.uniformsView.arrayBuffer)
    return buff
  }

  async encodeRepeatedPasses (workgroupSize) {
    const commandEncoder = this.device.createCommandEncoder()
    const computePass = commandEncoder.beginComputePass({
      label: 'Gaussian compute pass',
    })
    computePass.setPipeline(this.pipeline)
    for (let mipLevel = 0; mipLevel < this.outputTexture.mipLevelCount; mipLevel++) {
      let inputMipLevel = mipLevel === 0 ? 0 : mipLevel - 1
      let inputTexture = mipLevel === 0 ? this.outViewMipmap[0] : this.outViewMipmap[this.outViewMipmap.length - 3]
      const mipsRatio = 2 ** mipLevel
      const [outputWidth, outputHeight] = [this.outputWidth, this.outputHeight].map(d => Math.floor(d / mipsRatio))
      const horizontalWorkGroups = [Math.ceil(outputWidth / workgroupSize), outputHeight]
      const verticalWorkGroups = [Math.ceil(outputHeight / workgroupSize), outputWidth]
      for (const [index, kernel] of this.kernelBuffers.entries()) {
        console.log('kernel index: ' + index, 'mip', mipLevel)
        let inputParamBuffer = this.createUniformBuffer({
          horizontal: 1,
          from_mip: inputMipLevel
        })
        computePass.setBindGroup(0, this.device.createBindGroup({
          label: 'Gauss bind group 0',
          layout: this.pipeline.getBindGroupLayout(0),
          entries: [{binding: 0, resource: inputTexture},
            {binding: 1, resource: this.tempViewStorage[mipLevel]},
            {binding: 2, resource: {buffer: inputParamBuffer}},
            {binding: 3, resource: {buffer: kernel}},]
        }))
        computePass.dispatchWorkgroups(...horizontalWorkGroups)
        inputParamBuffer = this.createUniformBuffer({
          horizontal: 0,
          from_mip: mipLevel
        })
        computePass.setBindGroup(0, this.device.createBindGroup({
          label: 'Gauss bind group 0',
          layout: this.pipeline.getBindGroupLayout(0),
          entries: [{binding: 0, resource: this.tempView},
            {binding: 1, resource: this.outViewsStorage[index][mipLevel]},
            {binding: 2, resource: {buffer: inputParamBuffer}},
            {binding: 3, resource: {buffer: kernel}}]
        }))
        computePass.dispatchWorkgroups(...verticalWorkGroups)
        inputTexture = this.outViewMipmap[index]
        inputMipLevel = mipLevel
      }
    }
    computePass.end()
    const copyMipLevel = 0
    const mipRatio = 2 ** copyMipLevel
    const bytesPerRow = Math.ceil((this.outputWidth * 4) / 256) * 256
    commandEncoder.copyTextureToBuffer({texture: this.outputTexture, origin: [0, 0, 4], mipLevel: copyMipLevel},
      {buffer: this.outputBuffer, bytesPerRow},
      [Math.floor(this.outputWidth / mipRatio), Math.floor(this.outputHeight / mipRatio), 1])
    return commandEncoder.finish()
  }

  async encodeSinglePass (callHorizontal, callVertical, horizontalWorkGroups, verticalWorkGroups) {
    const commandEncoder = this.device.createCommandEncoder()
    // Calculate bytesPerRow, which must be a multiple of 256 for WebGPU
    const bytesPerRow = Math.ceil((this.outputWidth * 4) / 256) * 256
    const computePass = commandEncoder.beginComputePass({
      label: 'Gaussian compute pass',
    })
    computePass.setPipeline(this.pipeline)
    let vInput
    let vOutput = this.outViewsStorage[1][0]
    let outCopyOrig
    if (callHorizontal) {
      computePass.setBindGroup(0, this.device.createBindGroup({
        label: 'Gauss bind group 0',
        layout: this.pipeline.getBindGroupLayout(0),
        entries: [{binding: 0, resource: this.outViewsStorage[0][0]},
          {binding: 1, resource: this.tempViewStorage[0]}, {
            binding: 2, resource: {
              buffer: this.createUniformBuffer({
                horizontal: 1,
                from_mip: 0
              })
            }
          }, {binding: 3, resource: {buffer: this.kernelBuffers[0]}},]
      }))
      computePass.dispatchWorkgroups(...horizontalWorkGroups)
      outCopyOrig = {texture: this.tempTexture, origin: [0, 0, 0]}
      vInput = this.tempView
    } else {
      vInput = this.outViewsStorage[0][0]
      outCopyOrig = {texture: this.outputTexture, origin: [0, 0, 0]}
    }
    if (callVertical) {
      computePass.setBindGroup(0, this.device.createBindGroup({
        label: 'Gauss bind group 0',
        layout: this.pipeline.getBindGroupLayout(0),
        entries: [{binding: 0, resource: vInput}, {binding: 1, resource: vOutput}, {
          binding: 2, resource: {
            buffer: this.createUniformBuffer({
              horizontal: 0,
              from_mip: 0
            })
          }
        }, {binding: 3, resource: {buffer: this.kernelBuffers[0]}},]
      }))

      computePass.dispatchWorkgroups(...verticalWorkGroups)
      outCopyOrig = {texture: this.outputTexture, origin: [0, 0, 1]}
    }
    computePass.end()
    commandEncoder.copyTextureToBuffer(outCopyOrig, {
      buffer: this.outputBuffer, bytesPerRow
    }, [this.outputWidth, this.outputHeight, 1])
    return commandEncoder.finish()
  }

  async getOutputTexture (index, mipLevel) {
    const commandEncoder = this.device.createCommandEncoder()
    const mipRatio = 2 ** mipLevel
    let outW = Math.floor(this.outputWidth / mipRatio)
    let outH = Math.floor(this.outputHeight / mipRatio)
    const bytesPerRow = Math.ceil((outW * 4) / 256) * 256
    commandEncoder.copyTextureToBuffer({texture: this.outputTexture, origin: [0, 0, index], mipLevel: mipLevel},
      {buffer: this.outputBuffer, bytesPerRow},
      [outW, outH, 1])
    this.device.queue.submit([commandEncoder.finish()])
    await this.device.queue.onSubmittedWorkDone()
    await this.outputBuffer.mapAsync(GPUMapMode.READ)
    try {
      const outputData = new Uint8ClampedArray(this.outputBuffer.getMappedRange()).slice(0, bytesPerRow * outH)
      console.log('out buffer length: ', outputData.byteLength, bytesPerRow * outH, bytesPerRow, bytesPerRow / 4, outH)
      return createImageBitmap(new ImageData(outputData, bytesPerRow / 4, outH), 0, 0, outW, outH)
    } finally {
      this.outputBuffer.unmap()
    }
  }

  async runShader () {
    console.time('runShader')
    const bytesPerRow = Math.ceil((this.outputWidth * 4) / 256) * 256
    const device = this.shader.device

    console.log('Submitting command buffer to GPU queue', this.encodedCommands)
    device.queue.submit([this.encodedCommands])
    await device.queue.onSubmittedWorkDone()

    await this.outputBuffer.mapAsync(GPUMapMode.READ)
    try {
      console.time('result copy')
      const outputData = new Uint8ClampedArray(this.outputBuffer.getMappedRange()).slice()
      let outputImage = new ImageData(outputData, bytesPerRow / 4, this.outputHeight)
      console.timeEnd('result copy')
      return outputImage
    } finally {
      this.outputBuffer.unmap()
      console.timeEnd('runShader')
    }
  }
}

export class RadialShader {

  constructor (device, pipelineDesc, defs, module) {
    this.device = device
    this.pipelineDesc = pipelineDesc
    this.defs = defs
    this.module = module
  }

  static async createShader (device) {
    const shaderCode = await (await fetch('radial.wgsl')).text()
    const defs = makeShaderDataDefinitions(shaderCode)
    console.log('DEF', defs)
    const pipelineDescs = Object.fromEntries(Object.entries(defs.entryPoints).map(([key, v], _) =>
      [key, {label: `${key} pipeline`, layout: 'auto', compute: {entryPoint: key, constants: {}}}]))
    console.log(pipelineDescs)
    // partial pipeline to generate layout
    const pipelineDesc = {
      label: 'Gaussian Blur Pipeline Test', layout: 'auto', compute: {
        entryPoint: 'single_pass_radial', constants: {},
      }
    }
    const module = device.createShaderModule({
      label: 'Gaussian Blur Shader Test', code: shaderCode,
    })
    let shader = new RadialShader(device, pipelineDesc, defs, module)
    shader.uniformsView = makeStructuredView(shader.defs.uniforms.parameters)
    return shader
  }

  async createGPUResources (workgroupSize, inputImage, outputWidth, outputHeight, kernels, oneDirection = null) {
    this.pipelineDesc.compute.constants.workgroup_size = workgroupSize
    let pipelineDesc = structuredClone(this.pipelineDesc)
    console.log('pipelineDesc', JSON.stringify(pipelineDesc))
    pipelineDesc.compute.module = this.module
    return AllocatedRadialShader.createGPUResources(this, pipelineDesc, inputImage, outputWidth, outputHeight, kernels, oneDirection)
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

