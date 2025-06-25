// SIFT WebGPU Shader Definitions
// This module contains shared shader code used by both the worker and test files

// SIFT implementation constants
import { makeShaderDataDefinitions, makeStructuredView } from './lib/webgpu-utils.module.js'

export const NUM_OCTAVES = 4
export const SCALES_PER_OCTAVE = 5
export const SIGMA_INITIAL = 1.6
export const SIGMA_MULTIPLIER = Math.sqrt(2)
export const CONTRAST_THRESHOLD = 0.001 // Reduced threshold to detect more features
export const EDGE_THRESHOLD = 5.0
export const MAX_KEYPOINTS = 10000

export const HORIZONTAL = 1
export const VERTICAL = 2

function createMipViewArray (texture, params = {}) {
  const result = []
  for (let i = 0; i < texture.mipLevelCount; i++) {
    result.push(texture.createView({
      mipLevelCount: 1,
      baseMipLevel: i,
      ...params
    }))
  }
  return result
}

class AllocatedRadialShader {
  constructor (shader) {
    this.shader = shader
  }

  static async createGPUResources (shader, inputImage, outputWidth, outputHeight, kernels, oneDirection = null) {
    console.assert(oneDirection == null || kernels.length === 1, `can use oneDirection parameter only when there is only one kernel, found ${kernels.length} kernels`)
    const resources = new AllocatedRadialShader(shader)
    resources.device = shader.device
    resources.pipelines = shader.pipelines
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
    resources.rgbaTexture = shader.device.createTexture({
      label: 'input',
      size: [outputWidth, outputHeight, 1],
      mipLevelCount: 1,
      format: 'rgba8unorm',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC | GPUTextureUsage.COPY_DST | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT
    })
    resources.rgbaTextureView = resources.rgbaTexture.createView()
    resources.outputTexture = shader.device.createTexture({
      label: 'output',
      size: [outputWidth, outputHeight, kernels.length],
      mipLevelCount: mipLevels,
      format: 'r32sint',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC | GPUTextureUsage.COPY_DST | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT
    })
    resources.diffTexture = shader.device.createTexture({
      label: 'diff',
      size: [outputWidth, outputHeight, Math.max(1, kernels.length - 1)],
      mipLevelCount: mipLevels,
      format: 'r32sint',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC | GPUTextureUsage.COPY_DST | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT
    })
    resources.diffTextureView = createMipViewArray(resources.diffTexture)
    resources.maxTexture = shader.device.createTexture({
      label: 'maxTexture',
      size: [outputWidth, outputHeight, Math.max(1, resources.diffTexture.depthOrArrayLayers - 2)],
      mipLevelCount: mipLevels,
      format: 'r32sint',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC | GPUTextureUsage.COPY_DST | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT
    })
    resources.maxTextureView = createMipViewArray(resources.maxTexture)
    resources.outViewsArray = resources.outputTexture.createView({
      dimension: '2d-array'
    })
    resources.tempTexture = shader.device.createTexture({
      label: 'temp',
      size: [outputWidth, outputHeight],
      mipLevelCount: mipLevels,
      format: 'r32sint',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC | GPUTextureUsage.COPY_DST | GPUTextureUsage.TEXTURE_BINDING
    })
    const copySize = [inputImage.width, inputImage.height, 1]
    console.log('#image size', copySize)
    // should handle both ImageBitmap and ImageData
    shader.device.queue.copyExternalImageToTexture({source: inputImage}, {
      texture: resources.rgbaTexture, origin: [0, 0, 0]
    }, copySize)
    resources.tempView = resources.tempTexture.createView({dimension: '2d'})
    resources.tempViewStorage = createMipViewArray(resources.tempTexture)
    resources.outViewsStorage = []
    resources.outViewMipmap = []
    for (let i = 0; i < resources.outputTexture.depthOrArrayLayers; i++) {
      resources.outViewsStorage.push(createMipViewArray(resources.outputTexture, {dimension: '2d', baseArrayLayer: i}))
      resources.outViewMipmap.push(resources.outputTexture.createView({
        dimension: '2d', baseArrayLayer: i
      }))
    }
    console.log(resources.outViewsStorage)
    const bytesPerRow = Math.ceil((outputWidth * 4) / 256) * 256
    resources.outputBuffer = shader.device.createBuffer({
      size: bytesPerRow * outputHeight, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
    })
    const workgroupSize = shader.pipelineDescs['single_pass_radial'].compute.constants.workgroup_size
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

  async encodeConvertToGray (computePass) {
    computePass.setPipeline(this.pipelines['convert_to_gray'])
    computePass.setBindGroup(0, this.device.createBindGroup({
      label: 'gray bind group 0',
      layout: this.pipelines['convert_to_gray'].getBindGroupLayout(0),
      entries: [
        {binding: 2, resource: {buffer: this.createUniformBuffer({from_mip: 0})}},
        {binding: 8, resource: this.rgbaTextureView},
        {binding: 9, resource: this.outViewsStorage[0][0]}]
    }))
    computePass.dispatchWorkgroups(Math.ceil(this.outputWidth / 8), Math.ceil(this.outputHeight / 8))
  }

  async encodeConvertFromGray (computePass, inputTextureView, convertNegative = false) {
    const pipeline = this.pipelines['convert_from_gray']
    computePass.setPipeline(pipeline)
    computePass.setBindGroup(0, this.device.createBindGroup({
      label: 'from gray bind group 0',
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        {binding: 0, resource: inputTextureView},
        {
          binding: 2,
          resource: {buffer: this.createUniformBuffer({from_mip: 0, from_gray_negative: convertNegative ? 1 : 0})}
        },
        {binding: 10, resource: this.rgbaTextureView}]
    }))
    computePass.dispatchWorkgroups(Math.ceil(this.outputWidth / 8), Math.ceil(this.outputHeight / 8))
  }

  mipSize (mipLevel) {
    const mipsRatio = 2 ** mipLevel
    return [this.outputWidth, this.outputHeight].map(d => Math.floor(d / mipsRatio))
  }

  workgroups88Mip (mipLevel) {
    return this.mipSize(mipLevel).map(d => Math.ceil(d / 8))
  }

  async encodeRepeatedPasses (workgroupSize) {
    const commandEncoder = this.device.createCommandEncoder({
      label: 'encodeRepeatedPasses'
    })
    const computePass = commandEncoder.beginComputePass({
      label: 'Gaussian repeated compute pass',
    })
    await this.encodeConvertToGray(computePass)
    let gaussPipeline = this.pipelines['single_pass_radial']
    computePass.setPipeline(gaussPipeline)
    for (let mipLevel = 0; mipLevel < this.outputTexture.mipLevelCount; mipLevel++) {
      let inputMipLevel = mipLevel === 0 ? 0 : mipLevel - 1
      let inputTexture = mipLevel === 0 ? this.outViewMipmap[0] : this.outViewMipmap[this.outViewMipmap.length - 3]
      const [outputWidth, outputHeight] = this.mipSize(mipLevel)
      const horizontalWorkGroups = [Math.ceil(outputWidth / workgroupSize), outputHeight]
      const verticalWorkGroups = [Math.ceil(outputHeight / workgroupSize), outputWidth]
      for (const [index, kernel] of this.kernelBuffers.entries()) {
        let inputParamBuffer = this.createUniformBuffer({
          horizontal: 1,
          from_mip: inputMipLevel,
          convert_to_gray: mipLevel === 0 && index === 0 ? 1 : 0
        })
        computePass.setBindGroup(0, this.device.createBindGroup({
          label: 'Gauss bind group 0',
          layout: gaussPipeline.getBindGroupLayout(0),
          entries: [{binding: 0, resource: inputTexture},
            {binding: 1, resource: this.tempViewStorage[mipLevel]},
            {binding: 2, resource: {buffer: inputParamBuffer}},
            {binding: 3, resource: {buffer: kernel}},]
        }))
        computePass.dispatchWorkgroups(...horizontalWorkGroups)
        inputParamBuffer = this.createUniformBuffer({
          horizontal: 0,
          from_mip: mipLevel,
          convert_to_gray: 0
        })
        computePass.setBindGroup(0, this.device.createBindGroup({
          label: 'Gauss bind group 0',
          layout: gaussPipeline.getBindGroupLayout(0),
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
    let diffPipeline = this.pipelines['subtract']
    computePass.setPipeline(diffPipeline)
    for (let mip = 0; mip < this.outputTexture.mipLevelCount; mip++) {
      computePass.setBindGroup(0, this.device.createBindGroup({
        label: 'diff bind group 0',
        layout: diffPipeline.getBindGroupLayout(0),
        entries: [
          {binding: 2, resource: {buffer: this.createUniformBuffer({from_mip: mip, convert_to_gray: 0})}},
          {binding: 4, resource: this.outViewsArray},
          {binding: 5, resource: this.diffTextureView[mip]}
        ]
      }))
      const [wgW, wgH] = this.workgroups88Mip(mip)
      computePass.dispatchWorkgroups(wgW, wgH, this.diffTexture.depthOrArrayLayers)
    }
    let extremaPipeline = this.pipelines['extrema']
    computePass.setPipeline(extremaPipeline)
    for (let mip = 0; mip < this.outputTexture.mipLevelCount; mip++) {
      computePass.setBindGroup(0, this.device.createBindGroup({
        layout: extremaPipeline.getBindGroupLayout(0),
        entries: [
          {binding: 5, resource: this.diffTextureView[mip]},
          {binding: 7, resource: this.maxTextureView[mip]}
        ]
      }))
      const [wgW, wgH] = this.workgroups88Mip(mip)
      computePass.dispatchWorkgroups(wgW, wgH, this.maxTexture.depthOrArrayLayers)
    }
    computePass.end()
    return commandEncoder.finish({
      label: 'encodeRepeatedPasses'
    })
  }

  async encodeSinglePass (callHorizontal, callVertical, horizontalWorkGroups, verticalWorkGroups) {
    const commandEncoder = this.device.createCommandEncoder()
    // Calculate bytesPerRow, which must be a multiple of 256 for WebGPU
    const bytesPerRow = Math.ceil((this.outputWidth * 4) / 256) * 256
    const computePass = commandEncoder.beginComputePass({
      label: 'Gaussian compute pass',
    })
    await this.encodeConvertToGray(computePass)
    computePass.setPipeline(this.pipelines['single_pass_radial'])
    let vInput
    let vOutput = this.outViewsStorage[0][0]
    let outCopyOrig
    if (callHorizontal) {
      computePass.setBindGroup(0, this.device.createBindGroup({
        label: 'Gauss bind group 0',
        layout: this.pipelines['single_pass_radial'].getBindGroupLayout(0),
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
      vInput = this.tempView
    } else {
      vInput = this.outViewsStorage[0][0]
      vOutput = this.tempViewStorage[0]
    }
    if (callVertical) {
      computePass.setBindGroup(0, this.device.createBindGroup({
        label: 'Gauss bind group 0',
        layout: this.pipelines['single_pass_radial'].getBindGroupLayout(0),
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
    }
    await this.encodeConvertFromGray(computePass, this.tempView)
    computePass.end()
    commandEncoder.copyTextureToBuffer({texture: this.rgbaTexture}, {
      buffer: this.outputBuffer, bytesPerRow
    }, [this.outputWidth, this.outputHeight, 1])
    return commandEncoder.finish()
  }

  async getTexture (name, index, mipLevel) {
    const commandEncoder = this.device.createCommandEncoder()
    const mipRatio = 2 ** mipLevel
    let outW = Math.floor(this.outputWidth / mipRatio)
    let outH = Math.floor(this.outputHeight / mipRatio)
    const bytesPerRow = Math.ceil((outW * 4) / 256) * 256
    let texture = this[name]
    let sourceTexture = {texture: texture, origin: [0, 0, index], mipLevel: mipLevel}
    if (texture.format === 'r32sint') {
      const computePass = commandEncoder.beginComputePass({
        label: 'gray to RGBA compute pass',
      })
      computePass.setPipeline(this.pipelines['single_pass_radial'])
      await this.encodeConvertFromGray(computePass, texture.createView({
        dimension: '2d',
        baseMipLevel: mipLevel,
        baseArrayLayer: index,
        mipLevelCount: 1,
        arrayLayerCount: 1
      }), texture === this.diffTexture)
      computePass.end()
      sourceTexture = {texture: this.rgbaTexture, origin: [0, 0, 0], mipLevel: 0}
    }
    commandEncoder.copyTextureToBuffer(sourceTexture,
      {buffer: this.outputBuffer, bytesPerRow},
      [outW, outH, 1])
    this.device.queue.submit([commandEncoder.finish()])
    await this.device.queue.onSubmittedWorkDone()
    await this.outputBuffer.mapAsync(GPUMapMode.READ)
    try {
      const outputData = new Uint8ClampedArray(this.outputBuffer.getMappedRange()).slice(0, bytesPerRow * outH)
      // createImageBitmap() cuts the extra pixels caused by the x256 bytes per row alignment
      return await createImageBitmap(new ImageData(outputData, bytesPerRow / 4, outH), 0, 0, outW, outH)
    } finally {
      this.outputBuffer.unmap()
    }
  }

  async runShader () {
    console.time('runShader')
    const bytesPerRow = Math.ceil((this.outputWidth * 4) / 256) * 256
    const device = this.shader.device

    console.log('Submitting command buffer to GPU queue', this.encodedCommands.label)
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

function objMap (obj, fun) {
  return Object.fromEntries(Object.entries(obj).map(([key, v], _) =>
    [key, fun(key, v)]))
}

export class RadialShader {

  constructor (device, pipelines, pipelineDescs, defs, module) {
    this.device = device
    this.pipelines = pipelines
    this.pipelineDescs = pipelineDescs
    this.defs = defs
    this.module = module
  }

  static async createShaders (device) {
    const shaderCode = await (await fetch('radial.wgsl')).text()
    const defs = makeShaderDataDefinitions(shaderCode)
    console.log('DEF', defs)
    const module = device.createShaderModule({
      label: 'Gaussian Blur Shader Test', code: shaderCode,
    })
    const pipelineDescs = objMap(defs.entryPoints, (key, v) =>
      ({label: `${key} pipeline`, layout: 'auto', compute: {module: module, entryPoint: key, constants: {}}}))
    console.log(pipelineDescs)
    pipelineDescs['single_pass_radial'].compute.constants.workgroup_size = 64
    const pipelines = objMap(pipelineDescs, (key, v) => {
      console.log(v)
      return device.createComputePipeline(v)
    })
    let shader = new RadialShader(device, pipelines, pipelineDescs, defs, module)
    shader.uniformsView = makeStructuredView(shader.defs.uniforms.parameters)
    return shader
  }

  async createGPUResources (workgroupSize, inputImage, outputWidth, outputHeight, kernels, oneDirection = null) {
    return AllocatedRadialShader.createGPUResources(this, inputImage, outputWidth, outputHeight, kernels, oneDirection)
  }

}
