// SIFT WebGPU Shader Definitions
// This module contains shared shader code used by both the worker and test files

// SIFT implementation constants
import {
  getSizeAndAlignmentOfUnsizedArrayElement,
  getSizeForMipFromTexture,
  makeShaderDataDefinitions,
  makeStructuredView,
  numMipLevels
} from './lib/webgpu-utils.module.js'

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

const EVERYTHING_TEXTURE = GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC | GPUTextureUsage.COPY_DST | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT

class AllocatedRadialShader {
  constructor (shader) {
    this.shader = shader
  }

  static async createGPUResources (shader, inputImage, kernels, oneDirection = null) {
    console.assert(oneDirection == null || kernels.length === 1, `can use oneDirection parameter only when there is only one kernel, found ${kernels.length} kernels`)
    const resources = new AllocatedRadialShader(shader)
    resources.device = shader.device
    resources.pipelines = shader.pipelines
    resources.kernelBuffers = kernels.map(k => {
      const kernelBuffer = shader.device.createBuffer({
        label: 'kernel',
        size: k.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      })
      shader.device.queue.writeBuffer(kernelBuffer, 0, k)
      return kernelBuffer
    })
    resources.rgbaTexture = await imageToTextureMaybeDoubled(shader.device, inputImage, oneDirection === null)
    const outputWidth = resources.rgbaTexture.width
    const outputHeight = resources.rgbaTexture.height
    resources.outputWidth = outputWidth
    resources.outputHeight = outputHeight
    resources.uniformsView = shader.uniformsView
    const mipLevels = numMipLevels([outputWidth, outputHeight])
    console.log('mipLevels', mipLevels)
    resources.rgbaTextureView = resources.rgbaTexture.createView()
    resources.outputTexture = shader.device.createTexture({
      label: 'output',
      size: [outputWidth, outputHeight, kernels.length],
      mipLevelCount: mipLevels,
      format: 'r32sint',
      usage: EVERYTHING_TEXTURE
    })
    resources.diffTexture = shader.device.createTexture({
      label: 'diff',
      size: [outputWidth, outputHeight, Math.max(1, kernels.length - 1)],
      mipLevelCount: mipLevels,
      format: 'r32sint',
      usage: EVERYTHING_TEXTURE
    })
    resources.diffTextureView = createMipViewArray(resources.diffTexture)
    resources.maxTexture = shader.device.createTexture({
      label: 'maxTexture',
      size: [outputWidth, outputHeight, Math.max(1, resources.diffTexture.depthOrArrayLayers - 2)],
      mipLevelCount: mipLevels,
      format: 'r32sint',
      usage: EVERYTHING_TEXTURE
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
      label: 'outputBuffer',
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
      label: 'uniforms',
      size: this.uniformsView.arrayBuffer.byteLength, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })

    this.uniformsView.set(values)
    this.device.queue.writeBuffer(buff, 0, this.uniformsView.arrayBuffer)
    return {buffer: buff}
  }

  async encodeConvertToGray (computePass) {
    await encodePipePrep(this.device, this.pipelines['convert_to_gray'], computePass, {
      input_rgba: this.rgbaTextureView,
      parameters: this.createUniformBuffer({from_mip: 0}),
      output_gray: this.outViewsStorage[0][0]
    }, this.shader.defs.entryPoints['convert_to_gray'].resources)
    computePass.dispatchWorkgroups(Math.ceil(this.outputWidth / 8), Math.ceil(this.outputHeight / 8))
  }

  async encodeConvertFromGray (computePass, inputTextureView, convertNegative = false) {
    const pipeline = this.pipelines['convert_from_gray']
    await encodePipePrep(this.device, pipeline, computePass, {
      inputTexture: inputTextureView,
      parameters: this.createUniformBuffer({from_mip: 0, from_gray_negative: convertNegative ? 1 : 0}),
      output_rgba: this.rgbaTextureView
    }, this.shader.defs.entryPoints['convert_from_gray'].resources)
    computePass.dispatchWorkgroups(Math.ceil(this.outputWidth / 8), Math.ceil(this.outputHeight / 8))
  }

  mipSize (mipLevel) {
    return getSizeForMipFromTexture(this.outputTexture, mipLevel).slice(0, 2)
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
            {binding: 2, resource: inputParamBuffer},
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
            {binding: 2, resource: inputParamBuffer},
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
          {binding: 2, resource: this.createUniformBuffer({from_mip: mip, convert_to_gray: 0})},
          {binding: 4, resource: this.outViewsArray},
          {binding: 5, resource: this.diffTextureView[mip]}
        ]
      }))
      const [wgW, wgH] = this.workgroups88Mip(mip)
      computePass.dispatchWorkgroups(wgW, wgH, this.diffTexture.depthOrArrayLayers)
    }
    let extremaPipeline = this.pipelines['extrema']
    computePass.setPipeline(extremaPipeline)
    const maxExtremaPerWg = 4
    const extremaBuffers = []
    this.extremaBuffers = extremaBuffers
    for (let mip = 0; mip < this.outputTexture.mipLevelCount; mip++) {
      const [wgW, wgH] = this.workgroups88Mip(mip)
      const numElements = wgW * wgH * this.maxTexture.depthOrArrayLayers
      let extremaBuffer = this.device.createBuffer({
        label: 'extremas ' + mip,
        size: getSizeAndAlignmentOfUnsizedArrayElement(this.shader.defs.storages.extrema_storage).size * maxExtremaPerWg * numElements,
        usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.STORAGE,
      })
      let extremaCountBuffer = this.device.createBuffer({
        label: 'extrema count ' + mip,
        size: getSizeAndAlignmentOfUnsizedArrayElement(this.shader.defs.storages.extrema_count_storage).size * numElements,
        usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.STORAGE,
      })
      extremaBuffers.push({
        extrema: extremaBuffer,
        count: extremaCountBuffer,
        workgroups: [wgW, wgH, this.maxTexture.depthOrArrayLayers]
      })
      computePass.setBindGroup(0, this.device.createBindGroup({
        layout: extremaPipeline.getBindGroupLayout(0),
        entries: [
          {
            binding: 2,
            resource: this.createUniformBuffer({
              extrema_threshold: Math.floor(0.5 * 0.04 / 3),
              extrema_border: 5,
              max_extrema_per_wg: maxExtremaPerWg
            })
          },
          {binding: 5, resource: this.diffTextureView[mip]},
          {binding: 7, resource: this.maxTextureView[mip]},
          {binding: 11, resource: {buffer: extremaBuffer}},
          {binding: 12, resource: {buffer: extremaCountBuffer}}
        ]
      }))
      console.log('extrema dispatches', wgW, wgH, this.maxTexture.depthOrArrayLayers)
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
    let resourceDef = this.shader.defs.entryPoints['single_pass_radial'].resources
    let vInput
    let vOutput = this.outViewsStorage[0][0]
    if (callHorizontal) {
      await encodePipePrep(this.device, this.pipelines['single_pass_radial'], computePass, {
        inputTexture: this.outViewsStorage[0][0],
        gaussian_textures: this.tempViewStorage[0],
        parameters: this.createUniformBuffer({
          horizontal: 1,
          from_mip: 0
        })
        ,
        kernel: {buffer: this.kernelBuffers[0]}
      }, resourceDef)
      computePass.dispatchWorkgroups(...horizontalWorkGroups)
      vInput = this.tempView
    } else {
      vInput = this.outViewsStorage[0][0]
      vOutput = this.tempViewStorage[0]
    }
    if (callVertical) {
      await encodePipePrep(this.device, this.pipelines['single_pass_radial'], computePass, {
        inputTexture: vInput, gaussian_textures: vOutput, parameters: this.createUniformBuffer({
          horizontal: 0,
          from_mip: 0
        }),
        kernel: {buffer: this.kernelBuffers[0]}
      }, resourceDef)
      computePass.dispatchWorkgroups(...verticalWorkGroups)
    }
    await this.encodeConvertFromGray(computePass, this.tempView)
    computePass.end()
    commandEncoder.copyTextureToBuffer({texture: this.rgbaTexture}, {
      buffer: this.outputBuffer, bytesPerRow
    }, [this.outputWidth, this.outputHeight, 1])
    return commandEncoder.finish()
  }

  async getBuffer (buffer) {
    const resultBuffer = this.device.createBuffer({
      label: 'temp buffer',
      size: buffer.size,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    })
    const encoder = this.device.createCommandEncoder()
    encoder.copyBufferToBuffer(buffer, resultBuffer)
    this.device.queue.submit([encoder.finish()])
    await resultBuffer.mapAsync(GPUMapMode.READ)
    try {
      return resultBuffer.getMappedRange().slice()
    } finally {
      resultBuffer.unmap()
      resultBuffer.destroy()
    }
  }

  async getTexture (name, index, mipLevel) {
    const commandEncoder = this.device.createCommandEncoder()
    let texture = this[name]
    const [outW, outH, _layers] = getSizeForMipFromTexture(texture, mipLevel)
    const bytesPerRow = Math.ceil((outW * 4) / 256) * 256
    let sourceTexture = {texture: texture, origin: [0, 0, index], mipLevel: mipLevel}
    if (texture.format === 'r32sint') {
      const computePass = commandEncoder.beginComputePass({
        label: 'gray to RGBA compute pass',
      })
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

  constructor (device, pipelines, pipelineDescs, defs) {
    this.device = device
    this.pipelines = pipelines
    this.pipelineDescs = pipelineDescs
    this.defs = defs
  }

  static async createShaders (device) {
    const {pipelines, defs, descriptors} = await loadWgsl(device, 'radial.wgsl', {workgroup_size: 64})
    let shader = new RadialShader(device, pipelines, descriptors, defs)
    shader.uniformsView = makeStructuredView(shader.defs.uniforms.parameters)
    return shader
  }

  async createGPUResources (workgroupSize, inputImage, outputWidth, outputHeight, kernels, oneDirection = null) {
    return AllocatedRadialShader.createGPUResources(this, inputImage, kernels, oneDirection)
  }
}

export async function loadWgsl (device, url, constants = {}) {
  const shaderCode = await (await fetch(url)).text()
  const defs = makeShaderDataDefinitions(shaderCode)
  console.log('DEF', defs)
  const module = device.createShaderModule({
    label: `${url} module`, code: shaderCode,
  })
  const pipelineDescs = objMap(defs.entryPoints, (key, v) =>
    ({label: `${key} pipeline`, layout: 'auto', compute: {module: module, entryPoint: key, constants: constants}}))
  const pipelines = objMap(pipelineDescs, (key, v) => {
    return device.createComputePipeline(v)
  })
  return {pipelines, defs, descriptors: pipelineDescs}
}

async function encodePipePrep (device, pipeline, computePass, parameters, resourceDef) {
  computePass.setPipeline(pipeline)
  const bindGroups = new Map()
  for (const elem of resourceDef) {
    if (!bindGroups.has(elem.group)) {
      bindGroups.set(elem.group, [])
    }
    bindGroups.get(elem.group).push({binding: elem.entry.binding, resource: parameters[elem.name]})
  }
  for (const [gNum, desc] of bindGroups.entries()) {
    computePass.setBindGroup(gNum, device.createBindGroup({
      label: `group ${0} of ${pipeline.label}`,
      layout: pipeline.getBindGroupLayout(gNum),
      entries: desc
    }))
  }
}

export async function imageToTextureMaybeDoubled (device, inputImage, tryDouble = true) {
  const maxSize = device.limits.maxTextureDimension2D
  console.log('Max size', maxSize)
  const factor = tryDouble && Math.max(inputImage.width, inputImage.height) * 2 <= maxSize ? 2 : 1
  const inputTexture = device.createTexture({
    size: [inputImage.width, inputImage.height],
    format: 'rgba8unorm',
    usage: EVERYTHING_TEXTURE
  })
  // should handle both ImageBitmap and ImageData
  device.queue.copyExternalImageToTexture({source: inputImage}, {
    texture: inputTexture, origin: [0, 0, 0]
  }, [inputImage.width, inputImage.height, 1])
  if (factor === 1) {
    return inputTexture
  }
  const outputWidth = inputImage.width * factor
  const outputHeight = inputImage.height * factor
  const outputTexture = device.createTexture({
    label: 'doubled',
    size: [outputWidth, outputHeight, 1],
    mipLevelCount: 1,
    format: 'rgba8unorm',
    usage: EVERYTHING_TEXTURE
  })
  try {
    const {pipelines, defs, descriptors} = await loadWgsl(device, 'resize.wgsl')
    const commandEncoder = device.createCommandEncoder()
    const computePass = commandEncoder.beginComputePass({label: 'image doubling compute pass'})
    await encodePipePrep(device, pipelines['resize'], computePass, {
      input_sampler: device.createSampler({magFilter: 'linear', minFilter: 'linear'}),
      input_texture: inputTexture.createView(),
      output_texture: outputTexture.createView()
    }, defs.entryPoints['resize'].resources)
    computePass.dispatchWorkgroups(Math.ceil(outputWidth / 8), Math.ceil(outputHeight / 8))
    computePass.end()
    device.queue.submit([commandEncoder.finish()])
    return outputTexture
  } finally {
    inputTexture.destroy()
  }
}
