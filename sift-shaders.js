import {
  getSizeAndAlignmentOfUnsizedArrayElement,
  getSizeForMipFromTexture,
  makeShaderDataDefinitions,
  makeStructuredView,
} from './lib/webgpu-utils.module.js'

export const HORIZONTAL = 1
export const VERTICAL = 2

export function quadraticDiff (a, b) {
  return Math.sqrt(a ** 2 - b ** 2)
}

export function computeGaussianValue (radius, kernelRadius, sigma) {
  const twoSigmaSquared = 2.0 * sigma ** 2
  return Math.exp(-(radius ** 2) / twoSigmaSquared) / Math.sqrt(Math.PI * twoSigmaSquared)
}

export function computeGaussianKernel (sigma, kernelRadius = null) {
  if (kernelRadius === null) {
    kernelRadius = Math.ceil(Math.round(sigma * 8 + 1) / 2) // adapted from openCV
  }
  // value at 0 will be the center pixel, value at end will be the edge pixel
  let k = Float32Array.from({length: kernelRadius}, (_, i) => computeGaussianValue(i, kernelRadius, sigma))
  let sum = k[0]
  for (let i = 1; i < k.length; i++) {
    sum += k[i] * 2
  }
  return Float32Array.from(k.map(v => v / sum))
}

function createMipViewArray (texture, params = {}) {
  const result = []
  for (let i = 0; i < texture.mipLevelCount; i++) {
    result.push(texture.createView({
      mipLevelCount: 1, baseMipLevel: i, ...params
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
    const extremaBorder = 5
    const resources = new AllocatedRadialShader(shader)
    resources.device = shader.device
    resources.pipelines = shader.pipelines
    resources.kernelBuffers = kernels.map(k => {
      const kernelBuffer = shader.device.createBuffer({
        label: 'kernel', size: k.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      })
      shader.device.queue.writeBuffer(kernelBuffer, 0, k)
      return kernelBuffer
    })
    resources.rgbaTexture = await imageToTextureMaybeDoubled(shader.device, inputImage, oneDirection === null)
    resources.wasDoubled = resources.rgbaTexture.width !== inputImage.width
    const outputWidth = resources.rgbaTexture.width
    const outputHeight = resources.rgbaTexture.height
    resources.outputWidth = outputWidth
    resources.outputHeight = outputHeight
    resources.uniformsView = shader.uniformsView
    // 2**3 = 8 we need at least a 10pix wide image to find an extremum
    const borderMip = Math.floor(Math.log2(extremaBorder * 2))
    // Remove a few levels because we use a border when looking for extrema.
    const mipLevels = Math.ceil(Math.log2(Math.min(outputWidth, outputHeight))) - borderMip
    console.log('mipLevels', mipLevels)

    resources.rgbaTextureView = resources.rgbaTexture.createView()
    resources.outputTexture = shader.device.createTexture({
      label: 'output',
      size: [outputWidth, outputHeight, kernels.length + 1],
      mipLevelCount: mipLevels,
      format: 'r32sint',
      usage: EVERYTHING_TEXTURE
    })
    for (let i = 0; i < mipLevels; i++) {
      console.log('mipLevel', i, resources.mipSize(i))
    }
    resources.diffTexture = shader.device.createTexture({
      label: 'diff',
      size: [outputWidth, outputHeight, Math.max(1, resources.outputTexture.depthOrArrayLayers - 1)],
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
      label: 'outputBuffer', size: bytesPerRow * outputHeight, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
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
      size: this.uniformsView.arrayBuffer.byteLength,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })

    this.uniformsView.set(values)
    this.device.queue.writeBuffer(buff, 0, this.uniformsView.arrayBuffer)
    return {buffer: buff}
  }

  async encodeConvertToGray (computePass) {
    await encodePipePrep(this.device, computePass, this.pipelines['convert_to_gray'], this.shader.defs.entryPoints['convert_to_gray'].resources, {
      input_rgba: this.rgbaTextureView,
      parameters: this.createUniformBuffer({from_mip: 0}),
      output_gray: this.outViewsStorage[0][0]
    })
    dispatchSquare(computePass, this.outputWidth, this.outputHeight, 8)
  }

  async encodeConvertFromGray (computePass, inputTextureView, convertNegative = false) {
    const pipeline = this.pipelines['convert_from_gray']
    await encodePipePrep(this.device, computePass, pipeline, this.shader.defs.entryPoints['convert_from_gray'].resources, {
      inputTexture: inputTextureView,
      parameters: this.createUniformBuffer({from_mip: 0, from_gray_negative: convertNegative ? 1 : 0}),
      output_rgba: this.rgbaTextureView
    })
    dispatchSquare(computePass, this.outputWidth, this.outputHeight, 8)
  }

  mipSize (mipLevel) {
    return getSizeForMipFromTexture(this.outputTexture, mipLevel).slice(0, 2)
  }

  workgroups88Mip (mipLevel) {
    return this.mipSize(mipLevel).map(d => Math.ceil(d / 8))
  }

  async encodeRepeatedPasses (workgroupSize, extremaBorder) {
    const use_101_border = 1
    const commandEncoder = this.device.createCommandEncoder({
      label: 'encodeRepeatedPasses'
    })
    const computePass = commandEncoder.beginComputePass({
      label: 'Gaussian repeated compute pass',
    })
    await this.encodeConvertToGray(computePass)
    let gaussPipeline = this.pipelines['single_pass_radial']
    let resourceDef = this.shader.defs.entryPoints['single_pass_radial'].resources
    if (this.wasDoubled) {
      const [outputWidth, outputHeight] = this.mipSize(0)
      const sigma = 1.2489995996796799 // taken from pysift
      const kernel = computeGaussianKernel(sigma)
      console.log('gaussian', kernel)
      const kernelBuffer = this.device.createBuffer({
        label: 'kernel', size: kernel.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      })
      this.device.queue.writeBuffer(kernelBuffer, 0, kernel)
      await encodePipePrep(this.device, computePass, gaussPipeline, resourceDef, {
        inputTexture: this.outViewMipmap[0],
        gaussian_textures: this.tempViewStorage[0],
        parameters: this.createUniformBuffer({horizontal: 1, from_mip: 0, border_reflect_101: use_101_border}),
        kernel: {buffer: kernelBuffer}
      })
      computePass.dispatchWorkgroups(divCeil(outputWidth, workgroupSize), outputHeight)
      await encodePipePrep(this.device, computePass, gaussPipeline, resourceDef, {
        inputTexture: this.tempViewStorage[0],
        gaussian_textures: this.outViewsStorage[0][0],
        parameters: this.createUniformBuffer({horizontal: 0, from_mip: 0, border_reflect_101: use_101_border}),
        kernel: {buffer: kernelBuffer}
      })
      computePass.dispatchWorkgroups(divCeil(outputHeight, workgroupSize), outputWidth)
    }
    for (let mipLevel = 0; mipLevel < this.outputTexture.mipLevelCount; mipLevel++) {
      let inputTexture = this.outViewMipmap[0]
      if (mipLevel > 0) {
        // resize from higher res
        await encodePipePrep(this.device, computePass, this.pipelines['copy'], this.shader.defs.entryPoints['copy'].resources, {
          inputTexture: this.outViewMipmap[this.outViewMipmap.length - 3],
          gaussian_textures: this.outViewsStorage[0][mipLevel],
          parameters: this.createUniformBuffer({from_mip: mipLevel - 1})
        })
        const [outputWidth, outputHeight] = this.mipSize(mipLevel)
        dispatchSquare(computePass, outputWidth, outputHeight, 8)
      }
      const [outputWidth, outputHeight] = this.mipSize(mipLevel)
      const horizontalWorkGroups = [Math.ceil(outputWidth / workgroupSize), outputHeight]
      const verticalWorkGroups = [Math.ceil(outputHeight / workgroupSize), outputWidth]
      for (const [index, kernel] of this.kernelBuffers.entries()) {
        let inputParamBuffer = this.createUniformBuffer({
          horizontal: 1, from_mip: mipLevel, convert_to_gray: 0, border_reflect_101: use_101_border
        })
        await encodePipePrep(this.device, computePass, gaussPipeline, resourceDef, {
          inputTexture: inputTexture,
          gaussian_textures: this.tempViewStorage[mipLevel],
          parameters: inputParamBuffer,
          kernel: {buffer: kernel}
        })
        computePass.dispatchWorkgroups(...horizontalWorkGroups)
        inputParamBuffer = this.createUniformBuffer({
          horizontal: 0, from_mip: mipLevel, convert_to_gray: 0, border_reflect_101: use_101_border
        })
        await encodePipePrep(this.device, computePass, gaussPipeline, resourceDef, {
          inputTexture: this.tempView,
          gaussian_textures: this.outViewsStorage[index + 1][mipLevel],
          parameters: inputParamBuffer,
          kernel: {buffer: kernel}
        })
        computePass.dispatchWorkgroups(...verticalWorkGroups)
        inputTexture = this.outViewMipmap[index + 1]
      }
    }
    let diffPipeline = this.pipelines['subtract']
    for (let mip = 0; mip < this.outputTexture.mipLevelCount; mip++) {
      await encodePipePrep(this.device, computePass, diffPipeline, this.shader.defs.entryPoints['subtract'].resources, {
        parameters: this.createUniformBuffer({from_mip: mip, convert_to_gray: 0}),
        diff_input_stack: this.outViewsArray,
        diff_output_stack: this.diffTextureView[mip]
      })
      const [wgW, wgH] = this.workgroups88Mip(mip)
      computePass.dispatchWorkgroups(wgW, wgH, this.diffTexture.depthOrArrayLayers)
    }
    let extremaPipeline = this.pipelines['extrema']
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
        extrema: extremaBuffer, count: extremaCountBuffer, workgroups: [wgW, wgH, this.maxTexture.depthOrArrayLayers]
      })
      await encodePipePrep(this.device, computePass, extremaPipeline, this.shader.defs.entryPoints['extrema'].resources, {
        parameters: this.createUniformBuffer({
          extrema_threshold: 1 / 256,
          extrema_border: extremaBorder,
          max_extrema_per_wg: maxExtremaPerWg,
          from_mip: mip
        }),
        diff_output_stack: this.diffTextureView[mip],
        max_output_stack: this.maxTextureView[mip],
        extrema_storage: {buffer: extremaBuffer},
        extrema_count_storage: {buffer: extremaCountBuffer}
      })
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
      await encodePipePrep(this.device, computePass, this.pipelines['single_pass_radial'], resourceDef, {
        inputTexture: this.outViewsStorage[0][0],
        gaussian_textures: this.tempViewStorage[0],
        parameters: this.createUniformBuffer({
          horizontal: 1, from_mip: 0
        }),
        kernel: {buffer: this.kernelBuffers[0]}
      })
      computePass.dispatchWorkgroups(...horizontalWorkGroups)
      vInput = this.tempView
    } else {
      vInput = this.outViewsStorage[0][0]
      vOutput = this.tempViewStorage[0]
    }
    if (callVertical) {
      await encodePipePrep(this.device, computePass, this.pipelines['single_pass_radial'], resourceDef, {
        inputTexture: vInput, gaussian_textures: vOutput, parameters: this.createUniformBuffer({
          horizontal: 0, from_mip: 0
        }), kernel: {buffer: this.kernelBuffers[0]}
      })
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
      label: 'temp buffer', size: buffer.size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
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

  async getTexture (name, mipLevel, index, centerZero = null) {
    const commandEncoder = this.device.createCommandEncoder()
    let texture = this[name]
    if (centerZero == null) {
      centerZero = texture === this.diffTexture
    }
    const [outW, outH, _layers] = getSizeForMipFromTexture(texture, mipLevel)
    const bytesPerRow = Math.ceil((outW * 4) / 256) * 256
    let sourceTexture = {texture: texture, origin: [0, 0, index], mipLevel: mipLevel}
    if (texture.format === 'r32sint') {
      const computePass = commandEncoder.beginComputePass({
        label: 'gray to RGBA compute pass',
      })
      await this.encodeConvertFromGray(computePass, texture.createView({
        dimension: '2d', baseMipLevel: mipLevel, baseArrayLayer: index, mipLevelCount: 1, arrayLayerCount: 1
      }), centerZero)
      computePass.end()
      sourceTexture = {texture: this.rgbaTexture, origin: [0, 0, 0], mipLevel: 0}
    }
    commandEncoder.copyTextureToBuffer(sourceTexture, {buffer: this.outputBuffer, bytesPerRow}, [outW, outH, 1])
    this.device.queue.submit([commandEncoder.finish()])
    await this.outputBuffer.mapAsync(GPUMapMode.READ)
    try {
      const outputData = new Uint8ClampedArray(this.outputBuffer.getMappedRange()).slice(0, bytesPerRow * outH)
      // need to crop the 256-bytes line alignment constraint
      const croppedData = new ImageData(outW, outH)
      let videoFrame = new VideoFrame(outputData, {
        format: 'RGBA',
        codedWidth: bytesPerRow / 4,
        codedHeight: outH,
        timestamp: 0,
        visibleRect: {x: 0, y: 0, height: outH, width: outW}
      })
      await videoFrame.copyTo(croppedData.data)
      videoFrame.close()
      return croppedData
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
  return Object.fromEntries(Object.entries(obj).map(([key, v], _) => [key, fun(key, v)]))
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
  const pipelineDescs = objMap(defs.entryPoints, (key, _v) => ({
    label: `${key} pipeline`,
    layout: 'auto',
    compute: {module: module, entryPoint: key, constants: constants}
  }))
  const pipelines = objMap(pipelineDescs, (key, v) => {
    return device.createComputePipeline(v)
  })
  return {pipelines, defs, descriptors: pipelineDescs}
}

async function encodePipePrep (device, computePass, pipeline, resourceDef, parameters) {
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
      label: `group ${0} of ${pipeline.label}`, layout: pipeline.getBindGroupLayout(gNum), entries: desc
    }))
  }
}

export async function imageToTextureMaybeDoubled (device, inputImage, tryDouble = true) {
  const maxSize = device.limits.maxTextureDimension2D
  console.log('Max size', maxSize)
  const factor = tryDouble && Math.max(inputImage.width, inputImage.height) * 2 <= maxSize ? 2 : 1
  const inputTexture = device.createTexture({
    size: [inputImage.width, inputImage.height], format: 'rgba8unorm', usage: EVERYTHING_TEXTURE
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
    const wgSizeXY = 8
    const {pipelines, defs} = await loadWgsl(device, 'resize.wgsl', {workgroupxy_size: wgSizeXY})
    const commandEncoder = device.createCommandEncoder()
    const computePass = commandEncoder.beginComputePass({label: 'image doubling compute pass'})
    await encodePipePrep(device, computePass, pipelines['resize'], defs.entryPoints['resize'].resources, {
      input_sampler: device.createSampler({magFilter: 'linear', minFilter: 'linear'}),
      input_texture: inputTexture.createView(),
      output_texture: outputTexture.createView()
    })
    dispatchSquare(computePass, outputWidth, outputHeight, wgSizeXY)
    computePass.end()
    device.queue.submit([commandEncoder.finish()])
    return outputTexture
  } finally {
    inputTexture.destroy()
  }
}

function dispatchSquare (computePass, w, h, side, z = 1) {
  computePass.dispatchWorkgroups(divCeil(w, side), divCeil(h, side), z)
}

function divCeil (num, denominator) {
  return Math.ceil(num / denominator)
}
