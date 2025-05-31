import {
  dogShader, RadialShader,
} from './sift-shaders.js'

export async function runShaderTests () {
  console.log('Starting shader unit tests...')

  QUnit.test('WebGPU Support', assert => {
    assert.ok(navigator.gpu, 'WebGPU not supported on this browser')
  })

  try {
    const adapter = await navigator.gpu.requestAdapter()
    QUnit.test('WebGPU Adapter', assert => {
      assert.ok(adapter, 'No appropriate GPU adapter found')
    })
    const device = await adapter.requestDevice({
      label: 'Shader Test Device',
      requiredFeatures: ['timestamp-query'],
      requiredLimits: {maxTextureDimension2D: 16384}
    })
    QUnit.module('Shader Tests', {
      before: function () {
        console.log('Starting shader tests module')
      },
      after: function () {
        console.log('Completed shader tests module')
      }
    })

    await testGaussianBlurShader(device)
    await testDogShader(device)

  } catch (error) {
    console.error('Error running shader tests:', error)
    QUnit.test('Shader Test Setup', assert => {
      assert.ok(false, `Error setting up shader tests: ${error.message}`)
    })
  }
}

async function readTextureData (device, texture, width, height) {
  // Calculate bytesPerRow, which must be a multiple of 256 for WebGPU
  const bytesPerRow = Math.ceil((width * 4) / 256) * 256

  const outputBuffer = device.createBuffer({
    size: bytesPerRow * height,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
  })

  // Copy texture to buffer
  const commandEncoder = device.createCommandEncoder()
  commandEncoder.copyTextureToBuffer(
    {texture},
    {buffer: outputBuffer, bytesPerRow},
    [width, height]
  )
  device.queue.submit([commandEncoder.finish()])

  // Read the buffer
  await outputBuffer.mapAsync(GPUMapMode.READ)
  const outputData = new Uint8Array(outputBuffer.getMappedRange())

  // Create a new array with the correct pixel data, removing any padding
  const pixelData = new Uint8Array(width * height * 4)
  const actualBytesPerRow = width * 4

  for (let y = 0; y < height; y++) {
    const sourceOffset = y * bytesPerRow
    const destOffset = y * actualBytesPerRow
    pixelData.set(outputData.subarray(sourceOffset, sourceOffset + actualBytesPerRow), destOffset)
  }

  outputBuffer.unmap()
  return pixelData
}

function createDirectionalTestPattern (height, width, textureData) {
  // Create a weird checkerboard pattern where some rows and columns are white
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const i = (y * width + x) * 4
      const whiteLine = y % 4 === 0
      const whiteColumn = x % 4 === 0
      const isBlack = !whiteLine && !whiteColumn && ((Math.floor(x / 4) + Math.floor(y / 4)) % 2 === 0)

      if (isBlack) {
        // Black cell
        textureData[i] = 0     // R
        textureData[i + 1] = 0 // G
        textureData[i + 2] = 0 // B
      } else {
        // White cell
        textureData[i] = 255     // R
        textureData[i + 1] = 255 // G
        textureData[i + 2] = 255 // B
      }
      textureData[i + 3] = 255 // A always fully opaque
    }
  }
}

// Test for Gaussian Blur Shader
export async function testGaussianBlurShader (device) {
  console.log('Testing Gaussian Blur Shader...')

  let inputImage
  let outputImage

  const workgroupSize = 64
  const kernelRadius = 20

  function checkColor (y, width, x, outputData, euclideanDistance, gray, colorName) {
    const index = (y * width + x) * 4
    const pix = [outputData[index], outputData[index + 1], outputData[index + 2]]
    const colorDist = euclideanDistance(gray, pix)
    let isGray = colorDist < 3
    if (!isGray) {
      console.log(`expected ${colorName} ${gray} pixel at ${[x, y]}, got ${[outputData[index], outputData[index + 1], outputData[index + 2]]}`)
    }
    return isGray
  }

  QUnit.test('Gaussian Blur Shader', async assert => {

    try {
      console.log('creating shader')
      let ts = await RadialShader.createShader(device, kernelRadius, workgroupSize)

      // Test 2: Verify horizontal blur
      const width = 256
      const height = 128
      // Create a checkerboard pattern for better visibility of blur effect
      const textureData = new Uint8ClampedArray(width * height * 4)
      createDirectionalTestPattern(height, width, textureData)
      inputImage = new ImageData(textureData, width)
      // Keep track of the middle for testing
      const midX = Math.floor(width / 2)

      // Log the input texture data for debugging
      console.log('Input texture data (first few pixels):')
      for (let i = 0; i < 5; i++) {
        const idx = i * 4
        console.log(`Pixel ${i}: RGBA=(${textureData[idx]}, ${textureData[idx + 1]}, ${textureData[idx + 2]}, ${textureData[idx + 3]})`)
      }

      // Log the input texture data around the black line
      console.log('Input texture data (horizontal line through the middle, around the black line):')
      const midY = Math.floor(height / 2)
      for (let x = midX - 5; x <= midX + 5; x++) {
        if (x < 0 || x >= width) continue
        const idx = (midY * width + x) * 4
        console.log(`Pixel at (${x}, ${midY}): RGBA=(${textureData[idx]}, ${textureData[idx + 1]}, ${textureData[idx + 2]}, ${textureData[idx + 3]})`)
      }

      let gpuTime

      function computeGaussianValue (radius, kernelRadius, sigma = null) {
        if (sigma == null) {
          sigma = kernelRadius / 3 // 3*sigma covers >99% of Gaussian
        }
        const twoSigmaSquared = 2.0 * sigma * sigma
        return Math.exp(-(radius * radius) / twoSigmaSquared) / (Math.PI * twoSigmaSquared)
      }

      function computeGaussianKernel (kernelRadius, sigma = null) {
        return Float32Array.from({length: kernelRadius}, (_, i) => computeGaussianValue(i, kernelRadius, sigma))
      }

      const quenelle = computeGaussianKernel(kernelRadius)
      console.log('computeGaussianValue', quenelle)
      const kernelBuffer = device.createBuffer({
        size: quenelle.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      })
      device.queue.writeBuffer(kernelBuffer, 0, quenelle)

      let allocatedShader = await ts.createGPUResources(kernelRadius, workgroupSize, inputImage, width, height, kernelBuffer)
      let outputData = await allocatedShader.runShader(true, false)

      console.log('####outputData', outputData.width, outputData.height, outputData.constructor.name)
      // Test horizontal blur
      let whiteRowUntouched = true
      for (let x = 0; x < width; x++) {
        let y = 0 //first row should be white
        const index = (y * width + x) * 4
        const isWhite = outputData.data[index] === 255 && outputData.data[index + 1] === 255 && outputData.data[index + 2] === 255
        whiteRowUntouched &= isWhite
        if (!isWhite) {
          console.log(`expected white pixel at ${[x, y]}, got ${[outputData.data[index], outputData.data[index + 1], outputData.data[index + 2], outputData.data[index + 3]]}`)
        }
      }

      assert.ok(whiteRowUntouched, 'Horizontal blur: White row 0 should remain untouched')
      // Display input and output images for this assertion
      await assert.imageTest(inputImage, outputData, `Horizontal blur: Input and output. GPU time: ${allocatedShader.gpuTime}`, whiteRowUntouched)

      const euclideanDistance = (a, b) =>
        Math.hypot(...Object.keys(a).map(k => b[k] - a[k]))
      const gray = [160, 160, 160]
      let grayRowDetected = true
      for (let x = 20; x < width - 20; x++) {
        let y = 1
        grayRowDetected &= checkColor(y, width, x, outputData.data, euclideanDistance, gray, 'gray')
      }

      assert.ok(grayRowDetected, 'Horizontal blur: Checkered row 1 should be blurred to gray')
      // Display input and output images for this assertion
      await assert.imageTest(inputImage, outputData, `Horizontal blur: Input and output images. GPU time: ${allocatedShader.gpuTime}`, grayRowDetected)

      // Vertical test
      allocatedShader = await ts.createGPUResources(kernelRadius, workgroupSize, inputImage, width, height, kernelBuffer)
      outputData = await allocatedShader.runShader(false, true)
      let whiteColumnUntouched = true
      for (let y = 0; y < height; y++) {
        let x = 0 //first column should be white
        whiteColumnUntouched &= checkColor(y, width, x, outputData.data, euclideanDistance, [255, 255, 255], 'white')
      }

      assert.ok(whiteColumnUntouched, 'Vertical blur: White column 0 should remain untouched')
      // Display input and output images for this assertion
      await assert.imageTest(inputImage, outputData, `Vertical blur: Input and output images. GPU time: ${allocatedShader.gpuTime}`, whiteColumnUntouched)

      let grayColumnDetected = true
      for (let y = 20; y < height - 20; y++) {
        const x = 1
        grayColumnDetected &= checkColor(y, width, x, outputData.data, euclideanDistance, gray, 'gray')
      }

      assert.ok(grayColumnDetected, 'Vertical blur: Checkered column 1 should be blurred to gray')
      // Display input and output images for this assertion
      await assert.imageTest(inputImage, outputData, 'Vertical blur: Input and output images', grayColumnDetected)
      ts = await RadialShader.createShader(device, kernelRadius, workgroupSize)
      allocatedShader = await ts.createGPUResources(kernelRadius, workgroupSize, inputImage, inputImage.width, inputImage.height, kernelBuffer)
      outputImage = await allocatedShader.runShader(true, true)
      await assert.imageTest(inputImage, outputImage, `Blur, running both passes. GPU time: ${allocatedShader.gpuTime}`, whiteColumnUntouched)

      let testImage = await createImageBitmap(await (await fetch('3916587d9b.png')).blob())
      ts = await RadialShader.createShader(device, kernelRadius, workgroupSize)
      allocatedShader = await ts.createGPUResources(kernelRadius, workgroupSize, testImage, testImage.width, testImage.height, kernelBuffer)
      outputImage = await allocatedShader.runShader(true, true)
      await assert.imageTest(testImage, outputImage, `Complete blur: Input and output images. GPU time: ${allocatedShader.gpuTime}`, true)
      //NASM-A20150317000-NASM2018-10769.jpg
      console.time('createImageBitmap')
      testImage = await createImageBitmap(await (await fetch('NASM-A20150317000-NASM2018-10769.jpg')).blob())
      console.log('testImage size', testImage.width, testImage.height)
      console.timeEnd('createImageBitmap')
      ts = await RadialShader.createShader(device, kernelRadius, workgroupSize)
      allocatedShader = await ts.createGPUResources(kernelRadius, workgroupSize, testImage, testImage.width, testImage.height, kernelBuffer)
      outputImage = await allocatedShader.runShader(true, true)
      console.time('imageTest')
      await assert.imageTest(testImage, outputImage, `Complete blur on big image: Input and output images. GPU time: ${allocatedShader.gpuTime}`, true)
      console.timeEnd('imageTest')
      const imagesPerOctaves = 5
      const gaussians = [1.2262735, 1.54500779, 1.94658784, 2.452547, 3.09001559]
      console.log('tripple gaussians', gaussians, gaussians.map(g => g * 3))
      const gs = gaussians.map(sigma => computeGaussianKernel(Math.ceil(sigma * 3), sigma))
      console.log('gaussians', gs)
      kernelBuffer.destroy()
    } catch (error) {
      console.error('Error testing Gaussian Blur Shader:', error)
      assert.ok(false, `Error testing Gaussian Blur Shader: ${error.message}`)
    }
  })
}

// Test for DoG (Difference of Gaussians) Shader
export async function testDogShader (device) {
  console.log('Testing DoG Shader...')

  // Create QUnit test
  QUnit.test('DoG (Difference of Gaussians) Shader', async assert => {
    try {
      // Create shader module
      const dogModule = device.createShaderModule({
        label: 'DoG Shader Test',
        code: dogShader
      })

      // Create compute pipeline
      const dogPipeline = device.createComputePipeline({
        label: 'DoG Pipeline Test',
        layout: 'auto',
        compute: {
          module: dogModule,
          entryPoint: 'main'
        }
      })

      // Test 1: Verify shader compilation
      assert.ok(dogPipeline, 'Shader compiled successfully')

      // Test 2: Verify DoG computation
      const width = 32
      const height = 32

      // Create two input textures with different patterns
      // First texture: uniform gray
      const texture1Data = new Uint8Array(width * height * 4)
      texture1Data.fill(128) // Fill with gray (128, 128, 128, 255)
      for (let i = 0; i < width * height; i++) {
        texture1Data[i * 4 + 3] = 255 // Set alpha to 255
      }

      const texture1 = device.createTexture({
        size: [width, height],
        format: 'rgba8unorm',
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST
      })

      device.queue.writeTexture(
        {texture: texture1},
        texture1Data,
        {bytesPerRow: width * 4},
        [width, height]
      )

      // Second texture: darker gray
      const texture2Data = new Uint8Array(width * height * 4)
      texture2Data.fill(64) // Fill with darker gray (64, 64, 64, 255)
      for (let i = 0; i < width * height; i++) {
        texture2Data[i * 4 + 3] = 255 // Set alpha to 255
      }

      const texture2 = device.createTexture({
        size: [width, height],
        format: 'rgba8unorm',
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST
      })

      device.queue.writeTexture(
        {texture: texture2},
        texture2Data,
        {bytesPerRow: width * 4},
        [width, height]
      )

      // Create output texture
      const outputTexture = device.createTexture({
        size: [width, height],
        format: 'rgba8unorm',
        usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC
      })

      // Create bind group
      const bindGroup = device.createBindGroup({
        layout: dogPipeline.getBindGroupLayout(0),
        entries: [
          {binding: 0, resource: texture1.createView()},
          {binding: 1, resource: texture2.createView()},
          {binding: 2, resource: outputTexture.createView()}
        ]
      })

      // Run the shader
      const commandEncoder = device.createCommandEncoder()
      const computePass = commandEncoder.beginComputePass()
      computePass.setBindGroup(0, bindGroup)
      computePass.setPipeline(dogPipeline)
      computePass.dispatchWorkgroups(Math.ceil(width / 16), Math.ceil(height / 16))
      computePass.end()
      device.queue.submit([commandEncoder.finish()])

      // Read back the results
      const outputData = await readTextureData(device, outputTexture, width, height)

      // Verify that the DoG computation is correct
      // The difference should be stored in the R channel
      // Expected difference: 128 - 64 = 64 (scaled by luminance weights)
      // Luminance weights: 0.299, 0.587, 0.114
      // Expected luminance: (128 - 64) * 0.299 + (128 - 64) * 0.587 + (128 - 64) * 0.114 = 64

      const expectedDifference = 64 * (0.299 + 0.587 + 0.114)
      const tolerance = 5 // Allow some tolerance due to floating point precision

      // Check a few pixels
      let allPixelsCorrect = true
      let incorrectPixelValue = null

      for (let i = 0; i < 5; i++) {
        const index = i * 100 * 4 // Check a few scattered pixels
        const actualDifference = outputData[index] // R channel

        if (Math.abs(actualDifference - expectedDifference) > tolerance) {
          allPixelsCorrect = false
          incorrectPixelValue = actualDifference
          break
        }
      }

      assert.ok(
        allPixelsCorrect,
        incorrectPixelValue === null
          ? 'DoG computation correctly calculated the difference'
          : `DoG computation incorrect. Expected ~${expectedDifference}, got ${incorrectPixelValue}`
      )

      // Create ImageData objects for input and output images
      const inputImageData1 = new ImageData(new Uint8ClampedArray(texture1Data), width, height)
      const inputImageData2 = new ImageData(new Uint8ClampedArray(texture2Data), width, height)
      const outputImageData = new ImageData(new Uint8ClampedArray(outputData), width, height)

      // Display input and output images for this assertion
      await assert.imageTest(inputImageData1, outputImageData, 'DoG: Input 1 and output images', allPixelsCorrect)
      await assert.imageTest(inputImageData2, outputImageData, 'DoG: Input 2 and output images', allPixelsCorrect)

      // Clean up
      texture1.destroy()
      texture2.destroy()
      outputTexture.destroy()

    } catch (error) {
      console.error('Error testing DoG Shader:', error)
      assert.ok(false, `Error testing DoG Shader: ${error.message}`)
    }
  })
}

