import {
  computeGaussianKernel,
  computeGaussianValue,
  HORIZONTAL,
  quadraticDiff,
  RadialShader,
  VERTICAL,
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
      label: 'Shader Test Device', requiredLimits: {
        maxTextureDimension2D: adapter.limits.maxTextureDimension2D
      }, requiredFeatures: ['float32-filterable']
    })
    device.addEventListener('uncapturederror', (event) => {
      console.error(event.error.message)
    })
    QUnit.module('Shader Tests', {
      before: function () {
        console.log('Starting shader tests module')
      }, after: function () {
        console.log('Completed shader tests module')
      }
    })

    await testGaussianBlurShader(device)

  } catch (error) {
    console.error('Error running shader tests:', error)
    QUnit.test('Shader Test Setup', assert => {
      assert.ok(false, `Error setting up shader tests: ${error.message}`)
    })
  }
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

  QUnit.test('Gaussian computations', async assert => {
    const s1 = 1.6
    const s2 = 1
    const diff = quadraticDiff(s1, s2)
    const expectedDiff = 1.2489995996796799
    assert.closeTo(diff, expectedDiff, 0.00001, `quadraticDiff(${s1}, ${s2}) expected to be close to ${expectedDiff}`)
    const expectedGaussian = [2.49348081e-01, 2.05108137e-01, 1.14159870e-01, 4.29930011e-02, 1.09555901e-02, 1.88898063e-03, 2.20380394e-04]
    assert.closeTo(computeGaussianValue(0, 7, 1.6), expectedGaussian[0], 0.00001, `computeGaussianValue(0, 7, 1.6) expected to be close to ${expectedGaussian[0]}`)
    const computedGaussian = computeGaussianKernel(1.6)
    assert.equal(computedGaussian.length, expectedGaussian.length)
    const sqDiff = Math.sqrt(expectedGaussian.map((_, idx) => (expectedGaussian[idx] - computedGaussian[idx]) ** 2).reduce((acc, val) => acc + val, 0))
    assert.closeTo(sqDiff, 0, 0.0001, `computed to expected gaussian distance was ${sqDiff}`)
    let sum = computedGaussian[0]
    for (let i = 1; i < computedGaussian.length; i++) {
      sum += computedGaussian[i] * 2
    }
    assert.closeTo(sum, 1, 0.00001, `gaussian kernel sum expected to be one, actual: ${sum}`)
  })

  QUnit.test('Compare with pysift', async assert => {
    let ts = await RadialShader.createShaders(device, kernelRadius, workgroupSize)
    const gaussians = [1.2262734984654078, 1.5450077936447955, 1.9465878414647133, 2.4525469969308156, 3.090015587289591]
    const gs = gaussians.map(sigma => computeGaussianKernel(sigma))
    console.log('gaussians', gs)

    const testImage = await createImageBitmap(await (await fetch('box.png')).blob())
    const allocatedShader = await ts.createGPUResources(workgroupSize, testImage, testImage.width, testImage.height, gs)
    outputImage = await allocatedShader.runShader()

    async function compareToPysift (fileName, textureName, mipLevel, index, centerZero = false) {
      const refImage = await createImageBitmap(await (await fetch(fileName)).blob())
      const computedImage = await allocatedShader.getTexture(textureName, mipLevel, index, centerZero)
      const refVidFrame = new VideoFrame(refImage, {timestamp: 0})
      const refData = new Uint8Array(refVidFrame.allocationSize())
      await refVidFrame.copyTo(refData)
      refVidFrame.close()
      const imageData = new ImageData(refImage.width, refImage.height)

      let max_diff = 0
      let min_diff = 0
      let sqDiff = 0
      for (let i = 0; i < imageData.data.length; i += 4) {
        let diff = refData[i] - computedImage.data[i]
        sqDiff += diff ** 2
        max_diff = Math.max(max_diff, diff)
        min_diff = Math.min(min_diff, diff)
      }
      sqDiff = Math.sqrt(sqDiff)
      const factor = 255 / (max_diff - min_diff)
      for (let i = 0; i < imageData.data.length; i += 4) {
        let diff = (refData[i] - computedImage.data[i] - min_diff) * factor
        // noinspection PointlessArithmeticExpressionJS
        imageData.data[i + 0] = diff >= 0 ? diff : 0
        imageData.data[i + 1] = diff < 0 ? -diff : 0
        imageData.data[i + 2] = 0
        imageData.data[i + 3] = 255
      }
      let threshold = 400 / (mipLevel + 1) ** 2
      await assert.imagesTest([refImage, computedImage, imageData], ['pysift reference', 'computed', `magnified diff`], `pysift comparison ${fileName},  real diff range is ${max_diff - min_diff + 1}/255, sqDiff: ${sqDiff.toFixed(1)} < ${threshold.toFixed(1)}`, sqDiff < threshold)
    }

    for (let mip = 0; mip < 6; mip++) {
      for (let j = 0; j < 6; j++) {
        await compareToPysift(`pysift_ref/gaussian_image_mip${mip}_${j}.png`, 'outputTexture', mip, j, false)
      }
    }
    for (let mip = 0; mip < 6; mip++) {
      for (let j = 0; j < 5; j++) {
        await compareToPysift(`pysift_ref/dog_mip${mip}_${j}.png`, 'diffTexture', mip, j, true)
      }
    }

    const extremaCounter = new Uint32Array(await allocatedShader.getBuffer(allocatedShader.totalExtremaBuffer))
    const extremaBuffer = new Float32Array(await allocatedShader.getBuffer(allocatedShader.extremaBuffer, extremaCounter[0] * 4 * 4))
    console.log('extremaCounter', extremaCounter)
    let totalExtrema = 0
    const extrema = []
    for (let i = 0; i < extremaBuffer.length; i += 4) {
      const extremum = extremaBuffer.slice(i, (i + 4))
      if (extremum[0] > 0 && extremum[1] > 0) {
        totalExtrema++
        extrema.push(extremum)
      }
    }

    console.log('totalExtrema', totalExtrema)
    console.log('extrema', extrema)
    const pysiftKeypoints = await (await fetch('pysift_ref/raw_keypoints.json')).json()
    console.log('pysift_ref', pysiftKeypoints)
    const textureName = 'diffTexture'
    const texture = allocatedShader[textureName]
    const contexts = []
    for (let mipLevel = 0; mipLevel < texture.mipLevelCount; mipLevel++) {
      contexts.push([])
      for (let index = 0; index < texture.depthOrArrayLayers; index++) {
        let texture = await allocatedShader.getTexture(textureName, mipLevel, index)
        const canvas = new OffscreenCanvas(texture.width, texture.height)
        let context = canvas.getContext('2d')
        contexts[contexts.length - 1].push(context)
        context.putImageData(texture, 0, 0)
        context.fillStyle = 'green'
      }
    }
    const perImageSets = {}

    function getSets (key) {
      let pointSet = perImageSets[key]
      if (pointSet == null) {
        pointSet = {computed: new Set(), expected: new Set()}
        perImageSets[key] = pointSet
      }
      return pointSet
    }

    for (const extremum of extrema) {
      let pointSet = getSets(`${extremum[3]}|${extremum[2]}`)
      pointSet.computed.add(`${extremum[0]}|${extremum[1]}|${extremum[3]}|${extremum[2]}`)
      const ctx = contexts[extremum[3]][extremum[2]]
      ctx.beginPath()
      ctx.arc(extremum[0] + 0.5, extremum[1] + 0.5, 2, 0, 2 * Math.PI)
      ctx.fill()
    }
    for (const pyExtremum of pysiftKeypoints) {
      let pointSet = getSets(`${pyExtremum.octave}|${pyExtremum.idx}`)
      pointSet.expected.add(`${pyExtremum.x}|${pyExtremum.y}|${pyExtremum.octave}|${pyExtremum.idx}`)
      const ctx = contexts[pyExtremum.octave][pyExtremum.idx]
      ctx.globalCompositeOperation = 'lighter'
      ctx.fillStyle = 'red'
      ctx.beginPath()
      ctx.arc(pyExtremum.x + 0.5, pyExtremum.y + 0.5, 2, 0, 2 * Math.PI)
      ctx.fill()
    }

    for (let i = 0; i < contexts.length; i++) {
      const mipCtx = contexts[i]
      for (let j = 0; j < mipCtx.length; j++) {
        const key = `${i}|${j}`
        let pointSet = getSets(key)
        let missRatio = pointSet.expected.symmetricDifference(pointSet.computed).size / pointSet.expected.size * 100
        if (pointSet.expected.size === 0)
          missRatio = 0
        console.log('## missRatio', key, missRatio, pointSet.expected.size)
        const context = mipCtx[j]
        const overlayTexture = context.getImageData(0, 0, context.canvas.width, context.canvas.height)
        await assert.imageTest(testImage, overlayTexture, `Extrema. stack index: ${j}, mip level: ${i} 
        miss ratio: ${missRatio.toFixed(1)}%, red is pysift, green is computed, orange is overlap`, missRatio < 22)
      }
    }
  })

  QUnit.test('Gaussian Blur Shader', async assert => {

    try {
      console.log('creating shader')
      let ts = await RadialShader.createShaders(device, kernelRadius, workgroupSize)

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

      const quenelle = computeGaussianKernel(kernelRadius / 3, kernelRadius)

      let allocatedShader = await ts.createGPUResources(workgroupSize, inputImage, width, height, [quenelle], HORIZONTAL)
      let outputData = await allocatedShader.runShader()

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

      const euclideanDistance = (a, b) => Math.hypot(...Object.keys(a).map(k => b[k] - a[k]))
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
      allocatedShader = await ts.createGPUResources(workgroupSize, inputImage, width, height, [quenelle], VERTICAL)
      outputData = await allocatedShader.runShader()
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
      ts = await RadialShader.createShaders(device)
      allocatedShader = await ts.createGPUResources(workgroupSize, inputImage, inputImage.width, inputImage.height, [quenelle])
      outputImage = await allocatedShader.runShader()
      await assert.imageTest(inputImage, outputImage, `Blur, running both passes. GPU time: ${allocatedShader.gpuTime}`, whiteColumnUntouched)

      let testImage = await createImageBitmap(await (await fetch('3916587d9b.png')).blob())
      ts = await RadialShader.createShaders(device)
      allocatedShader = await ts.createGPUResources(workgroupSize, testImage, testImage.width, testImage.height, [quenelle])
      outputImage = await allocatedShader.runShader()
      await assert.imageTest(testImage, outputImage, `Complete blur: Input and output images. GPU time: ${allocatedShader.gpuTime}`, true)

      console.time('createImageBitmap')
      testImage = await createImageBitmap(await (await fetch('NASM-A20150317000-NASM2018-10769.jpg')).blob())
      console.log('testImage size', testImage.width, testImage.height)
      console.timeEnd('createImageBitmap')
      ts = await RadialShader.createShaders(device)
      const gaussians = [1.2262734984654078, 1.5450077936447955, 1.9465878414647133, 2.4525469969308156, 3.090015587289591]
      const gs = gaussians.map(sigma => computeGaussianKernel(sigma))
      allocatedShader = await ts.createGPUResources(workgroupSize, testImage, testImage.width, testImage.height, gs)
      outputImage = await allocatedShader.runShader()
      const index = 3
      const mipLevel = 4
      await assert.imageTest(testImage, await allocatedShader.getTexture('outputTexture', mipLevel, index), `Complete blur on big image stack index: ${index}, mip level: ${mipLevel}`, true)
      if (false) {
        for (let mipLevel = 0; mipLevel < 8; mipLevel++) {
          for (let index = 0; index < gaussians.length; index++) {
            await assert.imageTest(testImage, await allocatedShader.getTexture('outputTexture', mipLevel, index), `Complete blur on big image stack index: ${index}, mip level: ${mipLevel}`, true)
          }
        }
      }
    } catch (error) {
      console.error('Error testing Gaussian Blur Shader:', error)
      assert.ok(false, `Error testing Gaussian Blur Shader: ${error.message}`)
    }
  })
}
