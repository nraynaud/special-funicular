import * as twgl from 'https://cdnjs.cloudflare.com/ajax/libs/twgl.js/6.1.0/twgl-full.module.js'

const vertexShader = `
    attribute vec2 a_position;
    varying vec2 v_texCoord;
    // homogenous sampling coordinate transform
    uniform mat3 u_imageTransform;
    void main() {
        gl_Position = vec4(a_position * vec2(1, -1), 0, 1);
        vec2 a_texCoord = (a_position.xy + vec2(1.0))/2.0;
        v_texCoord = (u_imageTransform * vec3(a_texCoord, 1.0)).xy;
    }
  `

// 1. Shader definitions
/**
 *
 * @param kernelRadius pixel radius, including the center element.
 * @returns {string} the shader for a single pass of the separable kernel
 */
function generateRadialKernelShader (kernelRadius) {
  // language=GLSL
  return `
      precision mediump float;
      const int kernelRadius = ${kernelRadius};
      uniform sampler2D u_image;
      uniform float u_radialKernel[kernelRadius];
      uniform vec2 u_neighborDir;
      varying vec2 v_texCoord;
      void main() {
          vec3 result = vec3(0.0);
          for (float i = float(-kernelRadius+1); i < float(kernelRadius); i++) {
              result+=u_radialKernel[int(abs(i))] * texture2D(u_image, v_texCoord + u_neighborDir * i).rgb;
          }
          gl_FragColor = vec4(result, 1.0);
      }
  `
}

// language=GLSL
const derivativeShader = `
    #extension GL_OES_standard_derivatives : enable
    precision mediump float;
    uniform sampler2D u_image;
    varying vec2 v_texCoord;
    void main() {
        vec3 texel = texture2D(u_image, v_texCoord).rgb;
        vec3 dx = dFdx(texel);
        vec3 dy = dFdy(texel);
        // DiZenzo: Tensor Gradient
        float gxx = dot(dx, dx);
        float gyy = dot(dy, dy);
        float gxy = dot(dx, dy);
        vec2 gradVec = vec2(gxx-gyy, 2.0*gxy);
        float magnitude = length(gradVec);
        gl_FragColor = vec4(gradVec*10.0, magnitude*10.0, 1.0);
    }
`

/**
 *
 * @param scale
 * @param sigma_scale
 * @returns {any[]} a radial kernel, first element is the center, last element is at perimeter
 */
function generateGaussianRadialKernel (scale = 0.8, sigma_scale = 0.6) {
  // https://github.com/suryanshkumar/Line-Segment-De tector/blob/master/lsd.c
  const sigma = scale < 1.0 ? sigma_scale / scale : sigma_scale
  const prec = 3.0
  const kernel_radius = Math.ceil(sigma * Math.sqrt(2.0 * prec * Math.log(10.0)))

  //https://stackoverflow.com/a/8204886
  function generateGaussian (x, mu, sigma) {
    return Math.exp(-(((x - mu) / (sigma)) ** 2) / 2)
  }

  const kernel = new Array(kernel_radius * 2 + 1)
  for (const i of kernel.keys()) {
    kernel[i] = generateGaussian(i, 0, sigma)
  }
  return normalizeRadialKernel(kernel)
}

/**
 *
 * @param radialKernel by convention, element 0 is the center and is counted only once
 */
function normalizeRadialKernel (radialKernel) {
  // trashy, I ain't smart enough to compute the volume
  let sum = 0
  for (let i = -radialKernel.length + 1; i < radialKernel.length; i++) {
    sum += radialKernel[Math.abs(i)]
  }
  return radialKernel.map(v => v / sum)
}

const gaussKernel = generateGaussianRadialKernel()
console.log('generateGaussianRadialKernel', gaussKernel)
const gaussShader = generateRadialKernelShader(gaussKernel.length)
console.log('gaussShader', gaussShader)

/**
 * a bit like shadertoy, runs a fragment shader on a quad.
 * caller has to set the render target beforehand.
 * @param gl
 * @param fragmentShaderId
 * @param imageUniform input texture
 * @param extraUniforms a mat3 homogenous matrix to transform the image on the fly
 * @returns {Promise<void>}
 */
async function run2DShader (gl, fragmentShaderId, imageUniform, extraUniforms = {},) {
  const defautVShader = vertexShader
  let mesh
  if (gl.__2Dmesh) {
    mesh = gl.__2Dmesh
  } else {
    // just a quad in strip form
    const vertices = [-1, -1, 0,
      1, -1, 0,
      -1, 1, 0,
      1, 1, 0]
    mesh = twgl.createBufferInfoFromArrays(gl, {a_position: vertices})
    gl.__2Dmesh = mesh
  }
  console.log('createPlaneBufferInfo', mesh)
  const shaderCache = gl._programs = gl._programs || new Map()
  if (!shaderCache.has(defautVShader)) {
    shaderCache.set(defautVShader, new Map())
  }
  const fShaders = shaderCache.get(defautVShader)
  console.log('fShaders', fShaders)
  let programInfo
  if (!fShaders.has(fragmentShaderId)) {
    programInfo = twgl.createProgramInfo(gl, [defautVShader, fragmentShaderId])
    fShaders.set(fragmentShaderId, programInfo)
  } else {
    console.log('got program from cache')
    programInfo = fShaders.get(fragmentShaderId)
  }

  console.log('ProgramInfoLog', gl.getProgramInfoLog(programInfo.program))
  gl.useProgram(programInfo.program)
  twgl.setBuffersAndAttributes(gl, programInfo, mesh)
  twgl.setUniforms(programInfo, {
    u_image: imageUniform,
    u_imageTransform: DEFAULT_IMAGE_TRANSFORM,
    ...extraUniforms
  })
  twgl.drawBufferInfo(gl, mesh, gl.TRIANGLE_STRIP)
  gl.flush()
  gl.finish()
}

const DEFAULT_IMAGE_TRANSFORM = [
  1, 0, 0,
  0, 1, 0,
  0, 0, 1
]
const FLIP_UPSIDE_DOWN_IMAGE_TRANSFORM = [
  1, 0, 0,
  0, -1, 0,
  0, 1, 1
]
const FLIPXY_IMAGE_TRANSFORM = [
  0, 1, 0,
  1, 0, 0,
  0, 0, 1
]

onmessage = async (e) => {
  const {imageUrl} = e.data
  const blob = await (await fetch(imageUrl)).blob()
  const imageBitmap = await createImageBitmap(blob)
  console.log('imageBitmap', imageBitmap)
  console.log('worker message', imageBitmap.width, imageBitmap.height)
  const offscreen = new OffscreenCanvas(imageBitmap.width, imageBitmap.height)
  const gl = offscreen.getContext('webgl')
  twgl.addExtensionsToContext(gl)

  const favoriteTexParams = {
    minMag: gl.LINEAR,
    level: 0,
    auto: false,
    wrap: gl.CLAMP_TO_EDGE
  }

  const fbi1 = twgl.createFramebufferInfo(gl, [{
    format: gl.RGB,
    type: gl.FLOAT,
    ...favoriteTexParams
  }])

  const fbi2 = twgl.createFramebufferInfo(gl, [{
    format: gl.RGB,
    type: gl.FLOAT,
    ...favoriteTexParams
  }])

  const texture = await twgl.createTexture(gl, {
    src: imageBitmap,
    flipY: true,
    ...favoriteTexParams
  })

  // First pass - horizontal gaussian blur
  twgl.bindFramebufferInfo(gl, fbi2)
  await run2DShader(gl, gaussShader, texture, {
    u_radialKernel: gaussKernel,
    u_neighborDir: [1 / imageBitmap.width, 0]
  })

  // Second pass - vertical gaussian blur
  twgl.bindFramebufferInfo(gl, fbi1)
  await run2DShader(gl, gaussShader, fbi2.attachments[0], {
    u_radialKernel: gaussKernel,
    u_neighborDir: [0, 1 / imageBitmap.height]
  })

  // Final pass - derivative computation
  twgl.bindFramebufferInfo(gl, null)
  await run2DShader(gl, derivativeShader, fbi1.attachments[0])

  const outBitmap = await offscreen.convertToBlob()
  const url = URL.createObjectURL(outBitmap)
  postMessage({
    type: 'processed',
    result: url
  },)
}
