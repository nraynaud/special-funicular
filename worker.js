import * as twgl from 'https://cdnjs.cloudflare.com/ajax/libs/twgl.js/6.1.0/twgl-full.module.js'

const vertexShader = `#version 300 es
    in vec2 a_position;
    out vec2 v_texCoord;
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
  console.log(`generating RadialKernelShader with radius ${kernelRadius}`)
  // language=GLSL
  return `#version 300 es
  #extension GL_ANGLE_shader_pixel_local_storage : require
  precision mediump float;
  layout(binding=0, rgba8ui) uniform mediump upixelLocalANGLE framebuffer;
  
  const int kernelRadius = ${kernelRadius};
  uniform sampler2D u_image;
  uniform float u_radialKernel[kernelRadius];
  uniform vec2 u_neighborDir;
  in vec2 v_texCoord;
  out vec4 fragColor;
  void main() {
      vec3 result = vec3(0.0);
      for (float i = float(-kernelRadius+1); i < float(kernelRadius); i++) {
          result += u_radialKernel[int(abs(i))] * texture(u_image, v_texCoord + u_neighborDir * i).rgb;
      }
      fragColor = vec4(result, 1.0);
  }
  `
}

// language=GLSL
const derivativeShader = `#version 300 es
precision mediump float;
uniform sampler2D u_image;
in vec2 v_texCoord;
out vec4 fragColor;
void main() {
    vec3 texel = texture(u_image, v_texCoord).rgb;
    vec3 dx = dFdx(texel);
    vec3 dy = dFdy(texel);
    float gxx = dot(dx, dx);
    float gyy = dot(dy, dy);
    float gxy = dot(dx, dy);
    vec2 gradVec = vec2(gxx-gyy, 2.0*gxy);
    float magnitude = length(gradVec);
    fragColor = vec4(gradVec*10.0, magnitude*10.0, 1.0);
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
const gaussShader = generateRadialKernelShader(gaussKernel.length)

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
  let programInfo
  if (!fShaders.has(fragmentShaderId)) {
    programInfo = twgl.createProgramInfo(gl, [defautVShader, fragmentShaderId], (error)=>{
      console.log("Error", error)
    })

    fShaders.set(fragmentShaderId, programInfo)
  } else {
    console.log('got program from cache')
    programInfo = fShaders.get(fragmentShaderId)
  }

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
  const gl = twgl.getContext(offscreen, {depth: false})
  twgl.addExtensionsToContext(gl)
  let pixExt = gl.getExtension('WEBGL_shader_pixel_local_storage')
  twgl.glEnumToString(pixExt, 0) // register the enums
  console.log("PLS is coherent:", pixExt.isCoherent())

  const favoriteTexParams = {
    minMag: gl.LINEAR,
    level: 0,
    auto: false,
    wrap: gl.CLAMP_TO_EDGE
  }

  const fbi1 = twgl.createFramebufferInfo(gl, [{
    format: gl.RGBA,
    type: gl.FLOAT,
    internalFormat: gl.RGBA32F,
    ...favoriteTexParams
  }])

  const fbi2 = twgl.createFramebufferInfo(gl, [{
    format: gl.RGBA,
    type: gl.FLOAT,
    internalFormat: gl.RGBA32F,
    ...favoriteTexParams
  }])
  //https://github.com/dsanders11/WebGL/blob/de50dfcfd907b07caaa8a92869a79c9e5f8d7820/sdk/demos/rive/bubbles.html#L267
  const pixelStorage =  gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, pixelStorage);
  gl.texStorage2D(gl.TEXTURE_2D, 1, gl.RGBA8, imageBitmap.width, imageBitmap.height);
  const fbi3 = twgl.createFramebufferInfo(gl, [{
    attachment:pixelStorage
  }])

  twgl.bindFramebufferInfo(gl, fbi2, gl.READ_FRAMEBUFFER)
  pixExt.framebufferTexturePixelLocalStorageWEBGL(0, pixelStorage, 0, 0 )

  pixExt.framebufferPixelLocalClearValuefvWEBGL(0, [0, 0, 0, .1]);
  const texture = await twgl.createTexture(gl, {
    src: imageBitmap,
    flipY: true,
    ...favoriteTexParams
  })

  pixExt.beginPixelLocalStorageWEBGL([pixExt.LOAD_OP_CLEAR_WEBGL]);
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

  pixExt.endPixelLocalStorageWEBGL([pixExt.STORE_OP_STORE_WEBGL]);

  const outBitmap = await offscreen.convertToBlob()
  const url = URL.createObjectURL(outBitmap)
  postMessage({
    type: 'processed',
    result: url
  },)
}
