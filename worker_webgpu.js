// WebGPU SIFT Feature Extraction Worker
// This worker implements SIFT (Scale-Invariant Feature Transform) feature extraction using WebGPU

// Import shared shader code and constants
import {
  NUM_OCTAVES,
  SCALES_PER_OCTAVE,
  SIGMA_INITIAL,
  SIGMA_MULTIPLIER,
  CONTRAST_THRESHOLD,
  EDGE_THRESHOLD,
  MAX_KEYPOINTS,
  gaussianBlurShader,
  dogShader
} from './sift-shaders.js';

/**
 * Extract SIFT features from an image using WebGPU
 * @param {GPUDevice} device - The WebGPU device
 * @param {ImageData} imageData - The image data to process
 * @returns {Promise<Array>} - Array of SIFT keypoints
 */
async function extractSIFTFeatures(device, imageData) {
  return []
  debugLog('extractSIFTFeatures: Starting extraction');
  const { width, height, data } = imageData;
  debugLog(`extractSIFTFeatures: Image dimensions: ${width}x${height}`);

  // Create input texture from image data
  debugLog('extractSIFTFeatures: Creating input texture');
  const textureSize = [width, height];
  const inputTexture = device.createTexture({
    label: 'SIFT Input Texture',
    size: textureSize,
    format: 'rgba8unorm',
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.COPY_SRC | GPUTextureUsage.RENDER_ATTACHMENT
  });

  debugLog('extractSIFTFeatures: Writing image data to texture');
  try {
    device.queue.writeTexture(
      { texture: inputTexture },
      data,
      { bytesPerRow: width * 4 },
      { width, height }
    );
    debugLog('extractSIFTFeatures: Successfully wrote image data to texture');
  } catch (error) {
    debugLog('extractSIFTFeatures: Error writing to texture', {
      error: error.message,
      bytesPerRow: width * 4,
      width,
      height,
      dataLength: data.length
    });
    throw error;
  }

  // Create pipeline for Gaussian blur
  debugLog('extractSIFTFeatures: Creating Gaussian blur shader module');
  let gaussianModule;
  try {
    gaussianModule = device.createShaderModule({
      label: 'Gaussian Blur Shader',
      code: gaussianBlurShader
    });
    debugLog('extractSIFTFeatures: Gaussian blur shader module created successfully');
  } catch (error) {
    debugLog('extractSIFTFeatures: Error creating Gaussian blur shader module', {
      error: error.message,
      shaderLength: gaussianBlurShader.length
    });
    throw error;
  }

  debugLog('extractSIFTFeatures: Creating Gaussian blur pipeline');
  let gaussianPipeline;
  try {
    gaussianPipeline = device.createComputePipeline({
      label: 'Gaussian Blur Pipeline',
      layout: 'auto',
      compute: {
        module: gaussianModule,
        entryPoint: 'main'
      }
    });
    debugLog('extractSIFTFeatures: Gaussian blur pipeline created successfully');
  } catch (error) {
    debugLog('extractSIFTFeatures: Error creating Gaussian blur pipeline', {
      error: error.message
    });
    throw error;
  }

  // Create pipeline for DoG
  debugLog('extractSIFTFeatures: Creating DoG shader module');
  let dogModule;
  try {
    dogModule = device.createShaderModule({
      label: 'Difference of Gaussians Shader',
      code: dogShader
    });
    debugLog('extractSIFTFeatures: DoG shader module created successfully');
  } catch (error) {
    debugLog('extractSIFTFeatures: Error creating DoG shader module', {
      error: error.message,
      shaderLength: dogShader.length
    });
    throw error;
  }

  debugLog('extractSIFTFeatures: Creating DoG pipeline');
  let dogPipeline;
  try {
    dogPipeline = device.createComputePipeline({
      label: 'Difference of Gaussians Pipeline',
      layout: 'auto',
      compute: {
        module: dogModule,
        entryPoint: 'main'
      }
    });
    debugLog('extractSIFTFeatures: DoG pipeline created successfully');
  } catch (error) {
    debugLog('extractSIFTFeatures: Error creating DoG pipeline', {
      error: error.message
    });
    throw error;
  }


  // Note: Keypoint detection functionality has been removed

  // Create textures for Gaussian pyramid
  const gaussianPyramid = [];
  const dogPyramid = [];

  // For each octave
  for (let octave = 0; octave < NUM_OCTAVES; octave++) {
    const octaveWidth = Math.floor(width / Math.pow(2, octave));
    const octaveHeight = Math.floor(height / Math.pow(2, octave));

    if (octaveWidth < 8 || octaveHeight < 8) {
      continue; // Skip if the image is too small
    }

    const octaveTextures = [];
    const octaveDoGTextures = [];

    // For each scale in the octave
    for (let scale = 0; scale < SCALES_PER_OCTAVE + 2; scale++) {
      // Create texture for this scale
      const scaleTexture = device.createTexture({
        label: `Gaussian Pyramid Octave ${octave} Scale ${scale}`,
        size: [octaveWidth, octaveHeight],
        format: 'rgba8unorm',
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING |
               GPUTextureUsage.COPY_DST | GPUTextureUsage.COPY_SRC
      });

      octaveTextures.push(scaleTexture);

      // Create DoG texture (except for the first scale)
      if (scale > 0) {
        const dogTexture = device.createTexture({
          label: `DoG Pyramid Octave ${octave} Scale ${scale-1}`,
          size: [octaveWidth, octaveHeight],
          format: 'rgba8unorm',
          usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING |
                 GPUTextureUsage.COPY_DST | GPUTextureUsage.COPY_SRC
        });

        octaveDoGTextures.push(dogTexture);
      }
    }

    gaussianPyramid.push(octaveTextures);
    dogPyramid.push(octaveDoGTextures);
  }

  // Build Gaussian pyramid
  for (let octave = 0; octave < NUM_OCTAVES; octave++) {
    const octaveWidth = Math.floor(width / Math.pow(2, octave));
    const octaveHeight = Math.floor(height / Math.pow(2, octave));

    if (octaveWidth < 8 || octaveHeight < 8) {
      continue;
    }

    const octaveTextures = gaussianPyramid[octave];

    // For the first octave, use the input image
    // For subsequent octaves, downsample from the previous octave
    if (octave === 0) {
      // Copy input texture to first scale of first octave
      const commandEncoder = device.createCommandEncoder();
      commandEncoder.copyTextureToTexture(
        { texture: inputTexture },
        { texture: octaveTextures[0] },
        [octaveWidth, octaveHeight]
      );
      device.queue.submit([commandEncoder.finish()]);
    } else {
      // Downsample from previous octave
      const prevOctaveTexture = gaussianPyramid[octave - 1][0];

      // Create a simple downsampling shader
      const downsampleShader = `
        @group(0) @binding(0) var inputTexture: texture_2d<f32>;
        @group(0) @binding(1) var outputTexture: texture_storage_2d<rgba8unorm, write>;

        @compute @workgroup_size(16, 16)
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
          let outputPos = vec2<i32>(global_id.xy);
          let inputPos = outputPos * 2;

          let dimensions = vec2<i32>(textureDimensions(outputTexture));
          if (outputPos.x >= dimensions.x || outputPos.y >= dimensions.y) {
            return;
          }

          let texel = textureLoad(inputTexture, inputPos, 0);
          textureStore(outputTexture, outputPos, texel);
        }
      `;

      const downsampleModule = device.createShaderModule({
        label: `Downsample Shader Octave ${octave}`,
        code: downsampleShader
      });

      const downsamplePipeline = device.createComputePipeline({
        label: `Downsample Pipeline Octave ${octave}`,
        layout: 'auto',
        compute: {
          module: downsampleModule,
          entryPoint: 'main'
        }
      });

      const bindGroup = device.createBindGroup({
        label: `Downsample Bind Group Octave ${octave}`,
        layout: downsamplePipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: prevOctaveTexture.createView() },
          { binding: 1, resource: octaveTextures[0].createView() }
        ]
      });

      const commandEncoder = device.createCommandEncoder({
        label: `Downsample Command Encoder Octave ${octave}`
      });
      const computePass = commandEncoder.beginComputePass({
        label: `Downsample Compute Pass Octave ${octave}`
      });
      computePass.setPipeline(downsamplePipeline);
      computePass.setBindGroup(0, bindGroup);
      computePass.dispatchWorkgroups(
        Math.ceil(octaveWidth / 16),
        Math.ceil(octaveHeight / 16)
      );
      computePass.end();
      device.queue.submit([commandEncoder.finish()]);
    }

    // Generate the rest of the scales in this octave using Gaussian blur
    for (let scale = 1; scale < SCALES_PER_OCTAVE + 2; scale++) {
      const sigma = SIGMA_INITIAL * Math.pow(SIGMA_MULTIPLIER, scale / SCALES_PER_OCTAVE);
      const prevSigma = SIGMA_INITIAL * Math.pow(SIGMA_MULTIPLIER, (scale - 1) / SCALES_PER_OCTAVE);

      // Calculate the incremental sigma needed to go from prevSigma to sigma
      const sigmaDiff = Math.sqrt(sigma * sigma - prevSigma * prevSigma);

      // Create a temporary texture for the horizontal pass
      const tempTexture = device.createTexture({
        label: `Gaussian Blur Temp Texture Octave ${octave} Scale ${scale}`,
        size: [octaveWidth, octaveHeight],
        format: 'rgba8unorm',
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING
      });

      // Create uniform buffer for Gaussian parameters
      const paramsBuffer = device.createBuffer({
        label: `Gaussian Params Buffer Octave ${octave} Scale ${scale}`,
        size: 24, // 5 floats (sigma, dirX, dirY, imageWidth, imageHeight) + padding to meet 24-byte minimum
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
      });

      // Horizontal pass
      device.queue.writeBuffer(
        paramsBuffer,
        0,
        new Float32Array([sigmaDiff, 1.0, 0.0, octaveWidth, octaveHeight])
      );

      let bindGroup = device.createBindGroup({
        label: `Gaussian Horizontal Bind Group Octave ${octave} Scale ${scale}`,
        layout: gaussianPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: octaveTextures[scale - 1].createView() },
          { binding: 1, resource: tempTexture.createView() },
          { binding: 2, resource: { buffer: paramsBuffer } }
        ]
      });

      let commandEncoder = device.createCommandEncoder({
        label: `Gaussian Horizontal Command Encoder Octave ${octave} Scale ${scale}`
      });
      let computePass = commandEncoder.beginComputePass({
        label: `Gaussian Horizontal Compute Pass Octave ${octave} Scale ${scale}`
      });
      computePass.setPipeline(gaussianPipeline);
      computePass.setBindGroup(0, bindGroup);
      computePass.dispatchWorkgroups(
        Math.ceil(octaveWidth / 16),
        Math.ceil(octaveHeight / 16)
      );
      computePass.end();
      device.queue.submit([commandEncoder.finish()]);

      // Vertical pass
      device.queue.writeBuffer(
        paramsBuffer,
        0,
        new Float32Array([sigmaDiff, 0.0, 1.0, octaveWidth, octaveHeight])
      );

      bindGroup = device.createBindGroup({
        label: `Gaussian Vertical Bind Group Octave ${octave} Scale ${scale}`,
        layout: gaussianPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: tempTexture.createView() },
          { binding: 1, resource: octaveTextures[scale].createView() },
          { binding: 2, resource: { buffer: paramsBuffer } }
        ]
      });

      commandEncoder = device.createCommandEncoder({
        label: `Gaussian Vertical Command Encoder Octave ${octave} Scale ${scale}`
      });
      computePass = commandEncoder.beginComputePass({
        label: `Gaussian Vertical Compute Pass Octave ${octave} Scale ${scale}`
      });
      computePass.setPipeline(gaussianPipeline);
      computePass.setBindGroup(0, bindGroup);
      computePass.dispatchWorkgroups(
        Math.ceil(octaveWidth / 16),
        Math.ceil(octaveHeight / 16)
      );
      computePass.end();
      device.queue.submit([commandEncoder.finish()]);

      // Compute DoG
      bindGroup = device.createBindGroup({
        label: `DoG Bind Group Octave ${octave} Scale ${scale}`,
        layout: dogPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: octaveTextures[scale - 1].createView() },
          { binding: 1, resource: octaveTextures[scale].createView() },
          { binding: 2, resource: dogPyramid[octave][scale - 1].createView() }
        ]
      });

      commandEncoder = device.createCommandEncoder({
        label: `DoG Command Encoder Octave ${octave} Scale ${scale}`
      });
      computePass = commandEncoder.beginComputePass({
        label: `DoG Compute Pass Octave ${octave} Scale ${scale}`
      });
      computePass.setPipeline(dogPipeline);
      computePass.setBindGroup(0, bindGroup);
      computePass.dispatchWorkgroups(
        Math.ceil(octaveWidth / 16),
        Math.ceil(octaveHeight / 16)
      );
      computePass.end();
      device.queue.submit([commandEncoder.finish()]);
    }
  }


  // Note: Keypoint detection functionality has been removed
  // Return an empty array of keypoints
  const keypoints = [];

  // Clean up resources
  inputTexture.destroy();
  for (const octaveTextures of gaussianPyramid) {
    for (const texture of octaveTextures) {
      texture.destroy();
    }
  }
  for (const octaveTextures of dogPyramid) {
    for (const texture of octaveTextures) {
      texture.destroy();
    }
  }

  return keypoints;
}

/**
 * Visualize SIFT features on an image
 * @param {GPUDevice} device - The WebGPU device
 * @param {ImageData} imageData - The original image data
 * @param {Array} features - Array of SIFT keypoints
 * @returns {Promise<string>} - URL of the visualized image
 */
async function visualizeFeatures(device, imageData, features) {
  // Create a canvas with the original image
  const { width, height, data } = imageData;
  const canvas = new OffscreenCanvas(width, height);
  const ctx = canvas.getContext('2d');

  // Draw the original image
  ctx.putImageData(imageData, 0, 0);

  // Convert to blob and then to URL
  const blob = await canvas.convertToBlob();
  const url = URL.createObjectURL(blob);

  return url;
}

// Debug utility function
function debugLog(message, data = null) {
  const logMessage = data ? `${message}: ${JSON.stringify(data)}` : message;
  console.log(logMessage);

  // Also post a debug message to the main thread
  postMessage({
    type: 'debug',
    message: logMessage
  });
}

// Main worker code
onmessage = async (e) => {
  const { imageUrl } = e.data;

  if (!imageUrl) {
    postMessage({
      type: 'error',
      message: 'No image URL provided'
    });
    return;
  }

  // Flag to track if an uncaptured error has occurred
  let hasUncapturedError = false;

  try {
    const response = await fetch(imageUrl);
    if (!response.ok) {
      throw new Error(`Failed to fetch image: ${response.status} ${response.statusText}`);
    }

    const blob = await response.blob();
    debugLog(`Image fetched, size: ${blob.size} bytes`);

    const imageBitmap = await createImageBitmap(blob);
    debugLog(`Image bitmap created, dimensions: ${imageBitmap.width}x${imageBitmap.height}`);

    // Create ImageData from the bitmap
    const canvas = new OffscreenCanvas(imageBitmap.width, imageBitmap.height);
    const ctx = canvas.getContext('2d');
    ctx.drawImage(imageBitmap, 0, 0);
    const imageData = ctx.getImageData(0, 0, imageBitmap.width, imageBitmap.height);
    debugLog(`ImageData created, dimensions: ${imageData.width}x${imageData.height}`);

    // Check if WebGPU is supported
    if (!navigator.gpu) {
      throw new Error('WebGPU not supported on this browser');
    }
    debugLog('WebGPU is supported');

    // Request adapter and device
    debugLog('Requesting WebGPU adapter');
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
      throw new Error('No appropriate GPU adapter found');
    }
    debugLog('WebGPU adapter obtained', {
      name: adapter.name,
      features: [...adapter.features].map(f => f.toString())
    });

    debugLog('Requesting WebGPU device');
    const device = await adapter.requestDevice({
      label: 'SIFT WebGPU Device'
    });
    debugLog('WebGPU device obtained');

    // Set up error handling for the device
    device.addEventListener('uncapturederror', (event) => {
      debugLog('WebGPU device error', {
        error: event.error.message
      });

      // Mark the program as failed and send error message back to main thread
      postMessage({
        type: 'error',
        message: `WebGPU uncaptured error: ${event.error.message}`
      });

      // Set the flag to stop further processing
      hasUncapturedError = true;
    });

    // Check if an uncaptured error has occurred before proceeding
    if (!hasUncapturedError) {
      // Extract SIFT features using the algorithm for all images
      debugLog('Starting SIFT feature extraction');
      const features = await extractSIFTFeatures(device, imageData);

      // Check again if an uncaptured error occurred during feature extraction
      if (!hasUncapturedError) {
        debugLog(`SIFT feature extraction complete, found ${features.length} features`);

        // Visualize the features on a canvas
        debugLog('Starting feature visualization');
        const visualizedImage = await visualizeFeatures(device, imageData, features);

        // Check again if an uncaptured error occurred during visualization
        if (!hasUncapturedError) {
          debugLog('Feature visualization complete');

          // Send the result back
          debugLog('Sending results back to main thread');
          postMessage({
            type: 'processed',
            result: visualizedImage,
            features: features
          });
        }
      }
    }
  } catch (error) {
    console.error('Error in WebGPU SIFT extraction:', error);
    debugLog('Error in WebGPU SIFT extraction', {
      name: error.name,
      message: error.message,
      stack: error.stack
    });
    postMessage({
      type: 'error',
      message: error.message || 'Unknown error in WebGPU SIFT extraction'
    });
  }
};
