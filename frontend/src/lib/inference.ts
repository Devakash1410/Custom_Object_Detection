/**
 * ONNX model inference and preprocessing.
 *
 * - Dynamically parses output dimensions (no hardcoded [1, c+4, 8400]).
 * - Handles different model sizes and class counts.
 * - Designed to run inside a Web Worker.
 */

import * as ort from "onnxruntime-web";
import type { ModelMetadata, Detection, InferenceResult } from "@/types/detection";
import { getClassColor } from "./colors";
import { applyNMS } from "./postprocess";

let session: ort.InferenceSession | null = null;
let metadata: ModelMetadata | null = null;
let classNames: string[] = [];

/**
 * Load the ONNX model and metadata.
 */
export async function loadModel(
  modelUrl: string,
  metadataUrl: string,
  labelsUrl: string
): Promise<ModelMetadata> {
  // Load metadata
  const metaResp = await fetch(metadataUrl);
  if (metaResp.ok) {
    metadata = await metaResp.json();
  }

  // Load labels
  const labelsResp = await fetch(labelsUrl);
  if (labelsResp.ok) {
    classNames = await labelsResp.json();
  }

  // If metadata missing, infer from labels
  if (!metadata && classNames.length > 0) {
    metadata = {
      num_classes: classNames.length,
      class_names: classNames,
      input_size: 640,
    };
  }

  // Create inference session
  session = await ort.InferenceSession.create(modelUrl, {
    executionProviders: ["wasm"],
    graphOptimizationLevel: "all",
  });

  return metadata!;
}

/**
 * Preprocess an image for YOLOv8 inference.
 * Resize to inputSize x inputSize, normalize to [0,1], convert to CHW float32.
 */
export function preprocessImage(
  imageData: ImageData,
  inputSize: number
): { tensor: ort.Tensor; xRatio: number; yRatio: number } {
  const { width: srcW, height: srcH, data } = imageData;

  // Create a temporary canvas for resizing
  const canvas = new OffscreenCanvas(inputSize, inputSize);
  const ctx = canvas.getContext("2d")!;

  // Letterbox: scale preserving aspect ratio, pad with gray
  const scale = Math.min(inputSize / srcW, inputSize / srcH);
  const newW = Math.round(srcW * scale);
  const newH = Math.round(srcH * scale);
  const padX = (inputSize - newW) / 2;
  const padY = (inputSize - newH) / 2;

  ctx.fillStyle = "#808080";
  ctx.fillRect(0, 0, inputSize, inputSize);

  // Draw the source image (create ImageBitmap for OffscreenCanvas)
  const srcCanvas = new OffscreenCanvas(srcW, srcH);
  const srcCtx = srcCanvas.getContext("2d")!;
  srcCtx.putImageData(imageData, 0, 0);
  ctx.drawImage(srcCanvas, padX, padY, newW, newH);

  // Extract pixel data
  const resized = ctx.getImageData(0, 0, inputSize, inputSize);
  const pixels = resized.data;

  // Convert to CHW float32 normalized [0, 1]
  const numPixels = inputSize * inputSize;
  const float32Data = new Float32Array(3 * numPixels);

  for (let i = 0; i < numPixels; i++) {
    const idx = i * 4;
    float32Data[i] = pixels[idx] / 255.0;                    // R
    float32Data[numPixels + i] = pixels[idx + 1] / 255.0;    // G
    float32Data[2 * numPixels + i] = pixels[idx + 2] / 255.0; // B
  }

  const tensor = new ort.Tensor("float32", float32Data, [1, 3, inputSize, inputSize]);

  return {
    tensor,
    xRatio: srcW / newW,
    yRatio: srcH / newH,
  };
}

/**
 * Run inference and return detections.
 * Dynamically parses output shape — works with any YOLOv8 model.
 */
export async function runInference(
  imageData: ImageData,
  inputSize: number,
  confidenceThreshold: number,
  iouThreshold: number = 0.45
): Promise<InferenceResult> {
  if (!session || !metadata) {
    throw new Error("Model not loaded. Call loadModel() first.");
  }

  const t0 = performance.now();

  // Preprocess
  const { tensor, xRatio, yRatio } = preprocessImage(imageData, inputSize);
  const t1 = performance.now();

  // Inference
  const inputName = session.inputNames[0];
  const feeds: Record<string, ort.Tensor> = { [inputName]: tensor };
  const results = await session.run(feeds);
  const t2 = performance.now();

  // Get output tensor — dynamically parse shape
  const outputName = session.outputNames[0];
  const output = results[outputName];
  const outputData = output.data as Float32Array;
  const outputDims = output.dims as number[];

  // YOLOv8 output: [1, (4 + nc), numDetections] or [1, numDetections, (4 + nc)]
  // Detect layout dynamically
  const nc = metadata.num_classes;
  const expectedChannels = 4 + nc;

  let numDetections: number;
  let transposed = false;

  if (outputDims.length === 3) {
    if (outputDims[1] === expectedChannels) {
      // Shape: [1, 4+nc, N] — needs transpose
      numDetections = outputDims[2];
      transposed = true;
    } else if (outputDims[2] === expectedChannels) {
      // Shape: [1, N, 4+nc] — already in row format
      numDetections = outputDims[1];
      transposed = false;
    } else {
      // Try to infer: smaller dim is likely channels
      if (outputDims[1] < outputDims[2]) {
        numDetections = outputDims[2];
        transposed = true;
      } else {
        numDetections = outputDims[1];
        transposed = false;
      }
    }
  } else {
    throw new Error(`Unexpected output dims: [${outputDims.join(", ")}]`);
  }

  // Parse detections
  const rawDetections: Detection[] = [];

  // Compute pad offsets for letterbox
  const padX = (inputSize - (imageData.width / xRatio)) / 2;
  const padY = (inputSize - (imageData.height / yRatio)) / 2;

  for (let i = 0; i < numDetections; i++) {
    let cx: number, cy: number, w: number, h: number;
    let maxScore = 0;
    let maxClassId = 0;

    if (transposed) {
      // Shape [1, 4+nc, N]: column-major
      cx = outputData[0 * numDetections + i];
      cy = outputData[1 * numDetections + i];
      w = outputData[2 * numDetections + i];
      h = outputData[3 * numDetections + i];

      for (let c = 0; c < nc; c++) {
        const score = outputData[(4 + c) * numDetections + i];
        if (score > maxScore) {
          maxScore = score;
          maxClassId = c;
        }
      }
    } else {
      // Shape [1, N, 4+nc]: row-major
      const offset = i * expectedChannels;
      cx = outputData[offset];
      cy = outputData[offset + 1];
      w = outputData[offset + 2];
      h = outputData[offset + 3];

      for (let c = 0; c < nc; c++) {
        const score = outputData[offset + 4 + c];
        if (score > maxScore) {
          maxScore = score;
          maxClassId = c;
        }
      }
    }

    if (maxScore < confidenceThreshold) continue;

    // Convert from model coords to original image coords
    const x1 = (cx - w / 2 - padX) * xRatio;
    const y1 = (cy - h / 2 - padY) * yRatio;
    const x2 = (cx + w / 2 - padX) * xRatio;
    const y2 = (cy + h / 2 - padY) * yRatio;

    rawDetections.push({
      bbox: {
        x1: Math.max(0, x1),
        y1: Math.max(0, y1),
        x2: Math.max(0, x2),
        y2: Math.max(0, y2),
        width: Math.max(0, x2 - x1),
        height: Math.max(0, y2 - y1),
      },
      classId: maxClassId,
      className: classNames[maxClassId] || `class_${maxClassId}`,
      confidence: maxScore,
      color: getClassColor(maxClassId),
    });
  }

  // Apply NMS
  const detections = applyNMS(rawDetections, iouThreshold);
  const t3 = performance.now();

  return {
    detections,
    preprocessTimeMs: t1 - t0,
    inferenceTimeMs: t2 - t1,
    postprocessTimeMs: t3 - t2,
    totalTimeMs: t3 - t0,
  };
}

export function isModelLoaded(): boolean {
  return session !== null && metadata !== null;
}

export function getMetadata(): ModelMetadata | null {
  return metadata;
}
