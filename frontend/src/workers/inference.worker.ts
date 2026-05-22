/**
 * Inference Web Worker.
 *
 * Runs all heavy computation (preprocessing, model inference, NMS)
 * off the main UI thread to prevent camera lag and UI freezing.
 */

import * as ort from "onnxruntime-web";
import { getClassColor } from "../lib/colors";
import { applyNMS } from "../lib/postprocess";
import type { Detection, ModelMetadata, InferenceResult } from "../types/detection";

// Set WASM paths for onnxruntime-web
ort.env.wasm.numThreads = 1;

let session: ort.InferenceSession | null = null;
let metadata: ModelMetadata | null = null;
let classNames: string[] = [];

/**
 * Handle messages from main thread.
 */
self.onmessage = async (e: MessageEvent) => {
  const { type, payload } = e.data;

  switch (type) {
    case "load-model":
      await handleLoadModel(payload);
      break;
    case "run-inference":
      await handleRunInference(payload);
      break;
  }
};

async function handleLoadModel(payload: {
  modelUrl: string;
  metadataUrl: string;
  labelsUrl: string;
}) {
  try {
    // Load metadata
    const metaResp = await fetch(payload.metadataUrl);
    if (metaResp.ok) {
      metadata = await metaResp.json();
    }

    // Load labels
    const labelsResp = await fetch(payload.labelsUrl);
    if (labelsResp.ok) {
      classNames = await labelsResp.json();
    }

    if (!metadata && classNames.length > 0) {
      metadata = {
        num_classes: classNames.length,
        class_names: classNames,
        input_size: 640,
      };
    }

    // Create session
    session = await ort.InferenceSession.create(payload.modelUrl, {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all",
    });

    self.postMessage({
      type: "model-loaded",
      payload: metadata,
    });
  } catch (err) {
    self.postMessage({
      type: "model-error",
      payload: String(err),
    });
  }
}

async function handleRunInference(payload: {
  imageData: { data: Uint8ClampedArray; width: number; height: number };
  inputSize: number;
  confidenceThreshold: number;
  iouThreshold: number;
}) {
  if (!session || !metadata) {
    self.postMessage({
      type: "inference-error",
      payload: "Model not loaded",
    });
    return;
  }

  try {
    const { imageData, inputSize, confidenceThreshold, iouThreshold } = payload;
    const t0 = performance.now();

    // --- Preprocess ---
    const srcW = imageData.width;
    const srcH = imageData.height;
    const pixels = imageData.data;

    const scale = Math.min(inputSize / srcW, inputSize / srcH);
    const newW = Math.round(srcW * scale);
    const newH = Math.round(srcH * scale);
    const padX = Math.round((inputSize - newW) / 2);
    const padY = Math.round((inputSize - newH) / 2);

    const numPixels = inputSize * inputSize;
    const float32Data = new Float32Array(3 * numPixels);

    // Fill with 0.5 (gray padding = 128/255)
    float32Data.fill(128 / 255);

    // Bilinear-ish sampling from source
    for (let dy = 0; dy < newH; dy++) {
      for (let dx = 0; dx < newW; dx++) {
        const sx = Math.min(Math.round(dx / scale), srcW - 1);
        const sy = Math.min(Math.round(dy / scale), srcH - 1);
        const srcIdx = (sy * srcW + sx) * 4;

        const dstX = dx + padX;
        const dstY = dy + padY;
        if (dstX >= inputSize || dstY >= inputSize) continue;

        const dstIdx = dstY * inputSize + dstX;
        float32Data[dstIdx] = pixels[srcIdx] / 255.0;
        float32Data[numPixels + dstIdx] = pixels[srcIdx + 1] / 255.0;
        float32Data[2 * numPixels + dstIdx] = pixels[srcIdx + 2] / 255.0;
      }
    }

    const tensor = new ort.Tensor("float32", float32Data, [1, 3, inputSize, inputSize]);
    const t1 = performance.now();

    // --- Inference ---
    const inputName = session.inputNames[0];
    const results = await session.run({ [inputName]: tensor });
    const t2 = performance.now();

    // --- Postprocess ---
    const outputName = session.outputNames[0];
    const output = results[outputName];
    const outputData = output.data as Float32Array;
    const outputDims = output.dims as number[];

    const nc = metadata.num_classes;
    const expectedChannels = 4 + nc;

    let numDets: number;
    let transposed: boolean;

    if (outputDims.length === 3) {
      if (outputDims[1] === expectedChannels) {
        numDets = outputDims[2];
        transposed = true;
      } else if (outputDims[2] === expectedChannels) {
        numDets = outputDims[1];
        transposed = false;
      } else {
        numDets = outputDims[1] < outputDims[2] ? outputDims[2] : outputDims[1];
        transposed = outputDims[1] < outputDims[2];
      }
    } else {
      self.postMessage({
        type: "inference-error",
        payload: `Unexpected output dims: [${outputDims.join(", ")}]`,
      });
      return;
    }

    const rawDetections: Detection[] = [];
    const xRatio = srcW / newW;
    const yRatio = srcH / newH;

    for (let i = 0; i < numDets; i++) {
      let cx: number, cy: number, w: number, h: number;
      let maxScore = 0;
      let maxClassId = 0;

      if (transposed) {
        cx = outputData[0 * numDets + i];
        cy = outputData[1 * numDets + i];
        w = outputData[2 * numDets + i];
        h = outputData[3 * numDets + i];
        for (let c = 0; c < nc; c++) {
          const score = outputData[(4 + c) * numDets + i];
          if (score > maxScore) { maxScore = score; maxClassId = c; }
        }
      } else {
        const offset = i * expectedChannels;
        cx = outputData[offset];
        cy = outputData[offset + 1];
        w = outputData[offset + 2];
        h = outputData[offset + 3];
        for (let c = 0; c < nc; c++) {
          const score = outputData[offset + 4 + c];
          if (score > maxScore) { maxScore = score; maxClassId = c; }
        }
      }

      if (maxScore < confidenceThreshold) continue;

      const x1 = (cx - w / 2 - padX) * xRatio;
      const y1 = (cy - h / 2 - padY) * yRatio;
      const x2 = (cx + w / 2 - padX) * xRatio;
      const y2 = (cy + h / 2 - padY) * yRatio;

      rawDetections.push({
        bbox: {
          x1: Math.max(0, x1), y1: Math.max(0, y1),
          x2: Math.max(0, x2), y2: Math.max(0, y2),
          width: Math.max(0, x2 - x1), height: Math.max(0, y2 - y1),
        },
        classId: maxClassId,
        className: classNames[maxClassId] || `class_${maxClassId}`,
        confidence: maxScore,
        color: getClassColor(maxClassId),
      });
    }

    const detections = applyNMS(rawDetections, iouThreshold);
    const t3 = performance.now();

    const result: InferenceResult = {
      detections,
      preprocessTimeMs: t1 - t0,
      inferenceTimeMs: t2 - t1,
      postprocessTimeMs: t3 - t2,
      totalTimeMs: t3 - t0,
    };

    self.postMessage({ type: "inference-result", payload: result });
  } catch (err) {
    self.postMessage({
      type: "inference-error",
      payload: String(err),
    });
  }
}
