/**
 * Detection types shared across the frontend.
 */

export interface BBox {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  width: number;
  height: number;
}

export interface Detection {
  bbox: BBox;
  classId: number;
  className: string;
  confidence: number;
  color: string;
}

export interface ModelMetadata {
  num_classes: number;
  class_names: string[];
  input_size: number;
}

export interface InferenceResult {
  detections: Detection[];
  inferenceTimeMs: number;
  preprocessTimeMs: number;
  postprocessTimeMs: number;
  totalTimeMs: number;
}

export interface DetectionSettings {
  confidenceThreshold: number;
  showLabels: boolean;
  showConfidence: boolean;
  liveDetection: boolean;
  inputResolution: number;
  inferenceMode: "browser" | "api";
}

export type DetectionStatus =
  | "idle"
  | "loading-model"
  | "ready"
  | "detecting"
  | "error"
  | "camera-denied"
  | "unsupported";

/** Messages sent to / from the inference Web Worker. */
export interface WorkerMessage {
  type:
    | "load-model"
    | "model-loaded"
    | "model-error"
    | "run-inference"
    | "inference-result"
    | "inference-error";
  payload?: unknown;
}

export interface WorkerLoadPayload {
  modelUrl: string;
  metadataUrl: string;
  labelsUrl: string;
}

export interface WorkerInferencePayload {
  imageData: ImageData;
  inputSize: number;
  confidenceThreshold: number;
  iouThreshold: number;
}
