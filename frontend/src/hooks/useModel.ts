/**
 * useModel — Manages ONNX model loading via Web Worker.
 */

"use client";

import { useState, useEffect, useCallback, useRef } from "react";
import type { ModelMetadata, DetectionStatus } from "@/types/detection";

export function useModel() {
  const [status, setStatus] = useState<DetectionStatus>("idle");
  const [metadata, setMetadata] = useState<ModelMetadata | null>(null);
  const [error, setError] = useState<string | null>(null);
  const workerRef = useRef<Worker | null>(null);

  // Initialize worker
  useEffect(() => {
    const worker = new Worker(
      new URL("../workers/inference.worker.ts", import.meta.url),
      { type: "module" }
    );

    worker.onmessage = (e) => {
      const { type, payload } = e.data;
      switch (type) {
        case "model-loaded":
          setMetadata(payload);
          setStatus("ready");
          setError(null);
          break;
        case "model-error":
          setStatus("error");
          setError(String(payload));
          break;
      }
    };

    worker.onerror = (err) => {
      setStatus("error");
      setError(err.message || "Worker error");
    };

    workerRef.current = worker;
    return () => worker.terminate();
  }, []);

  const loadModel = useCallback(() => {
    if (!workerRef.current) return;
    setStatus("loading-model");
    setError(null);

    workerRef.current.postMessage({
      type: "load-model",
      payload: {
        modelUrl: "/model/best.onnx",
        metadataUrl: "/model/model_metadata.json",
        labelsUrl: "/model/labels.json",
      },
    });
  }, []);

  return { status, metadata, error, loadModel, workerRef };
}
