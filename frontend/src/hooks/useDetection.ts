/**
 * useDetection — Manages live detection loop with adaptive throttling.
 *
 * - Adaptive FPS: slows down if inference is slow, speeds up if fast
 * - Rolling average timing
 * - Web Worker based (no UI blocking)
 */

"use client";

import { useState, useRef, useCallback, useEffect } from "react";
import type { Detection, InferenceResult } from "@/types/detection";

interface UseDetectionReturn {
  detections: Detection[];
  fps: number;
  inferenceTime: number;
  isDetecting: boolean;
  startDetection: (
    videoElement: HTMLVideoElement,
    workerRef: React.RefObject<Worker | null>,
    inputSize: number,
    confidenceThreshold: number
  ) => void;
  stopDetection: () => void;
}

export function useDetection(): UseDetectionReturn {
  const [detections, setDetections] = useState<Detection[]>([]);
  const [fps, setFps] = useState(0);
  const [inferenceTime, setInferenceTime] = useState(0);
  const [isDetecting, setIsDetecting] = useState(false);

  const rafRef = useRef<number>(0);
  const busyRef = useRef(false);
  const activeRef = useRef(false);

  // Rolling average for adaptive throttling
  const timingsRef = useRef<number[]>([]);
  const frameCountRef = useRef(0);
  const fpsTimerRef = useRef(performance.now());

  // References to current params (avoid stale closures)
  const paramsRef = useRef({
    inputSize: 640,
    confidenceThreshold: 0.25,
  });

  const startDetection = useCallback(
    (
      videoElement: HTMLVideoElement,
      workerRef: React.RefObject<Worker | null>,
      inputSize: number,
      confidenceThreshold: number
    ) => {
      paramsRef.current = { inputSize, confidenceThreshold };
      activeRef.current = true;
      setIsDetecting(true);

      const worker = workerRef.current;
      if (!worker) return;

      // Handle results from worker
      const onMessage = (e: MessageEvent) => {
        const { type, payload } = e.data;
        if (type === "inference-result") {
          const result = payload as InferenceResult;
          setDetections(result.detections);
          setInferenceTime(Math.round(result.totalTimeMs));

          // Track timing for adaptive throttle
          timingsRef.current.push(result.totalTimeMs);
          if (timingsRef.current.length > 10) timingsRef.current.shift();

          // FPS counter
          frameCountRef.current++;
          const now = performance.now();
          if (now - fpsTimerRef.current >= 1000) {
            setFps(frameCountRef.current);
            frameCountRef.current = 0;
            fpsTimerRef.current = now;
          }

          busyRef.current = false;
        } else if (type === "inference-error") {
          busyRef.current = false;
        }
      };

      worker.addEventListener("message", onMessage);

      // Detection loop with adaptive throttling
      const detect = () => {
        if (!activeRef.current) return;

        if (!busyRef.current && videoElement.readyState >= 2) {
          busyRef.current = true;

          // Capture frame
          const { inputSize: sz, confidenceThreshold: ct } = paramsRef.current;
          const canvas = new OffscreenCanvas(videoElement.videoWidth || 640, videoElement.videoHeight || 480);
          const ctx = canvas.getContext("2d");
          if (ctx) {
            ctx.drawImage(videoElement, 0, 0);
            const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);

            worker.postMessage({
              type: "run-inference",
              payload: {
                imageData: {
                  data: imageData.data,
                  width: imageData.width,
                  height: imageData.height,
                },
                inputSize: sz,
                confidenceThreshold: ct,
                iouThreshold: 0.45,
              },
            });
          } else {
            busyRef.current = false;
          }
        }

        // Adaptive delay: if inference is slow, wait longer
        const avgTime =
          timingsRef.current.length > 0
            ? timingsRef.current.reduce((a, b) => a + b, 0) / timingsRef.current.length
            : 33;
        const delay = Math.max(16, Math.min(200, avgTime * 0.5));

        rafRef.current = window.setTimeout(() => {
          requestAnimationFrame(detect);
        }, delay) as unknown as number;
      };

      requestAnimationFrame(detect);

      // Cleanup function stored for stopDetection
      return () => {
        worker.removeEventListener("message", onMessage);
      };
    },
    []
  );

  const stopDetection = useCallback(() => {
    activeRef.current = false;
    busyRef.current = false;
    setIsDetecting(false);
    setDetections([]);
    if (rafRef.current) {
      clearTimeout(rafRef.current);
    }
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      activeRef.current = false;
      if (rafRef.current) clearTimeout(rafRef.current);
    };
  }, []);

  return { detections, fps, inferenceTime, isDetecting, startDetection, stopDetection };
}
