/**
 * useCamera — WebRTC camera access with graceful error handling.
 *
 * - Prefers front camera (facingMode: "user")
 * - Handles permission denial
 * - Handles unsupported devices
 * - Cleanup on unmount
 */

"use client";

import { useState, useRef, useCallback, useEffect } from "react";
import type { DetectionStatus } from "@/types/detection";

interface UseCameraReturn {
  videoRef: React.RefObject<HTMLVideoElement | null>;
  isActive: boolean;
  cameraStatus: DetectionStatus;
  cameraError: string | null;
  startCamera: () => Promise<void>;
  stopCamera: () => void;
}

export function useCamera(): UseCameraReturn {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const [isActive, setIsActive] = useState(false);
  const [cameraStatus, setCameraStatus] = useState<DetectionStatus>("idle");
  const [cameraError, setCameraError] = useState<string | null>(null);

  const stopCamera = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setIsActive(false);
    setCameraStatus("idle");
  }, []);

  const startCamera = useCallback(async () => {
    // Check browser support
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      setCameraStatus("unsupported");
      setCameraError("Camera API not supported in this browser");
      return;
    }

    try {
      setCameraStatus("loading-model");
      setCameraError(null);

      // Try front camera first
      let stream: MediaStream;
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: "user", width: { ideal: 640 }, height: { ideal: 480 } },
          audio: false,
        });
      } catch {
        // Fallback: any camera
        stream = await navigator.mediaDevices.getUserMedia({
          video: { width: { ideal: 640 }, height: { ideal: 480 } },
          audio: false,
        });
      }

      streamRef.current = stream;

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }

      setIsActive(true);
      setCameraStatus("ready");
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);

      if (msg.includes("Permission") || msg.includes("NotAllowed")) {
        setCameraStatus("camera-denied");
        setCameraError("Camera permission denied. Please allow camera access and refresh.");
      } else if (msg.includes("NotFound") || msg.includes("DevicesNotFound")) {
        setCameraStatus("unsupported");
        setCameraError("No camera found on this device.");
      } else {
        setCameraStatus("error");
        setCameraError(`Camera error: ${msg}`);
      }
    }
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((t) => t.stop());
      }
    };
  }, []);

  return { videoRef, isActive, cameraStatus, cameraError, startCamera, stopCamera };
}
