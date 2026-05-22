"use client";

import { useState, useRef, useCallback, useEffect } from "react";
import { useModel } from "@/hooks/useModel";
import { useCamera } from "@/hooks/useCamera";
import { useDetection } from "@/hooks/useDetection";
import type { Detection, DetectionSettings } from "@/types/detection";

type SourceMode = "upload" | "camera" | "live";

export default function Home() {
  // --- State ---
  const [source, setSource] = useState<SourceMode>("upload");
  const [settings, setSettings] = useState<DetectionSettings>({
    confidenceThreshold: 0.25,
    showLabels: true,
    showConfidence: true,
    liveDetection: true,
    inputResolution: 640,
    inferenceMode: "browser",
  });
  const [uploadedImage, setUploadedImage] = useState<string | null>(null);
  const [uploadDetections, setUploadDetections] = useState<Detection[]>([]);
  const [dragOver, setDragOver] = useState(false);

  // --- Hooks ---
  const { status: modelStatus, metadata, error: modelError, loadModel, workerRef } = useModel();
  const { videoRef, isActive: cameraActive, cameraStatus, cameraError, startCamera, stopCamera } = useCamera();
  const { detections: liveDetections, fps, inferenceTime, isDetecting, startDetection, stopDetection } = useDetection();

  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const uploadCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const imgRef = useRef<HTMLImageElement | null>(null);

  // Load model on mount
  useEffect(() => {
    loadModel();
  }, [loadModel]);

  // Draw detections on canvas (live mode)
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const video = videoRef.current;
    if (!video) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    drawDetections(ctx, liveDetections, canvas.width, canvas.height, settings);
  }, [liveDetections, settings, videoRef]);

  // Start/stop live detection
  useEffect(() => {
    if (
      source === "live" &&
      cameraActive &&
      modelStatus === "ready" &&
      settings.liveDetection &&
      videoRef.current &&
      workerRef.current
    ) {
      startDetection(
        videoRef.current,
        workerRef,
        settings.inputResolution,
        settings.confidenceThreshold
      );
    } else if (source !== "live" || !settings.liveDetection) {
      stopDetection();
    }

    return () => {
      if (source !== "live") stopDetection();
    };
  }, [source, cameraActive, modelStatus, settings.liveDetection, settings.inputResolution, settings.confidenceThreshold, startDetection, stopDetection, videoRef, workerRef]);

  // --- Handlers ---
  const handleFileUpload = useCallback(
    (file: File) => {
      if (!file.type.startsWith("image/")) {
        alert("Please upload a valid image file (JPG, PNG, WebP).");
        return;
      }
      if (file.size > 20 * 1024 * 1024) {
        alert("File too large. Maximum 20MB.");
        return;
      }

      const reader = new FileReader();
      reader.onload = (e) => {
        const dataUrl = e.target?.result as string;
        setUploadedImage(dataUrl);
        setUploadDetections([]);

        // Run inference on uploaded image
        if (workerRef.current && modelStatus === "ready") {
          const img = new Image();
          img.onload = () => {
            const c = new OffscreenCanvas(img.width, img.height);
            const cx = c.getContext("2d");
            if (!cx) return;
            cx.drawImage(img, 0, 0);
            const imageData = cx.getImageData(0, 0, img.width, img.height);

            const handler = (ev: MessageEvent) => {
              if (ev.data.type === "inference-result") {
                setUploadDetections(ev.data.payload.detections);
                workerRef.current?.removeEventListener("message", handler);
              }
            };
            workerRef.current!.addEventListener("message", handler);
            workerRef.current!.postMessage({
              type: "run-inference",
              payload: {
                imageData: {
                  data: imageData.data,
                  width: imageData.width,
                  height: imageData.height,
                },
                inputSize: settings.inputResolution,
                confidenceThreshold: settings.confidenceThreshold,
                iouThreshold: 0.45,
              },
            });
          };
          img.src = dataUrl;
        }
      };
      reader.readAsDataURL(file);
    },
    [modelStatus, settings, workerRef]
  );

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragOver(false);
      const file = e.dataTransfer.files[0];
      if (file) handleFileUpload(file);
    },
    [handleFileUpload]
  );

  // Draw upload detections
  useEffect(() => {
    if (!uploadedImage || !uploadCanvasRef.current || !imgRef.current) return;
    const canvas = uploadCanvasRef.current;
    const img = imgRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    canvas.width = img.naturalWidth || img.width;
    canvas.height = img.naturalHeight || img.height;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    drawDetections(ctx, uploadDetections, canvas.width, canvas.height, settings);
  }, [uploadDetections, uploadedImage, settings]);

  const currentDetections = source === "upload" ? uploadDetections : liveDetections;

  const getStatusText = () => {
    if (modelStatus === "loading-model") return "Loading model...";
    if (modelStatus === "error") return "Model error";
    if (modelStatus === "ready" && !isDetecting) return "Ready";
    if (isDetecting) return "Detecting";
    return "Idle";
  };

  const getStatusClass = () => {
    if (modelStatus === "loading-model") return "status-loading";
    if (modelStatus === "error") return "status-error";
    if (isDetecting) return "status-detecting";
    if (modelStatus === "ready") return "status-ready";
    return "status-idle";
  };

  return (
    <div className="app-container">
      {/* Header */}
      <header className="app-header">
        <h1>Urban Object Detection</h1>
        <p>
          YOLOv8 real-time detection
          {metadata ? ` - ${metadata.num_classes} classes` : ""} - powered by ONNX Runtime Web
        </p>
      </header>

      <div className="main-grid">
        {/* Left: Viewport */}
        <div>
          {/* Source Tabs */}
          <div className="source-tabs">
            {(["upload", "camera", "live"] as SourceMode[]).map((s) => (
              <button
                key={s}
                className={`source-tab ${source === s ? "active" : ""}`}
                onClick={() => {
                  setSource(s);
                  if (s !== "camera" && s !== "live") stopCamera();
                  if (s !== "live") stopDetection();
                }}
              >
                {s === "upload" ? "Upload" : s === "camera" ? "Camera" : "Live Feed"}
              </button>
            ))}
          </div>

          {/* Errors */}
          {modelError && <div className="error-banner">Model error: {modelError}</div>}
          {cameraError && <div className="error-banner">{cameraError}</div>}
          {modelStatus === "loading-model" && (
            <div className="info-banner">Loading ONNX model... This may take a few seconds on first load.</div>
          )}

          {/* Viewport */}
          <div className="viewport-wrapper">
            {/* Upload mode */}
            {source === "upload" && !uploadedImage && (
              <div
                className={`upload-zone viewport-placeholder ${dragOver ? "dragover" : ""}`}
                onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
                onDragLeave={() => setDragOver(false)}
                onDrop={handleDrop}
                onClick={() => fileInputRef.current?.click()}
              >
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                  <path d="M12 16V4m0 0L8 8m4-4l4 4M4 17v2a2 2 0 002 2h12a2 2 0 002-2v-2" />
                </svg>
                <span>Drop an image here or click to upload</span>
                <span style={{ fontSize: "0.75rem" }}>JPG, PNG, WebP (max 20MB)</span>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/jpeg,image/png,image/webp"
                  onChange={(e) => {
                    const file = e.target.files?.[0];
                    if (file) handleFileUpload(file);
                  }}
                />
              </div>
            )}

            {source === "upload" && uploadedImage && (
              <>
                <img
                  ref={imgRef}
                  src={uploadedImage}
                  alt="Uploaded"
                  onLoad={() => {
                    // Trigger redraw after image loads
                    if (uploadCanvasRef.current && imgRef.current) {
                      const canvas = uploadCanvasRef.current;
                      canvas.width = imgRef.current.naturalWidth;
                      canvas.height = imgRef.current.naturalHeight;
                    }
                  }}
                />
                <canvas ref={uploadCanvasRef} />
              </>
            )}

            {/* Camera / Live mode */}
            {(source === "camera" || source === "live") && (
              <>
                <video ref={videoRef} playsInline muted style={{ display: cameraActive ? "block" : "none" }} />
                <canvas ref={canvasRef} style={{ display: cameraActive ? "block" : "none" }} />
                {!cameraActive && (
                  <div className="viewport-placeholder">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                      <path d="M15.75 10.5l4.72-4.72a.75.75 0 011.28.53v11.38a.75.75 0 01-1.28.53l-4.72-4.72M4.5 18.75h9a2.25 2.25 0 002.25-2.25v-9a2.25 2.25 0 00-2.25-2.25h-9A2.25 2.25 0 002.25 7.5v9a2.25 2.25 0 002.25 2.25z" />
                    </svg>
                    <button className="btn btn-primary" onClick={startCamera} disabled={modelStatus !== "ready"}>
                      {modelStatus === "loading-model" ? "Loading model..." : "Start Camera"}
                    </button>
                  </div>
                )}
              </>
            )}
          </div>

          {/* Stats Bar */}
          <div className="stats-bar">
            <div className="stat-item">
              <span className={`status-badge ${getStatusClass()}`}>
                <span className="status-dot" />
                {getStatusText()}
              </span>
            </div>
            {(source === "live" || source === "camera") && isDetecting && (
              <>
                <div className="stat-item">
                  <span className="stat-label">FPS</span>
                  <span className={`stat-value ${fps >= 10 ? "good" : fps >= 5 ? "warn" : "bad"}`}>{fps}</span>
                </div>
                <div className="stat-item">
                  <span className="stat-label">Inference</span>
                  <span className="stat-value">{inferenceTime}ms</span>
                </div>
              </>
            )}
            <div className="stat-item">
              <span className="stat-label">Objects</span>
              <span className="stat-value">{currentDetections.length}</span>
            </div>
          </div>

          {/* Upload controls */}
          {source === "upload" && uploadedImage && (
            <div style={{ marginTop: 12 }}>
              <button className="btn" onClick={() => { setUploadedImage(null); setUploadDetections([]); }}>
                Upload Another Image
              </button>
            </div>
          )}
        </div>

        {/* Right: Controls */}
        <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
          {/* Controls */}
          <div className="card">
            <div className="card-title">Controls</div>

            <div className="control-group">
              <div className="control-label">
                <span>Confidence Threshold</span>
                <span className="control-value">{(settings.confidenceThreshold * 100).toFixed(0)}%</span>
              </div>
              <input
                type="range"
                min="10"
                max="90"
                value={settings.confidenceThreshold * 100}
                onChange={(e) =>
                  setSettings((s) => ({ ...s, confidenceThreshold: Number(e.target.value) / 100 }))
                }
              />
            </div>

            <div className="control-group">
              <div className="toggle-row">
                <span className="toggle-text">Show Labels</span>
                <label className="toggle-switch">
                  <input
                    type="checkbox"
                    checked={settings.showLabels}
                    onChange={(e) => setSettings((s) => ({ ...s, showLabels: e.target.checked }))}
                  />
                  <span className="toggle-slider" />
                </label>
              </div>
              <div className="toggle-row">
                <span className="toggle-text">Show Confidence</span>
                <label className="toggle-switch">
                  <input
                    type="checkbox"
                    checked={settings.showConfidence}
                    onChange={(e) => setSettings((s) => ({ ...s, showConfidence: e.target.checked }))}
                  />
                  <span className="toggle-slider" />
                </label>
              </div>
              {source === "live" && (
                <div className="toggle-row">
                  <span className="toggle-text">Live Detection</span>
                  <label className="toggle-switch">
                    <input
                      type="checkbox"
                      checked={settings.liveDetection}
                      onChange={(e) => setSettings((s) => ({ ...s, liveDetection: e.target.checked }))}
                    />
                    <span className="toggle-slider" />
                  </label>
                </div>
              )}
            </div>

            <div className="control-group">
              <div className="control-label">
                <span>Input Resolution</span>
              </div>
              <div className="resolution-group">
                {[320, 416, 640].map((res) => (
                  <button
                    key={res}
                    className={`resolution-btn ${settings.inputResolution === res ? "active" : ""}`}
                    onClick={() => setSettings((s) => ({ ...s, inputResolution: res }))}
                  >
                    {res}px
                    {res === 320 ? " (Fast)" : res === 640 ? " (Quality)" : ""}
                  </button>
                ))}
              </div>
            </div>
          </div>

          {/* Detections List */}
          <div className="card">
            <div className="card-title">Detections ({currentDetections.length})</div>
            {currentDetections.length === 0 ? (
              <div style={{ color: "var(--text-muted)", fontSize: "0.85rem", textAlign: "center", padding: "16px 0" }}>
                {modelStatus !== "ready" ? "Load model to start detecting" : "No objects detected"}
              </div>
            ) : (
              <div className="detection-list">
                {currentDetections.map((det, i) => (
                  <div key={i} className="detection-item">
                    <div className="detection-color" style={{ background: det.color }} />
                    <span className="detection-name">{det.className}</span>
                    <span className="detection-conf">{(det.confidence * 100).toFixed(1)}%</span>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Model Info */}
          {metadata && (
            <div className="card">
              <div className="card-title">Model Info</div>
              <div style={{ fontSize: "0.8rem", color: "var(--text-secondary)", lineHeight: 1.8 }}>
                <div>Classes: {metadata.num_classes}</div>
                <div>Input: {metadata.input_size}x{metadata.input_size}</div>
                <div>Runtime: ONNX (WASM)</div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

/* ======================== Drawing Helper ======================== */

function drawDetections(
  ctx: CanvasRenderingContext2D,
  detections: Detection[],
  canvasW: number,
  canvasH: number,
  settings: DetectionSettings
) {
  ctx.clearRect(0, 0, canvasW, canvasH);

  for (const det of detections) {
    const { x1, y1, width, height } = det.bbox;

    // Box
    ctx.strokeStyle = det.color;
    ctx.lineWidth = Math.max(2, canvasW / 300);
    ctx.strokeRect(x1, y1, width, height);

    // Semi-transparent fill
    ctx.fillStyle = det.color.replace(")", ", 0.08)").replace("hsl(", "hsla(");
    ctx.fillRect(x1, y1, width, height);

    // Label
    if (settings.showLabels || settings.showConfidence) {
      let label = "";
      if (settings.showLabels) label += det.className;
      if (settings.showLabels && settings.showConfidence) label += " ";
      if (settings.showConfidence) label += `${(det.confidence * 100).toFixed(0)}%`;

      const fontSize = Math.max(11, canvasW / 50);
      ctx.font = `600 ${fontSize}px Inter, sans-serif`;
      const textMetrics = ctx.measureText(label);
      const textH = fontSize + 6;
      const textW = textMetrics.width + 10;
      const labelY = Math.max(textH, y1);

      ctx.fillStyle = det.color;
      ctx.fillRect(x1, labelY - textH, textW, textH);

      ctx.fillStyle = "#ffffff";
      ctx.fillText(label, x1 + 5, labelY - 4);
    }
  }
}
