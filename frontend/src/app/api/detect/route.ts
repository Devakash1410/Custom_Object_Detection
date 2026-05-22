/**
 * Backend API route for hybrid inference.
 * Used when browser performance is poor or model is too large for WASM.
 *
 * POST /api/detect
 * Body: FormData with "image" file, "confidence" number, "inputSize" number
 * Returns: JSON { detections: Detection[], inferenceTimeMs: number }
 */

import { NextRequest, NextResponse } from "next/server";

export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData();
    const imageFile = formData.get("image") as File | null;
    const confidence = parseFloat((formData.get("confidence") as string) || "0.25");

    if (!imageFile) {
      return NextResponse.json(
        { error: "No image provided" },
        { status: 400 }
      );
    }

    // Validate file type
    if (!imageFile.type.startsWith("image/")) {
      return NextResponse.json(
        { error: "Invalid file type. Expected an image." },
        { status: 400 }
      );
    }

    // For now, return a placeholder response.
    // In production, this would run ONNX Runtime (Node.js) or call a Python backend.
    // To enable: install onnxruntime-node and implement server-side inference.
    return NextResponse.json({
      detections: [],
      inferenceTimeMs: 0,
      message:
        "Server-side inference not configured. " +
        "To enable: install onnxruntime-node and add model loading logic. " +
        "Browser inference (WASM) is the default mode.",
    });
  } catch (error) {
    return NextResponse.json(
      { error: `Inference failed: ${String(error)}` },
      { status: 500 }
    );
  }
}
