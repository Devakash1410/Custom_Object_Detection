# YOLOv8 Urban Object Detection — Full Stack

Production-grade object detection pipeline with real-time browser inference.

**Dataset** -> **Training** -> **Validation** -> **Testing** -> **Model Export** -> **Quantized ONNX** -> **Next.js Frontend** -> **Image Upload + Live Camera + Real-time Detection**

---

## Features

### ML Pipeline (Python)
- **37-class urban detection**: cars, people, bicycles, traffic lights, stop signs, potholes, and more
- **COCO-to-YOLO conversion** with symlink-first + copy fallback
- **GTX 1650 optimized**: auto-batch sizing, mixed precision, OOM recovery
- **Resume interrupted training** from checkpoints
- **Comprehensive evaluation**: per-class metrics, confusion analysis, underperforming class detection
- **ONNX export** with quantization for browser deployment
- **Versioned model artifacts** with metadata and config snapshots

### Frontend (Next.js)
- **Real-time browser inference** using ONNX Runtime Web (WASM)
- **Three input modes**: image upload, camera capture, live video feed
- **Web Worker inference** — zero UI freezing
- **Adaptive FPS throttling** — adjusts to device performance
- **Dynamic resolution**: 320px (fast) / 416px (balanced) / 640px (quality)
- **Dark glassmorphism UI** with responsive design
- **37 classes** with color-coded bounding boxes and confidence scores

---

## Project Structure

```
Custom_Object_Detection/
├── configs/
│   ├── config.yaml         # Central configuration
│   └── data.yaml            # Auto-generated YOLO dataset config
├── src/
│   ├── data/
│   │   ├── converter.py     # COCO -> YOLO conversion
│   │   ├── validator.py     # Dataset integrity checks
│   │   └── eda.py           # Exploratory data analysis
│   ├── training/
│   │   └── trainer.py       # Training with OOM recovery + resume
│   ├── evaluation/
│   │   └── evaluator.py     # Full evaluation + reports
│   ├── export/
│   │   └── exporter.py      # ONNX + quantized export
│   └── utils/
│       └── helpers.py       # Config, logging, versioning
├── scripts/
│   ├── run_pipeline.py      # Full pipeline (convert -> train -> eval -> export)
│   ├── run_training.py      # Training only
│   ├── run_evaluation.py    # Evaluation only
│   └── run_export.py        # Export only
├── frontend/                # Next.js application
│   ├── src/
│   │   ├── app/             # Pages + API routes
│   │   ├── hooks/           # useCamera, useDetection, useModel
│   │   ├── lib/             # inference, postprocess, colors
│   │   ├── workers/         # Web Worker for inference
│   │   └── types/           # TypeScript types
│   └── public/model/        # ONNX model + labels (auto-copied)
├── outputs/
│   ├── models/v{N}/         # Versioned model artifacts
│   ├── reports/             # Evaluation reports + plots
│   └── training_history/    # Metrics CSV
├── dataset/                 # Raw COCO dataset
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Install Dependencies

```bash
# Python ML pipeline
pip install -r requirements.txt

# CUDA PyTorch (for GPU training)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Frontend
cd frontend && npm install
```

### 2. Run the Full Pipeline

```bash
# End-to-end: convert dataset -> train -> evaluate -> export
python scripts/run_pipeline.py
```

Or run individual steps:
```bash
python scripts/run_pipeline.py --start-from train     # Skip conversion
python scripts/run_pipeline.py --start-from evaluate   # Evaluation only
python scripts/run_pipeline.py --start-from export     # Export only
```

### 3. Start the Frontend

```bash
cd frontend
npm run dev
```

Open http://localhost:3000

### 4. Add Your Trained Model

The export pipeline auto-copies model files to `frontend/public/model/`.
If running manually:
```bash
# Copy these files to frontend/public/model/:
# - best.onnx (or best_quantized.onnx)
# - labels.json
# - model_metadata.json
```

---

## Configuration

Edit `configs/config.yaml` to customize:

| Setting | Default | Description |
|---------|---------|-------------|
| `training.model` | `yolov8n.pt` | Model variant (n/s/m/l) |
| `training.epochs` | `25` | Training epochs |
| `training.batch` | `-1` | Batch size (-1 = auto) |
| `training.imgsz` | `640` | Image size |
| `training.amp` | `true` | Mixed precision |
| `training.patience` | `10` | Early stopping |
| `export.quantize` | `true` | INT8 quantization |
| `dataset.use_symlinks` | `true` | Symlinks for images |

---

## GPU Training (GTX 1650)

The pipeline is optimized for 4GB VRAM GPUs:
- **Auto-batch sizing**: finds optimal batch size automatically
- **OOM recovery**: halves batch and retries on out-of-memory
- **Workers=2**: reduces memory pressure
- **Cache=false**: saves ~3GB RAM
- **AMP**: mixed precision for faster training

---

## Deployment

### Browser Deployment (Default)
The frontend uses ONNX Runtime Web (WASM). No backend needed.
Just serve the Next.js app and the model files.

```bash
cd frontend
npm run build
npm start
```

### Hybrid Mode
For larger models or weak devices, the frontend can fallback to server-side inference
via the `/api/detect` endpoint. Configure `onnxruntime-node` on the server.

---

## Model Outputs

After training, the pipeline produces:

```
outputs/models/v1/
├── best.pt              # Best PyTorch weights
├── best.onnx            # ONNX model
├── best_quantized.onnx  # Quantized ONNX (smaller, faster)
├── labels.json          # Class names
├── metadata.json        # Training metadata
├── config_snapshot.yaml # Config used for training
└── last.pt              # Last checkpoint (for resume)
```
