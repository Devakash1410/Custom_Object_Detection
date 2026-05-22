"""
Model export for deployment:
- ONNX export with simplification
- Quantized ONNX (INT8) for smaller size + faster browser inference
- labels.json for frontend
- metadata.json with export details
- Auto-copy to frontend/public/model/
"""

import os
import json
import shutil
import yaml
from pathlib import Path
from typing import Dict, Optional

from src.utils.helpers import (
    load_config,
    resolve_path,
    setup_logger,
    get_device,
    save_metadata,
    save_labels_json,
    create_metadata,
    get_next_version,
)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
logger = setup_logger("exporter")


def export_model(
    model_path: Optional[str] = None,
    config: Optional[dict] = None,
    version: Optional[int] = None,
) -> Dict:
    """
    Export trained model to ONNX + quantized ONNX for browser deployment.

    Args:
        model_path: Path to best.pt. Auto-detected if None.
        config: Config dict.
        version: Model version number.

    Returns:
        Dict with export paths and metadata.
    """
    from ultralytics import YOLO

    if config is None:
        config = load_config()

    ex_cfg = config["export"]
    o_cfg = config["output"]

    # Find model
    if model_path is None:
        model_path = _find_best_model(o_cfg, version)
    if model_path is None or not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    logger.info("=" * 60)
    logger.info("EXPORT PIPELINE")
    logger.info(f"Model: {model_path}")
    logger.info("=" * 60)

    model = YOLO(model_path)

    # Determine output version dir
    if version is None:
        # Try to infer from model_path
        model_parent = Path(model_path).parent
        if model_parent.name.startswith("v") and model_parent.name[1:].isdigit():
            version = int(model_parent.name[1:])
        else:
            version = get_next_version(o_cfg["models_dir"])

    version_dir = resolve_path(o_cfg["models_dir"]) / f"v{version}"
    version_dir.mkdir(parents=True, exist_ok=True)

    results = {"version": version, "model_path": model_path}

    # --- ONNX Export ---
    logger.info("Exporting to ONNX...")
    imgsz = ex_cfg.get("imgsz", 640)
    simplify = ex_cfg.get("simplify", True)
    opset = ex_cfg.get("opset", 17)

    onnx_path = model.export(
        format="onnx",
        imgsz=imgsz,
        simplify=simplify,
        opset=opset,
    )
    onnx_path = Path(onnx_path)

    # Copy to version dir
    best_onnx = version_dir / "best.onnx"
    if onnx_path.exists():
        shutil.copy2(onnx_path, best_onnx)
        onnx_size = best_onnx.stat().st_size / (1024 * 1024)
        logger.info(f"ONNX exported: {best_onnx} ({onnx_size:.1f} MB)")
        results["onnx_path"] = str(best_onnx)
        results["onnx_size_mb"] = round(onnx_size, 2)

    # --- Quantized ONNX ---
    if ex_cfg.get("quantize", True):
        logger.info("Creating quantized ONNX model...")
        quantized_path = version_dir / "best_quantized.onnx"
        try:
            _quantize_onnx(best_onnx, quantized_path)
            q_size = quantized_path.stat().st_size / (1024 * 1024)
            logger.info(f"Quantized ONNX: {quantized_path} ({q_size:.1f} MB)")
            results["quantized_onnx_path"] = str(quantized_path)
            results["quantized_size_mb"] = round(q_size, 2)
        except Exception as e:
            logger.warning(f"Quantization failed (non-critical): {e}")
            logger.info("The standard ONNX model will be used instead.")

    # --- Labels JSON ---
    data_yaml = resolve_path("configs") / "data.yaml"
    if not data_yaml.exists():
        data_yaml = resolve_path("data.yaml")

    with open(data_yaml) as f:
        data_cfg = yaml.safe_load(f)

    class_names = data_cfg["names"]
    labels_path = version_dir / "labels.json"
    save_labels_json(class_names, labels_path)
    results["labels_path"] = str(labels_path)

    # --- Model metadata for frontend ---
    model_meta = {
        "version": version,
        "format": "onnx",
        "input_size": imgsz,
        "num_classes": len(class_names),
        "class_names": class_names,
        "opset": opset,
        "simplified": simplify,
        "quantized": ex_cfg.get("quantize", True),
    }
    meta_path = version_dir / "model_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(model_meta, f, indent=2)
    results["metadata_path"] = str(meta_path)

    # --- Copy to frontend ---
    if ex_cfg.get("copy_to_frontend", True):
        frontend_dir = resolve_path(ex_cfg.get("frontend_model_dir", "frontend/public/model"))
        _copy_to_frontend(version_dir, frontend_dir, class_names)
        results["frontend_dir"] = str(frontend_dir)

    logger.info("=" * 60)
    logger.info("Export complete!")
    logger.info(f"  ONNX: {results.get('onnx_path')}")
    if "quantized_onnx_path" in results:
        logger.info(f"  Quantized: {results['quantized_onnx_path']}")
    logger.info("=" * 60)

    return results


def _quantize_onnx(input_path: Path, output_path: Path) -> None:
    """Quantize ONNX model to INT8 using onnxruntime quantization."""
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType

        quantize_dynamic(
            model_input=str(input_path),
            model_output=str(output_path),
            weight_type=QuantType.QUInt8,
        )
    except ImportError:
        # Fallback: just copy the original
        logger.warning("onnxruntime quantization not available. Install: pip install onnxruntime")
        shutil.copy2(input_path, output_path)


def _copy_to_frontend(version_dir: Path, frontend_dir: Path, class_names: list) -> None:
    """Copy model artifacts to frontend public directory."""
    frontend_dir.mkdir(parents=True, exist_ok=True)

    files_to_copy = ["best.onnx", "best_quantized.onnx", "labels.json", "model_metadata.json"]
    for fname in files_to_copy:
        src = version_dir / fname
        if src.exists():
            shutil.copy2(src, frontend_dir / fname)
            logger.info(f"  Copied {fname} to frontend")

    logger.info(f"Frontend model dir: {frontend_dir}")


def _find_best_model(o_cfg: dict, version: Optional[int] = None) -> Optional[str]:
    """Find best.pt from the latest or specified version."""
    models_dir = resolve_path(o_cfg["models_dir"])
    if not models_dir.exists():
        return None

    if version:
        p = models_dir / f"v{version}" / "best.pt"
        return str(p) if p.exists() else None

    versions = sorted(
        [d for d in models_dir.iterdir() if d.is_dir() and d.name.startswith("v")],
        key=lambda d: int(d.name[1:]) if d.name[1:].isdigit() else 0,
        reverse=True,
    )
    for v in versions:
        p = v / "best.pt"
        if p.exists():
            return str(p)
    return None
