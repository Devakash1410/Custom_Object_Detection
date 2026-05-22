"""
Comprehensive model evaluation:
- Validation & test metrics
- Per-class AP, precision, recall
- Confusion matrix
- Confidence statistics
- Underperforming class detection
- Prediction visualization
- Auto-generated markdown report
"""

import os
import csv
import json
import random
import yaml
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
from typing import Dict, List, Optional

from src.utils.helpers import load_config, resolve_path, setup_logger, get_device

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
logger = setup_logger("evaluator")
random.seed(42)


def evaluate_model(
    model_path: Optional[str] = None,
    config: Optional[dict] = None,
    version: Optional[int] = None,
) -> Dict:
    """
    Full evaluation pipeline.

    Args:
        model_path: Path to best.pt. Auto-detected if None.
        config: Config dict.
        version: Model version number.

    Returns:
        Dict with all metrics, paths, and analysis.
    """
    from ultralytics import YOLO

    if config is None:
        config = load_config()

    e_cfg = config["evaluation"]
    o_cfg = config["output"]
    device_arg, device_name = get_device()

    # Find model
    if model_path is None:
        model_path = _find_best_model(o_cfg, version)
    if model_path is None or not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    logger.info("=" * 60)
    logger.info("EVALUATION PIPELINE")
    logger.info(f"Model: {model_path}")
    logger.info(f"Device: {device_name}")
    logger.info("=" * 60)

    model = YOLO(model_path)

    # Load data config
    data_yaml = resolve_path("configs") / "data.yaml"
    if not data_yaml.exists():
        data_yaml = resolve_path("data.yaml")

    with open(data_yaml) as f:
        data_cfg = yaml.safe_load(f)
    names = data_cfg["names"]
    nc = data_cfg["nc"]

    reports_dir = resolve_path(o_cfg["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)

    results = {"model_path": model_path, "device": device_name}

    # --- Validation metrics ---
    logger.info("Running validation...")
    val_metrics = model.val(
        data=str(data_yaml), split="val", imgsz=640, batch=8,
        device=device_arg, verbose=False, plots=True,
    )
    results["validation"] = _extract_metrics(val_metrics, names)

    # --- Test metrics ---
    logger.info("Running test evaluation...")
    try:
        test_metrics = model.val(
            data=str(data_yaml), split="test", imgsz=640, batch=8,
            device=device_arg, verbose=False, plots=True,
        )
        results["test"] = _extract_metrics(test_metrics, names)
    except Exception as e:
        logger.warning(f"Test evaluation failed: {e}")
        results["test"] = None

    # --- Per-class metrics CSV ---
    logger.info("Saving per-class metrics...")
    per_class = results["validation"].get("per_class", [])
    csv_path = reports_dir / "per_class_metrics.csv"
    _save_per_class_csv(per_class, csv_path)
    results["per_class_csv"] = str(csv_path)

    # --- Underperforming classes ---
    threshold = e_cfg.get("underperform_ap_threshold", 0.3)
    underperforming = [
        c for c in per_class if c["ap50"] < threshold
    ]
    results["underperforming_classes"] = underperforming
    if underperforming:
        logger.warning(f"Underperforming classes (AP50 < {threshold}):")
        for c in underperforming:
            logger.warning(f"  {c['name']}: AP50={c['ap50']:.4f}")

    # --- Confidence statistics ---
    logger.info("Computing confidence statistics...")
    conf_stats = _compute_confidence_stats(model, data_yaml, data_cfg, device_arg, e_cfg)
    results["confidence_stats"] = conf_stats

    # --- Prediction visualization ---
    logger.info("Generating prediction visualizations...")
    pred_path = _save_prediction_grid(model, data_cfg, device_arg, e_cfg, reports_dir)
    results["prediction_grid"] = str(pred_path) if pred_path else None

    # --- Generate markdown report ---
    report_path = reports_dir / "evaluation_report.md"
    _generate_report(results, names, report_path)
    results["report_path"] = str(report_path)
    logger.info(f"Report saved to: {report_path}")

    # Print summary
    v = results["validation"]
    logger.info("=" * 60)
    logger.info("RESULTS SUMMARY")
    logger.info(f"  mAP@50     : {v['mAP50']:.4f}")
    logger.info(f"  mAP@50-95  : {v['mAP50_95']:.4f}")
    logger.info(f"  Precision  : {v['precision']:.4f}")
    logger.info(f"  Recall     : {v['recall']:.4f}")
    if underperforming:
        logger.info(f"  Underperforming: {len(underperforming)} classes")
    logger.info("=" * 60)

    return results


def _extract_metrics(metrics, names: list) -> Dict:
    """Extract metrics from Ultralytics validation results."""
    result = {
        "mAP50": float(metrics.box.map50),
        "mAP50_95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
    }

    # Per-class
    per_class = []
    try:
        ap50_values = metrics.box.ap50
        for i in range(len(ap50_values)):
            if i < len(names):
                per_class.append({
                    "id": i,
                    "name": names[i],
                    "ap50": float(ap50_values[i]),
                })
    except Exception:
        pass
    result["per_class"] = per_class
    return result


def _save_per_class_csv(per_class: list, path: Path) -> None:
    """Save per-class metrics to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "name", "ap50"])
        writer.writeheader()
        writer.writerows(per_class)


def _compute_confidence_stats(model, data_yaml, data_cfg, device_arg, e_cfg) -> Dict:
    """Compute confidence score statistics per class."""
    yolo_dir = Path(data_cfg["path"])
    test_img_dir = yolo_dir / data_cfg.get("test", "images/test")
    if not test_img_dir.exists():
        test_img_dir = yolo_dir / data_cfg.get("val", "images/valid")

    images = list(test_img_dir.glob("*.jpg"))[:200]
    if not images:
        return {}

    from collections import defaultdict
    class_confs = defaultdict(list)

    for img_path in images:
        try:
            preds = model.predict(str(img_path), conf=0.1, device=device_arg, verbose=False)
            for r in preds:
                if r.boxes is not None:
                    for box in r.boxes:
                        cls = int(box.cls.item())
                        conf = float(box.conf.item())
                        class_confs[cls].append(conf)
        except Exception:
            continue

    stats = {}
    names = data_cfg["names"]
    for cls_id, confs in class_confs.items():
        name = names[cls_id] if cls_id < len(names) else str(cls_id)
        stats[name] = {
            "count": len(confs),
            "mean": round(float(np.mean(confs)), 4),
            "median": round(float(np.median(confs)), 4),
            "std": round(float(np.std(confs)), 4),
            "min": round(float(np.min(confs)), 4),
            "max": round(float(np.max(confs)), 4),
        }
    return stats


def _save_prediction_grid(model, data_cfg, device_arg, e_cfg, reports_dir) -> Optional[Path]:
    """Generate and save a grid of prediction visualizations."""
    yolo_dir = Path(data_cfg["path"])
    test_dir = yolo_dir / data_cfg.get("test", "images/test")
    if not test_dir.exists():
        return None

    images = list(test_dir.glob("*.jpg"))
    n = min(e_cfg.get("max_prediction_images", 12), len(images))
    if n == 0:
        return None

    samples = random.sample(images, n)
    cols = 4
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(24, 6 * rows))
    axes = np.array(axes).flatten() if rows * cols > 1 else [axes]

    for idx, img_path in enumerate(samples):
        try:
            preds = model.predict(str(img_path), conf=e_cfg.get("conf_threshold", 0.25),
                                  device=device_arg, verbose=False)
            annotated = preds[0].plot()
            annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            det_count = len(preds[0].boxes) if preds[0].boxes is not None else 0

            ax = axes[idx]
            ax.imshow(annotated)
            ax.set_title(f"{img_path.name[:30]}\n({det_count} detections)", fontsize=9)
            ax.axis("off")
        except Exception:
            axes[idx].axis("off")

    for idx in range(n, len(axes)):
        axes[idx].axis("off")

    plt.suptitle("Model Predictions on Test Images", fontsize=16, fontweight="bold")
    plt.tight_layout()

    pred_dir = reports_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    path = pred_dir / "prediction_grid.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def _generate_report(results: Dict, names: list, path: Path) -> None:
    """Generate a markdown evaluation report."""
    path.parent.mkdir(parents=True, exist_ok=True)

    v = results["validation"]
    lines = [
        "# Model Evaluation Report\n",
        f"**Model**: `{results['model_path']}`  ",
        f"**Device**: {results['device']}\n",
        "## Validation Metrics\n",
        "| Metric | Value |",
        "|--------|-------|",
        f"| mAP@50 | {v['mAP50']:.4f} |",
        f"| mAP@50-95 | {v['mAP50_95']:.4f} |",
        f"| Precision | {v['precision']:.4f} |",
        f"| Recall | {v['recall']:.4f} |",
        "",
    ]

    if results.get("test"):
        t = results["test"]
        lines += [
            "## Test Metrics\n",
            "| Metric | Value |",
            "|--------|-------|",
            f"| mAP@50 | {t['mAP50']:.4f} |",
            f"| mAP@50-95 | {t['mAP50_95']:.4f} |",
            f"| Precision | {t['precision']:.4f} |",
            f"| Recall | {t['recall']:.4f} |",
            "",
        ]

    # Per-class table
    per_class = v.get("per_class", [])
    if per_class:
        lines += [
            "## Per-Class Performance\n",
            "| Class | AP@50 | Status |",
            "|-------|-------|--------|",
        ]
        threshold = 0.3
        for c in sorted(per_class, key=lambda x: x["ap50"], reverse=True):
            status = "OK" if c["ap50"] >= threshold else "LOW"
            lines.append(f"| {c['name']} | {c['ap50']:.4f} | {status} |")
        lines.append("")

    # Underperforming
    up = results.get("underperforming_classes", [])
    if up:
        lines += [
            "## Underperforming Classes\n",
            "Classes with AP@50 below threshold:\n",
        ]
        for c in up:
            lines.append(f"- **{c['name']}**: AP50 = {c['ap50']:.4f}")
        lines.append("")

    # Confidence stats
    conf = results.get("confidence_stats", {})
    if conf:
        lines += [
            "## Confidence Score Statistics\n",
            "| Class | Count | Mean | Median | Std |",
            "|-------|-------|------|--------|-----|",
        ]
        for name, s in sorted(conf.items(), key=lambda x: x[1]["count"], reverse=True):
            lines.append(f"| {name} | {s['count']} | {s['mean']:.3f} | {s['median']:.3f} | {s['std']:.3f} |")
        lines.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


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
