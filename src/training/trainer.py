"""
Training pipeline with:
- Auto batch sizing (Ultralytics AutoBatch)
- OOM recovery with automatic batch reduction
- Resume interrupted training
- Mixed precision if supported
- Best model + checkpoints + metadata + versioned output
- GTX 1650 optimized defaults
"""

import os
import json
import time
import shutil
import yaml
import csv
from pathlib import Path
from typing import Dict, Optional

from src.utils.helpers import (
    load_config,
    resolve_path,
    setup_logger,
    get_device,
    supports_amp,
    get_next_version,
    create_metadata,
    save_metadata,
    save_labels_json,
)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
logger = setup_logger("trainer")


def train_model(config: Optional[dict] = None, resume_path: Optional[str] = None) -> Dict:
    """
    Train YOLOv8 model with production-grade features.

    Args:
        config: Config dict. Loaded from file if None.
        resume_path: Path to checkpoint to resume from. Overrides config.

    Returns:
        Dict with training results, paths, and metadata.
    """
    from ultralytics import YOLO

    if config is None:
        config = load_config()

    t_cfg = config["training"]
    o_cfg = config["output"]
    device_arg, device_name = get_device()

    logger.info("=" * 60)
    logger.info("TRAINING PIPELINE")
    logger.info("=" * 60)
    logger.info(f"Device: {device_name}")

    # --- Resolve data.yaml ---
    data_yaml = resolve_path("configs") / "data.yaml"
    if not data_yaml.exists():
        data_yaml = resolve_path("data.yaml")
    if not data_yaml.exists():
        raise FileNotFoundError("data.yaml not found. Run dataset conversion first.")

    with open(data_yaml, "r") as f:
        data_cfg = yaml.safe_load(f)
    class_names = data_cfg["names"]
    logger.info(f"Dataset: {data_cfg['nc']} classes, yaml={data_yaml}")

    # --- Load or resume model ---
    if resume_path or t_cfg.get("resume"):
        ckpt = resume_path or _find_last_checkpoint(o_cfg)
        if ckpt and Path(ckpt).exists():
            logger.info(f"Resuming from checkpoint: {ckpt}")
            model = YOLO(ckpt)
        else:
            logger.warning("No checkpoint found to resume. Starting fresh.")
            model = YOLO(t_cfg["model"])
    else:
        model = YOLO(t_cfg["model"])
        logger.info(f"Model: {t_cfg['model']}")

    # --- Training parameters ---
    batch = t_cfg.get("batch", -1)  # -1 = auto
    amp = t_cfg.get("amp", True) and supports_amp()
    epochs = t_cfg.get("epochs", 25)
    imgsz = t_cfg.get("imgsz", 640)

    logger.info(f"Epochs: {epochs}, ImgSize: {imgsz}, Batch: {'auto' if batch == -1 else batch}")
    logger.info(f"AMP: {amp}, Patience: {t_cfg.get('patience', 10)}")

    # --- Versioned output directory ---
    version = get_next_version(o_cfg["models_dir"])
    version_dir = resolve_path(o_cfg["models_dir"]) / f"v{version}"
    version_dir.mkdir(parents=True, exist_ok=True)

    project_dir = resolve_path("runs") / "detect"
    run_name = f"v{version}"

    # --- Train with OOM recovery ---
    start_time = time.time()
    results = None
    attempts = 0
    max_attempts = 3

    while attempts < max_attempts:
        attempts += 1
        try:
            logger.info(f"Training attempt {attempts} (batch={'auto' if batch == -1 else batch})...")

            results = model.train(
                data=str(data_yaml),
                epochs=epochs,
                imgsz=imgsz,
                batch=batch,
                device=device_arg,
                project=str(project_dir),
                name=run_name,
                exist_ok=True,
                plots=True,
                patience=t_cfg.get("patience", 10),
                save=True,
                save_period=t_cfg.get("save_period", 5),
                verbose=True,
                pretrained=True,
                optimizer=t_cfg.get("optimizer", "auto"),
                lr0=t_cfg.get("lr0", 0.01),
                lrf=t_cfg.get("lrf", 0.01),
                momentum=t_cfg.get("momentum", 0.937),
                weight_decay=t_cfg.get("weight_decay", 0.0005),
                warmup_epochs=t_cfg.get("warmup_epochs", 3.0),
                warmup_momentum=t_cfg.get("warmup_momentum", 0.8),
                warmup_bias_lr=t_cfg.get("warmup_bias_lr", 0.1),
                close_mosaic=t_cfg.get("close_mosaic", 10),
                amp=amp,
                workers=t_cfg.get("workers", 2),
                cache=t_cfg.get("cache", False),
                seed=t_cfg.get("seed", 42),
                resume=bool(resume_path or t_cfg.get("resume")),
            )
            break  # Success

        except RuntimeError as e:
            error_msg = str(e).lower()
            if "out of memory" in error_msg or "cuda" in error_msg:
                if not t_cfg.get("oom_retry", True) or attempts >= max_attempts:
                    logger.error(f"OOM error and no more retries. Error: {e}")
                    raise

                # Reduce batch size
                fraction = t_cfg.get("oom_batch_fraction", 0.5)
                if batch == -1:
                    batch = 8  # Start with 8 if auto failed
                else:
                    batch = max(1, int(batch * fraction))

                logger.warning(f"OOM detected! Reducing batch to {batch} and retrying...")

                # Clear GPU memory
                try:
                    import torch
                    torch.cuda.empty_cache()
                except Exception:
                    pass
            else:
                raise

    training_duration = time.time() - start_time
    logger.info(f"Training completed in {training_duration / 60:.1f} minutes")

    # --- Save artifacts ---
    run_dir = project_dir / run_name
    best_pt = run_dir / "weights" / "best.pt"
    last_pt = run_dir / "weights" / "last.pt"

    # Copy best model to versioned dir
    if best_pt.exists():
        shutil.copy2(best_pt, version_dir / "best.pt")
        logger.info(f"Best model saved to: {version_dir / 'best.pt'}")
    if last_pt.exists():
        shutil.copy2(last_pt, version_dir / "last.pt")

    # Save class labels
    save_labels_json(class_names, version_dir / "labels.json")

    # Save config snapshot
    config_snapshot = version_dir / "config_snapshot.yaml"
    with open(config_snapshot, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    # Extract metrics
    metrics = {}
    if results is not None:
        try:
            metrics = {
                "mAP50": float(results.results_dict.get("metrics/mAP50(B)", 0)),
                "mAP50-95": float(results.results_dict.get("metrics/mAP50-95(B)", 0)),
                "precision": float(results.results_dict.get("metrics/precision(B)", 0)),
                "recall": float(results.results_dict.get("metrics/recall(B)", 0)),
            }
        except Exception:
            pass

    # Save metadata
    meta = create_metadata(config, version, metrics, training_duration, {
        "batch_used": batch,
        "amp_used": amp,
        "attempts": attempts,
        "run_dir": str(run_dir),
    })
    save_metadata(meta, version_dir / "metadata.json")

    # Save training history CSV
    _save_training_history(run_dir, config)

    logger.info("=" * 60)
    logger.info(f"Model version: v{version}")
    logger.info(f"Best weights: {version_dir / 'best.pt'}")
    if metrics:
        logger.info(f"mAP@50: {metrics.get('mAP50', 'N/A'):.4f}")
        logger.info(f"mAP@50-95: {metrics.get('mAP50-95', 'N/A'):.4f}")
    logger.info("=" * 60)

    return {
        "version": version,
        "version_dir": str(version_dir),
        "best_pt": str(version_dir / "best.pt"),
        "run_dir": str(run_dir),
        "metrics": metrics,
        "training_duration": training_duration,
        "batch_used": batch,
        "class_names": class_names,
    }


def _find_last_checkpoint(o_cfg: dict) -> Optional[str]:
    """Find the most recent last.pt checkpoint."""
    models_dir = resolve_path(o_cfg["models_dir"])
    if not models_dir.exists():
        return None
    versions = sorted(
        [d for d in models_dir.iterdir() if d.is_dir() and d.name.startswith("v")],
        key=lambda d: int(d.name[1:]) if d.name[1:].isdigit() else 0,
        reverse=True,
    )
    for v in versions:
        last = v / "last.pt"
        if last.exists():
            return str(last)
    return None


def _save_training_history(run_dir: Path, config: dict) -> None:
    """Copy results.csv from training run to outputs."""
    results_csv = run_dir / "results.csv"
    if results_csv.exists():
        history_dir = resolve_path(config["output"]["history_dir"])
        history_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(results_csv, history_dir / "metrics.csv")
        logger.info(f"Training history saved to: {history_dir / 'metrics.csv'}")
