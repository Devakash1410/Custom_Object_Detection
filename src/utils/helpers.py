"""
Utility helpers: config loading, logging, versioning, device detection, OOM handling.
"""

import os
import sys
import json
import yaml
import logging
import platform
import datetime
from pathlib import Path
from typing import Any, Dict, Optional

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logger(name: str = "urban_detection", log_file: Optional[str] = None) -> logging.Logger:
    """Create a formatted logger."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)

    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    logger.addHandler(console)

    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load YAML config. Falls back to configs/config.yaml."""
    if config_path is None:
        config_path = PROJECT_ROOT / "configs" / "config.yaml"
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def resolve_path(relative: str) -> Path:
    """Resolve a path relative to project root."""
    return (PROJECT_ROOT / relative).resolve()


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

def get_device() -> tuple:
    """Detect best available device. Returns (device_arg, device_name)."""
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            mem_bytes = torch.cuda.get_device_properties(0).total_memory
            mem_gb = mem_bytes / (1024 ** 3)
            return 0, f"{name} ({mem_gb:.1f} GB)"
        else:
            return "cpu", "CPU"
    except ImportError:
        return "cpu", "CPU (torch not available)"


def supports_amp() -> bool:
    """Check if mixed-precision (AMP) training is supported."""
    try:
        import torch
        if torch.cuda.is_available():
            capability = torch.cuda.get_device_capability(0)
            # AMP requires compute capability >= 7.0 (Volta+)
            # GTX 1650 is 7.5 so this should be True
            return capability[0] >= 7
        return False
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Versioning
# ---------------------------------------------------------------------------

def get_next_version(models_dir: str) -> int:
    """Get next model version number (auto-increment)."""
    models_path = resolve_path(models_dir)
    if not models_path.exists():
        return 1
    existing = [
        int(d.name.replace("v", ""))
        for d in models_path.iterdir()
        if d.is_dir() and d.name.startswith("v") and d.name[1:].isdigit()
    ]
    return max(existing, default=0) + 1


def create_metadata(
    config: Dict[str, Any],
    version: int,
    metrics: Optional[Dict[str, float]] = None,
    training_duration: Optional[float] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Create metadata dict for a model version."""
    device_arg, device_name = get_device()
    meta = {
        "version": version,
        "timestamp": datetime.datetime.now().isoformat(),
        "platform": platform.platform(),
        "python_version": sys.version.split()[0],
        "device": device_name,
        "training_config": config.get("training", {}),
        "export_config": config.get("export", {}),
    }
    if metrics:
        meta["metrics"] = metrics
    if training_duration:
        meta["training_duration_seconds"] = round(training_duration, 2)
    if extra:
        meta.update(extra)

    # Try to add package versions
    try:
        import torch
        meta["torch_version"] = torch.__version__
    except ImportError:
        pass
    try:
        import ultralytics
        meta["ultralytics_version"] = ultralytics.__version__
    except ImportError:
        pass

    return meta


def save_metadata(meta: Dict[str, Any], path: Path) -> None:
    """Save metadata to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Symlink helpers
# ---------------------------------------------------------------------------

def try_symlink(src: Path, dst: Path) -> bool:
    """
    Try to create a symlink. Returns True on success, False on failure.
    Works cross-platform: Windows may need developer mode or admin.
    """
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() or dst.is_symlink():
            return True
        dst.symlink_to(src.resolve())
        return True
    except (OSError, PermissionError, NotImplementedError):
        return False


def can_create_symlinks(test_dir: Path) -> bool:
    """Test if symlinks are supported in the given directory."""
    test_dir.mkdir(parents=True, exist_ok=True)
    test_src = test_dir / "__symlink_test_src__.tmp"
    test_dst = test_dir / "__symlink_test_dst__.tmp"
    try:
        test_src.write_text("test")
        test_dst.symlink_to(test_src.resolve())
        result = test_dst.exists()
        return result
    except (OSError, PermissionError, NotImplementedError):
        return False
    finally:
        for f in [test_dst, test_src]:
            try:
                f.unlink(missing_ok=True)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------

def save_labels_json(class_names: list, path: Path) -> None:
    """Save class names as labels.json for frontend."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(class_names, f, indent=2, ensure_ascii=False)
