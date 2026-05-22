"""
COCO-to-YOLO format converter.

- Symlinks images first; falls back to copying if symlinks fail.
- Dynamically infers classes from COCO annotations.
- Auto-generates data.yaml with relative paths.
- Idempotent: skips already-converted files.
"""

import json
import shutil
import yaml
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from src.utils.helpers import (
    load_config,
    resolve_path,
    setup_logger,
    can_create_symlinks,
    try_symlink,
)

logger = setup_logger("converter")


def _parse_coco(coco_json_path: Path) -> dict:
    """Load and parse a COCO annotation JSON file."""
    with open(coco_json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _convert_split(
    coco_data: dict,
    images_dir: Path,
    out_images_dir: Path,
    out_labels_dir: Path,
    use_symlinks: bool,
    image_extensions: list,
) -> Tuple[int, int]:
    """
    Convert one split from COCO to YOLO format.
    Returns (converted_count, skipped_count).
    """
    out_images_dir.mkdir(parents=True, exist_ok=True)
    out_labels_dir.mkdir(parents=True, exist_ok=True)

    images_info = {img["id"]: img for img in coco_data["images"]}
    img_annotations = defaultdict(list)
    for ann in coco_data["annotations"]:
        img_annotations[ann["image_id"]].append(ann)

    converted = 0
    skipped = 0

    for img_id, img_info in images_info.items():
        img_filename = img_info["file_name"]
        img_w = img_info["width"]
        img_h = img_info["height"]

        if img_w <= 0 or img_h <= 0:
            skipped += 1
            continue

        src_img = images_dir / img_filename
        if not src_img.exists():
            # Try other extensions
            found = False
            stem = Path(img_filename).stem
            for ext in image_extensions:
                candidate = images_dir / f"{stem}{ext}"
                if candidate.exists():
                    src_img = candidate
                    img_filename = f"{stem}{ext}"
                    found = True
                    break
            if not found:
                skipped += 1
                continue

        dst_img = out_images_dir / img_filename
        if not dst_img.exists() and not dst_img.is_symlink():
            if use_symlinks:
                if not try_symlink(src_img, dst_img):
                    shutil.copy2(src_img, dst_img)
            else:
                shutil.copy2(src_img, dst_img)

        # Write YOLO label
        label_filename = Path(img_filename).stem + ".txt"
        label_path = out_labels_dir / label_filename
        annotations = img_annotations.get(img_id, [])

        with open(label_path, "w", encoding="utf-8") as lf:
            for ann in annotations:
                cat_id = ann["category_id"]
                x_min, y_min, bbox_w, bbox_h = ann["bbox"]

                x_center = (x_min + bbox_w / 2.0) / img_w
                y_center = (y_min + bbox_h / 2.0) / img_h
                norm_w = bbox_w / img_w
                norm_h = bbox_h / img_h

                # Clamp to [0, 1]
                x_center = max(0.0, min(1.0, x_center))
                y_center = max(0.0, min(1.0, y_center))
                norm_w = max(0.0, min(1.0, norm_w))
                norm_h = max(0.0, min(1.0, norm_h))

                lf.write(
                    f"{cat_id} {x_center:.6f} {y_center:.6f} "
                    f"{norm_w:.6f} {norm_h:.6f}\n"
                )

        converted += 1

    return converted, skipped


def infer_classes(dataset_dir: Path, splits: list, annotation_file: str) -> List[dict]:
    """Dynamically infer classes from COCO annotation files."""
    all_categories = {}
    for split in splits:
        coco_json = dataset_dir / split / annotation_file
        if not coco_json.exists():
            continue
        coco_data = _parse_coco(coco_json)
        for cat in coco_data.get("categories", []):
            all_categories[cat["id"]] = cat
    return [all_categories[k] for k in sorted(all_categories.keys())]


def convert_dataset(config: Optional[dict] = None) -> Dict:
    """
    Main entry point: Convert COCO dataset to YOLO format.

    Returns dict with class_names, num_classes, yaml_path, and per-split stats.
    """
    if config is None:
        config = load_config()

    ds_cfg = config["dataset"]
    dataset_dir = resolve_path(ds_cfg["raw_dir"])
    yolo_dir = resolve_path(ds_cfg["yolo_dir"])
    splits = ds_cfg["splits"]
    annotation_file = ds_cfg["annotation_file"]
    image_extensions = ds_cfg.get("image_extensions", [".jpg", ".jpeg", ".png"])

    # Decide symlink vs copy
    prefer_symlinks = ds_cfg.get("use_symlinks", True)
    use_symlinks = False
    if prefer_symlinks:
        use_symlinks = can_create_symlinks(yolo_dir / ".symlink_test")
        if use_symlinks:
            logger.info("Using symlinks for images (saves disk space)")
        else:
            logger.info("Symlinks not supported; falling back to file copy")

    # Infer classes dynamically
    categories = infer_classes(dataset_dir, splits, annotation_file)
    class_names = [cat["name"] for cat in categories]
    num_classes = len(class_names)
    logger.info(f"Inferred {num_classes} classes from annotations")

    # Convert each split
    stats = {}
    for split in splits:
        coco_json = dataset_dir / split / annotation_file
        if not coco_json.exists():
            logger.warning(f"  {split}: annotation file not found, skipping")
            stats[split] = {"converted": 0, "skipped": 0}
            continue

        coco_data = _parse_coco(coco_json)
        out_images = yolo_dir / "images" / split
        out_labels = yolo_dir / "labels" / split

        converted, skipped = _convert_split(
            coco_data, dataset_dir / split, out_images, out_labels,
            use_symlinks, image_extensions,
        )
        stats[split] = {"converted": converted, "skipped": skipped}
        logger.info(f"  {split}: {converted} converted, {skipped} skipped")

    # Generate data.yaml
    data_yaml = {
        "path": str(yolo_dir.resolve()),
        "train": "images/train",
        "val": "images/valid",
        "test": "images/test",
        "nc": num_classes,
        "names": class_names,
    }
    yaml_path = resolve_path("configs") / "data.yaml"
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(data_yaml, f, default_flow_style=False, sort_keys=False)

    logger.info(f"data.yaml saved to {yaml_path}")

    return {
        "class_names": class_names,
        "num_classes": num_classes,
        "yaml_path": str(yaml_path),
        "stats": stats,
        "use_symlinks": use_symlinks,
    }
