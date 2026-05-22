"""
Dataset integrity validation.

Checks: missing labels, corrupt images, invalid annotations, orphaned files.
"""

import cv2
from pathlib import Path
from typing import Dict, List, Optional

from src.utils.helpers import load_config, resolve_path, setup_logger

logger = setup_logger("validator")


def validate_dataset(config: Optional[dict] = None) -> Dict:
    """
    Validate the YOLO-format dataset for integrity issues.

    Returns a report dict with issues found per split.
    """
    if config is None:
        config = load_config()

    ds_cfg = config["dataset"]
    yolo_dir = resolve_path(ds_cfg["yolo_dir"])
    splits = ds_cfg["splits"]
    image_exts = set(ds_cfg.get("image_extensions", [".jpg", ".jpeg", ".png", ".bmp", ".webp"]))

    report = {}
    total_issues = 0

    for split in splits:
        img_dir = yolo_dir / "images" / split
        lbl_dir = yolo_dir / "labels" / split

        issues: List[str] = []
        stats = {"images": 0, "labels": 0, "missing_labels": 0,
                 "orphaned_labels": 0, "corrupt_images": 0, "invalid_labels": 0}

        if not img_dir.exists():
            issues.append(f"Image directory not found: {img_dir}")
            report[split] = {"issues": issues, "stats": stats}
            continue

        # Collect image stems
        image_files = [f for f in img_dir.iterdir() if f.suffix.lower() in image_exts]
        image_stems = {f.stem for f in image_files}
        stats["images"] = len(image_files)

        # Collect label stems
        label_files = list(lbl_dir.glob("*.txt")) if lbl_dir.exists() else []
        label_stems = {f.stem for f in label_files}
        stats["labels"] = len(label_files)

        # Missing labels
        missing = image_stems - label_stems
        stats["missing_labels"] = len(missing)
        if missing and len(missing) <= 10:
            for m in sorted(missing)[:10]:
                issues.append(f"Missing label for image: {m}")
        elif missing:
            issues.append(f"{len(missing)} images have no corresponding label file")

        # Orphaned labels
        orphaned = label_stems - image_stems
        stats["orphaned_labels"] = len(orphaned)
        if orphaned and len(orphaned) <= 5:
            for o in sorted(orphaned)[:5]:
                issues.append(f"Orphaned label (no image): {o}")
        elif orphaned:
            issues.append(f"{len(orphaned)} label files have no corresponding image")

        # Validate label format (sample check)
        invalid_count = 0
        for lbl_file in list(label_files)[:500]:  # Sample for speed
            try:
                with open(lbl_file, "r") as f:
                    for line_num, line in enumerate(f, 1):
                        parts = line.strip().split()
                        if not parts:
                            continue
                        if len(parts) < 5:
                            invalid_count += 1
                            break
                        cls_id = int(parts[0])
                        coords = [float(x) for x in parts[1:5]]
                        if cls_id < 0:
                            invalid_count += 1
                            break
                        if any(c < 0 or c > 1.01 for c in coords):
                            invalid_count += 1
                            break
            except (ValueError, UnicodeDecodeError):
                invalid_count += 1

        stats["invalid_labels"] = invalid_count
        if invalid_count:
            issues.append(f"{invalid_count} label files have formatting issues")

        # Check a few images for corruption
        corrupt_count = 0
        for img_file in list(image_files)[:100]:
            try:
                img = cv2.imread(str(img_file))
                if img is None:
                    corrupt_count += 1
            except Exception:
                corrupt_count += 1

        stats["corrupt_images"] = corrupt_count
        if corrupt_count:
            issues.append(f"{corrupt_count} images appear corrupt (of 100 sampled)")

        total_issues += len(issues)
        report[split] = {"issues": issues, "stats": stats}

        if issues:
            logger.warning(f"  {split}: {len(issues)} issue(s) found")
        else:
            logger.info(f"  {split}: OK ({stats['images']} images, {stats['labels']} labels)")

    report["total_issues"] = total_issues
    if total_issues == 0:
        logger.info("Dataset validation passed with no issues")
    else:
        logger.warning(f"Dataset validation found {total_issues} issue(s)")

    return report
