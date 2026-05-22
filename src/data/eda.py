"""
Exploratory Data Analysis — visualizations and statistics.

Uses standard fonts only (no emoji in axis labels). Saves all plots to output dir.
"""

import yaml
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import random
from pathlib import Path
from collections import Counter
from typing import Dict, Optional

from src.utils.helpers import load_config, resolve_path, setup_logger

logger = setup_logger("eda")

# Reproducibility
random.seed(42)
np.random.seed(42)

# Standard plot style
plt.rcParams.update({
    "figure.figsize": (14, 8),
    "font.size": 11,
    "font.family": "sans-serif",
    "axes.titleweight": "bold",
    "axes.grid": True,
    "grid.alpha": 0.3,
})


def run_eda(config: Optional[dict] = None, show: bool = False) -> Dict:
    """
    Run full EDA on the YOLO dataset.
    Generates plots and statistics. Saves to outputs/reports/.

    Returns dict with dataset stats and file paths.
    """
    if config is None:
        config = load_config()

    ds_cfg = config["dataset"]
    yolo_dir = resolve_path(ds_cfg["yolo_dir"])
    splits = ds_cfg["splits"]
    out_dir = resolve_path(config["output"]["reports_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load class names
    yaml_path = resolve_path("configs") / "data.yaml"
    if not yaml_path.exists():
        yaml_path = resolve_path("data.yaml")
    with open(yaml_path, "r") as f:
        data_cfg = yaml.safe_load(f)
    names = data_cfg["names"]
    num_classes = len(names)

    results = {"class_names": names, "num_classes": num_classes, "splits": {}}

    # --- Dataset statistics ---
    logger.info("Computing dataset statistics...")
    for split in splits:
        img_dir = yolo_dir / "images" / split
        lbl_dir = yolo_dir / "labels" / split
        num_imgs = len(list(img_dir.glob("*.jpg"))) if img_dir.exists() else 0
        total_objects = 0
        if lbl_dir.exists():
            for lf in lbl_dir.glob("*.txt"):
                with open(lf) as f:
                    total_objects += sum(1 for line in f if line.strip())
        avg_obj = total_objects / num_imgs if num_imgs else 0
        results["splits"][split] = {
            "images": num_imgs, "objects": total_objects, "avg_per_image": round(avg_obj, 1)
        }
        logger.info(f"  {split}: {num_imgs:,} images, {total_objects:,} objects, {avg_obj:.1f}/img")

    # --- Class distribution ---
    logger.info("Generating class distribution plot...")
    class_counts = Counter()
    train_lbl_dir = yolo_dir / "labels" / "train"
    if train_lbl_dir.exists():
        for lf in train_lbl_dir.glob("*.txt"):
            with open(lf) as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        class_counts[int(parts[0])] += 1

    sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
    labels = [names[cid] if cid < len(names) else str(cid) for cid, _ in sorted_classes]
    values = [cnt for _, cnt in sorted_classes]

    fig, ax = plt.subplots(figsize=(16, max(8, len(labels) * 0.35)))
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(labels)))
    bars = ax.barh(range(len(labels)), values, color=colors)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Number of Instances")
    ax.set_title("Class Distribution (Training Set)")
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + max(values) * 0.01,
                bar.get_y() + bar.get_height() / 2, f"{val:,}", va="center", fontsize=9)
    plt.tight_layout()
    dist_path = out_dir / "class_distribution.png"
    plt.savefig(dist_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()
    results["class_distribution_path"] = str(dist_path)

    # --- Sample images with bounding boxes ---
    logger.info("Generating sample images grid...")
    train_img_dir = yolo_dir / "images" / "train"
    all_imgs = list(train_img_dir.glob("*.jpg")) if train_img_dir.exists() else []
    sample_imgs = random.sample(all_imgs, min(9, len(all_imgs)))

    np.random.seed(42)
    class_colors = {i: tuple(np.random.randint(50, 255, 3).tolist()) for i in range(num_classes)}

    if sample_imgs:
        fig, axes = plt.subplots(3, 3, figsize=(18, 18))
        axes_flat = axes.flatten()
        for idx, img_path in enumerate(sample_imgs):
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w = img.shape[:2]
            ax = axes_flat[idx]
            ax.imshow(img)

            lbl_path = train_lbl_dir / (img_path.stem + ".txt")
            if lbl_path.exists():
                with open(lbl_path) as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            cid = int(parts[0])
                            xc, yc, bw, bh = [float(p) for p in parts[1:5]]
                            x1 = int((xc - bw / 2) * w)
                            y1 = int((yc - bh / 2) * h)
                            bw_px, bh_px = int(bw * w), int(bh * h)
                            c = [v / 255.0 for v in class_colors.get(cid, (200, 200, 200))]
                            rect = patches.Rectangle((x1, y1), bw_px, bh_px,
                                                     linewidth=2, edgecolor=c, facecolor="none")
                            ax.add_patch(rect)
                            name = names[cid] if cid < len(names) else str(cid)
                            ax.text(x1, max(0, y1 - 3), name, fontsize=7, color="white",
                                    bbox=dict(boxstyle="round,pad=0.2", facecolor=c, alpha=0.8))
            ax.set_title(img_path.name[:40], fontsize=9)
            ax.axis("off")
        plt.suptitle("Sample Training Images with Ground Truth", fontsize=16, fontweight="bold")
        plt.tight_layout()
        samples_path = out_dir / "sample_images.png"
        plt.savefig(samples_path, dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        plt.close()
        results["sample_images_path"] = str(samples_path)

    # --- BBox size distribution ---
    logger.info("Generating bbox distribution plot...")
    widths, heights, areas = [], [], []
    for lf in list(train_lbl_dir.glob("*.txt"))[:5000]:
        with open(lf) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    bw, bh = float(parts[3]), float(parts[4])
                    widths.append(bw)
                    heights.append(bh)
                    areas.append(bw * bh)

    if widths:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        axes[0].hist(widths, bins=50, color="steelblue", edgecolor="white", alpha=0.8)
        axes[0].set_title("BBox Width Distribution")
        axes[0].set_xlabel("Normalized Width")
        axes[0].set_ylabel("Count")
        axes[1].hist(heights, bins=50, color="coral", edgecolor="white", alpha=0.8)
        axes[1].set_title("BBox Height Distribution")
        axes[1].set_xlabel("Normalized Height")
        axes[2].hist(areas, bins=50, color="mediumseagreen", edgecolor="white", alpha=0.8)
        axes[2].set_title("BBox Area Distribution")
        axes[2].set_xlabel("Normalized Area")
        plt.suptitle("Bounding Box Size Distribution (Training Set)", fontsize=14, fontweight="bold")
        plt.tight_layout()
        bbox_path = out_dir / "bbox_distribution.png"
        plt.savefig(bbox_path, dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        plt.close()
        results["bbox_distribution_path"] = str(bbox_path)

    logger.info("EDA complete. Reports saved to: %s", out_dir)
    return results
