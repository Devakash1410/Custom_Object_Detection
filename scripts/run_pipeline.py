"""
Full pipeline: Dataset conversion -> Training -> Evaluation -> Export

Usage:
    python scripts/run_pipeline.py
    python scripts/run_pipeline.py --start-from train
    python scripts/run_pipeline.py --start-from evaluate
    python scripts/run_pipeline.py --start-from export
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.helpers import load_config, setup_logger

logger = setup_logger("pipeline")

STEPS = ["convert", "eda", "validate", "train", "evaluate", "export"]


def main():
    parser = argparse.ArgumentParser(description="YOLOv8 Urban Detection Pipeline")
    parser.add_argument("--config", type=str, default=None, help="Config file path")
    parser.add_argument("--start-from", type=str, default="convert",
                        choices=STEPS, help="Start from this step")
    parser.add_argument("--skip", nargs="*", default=[], choices=STEPS,
                        help="Steps to skip")
    args = parser.parse_args()

    config = load_config(args.config)
    start_idx = STEPS.index(args.start_from)

    logger.info("=" * 60)
    logger.info("YOLOv8 URBAN DETECTION PIPELINE")
    logger.info(f"Starting from: {args.start_from}")
    logger.info("=" * 60)

    version = None
    model_path = None

    for step in STEPS[start_idx:]:
        if step in args.skip:
            logger.info(f"Skipping: {step}")
            continue

        logger.info(f"\n{'='*60}")
        logger.info(f"STEP: {step.upper()}")
        logger.info(f"{'='*60}")

        if step == "convert":
            from src.data.converter import convert_dataset
            result = convert_dataset(config)
            logger.info(f"Converted {result['num_classes']} classes")

        elif step == "eda":
            from src.data.eda import run_eda
            run_eda(config, show=False)

        elif step == "validate":
            from src.data.validator import validate_dataset
            report = validate_dataset(config)
            if report.get("total_issues", 0) > 0:
                logger.warning("Dataset has issues — review before training")

        elif step == "train":
            from src.training.trainer import train_model
            result = train_model(config)
            version = result["version"]
            model_path = result["best_pt"]

        elif step == "evaluate":
            from src.evaluation.evaluator import evaluate_model
            evaluate_model(model_path=model_path, config=config, version=version)

        elif step == "export":
            from src.export.exporter import export_model
            export_model(model_path=model_path, config=config, version=version)

    logger.info(f"\n{'='*60}")
    logger.info("PIPELINE COMPLETE")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
