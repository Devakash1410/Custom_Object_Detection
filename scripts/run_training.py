"""Training-only script. Usage: python scripts/run_training.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.helpers import load_config
from src.training.trainer import train_model

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint path to resume")
    args = parser.parse_args()
    config = load_config(args.config)
    train_model(config=config, resume_path=args.resume)
