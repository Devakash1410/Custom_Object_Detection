"""Export-only script. Usage: python scripts/run_export.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.helpers import load_config
from src.export.exporter import export_model

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--model", type=str, default=None, help="Path to best.pt")
    parser.add_argument("--version", type=int, default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    export_model(model_path=args.model, config=config, version=args.version)
