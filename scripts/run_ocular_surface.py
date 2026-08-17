from __future__ import annotations

import argparse

from ophthalmic_transfer.common.io import read_yaml
from ophthalmic_transfer.common.utils import get_device, seed_everything
from ophthalmic_transfer.ocular_surface.pipeline import run_pipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Run the ocular-surface transfer-learning pipeline.")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = read_yaml(args.config)
    seed_everything(int(config["cross_validation"]["random_seed"]))
    device = get_device()
    print(f"Using device: {device}")
    run_pipeline(config=config, device=device)
