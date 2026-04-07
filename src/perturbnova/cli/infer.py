from __future__ import annotations

import argparse

from ..config import load_infer_config
from ..inference import DiffusionInferenceRunner
from ..utils.distributed import cleanup_distributed, init_distributed, seed_everything


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run PerturbNova inference.")
    parser.add_argument("--config", required=True, help="Path to a TOML config file.")
    parser.add_argument("--output-dir", default="", help="Optional override for experiment.output_dir.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_infer_config(args.config)
    if args.output_dir:
        config["experiment"]["output_dir"] = args.output_dir

    context = init_distributed(config["distributed"]["backend"] or None)
    seed_everything(config["experiment"]["seed"], rank=context.rank)
    try:
        runner = DiffusionInferenceRunner(config, context)
        runner.run()
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
