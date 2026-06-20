from __future__ import annotations

import argparse
from pathlib import Path

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
        output_dir = Path(args.output_dir)
        config["experiment"]["output_dir"] = str(output_dir)
        config["output"]["prediction_path"] = str(output_dir / "predictions.h5ad")
        config["output"]["real_copy_path"] = str(output_dir / "cell_eval_real.h5ad")
        if config.get("cell_eval", {}).get("enabled", False):
            config["cell_eval"]["outdir"] = str(output_dir / "cell_eval")

    context = init_distributed(config["distributed"]["backend"] or None)
    seed_everything(config["experiment"]["seed"], rank=context.rank)
    try:
        runner = DiffusionInferenceRunner(config, context)
        runner.run()
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
