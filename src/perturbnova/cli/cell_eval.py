from __future__ import annotations

import argparse

from ..post_infer_eval import run_cell_eval_from_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run cell_eval on a completed PerturbNova inference output.")
    parser.add_argument("--config", required=True, help="Path to the inference TOML config.")
    parser.add_argument("--output-dir", default="", help="Optional override for experiment.output_dir.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_cell_eval_from_config(args.config, output_dir_override=args.output_dir)


if __name__ == "__main__":
    main()
