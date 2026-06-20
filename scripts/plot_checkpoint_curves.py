#!/usr/bin/env python3
"""Plot metric curves across checkpoint evaluations for two models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages


def load_curve(model_dir: Path) -> pd.DataFrame:
    rows = []
    for result_path in sorted(model_dir.glob("checkpoint_eval/epoch_*/results.json")):
        with open(result_path) as f:
            data = json.load(f)
        epoch = int(result_path.parent.name.split("_")[-1])
        rows.append(
            {
                "epoch": epoch,
                "mse": data["metrics"]["mse"],
                "r2": data["metrics"]["r2"],
            }
        )
    return pd.DataFrame(rows).sort_values("epoch")


def load_cell_eval_curve(model_dir: Path) -> pd.DataFrame:
    rows = []
    for agg_path in sorted(model_dir.glob("checkpoint_eval/epoch_*/cell_eval/*_agg_results.csv")):
        epoch = int(agg_path.parent.parent.name.split("_")[-1])
        df = pd.read_csv(agg_path)
        df.columns = [c.strip() for c in df.columns]
        mean = df[df["statistic"].astype(str).str.strip().eq("mean")].iloc[0]
        row = {"epoch": epoch}
        for col in df.columns:
            if col == "statistic":
                continue
            row[col] = float(mean[col])
        rows.append(row)
    return pd.DataFrame(rows).sort_values("epoch")


def plot_metric(ax, df1, df2, metric, label1, label2, title):
    ax.plot(df1["epoch"], df1[metric], marker="o", label=label1)
    ax.plot(df2["epoch"], df2[metric], marker="o", label=label2)
    ax.set_title(title)
    ax.set_xlabel("epoch")
    ax.set_ylabel(metric)
    ax.legend()
    ax.grid(True, alpha=0.3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vae-dir", type=Path, required=True)
    parser.add_argument("--evo-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("checkpoint_curves.pdf"))
    parser.add_argument("--csv-output", type=Path, default=None)
    args = parser.parse_args()

    vae_train = load_curve(args.vae_dir)
    evo_train = load_curve(args.evo_dir)
    vae_ce = load_cell_eval_curve(args.vae_dir)
    evo_ce = load_cell_eval_curve(args.evo_dir)

    vae = vae_train.merge(vae_ce, on="epoch", how="outer", suffixes=("", "_cell_eval"))
    evo = evo_train.merge(evo_ce, on="epoch", how="outer", suffixes=("", "_cell_eval"))
    vae.insert(0, "model", "VAE")
    evo.insert(0, "model", "Evoformer")
    combined = pd.concat([vae, evo], ignore_index=True).sort_values(["model", "epoch"])

    csv_output = args.csv_output or args.output.with_suffix(".csv")
    combined.to_csv(csv_output, index=False)

    metrics = [
        col
        for col in combined.columns
        if col not in {"model", "epoch"} and pd.api.types.is_numeric_dtype(combined[col])
    ]
    metrics_per_page = 12
    with PdfPages(args.output) as pdf:
        for start in range(0, len(metrics), metrics_per_page):
            page_metrics = metrics[start : start + metrics_per_page]
            nrows = 4
            ncols = 3
            fig, axes = plt.subplots(nrows, ncols, figsize=(16, 18))
            axes = axes.ravel()
            for ax, metric in zip(axes, page_metrics):
                plot_metric(ax, vae, evo, metric, "VAE", "Evoformer", metric)
            for ax in axes[len(page_metrics):]:
                ax.axis("off")
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    print(csv_output)
    print(args.output)


if __name__ == "__main__":
    main()
