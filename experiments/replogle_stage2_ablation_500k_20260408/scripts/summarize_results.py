#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path

EXP_DIR = Path('/work/home/cryoem666/xyf/temp/pycharm/PerturbNova/experiments/replogle_stage2_ablation_500k_20260408')
OUT_ROOT = Path('/work/home/cryoem666/xyf/temp/pycharm/PerturbNova/outputs/experiments/replogle_stage2_ablation_500k_20260408')
RESULT_PATH = EXP_DIR / 'summary_results.csv'
SPLITS = ['hepg2', 'jurkat', 'k562', 'rpe1']
TASKS = ['zeroshot', 'fewshot']
VARIANTS = ['frozen_500k', 'unfrozen_500k', 'v1_500k']
METRICS = ['pearson_delta', 'mse', 'mae', 'mse_delta', 'mae_delta', 'discrimination_score_l1', 'discrimination_score_l2', 'discrimination_score_cosine']


def load_mean_row(path: Path) -> dict[str, str]:
    with path.open('r', encoding='utf-8') as f:
        reader = csv.reader(f)
        rows = list(reader)
    header = [c.strip() for c in rows[0]]
    mean_row = None
    for row in rows[1:]:
        if row and row[0].strip() == 'mean':
            mean_row = [c.strip() for c in row]
            break
    if mean_row is None:
        raise ValueError(f'mean row not found in {path}')
    return {h: v for h, v in zip(header, mean_row)}


records = []
for task in TASKS:
    for split in SPLITS:
        for variant in VARIANTS:
            agg = OUT_ROOT / task / split / f'{variant}_infer' / 'cell_eval' / f'{split}_agg_results.csv'
            record = {'task': task, 'split': split, 'variant': variant, 'agg_path': str(agg), 'exists': str(agg.exists())}
            if agg.exists():
                row = load_mean_row(agg)
                for m in METRICS:
                    record[m] = row.get(m, '')
            else:
                for m in METRICS:
                    record[m] = ''
            records.append(record)

with RESULT_PATH.open('w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['task', 'split', 'variant', 'exists', 'agg_path', *METRICS])
    writer.writeheader()
    writer.writerows(records)

print(f'wrote {RESULT_PATH}')
