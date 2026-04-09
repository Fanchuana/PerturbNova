#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
sbatch pn_fewshot_frozen_500k.sbatch
sbatch pn_fewshot_unfrozen_500k.sbatch
sbatch pn_fewshot_v1_500k.sbatch
