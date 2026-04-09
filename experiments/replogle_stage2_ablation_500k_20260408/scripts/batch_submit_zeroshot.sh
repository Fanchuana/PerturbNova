#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
sbatch pn_zeroshot_frozen_500k.sbatch
sbatch pn_zeroshot_unfrozen_500k.sbatch
sbatch pn_zeroshot_v1_500k.sbatch
