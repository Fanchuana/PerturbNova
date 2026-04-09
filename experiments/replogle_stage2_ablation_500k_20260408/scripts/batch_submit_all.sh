#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
bash ./batch_submit_fewshot.sh
bash ./batch_submit_zeroshot.sh
