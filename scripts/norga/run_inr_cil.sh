#!/usr/bin/env bash
# NoRGa (HiDe-Prompt framework) on Split ImageNet-R, 10 tasks of 20 classes.
# Extra arguments become --set overrides, e.g.
#   bash scripts/norga/run_inr_cil.sh seed='[1993,1994,1995]' wandb=true
set -euo pipefail
cd "$(dirname "$0")/../.."
python3 main.py --config exps/norga/norga_inr_10task.json --set "$@"
