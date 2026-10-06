#!/usr/bin/env bash
# HiDe-Prompt (plain prefix tuning) vs NoRGa with different gate activations, over seeds.
# Usage: bash scripts/norga/run_ablation.sh [config] [extra --set overrides...]
#   SEEDS="1993 1994 1995" bash scripts/norga/run_ablation.sh exps/norga/norga_inr_10task.json wandb=true
set -euo pipefail
cd "$(dirname "$0")/../.."
CONFIG=${1:-exps/norga/norga_inr_10task.json}
if [ $# -gt 0 ]; then shift; fi
SEEDS=${SEEDS:-"1993 1994 1995"}
for seed in $SEEDS; do
  python3 main.py --config "$CONFIG" --set "seed=[$seed]" norga=false prefix=hideprompt "$@"
  for act in tanh sigmoid gelu; do
    python3 main.py --config "$CONFIG" --set "seed=[$seed]" norga=true gate_act=$act \
      prefix=norga_$act "$@"
  done
done
