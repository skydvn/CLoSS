#!/bin/bash
# Paper's alignment controls under the same architecture and expansion procedure:
#   subset (proposed) | none (same supports and J_t, no L_ssi) | global | random (size-matched)
# Usage: bash scripts/moss/run_alignment_ablation.sh [config] [seed ...]
# Set WANDB=1 to log every run to Weights & Biases, grouped by alignment mode.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG="${1:-$ROOT_DIR/exps/moss/moss_domainnet_dil.json}"
shift || true
if (($#)); then SEEDS=("$@"); else SEEDS=(1993 1994 1995); fi
NAME="$(basename "$CONFIG" .json)"
WANDB_ARGS=()
if [[ "${WANDB:-0}" == "1" ]]; then
  WANDB_ARGS=(wandb=true)
fi

for MODE in subset none global random; do
  for SEED in "${SEEDS[@]}"; do
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
    python3 "$ROOT_DIR/main.py" --config "$CONFIG" \
      --prefix "${NAME}_${MODE}_s${SEED}" \
      --set ssi_mode="\"$MODE\"" seed="[$SEED]" \
            wandb_group="\"${NAME}_${MODE}\"" wandb_tags="[\"$MODE\"]" "${WANDB_ARGS[@]}"
  done
done
