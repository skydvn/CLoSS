#!/bin/bash
# Paper's alignment controls under the same architecture and expansion procedure:
#   subset (proposed) | none (same supports and J_t, no L_ssi) | global | random (size-matched)
# Usage: bash scripts/moss/run_alignment_ablation.sh [config] [seed ...]
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG="${1:-$ROOT_DIR/exps/moss/moss_domainnet_dil.json}"
shift || true
SEEDS=("${@:-1993 1994 1995}")
NAME="$(basename "$CONFIG" .json)"

for MODE in subset none global random; do
  for SEED in ${SEEDS[@]}; do
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
    python3 "$ROOT_DIR/main.py" --config "$CONFIG" \
      --prefix "${NAME}_${MODE}_s${SEED}" \
      --set ssi_mode="\"$MODE\"" seed="[$SEED]"
  done
done
