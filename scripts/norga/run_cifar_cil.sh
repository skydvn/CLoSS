#!/usr/bin/env bash
# NoRGa (HiDe-Prompt framework) on Split CIFAR-100, 10 tasks of 10 classes.
set -euo pipefail
cd "$(dirname "$0")/../.."
python3 main.py --config exps/norga/norga_cifar_10task.json --set "$@"
