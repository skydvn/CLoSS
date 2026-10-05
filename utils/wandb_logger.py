"""
Optional Weights & Biases logging for CaRE and CLoSS/MoSS runs.

Every function here is a no-op unless the config sets "wandb": true, so runs without a wandb
account behave exactly as before. Enable it from the command line:

    python main.py --config exps/moss/moss_domainnet_dil.json --set wandb=true
    python main.py --config ... --set wandb=true wandb_mode='"offline"'   # sync later: wandb sync <dir>

Config keys
  wandb               bool, default false
  wandb_project       default "closs"
  wandb_entity        team or user name; default: your wandb default entity
  wandb_group         default: the config's "prefix" (seeds of one setting share a group)
  wandb_name          default: "<prefix>-s<seed>"
  wandb_tags          extra tags; model name, dataset and scenario are always added
  wandb_mode          "online" (default) | "offline" | "disabled"
  wandb_log_interval  CLoSS only: log training losses every N optimizer steps (0 = off)
  wandb_save_model    upload latest_model.pth at the end (default false; ViT checkpoints are large)

Charts use "task" as their x-axis for per-task metrics and "train_step" for training losses.
"""
import glob
import json
import os

import numpy as np

_run = None
_logs_dir = None
_save_model = False

TASK_METRICS = ("acc/*", "acc_task/*", "forgetting/*", "params/*", "moss/*")


def active():
    return _run is not None


def _jsonable(v):
    try:
        json.dumps(v)
        return v
    except TypeError:
        if isinstance(v, dict):
            return {str(k): _jsonable(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [_jsonable(x) for x in v]
        return str(v)


def init(args):
    """Start one wandb run for one seed. Call after the logs folder exists."""
    global _run, _logs_dir, _save_model
    if not args.get("wandb", False):
        return None
    try:
        import wandb
    except ImportError as e:
        raise ImportError('The config sets "wandb": true but wandb is not installed. '
                          "Install it with: python -m pip install wandb") from e
    prefix = args.get("prefix", "run")
    seed = args["seed"]
    tags = list(args.get("wandb_tags") or [])
    tags += [str(args.get("model_name")), str(args.get("dataset")), str(args.get("scenario", "cil"))]
    _logs_dir = args.get("logs_name")
    _save_model = bool(args.get("wandb_save_model", False))
    _run = wandb.init(
        project=args.get("wandb_project", "closs"),
        entity=args.get("wandb_entity"),
        group=args.get("wandb_group") or prefix,
        name=args.get("wandb_name") or f"{prefix}-s{seed}",
        tags=sorted(set(tags)),
        mode=args.get("wandb_mode", "online"),
        dir=_logs_dir,
        config=_jsonable(dict(args)),
    )
    _run.define_metric("task")
    for pattern in TASK_METRICS:
        _run.define_metric(pattern, step_metric="task")
    _run.define_metric("train_step")
    _run.define_metric("train/*", step_metric="train_step")
    return _run


def log(data):
    if _run is not None:
        _run.log(_jsonable(data))


def _acc_table(matrix):
    """rows = evaluated task groups, columns = training stage (same layout as trainer.py)."""
    width = max(len(line) for line in matrix)
    a = np.zeros((len(matrix), width))
    for i, line in enumerate(matrix):
        a[i, : len(line)] = np.asarray(line, dtype=float)
    return a.T


def _forgetting(matrix):
    """Mean over previously seen groups of (best accuracy so far - current accuracy)."""
    if len(matrix) < 2:
        return 0.0
    a = _acc_table(matrix)
    prev = len(matrix[-2])
    return float(np.mean((a.max(axis=1) - a[:, -1])[:prev]))


def log_task(task, accy, top1_curve, matrix, topk_key, n_params=None):
    """Per-task accuracy metrics, logged by trainer.py after each evaluation."""
    if _run is None:
        return
    grouped = accy.get("grouped", {})
    data = {
        "task": task,
        "acc/top1": accy.get("top1"),
        f"acc/{topk_key}": accy.get(topk_key),
        "acc/avg_top1": float(np.mean(top1_curve)),
        "forgetting/top1": _forgetting(matrix),
    }
    for k in ("old", "new"):
        if k in grouped:
            data[f"acc/{k}"] = grouped[k]
    for k, v in grouped.items():
        if "-" in k:
            data[f"acc_task/{k}"] = v
    if n_params is not None:
        data["params/total"] = n_params
    _run.log(_jsonable(data))


def log_final(matrix, top1_curve):
    """Accuracy matrix as a table, plus final numbers in the run summary."""
    if _run is None or not matrix:
        return
    import wandb
    a = _acc_table(matrix)
    cols = ["task group"] + [f"after task {i}" for i in range(a.shape[1])]
    rows = [[f"group {j}"] + [round(float(x), 2) for x in a[j]] for j in range(a.shape[0])]
    _run.log({"acc/matrix": wandb.Table(columns=cols, data=rows)})
    _run.summary["final/avg_top1"] = float(np.mean(top1_curve))
    _run.summary["final/last_top1"] = float(top1_curve[-1])
    _run.summary["final/forgetting"] = _forgetting(matrix)


def finish(exit_code=0):
    """Upload the run's small files (logs, configs, diagnostics) and close the run."""
    global _run
    if _run is None:
        return
    if _logs_dir and os.path.isdir(_logs_dir):
        patterns = ["*.log", "*.json"] + (["*.pth"] if _save_model else [])
        for pattern in patterns:
            for path in glob.glob(os.path.join(_logs_dir, pattern)):
                _run.save(path, base_path=_logs_dir, policy="now")
    _run.finish(exit_code=exit_code)
    _run = None
