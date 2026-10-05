# MoSS in this repository

This adds **MoSS (Mixture of Specialized Skills with subset-shared invariance)** as a second learner next to CaRE. It reuses the repo's training loop, logging, and accuracy/forgetting reporting. Select it with `"model_name": "moss"`. The original CaRE code path is unchanged.

MoSS is defined for **domain-incremental** learning (one label space, tasks are domains), so the integration also adds a domain-incremental data manager. The repo's class-incremental benchmarks still run, but see [the CIL caveat](#class-incremental-runs).

## What was added

| File | Purpose |
|---|---|
| `models/moss.py` | Learner. Reuse, support estimation, selective adaptation, expansion trials, retention, evaluation, diagnostics. |
| `backbone/moss_moe.py` | Expert bank (two-layer MLPs), linear top-k router, shared head, warm-up routing, leave-one-expert-out composition. |
| `utils/moss_losses.py` | Gaussian MMD² (V- or U-statistic), normalized cross-covariance (linear CKA), distillation KL. |
| `utils/moss_memory.py` | Fit/val reservoirs of `(u, y, task)` features; task-uniform replay sampling. |
| `utils/wandb_logger.py` | Optional Weights & Biases logging (off unless `"wandb": true`). |
| `utils/dil_data_manager.py` | Domain-incremental data: DomainNet, ImageNet-R by rendition, generic domain folders, synthetic subset-sharing benchmark. |
| `utils/inc_net.py` (appended) | `MoSSNet` = frozen backbone + expert mixture; `get_frozen_backbone`. |
| `trainer.py`, `main.py`, `utils/factory.py` | `scenario` switch (`"cil"` / `"dil"`), `--set KEY=VALUE` overrides, `moss` registration. |
| `exps/moss/*.json`, `scripts/moss/*.sh` | Configs and launchers, including the alignment ablation. |
| `tests/test_moss.py` | Unit tests for the mathematical components. |

## Quick start

```bash
# Download the domain-incremental datasets (DomainNet, ImageNet-R split by rendition, Office-Home)
bash scripts/download_data.sh moss

# CPU smoke test on the synthetic benchmark (seconds)
bash scripts/moss/run_synthetic_dil.sh

# Domain-incremental benchmarks (GPU)
bash scripts/moss/run_domainnet_dil.sh
bash scripts/moss/run_imagenetr_dil.sh
bash scripts/moss/run_officehome_dil.sh

# The paper's alignment controls (subset / none / global / random), 3 seeds
bash scripts/moss/run_alignment_ablation.sh exps/moss/moss_domainnet_dil.json 1993 1994 1995

# Any config entry can be overridden from the command line (values parsed as JSON)
python3 main.py --config exps/moss/moss_domainnet_dil.json --set lambda_grow=1.0 eps_f=0.2

# Unit tests
python3 -m tests.test_moss
```

Each run writes `logs/moss/<prefix>/<time>/<dataset>/dil/` containing `summary.log`, the copied config, `latest_model.pth`, and `moss_diagnostics.json`.

## Dataset layouts

`scripts/download_data.sh` downloads and arranges every dataset below, as well as CaRE's class-incremental benchmarks. Run `bash scripts/download_data.sh list` to see the targets. Set `DATA_ROOT=/big/disk/dataset` to store data elsewhere; `<repo>/dataset` then becomes a symlink to it. Downloads resume, and finished datasets are skipped on re-runs. On an offline machine, put the archives in `dataset/_archives/` first and the script uses them.

Paths are relative to the repo root unless `data_path` is set.

- **`domainnet`**: `dataset/domainnet/<domain>_{train,test}.txt` (official split files; lines `<domain>/<class>/<file> <label>`), with images under the same root.
- **`imagenetr_dil`**: `dataset/imagenet-r-dil/{train,test}/<wnid>/<rendition>_<n>.jpg`, built by `bash scripts/download_data.sh imagenetr_dil` from the official ImageNet-R tar with a seeded 80/20 split per (class, rendition), so every rendition has train and test images. Each task is one rendition, parsed from the file name. All discovered renditions are used unless `domains` is set. This is separate from CaRE's `dataset/imagenet-r`, which uses the LAMDA-PILOT class-incremental split.
- **`folder_dil`**: `<data_path>/<domain>/{train,test}/<class>/*`, or `<data_path>/<domain>/<class>/*` with a seeded `test_frac` split. Only classes present in every domain are kept. This works for Office-Home, PACS, VLCS, and CORe50 sessions arranged as folders.
- **`synthetic_subset`**: generated in memory, with `backbone_type: "identity"`. Task `s` carries class signal only in its factor subset `syn_task_factors[s]`; other blocks are task-specific nuisance. The default schedule `[[0,1],[1,2],[0,2],[3],[1,3]]` mixes new combinations of familiar factors with a genuinely new factor, and the ground truth is saved in the diagnostics.

In any DIL dataset, a `domains` entry can be a list of names; those domains are merged into one task, e.g. `"domains": [["art","cartoon"],"sketch"]`.

## Paper → code

| Paper | Code |
|---|---|
| Eq. 2–7: frozen `b0`, experts, router `s_m(u)`, top-K softmax, shared head | `MoSSNet`, `ExpertMixture.routing / forward` |
| Eq. 9–12: cells, `ρ`, leave-one-out utility `U`, reference laws | `Learner._cell_stats`, `ExpertMixture.leave_one_out_logits` |
| Eq. 13–14: affinity gates, top-L supports, weights `w` | `Learner._estimate_supports` |
| Eq. 16: `L_ssi`, sampled by `w`, fixed examples per cell | `Learner._ssi_loss` |
| Eq. 17–18: `L_div` over co-active pairs with a trainable member | `cross_cov_div`, `Learner._loss` |
| Eq. 19–20: `F_t`, feasibility, best feasible checkpoint, input fallback | `Learner._checkpoint_stats`, `_run_phase` |
| Eq. 21: `J_t` | `_estimate_supports` → `sup["J"]` |
| Eq. 22: warm-up routing, zero router weight, K-th-score bias | `ExpertMixture.active_set`, `Learner._kth_score_bias` |
| Eq. 23–25: matched control/candidate branches, acceptance rule, ≤ G trials, `P_max` | `Learner._expansion_trials` |
| Eq. 26: `L_sup` with task-uniform replay | `MoSSMemory.sample_replay_indices`, `_loss` |
| Eq. 27: `L_pred` against `θ⁻` | `kd_kl`, `_loss` |
| Eq. 28: `L_feat` on updated old experts | `_loss` |
| Eq. 29 + phase restrictions | `_learn_task`, `_set_trainable` |
| Reservoir memory update | `Reservoir.add` (Algorithm R, separate fit/val reservoirs) |

## Key configuration entries

All are optional, with defaults in `models/moss.py: DEFAULTS`.

- **Architecture:** `num_experts_init` (M₀ ≥ 2), `expert_hidden_dim` (d_e), `expert_out_dim` (r), `topk_experts` (k ≥ 2), `router_tau` (τ).
- **Supports:** `ssi_mode`, `delta_u` (δ_u), `delta_d` (δ_d), `mmd_sigma` (σ), `mmd_estimator`, `ssi_max_hist_tasks` (L), `min_cell_size`, `ssi_max_cell_samples`, `ssi_triples_per_step`, `ssi_samples_per_cell`.
- **Loss weights:** `lambda_rep`, `lambda_pred`, `lambda_feat`, `lambda_ssi`, `lambda_div`.
- **Retention and expansion:** `eps_f` (ε_f), `p_max` or `p_max_factor`, `lambda_grow`, `max_expansion_trials` (G), `warmup_steps`.
- **Memory and data:** `memory_size` (B), `memory_val_frac` (B_val/B), `val_frac`.
- **Phase lengths:** `first_task_epochs`, `reuse_epochs`, `adapt_epochs`, `expand_epochs`, `eval_interval`.

`ssi_mode` selects the alignment controls described in the paper:

- `subset`: the proposed rule.
- `none`: identical supports and `J_t`, but no `L_ssi`. This isolates the alignment term from the adaptation it gates.
- `global`: every eligible triple, uniform weights, all old experts adapted.
- `random`: the same number of triples as `subset`, drawn uniformly from the eligible ones.

## Diagnostics

`moss_diagnostics.json` records the following per task:

- **Supports:** eligible current and historical cells (vs. possible), candidate triples, pairs passing the utility gate, `|I_t|`, `J_t`, MMD quantiles (useful for setting `delta_d`), and support counts per expert.
- **Per phase:** checkpoints evaluated, how many were feasible, whether the phase fell back to its input, the selected `val_risk` and `F_t`, and mean loss terms.
- **Expansion trials:** `delta`, `threshold`, accept/reject, and the reason for stopping.
- **Final state:** the fixed σ, the expert count and `P_E` after the task, and per-task accuracies.

## Experiment tracking with Weights & Biases

Tracking is off by default. To use it:

```bash
python -m pip install wandb && wandb login        # once

bash scripts/moss/run_officehome_dil.sh --set wandb=true
bash scripts/moss/run_officehome_dil.sh --set wandb=true wandb_project='"closs"' wandb_entity='"your-team"'
WANDB=1 bash scripts/moss/run_alignment_ablation.sh exps/moss/moss_domainnet_dil.json   # grouped by mode

# no internet on the training machine: log offline, upload later
bash scripts/moss/run_synthetic_dil.sh --set wandb=true wandb_mode='"offline"'
wandb sync logs/.../wandb/offline-run-*
```

It also works for CaRE runs, e.g. `python main.py --config exps/imagenet_r/care_inr_inc20.json --set wandb=true`, because the per-task accuracy logging lives in `trainer.py`.

Each seed is one wandb run, named `<prefix>-s<seed>` and grouped by `prefix`. The ablation script groups by alignment mode instead, so seeds average together. What gets logged:

- **The full config**, including every CLoSS hyperparameter.
- **Accuracy per task** (x-axis `task`): `acc/top1`, `acc/avg_top1`, `acc/old`, `acc/new`, top-5 accuracy, accuracy on each task group (`acc_task/*`), running `forgetting/top1`, and the total parameter count.
- **CLoSS state per task:** expert count, experts added so far, `P_E` and its fraction of `P_max`, expansion trials and acceptances with the validation gain and threshold, support sizes (`I_t`, `J_t`, eligible historical cells, median MMD), and for each phase its fallback flag, fraction of feasible checkpoints, validation risk, `F_t`, and mean losses.
- **Training losses** every `wandb_log_interval` steps (default 50; x-axis `train_step`), per phase and loss term.
- **At the end:** the accuracy matrix as a table, `final/avg_top1`, `final/last_top1`, and `final/forgetting` in the run summary, plus the run's `summary.log`, config, and `moss_diagnostics.json` as files. Set `wandb_save_model=true` to also upload `latest_model.pth` (large for ViT models).

A run that crashes is closed with exit code 1, so it shows as failed in wandb. Other keys: `wandb_group`, `wandb_name`, `wandb_tags`, `wandb_mode` (`online`, `offline`, `disabled`).

## Implementation choices where the paper is not specific

1. **Feature caching.** Features are extracted once per task with test-time transforms, since `b0` is frozen. There is no augmentation in feature space.
2. **σ = `"auto"`.** The bandwidth is set by the median pairwise distance of expert outputs on task-0 probes, then fixed for the rest of the sequence. A float value fixes it from the start.
3. **Cell subsampling for MMD.** Support estimation caps each cell at `ssi_max_cell_samples` records. Set it to `-1` to use every record, as the paper describes.
4. **Checkpoint cadence and fallback.** Checkpoints are evaluated once per epoch by default. Following the text literally, a phase's input checkpoint is used only as a fallback when no checkpoint of the phase is feasible.
5. **`L_ssi` reference features.** These come from the frozen `θ⁻` experts. They are identical to the reuse snapshot's experts, because reuse freezes all experts.
6. **Dense expert evaluation.** All experts are evaluated on every input, and inactive experts get zero weight. This is mathematically identical to sparse evaluation, but it does not realize the compute saving of top-k routing.
7. **Reported accuracy.** In DIL, `top1` is the task-averaged accuracy (Eq. 1). In CIL it is sample-weighted, matching the repo's other learners.

## Class-incremental runs

In the repo's class-incremental splits, no class appears in two tasks, so no class-conditional cross-task cell exists. As a result, `I_t` is always empty, `L_ssi` never activates, and old experts are never adapted. The learner logs a warning when this happens. The run is still a valid reference point (replay, distillation, expansion), but it does not test subset-shared invariance. For CIL runs, `F_t` and `L_pred` are computed on the previously seen class subspace, because `θ⁻` cannot predict new classes.

## Things to watch

- **The retention constraint can bind.** `F_t` is always measured against the pre-task model `θ⁻` and is never reset within a task, so the allowance can run out. On the synthetic benchmark with `eps_f=0.15`, tasks 3–4 had 0–1 feasible adaptation checkpoints and the task-4 expansion control fell back to its input. Track `feasible_checkpoints` and `fallback_to_input` in the diagnostics, and sweep `eps_f`.
- **Memory is counted in records, not bytes.** A 768-d float feature is about 3 KB, far smaller than an image. Compare against image-replay baselines at matched bytes.
- **Small cells weaken support estimation.** The default MMD is the paper's V-statistic, which is biased upward for small cells. On long or many-class sequences, check `hist_cells_eligible` in the diagnostics; `mmd_estimator: "u"` is available.
- **Expansion is expensive.** Each trial trains two branches, so one task costs up to `1 + 1 + 2G` training phases.
