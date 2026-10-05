"""
MoSS: Mixture of Specialized Skills with subset-shared invariance for continual learning.

Implements Sec. 3 of the paper on top of the CaRE / PILOT training loop:

  3.2  Frozen backbone b0, expert bank, linear top-k router, shared head        (backbone/moss_moe.py)
  3.3  Expert-specific task supports (rho, U, MMD gate, top-L) + L_ssi, L_div    (_estimate_supports, _loss)
  3.4  Reuse -> selective adaptation -> matched expansion trials                  (_learn_task, _expansion_trials)
  3.5  L_sup with task-uniform replay, L_pred, L_feat, reservoir memory update   (_loss, utils/moss_memory.py)

Scenarios
  "dil"  domain-incremental (the paper's setting): one label space, tasks are domains.
  "cil"  class-incremental (the repo's benchmarks). No class appears in two tasks, so no
         class-conditional cross-task cell exists and the support set I_t is always empty: L_ssi is
         inactive and old experts are never adapted. The learner still runs (replay, retention,
         expansion), which is useful as a reference point, but it does not test subset-shared invariance.

Alignment controls (Sec. 1 / evaluation design), selected with "ssi_mode":
  "subset"  the proposed rule (Eq. 13-14).
  "none"    identical supports and J_t, but lambda_ssi is not applied (isolates the alignment term).
  "global"  every eligible (m, s, c) triple, uniform weights; J_t = all old experts.
  "random"  as many triples as "subset" would select, drawn uniformly from the eligible triples.
"""
import copy
import json
import logging
import math
import os
import re
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.base import BaseLearner
from utils.inc_net import MoSSNet
from utils.moss_losses import cross_cov_div, kd_kl, mmd2, pairwise_sq_dists
from utils.moss_memory import MoSSMemory
from utils import wandb_logger

DEFAULTS = dict(
    scenario="cil",
    # data / memory (Sec. 3.1)
    val_frac=0.1,                 # validation portion of the current task
    memory_size=2000,             # B = B_tr + B_val
    memory_val_frac=0.2,          # B_val / B
    extract_batch_size=128,
    num_workers=4,
    # optimization
    optimizer="adam",
    lr=1e-3,
    weight_decay=0.0,
    batch_size=64,
    replay_batch_size=64,
    first_task_epochs=10,
    reuse_epochs=5,
    adapt_epochs=5,
    expand_epochs=5,
    eval_interval=0,              # steps between checkpoint evaluations; 0 = once per epoch
    # loss weights (Eq. 29), fixed across tasks
    lambda_rep=1.0,
    lambda_pred=1.0,
    lambda_feat=1.0,
    lambda_ssi=1.0,
    lambda_div=0.1,
    # supports and alignment (Sec. 3.3)
    ssi_mode="subset",
    delta_u=0.0,
    delta_d=0.3,
    mmd_sigma="auto",             # float, or "auto": median heuristic on task-0 expert outputs, then frozen
    mmd_estimator="v",            # "v" = paper (all pairs incl. diagonal); "u" = unbiased
    ssi_max_hist_tasks=3,         # L
    min_cell_size=2,
    ssi_max_cell_samples=256,     # cap per cell for support estimation; -1 = use every record
    ssi_triples_per_step=4,
    ssi_samples_per_cell=32,
    # retention and checkpoints (Sec. 3.4)
    eps_f=0.1,
    # expansion (Sec. 3.4)
    p_max=None,                   # absolute P_max; if None, p_max_factor * P_E(initial bank)
    p_max_factor=4.0,
    lambda_grow=0.5,
    max_expansion_trials=2,       # G
    warmup_steps=50,
    probe_size=2048,
    # experiment tracking (only used when the config sets "wandb": true)
    wandb_log_interval=50,        # log training losses every N optimizer steps; 0 = off
)


class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.hp = {k: args.get(k, v) for k, v in DEFAULTS.items()}
        self.scenario = str(self.hp["scenario"]).lower()
        assert self.scenario in ("dil", "cil"), f"Unknown scenario {self.scenario}"
        assert self.hp["ssi_mode"] in ("subset", "none", "global", "random")
        seed = args["seed"] if isinstance(args["seed"], int) else int(args["seed"][0])

        self._network = MoSSNet(args).to(self._device)
        moe = self._network.moe
        self.memory = MoSSMemory(self.hp["memory_size"], self.hp["memory_val_frac"], seed=seed)
        p_init = moe.stored_expert_router_params()
        self.p_max = int(self.hp["p_max"]) if self.hp["p_max"] else int(self.hp["p_max_factor"] * p_init)
        if self.p_max < p_init:
            raise ValueError(f"P_max={self.p_max} must accommodate the initial bank ({p_init}).")
        self.sigma = None if self.hp["mmd_sigma"] == "auto" else float(self.hp["mmd_sigma"])

        self.np_rng = np.random.default_rng(seed + 17)
        self._test_cache = OrderedDict()
        self._teacher = None
        self._supports = None
        self._n_old_experts = moe.num_experts
        self.diagnostics = []
        self._cil_warned = False
        self._train_step = 0
        self._experts_added = 0

        logging.info(f"[MoSS] scenario={self.scenario}, ssi_mode={self.hp['ssi_mode']}, "
                     f"feature_dim={self._network.feature_dim}, M0={moe.num_experts}, k={moe.topk}, "
                     f"P_E(initial)={p_init:,}, P_max={self.p_max:,}")

    # ============================================================================ task loop
    def after_task(self):
        self._known_classes = self._total_classes
        self._dump_diagnostics()  # after eval_task, so per-task accuracies are included

    def incremental_train(self, data_manager):
        self._cur_task += 1
        t = self._cur_task
        self.data_manager = data_manager

        if self.scenario == "dil":
            self._total_classes = data_manager.nb_classes
            train_ds = data_manager.get_task_dataset(t, source="train", mode="test")
            test_ds = data_manager.get_task_dataset(t, source="test", mode="test")
            logging.info(f"[MoSS] Task {t}: domain '{data_manager.domain_names[t]}'")
        else:
            self._total_classes = self._known_classes + data_manager.get_task_size(t)
            cls = np.arange(self._known_classes, self._total_classes)
            train_ds = data_manager.get_dataset(cls, source="train", mode="test")
            test_ds = data_manager.get_dataset(cls, source="test", mode="test")
            logging.info(f"[MoSS] Task {t}: classes {self._known_classes}-{self._total_classes}")

        # The backbone is frozen, so features are extracted once per task (test-time transforms).
        u, y = self._extract(train_ds)
        self._test_cache[t] = self._extract(test_ds)
        fit_idx, val_idx = self._stratified_split(y)
        self.cur = {"u_fit": u[fit_idx], "y_fit": y[fit_idx], "u_val": u[val_idx], "y_val": y[val_idx]}
        logging.info(f"[MoSS] fit/val = {len(fit_idx)}/{len(val_idx)}, memory fit/val = "
                     f"{len(self.memory.fit)}/{len(self.memory.val)}")

        self._learn_task(t)

        self.memory.update(self.cur["u_fit"], self.cur["y_fit"], self.cur["u_val"], self.cur["y_val"], t)

    # ---------------------------------------------------------------------------- data helpers
    @torch.no_grad()
    def _extract(self, dataset):
        loader = DataLoader(dataset, batch_size=self.hp["extract_batch_size"], shuffle=False,
                            num_workers=self.hp["num_workers"])
        self._network.eval()
        feats, ys = [], []
        for _, x, y in loader:
            feats.append(self._network.backbone_features(x.to(self._device)).float().cpu())
            ys.append(torch.as_tensor(y).long())
        return torch.cat(feats), torch.cat(ys)

    def _stratified_split(self, y):
        frac = float(self.hp["val_frac"])
        fit, val = [], []
        for c in torch.unique(y).tolist():
            idx = torch.nonzero(y == c, as_tuple=True)[0].numpy()
            idx = self.np_rng.permutation(idx)
            n_val = int(round(frac * len(idx))) if len(idx) >= 2 else 0
            n_val = max(1, n_val) if len(idx) >= 2 and frac > 0 else n_val
            val.append(idx[:n_val])
            fit.append(idx[n_val:])
        fit, val = np.concatenate(fit), np.concatenate(val)
        if len(val) == 0:
            raise ValueError("The current-task validation portion is empty; increase val_frac.")
        return torch.as_tensor(fit), torch.as_tensor(val)

    @property
    def _n_teacher_classes(self):
        # The pre-task model's class space: all classes in DIL, previously seen classes in CIL.
        return self._total_classes if self.scenario == "dil" else self._known_classes

    # ============================================================================ one task
    def _learn_task(self, t):
        hp = self.hp
        moe = self._network.moe.to(self._device)
        self._n_old_experts = moe.num_experts
        diag = {"task": t, "experts_before": moe.num_experts}

        # theta^- : frozen pre-task model; the retention reference for every phase and trial.
        self._teacher, self._ref_hist_risk, self._supports = None, {}, None
        if t > 0:
            self._teacher = copy.deepcopy(moe).eval()
            for p in self._teacher.parameters():
                p.requires_grad = False
            self._ref_hist_risk = self._hist_risks(self._teacher)

        if t == 0:
            # Initial bank, router and head trained jointly: supervision + expert differentiation.
            diag["first"] = self._run_phase(moe, "first", list(range(moe.num_experts)),
                                            epochs=hp["first_task_epochs"], use_div=True)
        else:
            # (1) Reuse: experts frozen; router + head with L_sup + lambda_pred L_pred.
            diag["reuse"] = self._run_phase(moe, "reuse", [], epochs=hp["reuse_epochs"])
            # (2) Supports from the reuse snapshot theta-bar (fitting data only).
            sup, sup_diag = self._estimate_supports(t)
            self._supports = sup
            diag["supports"] = sup_diag
            # (3) Selective adaptation of J_t with the complete objective.
            use_ssi = hp["ssi_mode"] != "none" and hp["lambda_ssi"] > 0 and len(sup["triples"]) > 0
            diag["adapt"] = self._run_phase(moe, "adapt", sup["J"], epochs=hp["adapt_epochs"],
                                            use_div=True, use_feat=True, use_ssi=use_ssi)
            self._supports = None

        # (4) Expansion trials (also on the first task).
        diag["expansion"] = self._expansion_trials(t)
        if self.sigma is None:
            self.sigma = self._median_sigma()
            logging.info(f"[MoSS] MMD bandwidth fixed to sigma={self.sigma:.4f} (median heuristic)")
        diag["sigma"] = self.sigma
        diag["experts_after"] = self._network.moe.num_experts
        diag["P_E"] = self._network.moe.stored_expert_router_params()
        self.diagnostics.append(diag)
        self._log_task_wandb(t, diag)
        logging.info(f"[MoSS] Task {t} done: experts {diag['experts_before']} -> {diag['experts_after']}, "
                     f"P_E={diag['P_E']:,}/{self.p_max:,}")

    # ============================================================================ phases
    def _set_trainable(self, moe, trainable_experts):
        for p in moe.parameters():
            p.requires_grad = False
        for p in moe.routers.parameters():
            p.requires_grad = True
        for p in moe.head.parameters():
            p.requires_grad = True
        for m in trainable_experts:
            for p in moe.experts[m].parameters():
                p.requires_grad = True

    def _make_optimizer(self, params):
        hp = self.hp
        if hp["optimizer"] == "sgd":
            return optim.SGD(params, lr=hp["lr"], momentum=0.9, weight_decay=hp["weight_decay"])
        if hp["optimizer"] == "adamw":
            return optim.AdamW(params, lr=hp["lr"], weight_decay=hp["weight_decay"])
        return optim.Adam(params, lr=hp["lr"], weight_decay=hp["weight_decay"])

    def _make_plan(self, epochs):
        """Pre-generated (current minibatch, replay minibatch) indices, so the two expansion branches
        see identical fitting and replay minibatches."""
        n = len(self.cur["y_fit"])
        bs = int(self.hp["batch_size"])
        g = torch.Generator().manual_seed(int(self.np_rng.integers(1 << 31)))
        plan = []
        for _ in range(int(epochs)):
            perm = torch.randperm(n, generator=g)
            for i in range(0, n, bs):
                ci = perm[i:i + bs]
                if len(ci) < 2 <= n:
                    continue
                ri = self.memory.sample_replay_indices(int(self.hp["replay_batch_size"]), g)
                plan.append((ci, ri))
        return plan

    def _run_phase(self, moe, name, trainable_experts, epochs=None, plan=None, use_div=False,
                   use_feat=False, use_ssi=False, warmup_steps=0, forced_expert=None,
                   allow_input_fallback=True):
        """Train the given parameter subset and return to the best feasible checkpoint
        (Eq. 20: F_t <= eps_f and P_E <= P_max; lowest risk on D_t^val). If no checkpoint of the
        phase is feasible, the phase's input checkpoint is restored (when allowed)."""
        hp = self.hp
        self._set_trainable(moe, trainable_experts)
        opt = self._make_optimizer([p for p in moe.parameters() if p.requires_grad])
        plan = self._make_plan(epochs) if plan is None else plan
        V = [m for m in trainable_experts if m < self._n_old_experts] if self._teacher is not None else []
        input_state = copy.deepcopy(moe.state_dict()) if allow_input_fallback else None

        steps_per_epoch = max(1, math.ceil(len(self.cur["y_fit"]) / hp["batch_size"]))
        interval = int(hp["eval_interval"]) or steps_per_epoch
        eval_points = {i for i in range(len(plan)) if (i + 1) % interval == 0} | {len(plan) - 1}

        best_risk, best_state, best_f, n_eval, n_feasible = math.inf, None, None, 0, 0
        sums = {}
        prog = tqdm(range(len(plan)), desc=f"[MoSS] {name}", leave=False)
        for step in prog:
            moe.train()
            moe.forced_expert = forced_expert if (forced_expert is not None and step < warmup_steps) else None
            ci, ri = plan[step]
            loss, parts = self._loss(moe, ci, ri, trainable_experts, V, use_div, use_feat, use_ssi)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            for k, v in parts.items():
                sums[k] = sums.get(k, 0.0) + v
            self._train_step += 1
            interval = int(hp["wandb_log_interval"])
            if wandb_logger.active() and interval > 0 and self._train_step % interval == 0:
                kind = re.sub(r"\d+", "", name)  # expand0-candidate -> expand-candidate
                wandb_logger.log({"train_step": self._train_step, "train/task": self._cur_task,
                                  **{f"train/{kind}/{k}": v for k, v in parts.items()},
                                  f"train/{kind}/total": loss.item()})
            if step in eval_points and step >= warmup_steps:
                moe.forced_expert = None
                risk, f_t, p_e = self._checkpoint_stats(moe)
                n_eval += 1
                feasible = (f_t <= hp["eps_f"]) and (p_e <= self.p_max)
                n_feasible += int(feasible)
                if feasible and risk < best_risk:
                    best_risk, best_state, best_f = risk, copy.deepcopy(moe.state_dict()), f_t
                prog.set_description(f"[MoSS] {name} val={risk:.3f} F={f_t:.3f}")
        moe.forced_expert = None

        fallback = best_state is None
        if not fallback:
            moe.load_state_dict(best_state)
        elif input_state is not None:
            moe.load_state_dict(input_state)
        for p in moe.parameters():
            p.requires_grad = False
        moe.eval()

        info = {
            "steps": len(plan),
            "trainable_experts": list(map(int, trainable_experts)),
            "checkpoints": n_eval,
            "feasible_checkpoints": n_feasible,
            "fallback_to_input": bool(fallback),
            "val_risk": None if fallback else float(best_risk),
            "F_t": None if fallback else float(best_f),
            "mean_losses": {k: v / max(1, len(plan)) for k, v in sums.items()},
        }
        logging.info(f"[MoSS] {name}: {json.dumps(info)}")
        return info

    # ============================================================================ objective
    def _loss(self, moe, ci, ri, trainable, V, use_div, use_feat, use_ssi):
        """L = L_sup + lambda_pred L_pred + lambda_feat L_feat + lambda_ssi L_ssi + lambda_div L_div."""
        hp, dev = self.hp, self._device
        n_out = self._total_classes
        parts = {}

        u_c = self.cur["u_fit"][ci].to(dev)
        y_c = self.cur["y_fit"][ci].to(dev)
        out_c = moe(u_c)
        loss = F.cross_entropy(out_c["logits"][:, :n_out], y_c)
        parts["cur"] = loss.item()
        outs = [out_c]

        if ri is not None:
            mu, my, _ = self.memory.fit.data()
            u_r, y_r = mu[ri].to(dev), my[ri].to(dev)
            out_r = moe(u_r)
            outs.append(out_r)
            l_rep = F.cross_entropy(out_r["logits"][:, :n_out], y_r)
            loss = loss + hp["lambda_rep"] * l_rep
            parts["rep"] = l_rep.item()
            if self._teacher is not None:
                with torch.no_grad():
                    t_out = self._teacher(u_r)
                nt = self._n_teacher_classes
                if hp["lambda_pred"] > 0 and nt > 0:
                    l_pred = kd_kl(out_r["logits"][:, :nt], t_out["logits"][:, :nt])
                    loss = loss + hp["lambda_pred"] * l_pred
                    parts["pred"] = l_pred.item()
                if use_feat and len(V) > 0 and hp["lambda_feat"] > 0:
                    cur_h = out_r["expert_out"][:, V, :]
                    ref_h = t_out["expert_out"][:, V, :]
                    r = cur_h.shape[-1]
                    l_feat = (cur_h - ref_h).pow(2).sum(dim=(1, 2)).mean() / (r * max(1, len(V)))
                    loss = loss + hp["lambda_feat"] * l_feat
                    parts["feat"] = l_feat.item()

        if use_div and hp["lambda_div"] > 0 and len(trainable) > 0:
            zs = torch.cat([o["expert_out"] for o in outs], dim=0)
            act = torch.cat([o["active"] for o in outs], dim=0).float()
            co = (act.t() @ act) > 0
            tr = set(int(m) for m in trainable)
            M = zs.shape[1]
            pairs = [(m, n) for m in range(M) for n in range(m + 1, M)
                     if bool(co[m, n]) and (m in tr or n in tr)]
            if pairs:
                l_div = cross_cov_div(zs, pairs)
                loss = loss + hp["lambda_div"] * l_div
                parts["div"] = l_div.item()

        if use_ssi:
            l_ssi = self._ssi_loss(moe)
            loss = loss + hp["lambda_ssi"] * l_ssi
            parts["ssi"] = l_ssi.item()

        return loss, parts

    def _ssi_loss(self, moe):
        """Stochastic L_ssi (Eq. 16): triples sampled according to w^t, a fixed number of examples
        per cell; current features through the trainable expert, historical reference features
        through the frozen reference expert (h-bar_m = h^-_m, because reuse froze the experts)."""
        hp, dev, S = self.hp, self._device, self._supports
        draws = self.np_rng.choice(len(S["triples"]), size=int(hp["ssi_triples_per_step"]), p=S["w"])
        mu, _, _ = self.memory.fit.data()
        n = int(hp["ssi_samples_per_cell"])
        vals = []
        for k in draws:
            m, s, c = S["triples"][k]
            ci = self._subsample(S["cur_cells"][c], n)
            hi = self._subsample(S["hist_cells"][(s, c)], n)
            x = moe.experts[m](self.cur["u_fit"][ci].to(dev))
            with torch.no_grad():
                ref = self._teacher.experts[m](mu[hi].to(dev))
            vals.append(mmd2(x, ref, self.sigma, self.hp["mmd_estimator"]))
        return torch.stack(vals).mean()

    def _subsample(self, idx, n):
        if n <= 0 or len(idx) <= n:
            return idx
        return idx[torch.as_tensor(self.np_rng.choice(len(idx), size=n, replace=False))]

    # ============================================================================ supports
    @torch.no_grad()
    def _cell_stats(self, moe, u, c):
        """rho_{m,cell} (Eq. 10), U_{m,cell} (Eq. 11), and reference features h-bar_m(u) (Eq. 12)."""
        n_out = self._total_classes
        rho, util, feats = 0.0, 0.0, []
        for i in range(0, len(u), 1024):
            out = moe(u[i:i + 1024].to(self._device))
            logp = F.log_softmax(out["logits"][:, :n_out], dim=-1)[:, c]
            loo = moe.leave_one_out_logits(out["z"], out["weights"], out["expert_out"])[:, :, :n_out]
            logp_minus = F.log_softmax(loo, dim=-1)[:, :, c]
            rho = rho + out["weights"].sum(0)
            util = util + (logp.unsqueeze(1) - logp_minus).sum(0)
            feats.append(out["expert_out"])
        feats = torch.cat(feats)
        cap = int(self.hp["ssi_max_cell_samples"])
        if 0 < cap < len(feats):
            feats = feats[torch.as_tensor(self.np_rng.choice(len(feats), size=cap, replace=False))]
        return rho / len(u), util / len(u), feats

    @torch.no_grad()
    def _estimate_supports(self, t):
        hp = self.hp
        moe = self._network.moe.eval()
        M = moe.num_experts
        min_n = int(hp["min_cell_size"])
        y_fit = self.cur["y_fit"]

        cur_cells = {}
        for c in torch.unique(y_fit).tolist():
            idx = torch.nonzero(y_fit == c, as_tuple=True)[0]
            if len(idx) >= min_n:
                cur_cells[int(c)] = idx
        all_hist = self.memory.fit.indices_by_cell()
        hist_cells = {k: v for k, v in all_hist.items() if len(v) >= min_n and k[0] < t}

        mu, _, _ = self.memory.fit.data()
        st_cur = {c: self._cell_stats(moe, self.cur["u_fit"][idx], c) for c, idx in cur_cells.items()}
        st_hist = {k: self._cell_stats(moe, mu[idx], k[1]) for k, idx in hist_cells.items()}

        candidates, subset, d_vals = [], [], []
        n_pass_u = 0
        for c, (rho_t, U_t, f_t) in st_cur.items():
            hist_s = sorted(s for (s, cc) in st_hist if cc == c)
            for m in range(M):
                cands = []
                for s in hist_s:
                    candidates.append((m, s, c))
                    rho_s, U_s, f_s = st_hist[(s, c)]
                    if not (U_s[m] > hp["delta_u"] and U_t[m] > hp["delta_u"]):
                        continue
                    n_pass_u += 1
                    d = float(mmd2(f_s[:, m], f_t[:, m], self.sigma, hp["mmd_estimator"]))
                    d_vals.append(d)
                    if d > hp["delta_d"]:
                        continue
                    a = float(rho_s[m] * rho_t[m])
                    if a > 0:
                        cands.append((a, s, d))
                cands.sort(key=lambda z: (-z[0], z[1]))  # largest affinity, ties by task index
                subset += [(m, s, c, a, d) for a, s, d in cands[: int(hp["ssi_max_hist_tasks"])]]

        mode = hp["ssi_mode"]
        if mode in ("subset", "none"):
            triples = [(m, s, c) for m, s, c, _, _ in subset]
            a = np.array([z[3] for z in subset], dtype=np.float64)
            w = a / a.sum() if len(a) else a
        elif mode == "global":
            triples = list(candidates)
            w = np.full(len(triples), 1.0 / max(1, len(triples)))
        else:  # random, matched to the size of the subset rule
            k = min(len(subset), len(candidates))
            pick = self.np_rng.choice(len(candidates), size=k, replace=False) if k > 0 else []
            triples = [candidates[i] for i in pick]
            w = np.full(len(triples), 1.0 / max(1, len(triples)))

        if mode == "global":
            J = list(range(M))
        else:
            J = sorted({m for m, _, _ in triples})

        if self.scenario == "cil" and not self._cil_warned and not candidates:
            logging.warning("[MoSS] Class-incremental split: no class occurs in two tasks, so no "
                            "class-conditional cross-task cell exists. I_t is empty, L_ssi is inactive "
                            "and old experts are not adapted. Use scenario 'dil' to test subset-shared "
                            "invariance.")
            self._cil_warned = True

        n_hist_tasks = len({s for (s, _) in all_hist})
        diag = {
            "mode": mode,
            "current_cells_eligible": len(cur_cells),
            "hist_cells_stored": len(all_hist),
            "hist_cells_eligible": len(hist_cells),
            "hist_cells_possible": n_hist_tasks * self._total_classes if self.scenario == "dil" else 0,
            "candidate_triples": len(candidates),
            "pairs_passing_utility": n_pass_u,
            "subset_triples": len(subset),
            "I_t": len(triples),
            "J_t": list(map(int, J)),
            "mmd_quantiles": (np.quantile(d_vals, [0.1, 0.5, 0.9]).tolist() if d_vals else None),
            "affinity_by_expert": {int(m): int(sum(1 for z in triples if z[0] == m)) for m in range(M)},
        }
        logging.info(f"[MoSS] supports: {json.dumps(diag)}")
        sup = {"triples": triples, "w": w, "J": J, "cur_cells": cur_cells, "hist_cells": hist_cells}
        return sup, diag

    # ============================================================================ expansion
    @torch.no_grad()
    def _probes(self):
        n = len(self.cur["y_fit"])
        k = min(n, int(self.hp["probe_size"]))
        idx = torch.as_tensor(self.np_rng.choice(n, size=k, replace=False))
        return self.cur["u_fit"][idx].to(self._device)

    @torch.no_grad()
    def _kth_score_bias(self, moe):
        """Mean min(k, M)-th largest existing router score on current fitting probes."""
        scores = moe.router_scores(self._probes())
        K = min(moe.topk, moe.num_experts)
        return float(scores.topk(K, dim=-1).values[:, -1].mean())

    def _expansion_trials(self, t):
        hp = self.hp
        trials = []
        for g in range(int(hp["max_expansion_trials"])):
            cur = self._network.moe
            if cur.stored_expert_router_params() + cur.expert_param_count() > self.p_max:
                trials.append({"trial": g, "stopped": "P_max"})
                break
            plan = self._make_plan(hp["expand_epochs"])
            if hp["warmup_steps"] >= len(plan):
                logging.warning(f"[MoSS] warmup_steps={hp['warmup_steps']} >= trial length {len(plan)}: "
                                "no post-warm-up candidate checkpoint can exist.")

            # Control branch: existing bank (all experts frozen), router + head; fallback = theta^cur.
            ctrl = copy.deepcopy(cur)
            ctrl_info = self._run_phase(ctrl, f"expand{g}-control", [], plan=plan)
            # Candidate branch: one new expert + router score; previously accepted experts frozen.
            cand = copy.deepcopy(cur)
            m_new = cand.add_expert(router_bias=self._kth_score_bias(cand))
            cand_info = self._run_phase(cand, f"expand{g}-candidate", [m_new], plan=plan, use_div=True,
                                        warmup_steps=int(hp["warmup_steps"]), forced_expert=m_new,
                                        allow_input_fallback=False)
            rec = {"trial": g, "control": ctrl_info, "candidate": cand_info}
            if cand_info["fallback_to_input"]:
                rec.update(accepted=False, reason="no feasible candidate checkpoint")
                self._network.moe = ctrl
                trials.append(rec)
                break
            r_ctrl = self._risk(ctrl, self.cur["u_val"], self.cur["y_val"], self._total_classes)
            r_cand = self._risk(cand, self.cur["u_val"], self.cur["y_val"], self._total_classes)
            delta = r_ctrl - r_cand
            dP = cand.stored_expert_router_params() - ctrl.stored_expert_router_params()
            thr = hp["lambda_grow"] * dP / self.p_max
            accepted = delta > thr
            rec.update(accepted=bool(accepted), delta=float(delta), threshold=float(thr), delta_P=int(dP))
            logging.info(f"[MoSS] expansion trial {g}: delta={delta:.4f} vs threshold={thr:.4f} -> "
                         f"{'ACCEPT' if accepted else 'reject'}")
            trials.append(rec)
            if accepted:
                self._network.moe = cand
            else:
                self._network.moe = ctrl
                break
        return trials

    # ============================================================================ risks
    @torch.no_grad()
    def _risk(self, moe, u, y, n_cls):
        moe.eval()
        tot, n = 0.0, 0
        for i in range(0, len(y), 2048):
            logits = moe(u[i:i + 2048].to(self._device))["logits"][:, :n_cls]
            tot += F.cross_entropy(logits, y[i:i + 2048].to(self._device), reduction="sum").item()
            n += len(y[i:i + 2048])
        return tot / max(1, n)

    @torch.no_grad()
    def _hist_risks(self, moe):
        u, y, s = self.memory.val.data()
        if u is None or self._n_teacher_classes == 0:
            return {}
        return {int(k): self._risk(moe, u[s == k], y[s == k], self._n_teacher_classes)
                for k in torch.unique(s).tolist()}

    @torch.no_grad()
    def _checkpoint_stats(self, moe):
        risk = self._risk(moe, self.cur["u_val"], self.cur["y_val"], self._total_classes)
        f_t = 0.0  # Eq. 19; 0 when historical validation memory is empty
        if self._ref_hist_risk:
            cur = self._hist_risks(moe)
            f_t = max(max(cur[s] - self._ref_hist_risk[s], 0.0) for s in self._ref_hist_risk)
        return risk, f_t, moe.stored_expert_router_params()

    @torch.no_grad()
    def _median_sigma(self):
        moe = self._network.moe.eval()
        zs = moe.expert_outputs(self._probes()[:512])
        meds = []
        for m in range(zs.shape[1]):
            d = pairwise_sq_dists(zs[:, m], zs[:, m]).sqrt()
            iu = torch.triu_indices(d.shape[0], d.shape[0], offset=1)
            meds.append(d[iu[0], iu[1]].median())
        return float(torch.stack(meds).median().clamp_min(1e-6))

    # ============================================================================ evaluation
    @torch.no_grad()
    def eval_task(self):
        moe = self._network.moe.eval()
        n_out = self._total_classes
        grouped = OrderedDict()
        accs, topk_hits, n_samples, n_correct = [], 0, 0, 0
        for t, (u, y) in self._test_cache.items():
            correct, hits = 0, 0
            for i in range(0, len(y), 2048):
                logits = moe(u[i:i + 2048].to(self._device))["logits"][:, :n_out]
                yy = y[i:i + 2048].to(self._device)
                k = min(self.topk, n_out)
                top = logits.topk(k, dim=-1).indices
                correct += (top[:, 0] == yy).sum().item()
                hits += (top == yy.unsqueeze(1)).any(dim=1).sum().item()
            acc = 100.0 * correct / max(1, len(y))
            grouped[f"task-{t:02d}"] = round(acc, 2)
            accs.append(acc)
            topk_hits += hits
            n_samples += len(y)
            n_correct += correct
        if self.scenario == "dil":
            total = float(np.mean(accs))  # each task weighted equally (Eq. 1)
        else:
            total = 100.0 * n_correct / max(1, n_samples)  # sample-weighted, as in the CIL baselines
        grouped["total"] = round(total, 2)
        grouped["old"] = round(float(np.mean(accs[:-1])), 2) if len(accs) > 1 else 0.0
        grouped["new"] = round(accs[-1], 2)
        ret = {"grouped": grouped, "top1": round(total, 2),
               f"top{self.topk}": round(100.0 * topk_hits / max(1, n_samples), 2)}
        if self.diagnostics:
            self.diagnostics[-1]["eval"] = {k: float(v) for k, v in grouped.items()}
        logging.info(f"[MoSS] eval: {dict(grouped)}")
        return ret, None

    def _log_task_wandb(self, t, diag):
        """CLoSS-specific per-task metrics: capacity, expansion decisions, supports, phase health."""
        if not wandb_logger.active():
            return
        trials = [e for e in diag["expansion"] if "accepted" in e]
        accepted = sum(1 for e in trials if e["accepted"])
        self._experts_added += accepted
        data = {
            "task": t,
            "moss/experts": diag["experts_after"],
            "moss/experts_added_total": self._experts_added,
            "moss/P_E": diag["P_E"],
            "moss/P_E_frac_of_max": diag["P_E"] / self.p_max,
            "moss/expansion/trials": len(trials),
            "moss/expansion/accepted": accepted,
            "moss/sigma": diag["sigma"],
        }
        if trials:
            data["moss/expansion/first_delta"] = trials[0].get("delta")
            data["moss/expansion/threshold"] = trials[0].get("threshold")
        sup = diag.get("supports")
        if sup:
            data["moss/supports/I_t"] = sup["I_t"]
            data["moss/supports/J_t"] = len(sup["J_t"])
            data["moss/supports/candidate_triples"] = sup["candidate_triples"]
            data["moss/supports/hist_cells_eligible"] = sup["hist_cells_eligible"]
            if sup["hist_cells_possible"]:
                data["moss/supports/hist_cells_eligible_frac"] = sup["hist_cells_eligible"] / sup["hist_cells_possible"]
            if sup["mmd_quantiles"]:
                data["moss/supports/mmd_median"] = sup["mmd_quantiles"][1]
        for phase in ("first", "reuse", "adapt"):
            info = diag.get(phase)
            if not info:
                continue
            data[f"moss/{phase}/fallback_to_input"] = int(info["fallback_to_input"])
            data[f"moss/{phase}/feasible_frac"] = info["feasible_checkpoints"] / max(1, info["checkpoints"])
            if info["val_risk"] is not None:
                data[f"moss/{phase}/val_risk"] = info["val_risk"]
                data[f"moss/{phase}/F_t"] = info["F_t"]
            for k, v in info["mean_losses"].items():
                data[f"moss/{phase}/loss_{k}"] = v
        wandb_logger.log(data)

    def _dump_diagnostics(self):
        path = self.args.get("logs_name")
        if not path:
            return
        os.makedirs(path, exist_ok=True)
        payload = {"hyperparameters": {**self.hp, "p_max_resolved": self.p_max}, "tasks": self.diagnostics}
        tf = getattr(self.data_manager, "task_factors", None)
        if tf is not None:
            payload["ground_truth_task_factors"] = tf
        with open(os.path.join(path, "moss_diagnostics.json"), "w") as f:
            json.dump(payload, f, indent=1, default=float)
