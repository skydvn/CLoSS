"""
NoRGa (Le et al., "Mixture of Experts Meets Prompt-Based Continual Learning", NeurIPS
2024) inside the CLoSS training loop. NoRGa is built on the HiDe-Prompt framework
(Wang et al., NeurIPS 2023), which this learner re-implements:

  per task t
    1. WTP   train prompt t, its NoRGa gates (tau, alpha) and the shared head on task t
             with the ground-truth prompt: masked CE + contrastive regularisation.
    2. Stats Gaussian statistics of uninstructed (no prompt) and instructed (prompt t)
             features of task t.
    3. TII   task-identity head on uninstructed features (real features of task t, then
             Gaussian replay of every seen class).
    4. TAP   re-align the shared head on Gaussian samples of instructed features of every
             seen class.
  inference: TII on uninstructed features -> task id -> prompted forward -> shared head.

``"norga": false`` gives plain prefix tuning, i.e. HiDe-Prompt (model_name "hideprompt").
The learner implements the interface trainer.py uses: ``_network``, ``incremental_train``,
``eval_task`` and ``after_task``. See NORGA.md for the paper-to-code map and the list of
choices that were not verifiable against the official trainers.
"""
import json
import logging
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from backbone.norga_vit import PromptViT, build_vit, get_gate_act
from utils import norga_data
from utils.norga_stats import GaussianBank, contrastive_reg

DEFAULTS = {
    # NoRGa gate, Eq. 15: A_prompt + alpha * act(tau * A_prompt)
    "norga": True,
    "gate_act": "tanh",
    "act_scale_init": [1.0, 1.0],          # [tau, alpha], learned per task and prompt layer
    # prefix prompts (HiDe-Prompt / DualPrompt e-prompts)
    "prompt_layers": [0, 1, 2, 3, 4],
    "prompt_length": 5,
    "prompt_init": "uniform",              # "uniform" (U[-1, 1]) or "zero"
    "prompt_init_from_prev": True,         # start prompt t from prompt t-1
    "prompt_momentum": 0.01,               # pull prompt t towards the mean of old prompts
    # within-task prediction (WTP)
    "wtp_epochs": 20,
    "wtp_batch_size": 24,
    "wtp_optimizer": "adam",
    "wtp_lr": 0.005,
    "head_lr": None,                       # None -> wtp_lr
    "wtp_weight_decay": 0.0,
    "wtp_scheduler": "cosine",             # "cosine" or "constant"
    "train_mask": True,                    # CIL: CE over the current task's classes only
    "reg": 0.1,                            # weight of the contrastive regularisation
    "cr_temperature": 0.8,
    # Gaussian statistics
    "stats_cov_type": "full",              # "full" or "diag"
    "stats_shrink": 1e-4,
    # task-identity inference (TII)
    "tii_head": "linear",                  # "linear" or "mlp"
    "tii_hidden_dim": 1024,
    "tii_n_centroids": 1,
    "tii_epochs": 20,
    "tii_lr": 5e-4,
    "tii_batch_size": 128,
    "tii_ca_epochs": 30,
    "tii_ca_lr": 0.005,
    # task-adaptive prediction (TAP)
    "ca_epochs": 30,
    "ca_lr": 0.005,
    "ca_samples_per_class": 120,
    "ca_batch_size": None,                 # None -> ca_samples_per_class
    "ca_weight_decay": 5e-4,
    "ca_logit_norm": 0.0,                  # > 0 divides L2-normalised logits by this value
    # misc
    "num_workers": 8,
    "eval_batch_size": 128,
    "eval_oracle": False,                  # also report accuracy with ground-truth task ids
}

# Names used by the official HiDe-Prompt / NoRGa scripts.
ALIASES = {"crct_epochs": "ca_epochs"}


def _resolve_cfg(args):
    cfg = dict(DEFAULTS)
    for alias, key in ALIASES.items():
        if alias in args and key not in args:
            cfg[key] = args[alias]
    for key in DEFAULTS:
        if key in args:
            cfg[key] = args[key]
    if str(args.get("model_name", "")).lower() == "hideprompt" and "norga" not in args:
        cfg["norga"] = False
    if cfg["ca_batch_size"] is None:
        cfg["ca_batch_size"] = cfg["ca_samples_per_class"]
    if cfg["head_lr"] is None:
        cfg["head_lr"] = cfg["wtp_lr"]
    if len(cfg["act_scale_init"]) != 2:
        raise ValueError("act_scale_init must be [tau, alpha]")
    cfg["prompt_layers"] = [int(v) for v in cfg["prompt_layers"]]
    return cfg


def _make_optimizer(name, groups, weight_decay):
    name = str(name).lower()
    lr = groups[0]["lr"]
    if name == "adam":
        return torch.optim.Adam(groups, lr=lr, betas=(0.9, 0.999), weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(groups, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(groups, lr=lr, momentum=0.9, weight_decay=weight_decay)
    raise ValueError(f"Unknown optimizer {name}")


def _make_scheduler(name, opt, total_steps):
    name = str(name).lower()
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(int(total_steps), 1))
    if name == "constant":
        return None
    raise ValueError(f"Unknown scheduler {name}")


def _pct(hits):
    if hits.numel() == 0:
        return 0.0
    return float(np.around(hits.float().mean().item() * 100.0, decimals=2))


class NoRGaNet(nn.Module):
    """Frozen ViT + per-task prefix prompts and NoRGa gates + shared head + TII head."""

    def __init__(self, vit, cfg, nb_tasks, nb_classes, scenario):
        super().__init__()
        self.backbone = PromptViT(vit, cfg["prompt_layers"])
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        D, H, Dh = self.backbone.embed_dim, self.backbone.num_heads, self.backbone.head_dim
        n_layers = len(cfg["prompt_layers"])
        self.scenario = scenario
        self.use_norga = bool(cfg["norga"])
        self.gate_act_name = str(cfg["gate_act"])
        self._gate_act = get_gate_act(self.gate_act_name)

        init_scale = torch.tensor([float(v) for v in cfg["act_scale_init"]])
        self.prompts = nn.ParameterList()
        self.act_scales = nn.ParameterList()
        for _ in range(nb_tasks):
            p = torch.empty(n_layers, 2, int(cfg["prompt_length"]), H, Dh)
            if cfg["prompt_init"] == "uniform":
                nn.init.uniform_(p, -1.0, 1.0)
            elif cfg["prompt_init"] == "zero":
                nn.init.zeros_(p)
            else:
                raise ValueError(f"Unknown prompt_init {cfg['prompt_init']}")
            self.prompts.append(nn.Parameter(p, requires_grad=False))
            self.act_scales.append(
                nn.Parameter(init_scale.repeat(n_layers, 1).clone(), requires_grad=False))

        self.head = nn.Linear(D, nb_classes)
        tii_out = nb_tasks if scenario == "dil" else nb_classes
        if cfg["tii_head"] == "mlp":
            hid = int(cfg["tii_hidden_dim"])
            self.tii_head = nn.Sequential(nn.Linear(D, hid), nn.GELU(), nn.Linear(hid, tii_out))
        elif cfg["tii_head"] == "linear":
            self.tii_head = nn.Linear(D, tii_out)
        else:
            raise ValueError(f"Unknown tii_head {cfg['tii_head']}")
        for p in self.tii_head.parameters():
            p.requires_grad_(False)

        # Saved with the state dict so a reloaded network can run inference on its own.
        self.register_buffer("class_to_task", torch.full((nb_classes,), -1, dtype=torch.long))
        self.register_buffer("seen_counts", torch.zeros(2, dtype=torch.long))  # tasks, classes

    def _gather(self, task_ids):
        B = task_ids.shape[0]
        uniq = torch.unique(task_ids)
        if uniq.numel() == 1:
            t = int(uniq.item())
            prompts = self.prompts[t].unsqueeze(0).expand(B, *self.prompts[t].shape)
            scales = (self.act_scales[t].unsqueeze(0).expand(B, -1, -1)
                      if self.use_norga else None)
        else:
            n = int(task_ids.max().item()) + 1
            prompts = torch.stack([self.prompts[i] for i in range(n)])[task_ids]
            scales = (torch.stack([self.act_scales[i] for i in range(n)])[task_ids]
                      if self.use_norga else None)
        return prompts, scales

    def features(self, x, task_ids=None):
        """Uninstructed features if task_ids is None, else features under those prompts."""
        if task_ids is None:
            return self.backbone(x)
        prompts, scales = self._gather(task_ids)
        return self.backbone(x, prompts, scales, self._gate_act)

    @torch.no_grad()
    def infer_task(self, x):
        n_tasks = int(self.seen_counts[0])
        if n_tasks <= 1:
            return torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        logits = self.tii_head(self.backbone(x))
        if self.scenario == "dil":
            return logits[:, :n_tasks].argmax(dim=1)
        n_cls = int(self.seen_counts[1])
        return self.class_to_task[logits[:, :n_cls].argmax(dim=1)]

    def forward(self, x, task_ids=None):
        if task_ids is None:
            task_ids = self.infer_task(x)
        feats = self.features(x, task_ids)
        logits = self.head(feats)
        n_cls = int(self.seen_counts[1])
        if self.scenario != "dil" and 0 < n_cls < logits.shape[1]:
            unseen = torch.zeros(logits.shape[1], dtype=torch.bool, device=logits.device)
            unseen[n_cls:] = True
            logits = logits.masked_fill(unseen, float("-inf"))
        return {"logits": logits, "features": feats, "task_ids": task_ids}


class Learner:
    def __init__(self, args):
        self.args = args
        self.cfg = _resolve_cfg(args)
        self.scenario = str(args.get("scenario", "cil")).lower()
        self._device = args["device"][0]
        self._multiple_gpus = args["device"]
        self._cur_task = -1
        self._known_classes = 0
        self._total_classes = 0
        self.nb_tasks = int(args["nb_tasks"])
        self.nb_classes = int(args["nb_classes"])
        # trainer.py reads cnn_accy[f"top{min(5, init_cls)}"]
        self.topk = min(5, int(args.get("init_cls", self.nb_classes)))

        vit = build_vit(args)
        self._network = NoRGaNet(vit, self.cfg, self.nb_tasks, self.nb_classes,
                                 self.scenario).to(self._device)
        seed = int(args.get("seed", 1993)) if not isinstance(args.get("seed"), list) else 1993
        self._gen = torch.Generator().manual_seed(seed)
        self.tii_bank = GaussianBank(self.cfg["stats_cov_type"], self.cfg["stats_shrink"],
                                     self.cfg["tii_n_centroids"], seed=seed)
        self.tap_bank = GaussianBank(self.cfg["stats_cov_type"], self.cfg["stats_shrink"],
                                     1, seed=seed + 1)
        self.task_ranges = []      # CIL: [start, end) class range per task
        self.data_manager = None
        self.diagnostics = []
        self._task_diag = {}
        logging.info(f"[NoRGa] scenario={self.scenario} norga={self.cfg['norga']} "
                     f"gate_act={self.cfg['gate_act']} config={self.cfg}")

    # ------------------------------------------------------------------ data helpers
    def _task_dataset(self, task, source, mode):
        if self.scenario == "dil":
            return norga_data.dil_task_dataset(self.data_manager, task, source, mode)
        lo, hi = self.task_ranges[task]
        return norga_data.cil_dataset(self.data_manager, lo, hi, source, mode)

    def _loader(self, dataset, batch_size, shuffle):
        return DataLoader(dataset, batch_size=int(batch_size), shuffle=shuffle,
                          num_workers=int(self.cfg["num_workers"]),
                          pin_memory=getattr(self._device, "type", "cpu") == "cuda")

    # ------------------------------------------------------------------ training
    def incremental_train(self, data_manager):
        self.data_manager = data_manager
        self._cur_task += 1
        t = self._cur_task
        net = self._network
        if self.scenario == "dil":
            self._total_classes = self.nb_classes
        else:
            size = int(data_manager.get_task_size(t))
            self._total_classes = self._known_classes + size
            self.task_ranges.append((self._known_classes, self._total_classes))
            net.class_to_task[self._known_classes:self._total_classes] = t
        net.seen_counts[0] = t + 1
        net.seen_counts[1] = self._total_classes
        logging.info(f"[NoRGa] task {t}: classes {self._known_classes}-{self._total_classes}")

        self._task_diag = {"task": t}
        self._init_prompt(t)
        self._task_diag["wtp"] = self._train_wtp(self._task_dataset(t, "train", "train"), t)
        f_un, f_in, labels = self._extract(self._task_dataset(t, "train", "test"), t)
        self._update_banks(f_un, f_in, labels, t)
        self._task_diag["tii"] = self._train_tii(f_un, labels, t)
        self._task_diag["tap"] = self._train_tap()

    def _init_prompt(self, t):
        net = self._network
        with torch.no_grad():
            if t > 0 and self.cfg["prompt_init_from_prev"]:
                net.prompts[t].copy_(net.prompts[t - 1])
                net.act_scales[t].copy_(net.act_scales[t - 1])
        for i in range(self.nb_tasks):
            net.prompts[i].requires_grad_(i == t)
            net.act_scales[i].requires_grad_(i == t and net.use_norga)

    def _train_wtp(self, dataset, t):
        cfg, net, dev = self.cfg, self._network, self._device
        loader = self._loader(dataset, cfg["wtp_batch_size"], shuffle=True)
        for p in net.head.parameters():
            p.requires_grad_(True)
        prompt_params = [net.prompts[t]] + ([net.act_scales[t]] if net.use_norga else [])
        groups = [{"params": prompt_params, "lr": float(cfg["wtp_lr"])},
                  {"params": list(net.head.parameters()), "lr": float(cfg["head_lr"])}]
        opt = _make_optimizer(cfg["wtp_optimizer"], groups, float(cfg["wtp_weight_decay"]))
        epochs = int(cfg["wtp_epochs"])
        sched = _make_scheduler(cfg["wtp_scheduler"], opt, epochs * max(len(loader), 1))

        protos, proto_labels = self.tap_bank.prototypes(max_task=t)
        if protos is not None:
            protos, proto_labels = protos.to(dev), proto_labels.to(dev)
        momentum = float(cfg["prompt_momentum"])
        prev_mean = None
        if t > 0 and momentum > 0:
            with torch.no_grad():
                prev_mean = torch.stack([net.prompts[i].detach() for i in range(t)]).mean(0)

        mask = None
        if cfg["train_mask"] and self.scenario != "dil":
            lo, hi = self.task_ranges[t]
            mask = torch.ones(self.nb_classes, dtype=torch.bool, device=dev)
            mask[lo:hi] = False

        net.train()
        net.backbone.eval()
        history = []
        for epoch in range(epochs):
            n, correct, loss_sum, reg_sum = 0, 0, 0.0, 0.0
            for batch in loader:
                x, y = norga_data.unpack_batch(batch)
                x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
                tids = torch.full((x.shape[0],), t, dtype=torch.long, device=dev)
                feats = net.features(x, tids)
                logits = net.head(feats)
                if mask is not None:
                    logits = logits.masked_fill(mask, float("-inf"))
                loss = F.cross_entropy(logits, y)
                if cfg["reg"] > 0 and protos is not None:
                    reg = contrastive_reg(feats, y, protos, proto_labels, cfg["cr_temperature"])
                    loss = loss + float(cfg["reg"]) * reg
                    reg_sum += float(reg.detach()) * x.shape[0]
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                if sched is not None:
                    sched.step()
                if prev_mean is not None:
                    with torch.no_grad():
                        net.prompts[t].mul_(1.0 - momentum).add_(prev_mean, alpha=momentum)
                n += x.shape[0]
                loss_sum += float(loss.detach()) * x.shape[0]
                correct += int((logits.argmax(1) == y).sum())
            row = {"epoch": epoch + 1, "loss": loss_sum / max(n, 1),
                   "reg": reg_sum / max(n, 1), "train_acc": 100.0 * correct / max(n, 1)}
            history.append(row)
            logging.info(f"[NoRGa] task {t} WTP epoch {epoch + 1}/{epochs} "
                         f"loss {row['loss']:.4f} reg {row['reg']:.4f} "
                         f"train_acc {row['train_acc']:.2f}")
        net.eval()
        return history[-1] if history else {}

    @torch.no_grad()
    def _extract(self, dataset, t):
        net, dev = self._network, self._device
        net.eval()
        f_un, f_in, ys = [], [], []
        for batch in self._loader(dataset, self.cfg["eval_batch_size"], shuffle=False):
            x, y = norga_data.unpack_batch(batch)
            x = x.to(dev, non_blocking=True)
            tids = torch.full((x.shape[0],), t, dtype=torch.long, device=dev)
            f_un.append(net.features(x).float().cpu())
            f_in.append(net.features(x, tids).float().cpu())
            ys.append(y.cpu())
        return torch.cat(f_un), torch.cat(f_in), torch.cat(ys)

    def _update_banks(self, f_un, f_in, labels, t):
        for c in torch.unique(labels).tolist():
            m = labels == c
            self.tii_bank.add(f_un[m], label=c, task=t)
            self.tap_bank.add(f_in[m], label=c, task=t)

    def _align_head(self, head, bank, target, n_valid, epochs, lr):
        """Train `head` on Gaussian samples from `bank` (HiDe-Prompt classifier alignment)."""
        cfg, dev = self.cfg, self._device
        for p in head.parameters():
            p.requires_grad_(True)
        head.train()
        opt = torch.optim.SGD(head.parameters(), lr=float(lr), momentum=0.9,
                              weight_decay=float(cfg["ca_weight_decay"]))
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(int(epochs), 1))
        bs, temp = int(cfg["ca_batch_size"]), float(cfg["ca_logit_norm"])
        last = 0.0
        for _ in range(int(epochs)):
            xs, ys = bank.sample(int(cfg["ca_samples_per_class"]), target=target,
                                 generator=self._gen)
            perm = torch.randperm(xs.shape[0], generator=self._gen)
            xs, ys = xs[perm], ys[perm]
            total, n = 0.0, 0
            for i in range(0, xs.shape[0], bs):
                xb, yb = xs[i:i + bs].to(dev), ys[i:i + bs].to(dev)
                logits = head(xb)[:, :n_valid]
                if temp > 0:
                    logits = logits / (logits.norm(dim=1, keepdim=True) + 1e-7) / temp
                loss = F.cross_entropy(logits, yb)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                total += float(loss.detach()) * xb.shape[0]
                n += xb.shape[0]
            sched.step()
            last = total / max(n, 1)
        for p in head.parameters():
            p.requires_grad_(False)
        head.eval()
        return last

    def _train_tii(self, feats, labels, t):
        cfg, net, dev = self.cfg, self._network, self._device
        head = net.tii_head
        out = {}
        if self.scenario != "dil" and int(cfg["tii_epochs"]) > 0:
            lo, hi = self.task_ranges[t]
            for p in head.parameters():
                p.requires_grad_(True)
            head.train()
            opt = torch.optim.Adam(head.parameters(), lr=float(cfg["tii_lr"]))
            n, bs = feats.shape[0], int(cfg["tii_batch_size"])
            for _ in range(int(cfg["tii_epochs"])):
                perm = torch.randperm(n, generator=self._gen)
                total = 0.0
                for i in range(0, n, bs):
                    idx = perm[i:i + bs]
                    xb, yb = feats[idx].to(dev), labels[idx].to(dev)
                    loss = F.cross_entropy(head(xb)[:, lo:hi], yb - lo)
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()
                    total += float(loss.detach()) * xb.shape[0]
                out["real_loss"] = total / max(n, 1)
            for p in head.parameters():
                p.requires_grad_(False)
            head.eval()
        n_valid = (t + 1) if self.scenario == "dil" else self._total_classes
        if int(cfg["tii_ca_epochs"]) > 0 and n_valid > 1:
            out["ca_loss"] = self._align_head(
                head, self.tii_bank, "task" if self.scenario == "dil" else "label",
                n_valid, cfg["tii_ca_epochs"], cfg["tii_ca_lr"])
        return out

    def _train_tap(self):
        if int(self.cfg["ca_epochs"]) <= 0:
            return {}
        n_valid = self.nb_classes if self.scenario == "dil" else self._total_classes
        loss = self._align_head(self._network.head, self.tap_bank, "label", n_valid,
                                self.cfg["ca_epochs"], self.cfg["ca_lr"])
        return {"ca_loss": loss}

    # ------------------------------------------------------------------ evaluation
    @torch.no_grad()
    def _predict(self, dataset, oracle=None):
        """Top-k predictions, labels and the task ids used (inferred, or oracle(y))."""
        net, dev = self._network, self._device
        net.eval()
        preds, ys, tids = [], [], []
        for batch in self._loader(dataset, self.cfg["eval_batch_size"], shuffle=False):
            x, y = norga_data.unpack_batch(batch)
            x = x.to(dev, non_blocking=True)
            task_ids = None if oracle is None else oracle(y).to(dev)
            out = net(x, task_ids=task_ids)
            k = min(self.topk, out["logits"].shape[1])
            preds.append(torch.topk(out["logits"], k=k, dim=1)[1].cpu())
            ys.append(y.cpu())
            tids.append(out["task_ids"].cpu())
        return torch.cat(preds), torch.cat(ys), torch.cat(tids)

    def eval_task(self):
        t = self._cur_task
        oracle_on = bool(self.cfg["eval_oracle"])
        grouped = {}
        if self.scenario == "dil":
            accs, accks, tiis, oracles = [], [], [], []
            for s in range(t + 1):
                ds = self._task_dataset(s, "test", "test")
                pred, y, tid = self._predict(ds)
                acc = _pct(pred[:, 0] == y)
                grouped[f"{s:02d}-{s:02d}"] = acc
                accs.append(acc)
                accks.append(_pct((pred == y[:, None]).any(dim=1)))
                tiis.append(_pct(tid == s))
                if oracle_on:
                    o_pred, o_y, _ = self._predict(ds, oracle=lambda yy, s=s: torch.full_like(yy, s))
                    oracles.append(_pct(o_pred[:, 0] == o_y))
            # DIL reports the task-averaged accuracy, as MoSS does.
            top1 = round(float(np.mean(accs)), 2)
            topk = round(float(np.mean(accks)), 2)
            tii_acc = round(float(np.mean(tiis)), 2)
            oracle_acc = round(float(np.mean(oracles)), 2) if oracles else None
            grouped["total"] = top1
            grouped["old"] = round(float(np.mean(accs[:-1])), 2) if t > 0 else 0.0
            grouped["new"] = accs[-1]
        else:
            ds = norga_data.cil_dataset(self.data_manager, 0, self._total_classes, "test", "test")
            pred, y, tid = self._predict(ds)
            c2t = self._network.class_to_task.cpu()
            hit1 = pred[:, 0] == y
            for lo, hi in self.task_ranges:
                m = (y >= lo) & (y < hi)
                grouped[f"{lo:02d}-{hi - 1:02d}"] = _pct(hit1[m])
            grouped["total"] = _pct(hit1)
            old = y < self._known_classes
            grouped["old"] = _pct(hit1[old])
            grouped["new"] = _pct(hit1[~old])
            top1 = grouped["total"]
            topk = _pct((pred == y[:, None]).any(dim=1))
            tii_acc = _pct(tid == c2t[y])
            oracle_acc = None
            if oracle_on:
                o_pred, o_y, _ = self._predict(ds, oracle=lambda yy: c2t[yy])
                oracle_acc = _pct(o_pred[:, 0] == o_y)

        msg = f"[NoRGa] task {t}: top1 {top1:.2f}  task-identity acc {tii_acc:.2f}"
        if oracle_acc is not None:
            msg += f"  oracle-task top1 {oracle_acc:.2f}"
        logging.info(msg)
        self._record(top1, tii_acc, oracle_acc)
        cnn_accy = {"grouped": grouped, "top1": top1, f"top{self.topk}": topk}
        return cnn_accy, None

    def after_task(self):
        t = self._cur_task
        net = self._network
        self._known_classes = self._total_classes
        net.prompts[t].requires_grad_(False)
        net.act_scales[t].requires_grad_(False)
        for p in net.head.parameters():
            p.requires_grad_(False)
        if net.use_norga:
            tau_alpha = net.act_scales[t].detach().cpu().tolist()
            logging.info(f"[NoRGa] task {t} learned [tau, alpha] per prompt layer: {tau_alpha}")

    # ------------------------------------------------------------------ diagnostics
    def _record(self, top1, tii_acc, oracle_acc):
        net = self._network
        entry = dict(self._task_diag)
        entry.update({"top1": top1, "tii_acc": tii_acc, "oracle_top1": oracle_acc,
                      "act_scale": net.act_scales[self._cur_task].detach().cpu().tolist()
                      if net.use_norga else None})
        self.diagnostics.append(entry)
        logs_name = self.args.get("logs_name")
        if logs_name and os.path.isdir(logs_name):
            with open(os.path.join(logs_name, "norga_diagnostics.json"), "w") as f:
                json.dump(self.diagnostics, f, indent=2)
