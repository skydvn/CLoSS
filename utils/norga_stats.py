"""
Feature statistics and losses for the NoRGa / HiDe-Prompt learner.

* ``GaussianBank`` keeps one Gaussian per entry (a class in CIL, a (task, class) cell in
  DIL) and samples pseudo-features from it. HiDe-Prompt uses such samples to train the
  task-identity head and to re-align the shared head after every task.
* ``contrastive_reg`` is the contrastive regularisation that pushes the current task's
  instructed features away from the prototypes of earlier tasks.
"""
import torch
import torch.nn.functional as F


def _robust_cholesky(cov):
    """Cholesky factor of a covariance, adding jitter if needed (falls back to diagonal)."""
    eye = torch.eye(cov.shape[0], dtype=cov.dtype)
    scale = float(cov.diagonal().mean().clamp_min(1e-12))
    for jitter in [0.0] + [scale * 10.0 ** e for e in range(-6, 1)]:
        L, info = torch.linalg.cholesky_ex(cov + jitter * eye)
        if int(info) == 0:
            return L
    return torch.diag(cov.diagonal().clamp_min(1e-12).sqrt())


def kmeans(x, k, iters=25, seed=0):
    """Plain Lloyd's k-means on a (n, d) tensor; returns (centroids, assignment)."""
    gen = torch.Generator().manual_seed(int(seed))
    n = x.shape[0]
    centroids = x[torch.randperm(n, generator=gen)[:k]].clone()
    for _ in range(iters):
        assign = torch.cdist(x, centroids).argmin(dim=1)
        for j in range(k):
            members = assign == j
            if members.any():
                centroids[j] = x[members].mean(dim=0)
            else:
                centroids[j] = x[torch.randint(n, (1,), generator=gen)].squeeze(0)
    return centroids, torch.cdist(x, centroids).argmin(dim=1)


class GaussianBank:
    """Per-entry Gaussian statistics of features.

    cov_type: "full" keeps a Cholesky factor (D x D floats per entry, about 2.4 MB at
              D=768); "diag" keeps per-dimension standard deviations (D floats).
    n_centroids: > 1 fits k-means centroids per entry and a within-cluster covariance
                 (HiDe-Prompt's multi-centroid option for task-identity inference).
    """

    def __init__(self, cov_type="full", shrink=1e-4, n_centroids=1, seed=0):
        if cov_type not in ("full", "diag"):
            raise ValueError(f"cov_type must be 'full' or 'diag', got {cov_type}")
        self.cov_type = cov_type
        self.shrink = float(shrink)
        self.n_centroids = max(1, int(n_centroids))
        self.seed = int(seed)
        self.entries = []

    def __len__(self):
        return len(self.entries)

    @torch.no_grad()
    def add(self, feats, label, task):
        x = feats.detach().double().cpu()
        n, d = x.shape
        mean = x.mean(dim=0)
        k = self.n_centroids if n >= 2 * self.n_centroids else 1
        if k > 1:
            centroids, assign = kmeans(x, k, seed=self.seed + len(self.entries))
            resid = x - centroids[assign]
        else:
            centroids, resid = mean.unsqueeze(0), x - mean
        dof = max(n - k, 1)
        if self.cov_type == "full":
            cov = resid.T @ resid / dof + self.shrink * torch.eye(d, dtype=x.dtype)
            scale = _robust_cholesky(cov).float()
        else:
            scale = ((resid ** 2).sum(dim=0) / dof + self.shrink).sqrt().float()
        self.entries.append({
            "label": int(label), "task": int(task), "count": int(n),
            "mean": mean.float(), "centroids": centroids.float(), "scale": scale,
        })

    def prototypes(self, max_task=None):
        """Means and labels of entries whose task < max_task (all entries if None)."""
        chosen = [e for e in self.entries if max_task is None or e["task"] < max_task]
        if not chosen:
            return None, None
        return (torch.stack([e["mean"] for e in chosen]),
                torch.tensor([e["label"] for e in chosen], dtype=torch.long))

    @torch.no_grad()
    def sample(self, n_per_entry, target="label", generator=None):
        """Draw n_per_entry pseudo-features from every entry; target is "label" or "task"."""
        xs, ys = [], []
        for e in self.entries:
            c = e["centroids"]
            pick = torch.randint(c.shape[0], (n_per_entry,), generator=generator)
            z = torch.randn(n_per_entry, c.shape[1], generator=generator)
            noise = z @ e["scale"].T if self.cov_type == "full" else z * e["scale"]
            xs.append(c[pick] + noise)
            ys.append(torch.full((n_per_entry,), e[target], dtype=torch.long))
        return torch.cat(xs), torch.cat(ys)


def contrastive_reg(feats, labels, protos=None, proto_labels=None, temperature=0.8):
    """Supervised contrastive loss over the batch, with old prototypes as extra negatives.

    Positives are other batch samples of the same class. The denominator also contains
    prototypes of earlier tasks; prototypes that share the anchor's label are left out
    (this only happens in DIL, where every task contains every class).
    """
    f = F.normalize(feats, dim=1)
    B = f.shape[0]
    self_mask = torch.eye(B, dtype=torch.bool, device=f.device)
    pos = (labels[:, None] == labels[None, :]) & ~self_mask
    has_pos = pos.any(dim=1)
    if not bool(has_pos.any()):
        return feats.sum() * 0.0

    sim = (f @ f.T / temperature).masked_fill(self_mask, float("-inf"))
    denom = sim
    if protos is not None and len(protos) > 0:
        p = F.normalize(protos.to(device=f.device, dtype=f.dtype), dim=1)
        sp = f @ p.T / temperature
        same = labels[:, None] == proto_labels.to(labels.device)[None, :]
        denom = torch.cat([sim, sp.masked_fill(same, float("-inf"))], dim=1)
    log_prob = sim - torch.logsumexp(denom, dim=1, keepdim=True)
    per_anchor = -log_prob.masked_fill(~pos, 0.0).sum(dim=1) / pos.sum(dim=1).clamp_min(1)
    return per_anchor[has_pos].mean()
