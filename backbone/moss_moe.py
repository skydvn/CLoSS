"""
Expert bank and input-dependent composition for MoSS (Sec. 3.2).

    u      = b0(x)                frozen backbone feature (computed outside)
    z_m    = h_{phi_m}(u)         two-layer MLP experts, R^d -> R^r
    s_m(u) = a_m^T u + c_m        one linear score function per expert
    pi(u)  = softmax(s/tau) over the top-K scores, K = min(k, M), zero elsewhere
    z(x)   = sum_m pi_m(u) z_m(u)
    p(y|x) = softmax(W z + b)     shared head; input dim r is independent of M

Two forward modes:
  * forward(u, expert_out=False): routing-based dispatch. Each expert runs only on the inputs
    that route to it, so prediction costs B*K expert evaluations (Sec. 3.2 compute claim).
  * forward(u) / expert_out=True: every expert output is also computed and returned as
    "expert_out", for the auxiliary objectives that need them (L_div, L_feat, support
    statistics). The mixture z is identical in both modes.
Top-k membership is a non-differentiable index selection, held fixed in each backward pass, so
prediction gradients only reach active experts and their routing weights.

`dense_routing = True` makes every expert active (softmax over all M scores). The reuse phase
uses it for its exploration updates so that every existing router receives gradient; it is
never used for checkpoint evaluation.
"""
import torch
import torch.nn as nn


class Expert(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, u):
        return self.net(u)


class ExpertMixture(nn.Module):
    def __init__(self, in_dim, num_classes, num_experts=2, hidden_dim=256, out_dim=128, topk=2, tau=1.0):
        super().__init__()
        assert num_experts >= 2, "MoSS initializes M0 >= 2 experts."
        assert topk >= 2, "The active-expert budget k must be >= 2 (Eq. 11 needs K >= 2)."
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.num_classes = num_classes
        self.topk = topk
        self.tau = tau
        self.experts = nn.ModuleList([Expert(in_dim, hidden_dim, out_dim) for _ in range(num_experts)])
        self.routers = nn.ModuleList([nn.Linear(in_dim, 1) for _ in range(num_experts)])
        self.head = nn.Linear(out_dim, num_classes)
        # Warm-up routing for a candidate expert (Eq. 22); None means ordinary top-k routing.
        self.forced_expert = None
        # Dense exploration routing used by the reuse phase; False means ordinary top-k routing.
        self.dense_routing = False
        # Number of (input, expert) evaluations since the last reset (diagnostics / tests).
        self.expert_evals = 0

    # ------------------------------------------------------------------ bookkeeping
    @property
    def num_experts(self):
        return len(self.experts)

    def expert_param_count(self):
        """Stored parameters of one expert plus its router score function."""
        e = sum(p.numel() for p in self.experts[0].parameters())
        r = sum(p.numel() for p in self.routers[0].parameters())
        return e + r

    def stored_expert_router_params(self):
        """P_E(theta): stored expert and router parameters (the head is excluded)."""
        return sum(p.numel() for p in self.experts.parameters()) + sum(p.numel() for p in self.routers.parameters())

    @torch.no_grad()
    def add_expert(self, router_bias=0.0):
        """Append one expert (same init rule as the initial bank) and one router score function,
        whose weight is zero and whose bias is given (Sec. 3.4)."""
        device = self.head.weight.device
        expert = Expert(self.in_dim, self.hidden_dim, self.out_dim).to(device)
        router = nn.Linear(self.in_dim, 1).to(device)
        router.weight.zero_()
        router.bias.fill_(float(router_bias))
        self.experts.append(expert)
        self.routers.append(router)
        return self.num_experts - 1

    # ------------------------------------------------------------------ routing
    def router_scores(self, u):
        return torch.cat([r(u) for r in self.routers], dim=-1)  # (B, M)

    def active_set(self, scores):
        b, m = scores.shape
        if self.dense_routing:
            return torch.ones_like(scores, dtype=torch.bool)
        k = min(self.topk, m)
        active = torch.zeros_like(scores, dtype=torch.bool)
        if self.forced_expert is not None and m > 1:
            f = self.forced_expert
            others = scores.detach().clone()
            others[:, f] = float("-inf")
            idx = others.topk(k - 1, dim=-1).indices
            active.scatter_(-1, idx, True)
            active[:, f] = True
        else:
            idx = scores.detach().topk(k, dim=-1).indices
            active.scatter_(-1, idx, True)
        return active

    def routing(self, u):
        scores = self.router_scores(u)
        active = self.active_set(scores)
        masked = (scores / self.tau).masked_fill(~active, float("-inf"))
        weights = torch.softmax(masked, dim=-1)
        return scores, weights, active

    # ------------------------------------------------------------------ forward
    def expert_outputs(self, u, idx=None):
        idx = range(self.num_experts) if idx is None else idx
        idx = list(idx)
        self.expert_evals += u.shape[0] * len(idx)
        return torch.stack([self.experts[m](u) for m in idx], dim=1)  # (B, |idx|, r)

    def _dispatch(self, u, weights, active):
        """z = sum_m pi_m z_m, evaluating each expert only on the inputs routed to it."""
        z = u.new_zeros(u.shape[0], self.out_dim)
        for m in range(self.num_experts):
            rows = torch.nonzero(active[:, m], as_tuple=True)[0]
            if rows.numel() == 0:
                continue
            self.expert_evals += int(rows.numel())
            z = z.index_add(0, rows, weights[rows, m].unsqueeze(-1) * self.experts[m](u[rows]))
        return z

    def forward(self, u, expert_out=True):
        scores, weights, active = self.routing(u)
        if expert_out:
            zs = self.expert_outputs(u)
            z = torch.einsum("bm,bmr->br", weights, zs)
        else:
            zs = None
            z = self._dispatch(u, weights, active)
        logits = self.head(z)
        return {
            "logits": logits,
            "z": z,
            "weights": weights,
            "active": active,
            "scores": scores,
            "expert_out": zs,
        }

    def leave_one_out_logits(self, scores, active, zs):
        """
        Logits of p^(-m) for every m (Eq. 11): expert m is removed from the same active set and
        the softmax is recomputed over the remaining active experts, without admitting a
        replacement expert. Rows for inactive experts reproduce the full mixture.

        Working in score space keeps this exact when one routing weight is close to 1, where
        the closed form (z - pi_m z_m) / (1 - pi_m) suffers cancellation and the clamp on
        1 - pi_m distorts the remaining mixture.

        Args: scores (B, M) raw router scores, active (B, M) bool active set, zs (B, M, r).
        A legacy call leave_one_out_logits(z, weights, zs) is still accepted (see below).
        Returns (B, M, C).
        """
        if active.dtype != torch.bool:
            return self._leave_one_out_from_weights(active, zs)
        B, M = scores.shape
        s = (scores / self.tau).masked_fill(~active, float("-inf"))
        s = s.unsqueeze(1).expand(B, M, M).clone()               # row m: the set without m
        eye = torch.eye(M, dtype=torch.bool, device=s.device).unsqueeze(0)
        s = s.masked_fill(eye, float("-inf"))
        w = torch.softmax(s, dim=-1)                             # (B, M, M), rows sum to 1
        z_minus = torch.einsum("bjm,bmr->bjr", w, zs)
        return self.head(z_minus)

    def _leave_one_out_from_weights(self, weights, zs):
        """Legacy signature (z, weights, zs). Renormalises the remaining weights by their own sum
        (no subtraction from z), which avoids the cancellation of the old closed form; it is
        still inexact when the remaining weights underflow, so new code passes scores."""
        B, M = weights.shape
        w = weights.unsqueeze(1).expand(B, M, M).clone()
        w = w.masked_fill(torch.eye(M, dtype=torch.bool, device=w.device).unsqueeze(0), 0.0)
        w = w / w.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(w.dtype).tiny)
        return self.head(torch.einsum("bjm,bmr->bjr", w, zs))
