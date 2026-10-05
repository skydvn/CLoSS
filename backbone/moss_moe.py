"""
Expert bank and input-dependent composition for MoSS (Sec. 3.2).

    u      = b0(x)                                    frozen backbone feature (computed outside)
    z_m    = h_{phi_m}(u)                             two-layer MLP experts, R^d -> R^r
    s_m(u) = a_m^T u + c_m                            one linear score function per expert
    pi(u)  = softmax(s/tau) over the top-K scores     K = min(k, M), zero elsewhere
    z(x)   = sum_m pi_m(u) z_m(u)
    p(y|x) = softmax(W z + b)                         shared head; input dim r is independent of M

Every expert output is computed densely for simplicity; inactive experts receive zero weight, so
prediction gradients only reach active experts and their routing weights (top-k membership is a
non-differentiable index selection, i.e. held fixed in each backward pass).
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
        return torch.stack([self.experts[m](u) for m in idx], dim=1)  # (B, |idx|, r)

    def forward(self, u):
        scores, weights, active = self.routing(u)
        zs = self.expert_outputs(u)
        z = torch.einsum("bm,bmr->br", weights, zs)
        logits = self.head(z)
        return {
            "logits": logits,
            "z": z,
            "weights": weights,
            "active": active,
            "scores": scores,
            "expert_out": zs,
        }

    def leave_one_out_logits(self, z, weights, zs):
        """
        Logits of p^(-m) for every m (Eq. 11): expert m is removed from the same active set and the
        remaining weights are divided by 1 - pi_m, without admitting a replacement expert.
        Closed form: z^(-m) = (z - pi_m z_m) / (1 - pi_m); rows with pi_m = 0 are unchanged.
        Returns (B, M, C).
        """
        pi = weights.unsqueeze(-1)  # (B, M, 1)
        denom = (1.0 - pi).clamp_min(1e-6)
        z_minus = (z.unsqueeze(1) - pi * zs) / denom
        return self.head(z_minus)
