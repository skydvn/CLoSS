"""Unit tests for the MoSS components. Run: python -m tests.test_moss  (or pytest tests/)."""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backbone.moss_moe import ExpertMixture  # noqa: E402
from utils.moss_losses import cross_cov_div, kd_kl, mmd2  # noqa: E402
from utils.moss_memory import MoSSMemory, Reservoir  # noqa: E402


def test_reservoir_capacity_and_uniformity():
    B, N, trials = 20, 200, 400
    counts = np.zeros(N)
    for seed in range(trials):
        r = Reservoir(B, seed=seed)
        u = torch.arange(N, dtype=torch.float32).unsqueeze(1)
        r.add(u[:120], torch.zeros(120), task_id=0)   # records arrive over two tasks
        r.add(u[120:], torch.zeros(80), task_id=1)
        assert len(r) == B and r.n_seen == N
        counts[r.u[:, 0].long().numpy()] += 1
    p = counts / trials
    # every record is retained with probability B / N = 0.1
    assert abs(p.mean() - B / N) < 1e-9
    assert np.abs(p - B / N).max() < 0.06, np.abs(p - B / N).max()


def test_replay_is_task_uniform():
    mem = MoSSMemory(total_size=1000, val_frac=0.0, seed=0)
    mem.update(torch.randn(900, 4), torch.zeros(900), torch.randn(0, 4), torch.zeros(0), task_id=0)
    mem.update(torch.randn(100, 4), torch.zeros(100), torch.randn(0, 4), torch.zeros(0), task_id=1)
    g = torch.Generator().manual_seed(0)
    idx = mem.sample_replay_indices(20000, g)
    _, _, s = mem.fit.data()
    frac_task1 = (s[idx] == 1).float().mean().item()
    assert abs(frac_task1 - 0.5) < 0.02, frac_task1  # not 0.1, despite 9x fewer stored records


def test_mmd_properties():
    torch.manual_seed(0)
    x = torch.randn(200, 8)
    y = torch.randn(200, 8)
    z = torch.randn(200, 8) + 2.0
    sigma = 4.0  # ~ median pairwise distance of 8-d standard normals
    assert mmd2(x, x, sigma, "v").abs() < 1e-6
    assert mmd2(x, y, sigma, "u").abs() < 0.02
    assert mmd2(x, z, sigma, "v") > 10 * mmd2(x, y, sigma, "v")
    # V-statistic is biased upward for small cells; U-statistic is not
    small_v = np.mean([mmd2(torch.randn(3, 8), torch.randn(3, 8), 3.0, "v").item() for _ in range(300)])
    small_u = np.mean([mmd2(torch.randn(3, 8), torch.randn(3, 8), 3.0, "u").item() for _ in range(300)])
    assert small_v > 0.05 and abs(small_u) < 0.03, (small_v, small_u)


def _mixture(M=4, k=2):
    torch.manual_seed(0)
    moe = ExpertMixture(in_dim=6, num_classes=5, num_experts=M, hidden_dim=16, out_dim=8, topk=k)
    for r in moe.routers:  # spread scores so the top-k sets differ across inputs
        torch.nn.init.normal_(r.weight, std=1.0)
    return moe


def test_routing_topk_and_warmup():
    moe = _mixture(M=5, k=3)
    out = moe(torch.randn(64, 6))
    assert torch.allclose(out["weights"].sum(-1), torch.ones(64))
    assert (out["active"].sum(-1) == 3).all()
    assert ((out["weights"] > 0) == out["active"]).all()
    moe.forced_expert = 4
    out = moe(torch.randn(64, 6))
    assert out["active"][:, 4].all() and (out["active"].sum(-1) == 3).all()
    moe.forced_expert = None


def test_leave_one_out_matches_explicit_renormalization():
    moe = _mixture(M=4, k=3)
    u = torch.randn(32, 6)
    out = moe(u)
    loo = moe.leave_one_out_logits(out["z"], out["weights"], out["expert_out"])
    for m in range(4):
        w = out["weights"].clone()
        w[:, m] = 0.0
        w = w / w.sum(-1, keepdim=True)  # remaining active weights divided by 1 - pi_m
        z = torch.einsum("bm,bmr->br", w, out["expert_out"])
        assert torch.allclose(moe.head(z), loo[:, m], atol=1e-5)


def test_diversity_is_linear_cka():
    torch.manual_seed(0)
    h = torch.randn(50, 3, 7)
    h[:, 1] = h[:, 0] @ torch.randn(7, 7)          # expert 1 is a linear map of expert 0
    val_dep = cross_cov_div(h, [(0, 1)]).item()
    val_ind = cross_cov_div(h, [(0, 2)]).item()

    def cka(a, b):
        a = a - a.mean(0)
        b = b - b.mean(0)
        return ((a.t() @ b).norm() ** 2 / ((a.t() @ a).norm() * (b.t() @ b).norm())).item()

    assert abs(val_dep - cka(h[:, 0], h[:, 1])) < 1e-5
    assert val_dep > val_ind


def test_add_expert_bookkeeping():
    moe = _mixture(M=2, k=2)
    p0 = moe.stored_expert_router_params()
    m = moe.add_expert(router_bias=1.25)
    assert m == 2 and moe.num_experts == 3
    assert moe.stored_expert_router_params() - p0 == moe.expert_param_count()
    assert moe.routers[m].weight.abs().sum() == 0 and abs(moe.routers[m].bias.item() - 1.25) < 1e-6


def test_kd_is_zero_for_identical_logits():
    a = torch.randn(10, 5)
    assert kd_kl(a, a).abs() < 1e-6


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
