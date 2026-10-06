"""
Tests for the revision of commit 173badb (CPU, seconds).

    python3 -m tests.test_moss_revision

1. Eq. 11 leave-one-out matches a float64 brute force, including when one routing weight is
   1.0 in float32, where the old closed form fails;
2. legacy call signature leave_one_out_logits(z, weights, zs) still works;
3. routing-based dispatch gives the same predictions and gradients as dense evaluation, with
   B*K instead of B*M expert evaluations;
4. dense exploration routing gives every router a gradient; top-k routing does not;
5. checkpoint selection keeps a better feasible input checkpoint (and the old behaviour is
   available with select_include_input=False);
6. matched alignment controls keep J_t equal to the subset rule's J_t.
"""
import numpy as np
import torch
import torch.nn.functional as F

from backbone.moss_moe import ExpertMixture
from models.moss import DEFAULTS, Learner, select_control_triples


def _moe(M=4, k=2, seed=0):
    torch.manual_seed(seed)
    return ExpertMixture(in_dim=8, num_classes=3, num_experts=M, hidden_dim=16, out_dim=4, topk=k)


def _set_router_bias(moe, biases):
    with torch.no_grad():
        for r, b in zip(moe.routers, biases):
            r.weight.zero_()
            r.bias.fill_(float(b))


def _bruteforce_loo(moe, u):
    """p^(-m) logits in float64: softmax over the active set without m, no replacement."""
    out = moe(u)
    scores, active, zs = out["scores"].double(), out["active"], out["expert_out"].double()
    W, b = moe.head.weight.double(), moe.head.bias.double()
    B, M = scores.shape
    res = torch.empty(B, M, W.shape[0], dtype=torch.float64)
    for i in range(B):
        for m in range(M):
            keep = active[i].clone()
            keep[m] = False if bool(active[i, m]) else keep[m]
            w = torch.softmax((scores[i] / moe.tau).masked_fill(~keep, float("-inf")), dim=-1)
            res[i, m] = (w @ zs[i]) @ W.T + b
    return res


def _old_closed_form(moe, out):
    pi = out["weights"].unsqueeze(-1)
    z_minus = (out["z"].unsqueeze(1) - pi * out["expert_out"]) / (1.0 - pi).clamp_min(1e-6)
    return moe.head(z_minus)


def test_leave_one_out_matches_bruteforce():
    moe = _moe()
    u = torch.randn(6, 8)
    with torch.no_grad():
        out = moe(u)
        got = moe.leave_one_out_logits(out["scores"], out["active"], out["expert_out"])
        assert torch.allclose(got.double(), _bruteforce_loo(moe, u), atol=1e-5)


def test_leave_one_out_stable_when_one_weight_is_one():
    moe = _moe()
    _set_router_bias(moe, [30.0, 0.0, -5.0, -10.0])  # pi_0 == 1.0 in float32, active = {0, 1}
    u = torch.randn(6, 8)
    with torch.no_grad():
        out = moe(u)
        assert float(out["weights"][0, 0]) == 1.0
        got = moe.leave_one_out_logits(out["scores"], out["active"], out["expert_out"])
        ref = _bruteforce_loo(moe, u)
        assert torch.allclose(got.double(), ref, atol=1e-5)
        # removing expert 0 leaves expert 1 alone: z^(-0) = z_1
        assert torch.allclose(got[:, 0], moe.head(out["expert_out"][:, 1]), atol=1e-5)
        old = _old_closed_form(moe, out)
        assert not torch.allclose(old[:, 0].double(), ref[:, 0], atol=1e-3)  # the bug the fix removes


def test_leave_one_out_legacy_signature():
    moe = _moe()
    u = torch.randn(5, 8)
    with torch.no_grad():
        out = moe(u)
        legacy = moe.leave_one_out_logits(out["z"], out["weights"], out["expert_out"])
        assert torch.allclose(legacy.double(), _bruteforce_loo(moe, u), atol=1e-5)


def test_sparse_dispatch_matches_dense():
    moe = _moe()
    u = torch.randn(4, 8)
    moe.expert_evals = 0
    dense = moe(u)
    assert moe.expert_evals == 4 * 4
    moe.expert_evals = 0
    sparse = moe(u, expert_out=False)
    assert moe.expert_evals == 4 * 2 and sparse["expert_out"] is None
    assert torch.allclose(dense["logits"], sparse["logits"], atol=1e-6)
    y = torch.tensor([0, 1, 2, 0])
    grads = []
    for kw in ({}, {"expert_out": False}):
        moe.zero_grad(set_to_none=True)
        F.cross_entropy(moe(u, **kw)["logits"], y).backward()
        grads.append([None if p.grad is None else p.grad.clone() for p in moe.parameters()])
    for gd, gs in zip(*grads):
        if gd is None or gs is None:
            assert (gd is None or float(gd.abs().max()) == 0.0) and (gs is None or float(gs.abs().max()) == 0.0)
        else:
            assert torch.allclose(gd, gs, atol=1e-6)


def test_dense_exploration_reaches_every_router():
    moe = _moe()
    _set_router_bias(moe, [2.0, 2.0, -2.0, -2.0])  # top-2 is always {0, 1}
    u, y = torch.randn(8, 8), torch.randint(0, 3, (8,))

    def router_grad_norms():
        moe.zero_grad(set_to_none=True)
        F.cross_entropy(moe(u)["logits"], y).backward()
        return [0.0 if r.weight.grad is None else float(r.weight.grad.abs().sum()) for r in moe.routers]

    sparse = router_grad_norms()
    assert sparse[2] == 0.0 and sparse[3] == 0.0 and sparse[0] > 0.0
    moe.dense_routing = True
    dense = router_grad_norms()
    moe.dense_routing = False
    assert all(g > 0.0 for g in dense)


class _StubLearner(Learner):
    """Just enough state to run Learner._run_phase with scripted checkpoint risks."""

    def __init__(self, risks, include_input=True):
        self.hp = dict(DEFAULTS, batch_size=4, eval_interval=1, wandb_log_interval=0,
                       select_include_input=include_input)
        self.cur = {"y_fit": torch.zeros(4, dtype=torch.long)}
        self._teacher, self._n_old_experts, self._train_step, self._cur_task = None, 4, 0, 0
        self.p_max = 10 ** 9
        self._risks = list(risks)

    def _loss(self, moe, ci, ri, trainable, V, use_div, use_feat, use_ssi):
        return (moe.head.bias - 1.0).pow(2).sum(), {}

    def _checkpoint_stats(self, moe):
        return self._risks.pop(0), 0.0, moe.stored_expert_router_params()


def test_selection_keeps_better_feasible_input():
    plan = [(None, None)] * 2
    moe = _moe()
    before = moe.head.bias.detach().clone()
    info = _StubLearner([0.000335, 0.018126, 0.02])._run_phase(moe, "reuse", [], plan=plan)
    assert info["selected_input"] and info["input_feasible"] and not info["fallback_to_input"]
    assert abs(info["val_risk"] - 0.000335) < 1e-12
    assert torch.equal(moe.head.bias, before)
    # earlier behaviour: the worse post-update checkpoint replaces the input
    moe = _moe()
    info = _StubLearner([0.018126, 0.02], include_input=False)._run_phase(moe, "reuse", [], plan=plan)
    assert not info["selected_input"] and abs(info["val_risk"] - 0.018126) < 1e-12
    assert not torch.equal(moe.head.bias, before)
    # the candidate branch never competes with its untrained input
    moe = _moe()
    info = _StubLearner([0.018126, 0.02])._run_phase(moe, "cand", [3], plan=plan,
                                                       allow_input_fallback=False)
    assert info["input_feasible"] is None and not info["selected_input"]


def test_matched_controls_keep_J():
    subset = [(0, 0, 1, 0.5, 0.1), (0, 1, 2, 0.3, 0.2)]          # (m, s, c, a, d)
    candidates = [(m, s, c) for m in range(3) for s in range(2) for c in range(3)]
    rng = np.random.default_rng(0)
    for mode in ("subset", "none", "global", "random"):
        triples, w, J = select_control_triples(mode, subset, candidates, True, rng)
        assert J == [0], (mode, J)
        assert all(m == 0 for m, _, _ in triples)
        assert len(w) == len(triples) and (len(w) == 0 or abs(w.sum() - 1.0) < 1e-12)
    _, _, J_global = select_control_triples("global", subset, candidates, False, rng)
    assert J_global is None                                     # caller expands to all experts
    tr, _, J_random = select_control_triples("random", subset, candidates, False, rng)
    assert len(tr) == len(subset) and J_random == sorted({m for m, _, _ in tr})


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in tests:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"{len(tests)} tests passed")


if __name__ == "__main__":
    main()
