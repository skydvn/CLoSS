"""
Tests for the NoRGa integration. CPU only; needs torch and timm, downloads nothing.

    python3 -m tests.test_norga

1. the prompt-free wrapper, and a prompted block with an empty prefix, reproduce timm's
   forward (guards against timm version drift);
2. prefix_attention matches the official NoRGa_Attention math (copied verbatim below);
3. alpha = 0 reduces NoRGa to plain prefix tuning;
4. a batch mixing task prompts equals per-task forwards;
5. Gaussian statistics reproduce the data moments (full, diag, multi-centroid);
6. the contrastive regulariser is finite, differentiable and ignores same-label prototypes;
7. two-task CIL and DIL smoke runs through the Learner API that trainer.py calls.
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

import timm

from backbone.norga_vit import PromptViT, prefix_attention
from models.norga import DEFAULTS, Learner, NoRGaNet
from utils.norga_stats import GaussianBank, contrastive_reg


class RefNoRGaAttention(nn.Module):
    """Verbatim forward of NoRGa_Attention from MoE_PromptCL/attention.py (MIT License)."""

    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, prompt, act_scale, gate_act):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        if prompt is not None:
            prompt = prompt.permute(1, 0, 3, 2, 4).contiguous()
            key_prefix = prompt[0]
            value_prefix = prompt[1]
            v = torch.cat([value_prefix, v], dim=2)
            prompt_attn = (q @ key_prefix.transpose(-2, -1)) * self.scale
            prompt_attn = (prompt_attn + gate_act(prompt_attn * act_scale[0]) * act_scale[1])
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = torch.cat([prompt_attn, attn], dim=-1)
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


TINY = {"backbone_type": "vit_tiny_patch16_224", "pretrained": False,
        "backbone_kwargs": {"img_size": 32}}


def tiny_vit():
    torch.manual_seed(0)
    return timm.create_model("vit_tiny_patch16_224", pretrained=False, num_classes=0,
                             img_size=32, drop_path_rate=0.0).eval()


def small_cfg(**kw):
    cfg = dict(DEFAULTS)
    cfg.update(prompt_layers=[0, 1], prompt_length=3, ca_batch_size=8)
    cfg.update(kw)
    return cfg


def test_no_prompt_matches_timm():
    vit = tiny_vit()
    wrapper = PromptViT(vit, [0, 1])
    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        assert torch.allclose(wrapper(x), vit(x), atol=1e-5)


def test_prompted_block_matches_timm_block():
    # an empty prefix must leave every prompted block identical to timm's Block.forward
    vit = tiny_vit()
    wrapper = PromptViT(vit, [0, 1])
    H, Dh = wrapper.num_heads, wrapper.head_dim
    x = torch.randn(2, 3, 32, 32)
    empty = torch.zeros(2, 2, 2, 0, H, Dh)
    scales = torch.ones(2, 2, 2)
    with torch.no_grad():
        assert torch.allclose(wrapper(x, empty, scales), vit(x), atol=1e-5)


def test_matches_reference_norga():
    torch.manual_seed(0)
    attn = tiny_vit().blocks[0].attn
    C, H = attn.qkv.in_features, attn.num_heads
    ref = RefNoRGaAttention(C, H, qkv_bias=attn.qkv.bias is not None)
    ref.load_state_dict(attn.state_dict(), strict=False)
    assert torch.equal(ref.qkv.weight, attn.qkv.weight)
    B, N = 2, 5
    x = torch.randn(B, N, C)
    prompt = torch.randn(B, 2, 4, H, C // H)
    tau, alpha = torch.tensor(0.7), torch.tensor(1.3)
    with torch.no_grad():
        expected = ref(x, prompt, (tau, alpha), torch.tanh)
        got = prefix_attention(attn, x, prompt, torch.stack([tau, alpha]).expand(B, 2), torch.tanh)
        assert torch.allclose(got, expected, atol=1e-5)
        # without prompts both reduce to the standard attention block
        assert torch.allclose(prefix_attention(attn, x), attn(x), atol=1e-5)


def test_alpha_zero_is_plain_prefix_tuning():
    torch.manual_seed(0)
    attn = tiny_vit().blocks[0].attn
    C, H = attn.qkv.in_features, attn.num_heads
    x, prompt = torch.randn(2, 5, C), torch.randn(2, 2, 4, H, C // H)
    zero_alpha = torch.tensor([[3.0, 0.0], [3.0, 0.0]])
    with torch.no_grad():
        assert torch.allclose(prefix_attention(attn, x, prompt, zero_alpha),
                              prefix_attention(attn, x, prompt, None), atol=1e-6)


def test_mixed_task_batch_matches_separate():
    net = NoRGaNet(tiny_vit(), small_cfg(), nb_tasks=2, nb_classes=4, scenario="cil").eval()
    with torch.no_grad():
        net.act_scales[1].fill_(0.5)
    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        mixed = net.features(x, torch.tensor([0, 1]))
        first = net.features(x[:1], torch.tensor([0]))
        second = net.features(x[1:], torch.tensor([1]))
    assert torch.allclose(mixed, torch.cat([first, second]), atol=1e-5)


def test_gaussian_bank_moments():
    gen = torch.Generator().manual_seed(0)
    A = torch.randn(4, 4, generator=gen)
    cov = A @ A.T + 0.1 * torch.eye(4)
    dist = torch.distributions.MultivariateNormal(torch.full((4,), 3.0), cov)
    x = dist.sample((6000,))
    for cov_type in ("full", "diag"):
        bank = GaussianBank(cov_type, shrink=1e-6)
        bank.add(x, label=7, task=0)
        s, y = bank.sample(6000, generator=torch.Generator().manual_seed(1))
        assert bool((y == 7).all())
        assert torch.allclose(s.mean(0), x.mean(0), atol=0.15)
        tol = 0.15 * float(cov.diagonal().max())
        if cov_type == "full":
            assert torch.allclose(torch.cov(s.T), torch.cov(x.T), atol=tol)
        else:
            assert torch.allclose(s.var(0), x.var(0), atol=tol)
    # two well separated modes: multi-centroid sampling keeps each mode tight
    modes = torch.cat([torch.randn(500, 2, generator=gen) * 0.1 - 5,
                       torch.randn(500, 2, generator=gen) * 0.1 + 5])
    bank = GaussianBank("full", shrink=1e-6, n_centroids=2)
    bank.add(modes, label=0, task=0)
    s, _ = bank.sample(2000, generator=torch.Generator().manual_seed(2))
    assert float((s.abs() - 5).abs().mean()) < 0.3


def test_contrastive_reg():
    torch.manual_seed(0)
    feats = torch.randn(8, 16, requires_grad=True)
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    protos, proto_labels = torch.randn(5, 16), torch.tensor([4, 5, 6, 7, 8])
    loss = contrastive_reg(feats, labels, protos, proto_labels, 0.8)
    loss.backward()
    assert torch.isfinite(loss) and feats.grad is not None
    # prototypes sharing every anchor's label are ignored entirely
    same = contrastive_reg(feats.detach(), torch.zeros(8, dtype=torch.long),
                           torch.randn(3, 16), torch.zeros(3, dtype=torch.long), 0.8)
    alone = contrastive_reg(feats.detach(), torch.zeros(8, dtype=torch.long), None, None, 0.8)
    assert torch.allclose(same, alone)


class _TensorDS(Dataset):
    def __init__(self, x, y):
        self.x, self.y = x, y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return i, self.x[i], self.y[i]


class FakeCILDataManager:
    """Mimics the PILOT DataManager calls used by the learner."""

    def __init__(self, n_classes=4, task_size=2, per_class=6):
        gen = torch.Generator().manual_seed(0)
        self.nb_classes, self.task_size = n_classes, task_size
        self.nb_tasks = n_classes // task_size
        y = torch.arange(n_classes).repeat_interleave(per_class)
        self.data = {s: (torch.randn(len(y), 3, 32, 32, generator=gen) + y.view(-1, 1, 1, 1), y)
                     for s in ("train", "test")}

    def get_task_size(self, task):
        return self.task_size

    def get_dataset(self, indices, source, mode):
        x, y = self.data[source]
        keep = torch.from_numpy(np.isin(y.numpy(), indices))
        return _TensorDS(x[keep], y[keep])


class FakeDILDataManager:
    """One label space; each task is a domain (here a different input offset)."""

    def __init__(self, n_classes=3, n_domains=2, per_class=6):
        gen = torch.Generator().manual_seed(0)
        self.nb_classes, self.nb_tasks = n_classes, n_domains
        y = torch.arange(n_classes).repeat_interleave(per_class)
        self.data = {(d, s): (torch.randn(len(y), 3, 32, 32, generator=gen) + 3 * d, y)
                     for d in range(n_domains) for s in ("train", "test")}

    def get_task_dataset(self, task, source, mode):
        return _TensorDS(*self.data[(task, source)])


def _base_args(**kw):
    args = {"device": [torch.device("cpu")], "seed": 0, "model_name": "norga",
            "prompt_layers": [0, 1], "prompt_length": 3, "wtp_epochs": 1, "wtp_batch_size": 4,
            "tii_epochs": 1, "tii_ca_epochs": 1, "ca_epochs": 1, "ca_samples_per_class": 8,
            "num_workers": 0, "eval_batch_size": 8, "stats_cov_type": "diag",
            "eval_oracle": True, **TINY}
    args.update(kw)
    return args


def _run(learner, dm):
    for t in range(dm.nb_tasks):
        learner.incremental_train(dm)
        accy, nme = learner.eval_task()
        learner.after_task()
        assert nme is None and "top1" in accy and f"top{learner.topk}" in accy
        assert len([k for k in accy["grouped"] if "-" in k]) == t + 1
        if t == 0:
            frozen = learner._network.prompts[0].detach().clone()
    assert torch.equal(frozen, learner._network.prompts[0].detach())


def test_cil_smoke():
    dm = FakeCILDataManager()
    learner = Learner(_base_args(scenario="cil", nb_tasks=dm.nb_tasks, nb_classes=dm.nb_classes,
                                 init_cls=2, increment=2))
    _run(learner, dm)


def test_dil_smoke():
    dm = FakeDILDataManager()
    learner = Learner(_base_args(scenario="dil", nb_tasks=dm.nb_tasks, nb_classes=dm.nb_classes,
                                 init_cls=dm.nb_classes, increment=dm.nb_classes))
    _run(learner, dm)


def test_hideprompt_switch():
    dm = FakeCILDataManager()
    learner = Learner(_base_args(model_name="hideprompt", scenario="cil", nb_tasks=dm.nb_tasks,
                                 nb_classes=dm.nb_classes, init_cls=2, increment=2))
    assert learner.cfg["norga"] is False and learner._network.use_norga is False


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in tests:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"{len(tests)} tests passed")


if __name__ == "__main__":
    main()
