"""
Prompt-capable ViT for NoRGa / HiDe-Prompt style prefix tuning.

The prefix-attention math is ported from ``NoRGa_Attention`` in
https://github.com/Minhchuyentoancbn/MoE_PromptCL/blob/master/attention.py
(MIT License, see third_party/MoE_PromptCL/LICENSE):

    A_prompt_hat = A_prompt + alpha * act(tau * A_prompt)        (NoRGa, Eq. 15)
    attention    = softmax([A_prompt_hat, A_pretrain]) @ [V_prompt, V]

Differences from the original module (the math is unchanged):
  * it reuses the frozen timm ``Attention`` weights instead of owning a copy, so the
    pre-trained backbone is loaded exactly once;
  * prefixes and (tau, alpha) are given per sample, so one batch can mix samples routed
    to different task prompts at inference time.

With ``act_scale=None`` the function is plain prefix tuning (HiDe-Prompt / DualPrompt
``PreT_Attention``), which is what ``"norga": false`` runs.
"""
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F


def _identity(x):
    return x


GATE_ACTS = {
    "tanh": torch.tanh,
    "sigmoid": torch.sigmoid,
    "gelu": F.gelu,
    "relu": F.relu,
    "silu": F.silu,
    "identity": _identity,
}


def get_gate_act(name):
    key = str(name).lower()
    if key not in GATE_ACTS:
        raise ValueError(f"Unknown gate_act '{name}'. Choose from {sorted(GATE_ACTS)}.")
    return GATE_ACTS[key]


def prefix_attention(attn, x, prefix=None, act_scale=None, gate_act=torch.tanh):
    """Multi-head self-attention with optional (NoRGa-gated) key/value prefixes.

    Args:
        attn: a timm ``Attention`` module; its qkv / proj / dropout / scale are reused.
        x: (B, N, C) tokens after ``norm1``.
        prefix: None or (B, 2, Lp, H, Dh) key (index 0) and value (index 1) prefixes.
        act_scale: None, or (B, 2) holding [tau, alpha] per sample. None gives plain
            prefix tuning.
        gate_act: the non-linearity sigma in Eq. 15.
    """
    B, N, C = x.shape
    H = attn.num_heads
    Dh = C // H
    qkv = attn.qkv(x).reshape(B, N, 3, H, Dh).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)                                # each B, H, N, Dh
    q_norm = getattr(attn, "q_norm", None)                 # present in newer timm only
    k_norm = getattr(attn, "k_norm", None)
    if q_norm is not None:
        q = q_norm(q)
    if k_norm is not None:
        k = k_norm(k)
    scale = getattr(attn, "scale", Dh ** -0.5)

    scores = (q @ k.transpose(-2, -1)) * scale             # B, H, N, N
    if prefix is not None:
        k_prefix = prefix[:, 0].permute(0, 2, 1, 3)        # B, H, Lp, Dh
        v_prefix = prefix[:, 1].permute(0, 2, 1, 3)        # B, H, Lp, Dh
        p_scores = (q @ k_prefix.transpose(-2, -1)) * scale  # B, H, N, Lp
        if act_scale is not None:
            tau = act_scale[:, 0].reshape(B, 1, 1, 1)
            alpha = act_scale[:, 1].reshape(B, 1, 1, 1)
            p_scores = p_scores + gate_act(p_scores * tau) * alpha
        scores = torch.cat([p_scores, scores], dim=-1)     # B, H, N, Lp + N
        v = torch.cat([v_prefix, v], dim=2)

    weights = attn.attn_drop(scores.softmax(dim=-1))
    out = (weights @ v).transpose(1, 2).reshape(B, N, C)
    return attn.proj_drop(attn.proj(out))


class PromptViT(nn.Module):
    """Wraps a timm VisionTransformer and injects prefixes into selected blocks.

    ``forward(x)`` without prompts reproduces ``vit(x)`` (pooled features, head removed);
    tests/test_norga.py checks this against the installed timm version.
    """

    def __init__(self, vit, prompt_layers):
        super().__init__()
        self.vit = vit
        n_blocks = len(vit.blocks)
        layers = [int(layer) for layer in prompt_layers]
        bad = [layer for layer in layers if not 0 <= layer < n_blocks]
        if bad:
            raise ValueError(f"prompt_layers {bad} out of range for a {n_blocks}-block ViT")
        if len(set(layers)) != len(layers):
            raise ValueError(f"prompt_layers contains duplicates: {layers}")
        self.prompt_layers = layers
        self._slot = {layer: i for i, layer in enumerate(layers)}
        self.embed_dim = vit.embed_dim
        self.num_heads = vit.blocks[0].attn.num_heads
        self.head_dim = self.embed_dim // self.num_heads

    def _embed(self, x):
        vit = self.vit
        x = vit.patch_embed(x)
        if hasattr(vit, "_pos_embed"):
            x = vit._pos_embed(x)
        else:  # older timm
            cls = vit.cls_token.expand(x.shape[0], -1, -1)
            x = vit.pos_drop(torch.cat([cls, x], dim=1) + vit.pos_embed)
        norm_pre = getattr(vit, "norm_pre", None)
        if norm_pre is not None:
            x = norm_pre(x)
        return x

    def _pool(self, x):
        vit = self.vit
        if getattr(vit, "global_pool", "token") == "avg":
            x = x[:, getattr(vit, "num_prefix_tokens", 1):].mean(dim=1)
        else:
            x = x[:, 0]
        fc_norm = getattr(vit, "fc_norm", None)
        if fc_norm is not None:
            x = fc_norm(x)
        pre_logits = getattr(vit, "pre_logits", None)
        if isinstance(pre_logits, nn.Module):  # representation layer in old timm
            x = pre_logits(x)
        return x

    @staticmethod
    def _prompted_block(blk, x, prefix, act_scale, gate_act):
        def _get(*names):
            for name in names:
                mod = getattr(blk, name, None)
                if mod is not None:
                    return mod
            return _identity

        ls1, ls2 = _get("ls1"), _get("ls2")
        dp1, dp2 = _get("drop_path1", "drop_path"), _get("drop_path2", "drop_path")
        x = x + dp1(ls1(prefix_attention(blk.attn, blk.norm1(x), prefix, act_scale, gate_act)))
        x = x + dp2(ls2(blk.mlp(blk.norm2(x))))
        return x

    def forward(self, x, prompts=None, act_scales=None, gate_act=torch.tanh):
        """
        Args:
            prompts: None (uninstructed forward) or (B, n_prompt_layers, 2, Lp, H, Dh).
            act_scales: None (plain prefix tuning) or (B, n_prompt_layers, 2).
        Returns:
            (B, embed_dim) pooled features (what the classification head consumes).
        """
        x = self._embed(x)
        for i, blk in enumerate(self.vit.blocks):
            slot = self._slot.get(i) if prompts is not None else None
            if slot is None:
                x = blk(x)
            else:
                scale = None if act_scales is None else act_scales[:, slot]
                x = self._prompted_block(blk, x, prompts[:, slot], scale, gate_act)
        x = self.vit.norm(x)
        return self._pool(x)


def build_vit(args):
    """Create the frozen backbone with timm.

    Keys: ``backbone_type`` (timm name), ``pretrained`` (default True),
    ``pretrained_path`` (optional local checkpoint: .npz as released by Google, e.g. the
    Sup-21K ViT-B_16.npz used by NoRGa, or a torch state dict), ``backbone_kwargs``.
    """
    import timm

    name = args.get("backbone_type", "vit_base_patch16_224_in21k")
    path = args.get("pretrained_path")
    pretrained = bool(args.get("pretrained", True)) and not path
    kwargs = dict(args.get("backbone_kwargs") or {})
    kwargs.setdefault("num_classes", 0)
    kwargs.setdefault("drop_path_rate", 0.0)
    vit = timm.create_model(name, pretrained=pretrained, **kwargs)

    if path:
        if str(path).endswith(".npz"):
            vit.load_pretrained(path)
            logging.info(f"[NoRGa] loaded npz backbone weights from {path}")
        else:
            state = torch.load(path, map_location="cpu")
            for key in ("model", "state_dict", "teacher"):
                if isinstance(state, dict) and isinstance(state.get(key), dict):
                    state = state[key]
                    break
            cleaned = {}
            for k, v in state.items():
                for prefix in ("module.", "backbone."):
                    if k.startswith(prefix):
                        k = k[len(prefix):]
                if k.startswith(("head.", "fc.")):
                    continue
                cleaned[k] = v
            missing, unexpected = vit.load_state_dict(cleaned, strict=False)
            logging.info(f"[NoRGa] loaded backbone weights from {path}; "
                         f"missing={len(missing)} unexpected={len(unexpected)}")
            if missing:
                logging.warning(f"[NoRGa] missing backbone keys (first 10): {missing[:10]}")
    return vit
