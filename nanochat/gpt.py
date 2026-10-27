"""
GPT model (rewrite, a lot simpler)
Notable features:
- rotary embeddings (and no positional embeddings)
- QK norm
- untied weights for token embedding and lm_head
- relu^2 activation in MLP
- norm after token embedding
- no learnable params in rmsnorm
- no bias in linear layers
- Group-Query Attention (GQA) support for more efficient inference
- TPU Splash Attention integration
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from nanochat.common import print0, COMPUTE_DTYPE
from nanochat.optim import MuonAdamW
from nanochat.flash_attention import flash_attn

@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6 # number of query heads
    n_kv_head: int = 6 # number of key/value heads (GQA)
    n_embd: int = 768
    # Sliding window attention pattern string, tiled across layers. Final layer always L.
    # Characters: L=long (full context), S=short (quarter context)
    # Examples: "L"=all full context, "SL"=alternating, "SSL"=two short then one long
    window_pattern: str = "SSSL"


class _RMSNormFn(torch.autograd.Function):
    """Memory-efficient RMSNorm (with optional output scale).

    Saves the normalized output ``y`` and ``rstd`` instead of the pre-norm input ``x``.
    When ``y`` is consumed by downstream Linear layers or Splash Attention, AOTAutograd
    deduplicates the saved ``y`` tensor so the pre-norm input ``x`` is never saved
    across forward to backward.
    """

    @staticmethod
    def forward(ctx, x, scale: float = 1.0):
        eps = torch.finfo(torch.float32).eps
        x_f = x.float()
        rstd = torch.rsqrt(x_f.square().mean(dim=-1, keepdim=True) + eps)
        y = (x_f * rstd).to(dtype=x.dtype)
        if scale != 1.0:
            y = y * scale
        ctx.save_for_backward(y, rstd)
        ctx.scale = scale
        return y

    @staticmethod
    def backward(ctx, grad_out):
        y, rstd = ctx.saved_tensors
        scale = ctx.scale
        g_f = grad_out.float()
        y_unit = y.float() * (1.0 / scale) if scale != 1.0 else y.float()
        grad_y_norm = g_f * scale if scale != 1.0 else g_f
        grad_x = rstd * (grad_y_norm - y_unit * (grad_y_norm * y_unit).mean(dim=-1, keepdim=True))
        return grad_x.to(dtype=y.dtype), None


def norm(x, scale: float = 1.0):
    if torch.is_grad_enabled():
        return _RMSNormFn.apply(x, scale)
    y = F.rms_norm(x, (x.size(-1),))
    return y * scale if scale != 1.0 else y


class _CastLinearFn(torch.autograd.Function):
    """Linear layer with fp32 master weights cast to activation dtype in forward and backward.

    Using a custom autograd Function prevents AOTAutograd's min-cut partitioner
    from saving cast bf16 weight copies (1.27 GiB on d24) as forward-to-backward
    activations alongside the fp32 master weights. Also accepts 4D ``(B, T, H, D)``
    views (such as attention outputs) so ``c_proj`` shares saved storage with the
    attention output tensor without saving a second contiguous copy.
    """

    @staticmethod
    def forward(ctx, x, weight):
        ctx.save_for_backward(x, weight)
        x_2d = x.contiguous().view(-1, weight.size(1))
        out = F.linear(x_2d, weight.to(dtype=x.dtype))
        return out.view(*x.shape[:2], weight.size(0)) if x.ndim > 2 else out

    @staticmethod
    def backward(ctx, grad_out):
        x, weight = ctx.saved_tensors
        dtype = x.dtype
        w_t = weight.t().to(dtype=dtype)
        go_2d = grad_out.reshape(-1, weight.size(0))
        x_2d = x.reshape(-1, weight.size(1))
        grad_x = (go_2d @ w_t.t()).view_as(x)
        grad_w = (go_2d.t() @ x_2d).to(dtype=weight.dtype)
        return grad_x, grad_w


class Linear(nn.Linear):
    """nn.Linear that casts weights to match input dtype in forward.
    Replaces autocast: master weights stay fp32 for optimizer precision,
    but matmuls run in the activation dtype (typically bf16 from embeddings)."""
    def forward(self, x):
        if torch.is_grad_enabled():
            return _CastLinearFn.apply(x, self.weight)
        x_in = x.contiguous().view(*x.shape[:2], -1) if x.ndim > 3 else x
        return F.linear(x_in, self.weight.to(dtype=x.dtype))


def has_ve(layer_idx, n_layer):
    """Returns True if GPT layer should have Value Embedding (alternating, last layer always included)."""
    return layer_idx % 2 == (n_layer - 1) % 2

def apply_rotary_emb(x, cos, sin):
    # note: this rotates by -theta, the transpose of the textbook convention. Functionally
    # equivalent (only the relative q/k rotation matters), kept for checkpoint compatibility.
    assert x.ndim == 4  # multihead attention
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:] # split up last dim into two halves
    y1 = x1 * cos + x2 * sin # rotate pairs of dims
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)

class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        self.c_q = Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = Linear(self.n_embd, self.n_embd, bias=False)
        self.ve_gate_channels = 12
        self.ve_gate = Linear(self.ve_gate_channels, self.n_kv_head, bias=False) if has_ve(layer_idx, config.n_layer) else None

    def project_qkv(self, x, ve, cos_sin, bhtd=False):
        B, T, _ = x.size()

        # Project the input to get queries, keys, and values
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        # Value residual (ResFormer): mix in value embedding with input-dependent gate per head
        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 3 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))  # (B, T, n_kv_head), range (0, 3)
            v = v + gate.unsqueeze(-1) * ve

        # Apply Rotary Embeddings to queries and keys to get relative positional encoding
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        if bhtd:
            q = q.transpose(1, 2).contiguous()
            k = k.transpose(1, 2).contiguous()
            v = v.transpose(1, 2).contiguous()
        q = norm(q, scale=1.2)  # QK norm + 1.2x sharper attention
        k = norm(k, scale=1.2)
        return q, k, v

    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        _, T, _ = x.size()
        bhtd = kv_cache is None
        q, k, v = self.project_qkv(x, ve, cos_sin, bhtd=bhtd)

        # Attention (TPU Splash Attention or SDPA fallback)
        # window_size is (left, right) tuple: (N, 0) for causal, (-1, 0) for full context
        if kv_cache is None:
            # Training: causal attention with optional sliding window
            y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size, bhtd=True)
        else:
            # Inference: use flash_attn_with_kvcache which handles cache management
            k_cache, v_cache = kv_cache.get_layer_cache(self.layer_idx)
            y = flash_attn.flash_attn_with_kvcache(
                q, k_cache, v_cache,
                k=k, v=v,
                cache_seqlens=kv_cache.cache_seqlens,
                causal=True,
                window_size=window_size,
            )
            # Advance position after last layer processes
            if self.layer_idx == kv_cache.n_layers - 1:
                kv_cache.advance(T)

        # Project heads back to residual stream (Linear handles (B, T, H, D) views directly)
        y = self.c_proj(y)
        return y


class _FusedNormMLP(torch.autograd.Function):
    """Fused residual + RMSNorm + ReLU^2 MLP with custom backward.

    Computes ``x + c_proj(relu(c_fc(rms_norm(x)))^2)`` without saving ``rms_norm(x)``
    or ``relu(h)^2`` across forward to backward. When ``recompute_cfc=True``, only
    ``x`` (B, T, D) is saved (recomputing ``c_fc`` in backward), reducing MLP saved
    activations by 8x without triggering ``tpu._optimization_barrier`` or recomputing
    Splash Attention.
    """

    @staticmethod
    def forward(ctx, x, w_fc, w_proj, recompute_cfc: bool):
        B, T, D = x.size()
        dtype = x.dtype
        x_norm = F.rms_norm(x, (D,))
        h = F.linear(x_norm, w_fc.to(dtype))
        z = F.relu(h).square()
        out = x + F.linear(z, w_proj.to(dtype))
        ctx.recompute_cfc = recompute_cfc
        if recompute_cfc:
            ctx.save_for_backward(x, w_fc, w_proj)
        else:
            ctx.save_for_backward(x, h, w_fc, w_proj)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        if ctx.recompute_cfc:
            x, w_fc, w_proj = ctx.saved_tensors
            h = None
        else:
            x, h, w_fc, w_proj = ctx.saved_tensors
        B, T, D = x.size()
        dtype = x.dtype
        w_fc_t = w_fc.t().to(dtype)
        w_proj_t = w_proj.t().to(dtype)

        eps = torch.finfo(torch.float32).eps
        x_f = x.float()
        rstd = torch.rsqrt(x_f.square().mean(dim=-1, keepdim=True) + eps)
        x_norm_f = x_f * rstd
        xn_2d = x_norm_f.to(dtype).reshape(B * T, D)

        if h is None:
            h = (xn_2d @ w_fc_t).view(B, T, -1)
        relu_h = torch.clamp_min(h, 0)
        z = relu_h * relu_h

        go_2d = grad_out.reshape(B * T, D)
        z_2d = z.reshape(B * T, -1)
        grad_w_proj = (go_2d.t() @ z_2d).to(w_proj.dtype)
        grad_z = (go_2d @ w_proj_t.t()).view_as(z)
        grad_h = (grad_z * (relu_h + relu_h)).to(dtype)
        gh_2d = grad_h.reshape(B * T, -1)
        grad_w_fc = (gh_2d.t() @ xn_2d).to(w_fc.dtype)
        grad_x_norm = (gh_2d @ w_fc_t.t()).view(B, T, D).float()
        grad_x = grad_out + (rstd * (grad_x_norm - x_norm_f * (grad_x_norm * x_norm_f).mean(dim=-1, keepdim=True))).to(dtype)
        return grad_x, grad_w_fc, grad_w_proj, None


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, ve, cos_sin, window_size, kv_cache, recompute_mlp=False):
        x = x + self.attn(norm(x), ve, cos_sin, window_size, kv_cache)
        if torch.is_grad_enabled():
            x = _FusedNormMLP.apply(x, self.mlp.c_fc.weight, self.mlp.c_proj.weight, recompute_mlp)
        else:
            x = x + self.mlp(norm(x))
        return x

    def forward_checkpointed(self, x, x0, resid_lambda, x0_lambda, idx, ve_embed, cos_sin, window_size, full_recompute=False):
        """Activation-checkpointed block forward for training.

        When ``full_recompute=False`` (default for B*T <= 8192), checkpoints the MLP
        via ``_FusedNormMLP`` while keeping attention activations in memory — eliminating
        all Splash Attention forward recomputations and ``tpu._optimization_barrier`` ops.
        When ``full_recompute=True``, checkpoints the entire block so only ``x`` is saved.
        """
        if not full_recompute:
            x_s = resid_lambda * x + x0_lambda * x0
            ve = ve_embed(idx).to(x_s.dtype) if ve_embed is not None else None
            return self.forward(x_s, ve, cos_sin, window_size, None, recompute_mlp=True)

        def _block(x_in, x0_in, resid_l, x0_l, idx_in):
            x_s = resid_l * x_in + x0_l * x0_in
            ve = ve_embed(idx_in).to(x_s.dtype) if ve_embed is not None else None
            return self.forward(x_s, ve, cos_sin, window_size, None, recompute_mlp=True)

        return checkpoint(_block, x, x0, resid_lambda, x0_lambda, idx, use_reentrant=False)


def _linear_softcap_cross_entropy_fwd(x_2d, weight, targets_1d, vocab_size, softcap, reduction):
    w = (weight[:vocab_size] if vocab_size < weight.size(0) else weight).to(dtype=x_2d.dtype)
    num_chunks = min(2, max(1, x_2d.size(0) // 4096))
    if num_chunks <= 1:
        logits = F.linear(x_2d, w).float()
        logits = softcap * torch.tanh(logits / softcap)
        return F.cross_entropy(logits, targets_1d, ignore_index=-1, reduction=reduction)
    chunk_red = 'none' if reduction == 'none' else 'sum'
    chunk_losses = []
    for x_c, t_c in zip(x_2d.chunk(num_chunks), targets_1d.chunk(num_chunks)):
        logits = F.linear(x_c, w).float()
        logits = softcap * torch.tanh(logits / softcap)
        chunk_losses.append(F.cross_entropy(logits, t_c, ignore_index=-1, reduction=chunk_red))
    if reduction == 'none':
        return torch.cat(chunk_losses, dim=0)
    loss = torch.stack(chunk_losses).sum()
    if reduction == 'mean':
        loss = loss / (targets_1d != -1).sum().clamp(min=1)
    return loss


def _linear_softcap_cross_entropy_bwd_chunk(x_c, w_t, t_c, vocab_arange, softcap, scale_c):
    logits_raw = (x_c @ w_t).float()
    t = torch.tanh(logits_raw / softcap)
    probs = F.softmax(softcap * t, dim=-1)
    valid_mask = t_c != -1
    safe_targets = torch.where(valid_mask, t_c, torch.zeros_like(t_c))
    one_hot = (safe_targets.unsqueeze(1) == vocab_arange).to(torch.float32)
    dlogits = (probs - one_hot) * (1.0 - t * t) * scale_c.unsqueeze(1)
    dlogits_bf16 = dlogits.to(dtype=x_c.dtype)
    grad_x_c = dlogits_bf16 @ w_t.t()
    grad_w_c = (dlogits_bf16.t() @ x_c).to(dtype=torch.float32)
    return grad_x_c, grad_w_c


class _LinearSoftcapCrossEntropy(torch.autograd.Function):
    """Fused linear + logit softcap + cross-entropy loss with memory-efficient backward.

    Saves only ``x_2d`` (B*T, D), ``weight``, and ``targets_1d`` (B*T,) from forward
    to backward instead of ``(B*T, vocab_size)`` fp32 logits, and computes ``grad_x``
    and ``grad_weight`` without activation-checkpoint optimization barriers.
    """

    @staticmethod
    def forward(ctx, x_2d, weight, targets_1d, vocab_size: int, softcap: float, reduction: str):
        ctx.save_for_backward(x_2d, weight, targets_1d)
        ctx.vocab_size = vocab_size
        ctx.softcap = float(softcap)
        ctx.reduction = reduction
        return _linear_softcap_cross_entropy_fwd(x_2d, weight, targets_1d, vocab_size, softcap, reduction)

    @staticmethod
    def backward(ctx, grad_out):
        x_2d, weight, targets_1d = ctx.saved_tensors
        vocab_size = ctx.vocab_size
        softcap = ctx.softcap
        reduction = ctx.reduction

        w_slice = weight[:vocab_size] if vocab_size < weight.size(0) else weight
        w_t = w_slice.t().to(dtype=x_2d.dtype)

        valid_f = (targets_1d != -1).to(torch.float32)
        if reduction == 'mean':
            scale = (grad_out.float() / valid_f.sum().clamp(min=1)) * valid_f
        elif reduction == 'sum':
            scale = grad_out.float() * valid_f
        else:
            scale = grad_out.float().view(-1) * valid_f

        vocab_arange = torch.arange(vocab_size, device=targets_1d.device).unsqueeze(0)
        num_chunks = max(1, x_2d.size(0) // 8192)
        if num_chunks <= 1:
            grad_x, grad_w_f32 = _linear_softcap_cross_entropy_bwd_chunk(
                x_2d, w_t, targets_1d, vocab_arange, softcap, scale
            )
        else:
            grad_x_chunks = []
            grad_w_f32 = None
            for x_c, t_c, s_c in zip(x_2d.chunk(num_chunks), targets_1d.chunk(num_chunks), scale.chunk(num_chunks)):
                gx_c, gw_c = _linear_softcap_cross_entropy_bwd_chunk(
                    x_c, w_t, t_c, vocab_arange, softcap, s_c
                )
                grad_x_chunks.append(gx_c)
                grad_w_f32 = gw_c if grad_w_f32 is None else (grad_w_f32 + gw_c)
            grad_x = torch.cat(grad_x_chunks, dim=0)

        grad_w_valid = grad_w_f32.to(dtype=weight.dtype)
        if vocab_size < weight.size(0):
            grad_w = torch.zeros_like(weight)
            grad_w[:vocab_size] = grad_w_valid
        else:
            grad_w = grad_w_valid
        return grad_x, grad_w, None, None, None, None


class GPT(nn.Module):
    def __init__(self, config, pad_vocab_size_to=64):
        """
        NOTE a major footgun: this __init__ function runs in meta device context (!!)
        Therefore, any calculations inside here are shapes and dtypes only, no actual data.
        => We actually initialize all data (parameters, buffers, etc.) in init_weights() instead.
        """
        super().__init__()
        self.config = config
        self.grad_checkpoint = False
        # Compute per-layer window sizes for sliding window attention
        # window_size is (left, right) tuple: (-1, 0) for full context, (N, 0) for sliding window
        self.window_sizes = self._compute_window_sizes(config)
        # Pad vocab for efficiency (DDP, tensor cores). This is just an optimization - outputs are cropped in forward().
        # https://huggingface.co/docs/transformers/main_classes/model#transformers.PreTrainedModel.resize_token_embeddings
        padded_vocab_size = ((config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
        if padded_vocab_size != config.vocab_size:
            print0(f"Padding vocab_size from {config.vocab_size} to {padded_vocab_size} for efficiency")
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(padded_vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config, layer_idx) for layer_idx in range(config.n_layer)]),
        })
        self.lm_head = Linear(config.n_embd, padded_vocab_size, bias=False)
        # Per-layer learnable scalars (inspired by modded-nanogpt)
        # resid_lambdas: scales the residual stream at each layer (init 1.0 = neutral)
        # x0_lambdas: blends initial embedding back in at each layer (init 0.0 = disabled)
        # Separate parameters so they can have different optimizer treatment
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))   # fake init, real init in init_weights()
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))     # fake init, real init in init_weights()
        # Smear: mix previous token's embedding into current token (cheap bigram-like info)
        self.smear_gate = Linear(24, 1, bias=False)
        self.smear_lambda = nn.Parameter(torch.zeros(1))
        # Backout: subtract cached mid-layer residual before final norm to remove low-level features
        self.backout_lambda = nn.Parameter(0.2 * torch.ones(1))
        # Value embeddings (ResFormer-style): alternating layers, last layer always included
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = nn.ModuleDict({str(i): nn.Embedding(padded_vocab_size, kv_dim) for i in range(config.n_layer) if has_ve(i, config.n_layer)})
        # To support meta device initialization, we init the rotary embeddings here, but it's just "fake" meta tensors only.
        # As for rotary_seq_len, these rotary embeddings are pretty small/cheap in memory,
        # so let's just over-compute them by 10X, but assert fail if we ever reach that amount.
        # In the future we can dynamically grow the cache, for now it's fine.
        self.rotary_seq_len = config.sequence_len * 10 # 10X over-compute should be enough, TODO make nicer?
        head_dim = config.n_embd // config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False) # persistent=False means it's not saved to the checkpoint
        self.register_buffer("sin", sin, persistent=False)

    @torch.no_grad()
    def init_weights(self):
        """
        Initialize the full model in this one function for maximum clarity.

        wte (embedding):     normal, std=1.0
        lm_head:             normal, std=0.001
        for each block:
            attn.c_q:        uniform, std=1/sqrt(n_embd)
            attn.c_k:        uniform, std=1/sqrt(n_embd)
            attn.c_v:        uniform, std=1/sqrt(n_embd)
            attn.c_proj:     zeros
            mlp.c_fc:        uniform, std=1/sqrt(n_embd)
            mlp.c_proj:      zeros
        """

        # Embedding and unembedding
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=0.8)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)

        # Transformer blocks: uniform init with bound = sqrt(3) * std (same standard deviation as normal)
        n_embd = self.config.n_embd
        s = 3**0.5 * n_embd**-0.5 # sqrt(3) multiplier makes sure Uniform achieves the same std as Normal
        for block in self.transformer.h:
            torch.nn.init.uniform_(block.attn.c_q.weight, -s, s) # weights use Uniform to avoid outliers
            torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            torch.nn.init.zeros_(block.attn.c_proj.weight) # projections are zero
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -s * 0.4, s * 0.4)  # 0.4x init scale for c_fc
            torch.nn.init.zeros_(block.mlp.c_proj.weight)

        # Per-layer scalars
        # Per-layer resid init: stronger residual at early layers, weaker at deep layers
        n_layer = self.config.n_layer
        for i in range(n_layer):
            self.resid_lambdas.data[i] = 1.15 - (0.10 * i / max(n_layer - 1, 1))
        # Decaying x0 init: earlier layers get more input embedding blending
        for i in range(n_layer):
            self.x0_lambdas.data[i] = 0.20 - (0.15 * i / max(n_layer - 1, 1))

        # Smear/backout scalars and smear gate must be explicitly initialized 
        torch.nn.init.zeros_(self.smear_lambda)
        torch.nn.init.constant_(self.backout_lambda, 0.2)
        torch.nn.init.uniform_(self.smear_gate.weight, 0.0, 0.02)

        # Value embeddings (init like c_v: uniform with same std)
        for ve in self.value_embeds.values():
            torch.nn.init.uniform_(ve.weight, -s, s)

        # Gate weights init with small positive values so gates start slightly above neutral
        for block in self.transformer.h:
            if block.attn.ve_gate is not None:
                torch.nn.init.uniform_(block.attn.ve_gate.weight, 0.0, 0.02)

        # Rotary embeddings
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin

        # Cast embeddings to COMPUTE_DTYPE (bf16 on TPU) to save memory
        self.transformer.wte.to(dtype=COMPUTE_DTYPE)
        for ve in self.value_embeds.values():
            ve.to(dtype=COMPUTE_DTYPE)

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=100000, device=None):
        # TODO: bump base theta more? e.g. 100K is more common more recently
        # autodetect the device from model embeddings
        if device is None:
            device = self.transformer.wte.weight.device
        # stride the channels
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        # stride the time steps
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        # calculate the rotation frequencies at each (time, channel) pair
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.to(COMPUTE_DTYPE), sin.to(COMPUTE_DTYPE)
        cos, sin = cos[None, :, None, :], sin[None, :, None, :] # add batch and head dims for later broadcasting
        return cos, sin

    def _compute_window_sizes(self, config):
        """
        Compute per-layer window sizes for sliding window attention.

        Returns list of (left, right) tuples for FA3's window_size parameter:
        - left: how many tokens before current position to attend to (-1 = unlimited)
        - right: how many tokens after current position to attend to (0 for causal)

        Pattern string is tiled across layers. Final layer always gets L (full context).
        Characters: L=long (full context), S=short (quarter context)
        """
        pattern = config.window_pattern.upper()
        assert all(c in "SL" for c in pattern), f"Invalid window_pattern: {pattern}. Use only S and L."
        # Map characters to window sizes
        long_window = config.sequence_len
        short_window = -(-long_window // 4 // 128) * 128  # ceil to FA3 tile size (2048 -> 768)
        char_to_window = {
            "L": (long_window, 0),
            "S": (short_window, 0),
        }
        # Tile pattern across layers
        window_sizes = []
        for layer_idx in range(config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])
        # Final layer always gets full context
        window_sizes[-1] = (long_window, 0)
        return window_sizes

    def get_device(self):
        return self.transformer.wte.weight.device

    def estimate_flops(self):
        """
        Return the estimated FLOPs per token for the model (forward + backward).
        Each matmul weight parameter contributes 2 FLOPs (multiply *, accumulate +) in forward, and 2X that in backward => 2+4=6.
        Cleanest explanation of this: https://medium.com/@dzmitrybahdanau/the-flops-calculus-of-language-model-training-3b19c1f025e4
        On top of that, 12 * h * q * effective_seq_len accounts for key @ query matmul flops inside attention.
        With sliding windows, effective_seq_len varies per layer (capped by window size).
        Ref: https://arxiv.org/abs/2204.02311 (PaLM paper).
        This is ~1% off from the exact formulas of Chinchilla paper, the difference is:
        - Chinchilla counts the embedding layer as flops (? weird, it's just a lookup => we ignore)
        - Chinchilla counts exp/sum/divide in attention softmax as flops (a little sus and very tiny => we ignore)
        """
        h, q, t = self.config.n_head, self.config.n_embd // self.config.n_head, self.config.sequence_len
        # Sum attention FLOPs per layer, accounting for sliding window
        attn_flops = 0
        for window_size in self.window_sizes:
            window = window_size[0]  # (left, right) tuple, we use left
            effective_seq = t if window < 0 else min(window, t)
            attn_flops += 12 * h * q * effective_seq
        num_flops_per_token = 6 * self.num_matmul_params() + attn_flops
        return num_flops_per_token

    def num_matmul_params(self):
        """
        The number of parameters that participate in matmuls with the token stream,
        i.e. contribute 2 FLOPs/param to the forward pass. Counted structurally: every
        matmul in this model goes through an nn.Linear subclass (Linear or Float8Linear),
        while non-matmul params (embeddings = lookups, per-layer scalars) are nn.Embedding
        or raw Parameters.
        """
        matmul_params = sum(m.weight.numel() for m in self.modules() if isinstance(m, nn.Linear))
        return matmul_params

    def estimate_decode_flops(self, context_len):
        """
        Forward FLOPs to decode one token at a given context length during inference:
        2 FLOPs per matmul param, plus attention over min(context, window) per layer.
        """
        h = self.config.n_head
        q = self.config.n_embd // self.config.n_head
        attn_flops = sum(4 * h * q * min(context_len, window) for window, _ in self.window_sizes)
        decode_flops = 2 * self.num_matmul_params() + attn_flops
        return decode_flops

    def estimate_prefill_flops(self, num_tokens):
        """Forward FLOPs to prefill a prompt: causal, so token t attends to min(t, window)."""
        h = self.config.n_head
        q = self.config.n_embd // self.config.n_head
        attn_flops = 0
        for window, _ in self.window_sizes:
            w = min(window, num_tokens)
            attended_tokens = w * (w + 1) // 2 + (num_tokens - w) * w # ramp up to w, then flat
            attn_flops += 4 * h * q * attended_tokens
        prefill_flops = 2 * self.num_matmul_params() * num_tokens + attn_flops
        return prefill_flops

    def kv_bytes_per_token(self):
        """Bytes to *store* one token of KV cache during inference, per row (all layers)."""
        head_dim = self.config.n_embd // self.config.n_head
        kv_dtype_bytes = COMPUTE_DTYPE.itemsize # the KV cache is kept in the compute dtype
        return self.config.n_layer * 2 * self.config.n_kv_head * head_dim * kv_dtype_bytes

    def kv_read_bytes(self, context_len):
        """Bytes of KV cache *read* by one decode step at a given context length, per row.
        Sliding window layers only attend to (and read) the last `window` tokens."""
        head_dim = self.config.n_embd // self.config.n_head
        kv_dtype_bytes = COMPUTE_DTYPE.itemsize
        total = 0
        for window, _ in self.window_sizes:
            total += 2 * self.config.n_kv_head * head_dim * kv_dtype_bytes * min(context_len, window)
        return total

    def num_scaling_params(self):
        """
        Return detailed parameter counts for scaling law analysis.
        Different papers use different conventions:
        - Kaplan et al. excluded embedding parameters
        - Chinchilla included all parameters
        Ref: https://arxiv.org/abs/2203.15556 (Chinchilla paper)
        Ref: https://arxiv.org/abs/2001.08361 (Kaplan et al. original scaling laws paper)

        Returns a dict with counts for each parameter group, so downstream analysis
        can experiment with which combination gives the cleanest scaling laws.
        """
        # Count each group separately (mirrors the grouping in setup_optimizers)
        wte = sum(p.numel() for p in self.transformer.wte.parameters())
        value_embeds = sum(p.numel() for p in self.value_embeds.parameters())
        lm_head = sum(p.numel() for p in self.lm_head.parameters())
        transformer_matrices = sum(p.numel() for p in self.transformer.h.parameters())
        scalars = self.resid_lambdas.numel() + self.x0_lambdas.numel() + self.smear_gate.weight.numel() + self.smear_lambda.numel() + self.backout_lambda.numel()
        total = wte + value_embeds + lm_head + transformer_matrices + scalars
        assert total == sum(p.numel() for p in self.parameters()), "Parameter count mismatch"
        return {
            'wte': wte,
            'value_embeds': value_embeds,
            'lm_head': lm_head,
            'transformer_matrices': transformer_matrices,
            'scalars': scalars,
            'total': total,
        }

    def setup_optimizer(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0, scalar_lr=0.5):
        model_dim = self.config.n_embd

        # Separate out all parameters into groups
        matrix_params = list(self.transformer.h.parameters())
        value_embeds_params = list(self.value_embeds.parameters())
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        resid_params = [self.resid_lambdas]
        x0_params = [self.x0_lambdas]
        smear_params = [self.smear_gate.weight, self.smear_lambda, self.backout_lambda]
        assert len(list(self.parameters())) == len(matrix_params) + len(embedding_params) + len(lm_head_params) + len(value_embeds_params) + len(resid_params) + len(x0_params) + len(smear_params)

        # Scale the LR for the AdamW parameters by ∝1/√dmodel (tuned for 768 dim model)
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print0(f"Scaling the LR for the AdamW parameters ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}")

        # Build param_groups with all required fields explicit
        param_groups = [
            # AdamW groups (embeddings, lm_head, scalars)
            dict(kind='adamw', params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale, betas=(0.8, 0.96), eps=1e-10, weight_decay=0.01),
            dict(kind='adamw', params=embedding_params, lr=embedding_lr * dmodel_lr_scale, betas=(0.8, 0.995), eps=1e-10, weight_decay=0.001),
            dict(kind='adamw', params=value_embeds_params, lr=embedding_lr * dmodel_lr_scale * 0.5, betas=(0.8, 0.995), eps=1e-10, weight_decay=0.01),
            dict(kind='adamw', params=resid_params, lr=scalar_lr * 0.01, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.05),
            dict(kind='adamw', params=x0_params, lr=scalar_lr, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),  # higher beta1 for x0
            dict(kind='adamw', params=smear_params, lr=0.2, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.0),
        ]
        # Muon groups (matrix params, grouped by shape for stacking)
        for shape in sorted({p.shape for p in matrix_params}):
            group_params = [p for p in matrix_params if p.shape == shape]
            param_groups.append(dict(
                kind='muon', params=group_params, lr=matrix_lr,
                momentum=0.95, ns_steps=5, beta2=0.9, weight_decay=weight_decay,
            ))

        optimizer = MuonAdamW(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction='mean', loss_scale=None):
        B, T = idx.size()

        # Grab the rotary embeddings for the current sequence length (they are of shape (1, seq_len, 1, head_dim/2))
        assert T <= self.cos.size(1), f"Sequence length grew beyond the rotary embeddings cache: {T} > {self.cos.size(1)}"
        assert idx.device == self.cos.device, f"Rotary embeddings and idx are on different devices: {idx.device} != {self.cos.device}"
        assert self.cos.dtype == COMPUTE_DTYPE, f"Rotary embeddings must be in {COMPUTE_DTYPE}, got {self.cos.dtype}"
        # if kv cache exists, we need to offset the rotary embeddings to the current position in the cache
        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        cos_sin = self.cos[:, T0:T0+T], self.sin[:, T0:T0+T] # truncate cache to current sequence length

        # Embed the tokens
        x = self.transformer.wte(idx) # embed current token
        x = x.to(COMPUTE_DTYPE) # ensure activations are in compute dtype (no-op usually, but active for fp16 code path)
        x = norm(x)

        # Smear: mix previous token's embedding into current position (cheap bigram info)
        if kv_cache is None:
            # Training / naive generate: full sequence available, use fast slice
            assert T > 1, "Training forward pass should have T > 1"
            gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[:, 1:, :24]))
            x = torch.cat([x[:, :1], x[:, 1:] + gate * x[:, :-1]], dim=1)
        else:
            # KV cache inference: read prev embedding from cache, store current for next step
            x_pre_smear = kv_cache.prev_embedding
            kv_cache.prev_embedding = x[:, -1:, :]
            if T > 1:
                # Prefill: apply smear to positions 1+, same as training
                gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[:, 1:, :24]))
                x = torch.cat([x[:, :1], x[:, 1:] + gate * x[:, :-1]], dim=1)
            elif x_pre_smear is not None:
                # Decode: single token, use cached prev embedding
                gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[:, :, :24]))
                x = x + gate * x_pre_smear

        # Forward the trunk of the Transformer
        x0 = x  # save initial normalized embedding for x0 residual
        n_layer = self.config.n_layer
        backout_layer = n_layer // 2  # cache at halfway point
        x_backout = None
        use_ckpt = bool(self.grad_checkpoint) and self.training and kv_cache is None and torch.is_grad_enabled()
        full_recompute = (self.grad_checkpoint == "full") or (B * T > 8192)
        for i, block in enumerate(self.transformer.h):
            ve_embed = self.value_embeds[str(i)] if str(i) in self.value_embeds else None
            if use_ckpt:
                x = block.forward_checkpointed(
                    x, x0, self.resid_lambdas[i], self.x0_lambdas[i],
                    idx, ve_embed, cos_sin, self.window_sizes[i],
                    full_recompute=full_recompute,
                )
            else:
                x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
                ve = ve_embed(idx).to(x.dtype) if ve_embed is not None else None
                x = block(x, ve, cos_sin, self.window_sizes[i], kv_cache)
            if i == backout_layer:
                x_backout = x
        # Subtract mid-layer residual to remove low-level features before logit projection
        if x_backout is not None:
            x = x - self.backout_lambda.to(x.dtype) * x_backout
        x = norm(x)

        # Forward the lm_head (compute logits)
        softcap = 15 # smoothly cap the logits to the range [-softcap, softcap]
        if targets is not None:
            # training / eval loss: use custom autograd Function when grad is enabled so
            # full (B, T, vocab_size) fp32 logits + softcap tensors are never saved across
            # forward to backward, and backward runs without activation-checkpoint barriers.
            x_flat = x.view(-1, x.size(-1))
            t_flat = targets.view(-1)
            if torch.is_grad_enabled():
                loss = _LinearSoftcapCrossEntropy.apply(
                    x_flat, self.lm_head.weight, t_flat,
                    self.config.vocab_size, softcap, loss_reduction,
                )
            else:
                loss = _linear_softcap_cross_entropy_fwd(
                    x_flat, self.lm_head.weight, t_flat,
                    self.config.vocab_size, softcap, loss_reduction,
                )
            if loss_scale is not None:
                loss = loss * loss_scale
            return loss
        else:
            # inference: just return the logits directly
            logits = self.lm_head(x) # (B, T, padded_vocab_size)
            logits = logits[..., :self.config.vocab_size] # slice to remove padding
            logits = logits.float() # switch to fp32 for logit softcap
            logits = softcap * torch.tanh(logits / softcap) # squash the logits
            return logits

    @torch.inference_mode()
    def generate(self, tokens, max_tokens, temperature=1.0, top_k=None, seed=42):
        """
        Naive autoregressive streaming inference.
        To make it super simple, let's assume:
        - batch size is 1
        - ids and the yielded tokens are simple Python lists and ints
        """
        assert isinstance(tokens, list)
        device = self.get_device()
        rng = None
        if temperature > 0:
            rng = torch.Generator(device=device)
            rng.manual_seed(seed)
        ids = torch.tensor([tokens], dtype=torch.long, device=device) # add batch dim
        for _ in range(max_tokens):
            logits = self.forward(ids) # (B, T, vocab_size)
            logits = logits[:, -1, :] # (B, vocab_size)
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            if temperature > 0:
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                next_ids = torch.multinomial(probs, num_samples=1, generator=rng)
            else:
                next_ids = torch.argmax(logits, dim=-1, keepdim=True)
            ids = torch.cat((ids, next_ids), dim=1)
            token = next_ids.item()
            yield token
