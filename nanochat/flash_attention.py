"""
TPU attention interface with automatic Splash Attention / SDPA switching.

Exports `flash_attn` module that routes to Google's TPU Splash Attention
(JAX Pallas softmax kernel) when available and compatible, or falls back to
PyTorch SDPA (e.g. for single-token decode steps during inference).

Backend selection (in order of precedence):
    1. set_attn_impl('splash' | 'sdpa' | 'auto')
    2. NANOCHAT_ATTN_IMPL environment variable (same values)
    3. 'auto': Splash on TPU, else SDPA
"""
import os
from types import SimpleNamespace

import torch
import torch.nn.functional as F

VALID_IMPLS = ("auto", "splash", "sdpa")

_splash_ok = False
_splash_load_error = None
_splash_checked = False


def _load_splash():
    """Whether Splash Attention can run on this TPU. Cached."""
    global _splash_ok, _splash_load_error, _splash_checked
    if _splash_checked:
        return _splash_ok
    _splash_checked = True
    from nanochat.splash_attention import splash_available
    _splash_ok, _splash_load_error = splash_available()
    return _splash_ok


def has_splash():
    """Whether TorchTPU Splash Attention is importable on this machine."""
    return _load_splash()


_override_impl = None


def _requested_impl():
    if _override_impl is not None:
        return _override_impl
    env = os.environ.get("NANOCHAT_ATTN_IMPL", "auto").strip().lower() or "auto"
    assert env in VALID_IMPLS, f"Invalid NANOCHAT_ATTN_IMPL={env!r}, expected one of {VALID_IMPLS}"
    return env


def _resolve_attn_impl():
    """Decide once which attention backend to use, based on availability, override, and dtype."""
    from nanochat.common import COMPUTE_DTYPE
    requested = _requested_impl()
    if requested == "sdpa":
        return "sdpa"
    if requested == "splash":
        assert has_splash(), f"Cannot use Splash Attention: {_splash_load_error}"
        assert COMPUTE_DTYPE in (torch.bfloat16, torch.float32), \
            f"Splash Attention supports bf16/fp32, but COMPUTE_DTYPE={COMPUTE_DTYPE}"
        return "splash"
    if has_splash():
        return "splash"
    return "sdpa"


def _apply_impl(impl):
    global ATTN_IMPL, USE_SPLASH
    ATTN_IMPL = impl
    USE_SPLASH = impl == "splash"


try:
    _apply_impl(_resolve_attn_impl())
except AssertionError:
    _apply_impl("sdpa")


def set_attn_impl(impl):
    """
    Select the attention backend at runtime: 'splash', 'sdpa', or 'auto'/None.
    Must be called before the model is torch.compile'd / run. Returns the resolved backend.
    """
    global _override_impl
    if impl is not None:
        impl = impl.strip().lower()
        assert impl in VALID_IMPLS, f"Invalid attention impl {impl!r}, expected one of {VALID_IMPLS}"
    _override_impl = None if impl in (None, "auto") else impl
    _apply_impl(_resolve_attn_impl())
    return ATTN_IMPL


ATTN_IMPL_NAMES = {
    "splash": "TPU Splash Attention",
    "sdpa": "PyTorch SDPA",
}


def _sdpa_attention(q, k, v, window_size, enable_gqa, causal=True):
    """
    SDPA attention with sliding window support.
    q, k, v are (B, H, T, D) format.
    """
    Tq = q.size(2)
    Tk = k.size(2)
    window = window_size[0]

    if not causal:
        left, right = window_size
        left_unlimited = left is None or left < 0 or left >= Tk - 1
        right_unlimited = right is None or right < 0 or right >= Tk - 1
        if left_unlimited and right_unlimited:
            return F.scaled_dot_product_attention(q, k, v, is_causal=False, enable_gqa=enable_gqa)
        device = q.device
        row_idx = (Tk - Tq) + torch.arange(Tq, device=device).unsqueeze(1)
        col_idx = torch.arange(Tk, device=device).unsqueeze(0)
        mask = torch.ones((Tq, Tk), dtype=torch.bool, device=device)
        if not left_unlimited:
            mask = mask & ((row_idx - col_idx) <= left)
        if not right_unlimited:
            mask = mask & ((col_idx - row_idx) <= right)
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=enable_gqa)

    # Full context, same length
    if (window < 0 or window >= Tq) and Tq == Tk:
        return F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=enable_gqa)

    # Single token generation
    if Tq == 1:
        if window >= 0 and window < Tk:
            start = max(0, Tk - (window + 1))
            k = k[:, :, start:, :]
            v = v[:, :, start:, :]
        return F.scaled_dot_product_attention(q, k, v, is_causal=False, enable_gqa=enable_gqa)

    # Need explicit mask for sliding window/chunk inference
    device = q.device
    row_idx = (Tk - Tq) + torch.arange(Tq, device=device).unsqueeze(1)
    col_idx = torch.arange(Tk, device=device).unsqueeze(0)
    mask = col_idx <= row_idx

    if window >= 0 and window < Tk:
        mask = mask & ((row_idx - col_idx) <= window)

    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=enable_gqa)


def _sdpa_bthd(q, k, v, window_size, causal=True, bhtd=False):
    """SDPA on ``(B, T, H, D)`` (or ``(B, H, T, D)`` when ``bhtd=True``). Returns ``(B, T, H, D)``."""
    if not bhtd:
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
    enable_gqa = q.size(1) != k.size(1)
    y = _sdpa_attention(q, k, v, window_size, enable_gqa, causal=causal)
    return y.transpose(1, 2)


_splash_fallback_warned = False


def _splash_bthd(q, k, v, causal, window_size, warn_on_fallback=True, bhtd=False):
    """Splash Attention on ``(B, T, H, D)`` (or ``(B, H, T, D)`` when ``bhtd=True``).

    Shapes the kernel cannot tile (not a multiple of 128, including decode
    steps of length 1) fall back to SDPA. Returns ``(B, T, H, D)``.
    """
    global _splash_fallback_warned
    from nanochat.splash_attention import splash_local_window, splash_sdpa, splash_shapes_ok
    if bhtd:
        q_len, kv_len, head_dim = q.size(2), k.size(2), q.size(3)
        n_q, n_kv = q.size(1), k.size(1)
    else:
        q_len, kv_len, head_dim = q.size(1), k.size(1), q.size(3)
        n_q, n_kv = q.size(2), k.size(2)
    shapes_ok = splash_shapes_ok(q_len, kv_len, head_dim) and (n_q % n_kv == 0)
    if not causal:
        left, right = window_size
        window_limited = (left is not None and left >= 0) or (right is not None and right >= 0)
        if window_limited or not shapes_ok:
            return _sdpa_bthd(q, k, v, window_size, causal=False, bhtd=bhtd)
        local_window = None
    elif not shapes_ok:
        if warn_on_fallback and not _splash_fallback_warned:
            _splash_fallback_warned = True
            from nanochat.common import print0
            print0(
                f"Splash Attention needs sequence lengths and head_dim that are multiples of 128 "
                f"(got q={q_len}, kv={kv_len}, head_dim={head_dim}); using PyTorch SDPA for this shape."
            )
        return _sdpa_bthd(q, k, v, window_size, causal=True, bhtd=bhtd)
    else:
        local_window = splash_local_window(window_size, kv_len)
    if not bhtd:
        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()
    y = splash_sdpa(
        q,
        k,
        v,
        is_causal=causal,
        local_window_size=local_window,
        enable_gqa=n_q != n_kv,
    )
    return y.transpose(1, 2)


def flash_attn_func(q, k, v, causal=False, window_size=(-1, -1), bhtd=False):
    """
    Attention for training (no KV cache).

    Args:
        q, k, v: Tensors of shape (B, T, H, D) (or (B, H, T, D) when bhtd=True)
        causal: Whether to use causal masking
        window_size: (left, right) sliding window. -1 means unlimited.
        bhtd: If True, q, k, v are already in contiguous (B, H, T, D) layout.

    Returns:
        Output tensor of shape (B, T, H, D)
    """
    if ATTN_IMPL == "splash":
        return _splash_bthd(q, k, v, causal, window_size, bhtd=bhtd)
    return _sdpa_bthd(q, k, v, window_size, causal=causal, bhtd=bhtd)


def flash_attn_with_kvcache(q, k_cache, v_cache, k=None, v=None, cache_seqlens=None,
                            causal=False, window_size=(-1, -1)):
    """
    Attention with KV cache for inference.

    Args:
        q: Queries, shape (B, T_new, H, D)
        k_cache, v_cache: Pre-allocated cache tensors, shape (B, T_max, H_kv, D)
        k, v: New keys/values to insert, shape (B, T_new, H_kv, D)
        cache_seqlens: Current position in cache, shape (B,) int32
        causal: Whether to use causal masking
        window_size: (left, right) sliding window. -1 means unlimited.

    Returns:
        Output tensor of shape (B, T_new, H, D)
    """
    T_new = q.size(1)
    pos = cache_seqlens[0].item()  # assume uniform position across batch

    if k is not None and v is not None:
        k_cache[:, pos:pos+T_new, :, :] = k
        v_cache[:, pos:pos+T_new, :, :] = v

    end_pos = pos + T_new
    k_full = k_cache[:, :end_pos, :, :]
    v_full = v_cache[:, :end_pos, :, :]

    if ATTN_IMPL == "splash":
        return _splash_bthd(q, k_full, v_full, causal, window_size, warn_on_fallback=False)
    return _sdpa_bthd(q, k_full, v_full, window_size, causal=causal)


flash_attn = SimpleNamespace(
    flash_attn_func=flash_attn_func,
    flash_attn_with_kvcache=flash_attn_with_kvcache,
)
