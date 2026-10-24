"""
Unified attention interface with automatic FA3/FA4/FA2/Splash/SDPA switching.

Exports `flash_attn` module that matches the FA3 API exactly, but can route to
Flash Attention 4 (`kernels-community/flash-attn4`, CuTe DSL), Flash Attention 2
(`kernels-community/flash-attn2`), Google's TPU Splash Attention (JAX Pallas
softmax kernel, not AdaSplash), or fall back to PyTorch SDPA on incompatible
CUDA GPUs, MPS, and CPU.

Backend selection (in order of precedence):
    1. set_attn_impl('fa4' | 'fa3' | 'fa2' | 'splash' | 'sdpa' | 'auto')
    2. NANOCHAT_ATTN_IMPL environment variable                  (same values)
    3. 'auto': FA3 (sm80/sm90), else FA4 (sm100/sm110), else FA2, else FA4,
       else Splash on TPU, else SDPA

Usage (drop-in replacement for FA3):
    from nanochat.flash_attention import flash_attn

    # Training (no KV cache)
    y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)

    # Inference (with KV cache)
    y = flash_attn.flash_attn_with_kvcache(q, k_cache, v_cache, k=k, v=v, ...)
"""
import os

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import torch
import torch.nn.functional as F

VALID_IMPLS = ("auto", "fa4", "fa3", "fa2", "splash", "sdpa")


# =============================================================================
# Detection: Try to load FA3 on CUDA GPUs
# =============================================================================
def _load_flash_attention_3():
    """Try to load Flash Attention 3."""
    if not torch.cuda.is_available():
        return None
    try:
        major, _ = torch.cuda.get_device_capability()
        # FA3 kernels are compiled for Hopper (sm90) and Ampere/Ada (sm80/sm86/sm89)
        if major not in (8, 9):
            return None
        from kernels import get_kernel, has_kernel
        # The varunneal kernel obtains better results for H100/Hopper
        if major == 9:
            return get_kernel("varunneal/flash-attention-3", revision="main", trust_remote_code=True).flash_attn_interface
        hf_kernel = "kernels-community/flash-attn3"
        for version in (2, 1):
            if has_kernel(hf_kernel, version=version):
                return get_kernel(hf_kernel, version=version).flash_attn_interface
        return None
    except Exception:
        return None


_fa3 = _load_flash_attention_3()
HAS_FA3 = _fa3 is not None


# =============================================================================
# Detection: Try to load FA2 via kernels-community/flash-attn2 (lazy, cached)
# =============================================================================
_fa2 = None
_fa2_load_error = None
_fa2_checked = False


def _load_flash_attention_2():
    """Try to load Flash Attention 2 (kernels-community/flash-attn2). Cached."""
    global _fa2, _fa2_load_error, _fa2_checked
    if _fa2_checked:
        return _fa2
    _fa2_checked = True
    if not torch.cuda.is_available():
        _fa2_load_error = "CUDA is not available"
        return None
    try:
        from kernels import get_kernel, has_kernel
        hf_kernel = "kernels-community/flash-attn2"
        for version in (3, 2, 1):
            if has_kernel(hf_kernel, version=version):
                _fa2 = get_kernel(hf_kernel, version=version).flash_attn_interface
                return _fa2
        _fa2_load_error = "no compatible kernels-community/flash-attn2 build for this GPU/PyTorch version"
    except Exception as e:
        _fa2_load_error = f"failed to load kernels-community/flash-attn2: {type(e).__name__}: {e}"
        _fa2 = None
    return _fa2


def has_fa2():
    """Whether FA2 (kernels-community/flash-attn2) is available on this hardware."""
    return _load_flash_attention_2() is not None


# =============================================================================
# Detection: Try to load FA4 via kernels-community/flash-attn4 (lazy, cached)
# =============================================================================
_fa4 = None            # the flash-attn4 interface module, once loaded
_fa4_load_error = None  # reason FA4 could not be loaded (for error messages)
_fa4_checked = False


def _load_flash_attention_4():
    """Try to load Flash Attention 4 (kernels-community/flash-attn4, CuTe DSL). Cached."""
    global _fa4, _fa4_load_error, _fa4_checked
    if _fa4_checked:
        return _fa4
    _fa4_checked = True
    if not torch.cuda.is_available():
        _fa4_load_error = "CUDA is not available"
        return None
    major, minor = torch.cuda.get_device_capability()
    # FA4 CuTe kernels support sm80 (Ampere), sm90 (Hopper), sm100/sm110 (Blackwell DC), sm120 (Blackwell GeForce)
    if major not in (8, 9, 10, 11, 12):
        _fa4_load_error = f"unsupported compute capability sm{major}{minor}"
        return None
    try:
        from kernels import get_kernel, has_kernel
        hf_kernel = "kernels-community/flash-attn4"
        if has_kernel(hf_kernel, version=1):
            _fa4 = get_kernel(hf_kernel, version=1).interface
        else:
            _fa4_load_error = "no compatible kernels-community/flash-attn4 build found"
    except Exception as e:
        _fa4_load_error = f"failed to load kernels-community/flash-attn4: {type(e).__name__}: {e}"
        _fa4 = None
    return _fa4


def has_fa4():
    """Whether FA4 is installed and loadable on this hardware (triggers the lazy import)."""
    return _load_flash_attention_4() is not None


# =============================================================================
# Detection: Splash Attention on TPU (lazy, cached)
# =============================================================================
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


# =============================================================================
# Backend resolution
# =============================================================================
# Override (e.g. for testing or CLI): 'fa4', 'fa3', 'fa2', 'sdpa', or None (auto / env var)
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
    if requested == "fa3":
        assert HAS_FA3, "Cannot use FA3: not available on this hardware"
        assert COMPUTE_DTYPE == torch.bfloat16, \
            f"FA3 only supports bf16, but COMPUTE_DTYPE={COMPUTE_DTYPE}"
        return "fa3"
    if requested == "fa2":
        assert has_fa2(), f"Cannot use FA2: {_fa2_load_error}"
        assert COMPUTE_DTYPE in (torch.bfloat16, torch.float16), \
            f"FA2 only supports bf16/fp16, but COMPUTE_DTYPE={COMPUTE_DTYPE}"
        return "fa2"
    if requested == "fa4":
        assert has_fa4(), f"Cannot use FA4: {_fa4_load_error}"
        assert COMPUTE_DTYPE in (torch.bfloat16, torch.float16), \
            f"FA4 only supports bf16/fp16, but COMPUTE_DTYPE={COMPUTE_DTYPE}"
        return "fa4"
    if requested == "splash":
        assert has_splash(), f"Cannot use Splash Attention: {_splash_load_error}"
        assert COMPUTE_DTYPE in (torch.bfloat16, torch.float32, torch.float16), \
            f"Splash Attention supports bf16/fp32/fp16, but COMPUTE_DTYPE={COMPUTE_DTYPE}"
        return "splash"
    # auto
    # FA3 Hopper kernels only support bf16 and fp8; fp16/fp32 must use another backend
    if HAS_FA3 and COMPUTE_DTYPE == torch.bfloat16:
        return "fa3"
    if COMPUTE_DTYPE in (torch.bfloat16, torch.float16):
        # On datacenter Blackwell (sm100/sm110), FA4 uses native tcgen05/TMEM instructions;
        # on consumer Blackwell (sm120) and older, FA2 C++ kernels are faster when available.
        major = torch.cuda.get_device_capability()[0] if torch.cuda.is_available() else 0
        if major in (10, 11) and has_fa4():
            return "fa4"
        if has_fa2():
            return "fa2"
        if has_fa4():
            return "fa4"
    if has_splash():
        return "splash"
    return "sdpa"


def _apply_impl(impl):
    global ATTN_IMPL, USE_FA3, USE_FA4, USE_FA2, USE_SPLASH
    ATTN_IMPL = impl
    USE_FA3 = impl == "fa3"
    USE_FA4 = impl == "fa4"
    USE_FA2 = impl == "fa2"
    USE_SPLASH = impl == "splash"


def _resolve_use_fa3():
    """Backwards-compatible helper: whether the resolved backend is FA3."""
    _apply_impl(_resolve_attn_impl())
    return USE_FA3


try:
    _apply_impl(_resolve_attn_impl())
except AssertionError:
    # Defer hard failure on invalid/unavailable NANOCHAT_ATTN_IMPL until set_attn_impl()
    # so CLI --attn-impl can take precedence over the environment variable.
    _apply_impl("sdpa")


def set_attn_impl(impl):
    """
    Select the attention backend at runtime: 'fa4', 'fa3', 'fa2', 'splash', 'sdpa', or 'auto'/None.
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
    "fa3": "Flash Attention 3",
    "fa4": "Flash Attention 4",
    "fa2": "Flash Attention 2",
    "splash": "TPU Splash Attention",
    "sdpa": "PyTorch SDPA",
}


# =============================================================================
# FA4 helpers
# =============================================================================
# FA4's autograd.Function is not traceable by torch.compile, so we expose its forward/backward
# kernels as opaque torch.library custom ops. This keeps the compiled graph free of graph breaks.
@torch.library.custom_op("nanochat::fa4_fwd", mutates_args=())
def _fa4_fwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool,
             window_left: int | None, window_right: int | None) -> tuple[torch.Tensor, torch.Tensor]:
    res = _fa4._flash_attn_fwd(
        q, k, v, causal=causal,
        window_size_left=window_left, window_size_right=window_right,
        return_lse=True,
    )
    return res[0], res[1]  # out (B, T, H, D), lse (B, H, T) fp32


@_fa4_fwd.register_fake
def _(q, k, v, causal, window_left, window_right):
    B, T, H, _ = q.shape
    out = q.new_empty(B, T, H, v.shape[-1])
    lse = q.new_empty(B, H, T, dtype=torch.float32)
    return out, lse


@torch.library.custom_op("nanochat::fa4_bwd", mutates_args=())
def _fa4_bwd(dout: torch.Tensor, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
             out: torch.Tensor, lse: torch.Tensor, causal: bool,
             window_left: int | None, window_right: int | None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    res = _fa4._flash_attn_bwd(
        q, k, v, out, dout, lse, causal=causal,
        window_size_left=window_left, window_size_right=window_right,
    )
    return res[0], res[1], res[2]


@_fa4_bwd.register_fake
def _(dout, q, k, v, out, lse, causal, window_left, window_right):
    return torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)


def _fa4_setup_context(ctx, inputs, output):
    q, k, v, causal, window_left, window_right = inputs
    out, lse = output
    ctx.save_for_backward(q, k, v, out, lse)
    ctx.causal = causal
    ctx.window_left = window_left
    ctx.window_right = window_right


def _fa4_backward(ctx, dout, dlse):
    q, k, v, out, lse = ctx.saved_tensors
    dq, dk, dv = _fa4_bwd(dout.contiguous(), q, k, v, out, lse, ctx.causal, ctx.window_left, ctx.window_right)
    return dq, dk, dv, None, None, None


_fa4_fwd.register_autograd(_fa4_backward, setup_context=_fa4_setup_context)


def _fa4_window(window_size, Tk):
    """
    Convert FA3-style window_size (left, right) with -1 = unlimited into FA4-style (None = unlimited).
    A left window that covers the whole key sequence is dropped so FA4 uses its faster pure-causal path.
    """
    left, right = window_size
    left = None if (left is None or left < 0 or left >= Tk - 1) else left
    right = None if (right is None or right < 0) else right
    return left, right


def _fa4_attention(q, k, v, causal, window_size):
    """q, k, v are (B, T, H, D). Returns (B, Tq, H, D)."""
    left, right = _fa4_window(window_size, k.size(1))
    if causal:
        right = None  # implied by causal; lets FA4 pick the pure-causal kernel when left is None too
    out, _ = _fa4_fwd(q, k, v, causal, left, right)
    return out


# =============================================================================
# SDPA helpers
# =============================================================================
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
            # window is "left" tokens we need to include (window + 1) keys total
            start = max(0, Tk - (window + 1))
            k = k[:, :, start:, :]
            v = v[:, :, start:, :]
        return F.scaled_dot_product_attention(q, k, v, is_causal=False, enable_gqa=enable_gqa)

    # Need explicit mask for sliding window/chunk inference
    device = q.device
    # For chunk inference (Tq != Tk), is_causal is not aligned to cache position => build an explicit bool mask
    row_idx = (Tk - Tq) + torch.arange(Tq, device=device).unsqueeze(1)
    col_idx = torch.arange(Tk, device=device).unsqueeze(0)
    mask = col_idx <= row_idx

    # sliding window (left)
    if window >= 0 and window < Tk:
        mask = mask & ((row_idx - col_idx) <= window)

    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=enable_gqa)


def _sdpa_bthd(q, k, v, window_size, causal=True):
    """SDPA on FA layout ``(B, T, H, D)``. Returns the same layout."""
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    enable_gqa = q.size(1) != k.size(1)
    y = _sdpa_attention(q, k, v, window_size, enable_gqa, causal=causal)
    return y.transpose(1, 2)


_splash_fallback_warned = False


def _splash_bthd(q, k, v, causal, window_size, warn_on_fallback=True):
    """Splash Attention on FA layout ``(B, T, H, D)``.

    Shapes the kernel cannot tile (not a multiple of 128, including decode
    steps of length 1) fall back to SDPA. That path is exact; it is just
    slower, and it is what inference uses between prefill chunks.
    """
    global _splash_fallback_warned
    from nanochat.splash_attention import splash_local_window, splash_sdpa, splash_shapes_ok
    q_len, kv_len, head_dim = q.size(1), k.size(1), q.size(3)
    shapes_ok = splash_shapes_ok(q_len, kv_len, head_dim) and (q.size(2) % k.size(2) == 0)
    if not causal:
        left, right = window_size
        window_limited = (left is not None and left >= 0) or (right is not None and right >= 0)
        if window_limited or not shapes_ok:
            return _sdpa_bthd(q, k, v, window_size, causal=False)
        local_window = None
    elif not shapes_ok:
        if warn_on_fallback and not _splash_fallback_warned:
            _splash_fallback_warned = True
            from nanochat.common import print0
            print0(
                f"Splash Attention needs sequence lengths and head_dim that are multiples of 128 "
                f"(got q={q_len}, kv={kv_len}, head_dim={head_dim}); using PyTorch SDPA for this shape."
            )
        return _sdpa_bthd(q, k, v, window_size, causal=True)
    else:
        local_window = splash_local_window(window_size, kv_len)
    y = splash_sdpa(
        q.transpose(1, 2).contiguous(),
        k.transpose(1, 2).contiguous(),
        v.transpose(1, 2).contiguous(),
        is_causal=causal,
        local_window_size=local_window,
        enable_gqa=q.size(2) != k.size(2),
    )
    return y.transpose(1, 2)


# =============================================================================
# Public API: Same interface as FA3
# =============================================================================
def flash_attn_func(q, k, v, causal=False, window_size=(-1, -1)):
    """
    Flash Attention for training (no KV cache).

    Args:
        q, k, v: Tensors of shape (B, T, H, D)
        causal: Whether to use causal masking
        window_size: (left, right) sliding window. -1 means unlimited.

    Returns:
        Output tensor of shape (B, T, H, D)
    """
    if ATTN_IMPL == "fa3":
        return _fa3.flash_attn_func(q, k, v, causal=causal, window_size=window_size)
    if ATTN_IMPL == "fa2":
        return _fa2.flash_attn_func(q, k, v, causal=causal, window_size=window_size)
    if ATTN_IMPL == "fa4":
        return _fa4_attention(q, k, v, causal, window_size)
    if ATTN_IMPL == "splash":
        return _splash_bthd(q, k, v, causal, window_size)

    return _sdpa_bthd(q, k, v, window_size, causal=causal)


def flash_attn_with_kvcache(q, k_cache, v_cache, k=None, v=None, cache_seqlens=None,
                            causal=False, window_size=(-1, -1)):
    """
    Flash Attention with KV cache for inference.

    FA3 and FA2 update k_cache/v_cache in-place. Our FA4 and SDPA paths do the same.

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
    if ATTN_IMPL == "fa3":
        return _fa3.flash_attn_with_kvcache(
            q, k_cache, v_cache, k=k, v=v, cache_seqlens=cache_seqlens,
            causal=causal, window_size=window_size
        )
    if ATTN_IMPL == "fa2":
        return _fa2.flash_attn_with_kvcache(
            q, k_cache, v_cache, k=k, v=v, cache_seqlens=cache_seqlens,
            causal=causal, window_size=window_size
        )

    # FA4 / Splash / SDPA: manually manage KV cache
    T_new = q.size(1)
    pos = cache_seqlens[0].item()  # assume uniform position across batch

    # Insert new k, v into cache (in-place, matching FA3 behavior)
    if k is not None and v is not None:
        k_cache[:, pos:pos+T_new, :, :] = k
        v_cache[:, pos:pos+T_new, :, :] = v

    # Get full cache up to current position + new tokens
    end_pos = pos + T_new
    k_full = k_cache[:, :end_pos, :, :]
    v_full = v_cache[:, :end_pos, :, :]

    if ATTN_IMPL == "fa4":
        # FA4 has no kvcache API; causal masking is bottom-right aligned when Tq != Tk, matching FA3
        return _fa4_attention(q, k_full, v_full, causal, window_size)
    if ATTN_IMPL == "splash":
        # Splash has no KV-cache API. Prefill lengths that tile by 128 run the
        # kernel; decode steps fall back to SDPA inside _splash_bthd without warning.
        return _splash_bthd(q, k_full, v_full, causal, window_size, warn_on_fallback=False)

    return _sdpa_bthd(q, k_full, v_full, window_size, causal=causal)


# =============================================================================
# Export: flash_attn module interface (drop-in replacement for FA3)
# =============================================================================
from types import SimpleNamespace
flash_attn = SimpleNamespace(
    flash_attn_func=flash_attn_func,
    flash_attn_with_kvcache=flash_attn_with_kvcache,
)
