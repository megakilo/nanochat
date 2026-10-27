"""Google TPU Splash Attention for TorchTPU.

Drop-in replacement for ``F.scaled_dot_product_attention`` on TPU. The kernel
is JAX Pallas ``jax.experimental.pallas.ops.tpu.splash_attention``: softmax
attention with static causal and local masks, tiled for the TPU MXU. This is
Google's Splash Attention. It is not AdaSplash, which is an α-entmax sparse
attention algorithm with Triton kernels for GPUs.

Adapted from the TorchTPU example (Apache-2.0, Copyright 2026 Google LLC):
https://github.com/google-pytorch/torch_tpu/blob/main/examples/FLUX_optimization/splash_attention.py

JAX is imported only as the kernel-authoring DSL. ``torch.tpu.pallas.jax_op``
traces the JAX functions once per shape and dtype, exports a StableHLO
module, and after that the kernel runs on ``torch.Tensor`` TPU buffers.
No JAX runtime executes during training or inference.

q, k, v are ``(batch, heads, seq, dim)``, matching PyTorch SDPA. The kernel
grids over heads, so the JAX functions fold batch into heads with a reshape.
That reshape stays inside the StableHLO module: the PyTorch op is called on the
original ``(batch, heads, seq, dim)`` tensors, which lets AOTAutograd deduplicate
them with the QK-norm outputs. Causal masks are bottom-right aligned, so a short
query against a longer KV cache (prefill already consumed) lines up with Flash
Attention. Sequence lengths must be multiples of 128; token-by-token decode is
not a legal Splash tile.
"""

import hashlib
import math
import os
import types
import typing

import torch

# TPU vector lanes. Splash block sizes and sequence lengths must be multiples of this.
SPLASH_BLOCK_LANES = 128


def splash_shapes_ok(q_len: int, kv_len: int, head_dim: int = SPLASH_BLOCK_LANES) -> bool:
    """True when sequence lengths and head_dim can be tiled by the Splash kernel."""
    return (
        q_len >= SPLASH_BLOCK_LANES
        and kv_len >= SPLASH_BLOCK_LANES
        and q_len % SPLASH_BLOCK_LANES == 0
        and kv_len % SPLASH_BLOCK_LANES == 0
        and head_dim > 0
        and head_dim % SPLASH_BLOCK_LANES == 0
    )


def splash_local_window(window_size, kv_len: int) -> int | None:
    """FA3 ``(left, right)`` window -> Splash left-window size.

    ``-1`` or a left window that already covers every key means unlimited,
    which lets the kernel use its causal mask instead of a local one.
    The right side is not returned: causal attention uses right=0, and Splash
    encodes that in the mask (see ``splash_sdpa``).
    """
    left = window_size[0]
    if left is None or left < 0 or left >= kv_len - 1:
        return None
    return int(left)


def splash_available() -> tuple[bool, str | None]:
    """Whether TorchTPU and the JAX Splash kernel can be imported."""
    from nanochat.common import tpu_is_available
    if not tpu_is_available():
        return False, "TPU is not available (install TorchTPU with `uv sync --extra tpu` on a TPU VM)"
    try:
        import torch_tpu  # noqa: F401
        from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel  # noqa: F401
        torch.tpu.pallas.jax_op  # the bridge Splash Attention is lowered through
    except Exception as e:
        return False, f"failed to import Splash Attention: {type(e).__name__}: {e}"
    return True, None


def _adjust_block(preferred: int, seq_len: int) -> int:
    """Largest tile <= preferred that divides seq_len and is a multiple of 128."""
    if seq_len % SPLASH_BLOCK_LANES != 0:
        raise ValueError(
            f"Splash Attention sequence length {seq_len} must be a multiple of {SPLASH_BLOCK_LANES}"
        )
    preferred = max(SPLASH_BLOCK_LANES, int(preferred))
    if preferred % SPLASH_BLOCK_LANES == 0 and seq_len % preferred == 0:
        return preferred
    for candidate in (512, 256, SPLASH_BLOCK_LANES):
        if candidate <= preferred and seq_len % candidate == 0:
            return candidate
    block = min(preferred, seq_len)
    block -= block % SPLASH_BLOCK_LANES
    while block >= SPLASH_BLOCK_LANES and seq_len % block != 0:
        block -= SPLASH_BLOCK_LANES
    if block < SPLASH_BLOCK_LANES:
        raise ValueError(
            f"Splash Attention sequence length {seq_len} has no legal block size"
        )
    return block


def _ensure_kernel_modules():
    """Import TorchTPU and JAX Splash. Called only when a kernel is built."""
    import torch_tpu  # noqa: F401
    import jax
    from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel
    from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_mask
    from jax.experimental.pallas.ops.tpu.splash_attention.splash_attention_kernel import QKVLayout
    import jax.numpy as jnp
    return jax, jnp, splash_attention_kernel, splash_attention_mask, QKVLayout


def _make_splash_attention_fn(
    q_seq_len: int,
    kv_seq_len: int,
    n_heads: int,
    n_kv_heads: int,
    batch: int = 1,
    scale: float | None = None,
    is_causal: bool = True,
    local_window_size: int | None = None,
    block_q: int = 512,
    block_kv: int = 512,
    block_kv_compute: int = 512,
    block_q_dkv: int = 512,
    block_kv_dkv: int = 512,
    block_kv_dkv_compute: int = 512,
    block_q_dq: int | None = None,
    block_kv_dq: int | None = None,
    use_fused_bwd_kernel: bool = True,
    q_layout: str = "HEAD_DIM_MINOR",
    k_layout: str = "HEAD_DIM_MINOR",
    v_layout: str = "HEAD_DIM_MINOR",
    use_vmap_bwd: bool = True,
):
    """Build JAX Splash forward and backward callables for one static shape.

    Forward takes ``(q, k, v)`` shaped ``(batch, heads, seq, dim)`` and returns
    ``(out, logsumexp)`` in that same layout. Backward takes those saved tensors
    plus ``grad_out`` and returns ``(dq, dk, dv)``. The batch axis is folded into
    heads inside these functions so the Pallas grid is ``(batch * heads, ...)``.
    """
    jax, jnp, splash_attention_kernel, splash_attention_mask, QKVLayout = _ensure_kernel_modules()
    batch = max(int(batch), 1)
    kernel_q_heads = batch * n_heads
    kernel_kv_heads = batch * n_kv_heads
    default_mask_value = splash_attention_kernel.DEFAULT_MASK_VALUE

    def _get_layout(layout_str):
        if layout_str == "HEAD_DIM_MINOR":
            return QKVLayout.HEAD_DIM_MINOR
        if layout_str == "SEQ_MINOR":
            return QKVLayout.SEQ_MINOR
        raise ValueError(f"Unknown Splash layout: {layout_str}")

    # Fused backward rejects explicit dq tiles. Keep them None in that case.
    block_sizes = splash_attention_kernel.BlockSizes(
        block_q=min(block_q, q_seq_len),
        block_kv=min(block_kv, kv_seq_len),
        block_kv_compute=min(block_kv_compute, kv_seq_len),
        block_q_dkv=min(block_q_dkv, q_seq_len),
        block_kv_dkv=min(block_kv_dkv, kv_seq_len),
        block_kv_dkv_compute=min(block_kv_dkv_compute, kv_seq_len),
        block_q_dq=(
            min(block_q_dq, q_seq_len)
            if not use_fused_bwd_kernel and block_q_dq is not None
            else None
        ),
        block_kv_dq=(
            min(block_kv_dq, kv_seq_len)
            if not use_fused_bwd_kernel and block_kv_dq is not None
            else None
        ),
        use_fused_bwd_kernel=use_fused_bwd_kernel,
        q_layout=_get_layout(q_layout),
        k_layout=_get_layout(k_layout),
        v_layout=_get_layout(v_layout),
    )

    # Bottom-right causal: query i aligns with key (kv_len - q_len + i).
    # Equal lengths give offset 0, which is the standard training triangle.
    mask_shape = (q_seq_len, kv_seq_len)
    offset = (kv_seq_len - q_seq_len) if is_causal else 0
    if local_window_size is not None:
        single_mask = splash_attention_mask.LocalMask(
            shape=mask_shape,
            window_size=(local_window_size, 0 if is_causal else None),
            offset=offset,
        )
    elif is_causal:
        single_mask = splash_attention_mask.CausalMask(shape=mask_shape, offset=offset)
    else:
        single_mask = splash_attention_mask.FullMask(mask_shape)

    multi_head_mask = splash_attention_mask.MultiHeadMask(masks=(single_mask,) * kernel_q_heads)
    is_mqa = kernel_kv_heads == 1
    if is_mqa:
        splash_kernel = splash_attention_kernel.make_splash_mqa(
            mask=multi_head_mask,
            head_shards=1,
            q_seq_shards=1,
            block_sizes=block_sizes,
        )
    else:
        splash_kernel = splash_attention_kernel.make_splash_mha(
            mask=multi_head_mask,
            head_shards=1,
            q_seq_shards=1,
            block_sizes=block_sizes,
        )

    fwd_mask_info = splash_kernel.fwd_mask_info
    dq_mask_info = splash_kernel.dq_mask_info
    dkv_mask_info = splash_kernel.dkv_mask_info
    mask_value = splash_kernel.kwargs.get("mask_value", default_mask_value)
    mask_function = splash_kernel.kwargs.get("mask_function", None)
    attn_logits_soft_cap = splash_kernel.kwargs.get("attn_logits_soft_cap", None)
    residual_checkpoint_name = splash_kernel.kwargs.get("residual_checkpoint_name", None)
    interpret = splash_kernel.kwargs.get("interpret", False)

    def _single_fwd_with_lse(q_b, k_b, v_b):
        out_b, (logsumexp_b,) = splash_attention_kernel._splash_attention_forward(
            fwd_mask_info,
            q_b,
            k_b,
            v_b,
            segment_ids=None,
            sinks=None,
            mask_value=mask_value,
            is_mqa=is_mqa,
            block_sizes=block_sizes,
            residual_checkpoint_name=residual_checkpoint_name,
            save_residuals=True,
            mask_function=mask_function,
            attn_logits_soft_cap=attn_logits_soft_cap,
            interpret=interpret,
        )
        return out_b, logsumexp_b

    def _fold_qkv(q, k, v):
        """(batch, heads, seq, dim) -> one batch row whose head count is batch * heads."""
        b, h, t, d = q.shape
        kv_h = k.shape[1]
        q_f = jnp.reshape(q, (1, b * h, t, d))
        k_f = jnp.reshape(k, (1, b * kv_h, k.shape[2], k.shape[3]))
        v_f = jnp.reshape(v, (1, b * kv_h, v.shape[2], v.shape[3]))
        if is_mqa:
            k_f = jnp.squeeze(k_f, axis=1)
            v_f = jnp.squeeze(v_f, axis=1)
        return q_f, k_f, v_f

    # jax.Array annotations are how jax_op builds the PyTorch custom-op schema.
    # Shapes here are the PyTorch tensors. The fold is a bitcast in the HLO module,
    # not a second saved activation at the autograd boundary.
    def splash_fn(q: jax.Array, k: jax.Array, v: jax.Array) -> tuple[jax.Array, jax.Array]:
        b, h, t, d = q.shape
        q_f, k_f, v_f = _fold_qkv(q, k, v)
        head_dim = q.shape[-1]
        scale_val = (1.0 / math.sqrt(head_dim)) if scale is None else float(scale)
        scale_arr = jnp.array(scale_val, dtype=q.dtype)
        out, logsumexp = jax.vmap(_single_fwd_with_lse)(q_f * scale_arr, k_f, v_f)
        out = jnp.reshape(out, (b, h, t, d))
        logsumexp = jnp.reshape(logsumexp, (b, h, t))
        return out, logsumexp

    def _single_bwd(q_scaled_b, k_b, v_b, out_b, logsumexp_b, g_b):
        residuals = (
            q_scaled_b,
            k_b,
            v_b,
            None,  # segment_ids
            None,  # sinks
            out_b,
            logsumexp_b,
            dq_mask_info,
            dkv_mask_info,
        )
        all_grads = splash_attention_kernel._splash_attention_bwd(
            False,  # save_residuals (nondiff)
            mask_value,
            is_mqa,
            block_sizes,
            residual_checkpoint_name,
            mask_function,
            attn_logits_soft_cap,
            interpret,
            residuals,
            g_b,
        )
        return all_grads[3], all_grads[4], all_grads[5]  # dq_scaled, dk, dv

    def _scan_bwd_body(carry, args):
        dq_scaled_b, dk_b, dv_b = _single_bwd(*args)
        return carry, (dq_scaled_b, dk_b, dv_b)

    def splash_bwd_fn(
        q: jax.Array,
        k: jax.Array,
        v: jax.Array,
        out: jax.Array,
        logsumexp: jax.Array,
        g: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        b, h, t, d = q.shape
        kv_h = k.shape[1]
        kv_t = k.shape[2]
        q_f, k_f, v_f = _fold_qkv(q, k, v)
        out_f = jnp.reshape(out, (1, b * h, t, d))
        lse_f = jnp.reshape(logsumexp, (1, b * h, t))
        g_f = jnp.reshape(g, (1, b * h, t, d))
        head_dim = q.shape[-1]
        scale_val = (1.0 / math.sqrt(head_dim)) if scale is None else float(scale)
        scale_arr = jnp.array(scale_val, dtype=q.dtype)
        q_scaled = q_f * scale_arr
        if use_vmap_bwd:
            dq_scaled, dk, dv = jax.vmap(_single_bwd)(q_scaled, k_f, v_f, out_f, lse_f, g_f)
        else:
            _, (dq_scaled, dk, dv) = jax.lax.scan(
                _scan_bwd_body, None, (q_scaled, k_f, v_f, out_f, lse_f, g_f),
            )
        dq = jax.lax.optimization_barrier(dq_scaled * scale_arr)
        dk = jax.lax.optimization_barrier(dk)
        dv = jax.lax.optimization_barrier(dv)
        dq = jnp.reshape(dq, (b, h, t, d))
        dk = jnp.reshape(dk, (b, kv_h, kv_t, d))
        dv = jnp.reshape(dv, (b, kv_h, kv_t, d))
        return dq, dk, dv

    return splash_fn, splash_bwd_fn


# cache_key -> registered forward op, plus an attribute namespace of op_fwd_name -> op.
# The op object itself cannot be returned through assume_constant_result (Dynamo will not
# guard a CustomOpDef). Indexing a global dict in traced code also fails when a second
# window size is registered mid-trace (Dynamo snapshots the dict into a ConstDictVariable
# on the first layer and raises KeyError on the new key), so _get_splash_op returns the
# op attribute name and the caller looks it up via getattr on _splash_ops.
_splash_fwd_ops: dict[tuple, typing.Any] = {}
_splash_ops = types.SimpleNamespace()


@torch._dynamo.assume_constant_result
def _get_splash_op(**splash_kwargs) -> str:
    """Register the forward op for this shape and return its attribute name on ``_splash_ops``.

    The name is a compile-time constant. The callable lives on ``_splash_ops``
    (and in ``_splash_fwd_ops`` keyed by ``cache_key``).
    """
    cache_key = tuple(sorted(splash_kwargs.items()))
    identifier = hashlib.sha256(str(cache_key).encode("utf-8")).hexdigest()[:16]
    op_fwd_name = f"splash_attn_fwd_{identifier}"
    if cache_key in _splash_fwd_ops:
        return op_fwd_name

    splash_fn, splash_bwd_fn = _make_splash_attention_fn(**splash_kwargs)
    op_bwd_name = f"splash_attn_bwd_{identifier}"
    torch_fwd_fn: typing.Any = torch.tpu.pallas.jax_op(f"pallas::{op_fwd_name}", splash_fn)
    torch_bwd_fn: typing.Any = torch.tpu.pallas.jax_op(f"pallas::{op_bwd_name}", splash_bwd_fn)

    def setup_context(ctx, inputs, output):
        q, k, v = inputs
        out, lse = output
        ctx.save_for_backward(q, k, v, out, lse)

    def backward(ctx, grad_out, grad_lse):
        del grad_lse
        q, k, v, out, lse = ctx.saved_tensors
        return torch_bwd_fn(q, k, v, out, lse, grad_out.contiguous())

    torch_fwd_fn.register_autograd(backward, setup_context=setup_context)
    _splash_fwd_ops[cache_key] = torch_fwd_fn
    setattr(_splash_ops, op_fwd_name, torch_fwd_fn)
    return op_fwd_name


def splash_sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    scale: float | None = None,
    is_causal: bool = True,
    local_window_size: int | None = None,
    enable_gqa: bool = False,
    block_q: int = 512,
    block_kv: int = 512,
    block_dkv: int | None = None,
    block_kv_compute: int = 512,
    block_q_dkv: int = 512,
    block_kv_dkv: int = 512,
    block_kv_dkv_compute: int = 512,
    block_q_dq: int | None = None,
    block_kv_dq: int | None = None,
    use_fused_bwd_kernel: bool = True,
    use_vmap_bwd: bool = True,
    q_layout: str = "HEAD_DIM_MINOR",
    k_layout: str = "HEAD_DIM_MINOR",
    v_layout: str = "HEAD_DIM_MINOR",
) -> torch.Tensor:
    """Splash Attention with the ``F.scaled_dot_product_attention`` argument layout.

    q, k, v are ``(batch, heads, seq, dim)`` on TPU. The op is invoked on those
    tensors unchanged. Grouped-query attention is native. A single KV head is
    squeezed inside the JAX function, which is what the MQA kernel expects.
    """
    block_q = int(os.environ.get("SPLASH_BLOCK_Q", block_q))
    block_kv = int(os.environ.get("SPLASH_BLOCK_KV", block_kv))
    dkv_env = os.environ.get("SPLASH_BLOCK_DKV")
    if dkv_env is not None:
        block_dkv = int(dkv_env)
    if block_dkv is not None:
        block_q_dkv = block_dkv
        block_kv_dkv = block_dkv
        block_kv_dkv_compute = block_dkv
    vmap_env = os.environ.get("SPLASH_VMAP_BWD")
    if vmap_env is not None:
        use_vmap_bwd = vmap_env.strip().lower() not in ("0", "false", "no")

    # Folding happens in splash_fn / splash_bwd_fn. Reshaping here would make
    # the saved q, k a different FX node from the QK-norm outputs.
    batch, n_heads, q_seq_len, _head_dim = q.shape
    n_kv_heads = k.shape[1]
    kv_seq_len = k.shape[2]
    if enable_gqa != (n_heads != n_kv_heads):
        raise ValueError(
            f"enable_gqa ({enable_gqa}) does not match inputs: "
            f"n_heads={n_heads}, n_kv_heads={n_kv_heads}"
        )
    if n_kv_heads <= 0 or n_heads % n_kv_heads != 0:
        raise ValueError(
            f"Splash Attention requires n_heads ({n_heads}) to be divisible by n_kv_heads ({n_kv_heads})"
        )

    block_q = _adjust_block(block_q, q_seq_len)
    block_kv = _adjust_block(block_kv, kv_seq_len)
    # The compute tile has to divide the memory tile. Fall back to the memory
    # tile itself when the requested compute tile does not.
    block_kv_compute = _adjust_block(min(block_kv_compute, block_kv), kv_seq_len)
    if block_kv % block_kv_compute != 0:
        block_kv_compute = block_kv
    block_q_dkv = _adjust_block(block_q_dkv, q_seq_len)
    block_kv_dkv = _adjust_block(block_kv_dkv, kv_seq_len)
    block_kv_dkv_compute = _adjust_block(min(block_kv_dkv_compute, block_kv_dkv), kv_seq_len)
    if block_kv_dkv % block_kv_dkv_compute != 0:
        block_kv_dkv_compute = block_kv_dkv
    if block_q_dq is not None:
        block_q_dq = _adjust_block(block_q_dq, q_seq_len)
    if block_kv_dq is not None:
        block_kv_dq = _adjust_block(block_kv_dq, kv_seq_len)

    splash_args = dict(
        q_seq_len=q_seq_len,
        kv_seq_len=kv_seq_len,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        batch=batch,
        scale=scale,
        is_causal=is_causal,
        local_window_size=local_window_size,
        block_q=block_q,
        block_kv=block_kv,
        block_kv_compute=block_kv_compute,
        block_q_dkv=block_q_dkv,
        block_kv_dkv=block_kv_dkv,
        block_kv_dkv_compute=block_kv_dkv_compute,
        block_q_dq=block_q_dq,
        block_kv_dq=block_kv_dq,
        use_fused_bwd_kernel=use_fused_bwd_kernel,
        use_vmap_bwd=use_vmap_bwd,
        q_layout=q_layout,
        k_layout=k_layout,
        v_layout=v_layout,
    )
    torch_fwd_fn = getattr(_splash_ops, _get_splash_op(**splash_args))

    input_dtype = q.dtype
    result, _lse = torch_fwd_fn(q, k, v)
    result = typing.cast(torch.Tensor, result)
    if result.dtype != input_dtype:
        result = result.to(input_dtype)
    return result
