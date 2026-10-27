"""Splash Attention helpers and TPU backend selection.

These tests do not need a TPU. They cover the shape rules the Pallas kernel
imposes (tiles of 128) and that auto-selection does not pick Splash on a
machine without TorchTPU.
"""

import pytest
import torch

from nanochat.common import get_peak_flops, tpu_is_available
from nanochat.flash_attention import VALID_IMPLS, set_attn_impl
from nanochat.splash_attention import splash_local_window, splash_shapes_ok
import nanochat.flash_attention as fa_module


def test_valid_impls_include_splash():
    assert "splash" in VALID_IMPLS


def test_splash_shapes():
    assert splash_shapes_ok(2048, 2048)
    assert splash_shapes_ok(128, 512)
    assert splash_shapes_ok(512, 768)
    assert splash_shapes_ok(2048, 2048, head_dim=128)
    assert splash_shapes_ok(2048, 2048, head_dim=256)
    # Decode writes one query against a long cache. 1 is not a Splash tile.
    assert not splash_shapes_ok(1, 2048)
    assert not splash_shapes_ok(64, 64)
    assert not splash_shapes_ok(192, 192)
    assert not splash_shapes_ok(2048, 2048, head_dim=64)
    assert not splash_shapes_ok(2048, 2048, head_dim=32)


def test_splash_non_causal_fallback_is_bidirectional():
    q = torch.randn(1, 16, 2, 32)
    k = torch.randn(1, 16, 2, 32)
    v = torch.randn(1, 16, 2, 32)
    out = fa_module._splash_bthd(q, k, v, causal=False, window_size=(-1, -1))
    ref = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=False
    ).transpose(1, 2)
    torch.testing.assert_close(out, ref)


def test_splash_kvcache_fallback_does_not_warn(monkeypatch):
    monkeypatch.setattr(fa_module, "_splash_fallback_warned", False)
    q = torch.randn(1, 1, 2, 128)
    k = torch.randn(1, 17, 2, 128)
    v = torch.randn(1, 17, 2, 128)
    _ = fa_module._splash_bthd(q, k, v, causal=True, window_size=(-1, 0), warn_on_fallback=False)
    assert fa_module._splash_fallback_warned is False


def test_splash_local_window():
    # Full-context causal (left covers every key) stays on the pure causal kernel.
    assert splash_local_window((2048, 0), kv_len=2048) is None
    assert splash_local_window((-1, 0), kv_len=2048) is None
    assert splash_local_window((2047, 0), kv_len=2048) is None
    # Short window is passed through. 768 is the quarter-context tile at seq 2048.
    assert splash_local_window((768, 0), kv_len=2048) == 768
    assert splash_local_window((0, 0), kv_len=128) == 0


def test_splash_requires_tpu():
    if tpu_is_available():
        pytest.skip("TPU is available; this checks the missing-device error")
    with pytest.raises(AssertionError, match="Splash Attention"):
        set_attn_impl("splash")
    set_attn_impl(None)
    assert fa_module.ATTN_IMPL != "splash"
    assert fa_module.USE_SPLASH is False


def test_tpu_peak_flops():
    assert get_peak_flops("TPU v6e") == 918e12
    assert get_peak_flops("tpu v5p") == 459e12
    assert get_peak_flops("v5litepod-16") == 197e12
    assert get_peak_flops("Ironwood") == 4.614e15
    assert get_peak_flops("unknown") == float("inf")


def test_compile_model_leaves_cpu_backend_default(monkeypatch):
    from nanochat.common import compile_model
    captured = {}

    def fake_compile(model, **kwargs):
        captured.update(kwargs)
        return model

    monkeypatch.setattr(torch, "compile", fake_compile)
    layer = torch.nn.Linear(4, 4)
    assert compile_model(layer) is layer
    assert captured == {"dynamic": False}


def test_checkpoint_policy_fits_v6e_d24():
    """Checkpoint mode follows saved attention bytes, not token count."""
    from nanochat.gpt import GPTConfig, attention_activation_bytes, full_block_checkpoint, tpu_microbatch_sync_every

    d24 = GPTConfig(sequence_len=2048, n_layer=24, n_head=12, n_kv_head=12, n_embd=1536)
    d12 = GPTConfig(sequence_len=2048, n_layer=12, n_head=6, n_kv_head=6, n_embd=768)
    d28 = GPTConfig(sequence_len=2048, n_layer=28, n_head=14, n_kv_head=14, n_embd=1792)
    # d24 B=8 is the 4.5 GiB boundary and stays selective. d24 B=16 and d28 B=8 recompute.
    assert full_block_checkpoint(True, d24, 8, 2048) is False
    assert full_block_checkpoint(True, d24, 16, 2048) is True
    assert full_block_checkpoint(True, d28, 8, 2048) is True
    # Same token counts that used to force a recompute on d12 now stay selective.
    assert full_block_checkpoint(True, d12, 16, 2048) is False
    assert full_block_checkpoint(True, d12, 32, 2048) is False
    assert full_block_checkpoint("full", d12, 1, 128) is True
    assert full_block_checkpoint(False, d24, 32, 2048) is False
    assert tpu_microbatch_sync_every(d24, 8, 2048, True) == 1
    assert tpu_microbatch_sync_every(d24, 4, 2048, True) == 2
    assert tpu_microbatch_sync_every(d12, 4, 2048, True) == 4
    assert tpu_microbatch_sync_every(d12, 32, 2048, True) == 1
    assert tpu_microbatch_sync_every(d24, 8, 2048, "full") == 4
    attn = attention_activation_bytes(24, 12, 12, 128, 8, 2048)
    assert attn == (9 << 30) // 2


def test_full_checkpoint_env_override(monkeypatch):
    from nanochat.gpt import GPTConfig, full_block_checkpoint
    cfg = GPTConfig()
    monkeypatch.setenv("NANOCHAT_FULL_CHECKPOINT", "1")
    assert full_block_checkpoint(True, cfg, 1, 128) is True
    assert full_block_checkpoint(False, cfg, 1, 128) is False


def test_loss_keeps_token_layout():
    """Flattening inside the loss must match an explicit flatten, including grads."""
    from nanochat.gpt import _LinearSoftcapCrossEntropy
    torch.manual_seed(0)
    x3 = torch.randn(4, 8, 16, dtype=torch.float32, requires_grad=True)
    w = torch.randn(32, 16, requires_grad=True)
    targets = torch.randint(0, 20, (4, 8))
    targets[0, 1] = -1
    y3 = _LinearSoftcapCrossEntropy.apply(x3, w, targets, 20, 15.0, "mean")
    x2 = x3.detach().reshape(-1, 16).requires_grad_(True)
    w2 = w.detach().clone().requires_grad_(True)
    y2 = _LinearSoftcapCrossEntropy.apply(x2, w2, targets.reshape(-1), 20, 15.0, "mean")
    torch.testing.assert_close(y3, y2)
    y3.backward()
    y2.backward()
    torch.testing.assert_close(x3.grad, x2.grad.view_as(x3))
    torch.testing.assert_close(w.grad, w2.grad)


def test_rotary_grad_matches_explicit():
    from nanochat.gpt import apply_rotary_emb
    torch.manual_seed(0)
    x = torch.randn(2, 16, 4, 32, requires_grad=True)
    cos = torch.randn(1, 16, 1, 16)
    sin = torch.randn(1, 16, 1, 16)
    y = apply_rotary_emb(x, cos, sin)
    d = 16
    x1, x2 = x[..., :d], x[..., d:]
    y_ref = torch.cat([x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos], dim=-1)
    torch.testing.assert_close(y, y_ref)
    grad = torch.randn_like(y)
    y.backward(grad)
    grad_x = x.grad
    x2p = x.detach().requires_grad_(True)
    x1, x2s = x2p[..., :d], x2p[..., d:]
    y2 = torch.cat([x1 * cos + x2s * sin, x1 * (-sin) + x2s * cos], dim=-1)
    y2.backward(grad)
    torch.testing.assert_close(grad_x, x2p.grad)


def test_bestfit_matches_linear_scan():
    import random
    from nanochat.dataloader import _BestFitDocs

    rng = random.Random(0)
    docs = [torch.arange(rng.randint(1, 40)) for _ in range(200)]
    buf = _BestFitDocs()
    ref = []
    for doc in docs:
        buf.append(doc.clone())
        ref.append(doc.clone())
    remainings = [rng.randint(1, 40) for _ in range(80)]
    for remaining in remainings:
        best_i, best_len = -1, 0
        for i, doc in enumerate(ref):
            n = doc.numel()
            if n <= remaining and n > best_len:
                best_i, best_len = i, n
        if best_i >= 0:
            expect = ref.pop(best_i)
            got = buf.pop_best_fit(remaining)
        else:
            shortest_i = min(range(len(ref)), key=lambda i: ref[i].numel())
            expect = ref.pop(shortest_i)
            assert buf.pop_best_fit(remaining) is None
            got = buf.pop_shortest()
        assert torch.equal(got, expect)
    assert buf.n == len(ref)


def test_prefetch_iterator_preserves_order_and_errors():
    from nanochat.dataloader import _prefetch_iterator
    assert list(_prefetch_iterator(iter([1, 2, 3, 4]), depth=2)) == [1, 2, 3, 4]

    def _boom():
        yield 1
        raise RuntimeError("pack failed")

    got = []
    with pytest.raises(RuntimeError, match="pack failed"):
        for item in _prefetch_iterator(_boom(), depth=2):
            got.append(item)
    assert got == [1]


def test_dataloader_device_buffers_own_storage():
    from nanochat.dataloader import _allocate_token_buffers

    B, T = 4, 2048
    cpu_inputs, cpu_targets, inputs, targets = _allocate_token_buffers(
        B, T, device="cpu", pin_memory=False
    )
    assert cpu_inputs.shape == (B, T)
    assert cpu_targets.shape == (B, T)
    assert inputs.untyped_storage().data_ptr() != targets.untyped_storage().data_ptr()
    for t in (inputs, targets):
        assert not t._is_view()
        assert t.shape == (B, T)
        assert t.dtype == torch.long
        assert t.storage_offset() == 0
        assert t.untyped_storage().size() == B * T * t.element_size()


def _install_fake_splash(monkeypatch):
    """Replace the JAX bridge so Splash tests can run without a TPU."""
    import types
    import nanochat.splash_attention as sa

    seen = []

    def fake_make_splash(**kwargs):
        seen.append(kwargs)

        def fwd(q, k, v):
            return q.clone(), torch.zeros(q.shape[:-1], dtype=torch.float32, device=q.device)

        def bwd(q, k, v, out, lse, g):
            return g.clone(), k.clone(), v.clone()

        return fwd, bwd

    def fake_jax_op(qualname, fn):
        if "_fwd_" in qualname:
            @torch.library.custom_op(qualname, mutates_args=())
            def op(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
                return fn(q, k, v)

            @op.register_fake
            def _(q, k, v):
                return q.new_empty(q.shape), q.new_empty(q.shape[:-1], dtype=torch.float32)

            return op

        @torch.library.custom_op(qualname, mutates_args=())
        def bwd_op(
            q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
            out: torch.Tensor, lse: torch.Tensor, g: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            return fn(q, k, v, out, lse, g)

        @bwd_op.register_fake
        def _(q, k, v, out, lse, g):
            return q.new_empty(q.shape), k.new_empty(k.shape), v.new_empty(v.shape)

        return bwd_op

    monkeypatch.setattr(sa, "_make_splash_attention_fn", fake_make_splash)
    monkeypatch.setattr(sa, "_splash_fwd_ops", {})
    monkeypatch.setattr(sa, "_splash_ops", types.SimpleNamespace())
    monkeypatch.setattr(
        torch, "tpu", types.SimpleNamespace(pallas=types.SimpleNamespace(jax_op=fake_jax_op)), raising=False
    )
    return seen


def test_splash_folds_batch_into_heads(monkeypatch):
    import nanochat.splash_attention as sa
    seen = _install_fake_splash(monkeypatch)
    q = torch.randn(4, 3, 128, 128)
    k = torch.randn(4, 1, 128, 128)
    v = torch.randn(4, 1, 128, 128)
    y = sa.splash_sdpa(q, k, v, is_causal=True, enable_gqa=True)
    assert y.shape == q.shape
    torch.testing.assert_close(y, q)
    # The PyTorch op sees the original heads. The kernel is told to fold batch.
    assert seen[-1]["n_heads"] == 3
    assert seen[-1]["n_kv_heads"] == 1
    assert seen[-1]["batch"] == 4
    q1 = torch.randn(1, 3, 128, 128)
    _ = sa.splash_sdpa(q1, q1, q1, is_causal=True)
    assert seen[-1]["n_heads"] == 3
    assert seen[-1]["n_kv_heads"] == 3
    assert seen[-1]["batch"] == 1


def test_splash_mixed_windows_compile_single_graph(monkeypatch):
    import types
    import nanochat.splash_attention as sa

    def fake_make_splash(**_kwargs):
        def fwd(q, k, v):
            return q.clone(), torch.zeros(q.shape[:-1], dtype=torch.float32, device=q.device)

        def bwd(q, k, v, out, lse, g):
            return g.clone(), k.clone(), v.clone()

        return fwd, bwd

    def fake_jax_op(qualname, fn):
        if "_fwd_" in qualname:
            @torch.library.custom_op(qualname, mutates_args=())
            def op(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
                return fn(q, k, v)

            @op.register_fake
            def _(q, k, v):
                return q.new_empty(q.shape), q.new_empty(q.shape[:-1], dtype=torch.float32)

            return op

        @torch.library.custom_op(qualname, mutates_args=())
        def bwd_op(
            q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
            out: torch.Tensor, lse: torch.Tensor, g: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            return fn(q, k, v, out, lse, g)

        @bwd_op.register_fake
        def _(q, k, v, out, lse, g):
            return q.new_empty(q.shape), k.new_empty(k.shape), v.new_empty(v.shape)

        return bwd_op

    monkeypatch.setattr(sa, "_make_splash_attention_fn", fake_make_splash)
    monkeypatch.setattr(sa, "_splash_fwd_ops", {})
    monkeypatch.setattr(sa, "_splash_ops", types.SimpleNamespace())
    monkeypatch.setattr(
        torch, "tpu", types.SimpleNamespace(pallas=types.SimpleNamespace(jax_op=fake_jax_op)), raising=False
    )

    def mixed_window_fn(q, k, v):
        y = sa.splash_sdpa(q, k, v, is_causal=True, local_window_size=128)
        y = sa.splash_sdpa(y, k, v, is_causal=True, local_window_size=None)
        return y

    q = torch.randn(2, 2, 512, 128)
    explanation = torch._dynamo.explain(mixed_window_fn)(q, q, q)
    assert explanation.graph_count == 1
    assert explanation.graph_break_count == 0


def test_gpt_grad_checkpoint_and_fused_loss(monkeypatch):
    import copy
    import types
    import torch.nn.functional as F
    import nanochat.splash_attention as sa
    from nanochat.gpt import GPT, GPTConfig, _LinearSoftcapCrossEntropy

    fwd_calls = []
    bwd_calls = []

    def fake_make_splash(**kwargs):
        assert kwargs.get("use_vmap_bwd") is True
        assert kwargs.get("block_q_dkv") == 512
        assert kwargs.get("block_kv_dkv") == 512

        def fwd(q, k, v):
            fwd_calls.append(1)
            return (q + k + v) / 3.0, torch.zeros(q.shape[:-1], dtype=torch.float32, device=q.device)

        def bwd(q, k, v, out, lse, g):
            assert g.is_contiguous()
            bwd_calls.append(1)
            g3 = g / 3.0
            return g3.clone(), g3.clone(), g3.clone()

        return fwd, bwd

    def fake_jax_op(qualname, fn):
        if "_fwd_" in qualname:
            @torch.library.custom_op(qualname, mutates_args=())
            def op(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
                return fn(q, k, v)

            @op.register_fake
            def _(q, k, v):
                return q.new_empty(q.shape), q.new_empty(q.shape[:-1], dtype=torch.float32)

            return op

        @torch.library.custom_op(qualname, mutates_args=())
        def bwd_op(
            q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
            out: torch.Tensor, lse: torch.Tensor, g: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            return fn(q, k, v, out, lse, g)

        @bwd_op.register_fake
        def _(q, k, v, out, lse, g):
            return q.new_empty(q.shape), k.new_empty(k.shape), v.new_empty(v.shape)

        return bwd_op

    monkeypatch.setattr(sa, "_make_splash_attention_fn", fake_make_splash)
    monkeypatch.setattr(sa, "_splash_fwd_ops", {})
    monkeypatch.setattr(sa, "_splash_ops", types.SimpleNamespace())
    monkeypatch.setattr(
        torch, "tpu", types.SimpleNamespace(pallas=types.SimpleNamespace(jax_op=fake_jax_op)), raising=False
    )
    monkeypatch.setattr(fa_module, "ATTN_IMPL", "splash")

    # Verify _LinearSoftcapCrossEntropy against exact eager reference (including ignore_index=-1 and padded vocab)
    for reduction in ("mean", "sum", "none"):
        x_ref = torch.randn(64, 32, dtype=torch.bfloat16, requires_grad=True)
        w_ref = torch.randn(128, 32, dtype=torch.float32, requires_grad=True)
        t_ref = torch.randint(0, 100, (64,), dtype=torch.long)
        t_ref[::7] = -1
        logits_ref = 15.0 * torch.tanh(F.linear(x_ref, w_ref[:100].to(x_ref.dtype)).float() / 15.0)
        out_ref = F.cross_entropy(logits_ref, t_ref, ignore_index=-1, reduction=reduction)
        if reduction == "none":
            grad_vec = torch.randn_like(out_ref)
            out_ref.backward(grad_vec)
        else:
            (out_ref * 0.25).backward()

        x_test = x_ref.detach().clone().requires_grad_(True)
        w_test = w_ref.detach().clone().requires_grad_(True)
        out_test = _LinearSoftcapCrossEntropy.apply(x_test, w_test, t_ref, 100, 15.0, reduction)
        if reduction == "none":
            out_test.backward(grad_vec)
        else:
            (out_test * 0.25).backward()
        torch.testing.assert_close(out_test, out_ref, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(x_test.grad, x_ref.grad, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(w_test.grad, w_ref.grad, rtol=1e-2, atol=1e-2)

    cfg = GPTConfig(
        sequence_len=512, vocab_size=256, n_layer=2, n_head=2, n_kv_head=2, n_embd=256, window_pattern="SL"
    )
    torch.manual_seed(0)
    m_ref = GPT(cfg)
    m_ref.init_weights()
    m_ckpt = copy.deepcopy(m_ref)
    m_ckpt.grad_checkpoint = True

    idx = torch.randint(0, 256, (4, 512))
    targets = torch.randint(0, 256, (4, 512))

    # Reference forward + backward (scaled outside)
    loss_ref = m_ref(idx, targets) * 0.25
    loss_ref.backward()

    # Checkpointed forward + backward under aot_eager compile with fused loss_scale
    fwd_calls.clear()
    bwd_calls.clear()
    explanation = torch._dynamo.explain(m_ckpt)(idx, targets, loss_scale=0.25)
    assert explanation.graph_count == 1
    assert explanation.graph_break_count == 0

    fwd_calls.clear()
    bwd_calls.clear()
    compiled_ckpt = torch.compile(m_ckpt, backend="aot_eager", dynamic=False)
    loss_ckpt = compiled_ckpt(idx, targets, loss_scale=0.25)
    loss_ckpt.backward()

    # Default TPU checkpointing (grad_checkpoint=True) does NOT recompute Splash Attention forward in backward
    assert len(fwd_calls) == cfg.n_layer
    assert len(bwd_calls) == cfg.n_layer
    torch.testing.assert_close(loss_ref, loss_ckpt, rtol=1e-5, atol=1e-5)
    for (n1, p1), (n2, p2) in zip(m_ref.named_parameters(), m_ckpt.named_parameters()):
        assert n1 == n2
        torch.testing.assert_close(p1.grad, p2.grad, rtol=1e-4, atol=1e-4)

    # Full-block checkpointing (grad_checkpoint="full") runs Splash forward once in fw and once in bw per layer
    m_full = copy.deepcopy(m_ref)
    m_full.zero_grad(set_to_none=True)
    m_full.grad_checkpoint = "full"
    fwd_calls.clear()
    bwd_calls.clear()
    compiled_full = torch.compile(m_full, backend="aot_eager", dynamic=False)
    loss_full = compiled_full(idx, targets, loss_scale=0.25)
    loss_full.backward()
    assert len(fwd_calls) == 2 * cfg.n_layer
    assert len(bwd_calls) == cfg.n_layer
    torch.testing.assert_close(loss_ref, loss_full, rtol=1e-5, atol=1e-5)
    for (n1, p1), (n2, p2) in zip(m_ref.named_parameters(), m_full.named_parameters()):
        assert n1 == n2
        torch.testing.assert_close(p1.grad, p2.grad, rtol=1e-4, atol=1e-4)


