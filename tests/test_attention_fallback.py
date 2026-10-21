"""
Test Flash Attention unified interface - verify FA3/FA4/FA2 and SDPA produce identical results.

Run: python -m pytest tests/test_attention_fallback.py -v -s

Note on test structure:
    Tests are split into classes due to dtype/device constraints:

    1. TestFA3VsSDPA / TestFA2VsSDPA / TestFA4VsSDPA: Comparison tests that run both the Flash
       Attention backend and SDPA on the same inputs and verify they produce identical results.
       These require a compatible GPU (FA3: sm80/sm90; FA2/FA4: sm80/sm90/sm100/sm120) and use
       bfloat16 (none of the Flash Attention backends support float32).

    2. TestSDPAOnly: Tests that only exercise the SDPA fallback path. These can run
       on any device (CUDA, CPU, MPS) with the appropriate dtype for that device.
"""
import torch
import pytest
import nanochat.flash_attention as fa_module
from nanochat.flash_attention import flash_attn, HAS_FA3, has_fa4, has_fa2
from nanochat.engine import KVCache

HAS_FA4 = has_fa4()
HAS_FA2 = has_fa2()


def set_impl(impl):
    """Set the implementation override ('fa4', 'fa3', 'fa2', 'sdpa', or None for auto) and re-resolve the backend."""
    fa_module.set_attn_impl(impl)


def run_both_impls(fn, impl='fa3'):
    """Run a function with both the flash impl (FA3, FA2, or FA4) and SDPA, return both outputs."""
    set_impl(impl)
    out_flash = fn()
    set_impl('sdpa')
    out_sdpa = fn()
    set_impl(None)  # reset
    return out_flash, out_sdpa


def assert_close(t1, t2, name, atol=1e-2, rtol=1e-2):
    """Assert two tensors are close, with helpful error message."""
    max_diff = (t1 - t2).abs().max().item()
    mean_diff = (t1 - t2).abs().mean().item()
    assert torch.allclose(t1, t2, atol=atol, rtol=rtol), \
        f"{name}: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}"
    return max_diff, mean_diff


# =============================================================================
# Flash (FA3 / FA2 / FA4) vs SDPA comparison tests
# =============================================================================
class _FlashVsSDPA:
    """Compare a flash implementation (IMPL) and SDPA produce identical results."""

    IMPL = None
    DEVICE = "cuda"
    DTYPE = torch.bfloat16

    def test_basic_causal(self):
        """Basic causal attention."""
        B, T, H, D = 2, 64, 4, 32
        q = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE)
        k = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE)
        v = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE)

        def run():
            return flash_attn.flash_attn_func(q, k, v, causal=True, window_size=(T, 0))

        y_flash, y_sdpa = run_both_impls(run, self.IMPL)
        max_diff, mean_diff = assert_close(y_flash, y_sdpa, "basic_causal")
        print(f"basic_causal: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")

    def test_full_context(self):
        """Full context (window_size=-1)."""
        B, T, H, D = 2, 128, 4, 32
        q = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE)
        k = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE)
        v = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE)

        def run():
            return flash_attn.flash_attn_func(q, k, v, causal=True, window_size=(-1, -1))

        y_flash, y_sdpa = run_both_impls(run, self.IMPL)
        max_diff, mean_diff = assert_close(y_flash, y_sdpa, "full_context")
        print(f"full_context: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")

    def test_sliding_window(self):
        """Sliding window attention (window < T)."""
        B, T, H, D = 2, 128, 4, 32
        window = 32  # Only attend to last 32 tokens
        q = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE)
        k = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE)
        v = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE)

        def run():
            return flash_attn.flash_attn_func(q, k, v, causal=True, window_size=(window, 0))

        y_flash, y_sdpa = run_both_impls(run, self.IMPL)
        max_diff, mean_diff = assert_close(y_flash, y_sdpa, "sliding_window")
        print(f"sliding_window: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")

    def test_gqa(self):
        """Group Query Attention (fewer KV heads than Q heads)."""
        B, T, D = 2, 64, 32
        n_heads = 8
        n_kv_heads = 2

        q = torch.randn(B, T, n_heads, D, device=self.DEVICE, dtype=self.DTYPE)
        k = torch.randn(B, T, n_kv_heads, D, device=self.DEVICE, dtype=self.DTYPE)
        v = torch.randn(B, T, n_kv_heads, D, device=self.DEVICE, dtype=self.DTYPE)

        def run():
            return flash_attn.flash_attn_func(q, k, v, causal=True, window_size=(T, 0))

        y_flash, y_sdpa = run_both_impls(run, self.IMPL)
        max_diff, mean_diff = assert_close(y_flash, y_sdpa, "gqa")
        print(f"gqa: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")

    def test_larger_model(self):
        """Larger dimensions closer to real model."""
        B, T, H, D = 4, 256, 12, 64
        q = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE)
        k = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE)
        v = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE)

        def run():
            return flash_attn.flash_attn_func(q, k, v, causal=True, window_size=(-1, -1))

        y_flash, y_sdpa = run_both_impls(run, self.IMPL)
        max_diff, mean_diff = assert_close(y_flash, y_sdpa, "larger_model")
        print(f"larger_model: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")

    def test_kvcache_prefill(self):
        """Test prefill (inserting multiple tokens into empty cache)."""
        B, T_max, H, D = 2, 64, 4, 32
        T_prefill = 16

        q = torch.randn(B, T_prefill, H, D, device=self.DEVICE, dtype=self.DTYPE)
        k = torch.randn(B, T_prefill, H, D, device=self.DEVICE, dtype=self.DTYPE)
        v = torch.randn(B, T_prefill, H, D, device=self.DEVICE, dtype=self.DTYPE)

        def run():
            k_cache = torch.zeros(B, T_max, H, D, device=self.DEVICE, dtype=self.DTYPE)
            v_cache = torch.zeros(B, T_max, H, D, device=self.DEVICE, dtype=self.DTYPE)
            cache_seqlens = torch.zeros(B, dtype=torch.int32, device=self.DEVICE)
            return flash_attn.flash_attn_with_kvcache(
                q, k_cache, v_cache, k=k, v=v,
                cache_seqlens=cache_seqlens,
                causal=True, window_size=(T_max, 0)
            )

        y_flash, y_sdpa = run_both_impls(run, self.IMPL)
        max_diff, mean_diff = assert_close(y_flash, y_sdpa, "prefill")
        print(f"prefill: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")

    def test_kvcache_single_token(self):
        """Test single token generation (cache already has content)."""
        B, T_max, H, D = 2, 64, 4, 32
        T_prefill = 16

        k_init = torch.randn(B, T_prefill, H, D, device=self.DEVICE, dtype=self.DTYPE)
        v_init = torch.randn(B, T_prefill, H, D, device=self.DEVICE, dtype=self.DTYPE)
        q_single = torch.randn(B, 1, H, D, device=self.DEVICE, dtype=self.DTYPE)
        k_single = torch.randn(B, 1, H, D, device=self.DEVICE, dtype=self.DTYPE)
        v_single = torch.randn(B, 1, H, D, device=self.DEVICE, dtype=self.DTYPE)

        def run():
            k_cache = torch.zeros(B, T_max, H, D, device=self.DEVICE, dtype=self.DTYPE)
            v_cache = torch.zeros(B, T_max, H, D, device=self.DEVICE, dtype=self.DTYPE)
            k_cache[:, :T_prefill, :, :] = k_init
            v_cache[:, :T_prefill, :, :] = v_init
            cache_seqlens = torch.full((B,), T_prefill, dtype=torch.int32, device=self.DEVICE)
            return flash_attn.flash_attn_with_kvcache(
                q_single, k_cache, v_cache, k=k_single, v=v_single,
                cache_seqlens=cache_seqlens,
                causal=True, window_size=(T_max, 0)
            )

        y_flash, y_sdpa = run_both_impls(run, self.IMPL)
        max_diff, mean_diff = assert_close(y_flash, y_sdpa, "single_token")
        print(f"single_token: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")

    def test_kvcache_single_token_sliding_window(self):
        """Test single token decode with sliding window smaller than cache size.

        This catches the bug where SDPA ignores window_size during Tq=1 decode.
        When window < Tk, FA3 only attends to the last (window+1) tokens,
        but SDPA was attending to all cached tokens.
        """
        B, T_max, H, D = 2, 64, 4, 32
        T_prefill = 32  # Enough tokens to exceed window
        window = 8      # Window SMALLER than cache size

        k_init = torch.randn(B, T_prefill, H, D, device=self.DEVICE, dtype=self.DTYPE)
        v_init = torch.randn(B, T_prefill, H, D, device=self.DEVICE, dtype=self.DTYPE)
        q_single = torch.randn(B, 1, H, D, device=self.DEVICE, dtype=self.DTYPE)
        k_single = torch.randn(B, 1, H, D, device=self.DEVICE, dtype=self.DTYPE)
        v_single = torch.randn(B, 1, H, D, device=self.DEVICE, dtype=self.DTYPE)

        def run():
            k_cache = torch.zeros(B, T_max, H, D, device=self.DEVICE, dtype=self.DTYPE)
            v_cache = torch.zeros(B, T_max, H, D, device=self.DEVICE, dtype=self.DTYPE)
            k_cache[:, :T_prefill, :, :] = k_init
            v_cache[:, :T_prefill, :, :] = v_init
            cache_seqlens = torch.full((B,), T_prefill, dtype=torch.int32, device=self.DEVICE)
            return flash_attn.flash_attn_with_kvcache(
                q_single, k_cache, v_cache, k=k_single, v=v_single,
                cache_seqlens=cache_seqlens,
                causal=True, window_size=(window, 0)  # window=8 < Tk=33
            )

        y_flash, y_sdpa = run_both_impls(run, self.IMPL)
        max_diff, mean_diff = assert_close(y_flash, y_sdpa, "single_token_sliding_window")
        print(f"single_token_sliding_window: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")

    def test_backward_gradients_match(self):
        """Verify gradients are similar between Flash Attention and SDPA."""
        B, T, H, D = 2, 32, 4, 16

        q_data = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE)
        k_data = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE)
        v_data = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE)

        def run():
            q = q_data.clone().requires_grad_(True)
            k = k_data.clone().requires_grad_(True)
            v = v_data.clone().requires_grad_(True)
            y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=(T, 0))
            loss = y.sum()
            loss.backward()
            return y.detach(), q.grad.detach(), k.grad.detach(), v.grad.detach()

        set_impl(self.IMPL)
        y_flash, q_grad_flash, k_grad_flash, v_grad_flash = run()
        set_impl('sdpa')
        y_sdpa, q_grad_sdpa, k_grad_sdpa, v_grad_sdpa = run()
        set_impl(None)

        max_diff, mean_diff = assert_close(y_flash, y_sdpa, "backward_output")
        print(f"backward_output: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")

        max_diff, mean_diff = assert_close(q_grad_flash, q_grad_sdpa, "q_grad", atol=0.05, rtol=0.05)
        print(f"q_grad: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")

        max_diff, mean_diff = assert_close(k_grad_flash, k_grad_sdpa, "k_grad", atol=0.05, rtol=0.05)
        print(f"k_grad: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")

        max_diff, mean_diff = assert_close(v_grad_flash, v_grad_sdpa, "v_grad", atol=0.05, rtol=0.05)
        print(f"v_grad: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")

    def _grads(self, q_data, k_data, v_data, window_size, compile=False):
        def attn(q, k, v):
            return flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)
        fn = torch.compile(attn, dynamic=False, fullgraph=True) if compile else attn
        q = q_data.clone().requires_grad_(True)
        k = k_data.clone().requires_grad_(True)
        v = v_data.clone().requires_grad_(True)
        y = fn(q, k, v)
        (y.float() * torch.linspace(-1, 1, y.size(-1), device=y.device)).sum().backward()
        return y.detach(), q.grad, k.grad, v.grad

    def _compare_grads(self, B, T, H, H_kv, D, window_size, compile=False):
        q = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE)
        k = torch.randn(B, T, H_kv, D, device=self.DEVICE, dtype=self.DTYPE)
        v = torch.randn(B, T, H_kv, D, device=self.DEVICE, dtype=self.DTYPE)
        set_impl(self.IMPL)
        out_flash = self._grads(q, k, v, window_size, compile=compile)
        set_impl('sdpa')
        out_sdpa = self._grads(q, k, v, window_size)
        set_impl(None)
        for name, a, b in zip(["out", "dq", "dk", "dv"], out_flash, out_sdpa):
            max_diff, mean_diff = assert_close(a, b, name, atol=0.05, rtol=0.05)
            print(f"{name}: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")

    def test_backward_sliding_window_hd128(self):
        """Gradients with sliding window at nanochat's default head_dim=128."""
        self._compare_grads(B=2, T=512, H=4, H_kv=4, D=128, window_size=(128, 0))

    def test_backward_gqa_hd128(self):
        """Gradients with GQA at head_dim=128."""
        self._compare_grads(B=2, T=256, H=8, H_kv=2, D=128, window_size=(256, 0))

    def test_torch_compile(self):
        """Flash backend must compile with fullgraph=True (no graph breaks) and match SDPA."""
        self._compare_grads(B=2, T=256, H=4, H_kv=4, D=128, window_size=(64, 0), compile=True)


@pytest.mark.skipif(not HAS_FA3, reason="FA3 required to compare implementations")
class TestFA3VsSDPA(_FlashVsSDPA):
    """Compare FA3 and SDPA produce identical results."""
    IMPL = 'fa3'


@pytest.mark.skipif(not HAS_FA2, reason="FA2 (kernels-community/flash-attn2) required to compare implementations")
class TestFA2VsSDPA(_FlashVsSDPA):
    """Compare FA2 and SDPA produce identical results."""
    IMPL = 'fa2'


@pytest.mark.skipif(not HAS_FA4, reason="FA4 (kernels-community/flash-attn4) required to compare implementations")
class TestFA4VsSDPA(_FlashVsSDPA):
    """Compare FA4 and SDPA produce identical results."""
    IMPL = 'fa4'


# =============================================================================
# SDPA-only tests (run on any device)
# =============================================================================
class TestSDPAOnly:
    """Test SDPA fallback works correctly. Runs on any device."""

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    def test_basic_forward(self):
        """Test SDPA forward pass produces valid output."""
        set_impl('sdpa')
        B, T, H, D = 2, 64, 4, 32
        q = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE)
        k = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE)
        v = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE)

        y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=(T, 0))

        assert y.shape == (B, T, H, D)
        assert not torch.isnan(y).any(), "Output contains NaN"
        set_impl(None)

    def test_backward(self):
        """Test gradients flow through SDPA."""
        set_impl('sdpa')
        B, T, H, D = 2, 32, 4, 16
        q = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE, requires_grad=True)
        k = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE, requires_grad=True)
        v = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE, requires_grad=True)

        y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=(T, 0))
        loss = y.sum()
        loss.backward()

        assert q.grad is not None, "No gradient for q"
        assert k.grad is not None, "No gradient for k"
        assert v.grad is not None, "No gradient for v"
        assert not torch.isnan(q.grad).any(), "NaN in q gradient"
        set_impl(None)

    def test_kvcache(self):
        """Test SDPA with KV cache."""
        set_impl('sdpa')
        B, T_max, H, D = 2, 64, 4, 32
        n_layers = 1

        cache = KVCache(
            batch_size=B, num_heads=H, seq_len=T_max, head_dim=D,
            num_layers=n_layers, device=self.DEVICE, dtype=self.DTYPE
        )
        k_cache, v_cache = cache.get_layer_cache(0)

        # Prefill
        T_prefill = 16
        q = torch.randn(B, T_prefill, H, D, device=self.DEVICE, dtype=self.DTYPE)
        k = torch.randn(B, T_prefill, H, D, device=self.DEVICE, dtype=self.DTYPE)
        v = torch.randn(B, T_prefill, H, D, device=self.DEVICE, dtype=self.DTYPE)

        y = flash_attn.flash_attn_with_kvcache(
            q, k_cache, v_cache, k=k, v=v,
            cache_seqlens=cache.cache_seqlens,
            causal=True, window_size=(T_max, 0)
        )
        cache.advance(T_prefill)

        assert y.shape == (B, T_prefill, H, D)
        assert cache.get_pos() == T_prefill

        # Generate single token
        q_single = torch.randn(B, 1, H, D, device=self.DEVICE, dtype=self.DTYPE)
        k_single = torch.randn(B, 1, H, D, device=self.DEVICE, dtype=self.DTYPE)
        v_single = torch.randn(B, 1, H, D, device=self.DEVICE, dtype=self.DTYPE)

        y_single = flash_attn.flash_attn_with_kvcache(
            q_single, k_cache, v_cache, k=k_single, v=v_single,
            cache_seqlens=cache.cache_seqlens,
            causal=True, window_size=(T_max, 0)
        )
        cache.advance(1)

        assert y_single.shape == (B, 1, H, D)
        assert cache.get_pos() == T_prefill + 1
        set_impl(None)


# =============================================================================
# Override mechanism tests
# =============================================================================
class TestOverrideMechanism:
    """Test that the override mechanism works correctly."""

    @pytest.mark.skipif(not HAS_FA3, reason="FA3 required")
    def test_override_fa3(self):
        """Test that override='fa3' uses FA3."""
        set_impl('fa3')
        assert fa_module.USE_FA3 == True
        assert fa_module.ATTN_IMPL == 'fa3'
        set_impl(None)

    @pytest.mark.skipif(not HAS_FA2, reason="FA2 required")
    def test_override_fa2(self):
        """Test that override='fa2' uses FA2."""
        set_impl('fa2')
        assert fa_module.USE_FA2 == True
        assert fa_module.USE_FA3 == False
        assert fa_module.USE_FA4 == False
        assert fa_module.ATTN_IMPL == 'fa2'
        set_impl(None)

    @pytest.mark.skipif(not HAS_FA4, reason="FA4 required")
    def test_override_fa4(self):
        """Test that override='fa4' uses FA4."""
        set_impl('fa4')
        assert fa_module.USE_FA4 == True
        assert fa_module.USE_FA3 == False
        assert fa_module.USE_FA2 == False
        assert fa_module.ATTN_IMPL == 'fa4'
        set_impl(None)

    def test_override_sdpa(self):
        """Test that override='sdpa' uses SDPA."""
        set_impl('sdpa')
        assert fa_module.USE_FA3 == False
        assert fa_module.USE_FA4 == False
        assert fa_module.USE_FA2 == False
        assert fa_module.ATTN_IMPL == 'sdpa'
        set_impl(None)

    def test_override_auto(self):
        """Test that override=None uses auto-detection."""
        from nanochat.common import COMPUTE_DTYPE
        set_impl(None)
        if HAS_FA3 and COMPUTE_DTYPE == torch.bfloat16:
            expected = 'fa3'
        elif COMPUTE_DTYPE in (torch.bfloat16, torch.float16):
            major = torch.cuda.get_device_capability()[0] if torch.cuda.is_available() else 0
            if major in (10, 11) and HAS_FA4:
                expected = 'fa4'
            elif HAS_FA2:
                expected = 'fa2'
            elif HAS_FA4:
                expected = 'fa4'
            else:
                expected = 'sdpa'
        else:
            expected = 'sdpa'
        assert fa_module.ATTN_IMPL == expected
        assert fa_module.USE_FA3 == (expected == 'fa3')
        assert fa_module.USE_FA2 == (expected == 'fa2')
        assert fa_module.USE_FA4 == (expected == 'fa4')

    def test_override_invalid(self):
        """Test that an unknown backend name is rejected."""
        with pytest.raises(AssertionError):
            set_impl('fa5')
        set_impl(None)


if __name__ == "__main__":
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")
        major, minor = torch.cuda.get_device_capability()
        print(f"Compute capability: {major}.{minor}")
    print(f"HAS_FA3: {HAS_FA3}")
    print(f"HAS_FA2: {HAS_FA2}")
    print(f"HAS_FA4: {HAS_FA4}")
    print()

    pytest.main([__file__, "-v", "-s"])
