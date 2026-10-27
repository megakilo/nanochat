"""
Test TPU Splash Attention / SDPA unified interface.

Run: python -m pytest tests/test_attention_fallback.py -v -s
"""
import torch
import pytest
import nanochat.flash_attention as fa_module
from nanochat.flash_attention import flash_attn, has_splash
from nanochat.engine import KVCache

HAS_SPLASH = has_splash()


def set_impl(impl):
    """Set the implementation override ('splash', 'sdpa', or None for auto) and re-resolve the backend."""
    fa_module.set_attn_impl(impl)


def run_both_impls(fn, impl='splash'):
    """Run a function with both Splash Attention and SDPA, return both outputs."""
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


@pytest.mark.skipif(not HAS_SPLASH, reason="TPU Splash Attention required to compare implementations")
class TestSplashVsSDPA:
    """Compare Splash Attention and SDPA produce identical results on TPU."""

    IMPL = "splash"
    DEVICE = "tpu"
    DTYPE = torch.bfloat16

    def test_basic_causal(self):
        B, T, H, D = 2, 128, 4, 128
        q = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE)
        k = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE)
        v = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE)

        def run():
            return flash_attn.flash_attn_func(q, k, v, causal=True, window_size=(T, 0))

        y_flash, y_sdpa = run_both_impls(run, self.IMPL)
        assert_close(y_flash, y_sdpa, "basic_causal")

    def test_sliding_window(self):
        B, T, H, D = 2, 256, 4, 128
        window = 128
        q = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE)
        k = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE)
        v = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE)

        def run():
            return flash_attn.flash_attn_func(q, k, v, causal=True, window_size=(window, 0))

        y_flash, y_sdpa = run_both_impls(run, self.IMPL)
        assert_close(y_flash, y_sdpa, "sliding_window")


# =============================================================================
# SDPA-only tests (run on any device)
# =============================================================================
class TestSDPAOnly:
    """Test SDPA fallback works correctly. Runs on any device."""

    DEVICE = "cpu"
    DTYPE = torch.float32

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

    @pytest.mark.skipif(not HAS_SPLASH, reason="Splash Attention required")
    def test_override_splash(self):
        set_impl('splash')
        assert fa_module.USE_SPLASH is True
        assert fa_module.ATTN_IMPL == 'splash'
        set_impl(None)

    def test_override_sdpa(self):
        set_impl('sdpa')
        assert fa_module.USE_SPLASH is False
        assert fa_module.ATTN_IMPL == 'sdpa'
        set_impl(None)

    def test_override_auto(self):
        set_impl(None)
        expected = 'splash' if fa_module.has_splash() else 'sdpa'
        assert fa_module.ATTN_IMPL == expected
        assert fa_module.USE_SPLASH == (expected == 'splash')

    def test_override_invalid(self):
        with pytest.raises(AssertionError):
            set_impl('fa5')
        set_impl(None)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
