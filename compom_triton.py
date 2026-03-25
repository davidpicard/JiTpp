"""Triton-accelerated kernels for ComPoM (Polynomial Mixer).

The hot path in polynomial_aggregation_ (no-mask branch) is:
  1. pom_activation  (element-wise)
  2. polynomial weighted sum  producing (B, N, D)
  3. mean over N  producing (B, 1, D)

PyTorch materialises a (B, N, D, K) intermediate before the mean.
The kernels here fuse all three steps into a single pass over the
input, eliminating the intermediate and halving memory bandwidth.

Key optimisations vs a naïve port:
  - coeff[d, k] is preloaded once per (b, d_blk) program and held in
    registers for the entire N loop (main memory-placement change).
  - X is a streaming tensor: each element is used once then discarded.
    eviction_policy="evict_first" hints the cache not to pollute L2.
  - cache_modifier=".ca" on coeff loads keeps them in L1 as a fallback
    if register spill occurs under high-BLOCK_D / high-K configs.
  - K is tl.constexpr so the inner polynomial loop is fully unrolled.
  - num_stages is autotuned: higher values pipeline X loads behind
    arithmetic, hiding global-memory latency.
  - Grid uses a lambda so autotune's chosen BLOCK_D is respected.
  - grad_x is accumulated in fp32 and cast to the input dtype in Python
    (avoids an implicit fp32→bf16 store inside the kernel).

Exposed API
-----------
TRITON_AVAILABLE : bool
poly_agg_mean_triton(x, coeff, k) -> Tensor  (B, 1, D)
"""
import os
import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = not os.environ.get("POM_DISABLE_TRITON", "")
except ImportError:
    TRITON_AVAILABLE = False


# ---------------------------------------------------------------------------
# Autotune config lists
# ---------------------------------------------------------------------------

if TRITON_AVAILABLE:

    _FWD_CONFIGS = [
        triton.Config({"BLOCK_D":  64}, num_warps=2, num_stages=3),
        triton.Config({"BLOCK_D":  64}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_D":  64}, num_warps=4, num_stages=6),
        triton.Config({"BLOCK_D": 128}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_D": 128}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_D": 128}, num_warps=4, num_stages=6),
        triton.Config({"BLOCK_D": 128}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_D": 256}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_D": 256}, num_warps=8, num_stages=3),
    ]

    _BWD_CONFIGS = [
        triton.Config({"BLOCK_D":  64}, num_warps=2, num_stages=3),
        triton.Config({"BLOCK_D":  64}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_D":  64}, num_warps=4, num_stages=6),
        triton.Config({"BLOCK_D": 128}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_D": 128}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_D": 128}, num_warps=4, num_stages=6),
        triton.Config({"BLOCK_D": 128}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_D": 256}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_D": 256}, num_warps=8, num_stages=3),
    ]

    # ---------------------------------------------------------------------------
    # Forward kernel
    # ---------------------------------------------------------------------------

    @triton.autotune(configs=_FWD_CONFIGS, key=["N", "D", "K"])
    @triton.jit
    def _poly_agg_mean_fwd(
        X_ptr,          # (B, N, D) – input, any dtype
        C_ptr,          # (D, K)    – polynomial coefficients, fp32
        O_ptr,          # (B, D)    – output, fp32
        N,              # sequence length (runtime)
        D,              # feature dim    (runtime)
        stride_xb,      # X.stride(0) = N*D
        stride_xn,      # X.stride(1) = D
        K: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        b     = tl.program_id(0)
        d_blk = tl.program_id(1)
        d_off = d_blk * BLOCK_D + tl.arange(0, BLOCK_D)
        dmask = d_off < D

        # Preload coeff[d_off, 0..K-1] once per program.
        # These BLOCK_D-wide vectors are held in registers for the entire N loop.
        # cache_modifier=".ca" (cache-all) provides an L1 fallback if K is large
        # enough to cause register spill; harmless otherwise.
        c0  = tl.load(C_ptr + d_off * K + 0,  mask=dmask, other=0.0, cache_modifier=".ca") if K > 0  else 0.0
        c1  = tl.load(C_ptr + d_off * K + 1,  mask=dmask, other=0.0, cache_modifier=".ca") if K > 1  else 0.0
        c2  = tl.load(C_ptr + d_off * K + 2,  mask=dmask, other=0.0, cache_modifier=".ca") if K > 2  else 0.0
        c3  = tl.load(C_ptr + d_off * K + 3,  mask=dmask, other=0.0, cache_modifier=".ca") if K > 3  else 0.0
        c4  = tl.load(C_ptr + d_off * K + 4,  mask=dmask, other=0.0, cache_modifier=".ca") if K > 4  else 0.0
        c5  = tl.load(C_ptr + d_off * K + 5,  mask=dmask, other=0.0, cache_modifier=".ca") if K > 5  else 0.0
        c6  = tl.load(C_ptr + d_off * K + 6,  mask=dmask, other=0.0, cache_modifier=".ca") if K > 6  else 0.0
        c7  = tl.load(C_ptr + d_off * K + 7,  mask=dmask, other=0.0, cache_modifier=".ca") if K > 7  else 0.0
        c8  = tl.load(C_ptr + d_off * K + 8,  mask=dmask, other=0.0, cache_modifier=".ca") if K > 8  else 0.0
        c9  = tl.load(C_ptr + d_off * K + 9,  mask=dmask, other=0.0, cache_modifier=".ca") if K > 9  else 0.0
        c10 = tl.load(C_ptr + d_off * K + 10, mask=dmask, other=0.0, cache_modifier=".ca") if K > 10 else 0.0
        c11 = tl.load(C_ptr + d_off * K + 11, mask=dmask, other=0.0, cache_modifier=".ca") if K > 11 else 0.0
        c12 = tl.load(C_ptr + d_off * K + 12, mask=dmask, other=0.0, cache_modifier=".ca") if K > 12 else 0.0
        c13 = tl.load(C_ptr + d_off * K + 13, mask=dmask, other=0.0, cache_modifier=".ca") if K > 13 else 0.0
        c14 = tl.load(C_ptr + d_off * K + 14, mask=dmask, other=0.0, cache_modifier=".ca") if K > 14 else 0.0
        c15 = tl.load(C_ptr + d_off * K + 15, mask=dmask, other=0.0, cache_modifier=".ca") if K > 15 else 0.0

        acc = tl.zeros((BLOCK_D,), dtype=tl.float32)

        for n in range(N):
            # X is streaming: each element is read once then never needed again.
            # evict_first frees cache lines immediately, keeping L2 pressure low.
            x = tl.load(
                X_ptr + b * stride_xb + n * stride_xn + d_off,
                mask=dmask, other=0.0,
                eviction_policy="evict_first",
            ).to(tl.float32)

            # pom_activation: clamp(leaky_relu(x, 0.01), -0.1, 6)
            h = tl.where(x >= 0.0, x, x * 0.01)
            h = tl.maximum(h, -0.1)
            h = tl.minimum(h,  6.0)

            # Polynomial weighted sum: Σ_k  coeff[d, k] * h^(k+1)
            # K is constexpr so the compiler fully unrolls this.
            poly = tl.zeros((BLOCK_D,), dtype=tl.float32)
            hp   = h                        # h^1
            if K > 0:
                poly += c0 * hp
            if K > 1:
                hp *= h; poly += c1 * hp    # h^2
            if K > 2:
                hp *= h; poly += c2 * hp    # h^3
            if K > 3:
                hp *= h; poly += c3 * hp
            if K > 4:
                hp *= h; poly += c4 * hp
            if K > 5:
                hp *= h; poly += c5 * hp
            if K > 6:
                hp *= h; poly += c6 * hp
            if K > 7:
                hp *= h; poly += c7 * hp
            if K > 8:
                hp *= h; poly += c8 * hp
            if K > 9:
                hp *= h; poly += c9 * hp
            if K > 10:
                hp *= h; poly += c10 * hp
            if K > 11:
                hp *= h; poly += c11 * hp
            if K > 12:
                hp *= h; poly += c12 * hp
            if K > 13:
                hp *= h; poly += c13 * hp
            if K > 14:
                hp *= h; poly += c14 * hp
            if K > 15:
                hp *= h; poly += c15 * hp

            acc += poly

        tl.store(O_ptr + b * D + d_off, acc / N, mask=dmask)

    # ---------------------------------------------------------------------------
    # Backward kernel – grad w.r.t. X
    # ---------------------------------------------------------------------------

    @triton.autotune(configs=_BWD_CONFIGS, key=["N", "D", "K"])
    @triton.jit
    def _poly_agg_mean_bwd_x(
        GO_ptr,         # (B, D)    – upstream gradient (squeezed), fp32
        X_ptr,          # (B, N, D) – saved input from forward
        C_ptr,          # (D, K)    – polynomial coefficients, fp32
        GX_ptr,         # (B, N, D) – output gradient, fp32
        N, D,
        stride_xb, stride_xn,
        K: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        b     = tl.program_id(0)
        d_blk = tl.program_id(1)
        d_off = d_blk * BLOCK_D + tl.arange(0, BLOCK_D)
        dmask = d_off < D

        # Preload upstream gradient and coeff – both reused across all N.
        go    = tl.load(GO_ptr + b * D + d_off, mask=dmask, other=0.0).to(tl.float32)
        inv_N = 1.0 / N

        c0  = tl.load(C_ptr + d_off * K + 0,  mask=dmask, other=0.0, cache_modifier=".ca") if K > 0  else 0.0
        c1  = tl.load(C_ptr + d_off * K + 1,  mask=dmask, other=0.0, cache_modifier=".ca") if K > 1  else 0.0
        c2  = tl.load(C_ptr + d_off * K + 2,  mask=dmask, other=0.0, cache_modifier=".ca") if K > 2  else 0.0
        c3  = tl.load(C_ptr + d_off * K + 3,  mask=dmask, other=0.0, cache_modifier=".ca") if K > 3  else 0.0
        c4  = tl.load(C_ptr + d_off * K + 4,  mask=dmask, other=0.0, cache_modifier=".ca") if K > 4  else 0.0
        c5  = tl.load(C_ptr + d_off * K + 5,  mask=dmask, other=0.0, cache_modifier=".ca") if K > 5  else 0.0
        c6  = tl.load(C_ptr + d_off * K + 6,  mask=dmask, other=0.0, cache_modifier=".ca") if K > 6  else 0.0
        c7  = tl.load(C_ptr + d_off * K + 7,  mask=dmask, other=0.0, cache_modifier=".ca") if K > 7  else 0.0
        c8  = tl.load(C_ptr + d_off * K + 8,  mask=dmask, other=0.0, cache_modifier=".ca") if K > 8  else 0.0
        c9  = tl.load(C_ptr + d_off * K + 9,  mask=dmask, other=0.0, cache_modifier=".ca") if K > 9  else 0.0
        c10 = tl.load(C_ptr + d_off * K + 10, mask=dmask, other=0.0, cache_modifier=".ca") if K > 10 else 0.0
        c11 = tl.load(C_ptr + d_off * K + 11, mask=dmask, other=0.0, cache_modifier=".ca") if K > 11 else 0.0
        c12 = tl.load(C_ptr + d_off * K + 12, mask=dmask, other=0.0, cache_modifier=".ca") if K > 12 else 0.0
        c13 = tl.load(C_ptr + d_off * K + 13, mask=dmask, other=0.0, cache_modifier=".ca") if K > 13 else 0.0
        c14 = tl.load(C_ptr + d_off * K + 14, mask=dmask, other=0.0, cache_modifier=".ca") if K > 14 else 0.0
        c15 = tl.load(C_ptr + d_off * K + 15, mask=dmask, other=0.0, cache_modifier=".ca") if K > 15 else 0.0

        for n in range(N):
            x = tl.load(
                X_ptr + b * stride_xb + n * stride_xn + d_off,
                mask=dmask, other=0.0,
                eviction_policy="evict_first",
            ).to(tl.float32)

            # Recompute activation (same as forward)
            h = tl.where(x >= 0.0, x, x * 0.01)
            h = tl.maximum(h, -0.1)
            h = tl.minimum(h,  6.0)

            # d(pom_activation)/dx
            d_act = tl.where(
                (x >= 0.0) & (x <= 6.0), 1.0,
                tl.where((x < 0.0) & (x >= -10.0), 0.01, 0.0),
            )

            # d(poly)/dh = Σ_k  coeff[d,k] * (k+1) * h^k
            grad_h = tl.zeros((BLOCK_D,), dtype=tl.float32)
            hp     = tl.full((BLOCK_D,), 1.0, dtype=tl.float32)   # h^0
            if K > 0:
                grad_h += c0 * hp                   # 1 * c0 * h^0
            if K > 1:
                hp *= h; grad_h += c1 * 2.0 * hp    # 2 * c1 * h^1
            if K > 2:
                hp *= h; grad_h += c2 * 3.0 * hp
            if K > 3:
                hp *= h; grad_h += c3 * 4.0 * hp
            if K > 4:
                hp *= h; grad_h += c4 * 5.0 * hp
            if K > 5:
                hp *= h; grad_h += c5 * 6.0 * hp
            if K > 6:
                hp *= h; grad_h += c6 * 7.0 * hp
            if K > 7:
                hp *= h; grad_h += c7 * 8.0 * hp
            if K > 8:
                hp *= h; grad_h += c8 * 9.0 * hp
            if K > 9:
                hp *= h; grad_h += c9 * 10.0 * hp
            if K > 10:
                hp *= h; grad_h += c10 * 11.0 * hp
            if K > 11:
                hp *= h; grad_h += c11 * 12.0 * hp
            if K > 12:
                hp *= h; grad_h += c12 * 13.0 * hp
            if K > 13:
                hp *= h; grad_h += c13 * 14.0 * hp
            if K > 14:
                hp *= h; grad_h += c14 * 15.0 * hp
            if K > 15:
                hp *= h; grad_h += c15 * 16.0 * hp

            grad_x = go * inv_N * grad_h * d_act
            # GX_ptr is fp32 (allocated explicitly in the Python wrapper);
            # no implicit dtype conversion at store time.
            tl.store(
                GX_ptr + b * stride_xb + n * stride_xn + d_off,
                grad_x,
                mask=dmask,
            )

    # ---------------------------------------------------------------------------
    # autograd.Function wrapper
    # ---------------------------------------------------------------------------

    class _PolyAggMean(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x: torch.Tensor, coeff: torch.Tensor, k: int):
            B, N, D = x.shape
            x_c     = x.contiguous()
            coeff_c = coeff.float().contiguous()
            out     = torch.empty(B, D, dtype=torch.float32, device=x.device)

            # Grid lambda: lets autotune's chosen BLOCK_D determine the tile count.
            # A hardcoded cdiv(D, 256) would produce a wrong grid for other configs.
            grid = lambda meta: (B, triton.cdiv(D, meta["BLOCK_D"]))
            _poly_agg_mean_fwd[grid](
                x_c, coeff_c, out,
                N, D,
                x_c.stride(0), x_c.stride(1),
                K=k,
            )

            ctx.save_for_backward(x_c, coeff_c)
            ctx.k = k
            return out.unsqueeze(1).to(x.dtype)

        @staticmethod
        def backward(ctx, grad_out: torch.Tensor):
            x, coeff = ctx.saved_tensors
            k        = ctx.k
            B, N, D  = x.shape

            go = grad_out.squeeze(1).float().contiguous()   # (B, D)

            # Allocate grad_x as fp32 so the Triton kernel stores without any
            # implicit dtype conversion; we cast to x.dtype afterwards.
            grad_x_buf = torch.empty(B, N, D, dtype=torch.float32, device=x.device)
            grid = lambda meta: (B, triton.cdiv(D, meta["BLOCK_D"]))
            _poly_agg_mean_bwd_x[grid](
                go, x, coeff, grad_x_buf,
                N, D,
                x.stride(0), x.stride(1),
                K=k,
            )

            # Gradient w.r.t. coeff – via PyTorch (sum over B and N)
            # grad_coeff[d, k] = (1/N) Σ_{b,n}  h[b,n,d]^(k+1) * go[b,d]
            from compom import pom_activation
            h      = pom_activation(x.float())              # (B, N, D)
            go_exp = go.unsqueeze(1)                         # (B, 1, D)
            grad_c = torch.zeros_like(coeff)
            hp     = h
            for ki in range(k):
                grad_c[:, ki] = (hp * go_exp / N).sum(dim=(0, 1))
                hp = hp * h

            return grad_x_buf.to(x.dtype), grad_c.to(coeff.dtype), None

    # ---------------------------------------------------------------------------
    # Public entry point
    # ---------------------------------------------------------------------------

    def poly_agg_mean_triton(
        x: torch.Tensor,
        coeff: torch.Tensor,
        k: int,
    ) -> torch.Tensor:
        """Fused polynomial aggregation + mean (no mask).

        Equivalent to the mask=None branch of polynomial_aggregation_ but
        with a single-pass Triton kernel that avoids the (B, N, D, K)
        intermediate.

        Args:
            x     : (B, N, D) input tensor
            coeff : (D, K)    polynomial coefficients
            k     : polynomial degree (≤ 16)

        Returns:
            (B, 1, D) mean-aggregated polynomial features
        """
        if k > 16:
            raise NotImplementedError(
                f"Triton kernel supports k ≤ 16 (got {k}). "
                "Extend the c0..c15 pattern or use the PyTorch fallback."
            )
        return _PolyAggMean.apply(x, coeff, k)
