"""Triton-accelerated kernels for ComPoM (Polynomial Mixer).

The hot path in polynomial_aggregation_ (no-mask branch) is:
  1. pom_activation  (element-wise)
  2. polynomial weighted sum  producing (B, N, D)
  3. mean over N  producing (B, 1, D)

PyTorch materialises a (B, N, D, K) intermediate before the mean.
The kernels here fuse all three steps into a single pass over the
input, eliminating the intermediate and halving memory bandwidth.

Exposed API
-----------
TRITON_AVAILABLE : bool
poly_agg_mean_triton(x, coeff, k) -> Tensor  (B, 1, D)
"""
import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False


# ---------------------------------------------------------------------------
# Forward kernel
# ---------------------------------------------------------------------------

if TRITON_AVAILABLE:

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

        acc = tl.zeros((BLOCK_D,), dtype=tl.float32)

        for n in range(N):
            x = tl.load(
                X_ptr + b * stride_xb + n * stride_xn + d_off,
                mask=dmask, other=0.0,
            ).to(tl.float32)

            # pom_activation: clamp(leaky_relu(x, 0.01), -0.1, 6)
            h = tl.where(x >= 0.0, x, x * 0.01)
            h = tl.maximum(h, -0.1)
            h = tl.minimum(h,  6.0)

            # Polynomial weighted sum: Σ_k  coeff[d, k] * h^(k+1)
            # K is constexpr so the inner loop is fully unrolled.
            # Coeff loads are loop-invariant in n; the compiler hoists them.
            poly = tl.zeros((BLOCK_D,), dtype=tl.float32)
            hp   = h                       # h^1
            for k in range(K):
                ck   = tl.load(C_ptr + d_off * K + k, mask=dmask, other=0.0)
                poly = poly + ck * hp
                hp   = hp * h

            acc = acc + poly

        tl.store(O_ptr + b * D + d_off, acc / N, mask=dmask)

    # ---------------------------------------------------------------------------
    # Backward kernel – grad w.r.t. X
    # ---------------------------------------------------------------------------

    @triton.jit
    def _poly_agg_mean_bwd_x(
        GO_ptr,         # (B, D)    – upstream gradient (squeezed)
        X_ptr,          # (B, N, D)
        C_ptr,          # (D, K)
        GX_ptr,         # (B, N, D) – output gradient
        N, D,
        stride_xb, stride_xn,
        K: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        b     = tl.program_id(0)
        d_blk = tl.program_id(1)
        d_off = d_blk * BLOCK_D + tl.arange(0, BLOCK_D)
        dmask = d_off < D

        # Load upstream gradient once; reused for every n
        go = tl.load(GO_ptr + b * D + d_off, mask=dmask, other=0.0).to(tl.float32)
        inv_N = 1.0 / N

        for n in range(N):
            x = tl.load(
                X_ptr + b * stride_xb + n * stride_xn + d_off,
                mask=dmask, other=0.0,
            ).to(tl.float32)

            # Recompute activation (same as forward)
            h = tl.where(x >= 0.0, x, x * 0.01)
            h = tl.maximum(h, -0.1)
            h = tl.minimum(h,  6.0)

            # d(pom_activation)/dx
            #   1    for  0  <= x <= 6
            #   0.01 for -10 <= x <  0   (leaky region, not clamped)
            #   0    otherwise            (clamped: x < -10 or x > 6)
            d_act = tl.where(
                (x >= 0.0) & (x <= 6.0), 1.0,
                tl.where((x < 0.0) & (x >= -10.0), 0.01, 0.0),
            )

            # Σ_k  coeff[d,k] * (k+1) * h^k   (derivative of poly sum w.r.t. h)
            grad_h = tl.zeros((BLOCK_D,), dtype=tl.float32)
            hp     = tl.full((BLOCK_D,), 1.0, dtype=tl.float32)   # h^0
            for k in range(K):
                ck     = tl.load(C_ptr + d_off * K + k, mask=dmask, other=0.0)
                grad_h = grad_h + ck * (k + 1) * hp
                hp     = hp * h

            grad_x = go * inv_N * grad_h * d_act
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

            BLOCK_D = min(512, triton.next_power_of_2(D))
            grid    = (B, triton.cdiv(D, BLOCK_D))
            out     = torch.empty(B, D, dtype=torch.float32, device=x.device)

            _poly_agg_mean_fwd[grid](
                x_c, coeff_c, out,
                N, D,
                x_c.stride(0), x_c.stride(1),
                K=k, BLOCK_D=BLOCK_D,
            )

            ctx.save_for_backward(x_c, coeff_c)
            ctx.k       = k
            ctx.N       = N
            ctx.D       = D
            ctx.BLOCK_D = BLOCK_D
            return out.unsqueeze(1).to(x.dtype)

        @staticmethod
        def backward(ctx, grad_out: torch.Tensor):
            x, coeff = ctx.saved_tensors
            k        = ctx.k
            N, D     = ctx.N, ctx.D
            B        = x.shape[0]

            go   = grad_out.squeeze(1).float().contiguous()   # (B, D)
            grid = (B, triton.cdiv(D, ctx.BLOCK_D))

            # Gradient w.r.t. x – via Triton
            grad_x = torch.empty_like(x)
            _poly_agg_mean_bwd_x[grid](
                go, x, coeff, grad_x,
                N, D,
                x.stride(0), x.stride(1),
                K=k, BLOCK_D=ctx.BLOCK_D,
            )

            # Gradient w.r.t. coeff – via PyTorch (sum over B and N)
            # grad_coeff[d, k] = (1/N) Σ_{b,n}  h[b,n,d]^(k+1) * go[b,d]
            from compom import pom_activation
            h         = pom_activation(x.float())            # (B, N, D)
            go_exp    = go.unsqueeze(1)                       # (B, 1, D)
            grad_c    = torch.zeros_like(coeff)
            hp        = h
            for ki in range(k):
                grad_c[:, ki] = (hp * go_exp / N).sum(dim=(0, 1))
                hp = hp * h

            return grad_x.to(x.dtype), grad_c.to(coeff.dtype), None

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
            k     : polynomial degree

        Returns:
            (B, 1, D) mean-aggregated polynomial features
        """
        return _PolyAggMean.apply(x, coeff, k)
