#!/usr/bin/env python3
"""Correctness verification and benchmarks for the Triton PoM kernels.

Correctness
-----------
  Compares Triton forward/backward against the reference PyTorch
  implementation across a range of (N, D, K) configs and dtypes.

Benchmark
---------
  Times the no-mask polynomial_aggregation_ path for
    N  ∈ {256, 512, 1024}
    D  ∈ {256, 512, 768, 1024}
  with B = 4, K = 3 (the common training case).

Usage:
    python bench_pom.py
"""
import sys
import torch

# ── make sure we're on GPU ────────────────────────────────────────────────────
if not torch.cuda.is_available():
    sys.exit("No CUDA device found – skipping benchmark.")

device = torch.device("cuda")

from compom_triton import TRITON_AVAILABLE, poly_agg_mean_triton
from compom import pom_activation, polynomial_aggregation_

if not TRITON_AVAILABLE:
    sys.exit("Triton not available – nothing to benchmark.")

print(f"Triton available: {TRITON_AVAILABLE}")
print()


# =============================================================================
# Reference PyTorch implementation (always pure-PT, no Triton dispatch)
# =============================================================================

def poly_agg_mean_pt(x: torch.Tensor, coeff: torch.Tensor, k: int) -> torch.Tensor:
    """Pure PyTorch polynomial aggregation + mean (reference)."""
    # Use non-inplace activation so grad-tracked inputs work
    h = torch.clamp(torch.nn.functional.leaky_relu(x, 0.01, inplace=False), -0.1, 6.0)
    h = h.unsqueeze(-1)                            # (B, N, D, 1)
    h = torch.cat([h ** (i + 1) for i in range(k)], dim=-1)  # (B, N, D, K)
    h = (h * coeff).sum(-1)                        # (B, N, D)
    return h.mean(dim=1, keepdim=True)             # (B, 1, D)


# =============================================================================
# Correctness
# =============================================================================

def check_close(a: torch.Tensor, b: torch.Tensor, tag: str,
                rtol: float = 1e-3, atol: float = 1e-3) -> bool:
    ok = torch.allclose(a.float(), b.float(), rtol=rtol, atol=atol)
    max_err = (a.float() - b.float()).abs().max().item()
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {tag}  max_err={max_err:.2e}")
    return ok


print("=" * 60)
print("CORRECTNESS CHECKS")
print("=" * 60)

all_pass = True

test_cases = [
    # (B, N, D, K, dtype)
    (2, 64,  256, 2, torch.float32),
    (2, 64,  256, 3, torch.float32),
    (2, 64,  256, 4, torch.float32),
    (2, 256, 512, 3, torch.float32),
    (2, 512, 768, 3, torch.float32),
    (4, 256, 512, 3, torch.bfloat16),
    (4, 512, 512, 3, torch.bfloat16),
]

for B, N, D, K, dt in test_cases:
    label = f"B={B} N={N} D={D} K={K} dtype={dt}"
    print(f"\n  {label}")

    torch.manual_seed(0)
    x      = torch.randn(B, N, D, device=device, dtype=dt)
    coeff  = torch.randn(D, K, device=device) * 0.01

    # ── Forward ───────────────────────────────────────────────────────────────
    ref_out  = poly_agg_mean_pt(x.float(), coeff, K)
    trt_out  = poly_agg_mean_triton(x, coeff, K)
    ok_fwd   = check_close(ref_out, trt_out, "forward")

    # ── Backward (grad_x and grad_coeff) ─────────────────────────────────────
    x_pt  = x.float().clone().detach().requires_grad_(True)
    x_tr  = x.clone().detach().requires_grad_(True)
    c_pt  = coeff.clone().detach().requires_grad_(True)
    c_tr  = coeff.clone().detach().requires_grad_(True)

    loss_pt = poly_agg_mean_pt(x_pt, c_pt, K).sum()
    loss_tr = poly_agg_mean_triton(x_tr, c_tr, K).to(torch.float32).sum()

    loss_pt.backward()
    loss_tr.backward()

    ok_gx = check_close(x_pt.grad, x_tr.grad.float(), "grad_x  ")
    ok_gc = check_close(c_pt.grad, c_tr.grad.float(), "grad_coeff")

    all_pass = all_pass and ok_fwd and ok_gx and ok_gc

print()
print("All correctness checks:", "PASSED ✓" if all_pass else "FAILED ✗")
if not all_pass:
    sys.exit(1)


# =============================================================================
# Benchmark helpers
# =============================================================================

def warmup_and_time(fn, *args, warmup: int = 5, reps: int = 50) -> float:
    """Returns median wall-time in milliseconds over `reps` runs."""
    import time

    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()

    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn(*args)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1e3)

    times.sort()
    return times[len(times) // 2]   # median


# =============================================================================
# Benchmark
# =============================================================================

print()
print("=" * 60)
print("BENCHMARK  (B=4, K=3, bfloat16, median over 50 runs)")
print("=" * 60)
print()

B = 4
K = 3

header = f"{'N':>6}  {'D':>6}  {'PyTorch (ms)':>14}  {'Triton (ms)':>13}  {'Speedup':>8}"
print(header)
print("-" * len(header))

Ns = [256, 512, 1024]
Ds = [256, 512, 768, 1024]

for N in Ns:
    for D in Ds:
        torch.manual_seed(0)
        x     = torch.randn(B, N, D, device=device, dtype=torch.bfloat16)
        coeff = torch.randn(D, K, device=device)

        t_pt = warmup_and_time(poly_agg_mean_pt, x.float(), coeff, K)
        t_tr = warmup_and_time(poly_agg_mean_triton, x, coeff, K)

        speedup = t_pt / t_tr
        print(f"{N:>6}  {D:>6}  {t_pt:>14.3f}  {t_tr:>13.3f}  {speedup:>7.2f}x")

    print()

print("Done.")
