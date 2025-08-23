#!/usr/bin/env python3
import argparse
import time


def make_data(B: int, D: int, backend: str, device: str | None = None):
    if backend == "numpy":
        import numpy as np

        rng = np.random.default_rng(0)
        coeffs = rng.standard_normal((B, D + 1)) + 1j * rng.standard_normal((B, D + 1))
        xs = rng.standard_normal((B,)) + 1j * rng.standard_normal((B,))
        return coeffs.astype(complex), xs.astype(complex)
    elif backend == "torch":
        import torch

        torch.manual_seed(0)
        coeffs = torch.randn(B, D + 1, dtype=torch.complex64)
        xs = torch.randn(B, dtype=torch.complex64)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        coeffs = coeffs.to(device)
        xs = xs.to(device)
        return coeffs, xs
    else:
        import jax
        import jax.numpy as jnp

        key = jax.random.PRNGKey(0)
        c1, c2 = jax.random.normal(key, (B, D + 1)), jax.random.normal(key, (B, D + 1))
        coeffs = (c1 + 1j * c2).astype(jnp.complex64)
        x1, x2 = jax.random.normal(key, (B,)), jax.random.normal(key, (B,))
        xs = (x1 + 1j * x2).astype(jnp.complex64)
        return coeffs, xs


def main():
    ap = argparse.ArgumentParser(description="GPU prototype bench: batched Newton steps")
    ap.add_argument("--backend", type=str, default="numpy", choices=["numpy", "torch", "jax"])
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--degree", type=int, default=16)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--device", type=str, default=None, help="torch device (cpu/cuda)")
    args = ap.parse_args()

    from geodepoly.batched import batched_newton_step, batched_poly_eval

    coeffs, xs = make_data(args.batch, args.degree, args.backend, device=args.device)

    # Warmup
    for _ in range(3):
        xs = batched_newton_step(coeffs, xs, backend=args.backend)

    t0 = time.perf_counter()
    for _ in range(args.steps):
        xs = batched_newton_step(coeffs, xs, backend=args.backend)
    dt = time.perf_counter() - t0

    # Residuals
    p = batched_poly_eval(coeffs, xs, backend=args.backend)
    if args.backend == "numpy":
        import numpy as np

        max_res = float(np.max(np.abs(p)))
    elif args.backend == "torch":
        import torch

        max_res = float(torch.max(torch.abs(p)).item())
    else:
        import jax.numpy as jnp

        max_res = float(jnp.max(jnp.abs(p)))

    steps_per_sec = (args.batch * args.steps) / dt if dt > 0 else float("inf")
    print(f"backend={args.backend} batch={args.batch} degree={args.degree} steps={args.steps}")
    print(f"time={dt:.4f}s steps/sec={steps_per_sec:,.0f} max|p(x)|={max_res:.2e}")


if __name__ == "__main__":
    main()


