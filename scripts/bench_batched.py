import argparse, time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["numpy", "torch", "jax"], default="numpy")
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--degree", type=int, default=16)
    args = ap.parse_args()

    from geodepoly.batched import batched_poly_eval

    B, D = args.batch, args.degree
    if args.backend == "numpy":
        import numpy as xp
    elif args.backend == "torch":
        import torch as xp
    else:
        import jax.numpy as xp

    rng = None
    if args.backend == "numpy":
        import numpy as np

        rng = np.random.default_rng(0)
        coeffs = (rng.standard_normal((B, D + 1)) + 1j * rng.standard_normal((B, D + 1))).astype(complex)
        xs = (rng.standard_normal((B, 8)) + 1j * rng.standard_normal((B, 8))).astype(complex)
    elif args.backend == "torch":
        import torch

        torch.manual_seed(0)
        coeffs = torch.randn(B, D + 1, dtype=torch.cdouble)
        xs = torch.randn(B, 8, dtype=torch.cdouble)
        if torch.cuda.is_available():
            coeffs = coeffs.cuda()
            xs = xs.cuda()
    else:
        import jax
        import jax.random as jr

        key = jr.PRNGKey(0)
        coeffs = (jr.normal(key, (B, D + 1)) + 1j * jr.normal(key, (B, D + 1))).astype(xp.complex128)
        xs = (jr.normal(key, (B, 8)) + 1j * jr.normal(key, (B, 8))).astype(xp.complex128)

    # warmup
    y = batched_poly_eval(coeffs, xs, backend=args.backend)
    if args.backend == "torch":
        xp.cuda.synchronize() if hasattr(xp, "cuda") and xp.cuda.is_available() else None
    t0 = time.perf_counter()
    y = batched_poly_eval(coeffs, xs, backend=args.backend)
    if args.backend == "torch":
        xp.cuda.synchronize() if hasattr(xp, "cuda") and xp.cuda.is_available() else None
    dt = time.perf_counter() - t0
    print(f"backend={args.backend} B={B} D={D} time={dt:.4f}s")


if __name__ == "__main__":
    main()


