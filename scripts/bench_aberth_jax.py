#!/usr/bin/env python3
import argparse
import time


def make_poly(deg: int):
    import numpy as np

    coeffs = np.array([1.0], dtype=complex)
    for k in range(1, deg + 1):
        coeffs = np.convolve(coeffs, np.array([-k, 1.0], dtype=complex))
    return coeffs


def main():
    ap = argparse.ArgumentParser(description="Bench JAX Aberth step")
    ap.add_argument("--deg", type=int, default=32)
    ap.add_argument("--iters", type=int, default=50)
    args = ap.parse_args()

    import jax
    import jax.numpy as jnp
    from geodepoly.batched import jax_aberth_solve

    coeffs_np = make_poly(args.deg)
    coeffs = jnp.array(coeffs_np, dtype=jnp.complex64)
    k = jnp.arange(args.deg, dtype=jnp.float32)
    theta = 2 * jnp.pi * (k / args.deg)
    z0 = 1.5 * jnp.exp(1j * theta)

    # JIT compile
    f = jax.jit(jax_aberth_solve)
    z = f(coeffs, z0, iters=1)
    _ = jnp.sum(jnp.abs(z)).block_until_ready()

    t0 = time.perf_counter()
    z = f(coeffs, z0, iters=args.iters)
    _ = jnp.sum(jnp.abs(z)).block_until_ready()
    dt = time.perf_counter() - t0
    print(f"deg={args.deg} iters={args.iters} JAX time={dt:.3f}s")


if __name__ == "__main__":
    main()



