import argparse, time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--degree", type=int, default=16)
    args = ap.parse_args()

    import jax
    import jax.numpy as jnp
    from geodepoly.ai import root_solve_jax

    B, D = args.batch, args.degree
    key = jax.random.PRNGKey(0)
    k1, k2 = jax.random.split(key)
    coeffs = (jax.random.normal(k1, (B, D + 1)) + 1j * jax.random.normal(k2, (B, D + 1))).astype(jnp.complex128)

    # JIT compile forward
    fwd = jax.jit(root_solve_jax)
    roots = fwd(coeffs)  # compile

    # Time forward
    t0 = time.perf_counter()
    roots = fwd(coeffs)
    jax.block_until_ready(roots)
    t_fwd = time.perf_counter() - t0

    # JIT compile backward
    def loss_fn(c):
        r = root_solve_jax(c)
        return jnp.mean(jnp.square(jnp.real(r)))

    val, grad = jax.jit(jax.value_and_grad(loss_fn))(coeffs)
    jax.block_until_ready(val)
    jax.block_until_ready(grad)

    # Time backward
    t0 = time.perf_counter()
    val, grad = jax.jit(jax.value_and_grad(loss_fn))(coeffs)
    jax.block_until_ready(val)
    jax.block_until_ready(grad)
    t_bwd = time.perf_counter() - t0

    print(f"JAX RootLayer: B={B} D={D} forward={t_fwd:.4f}s backward={t_bwd:.4f}s")


if __name__ == "__main__":
    main()


