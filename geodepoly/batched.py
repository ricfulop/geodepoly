from __future__ import annotations


def _as_backend(name: str):
    n = name.lower()
    if n == "numpy":
        import numpy as np  # type: ignore

        return np
    if n == "torch":
        import torch  # type: ignore

        return torch
    if n == "jax":
        import jax.numpy as jnp  # type: ignore

        return jnp
    raise ValueError("backend must be one of {'numpy','torch','jax'}")


def batched_poly_eval(coeffs, xs, backend: str = "numpy"):
    """Evaluate many polynomials at many points via Horner in the chosen backend.

    coeffs: shape (B, D+1) low-to-high
    xs:     shape (B,) or (B, M)
    returns shape (B,) or (B, M)
    """
    xp = _as_backend(backend)
    a = coeffs
    x = xs
    # normalize shapes
    if len(x.shape) == 1:
        x = x.reshape((-1, 1))
        squeezed = True
    else:
        squeezed = False
    # reversed for Horner (high-to-low)
    a_rev = a[..., ::-1]
    # p = 0; for c in a_rev: p = p*x + c
    # tile along M
    p = xp.zeros((x.shape[0], x.shape[1]), dtype=a_rev.dtype)
    for c in a_rev.T:
        p = p * x + c.reshape((-1, 1))
    if squeezed:
        p = p[:, 0]
    return p


def batched_newton_step(coeffs, xs, backend: str = "numpy"):
    """One Newton step x - p/ p' for each polynomial in the batch.

    coeffs: (B, D+1), xs: (B,) or (B, M)
    returns same shape as xs
    """
    xp = _as_backend(backend)
    x = xs
    if len(x.shape) == 1:
        x = x.reshape((-1, 1))
        squeezed = True
    else:
        squeezed = False
    # Horner for p and p'
    a_rev = coeffs[..., ::-1]
    p = xp.zeros_like(x, dtype=a_rev.dtype)
    dp = xp.zeros_like(x, dtype=a_rev.dtype)
    for c in a_rev.T:
        dp = dp * x + p
        p = p * x + c.reshape((-1, 1))
    step = x - p / xp.where(dp == 0, xp.asarray(1, dtype=dp.dtype), dp)
    if squeezed:
        step = step[:, 0]
    return step


def torch_root_layer(steps: int = 3, tol: float = 0.0):
    """Return a small torch.nn.Module performing several Newton steps.

    This is differentiable because it composes Torch ops.
    """
    import torch  # type: ignore[import-not-found]
    import torch.nn as nn  # type: ignore[import-not-found]

    class RootLayer(nn.Module):
        def __init__(self, steps: int, tol: float):
            super().__init__()
            self.steps = steps
            self.tol = tol

        def forward(self, coeffs: torch.Tensor, xs: torch.Tensor) -> torch.Tensor:
            x = xs
            for _ in range(self.steps):
                # Horner for p and p'
                a_rev = torch.flip(coeffs, dims=[-1])
                if x.ndim == 1:
                    xr = x[:, None]
                    squeeze = True
                else:
                    xr = x
                    squeeze = False
                p = torch.zeros_like(xr, dtype=coeffs.dtype)
                dp = torch.zeros_like(xr, dtype=coeffs.dtype)
                for c in a_rev.T:
                    dp = dp * xr + p
                    p = p * xr + c[:, None]
                upd = xr - p / torch.where(dp == 0, torch.ones_like(dp), dp)
                x = upd[:, 0] if squeeze else upd
                if self.tol > 0 and torch.max(torch.abs(p)) < self.tol:
                    break
            return x

    return RootLayer(steps, tol)


def torch_aberth_step(coeffs, roots, damping: float = 1.0):
    """One vectorized Aberth–Ehrlich update using Torch.

    Args:
      coeffs: 1D torch tensor of shape (D+1,) complex64/complex128 (low->high)
      roots: 1D torch tensor of shape (N,) complex
      damping: scalar in (0,1]

    Returns:
      updated roots tensor shape (N,)

    Note: This performs a single iteration without line search. Caller can run
    multiple steps and add damping heuristics if desired.
    """
    import torch  # type: ignore[import-not-found]

    assert coeffs.ndim == 1, "coeffs must be 1D (low->high)"
    assert roots.ndim == 1, "roots must be 1D"

    # Horner for p and p' at all roots (vectorized)
    a_rev = torch.flip(coeffs, dims=[-1])
    z = roots
    p = torch.zeros_like(z, dtype=coeffs.dtype)
    dp = torch.zeros_like(z, dtype=coeffs.dtype)
    for c in a_rev:
        dp = dp * z + p
        p = p * z + c

    # Pairwise differences matrix and reciprocal sums excluding diagonal
    Z_i = z[:, None]
    Z_j = z[None, :]
    diff = Z_i - Z_j
    # Avoid division by zero on diagonal by setting to inf
    diff = diff + torch.eye(z.shape[0], dtype=diff.dtype, device=diff.device) * (0.0 + 1j * 0.0)
    diff[torch.eye(z.shape[0], dtype=torch.bool, device=diff.device)] = torch.inf + 0j
    S = torch.sum(1.0 / diff, dim=1)

    denom = dp - p * S
    # Guard: avoid zero denom
    denom = torch.where(denom == 0, torch.ones_like(denom), denom)
    delta = p / denom
    return z - damping * delta


def torch_aberth_solve(
    coeffs,
    roots_init,
    iters: int = 50,
    damping: float = 0.8,
    backtracks: int = 3,
):
    """Multi-step Aberth–Ehrlich using Torch with simple backtracking on residual.

    Args:
      coeffs: 1D torch tensor (D+1,) complex (low->high)
      roots_init: 1D torch tensor (N,) complex
      iters: number of iterations
      damping: initial damping factor
      backtracks: backtracking steps if residual increases

    Returns:
      roots tensor shape (N,)
    """
    import torch  # type: ignore[import-not-found]

    z = roots_init.clone()
    # residual helper
    def max_residual(zv):
        # reuse batched_poly_eval with backend torch by building per-root coefficients
        from .batched import batched_poly_eval

        # Expand coeffs to (B, D+1) and z to (B,)
        B = zv.shape[0]
        c = coeffs.expand(B, -1)
        p = batched_poly_eval(c, zv, backend="torch")
        return torch.max(torch.abs(p))

    res_prev = max_residual(z)
    for _ in range(max(1, iters)):
        z_prop = torch_aberth_step(coeffs, z, damping=damping)
        res_prop = max_residual(z_prop)
        if torch.isfinite(res_prop) and res_prop <= res_prev:
            z, res_prev = z_prop, res_prop
            continue
        # backtrack
        alpha = damping
        accepted = False
        for _ in range(max(0, backtracks)):
            alpha *= 0.5
            z_bt = torch_aberth_step(coeffs, z, damping=alpha)
            res_bt = max_residual(z_bt)
            if torch.isfinite(res_bt) and res_bt <= res_prev:
                z, res_prev = z_bt, res_bt
                accepted = True
                break
        if not accepted:
            # accept proposed step anyway to avoid stalling
            z, res_prev = z_prop, res_prop
    return z


def torch_aberth_solve_batched(
    coeffs_batch,
    roots_init_batch,
    iters: int = 30,
    damping: float = 0.8,
):
    """Batched Aberth for many polynomials using Torch.

    Args:
      coeffs_batch: (B, D+1) complex, low->high
      roots_init_batch: (B, N) complex
    Returns:
      (B, N) updated roots
    """
    import torch  # type: ignore[import-not-found]

    a = coeffs_batch
    z = roots_init_batch
    B, N = z.shape
    # Precompute reversed coeffs
    a_rev = torch.flip(a, dims=[-1])  # (B, D+1)

    for _ in range(max(1, iters)):
        # Horner for p, dp per batch
        p = torch.zeros_like(z)
        dp = torch.zeros_like(z)
        for c in a_rev.T:  # iterate degree
            dp = dp * z + p
            p = p * z + c  # (B,)
        # Pairwise sums per batch
        Z_i = z.unsqueeze(2)  # (B, N, 1)
        Z_j = z.unsqueeze(1)  # (B, 1, N)
        diff = Z_i - Z_j
        # avoid diag
        eye = torch.eye(N, dtype=diff.dtype, device=diff.device).unsqueeze(0)
        diff = diff + eye * (0.0 + 1j * 0.0)
        diff[:, torch.arange(N), torch.arange(N)] = torch.inf + 0j
        S = torch.sum(1.0 / diff, dim=2)
        denom = dp - p * S
        denom = torch.where(denom == 0, torch.ones_like(denom), denom)
        delta = p / denom
        z = z - damping * delta
    return z


def jax_aberth_step(coeffs, roots, damping: float = 1.0):
    """Single JAX vectorized Aberth step."""
    import jax.numpy as jnp  # type: ignore[import-not-found]

    a_rev = jnp.flip(coeffs)
    z = roots
    p = jnp.zeros_like(z)
    dp = jnp.zeros_like(z)
    for c in a_rev:
        dp = dp * z + p
        p = p * z + c
    Z_i = z[:, None]
    Z_j = z[None, :]
    diff = Z_i - Z_j
    diff = diff + jnp.eye(z.shape[0], dtype=diff.dtype) * (0.0 + 0.0j)
    diff = diff.at[jnp.diag_indices(z.shape[0])].set(jnp.inf + 0.0j)
    S = jnp.sum(1.0 / diff, axis=1)
    denom = dp - p * S
    denom = jnp.where(denom == 0, jnp.ones_like(denom), denom)
    delta = p / denom
    return z - damping * delta


def jax_aberth_solve(coeffs, roots_init, iters: int = 50, damping: float = 0.8):
    import jax.numpy as jnp  # type: ignore[import-not-found]

    z = roots_init
    for _ in range(max(1, iters)):
        z = jax_aberth_step(coeffs, z, damping=damping)
    return z

def batched_solve_all(
    coeffs_batch, backend: str = "numpy", method: str = "newton", steps: int = 20
):
    """Solve many polynomials for one root each using a vectorized Newton path.

    - coeffs_batch: shape (B, D+1) low->high
    - Returns: shape (B,) complex roots (one per polynomial)

    Notes:
    - This is a simple baseline using Newton from a heuristic initial guess (x0 = 0).
    - Future: add Aberth/sharded multi-root per polynomial; support seeds per item.
    """
    xp = _as_backend(backend)
    a = coeffs_batch
    B = a.shape[0]
    # Start from zero; quick heuristic: shift by -a0/a1 if available and finite
    x = xp.zeros((B,), dtype=a.dtype)
    a0 = a[:, 0]
    a1 = a[:, 1] if a.shape[1] > 1 else xp.zeros_like(a0)
    with xp.errstate(all="ignore") if hasattr(xp, "errstate") else _nullcontext():  # type: ignore
        guess = -a0 / xp.where(a1 == 0, xp.asarray(1, dtype=a1.dtype), a1)
    # Blend: if |guess| finite and not huge, use it
    mask = (
        xp.isfinite(guess)
        if hasattr(xp, "isfinite")
        else xp.ones_like(guess, dtype=bool)
    )
    x = xp.where(mask, guess, x)

    # Newton iterations using batched_newton_step
    for _ in range(steps):
        x_next = batched_newton_step(a, x, backend=backend)
        # Stop if updates are tiny
        if xp.max(xp.abs(x_next - x)) < 1e-14:
            x = x_next
            break
        x = x_next
    return x


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False
