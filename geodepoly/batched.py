from __future__ import annotations

from typing import Any


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


def _batched_poly_eval_numpy(coeffs: Any, xs: Any) -> Any:
    import numpy as np  # type: ignore

    a = coeffs
    x = xs
    squeezed = False
    if x.ndim == 1:
        x = x.reshape((-1, 1))
        squeezed = True
    a_rev = a[..., ::-1]
    p = np.zeros((x.shape[0], x.shape[1]), dtype=a_rev.dtype)
    for c in a_rev.T:
        p = p * x + c.reshape((-1, 1))
    return p[:, 0] if squeezed else p


def _batched_poly_eval_torch(coeffs: Any, xs: Any) -> Any:
    import torch  # type: ignore

    a = coeffs
    x = xs
    squeezed = False
    if x.ndim == 1:
        x = x.reshape((-1, 1))
        squeezed = True
    a_rev = torch.flip(a, dims=[-1])
    p = torch.zeros((x.shape[0], x.shape[1]), dtype=a_rev.dtype, device=a_rev.device)
    for c in a_rev.T:
        p = torch.addcmul(c.reshape((-1, 1)), p, x, value=1.0)
    return p[:, 0] if squeezed else p


def _batched_poly_eval_jax(coeffs: Any, xs: Any) -> Any:
    import jax.numpy as jnp  # type: ignore
    from jax import lax  # type: ignore

    a = coeffs
    x = xs
    squeezed = False
    if x.ndim == 1:
        x = x.reshape((-1, 1))
        squeezed = True
    a_rev = jnp.flip(a, axis=-1)
    c_seq = jnp.transpose(a_rev)
    p0 = jnp.zeros((x.shape[0], x.shape[1]), dtype=a_rev.dtype)

    def body(p, c):
        return p * x + c[:, None], None

    p, _ = lax.scan(body, p0, c_seq)
    return p[:, 0] if squeezed else p


def batched_poly_eval(coeffs, xs, backend: str = "numpy"):
    """Evaluate many polynomials at many points via Horner.

    coeffs: (B, D+1) low->high; xs: (B,) or (B, M). Returns (B,) or (B, M).
    """
    if backend == "torch":
        return _batched_poly_eval_torch(coeffs, xs)
    if backend == "jax":
        return _batched_poly_eval_jax(coeffs, xs)
    return _batched_poly_eval_numpy(coeffs, xs)


def _batched_newton_step_numpy(coeffs: Any, xs: Any) -> Any:
    import numpy as np  # type: ignore

    x = xs
    squeezed = False
    if x.ndim == 1:
        x = x.reshape((-1, 1))
        squeezed = True
    a_rev = coeffs[..., ::-1]
    p = np.zeros_like(x, dtype=a_rev.dtype)
    dp = np.zeros_like(x, dtype=a_rev.dtype)
    for c in a_rev.T:
        dp = dp * x + p
        p = p * x + c.reshape((-1, 1))
    denom = np.where(dp == 0, np.asarray(1, dtype=dp.dtype), dp)
    step = x - p / denom
    return step[:, 0] if squeezed else step


def _batched_newton_step_torch(coeffs: Any, xs: Any) -> Any:
    import torch  # type: ignore

    x = xs
    squeezed = False
    if x.ndim == 1:
        x = x.reshape((-1, 1))
        squeezed = True
    a_rev = torch.flip(coeffs, dims=[-1])
    p = torch.zeros_like(x, dtype=a_rev.dtype)
    dp = torch.zeros_like(x, dtype=a_rev.dtype)
    for c in a_rev.T:
        dp = dp * x + p
        p = p * x + c.reshape((-1, 1))
    denom = torch.where(dp == 0, torch.ones_like(dp), dp)
    step = x - p / denom
    return step[:, 0] if squeezed else step


def _batched_newton_step_jax(coeffs: Any, xs: Any) -> Any:
    import jax.numpy as jnp  # type: ignore
    from jax import lax  # type: ignore

    x = xs
    squeezed = False
    if x.ndim == 1:
        x = x.reshape((-1, 1))
        squeezed = True
    a_rev = jnp.flip(coeffs, axis=-1)
    p0 = jnp.zeros_like(x, dtype=a_rev.dtype)
    dp0 = jnp.zeros_like(x, dtype=a_rev.dtype)
    c_seq = jnp.transpose(a_rev)

    def body(state, c):
        p, dp = state
        dp = dp * x + p
        p = p * x + c[:, None]
        return (p, dp), None

    (p, dp), _ = lax.scan(body, (p0, dp0), c_seq)
    denom = jnp.where(dp == 0, jnp.ones_like(dp), dp)
    step = x - p / denom
    return step[:, 0] if squeezed else step


def batched_newton_step(coeffs, xs, backend: str = "numpy"):
    """One Newton step x - p/ p' for each polynomial in the batch."""
    if backend == "torch":
        return _batched_newton_step_torch(coeffs, xs)
    if backend == "jax":
        return _batched_newton_step_jax(coeffs, xs)
    return _batched_newton_step_numpy(coeffs, xs)


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


def torch_batched_roots(coeffs_batch, iters: int = 60, damping: float = 0.8, backtracks: int = 3):
    """Solve many same-degree polynomials' roots with Torch vectorized Aberth.

    Args:
      coeffs_batch: torch.Tensor of shape (B, D+1) complex (low->high)
      iters: number of Aberth iterations
      damping: initial damping factor
      backtracks: backtracking steps per iteration

    Returns:
      torch.Tensor of shape (B, N) complex roots
    """
    import torch  # type: ignore[import-not-found]

    a = coeffs_batch
    assert a.ndim == 2, "coeffs_batch must be (B, D+1)"
    B, D1 = a.shape
    N = D1 - 1
    if N <= 0:
        return torch.zeros((B, 0), dtype=a.dtype, device=a.device)
    # Cauchy-like radius per batch: 1 + max |a_k|/|a_n|
    an = torch.abs(a[:, -1])
    # avoid div by zero
    an = torch.where(an == 0, torch.ones_like(an), an)
    max_ratio = torch.max(torch.abs(a[:, :-1]) / an[:, None], dim=1).values
    R = 1.0 + max_ratio
    # angles for N points
    k = torch.arange(N, dtype=torch.float32, device=a.device)
    theta = 2 * torch.pi * (k / N)
    unit = torch.exp(1j * theta.to(a.dtype))  # (N,)
    # broadcast radii to (B, N)
    z0 = (R[:, None].to(a.dtype)) * unit[None, :]
    z = torch_aberth_solve_batched(a.to(a.dtype), z0, iters=iters, damping=damping)
    return z

def batched_solve_all(
    coeffs_batch, backend: str = "numpy", method: str = "newton", steps: int = 20
):
    """Solve many polynomials for one root each using a vectorized Newton path.

    - coeffs_batch: shape (B, D+1) low->high
    - Returns: shape (B,) complex roots (one per polynomial)
    """
    if backend == "torch":
        import torch  # type: ignore

        a = coeffs_batch
        B = a.shape[0]
        x = torch.zeros((B,), dtype=a.dtype, device=a.device)
        a0 = a[:, 0]
        a1 = a[:, 1] if a.shape[1] > 1 else torch.zeros_like(a0)
        denom = torch.where(a1 == 0, torch.ones_like(a1), a1)
        guess = -a0 / denom
        mask = torch.isfinite(guess)
        x = torch.where(mask, guess, x)
        for _ in range(steps):
            x_next = _batched_newton_step_torch(a, x)
            if torch.max(torch.abs(x_next - x)) < 1e-14:
                x = x_next
                break
            x = x_next
        return x
    if backend == "jax":
        import jax.numpy as jnp  # type: ignore

        a = coeffs_batch
        B = a.shape[0]
        x = jnp.zeros((B,), dtype=a.dtype)
        a0 = a[:, 0]
        a1 = a[:, 1] if a.shape[1] > 1 else jnp.zeros_like(a0)
        denom = jnp.where(a1 == 0, jnp.asarray(1, dtype=a1.dtype), a1)
        guess = -a0 / denom
        mask = jnp.isfinite(guess)
        x = jnp.where(mask, guess, x)
        for _ in range(steps):
            x_next = _batched_newton_step_jax(a, x)
            if jnp.max(jnp.abs(x_next - x)) < 1e-14:
                x = x_next
                break
            x = x_next
        return x
    # numpy backend
    import numpy as np  # type: ignore

    a = coeffs_batch
    B = a.shape[0]
    x = np.zeros((B,), dtype=a.dtype)
    a0 = a[:, 0]
    a1 = a[:, 1] if a.shape[1] > 1 else np.zeros_like(a0)
    with np.errstate(all="ignore"):
        guess = -a0 / np.where(a1 == 0, np.asarray(1, dtype=a1.dtype), a1)
    mask = np.isfinite(guess)
    x = np.where(mask, guess, x)
    for _ in range(steps):
        x_next = _batched_newton_step_numpy(a, x)
        if np.max(np.abs(x_next - x)) < 1e-14:
            x = x_next
            break
        x = x_next
    return x


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False
