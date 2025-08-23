from __future__ import annotations

from typing import Optional


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
    import torch
    import torch.nn as nn

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



def batched_solve_all(coeffs_batch, backend: str = "numpy", method: str = "newton", steps: int = 20):
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
    mask = xp.isfinite(guess) if hasattr(xp, "isfinite") else xp.ones_like(guess, dtype=bool)
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

