"""AI extensions for GeodePoly.

This subpackage provides optional, differentiable root solving layers and
training losses for PyTorch and JAX. Framework imports are lazy so that
users without these dependencies can still import `geodepoly` normally.
"""

__all__ = [
    "root_solve_torch",
    "root_solve_jax",
]


def root_solve_torch(coeffs, method: str = "hybrid", resum: str = "pade"):
    """Torch root solve wrapper. Requires PyTorch installed.

    coeffs: torch.Tensor of shape (B, N+1), complex dtype preferred.
    Returns: torch.Tensor of shape (B, N) with roots (complex dtype).
    """
    try:
        from .rootlayer_torch import root_solve_torch as _rt
    except Exception as exc:  # pragma: no cover
        raise ImportError("PyTorch not installed or rootlayer unavailable") from exc
    return _rt(coeffs, method=method, resum=resum)


def root_solve_jax(coeffs, method: str = "hybrid", resum: str = "pade"):
    """JAX root solve wrapper. Requires jax/jaxlib installed.

    coeffs: jax.numpy array of shape (B, N+1), complex dtype preferred.
    Returns: jax.numpy array of shape (B, N) with roots (complex dtype).

    Note: method/resum are accepted for API symmetry but ignored in the pure-JAX path
    to keep `jit` usage simple (no static args)."""
    try:
        from .rootlayer_jax import root_solve_jax as _rj
    except Exception as exc:  # pragma: no cover
        raise ImportError("JAX not installed or rootlayer unavailable") from exc
    # Do not pass method/resum into JAX-traced function to avoid static arg handling.
    return _rj(coeffs)


