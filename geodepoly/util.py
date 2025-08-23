from __future__ import annotations
from typing import List
import os

_HAS_NUMBA = False
try:
    import numba as _nb  # type: ignore
    import numpy as _np  # type: ignore

    _HAS_NUMBA = True
except Exception:  # pragma: no cover
    _HAS_NUMBA = False

_USE_NUMBA = bool(int(os.getenv("GEODEPOLY_USE_NUMBA", "0"))) and _HAS_NUMBA


if _HAS_NUMBA:

    @_nb.njit(cache=True)
    def _poly_eval_nb(a: _np.ndarray, x: complex) -> complex:  # type: ignore
        acc = 0.0 + 0.0j
        for k in range(a.shape[0] - 1, -1, -1):
            acc = acc * x + a[k]
        return acc

    @_nb.njit(cache=True)
    def _poly_eval_dp_ddp_nb(a: _np.ndarray, x: complex):  # type: ignore
        p = 0.0 + 0.0j
        dp = 0.0 + 0.0j
        ddp = 0.0 + 0.0j
        for k in range(a.shape[0] - 1, -1, -1):
            ak = a[k]
            ddp = ddp * x + 2.0 * dp
            dp = dp * x + p
            p = p * x + ak
        return p, dp, ddp


def set_numba_acceleration(enabled: bool) -> None:
    global _USE_NUMBA
    _USE_NUMBA = bool(enabled) and _HAS_NUMBA


def poly_eval(coeffs, x):
    if _USE_NUMBA:
        a = _np.asarray([complex(c) for c in coeffs], dtype=_np.complex128)  # type: ignore
        return _poly_eval_nb(a, complex(x))  # type: ignore
    acc = 0j
    for a in reversed(coeffs):
        acc = acc * x + complex(a)
    return acc


def poly_eval_dp_ddp(coeffs, x):
    if _USE_NUMBA:
        a = _np.asarray([complex(c) for c in coeffs], dtype=_np.complex128)  # type: ignore
        return _poly_eval_dp_ddp_nb(a, complex(x))  # type: ignore
    p = 0j
    dp = 0j
    ddp = 0j
    for a in reversed(coeffs):
        ddp = ddp * x + 2 * dp
        dp = dp * x + p
        p = p * x + a
    return p, dp, ddp


def shift_expand(coeffs, mu):
    # q(y) = p(mu+y) coefficients (low-to-high)
    from math import comb

    n = len(coeffs) - 1
    q = [0j] * (n + 1)
    for k, a in enumerate(coeffs):
        ak = complex(a)
        if ak == 0:
            continue
        for j in range(k + 1):
            q[j] += ak * (comb(k, j) * (mu ** (k - j)))
    return q


def poly_eval_many(coeffs: List[complex], xs: List[complex]):
    """Vectorized polynomial evaluation for many points using NumPy Horner.
    Returns a list of complex values p(x) for x in xs.
    """
    try:
        import numpy as np
    except Exception:
        return [poly_eval(coeffs, x) for x in xs]
    a = np.array(
        list(reversed([complex(c) for c in coeffs])), dtype=complex
    )  # high->low
    xs_np = np.array(xs, dtype=complex)
    # Horner vectorized
    p = np.zeros_like(xs_np)
    for c in a:
        p = p * xs_np + c
    return [complex(v) for v in p]
