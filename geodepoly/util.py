\
from __future__ import annotations
import cmath, math
from typing import List

def poly_eval(coeffs, x):
    acc = 0j
    for a in reversed(coeffs):
        acc = acc * x + complex(a)
    return acc

def poly_eval_dp_ddp(coeffs, x):
    p = 0j; dp = 0j; ddp = 0j
    for a in reversed(coeffs):
        ddp = ddp * x + 2*dp
        dp  = dp  * x + p
        p   = p   * x + a
    return p, dp, ddp

def shift_expand(coeffs, mu):
    # q(y) = p(mu+y) coefficients (low-to-high)
    from math import comb
    n = len(coeffs)-1
    q = [0j]*(n+1)
    for k, a in enumerate(coeffs):
        ak = complex(a)
        if ak == 0: continue
        for j in range(k+1):
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
    a = np.array(list(reversed([complex(c) for c in coeffs])), dtype=complex)  # high->low
    xs_np = np.array(xs, dtype=complex)
    # Horner vectorized
    p = np.zeros_like(xs_np)
    for c in a:
        p = p * xs_np + c
    return [complex(v) for v in p]
