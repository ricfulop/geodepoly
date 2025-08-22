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
