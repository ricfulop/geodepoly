\
from __future__ import annotations
import cmath, math, random
from typing import List

from .util import poly_eval, poly_eval_dp_ddp

def halley_refine(coeffs, x, steps=2):
    for _ in range(steps):
        p, dp, ddp = poly_eval_dp_ddp(coeffs, x)
        denom = 2*dp*dp - p*ddp
        if denom == 0:
            break
        x = x - (2*p*dp)/denom
    return x

def durand_kerner(coeffs, iters=400, tol=1e-14, restarts=4):
    n = len(coeffs)-1
    an = coeffs[-1]
    if an == 0:
        raise ValueError("Leading coefficient is zero.")
    base_R = 1 + max((abs(a)/abs(an) for a in coeffs[:-1]), default=0)

    best_roots = None
    best_res = float('inf')
    for attempt in range(restarts):
        R = base_R * (1 + 0.2*(random.random()-0.5))
        roots = [R * cmath.exp(2j*math.pi*(k+0.25*attempt)/n) for k in range(n)]
        for it in range(iters):
            new_roots = []
            max_delta = 0.0
            for i, z in enumerate(roots):
                denom = 1.0 + 0j
                for j, w in enumerate(roots):
                    if j != i:
                        denom *= (z - w)
                if denom == 0:
                    z += (random.random()-0.5)*1e-8 + 1j*(random.random()-0.5)*1e-8
                    denom = 1.0 + 0j
                    for j, w in enumerate(roots):
                        if j != i:
                            denom *= (z - w)
                val = poly_eval(coeffs, z)
                z_new = z - val/denom
                new_roots.append(z_new)
                max_delta = max(max_delta, abs(z_new - z))
            roots = new_roots
            if max_delta < tol:
                break
        # residual measure
        res = max(abs(poly_eval(coeffs, r)) for r in roots)
        if res < best_res:
            best_res = res
            best_roots = roots
    return best_roots

def aberth_ehrlich(coeffs, iters=400, tol=1e-14, restarts=2, warm_starts: List[complex]|None=None):
    """
    Aberth–Ehrlich simultaneous iteration:
    z_i <- z_i - p(z_i) / ( p'(z_i) - p(z_i) * sum_{j!=i} 1/(z_i - z_j) )
    """
    n = len(coeffs)-1
    an = coeffs[-1]
    if an == 0:
        raise ValueError("Leading coefficient is zero.")
    base_R = 1 + max((abs(a)/abs(an) for a in coeffs[:-1]), default=0)

    best_roots = None
    best_res = float('inf')
    for attempt in range(restarts):
        if warm_starts and len(warm_starts) >= 1:
            # seed around warm starts + circle for remaining
            roots = []
            for ws in warm_starts[:min(len(warm_starts), n)]:
                roots.append(ws + 0.01*base_R*(random.random()-0.5) + 1j*0.01*base_R*(random.random()-0.5))
            # fill remaining on circle
            while len(roots) < n:
                k = len(roots)
                roots.append(base_R * cmath.exp(2j*math.pi*(k+0.17*attempt)/n))
        else:
            roots = [base_R * cmath.exp(2j*math.pi*(k+0.37*attempt)/n) for k in range(n)]

        for it in range(iters):
            max_step = 0.0
            new_roots = roots[:]
            # Precompute p and p' at all points
            pvals = []
            dpvals = []
            for z in roots:
                # Horner for p and dp
                p = 0j; dp = 0j
                for a in reversed(coeffs):
                    dp = dp*z + p
                    p = p*z + a
                pvals.append(p); dpvals.append(dp)

            for i, z in enumerate(roots):
                p = pvals[i]; dp = dpvals[i]
                if dp == 0:
                    # small jitter
                    z = z + (random.random()-0.5)*1e-6 + 1j*(random.random()-0.5)*1e-6
                    p = poly_eval(coeffs, z)
                    # recompute dp crudely
                    eps = 1e-8
                    dp = (poly_eval(coeffs, z+eps) - poly_eval(coeffs, z-eps))/(2*eps)
                denom = dp
                # Aberth correction term
                S = 0j
                for j, w in enumerate(roots):
                    if j != i:
                        S += 1/(z - w)
                denom = dp - p * S
                if denom == 0:
                    continue
                delta = p / denom
                new_roots[i] = z - delta
                max_step = max(max_step, abs(delta))
            roots = new_roots
            if max_step < tol:
                break

        res = max(abs(poly_eval(coeffs, r)) for r in roots)
        if res < best_res:
            best_res = res
            best_roots = roots
    return best_roots
