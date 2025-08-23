from __future__ import annotations
import cmath
import math
import random
from typing import List

from .util import poly_eval, poly_eval_dp_ddp


def halley_refine(coeffs, x, steps=2):
    for _ in range(steps):
        p, dp, ddp = poly_eval_dp_ddp(coeffs, x)
        denom = 2 * dp * dp - p * ddp
        if denom == 0:
            break
        x = x - (2 * p * dp) / denom
    return x


def estimate_multiplicity(coeffs, x) -> int:
    """Estimate local multiplicity m using p, p', p'' at x.
    m ≈ round( Re( p * p'' / p'^2 ) ), clamped to [1..3].
    """
    p, dp, ddp = poly_eval_dp_ddp(coeffs, x)
    if dp == 0:
        return 2
    ratio = (p * ddp) / (dp * dp)
    m = int(round(ratio.real))
    if m < 1:
        m = 1
    if m > 3:
        m = 3
    return m


def halley_refine_multiplicity(coeffs, x, steps=3):
    """Halley-like refinement aware of multiplicity m when m>=2.
    Update: x <- x - m * p / ( m*p' - (m-1) * p*p''/p' ).
    Falls back to standard Halley when estimates are unstable.
    """
    for _ in range(steps):
        p, dp, ddp = poly_eval_dp_ddp(coeffs, x)
        if dp == 0:
            break
        m = estimate_multiplicity(coeffs, x)
        if m <= 1:
            # standard Halley step
            denom = 2 * dp * dp - p * ddp
            if denom == 0:
                break
            x = x - (2 * p * dp) / denom
        else:
            # multiplicity-aware update
            denom = (m * dp) - (m - 1) * (p * ddp / dp)
            if denom == 0:
                break
            x = x - (m * p) / denom
    return x


def durand_kerner(coeffs, iters=400, tol=1e-14, restarts=4):
    n = len(coeffs) - 1
    an = coeffs[-1]
    if an == 0:
        raise ValueError("Leading coefficient is zero.")
    base_R = 1 + max((abs(a) / abs(an) for a in coeffs[:-1]), default=0)

    best_roots = None
    best_res = float("inf")
    for attempt in range(restarts):
        R = base_R * (1 + 0.2 * (random.random() - 0.5))
        roots = [
            R * cmath.exp(2j * math.pi * (k + 0.25 * attempt) / n) for k in range(n)
        ]
        for it in range(iters):
            new_roots = []
            max_delta = 0.0
            for i, z in enumerate(roots):
                denom = 1.0 + 0j
                for j, w in enumerate(roots):
                    if j != i:
                        denom *= z - w
                if denom == 0:
                    z += (random.random() - 0.5) * 1e-8 + 1j * (
                        random.random() - 0.5
                    ) * 1e-8
                    denom = 1.0 + 0j
                    for j, w in enumerate(roots):
                        if j != i:
                            denom *= z - w
                val = poly_eval(coeffs, z)
                z_new = z - val / denom
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


def aberth_ehrlich(
    coeffs, iters=400, tol=1e-14, restarts=2, warm_starts: List[complex] | None = None
):
    """
    Aberth–Ehrlich simultaneous iteration with adaptive damping and minimal repulsion:
    z_i <- z_i - alpha_i * p(z_i) / ( p'(z_i) - p(z_i) * sum_{j!=i} 1/(z_i - z_j) )
    where alpha_i in (0,1] chosen to reduce residual and temper steps near clusters.
    """
    n = len(coeffs) - 1
    an = coeffs[-1]
    if an == 0:
        raise ValueError("Leading coefficient is zero.")
    base_R = 1 + max((abs(a) / abs(an) for a in coeffs[:-1]), default=0)

    best_roots = None
    best_res = float("inf")
    for attempt in range(restarts):
        if warm_starts and len(warm_starts) >= 1:
            roots = []
            for ws in warm_starts[: min(len(warm_starts), n)]:
                roots.append(
                    ws
                    + 0.01 * base_R * (random.random() - 0.5)
                    + 1j * 0.01 * base_R * (random.random() - 0.5)
                )
            while len(roots) < n:
                k = len(roots)
                roots.append(
                    base_R * cmath.exp(2j * math.pi * (k + 0.17 * attempt) / n)
                )
        else:
            roots = [
                base_R * cmath.exp(2j * math.pi * (k + 0.37 * attempt) / n)
                for k in range(n)
            ]

        for it in range(iters):
            max_step = 0.0
            new_roots = roots[:]
            # Precompute p and p' at all points
            pvals = []
            dpvals = []
            for z in roots:
                p = 0j
                dp = 0j
                for a in reversed(coeffs):
                    dp = dp * z + p
                    p = p * z + a
                pvals.append(p)
                dpvals.append(dp)

            # Precompute cluster metric per root
            crowd = []
            for i, z in enumerate(roots):
                s = 0.0
                for j, w in enumerate(roots):
                    if j != i:
                        d = abs(z - w)
                        if d > 0:
                            s += 1.0 / d
                crowd.append(s)
            crowd_max = max(crowd) if crowd else 1.0

            for i, z in enumerate(roots):
                p = pvals[i]
                dp = dpvals[i]
                if dp == 0:
                    z = (
                        z
                        + (random.random() - 0.5) * 1e-6
                        + 1j * (random.random() - 0.5) * 1e-6
                    )
                    p = poly_eval(coeffs, z)
                    eps = 1e-8
                    dp = (poly_eval(coeffs, z + eps) - poly_eval(coeffs, z - eps)) / (
                        2 * eps
                    )
                # Aberth correction
                S = 0j
                for j, w in enumerate(roots):
                    if j != i:
                        S += 1 / (z - w)
                denom = dp - p * S
                if denom == 0:
                    continue
                delta = p / denom

                # Adaptive damping: smaller alpha if crowding is high
                local_crowd = crowd[i]
                crowd_factor = (local_crowd / crowd_max) if crowd_max > 0 else 0.0
                alpha = 1.0 / (1.0 + 2.0 * crowd_factor)
                alpha = max(0.2, min(1.0, alpha))

                # Backtracking line search on residual
                z_trial = z - alpha * delta
                res0 = abs(p)
                res1 = abs(poly_eval(coeffs, z_trial))
                back = 0
                while res1 > res0 and back < 4:
                    alpha *= 0.5
                    z_trial = z - alpha * delta
                    res1 = abs(poly_eval(coeffs, z_trial))
                    back += 1

                new_roots[i] = z_trial
                max_step = max(max_step, abs(alpha * delta))

            # Minimal repulsion for very close pairs
            eps_close = 1e-10 * (1 + base_R)
            for i in range(n):
                for j in range(i + 1, n):
                    if abs(new_roots[i] - new_roots[j]) < eps_close:
                        jitter = (1e-8 * base_R) * (
                            math.cos(2 * math.pi * random.random())
                            + 1j * math.sin(2 * math.pi * random.random())
                        )
                        new_roots[j] += jitter

            roots = new_roots
            if max_step < tol:
                break

        res = max(abs(poly_eval(coeffs, r)) for r in roots)
        if res < best_res:
            best_res = res
            best_roots = roots
    return best_roots
