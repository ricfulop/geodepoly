\
from __future__ import annotations
import cmath, math, time
from typing import List, Optional, Dict, Any

from .series_core import series_seed_step
from .finishers import durand_kerner, aberth_ehrlich, halley_refine, halley_refine_multiplicity
from .util import poly_eval

def solve_one(coeffs: List[complex],
              center: complex|None=None,
              max_order: int = 24,
              boots: int = 3,
              tol: float = 1e-14,
              resum: Optional[str] = None,   # None, 'pade', 'borel', 'borel-pade'
              refine_steps: int = 2) -> complex:
    """
    Find a single root using bootstrap series reversion around a center μ.
    If center is None, tries a small set of heuristic centers.
    """
    # Normalize to monic for stability
    if coeffs[-1] == 0:
        raise ValueError("Leading coefficient is zero.")
    scale = coeffs[-1]
    c = [complex(a)/scale for a in coeffs]

    # Heuristic centers: 0 and a Cauchy circle
    an = abs(c[-1])
    R = 1 + max((abs(a)/an for a in c[:-1]), default=0)
    centers = [center] if center is not None else [0j, R, -R, 1j*R, -1j*R, R/2]

    best = None
    best_res = float('inf')

    for mu in centers:
        x = complex(mu)
        ok_local = False
        for _ in range(max(1, boots)):
            y, a0, a1, ok = series_seed_step(c, x, max_order=max_order, resum=resum)
            if (not ok) or abs(a1) < 1e-18:
                ok_local = False
                break
            t = -a0/a1
            # Guard: if t ~ 1, we are near convergence boundary; resummation helps,
            # but still avoid stepping into divergence
            if abs(t) > 0.95:
                ok_local = False
                break
            x = x + y
            ok_local = True
            if abs(poly_eval(c, x)) < tol:
                break
        if ok_local:
            xr = halley_refine(c, x, steps=refine_steps)
            res = abs(poly_eval(c, xr))
            if res < best_res:
                best_res = res
                best = xr

    if best is None:
        # last resort: return center with smallest |p(mu)|
        best = min(centers, key=lambda mu: abs(poly_eval(c, mu)))
    return best

def solve_all(coeffs: List[complex],
              method: str = "hybrid",    # 'hybrid'|'aberth'|'dk'|'numpy'
              max_order: int = 24,
              boots: int = 2,
              tol: float = 1e-12,
              resum: Optional[str] = None,
              refine_steps: int = 3,
              verbose: bool = False) -> List[complex]:
    """
    Solve for all roots with a selectable finisher:
    - 'hybrid': obtain 1-2 high-quality series seeds, then finish with Aberth–Ehrlich.
    - 'aberth': pure Aberth–Ehrlich.
    - 'dk': Durand–Kerner.
    - 'numpy': companion eigenvalues (requires numpy).
    """
    n = len(coeffs)-1
    if n <= 0:
        return []

    if method not in {"hybrid","aberth","dk","numpy"}:
        raise ValueError("method must be one of {'hybrid','aberth','dk','numpy'}")

    if method == "numpy":
        try:
            import numpy as np
        except ImportError:
            raise ImportError("NumPy not installed; required for method='numpy'")
        # companion matrix eigenvalues
        a = [complex(x) for x in coeffs]
        if a[-1] == 0:
            raise ValueError("Leading coefficient is zero.")
        # normalize to monic
        a = [x/a[-1] for x in a]
        n = len(a)-1
        C = np.zeros((n,n), dtype=complex)
        C[1:, :-1] = np.eye(n-1, dtype=complex)
        C[:, -1] = -np.array(a[:-1], dtype=complex)
        w = np.linalg.eigvals(C)
        # Polish
        roots = [halley_refine(coeffs, complex(z), steps=refine_steps) for z in w]
        return roots

    if method == "dk":
        roots = durand_kerner(coeffs, iters=600, tol=1e-14, restarts=5)
        return [halley_refine_multiplicity(coeffs, z, steps=refine_steps) for z in roots]

    if method == "aberth":
        roots = aberth_ehrlich(coeffs, iters=400, tol=1e-14, restarts=3)
        return [halley_refine_multiplicity(coeffs, z, steps=refine_steps) for z in roots]

    # hybrid:
    # Get two seeds via series from different centers (0 and Cauchy radius)
    seeds = []
    try:
        seeds.append(solve_one(coeffs, center=None, max_order=max_order, boots=boots, tol=1e-14, resum=resum, refine_steps=refine_steps))
    except Exception:
        pass
    try:
        # second seed from opposite side
        an = abs(coeffs[-1])
        R = 1 + max((abs(a)/an for a in coeffs[:-1]), default=0)
        seeds.append(solve_one(coeffs, center=-R, max_order=max_order, boots=boots, tol=1e-14, resum=resum, refine_steps=refine_steps))
    except Exception:
        pass

    roots = aberth_ehrlich(coeffs, iters=400, tol=1e-14, restarts=3, warm_starts=seeds)
    return [halley_refine_multiplicity(coeffs, z, steps=refine_steps) for z in roots]

def solve_poly(coeffs: List[complex], **kwargs) -> List[complex]:
    """Public convenience wrapper (alias of solve_all)."""
    return solve_all(coeffs, **kwargs)
