from __future__ import annotations
import cmath
import math
from typing import List, Optional

from .series_core import series_seed_step
from .finishers import (
    durand_kerner,
    aberth_ehrlich,
    halley_refine,
    halley_refine_multiplicity,
)
from .batched import batched_solve_all
from .util import poly_eval, shift_expand
from .geode import Q_cubic


def solve_one(
    coeffs: List[complex],
    center: complex | None = None,
    max_order: int = 24,
    boots: int = 3,
    tol: float = 1e-14,
    resum: Optional[str] = None,  # None, 'pade', 'borel', 'borel-pade'
    refine_steps: int = 2,
) -> complex:
    """
    Find a single root using bootstrap series reversion around a center μ.
    If center is None, tries a small set of heuristic centers.
    """
    # Normalize to monic for stability
    if coeffs[-1] == 0:
        raise ValueError("Leading coefficient is zero.")
    scale = coeffs[-1]
    c = [complex(a) / scale for a in coeffs]

    # Candidate centers: minimize |t| = |-a0/a1|, a1!=0, sampled on radii
    an = abs(c[-1])
    R = 1 + max((abs(a) / an for a in c[:-1]), default=0)
    if center is not None:
        centers = [center]
    else:
        cand = [0j]
        for r in [R / 8, R / 4, R / 2, R]:
            # 16 angles + axes
            for k in range(16):
                theta = 2 * math.pi * k / 16
                cand.append(r * cmath.exp(1j * theta))
            cand.extend([r, -r, 1j * r, -1j * r])
        # Score by |t|, tie-break by |p(mu)|
        scored = []
        for mu in cand:
            q = shift_expand(c, mu)
            a0 = complex(q[0])
            a1 = complex(q[1]) if len(q) >= 2 else 0j
            if a1 == 0:
                continue
            tmag = abs(-a0 / a1)
            scored.append((tmag, abs(a0), mu))
        scored.sort(key=lambda x: (x[0], x[1]))
        centers = [mu for _, _, mu in scored[:4]] or [0j]

    best = None
    best_res = float("inf")

    for mu in centers:
        x = complex(mu)
        ok_local = False
        for _ in range(max(1, boots)):
            y, a0, a1, ok = series_seed_step(c, x, max_order=max_order, resum=resum)
            if (not ok) or abs(a1) < 1e-18:
                ok_local = False
                break
            t = -a0 / a1
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


def solve_all(
    coeffs: List[complex],
    method: str = "hybrid",  # 'hybrid'|'hybrid-cubic'|'aberth'|'dk'|'numpy'
    max_order: int = 16,
    boots: int = 1,
    tol: float = 1e-12,
    resum: Optional[str] = None,
    refine_steps: int = 2,
    verbose: bool = False,
) -> List[complex]:
    """
    Solve for all roots with a selectable finisher:
    - 'hybrid': obtain 1-2 high-quality series seeds, then finish with Aberth–Ehrlich.
    - 'aberth': pure Aberth–Ehrlich.
    - 'dk': Durand–Kerner.
    - 'numpy': companion eigenvalues (requires numpy).
    """
    n = len(coeffs) - 1
    if n <= 0:
        return []

    if method not in {"hybrid", "hybrid-cubic", "aberth", "dk", "numpy", "batched", "torch-aberth"}:
        raise ValueError(
            "method must be one of {'hybrid','hybrid-cubic','aberth','dk','numpy','batched','torch-aberth'}"
        )

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
        a = [x / a[-1] for x in a]
        n = len(a) - 1
        C = np.zeros((n, n), dtype=complex)
        C[1:, :-1] = np.eye(n - 1, dtype=complex)
        C[:, -1] = -np.array(a[:-1], dtype=complex)
        w = np.linalg.eigvals(C)
        # Polish
        roots = [halley_refine(coeffs, complex(z), steps=refine_steps) for z in w]
        return roots

    if method == "batched":
        # Use vectorized Newton baseline to obtain one root, then deflate and repeat.
        try:
            import numpy as np  # type: ignore
        except Exception:
            raise ImportError("numpy is required for method='batched'")

        def deflate(poly: List[complex], r: complex):
            # Synthetic division by (x - r)
            b: List[complex] = [poly[-1]]
            for a in reversed(poly[:-1]):
                b.append(complex(a) + b[-1] * r)
            b = list(reversed(b))
            return b[1:], complex(b[0])

        c = list(coeffs)
        roots_acc: List[complex] = []
        for _ in range(n):
            # single-item batch
            arr = np.asarray([c], dtype=complex)
            r_arr = batched_solve_all(arr, backend="numpy", steps=40)
            r = complex(r_arr[0])
            # polish
            r = halley_refine_multiplicity(c, complex(r), steps=refine_steps)
            # deflate
            c, rem = deflate(c, r)
            # Guard: if remainder large, fallback to Aberth for remaining
            if abs(rem) > 1e-6:
                remain = n - len(roots_acc)
                others = (
                    aberth_ehrlich(c + [0], iters=200, tol=1e-14, restarts=3)
                    if remain > 1
                    else []
                )
                roots_acc.append(r)
                roots_acc.extend(others)
                return [
                    halley_refine_multiplicity(coeffs, z, steps=refine_steps)
                    for z in roots_acc
                ]
            roots_acc.append(r)
            if len(c) <= 1:
                break
        return [
            halley_refine_multiplicity(coeffs, z, steps=refine_steps) for z in roots_acc
        ]

    if method == "torch-aberth":
        try:
            import torch  # type: ignore
            from .batched import torch_aberth_solve
        except Exception as e:
            raise ImportError("torch is required for method='torch-aberth'") from e

        n = len(coeffs) - 1
        if n <= 0:
            return []
        c = torch.tensor([complex(a) for a in coeffs], dtype=torch.complex64)
        k = torch.arange(n, dtype=torch.float32)
        theta = 2 * torch.pi * (k / n)
        z0 = (1.5 * torch.exp(1j * theta)).to(c.dtype)
        z = torch_aberth_solve(c, z0, iters=60, damping=0.8, backtracks=3)
        roots = [complex(zi.item()) for zi in z.detach().cpu().numpy()]
        return [halley_refine_multiplicity(coeffs, r, steps=refine_steps) for r in roots]

    if method == "dk":
        roots = durand_kerner(coeffs, iters=600, tol=1e-14, restarts=5)
        return [
            halley_refine_multiplicity(coeffs, z, steps=refine_steps) for z in roots
        ]

    if method == "aberth":
        roots = aberth_ehrlich(coeffs, iters=200, tol=1e-14, restarts=3)
        return [
            halley_refine_multiplicity(coeffs, z, steps=refine_steps) for z in roots
        ]

    if method == "hybrid-cubic":
        # A faster series-lite warm-start based on cubic Bi–Tri approximant
        # Build normalized coefficients for scoring centers
        a_norm = [complex(x) / coeffs[-1] for x in coeffs]
        # Candidate centers as in 'hybrid'
        import math as _math
        import cmath as _cmath

        an = abs(a_norm[-1])
        R = 1 + max((abs(a) / an for a in a_norm[:-1]), default=0)
        cand = [0j]
        for r in [R / 8, R / 4, R / 2, R]:
            for k in range(16):
                theta = 2 * _math.pi * k / 16
                cand.append(r * _cmath.exp(1j * theta))
            cand.extend([r, -r, 1j * r, -1j * r])

        def cubic_seed(mu: complex, passes: int = 2) -> Optional[complex]:
            try:
                x = complex(mu)
                for _ in range(max(1, passes)):
                    q = shift_expand(a_norm, x)
                    if len(q) < 2:
                        return None
                    a0 = complex(q[0])
                    a1 = complex(q[1])
                    if abs(a1) == 0:
                        return None
                    t = -a0 / a1
                    if abs(t) > 0.95:
                        return None
                    a2 = complex(q[2]) if len(q) > 2 else 0.0
                    a3 = complex(q[3]) if len(q) > 3 else 0.0
                    # t_k mapping around current center
                    t2 = (a0 * a2) / (a1 * a1) if a2 != 0 else 0.0
                    t3 = (a0 * a0 * a3) / (a1 * a1 * a1) if a3 != 0 else 0.0
                    alpha = Q_cubic(t2, t3)
                    y = t * alpha
                    x_new = x + y
                    # accept if residual improves
                    if abs(poly_eval(a_norm, x_new)) <= abs(poly_eval(a_norm, x)):
                        x = x_new
                    else:
                        x = x + 0.5 * y
                # polish a bit
                xr = halley_refine(a_norm, x, steps=refine_steps)
                return complex(xr)
            except Exception:
                return None

        seeds: List[complex] = []
        # best center by |t|
        scored = []
        for mu in cand:
            q = shift_expand(a_norm, mu)
            a0 = complex(q[0])
            a1 = complex(q[1]) if len(q) >= 2 else 0j
            if a1 == 0:
                continue
            scored.append((abs(-a0 / a1), mu))
        scored.sort(key=lambda x: x[0])
        for _, mu in scored[:2]:
            r = cubic_seed(mu, passes=max(1, boots))
            if r is not None:
                seeds.append(r)
        # Fallback: if no seeds, drop to standard hybrid behavior
        if not seeds:
            roots = aberth_ehrlich(coeffs, iters=200, tol=1e-14, restarts=3)
            return [halley_refine_multiplicity(coeffs, z, steps=refine_steps) for z in roots]

        roots = aberth_ehrlich(coeffs, iters=200, tol=1e-14, restarts=3, warm_starts=seeds)
        return [halley_refine_multiplicity(coeffs, z, steps=refine_steps) for z in roots]

    # hybrid:
    # Fast-path heuristic: if candidate centers all yield large |t|, skip series and go straight to Aberth
    try:
        # normalize to monic for scoring
        a_norm = [complex(x) / coeffs[-1] for x in coeffs]
        an = abs(a_norm[-1])
        R = 1 + max((abs(a) / an for a in a_norm[:-1]), default=0)
        cand = [0j]
        for r in [R / 8, R / 4, R / 2, R]:
            for k in range(16):
                theta = 2 * math.pi * k / 16
                cand.append(r * cmath.exp(1j * theta))
            cand.extend([r, -r, 1j * r, -1j * r])
        # score |t| = |-a0/a1|
        from .util import shift_expand

        t_mags = []
        for mu in cand:
            q = shift_expand(a_norm, mu)
            a0 = complex(q[0])
            a1 = complex(q[1]) if len(q) >= 2 else 0j
            if a1 == 0:
                continue
            t_mags.append(abs(-a0 / a1))
        min_t = min(t_mags) if t_mags else 1.0
        if min_t > 0.85:
            roots = aberth_ehrlich(coeffs, iters=200, tol=1e-14, restarts=3)
            return [
                halley_refine_multiplicity(coeffs, z, steps=refine_steps) for z in roots
            ]
    except Exception:
        pass

    # Get seeds via series from different centers
    seeds = []
    try:
        series_boots = 1 if n >= 12 else boots
        series_max_order = min(max_order, 16)
        seeds.append(
            solve_one(
                coeffs,
                center=None,
                max_order=series_max_order,
                boots=series_boots,
                tol=1e-14,
                resum=resum,
                refine_steps=refine_steps,
            )
        )
    except Exception:
        pass
    try:
        # second seed from opposite side
        an = abs(coeffs[-1])
        R = 1 + max((abs(a) / an for a in coeffs[:-1]), default=0)
        seeds.append(
            solve_one(
                coeffs,
                center=-R,
                max_order=series_max_order,
                boots=series_boots,
                tol=1e-14,
                resum=resum,
                refine_steps=refine_steps,
            )
        )
    except Exception:
        pass

    roots = aberth_ehrlich(coeffs, iters=200, tol=1e-14, restarts=3, warm_starts=seeds)
    return [halley_refine_multiplicity(coeffs, z, steps=refine_steps) for z in roots]


def solve_poly(coeffs: List[complex], **kwargs) -> List[complex]:
    """Public convenience wrapper (alias of solve_all)."""
    return solve_all(coeffs, **kwargs)
