\
from __future__ import annotations
import cmath, math, random
from typing import List, Tuple, Optional

def poly_eval(coeffs, x):
    """Evaluate polynomial a0 + a1 x + ... + aN x^N at x using Horner."""
    acc = 0j
    for a in reversed(coeffs):
        acc = acc * x + complex(a)
    return acc

def poly_deriv_at(coeffs, x):
    """First derivative at x by Horner (for diagnostics/fallbacks only)."""
    acc = 0j
    for k, a in enumerate(coeffs[1:], start=1):
        acc += k * a * (x ** (k - 1))
    return complex(acc)

def shift_expand(coeffs, mu):
    """Return coefficients of q(y) = p(mu + y), up to degree N."""
    # Binomial expansion via dynamic programming
    n = len(coeffs) - 1
    q = [0j]*(n+1)
    # p(mu+y) = sum_k a_k (mu+y)^k = sum_k a_k sum_j binom(k,j) mu^{k-j} y^j
    # Accumulate per j
    from math import comb
    for k, a in enumerate(coeffs):
        ak = complex(a)
        if abs(ak) == 0:
            continue
        for j in range(k+1):
            q[j] += ak * (comb(k, j) * (mu ** (k - j)))
    return q  # a0, a1, ..., aN

def inverseseries_g_coeffs(beta, max_order: int) -> List[complex]:
    """
    Compute coefficients g_m (m=1..max_order) of the compositional inverse of
    F(y) = y + sum_{k>=2} beta[k] * y^k, using Lagrange inversion:
      g_m = (1/m) * [y^{m-1}] (1 / F'(y))^m,
    where F'(y) = 1 + sum_{m>=1} a_m y^m, a_m = (m+1)*beta[m+1].

    beta is a dict-like with beta[k] defined for k=2..max_order+1.
    Returns list [g1, g2, ..., g_max_order].
    """
    # Build U(y) = sum_{m>=1} a_m y^m up to degree max_order-1
    a = [0j]*(max_order)  # a[0] unused, a[m] is coeff of y^m
    for m in range(1, max_order):
        b = beta.get(m+1, 0)
        a[m] = (m+1) * b

    # Precompute powers U(y)^j up to degree max_order-1
    # Represent a series as list c where c[d] is coeff of y^d, c[0] is constant term
    def series_mul(x, y, deg):
        out = [0j]*(deg+1)
        for i in range(min(len(x)-1, deg)+1):
            xi = x[i] if i < len(x) else 0
            if xi == 0: continue
            maxj = min(deg - i, len(y)-1)
            for j in range(maxj+1):
                yj = y[j] if j < len(y) else 0
                if yj == 0: continue
                out[i+j] += xi * yj
        return out

    deg = max_order-1
    U = [0j]*(deg+1)
    U[0] = 0
    for m in range(1, deg+1):
        U[m] = a[m] if m < len(a) else 0

    U_pows = []
    # U^0 == 1
    one = [0j]*(deg+1); one[0] = 1
    U_pows.append(one)
    if deg >= 1:
        U_pows.append(U)
    for j in range(2, max_order):  # up to power max_order-1
        U_pows.append(series_mul(U_pows[-1], U, deg))

    # Binomial coefficients for negative exponent: (1+U)^(-n) = sum_{j>=0} C(n,j) U^j
    # C(n,j) = (-n choose j) = (-1)^j * comb(n+j-1, j)
    from math import comb
    g = [0j]*max_order
    for m in range(1, max_order+1):
        # H_m(y) = (1 / F'(y))^m = (1 + U)^(-m)
        # coeff y^{m-1} = sum_{j=0}^{m-1} C(m,j) * [y^{m-1}] U^j
        target_deg = m-1
        coeff = 0j
        for j in range(0, m):  # only need up to degree m-1
            C_mj = ((-1)**j) * comb(m + j - 1, j)
            coeff += C_mj * (U_pows[j][target_deg] if target_deg < len(U_pows[j]) else 0)
        g[m-1] = coeff / m
    return g

def series_step(coeffs, center, max_order=24):
    """
    One analytic step: expand q(y) = p(center + y), compute inverse series,
    and evaluate y ≈ sum g_m t^m, with t = -a0/a1.
    Returns (y_est, a0, a1, ok_flag)
    """
    q = shift_expand(coeffs, center)
    a0 = complex(q[0])
    a1 = complex(q[1]) if len(q) >= 2 else 0j
    if abs(a1) == 0:
        return 0j, a0, a1, False
    # Build beta dict beta[k] = a_k / a1
    beta = {}
    for k in range(2, min(len(q), max_order+2)):
        beta[k] = q[k] / a1
    g = inverseseries_g_coeffs(beta, max_order=max_order)
    t = -a0 / a1
    # Horner-like evaluation of y = sum g_m t^m
    y = 0j
    for m in range(max_order, 0, -1):
        y = y * t + g[m-1]
    # Multiply by t since Horner above yields sum g_{m} t^{m-1}
    y = y * t
    return y, a0, a1, True

def halley_refine(coeffs, x, steps=6):
    """A couple of Halley refinement steps to polish, if desired."""
    # Compute p, p', p''
    for _ in range(steps):
        # p, p', p'' via Horner-like recurrences
        p = 0j; dp = 0j; ddp = 0j
        for a in reversed(coeffs):
            ddp = ddp * x + 2*dp
            dp  = dp * x + p
            p   = p  * x + a
        denom = 2*dp*dp - p*ddp
        if abs(denom) == 0:
            break
        x = x - (2*p*dp) / denom
    return x



def newton_refine(coeffs, x, steps=20, tol=1e-15):
    """Refine a root with standard Newton iterations until residual is tiny."""
    for _ in range(steps):
        # Evaluate p and p'
        p = 0j; dp = 0j
        for a in reversed(coeffs):
            dp = dp * x + p
            p = p * x + a
        if abs(p) < tol:
            break
        if dp == 0:
            break
        x = x - p / dp
    return x

def durand_kerner(coeffs, iters=1000, tol=1e-14, restarts=3):
    """Durand–Kerner with convergence check and random restarts."""
    n = len(coeffs)-1
    an = coeffs[-1]
    if an == 0:
        raise ValueError("Leading coefficient is zero.")
    base_R = 1 + max((abs(a)/abs(an) for a in coeffs[:-1]), default=0)

    for attempt in range(restarts):
        R = base_R * (1 + 0.1*(random.random()-0.5))
        roots = [R * cmath.exp(2j*math.pi*(k+0.3*attempt)/n) for k in range(n)]
        for it in range(iters):
            new_roots = []
            max_delta = 0.0
            for i, z in enumerate(roots):
                denom = 1.0 + 0j
                for j, w in enumerate(roots):
                    if j != i:
                        denom *= (z - w)
                if denom == 0:
                    z += (random.random()-0.5) * 1e-8 + 1j*(random.random()-0.5)*1e-8
                    denom = 1.0 + 0j
                    for j, w in enumerate(roots):
                        if j != i:
                            denom *= (z - w)
                val = poly_eval(coeffs, z)
                z_new = z - val/(an*denom)
                new_roots.append(z_new)
                max_delta = max(max_delta, abs(z_new - z))
            roots = new_roots
            if max_delta < tol:
                return roots
        # try another restart
    return roots

def deflate(coeffs, root):
    """Synthetic division by (x - root). Coeffs are a0..aN. Returns new coeffs of degree-1."""
    n = len(coeffs)-1
    b = [0j]*(n)
    b[-1] = coeffs[-1]
    for k in range(n-2, -1, -1):
        b[k] = coeffs[k+1] + root*b[k+1]
    # Remainder r = coeffs[0] + root*b[0]; we ignore here.
    return b




def choose_centers(coeffs, samples=16, topk=8):
    """Return up to topk centers (mu) minimizing |p(mu)/p'(mu)|."""
    an = abs(coeffs[-1])
    if an == 0:
        raise ValueError("Leading coefficient is zero.")
    R = 1 + max((abs(a)/an for a in coeffs[:-1]), default=0)
    radii = [0.0, R/12, R/8, R/6, R/4, R/2, R]
    candidates = set([0j])
    for r in radii[1:]:
        for k in range(samples):
            theta = 2*math.pi*(k/samples)
            candidates.add(r*cmath.exp(1j*theta))
        # axes
        candidates.update([r, -r, 1j*r, -1j*r])
    scored = []
    for mu in candidates:
        p = poly_eval(coeffs, mu)
        dp = poly_deriv_at(coeffs, mu)
        if dp == 0: 
            continue
        score = abs(p/dp)
        scored.append((score, mu))
    scored.sort(key=lambda x: x[0])
    return [mu for _, mu in scored[:topk]]


def choose_center(coeffs, samples=16):
    """
    Pick a center mu that minimizes |p(mu)/p'(mu)| using a coarse search on
    multiple radii from small to Cauchy bound.
    """
    an = abs(coeffs[-1])
    if an == 0:
        raise ValueError("Leading coefficient is zero.")
    R = 1 + max((abs(a)/an for a in coeffs[:-1]), default=0)
    radii = [0.0, R/8, R/4, R/2, R]
    candidates = [0j]
    for r in radii[1:]:
        for k in range(samples):
            theta = 2*math.pi*(k/samples)
            candidates.append(r*cmath.exp(1j*theta))
        # also probe along axes
        candidates.extend([r, -r, 1j*r, -1j*r])
    best_mu = 0j
    best_score = float("inf")
    for mu in candidates:
        p = poly_eval(coeffs, mu)
        dp = poly_deriv_at(coeffs, mu)
        if dp == 0: 
            continue
        score = abs(p/dp)
        if score < best_score:
            best_score = score
            best_mu = mu
    return best_mu




def series_one_root(coeffs: List[complex], center: complex|None=None, max_order=24, boots=3, tol=1e-14, refine=True):
    """
    Find a single root using bootstrap series steps from one of several candidate centers.
    If center is None, we score multiple centers mu by |p(mu)/p'(mu)| and try the best few.
    """
    if coeffs[-1] == 0:
        raise ValueError("Leading coefficient is zero.")
    # Normalize to monic
    scale = coeffs[-1]
    c = [complex(a)/scale for a in coeffs]

    # Candidate centers
    centers = [center] if center is not None else choose_centers(c, topk=10)

    best = None
    best_res = float('inf')

    for mu in centers:
        x = complex(mu)
        ok_local = False
        for _ in range(max(1, boots)):
            y, a0, a1, ok = series_step(c, x, max_order=max_order)
            if not ok or abs(a1) < 1e-18:
                ok_local = False
                break
            t = -a0/a1
            if abs(t) > 0.9:
                ok_local = False
                break
            x = x + y
            ok_local = True
            if abs(poly_eval(c, x)) < tol:
                break
        if ok_local:
            xr = halley_refine(c, x, steps=6) if refine else x
            res = abs(poly_eval(c, xr))
            if res < best_res:
                best_res = res
                best = xr

    if best is None:
        # last resort: pick smallest |p(mu)| among candidates
        best = min(centers, key=lambda mu: abs(poly_eval(c, mu)))

    return best



def series_solve_all(coeffs: List[complex], max_order=24, boots=3, tol=1e-12, max_deflation=None, verbose=False):
    """
    Solve for all roots. Try to grab at least one root by series bootstrap; then
    finish with Durand–Kerner as a robust, derivative-free method. This keeps the
    MVP reliable while showcasing the series step.
    """
    n = len(coeffs)-1
    if n <= 0:
        return []

    # Try to obtain a high-quality seed root via series
    try:
        r0 = series_one_root(coeffs, center=None, max_order=max_order, boots=boots, tol=tol, refine=True)
    except Exception:
        r0 = None

    # Compute all roots with Durand–Kerner
    dk_roots = durand_kerner(coeffs, iters=1000, tol=1e-14, restarts=4)

    # If we have a series root close to some dk root, replace/polish it
    if r0 is not None:
        # pick closest dk root and replace with refined series root
        idx = min(range(len(dk_roots)), key=lambda i: abs(dk_roots[i] - r0))
        dk_roots[idx] = r0

    # Final polish with Halley and then Newton (on original poly)
    polished = [newton_refine(coeffs, halley_refine(coeffs, z, steps=8), steps=20, tol=1e-15) for z in dk_roots]
    return polished
