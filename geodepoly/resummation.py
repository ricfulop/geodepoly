\
from __future__ import annotations
import math, cmath
from typing import List

# Plain evaluation of y = sum_{m>=1} g_m t^m
def eval_series_plain(g: List[complex], t: complex) -> complex:
    acc = 0j
    # Horner-like evaluation
    for gm in reversed(g):
        acc = acc * t + gm
    return acc * t

# Near-diagonal Pade via linear system; requires numpy
def pade_coeffs(c: List[complex], m: int, n: int):
    import numpy as np
    # c[0] + c[1] z + ... + c[m+n] z^{m+n}  ≈  P_m(z)/Q_n(z), Q_n(0)=1
    # Solve for Q (b1..bn): sum_{j=0}^n Q_j c_{k-j} = 0 for k=m+1..m+n, with Q_0=1
    # Then P_i by matching lower orders.
    kmin = m+1
    kmax = m+n
    A = np.zeros((n, n), dtype=complex)
    b = np.zeros((n,), dtype=complex)
    for row, k in enumerate(range(kmin, kmax+1)):
        for col, j in enumerate(range(1, n+1)):
            A[row, col-1] = c[k-j]
        b[row] = -c[k]
    q_tail = np.linalg.lstsq(A, b, rcond=None)[0]  # b1..bn
    Q = [1+0j] + list(q_tail)
    # Compute P by convolution up to order m
    P = [0j]*(m+1)
    for k in range(0, m+1):
        s = 0j
        for j in range(0, min(k, n)+1):
            s += Q[j]*c[k-j]
        P[k] = s
    return P, Q

def eval_series_pade(g: List[complex], t: complex) -> complex:
    # Our series is y = sum_{m>=1} g_m t^m = t * H(t), where H(0)=g_1.
    c = [0j] + list(g)  # make c[0]=0 so that y = sum c_k t^k
    order = min( max(4, len(g)//2) , len(g)-1)
    m = order
    n = order
    try:
        P, Q = pade_coeffs(c, m, n)
    except Exception:
        return eval_series_plain(g, t)
    # Evaluate P(t)/Q(t)
    num = 0j; den = 0j
    for a in reversed(P):
        num = num * t + a
    for b in reversed(Q):
        den = den * t + b
    if den == 0:
        return eval_series_plain(g, t)
    return num/den

# Simple Borel transform and Laplace inversion via Gauss-Laguerre quadrature
# y(t) = sum g_m t^m ≈ ∫_0^∞ e^{-s} B(t s) ds, where B(w) = sum (g_m/m!) w^m
# Use n=16 Gauss-Laguerre nodes (pretabulated)
GL16_x = [
    0.0876494104789, 0.462696328916, 1.14105777483, 2.12928364510,
    3.43708663364, 5.07801861455, 7.07033853505, 9.43831433639,
    12.213106787, 15.441527368, 19.190054314, 23.552663293,
    28.667338130, 34.782349972, 42.314838273, 52.312902457
]
GL16_w = [
    0.218234885940, 0.342210177922, 0.263027577942, 0.126425818106,
    0.040206864921, 0.008563877804, 0.001212436147, 0.000111674392,
    6.459926762e-06, 2.226316907e-07, 4.227430384e-09, 3.921897267e-11,
    1.456515266e-13, 8.059379366e-17, 4.211597427e-21, 1.772693296e-26
]

def borel_eval_B(g: List[complex], w: complex) -> complex:
    # B(w) = sum_{m>=1} (g_m/m!) w^m
    acc = 0j
    fact = 1.0
    poww = 1.0+0j
    for m, gm in enumerate(g, start=1):
        fact *= m
        poww *= w
        acc += gm * (poww / fact)
    return acc

def eval_series_borel(g: List[complex], t: complex) -> complex:
    # Approximate Laplace integral with Gauss-Laguerre
    s = 0j
    for x, w in zip(GL16_x, GL16_w):
        s += w * borel_eval_B(g, t * x)
    return s

def eval_series_borel_pade(g: List[complex], t: complex) -> complex:
    # Apply Pade to B(w) then Laplace
    # Build coefficients of B: b_m = g_m/m!
    import math
    b = []
    fact=1.0
    for m, gm in enumerate(g, start=1):
        fact *= m
        b.append(gm/fact)
    # Evaluate Pade for B at needed nodes and do Laguerre
    # Build c for B series: c[0]=0, c[m]=b_m
    c = [0j]+b
    order = min( max(4, len(b)//2) , len(b)-1)
    m = order; n = order
    try:
        P,Q = pade_coeffs(c, m, n)
    except Exception:
        return eval_series_borel(g, t)

    def B_pade(w):
        num=0j; den=0j
        for a in reversed(P):
            num = num*w+a
        for bq in reversed(Q):
            den = den*w+bq
        return num/den if den!=0 else num

    s = 0j
    for x, wgt in zip(GL16_x, GL16_w):
        s += wgt * B_pade(t*x)
    return s


def eval_series_auto(g: List[complex], t: complex) -> complex:
    """
    Adaptive evaluation of y = sum g_m t^m:
    - Try a small grid of near-diagonal Padé orders; score by denominator magnitude at t
      and simple last-term residual proxy.
    - If scores are poor or evaluation unstable, fall back to Borel–Padé; then to plain.
    """
    # Plain baseline
    plain = eval_series_plain(g, t)

    L = max(6, len(g))
    orders = []
    half = max(3, min(len(g)//2, 12))
    for delta in range(-2, 3):
        m = max(3, min(half + delta, len(g)-2))
        n = m
        orders.append((m, n))
    # Unique
    orders = list(dict.fromkeys(orders))

    best = None
    best_score = float('inf')
    best_val = plain

    # Build c for standard Padé of y(t)
    c = [0j] + list(g)
    for m, n in orders:
        try:
            P, Q = pade_coeffs(c, m, n)
            # Evaluate num/den and score
            num = 0j; den = 0j
            for a in reversed(P):
                num = num * t + a
            for b in reversed(Q):
                den = den * t + b
            if den == 0:
                continue
            val = num / den
            # Score: small |den| is bad; large last term magnitude is bad
            last_term = abs(g[-1] * (t ** len(g))) if g else 0.0
            score = (1.0 / max(1e-16, abs(den))) + last_term
            if score < best_score:
                best_score = score
                best = (m, n)
                best_val = val
        except Exception:
            continue

    # Heuristic threshold: if best score is too large, try Borel–Padé
    if best is None or best_score > 1e2:
        try:
            return eval_series_borel_pade(g, t)
        except Exception:
            return plain
    return best_val
