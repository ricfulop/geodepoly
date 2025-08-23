import random, math, cmath
import numpy as np
from geodepoly.series_core import inverseseries_g_coeffs
from geodepoly.util import shift_expand
from geodepoly.resummation import eval_series_plain, eval_series_pade, eval_series_borel_pade, eval_series_auto


def demo_series_step(coeffs, center=0j, t=None):
    q = shift_expand(coeffs, center)
    a0 = complex(q[0]); a1 = complex(q[1])
    beta = {}
    for k in range(2, min(len(q), 34)):
        beta[k] = q[k] / a1
    g = inverseseries_g_coeffs(beta, max_order=32)
    t = (-a0 / a1) if t is None else t
    plain = eval_series_plain(g, t)
    pade = eval_series_pade(g, t)
    bp = eval_series_borel_pade(g, t)
    auto = eval_series_auto(g, t)
    return t, plain, pade, bp, auto


def main():
    # cubic near boundary where |t| is not tiny
    coeffs = [1, -1.2, 0.3, 1.0]
    t, plain, pade, bp, auto = demo_series_step(coeffs, center=0.3)
    print(f"t={t:.3g}\nplain={plain:.6g}\npade={pade:.6g}\nborel-pade={bp:.6g}\nauto={auto:.6g}")


if __name__ == "__main__":
    main()


