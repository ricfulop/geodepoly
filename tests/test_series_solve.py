import math, cmath, random
from geodepoly import series_solve_all
import numpy as np

def poly_from_roots(roots):
    p = np.poly1d([1.0])
    for r in roots:
        p *= np.poly1d([1.0, -r])
    coeffs = p.c[::-1]  # low to high
    return [complex(x) for x in coeffs]

def test_random_polys():
    random.seed(0)
    for deg in [3,4,5,6]:
        for _ in range(5):
            roots_true = [cmath.exp(2j*math.pi*random.random()) for _ in range(deg)]
            coeffs = poly_from_roots(roots_true)
            roots = series_solve_all(coeffs, tol=1e-12)
            # Verify each true root is close to some computed root
            for rt in roots_true:
                assert min(abs(rt - r) for r in roots) < 1e-6
