import random, math, cmath
import numpy as np
from geodepoly.solver import solve_all
from geodepoly.util import poly_eval


def rand_poly(deg, seed):
    rnd = random.Random(seed)
    roots = [rnd.uniform(0.5, 2.0) * cmath.exp(2j * math.pi * rnd.random()) for _ in range(deg)]
    p = np.poly1d([1.0])
    for r in roots:
        p *= np.poly1d([1.0, -r])
    return [complex(x) for x in p.c[::-1]]


def test_random_degrees_up_to_20():
    degrees = [3, 5, 8, 12, 16]
    for deg in degrees:
        coeffs = rand_poly(deg, seed=deg)
        roots = solve_all(coeffs, method="hybrid", resum="auto", tol=1e-12)
        res = max(abs(poly_eval(coeffs, z)) for z in roots)
        assert res < 1e-6

