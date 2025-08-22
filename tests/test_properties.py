import random, math, cmath
from geodepoly.solver import solve_all
from geodepoly.util import poly_eval


def poly_from_roots(roots):
    coeffs = [1.0]
    for r in roots:
        coeffs = [0.0] + coeffs
        for k in range(len(coeffs)-1):
            coeffs[k] -= coeffs[k+1]*r
    return [complex(x) for x in coeffs[::-1]]


def rand_poly_edgey(deg, seed):
    rnd = random.Random(seed)
    # Random roots on annulus with optional near-multiple pair
    roots = []
    R = rnd.uniform(0.5, 2.0)
    for _ in range(deg):
        roots.append(R*cmath.exp(2j*math.pi*rnd.random()))
    if deg >= 2 and rnd.random() < 0.5:
        roots[0] = 1.0
        roots[1] = 1.0 + 1e-6
    coeffs = poly_from_roots(roots)
    # Global scaling to vary coefficient magnitudes
    scale = 10**rnd.uniform(-2, 2)
    coeffs = [scale*c for c in coeffs]
    return coeffs


def test_random_edge_cases_small_degrees():
    for deg in [3,4,5]:
        for trial in range(3):
            coeffs = rand_poly_edgey(deg, 1000 + 10*deg + trial)
            roots = solve_all(coeffs, method="hybrid", resum="auto", tol=1e-12)
            res = max(abs(poly_eval(coeffs, z)) for z in roots)
            assert res < 1e-6

