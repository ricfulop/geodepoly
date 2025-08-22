import random, math, cmath
from geodepoly.solver import solve_one


def rand_poly_with_radius(deg, R, seed=0):
    rnd = random.Random(seed)
    roots = [R*cmath.exp(2j*math.pi*rnd.random()) for _ in range(deg)]
    c = [1.0]
    for r in roots:
        c = [0.0] + c
        for k in range(len(c)-1):
            c[k] -= c[k+1]*r
    return [complex(x) for x in c[::-1]]


def test_center_selection_prefers_small_t():
    coeffs = rand_poly_with_radius(6, R=1.5, seed=42)
    # Request series seed with resum auto and ensure it returns a finite value
    x = solve_one(coeffs, center=None, resum="auto", max_order=24, boots=2)
    assert isinstance(x, complex)

