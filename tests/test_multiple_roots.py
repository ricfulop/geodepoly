from geodepoly.finishers import halley_refine_multiplicity
from geodepoly.util import poly_eval


def coeffs_from_roots(roots):
    c = [1.0]
    for r in roots:
        c = [0.0] + c
        for k in range(len(c)-1):
            c[k] -= c[k+1]*r
    return [complex(x) for x in c[::-1]]


def test_double_root_refinement():
    coeffs = coeffs_from_roots([1.0, 1.0, 2.0])
    x0 = 0.9
    x = halley_refine_multiplicity(coeffs, x0, steps=10)
    assert abs(poly_eval(coeffs, x)) < 1e-8


def test_triple_root_refinement():
    coeffs = coeffs_from_roots([1.0, 1.0, 1.0])
    x0 = 1.1
    x = halley_refine_multiplicity(coeffs, x0, steps=12)
    assert abs(poly_eval(coeffs, x)) < 1e-8


