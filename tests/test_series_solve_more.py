from geodepoly.series_solve import series_step, series_one_root, series_solve_all, deflate, choose_center
from geodepoly.util import poly_eval


def test_series_step_success_quadratic():
    coeffs = [2.0, -3.0, 1.0]  # (x-1)(x-2)
    y, a0, a1, ok = series_step(coeffs, 0.0, max_order=8)
    assert ok is True
    assert a1 != 0


def test_series_one_root_no_refine():
    coeffs = [2.0, -3.0, 1.0]
    r = series_one_root(coeffs, center=0.0, max_order=8, boots=2, refine=False)
    assert isinstance(r, (float, complex))


def test_deflate_known_root():
    coeffs = [2.0, -3.0, 1.0]
    q = deflate(coeffs, 1.0)
    # quotient should be x-2 => [-2,1]
    assert len(q) == 2 and abs(q[0] + 2.0) < 1e-12 and abs(q[1] - 1.0) < 1e-12


def test_series_solve_all_small_poly():
    coeffs = [2.0, -3.0, 1.0]
    roots = series_solve_all(coeffs, max_order=12, boots=2)
    assert len(roots) == 2
    assert max(abs(poly_eval(coeffs, z)) for z in roots) < 1e-6


def test_choose_center_runs_simple():
    coeffs = [2.0, -3.0, 1.0]
    mu = choose_center(coeffs, samples=8)
    assert isinstance(mu, complex)


