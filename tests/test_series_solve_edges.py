from geodepoly.series_solve import newton_refine
from geodepoly.series_solve import choose_centers, series_step


def test_newton_refine_dp_zero_guard():
    # For p(x)=constant, dp==0 path triggers early exit without crash
    coeffs = [3.0]
    x = newton_refine(coeffs, 0.0, steps=3)
    # Should return a numeric value without error
    assert isinstance(x, (float, complex))


def test_choose_centers_runs():
    coeffs = [1.0, 0.0, 0.0, 1.0]
    cs = choose_centers(coeffs, samples=8, topk=4)
    assert len(cs) > 0


def test_series_step_guard_paths():
    # dp==0 path if a1==0 at center
    coeffs = [1.0, 0.0, 1.0]
    y, a0, a1, ok = series_step(coeffs, 0.0)
    assert ok in (True, False)


