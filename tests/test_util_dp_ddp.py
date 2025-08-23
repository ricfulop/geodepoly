from geodepoly.util import poly_eval_dp_ddp, set_numba_acceleration


def test_util_poly_eval_dp_ddp_paths():
    coeffs = [1.0, 0.0, -7.0, 6.0]
    set_numba_acceleration(False)
    p, dp, ddp = poly_eval_dp_ddp(coeffs, 1.0)
    set_numba_acceleration(True)
    p2, dp2, ddp2 = poly_eval_dp_ddp(coeffs, 1.0)
    assert abs(p - p2) < 1e-12
    assert abs(dp - dp2) < 1e-12
    assert abs(ddp - ddp2) < 1e-12



