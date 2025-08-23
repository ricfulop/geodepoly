from geodepoly.poly import Polynomial


def test_poly_add_mul_eval():
    p = Polynomial([1, 2])       # 1 + 2x
    q = Polynomial([0, 1, 1])    # x + x^2
    r = p + q                    # 1 + 3x + x^2
    s = p * q                    # (1+2x)(x+x^2) = x + 3x^2 + 2x^3
    assert r.coeffs == [1+0j, 3+0j, 1+0j]
    assert s.coeffs == [0+0j, 1+0j, 3+0j, 2+0j]
    assert abs(r(2) - (1 + 3*2 + 4)) < 1e-12


def test_poly_shift_scale_diff_integrate():
    p = Polynomial([1, 0, 1])  # 1 + x^2
    q = p.shift_x(1)           # (1 + (1+y)^2) = 2 + 2y + y^2
    assert q.coeffs == [2+0j, 2+0j, 1+0j]
    s = p.scale_x(2)           # 1 + (2x)^2 = 1 + 4 x^2
    assert s.coeffs == [1+0j, 0+0j, 4+0j]
    dp = p.differentiate()     # 2x
    assert dp.coeffs == [0+0j, 2+0j]
    ip = dp.integrate(3)       # 3 + x^2
    assert ip.coeffs == [3+0j, 0+0j, 1+0j]


