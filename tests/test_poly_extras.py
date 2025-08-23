from geodepoly.poly import Polynomial


def test_poly_divmod_pow_edgecases():
    p = Polynomial([1, 2, 1])  # (x+1)^2
    q = Polynomial([1, -1])    # (x-1)
    div, rem = divmod(p, q)
    # (x+1)^2 / (x-1) = x+3 + 4/(x-1), so quotient degree 1
    assert div.degree == 1
    r = Polynomial([0, 1]) ** 3  # x^3
    assert r.coeffs == [0+0j, 0+0j, 0+0j, 1+0j]


def test_poly_scale_shift_zero_constant():
    p = Polynomial([5])  # constant
    assert p.shift_x(3).coeffs == [5+0j]
    assert p.scale_x(2).coeffs == [5+0j]



