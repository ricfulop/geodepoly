from geodepoly.util import shift_expand


def test_shift_expand_basic():
    # p(x) = 1 + 2x + 3x^2, shift by mu=1 => q(y) = p(1+y)
    coeffs = [1.0, 2.0, 3.0]
    q = shift_expand(coeffs, 1.0)
    # q0 = p(1) = 1 + 2 + 3 = 6
    # q1 = p'(1) = 2 + 6 = 8
    assert abs(q[0] - 6.0) < 1e-12
    assert abs(q[1] - 8.0) < 1e-12



