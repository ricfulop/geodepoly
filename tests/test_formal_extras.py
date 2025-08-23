import pytest

from geodepoly.formal import FormalSeries


def test_formal_truncate_and_compose_univariate():
    # f(x) = 1 + 2 x + 3 x^2
    f = FormalSeries({(): 1.0, (1,): 2.0, (2,): 3.0})
    # g(x) = x + x^2
    g = FormalSeries({(1,): 1.0, (2,): 1.0})
    h = f.compose_univariate(g, max_deg=3)
    # Check a couple of coefficients exist
    assert isinstance(h.coeff(()), complex)
    assert h.coeff((1,)) != 0
    ht = h.truncate_total_degree(2)
    # Degree-3 term dropped
    assert ht.coeff((3,)) == 0


def test_formal_to_sympy_guard():
    s = FormalSeries({(): 1.0, (1,): 2.0})
    try:
        expr = s.to_sympy()
        assert str(expr)  # convertible
    except ImportError:
        # acceptable if sympy not installed
        pass



