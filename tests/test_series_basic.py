import pytest

from geodepoly.formal import FormalSeries
from geodepoly.series import series_root, series_bootstrap, geode_factorize


def test_formalseries_basic_add_mul_truncate():
    f = FormalSeries({(): 1.0, (1,): 2.0})  # 1 + 2 t
    g = FormalSeries({(1,): 1.0, (2,): 3.0})  # t + 3 t^2
    h = (f + g) * g
    ht = h.truncate_total_degree(2)
    # quick structure check
    assert isinstance(ht.coeff(()), complex)
    assert ht.coeff((1,)) != 0


def test_series_root_stub_shape():
    s = series_root([1, 2, 3], order=4)
    # should be a univariate series
    assert isinstance(s, FormalSeries)
    assert s.coeff((1,)) != 0


def test_series_bootstrap_wallis():
    x = series_bootstrap([ -5, -2, 0, 1 ], x0=2.0, series_order=24, rounds=3)
    assert abs(x - 2.0945514815423265) < 1e-8


def test_geode_factorization_small():
    S, S1, G = geode_factorize(order=4)
    # check identity (S - 1) == S1 * G up to truncation
    from geodepoly.formal import FormalSeries

    lhs = (S + FormalSeries({(): -1.0}, var_names=S.vars))
    rhs = (S1 * G).truncate_total_degree(4)
    # compare a few coefficients
    for m in [(), (1,), (2,), (3,), (4,)]:
        assert abs(lhs.coeff(m) - rhs.coeff(m)) < 1e-12


