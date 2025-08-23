import pytest

from geodepoly.formal import FormalSeries


def test_formal_compose_univariate_raises_on_multivar():
    f = FormalSeries({(1, 0): 1.0, (0, 1): 1.0}, var_names=("t","u"))
    g = FormalSeries({(1,): 1.0})
    with pytest.raises(NotImplementedError):
        f.compose_univariate(g)



