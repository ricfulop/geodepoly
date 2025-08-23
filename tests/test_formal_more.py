from geodepoly.formal import FormalSeries


def test_formal_empty_and_scale():
    f = FormalSeries({})
    g = f.scale(2.0)
    assert g.coeff(()) == 0
    h = FormalSeries({(): 1.0}).scale(0.0)
    assert h.coeff(()) == 0


def test_formal_coeff_unknown_and_vars():
    f = FormalSeries({(2,): 3.0}, var_names=("x",))
    assert f.coeff((1,)) == 0
    # ensure var name is carried
    assert f.vars == ("x",)


