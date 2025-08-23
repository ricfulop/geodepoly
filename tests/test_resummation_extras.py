from geodepoly.resummation import eval_series_plain, eval_series_pade, eval_series_borel, eval_series_auto


def test_resummation_paths():
    g = [1.0, 0.5, 0.25, 0.125]
    t_small = 0.1
    t_mid = 0.9
    v_plain = eval_series_plain(g, t_small)
    v_pade = eval_series_pade(g, t_small)
    assert abs(v_pade - v_plain) < 1e-3
    # auto should produce a finite value near t~1 and not crash
    v_auto = eval_series_auto(g, t_mid)
    assert v_auto == v_auto  # not NaN



