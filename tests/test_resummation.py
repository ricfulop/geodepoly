from geodepoly.resummation import eval_series_plain, eval_series_pade, eval_series_auto


def test_eval_series_auto_prefers_pade_near_unit_circle():
    # Construct a geometric-like series with slow convergence near |t| ~ 1
    # g_m = 1 for m<=M (truncated), so y ~ t + t^2 + ...
    M = 20
    g = [1.0] * M
    t = 0.9
    plain = float(abs(eval_series_plain(g, t)))
    auto = float(abs(eval_series_auto(g, t)))
    # Heuristic: auto should not be worse than plain by large factor
    assert auto <= 1.2 * plain


