from __future__ import annotations

import cmath

from geodepoly.geode import bring_radical_series


def test_bring_quintic_series_residual_small():
    # For small t, the residual of y - t - y^5 should scale like O(t^{6}).
    for t in [1e-3, 5e-4, 1e-4]:
        y = bring_radical_series(t, d=5, terms=24)
        res = abs(y - t - y**5)
        # Very small for tiny t
        assert res < 1e-10


