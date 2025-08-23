import numpy as np
from geodepoly.batched import batched_poly_eval, batched_newton_step
from geodepoly.util import poly_eval


def test_batched_poly_eval_matches_numpy():
    B, D = 4, 5
    rng = np.random.default_rng(0)
    coeffs = rng.standard_normal((B, D + 1)) + 1j * rng.standard_normal((B, D + 1))
    xs = rng.standard_normal((B, 3)) + 1j * rng.standard_normal((B, 3))
    out = batched_poly_eval(coeffs, xs, backend="numpy")
    ref = np.stack([[poly_eval(list(coeffs[b]), x) for x in xs[b]] for b in range(B)])
    assert np.max(np.abs(out - ref)) < 1e-10


def test_batched_newton_step_shapes():
    B, D = 3, 4
    coeffs = np.ones((B, D + 1), dtype=complex)
    xs = np.zeros((B,), dtype=complex)
    step = batched_newton_step(coeffs, xs, backend="numpy")
    assert step.shape == xs.shape


