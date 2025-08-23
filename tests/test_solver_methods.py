import pytest

from geodepoly.solver import solve_all


def test_solver_numpy_method():
    coeffs = [-6.0, 11.0, -6.0, 1.0]
    roots = solve_all(coeffs, method="numpy")
    assert len(roots) == 3


def test_solver_batched_guard():
    coeffs = [-6.0, 11.0, -6.0, 1.0]
    roots = solve_all(coeffs, method="batched")
    assert len(roots) == 3



