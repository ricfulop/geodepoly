from geodepoly.solver import solve_all


def test_solver_hybrid_fastpath_triggers_aberth():
    # A polynomial where centers likely yield large |t| (e.g., well-scaled high-degree)
    # We just check it returns roots; this path is exercised indirectly.
    coeffs = [1.0] + [0.0]*9 + [1.0]  # 1 + x^10
    roots = solve_all(coeffs, method="hybrid")
    assert len(roots) == 10



