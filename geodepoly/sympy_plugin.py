def sympy_solve(poly, return_: str = "numeric", **kwargs):
    """
    Accept a SymPy Expr or Poly and return complex roots using geodepoly.
    Keyword args mirror geodepoly.solve_all: method, max_order, boots, tol, resum, refine_steps, verbose.
    """
    try:
        import sympy as sp
    except ImportError:
        raise ImportError("SymPy not installed. `pip install sympy`")

    from .solver import solve_all

    if isinstance(poly, sp.Expr):
        if len(poly.free_symbols) != 1:
            raise ValueError("Only univariate expressions are supported.")
        x = list(poly.free_symbols)[0]
        p = sp.Poly(poly, x)
    elif isinstance(poly, sp.Poly):
        p = poly
    else:
        raise TypeError("Expected sympy.Expr or sympy.Poly")

    coeffs = [complex(c) for c in p.all_coeffs()[::-1]]  # low->high
    roots = solve_all(coeffs, **kwargs)
    if return_ == "numeric":
        return roots
    elif return_ == "expr":
        # Round-trip back to SymPy complex numbers
        return [sp.nsimplify(r) for r in roots]
    else:
        raise ValueError("return_ must be 'numeric' or 'expr'")
