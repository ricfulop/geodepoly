def main():
    try:
        import sympy as sp
    except Exception as e:
        print("SymPy not installed; pip install sympy to run this example")
        return

    from geodepoly.sympy_plugin import sympy_solve

    x = sp.symbols('x')
    poly = x**8 - 3*x + 1

    roots_geode = sympy_solve(poly, method="hybrid", resum="auto")
    roots_sympy = [complex(r) for r in sp.nroots(poly)]

    def max_residual(roots):
        p = sp.Poly(poly, x)
        return max(abs(complex(p.eval(r))) for r in roots)

    print("geodepoly roots:", roots_geode)
    print("sympy nroots:", roots_sympy)
    print("max residual (geodepoly):", max_residual(roots_geode))
    print("max residual (sympy):   ", max_residual(roots_sympy))


if __name__ == "__main__":
    main()


