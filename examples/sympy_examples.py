def main():
    try:
        import sympy as sp
    except Exception:
        print("SymPy not installed; pip install sympy")
        return
    from geodepoly.sympy_plugin import sympy_solve

    x = sp.symbols('x')
    poly = x**5 + x - 1

    roots_num = sympy_solve(poly, method="hybrid", resum="auto", return_="numeric")
    roots_expr = sympy_solve(poly, method="hybrid", resum="auto", return_="expr")
    print("numeric roots:", roots_num)
    print("expr roots:", roots_expr)


if __name__ == "__main__":
    main()


