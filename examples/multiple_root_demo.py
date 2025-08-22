from geodepoly.solver import solve_all
from geodepoly.util import poly_eval


def poly_from_roots(roots):
    c = [1.0]
    for r in roots:
        c = [0.0] + c
        for k in range(len(c)-1):
            c[k] -= c[k+1]*r
    return [complex(x) for x in c[::-1]]


def main():
    coeffs = poly_from_roots([1.0, 1.0, 2.0])
    roots = solve_all(coeffs, method="hybrid", resum="auto")
    res = max(abs(poly_eval(coeffs, z)) for z in roots)
    print("roots:", roots)
    print("max residual:", res)


if __name__ == "__main__":
    main()


