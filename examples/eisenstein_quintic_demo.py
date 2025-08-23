"""Eisenstein/Bring quintic demo.

We consider the depressed quintic x^5 + x + a = 0 for a small parameter a.
Numerically compare geodepoly vs SymPy for a sample value of a.
"""

from geodepoly import solve_all


def main():
    a = 0.1
    # x^5 + x + a = 0 → coefficients low→high: [a, 1, 0, 0, 0, 1]
    coeffs = [a, 1.0, 0.0, 0.0, 0.0, 1.0]
    roots = solve_all(coeffs, method="hybrid", resum="auto")
    print("roots:", roots)


if __name__ == "__main__":
    main()


