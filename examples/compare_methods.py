import time, random, math, cmath
import numpy as np
from geodepoly.solver import solve_all


def rand_poly(deg, seed):
    rnd = random.Random(seed)
    roots = [rnd.uniform(0.5, 2.0) * cmath.exp(2j * math.pi * rnd.random()) for _ in range(deg)]
    p = np.poly1d([1.0])
    for r in roots:
        p *= np.poly1d([1.0, -r])
    return [complex(x) for x in p.c[::-1]]


def max_residual(coeffs, roots):
    p = np.poly1d(list(reversed(coeffs)))
    return float(max(abs(p(r)) for r in roots))


def main():
    deg = 8
    coeffs = rand_poly(deg, seed=0)
    methods = ["hybrid", "aberth", "dk", "numpy"]
    for m in methods:
        t0 = time.perf_counter()
        roots = solve_all(coeffs, method=m, resum="auto")
        dt = time.perf_counter() - t0
        res = max_residual(coeffs, roots)
        print(f"{m:6s}  time={dt:.4f}s  max|p(r)|={res:.3e}")


if __name__ == "__main__":
    main()


