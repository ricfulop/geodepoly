import argparse
import math
import random
import time
import cmath
from geodepoly import series_solve_all


def rand_poly(deg, seed):
    rnd = random.Random(seed)
    # Generate roots on random circle
    R = rnd.uniform(0.5, 2.0)
    roots = [R * cmath.exp(2j * math.pi * rnd.random()) for _ in range(deg)]
    # Build monic polynomial from roots
    coeffs = [1]
    for r in roots:
        coeffs = [0] + coeffs  # shift degree
        for k in range(len(coeffs) - 1):
            coeffs[k] -= coeffs[k + 1] * r
    coeffs = [complex(c) for c in coeffs]  # high to low
    coeffs_low_to_high = coeffs[::-1]
    return coeffs_low_to_high


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deg", type=int, default=6)
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    max_res = 0.0
    ok = 0
    t0 = time.time()
    for t in range(args.trials):
        coeffs = rand_poly(args.deg, args.seed + t)
        roots = series_solve_all(coeffs, verbose=False)
        # residual score
        import numpy as np

        p = np.poly1d(list(reversed(coeffs)))  # high-to-low for numpy
        res = max(abs(p(r)) for r in roots)
        max_res = max(max_res, res)
        if res < 1e-8:
            ok += 1
    dt = time.time() - t0
    print(
        f"Trials: {args.trials}, degree: {args.deg}, success {ok}/{args.trials}, max residual {max_res:.2e}, time {dt:.3f}s"
    )


if __name__ == "__main__":
    main()
