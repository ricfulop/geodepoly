import time, random, math, cmath, argparse, csv
import numpy as np
from geodepoly.solver import solve_all


def rand_poly(deg, seed):
    rnd = random.Random(seed)
    roots = [rnd.uniform(0.5, 2.0) * cmath.exp(2j * math.pi * rnd.random()) for _ in range(deg)]
    p = np.poly1d([1.0])
    for r in roots:
        p *= np.poly1d([1.0, -r])
    return [complex(x) for x in p.c[::-1]]


def newton_baseline(coeffs, max_iters=200, tol=1e-12):
    # Simple Newton on each root starting from Aberth circle guesses, no deflation
    a = [complex(x) for x in coeffs]
    if a[-1] == 0:
        raise ValueError("Leading coefficient is zero.")
    a = [x / a[-1] for x in a]
    n = len(a) - 1
    R = 1 + max((abs(c) for c in a[:-1]), default=0)
    guesses = [R * cmath.exp(2j * math.pi * k / n) for k in range(n)]

    def poly_dp(x):
        p = 0j
        dp = 0j
        for c in reversed(a):
            dp = dp * x + p
            p = p * x + c
        return p, dp

    roots = []
    for z in guesses:
        for _ in range(max_iters):
            p, dp = poly_dp(z)
            if dp == 0:
                break
            delta = p / dp
            z = z - delta
            if abs(delta) < tol:
                break
        roots.append(z)
    return roots


def max_residual(coeffs, roots):
    p = np.poly1d(list(reversed(coeffs)))
    return float(max(abs(p(r)) for r in roots))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--degrees", default="3,5,8,12,16,20")
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="bench_newton_vs_hybrid.csv")
    args = ap.parse_args()
    degrees = [int(x) for x in args.degrees.split(",") if x]

    fieldnames = ["degree","trial","method","time_s","max_residual"]
    with open(args.out, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fieldnames)
        wr.writeheader()
        for deg in degrees:
            for t in range(args.trials):
                coeffs = rand_poly(deg, args.seed + 1000 * deg + t)

                t0 = time.perf_counter();
                roots_h = solve_all(coeffs, method="hybrid", resum="auto", tol=1e-12)
                th = time.perf_counter() - t0
                rh = max_residual(coeffs, roots_h)
                wr.writerow({"degree":deg,"trial":t,"method":"hybrid","time_s":f"{th:.6f}","max_residual":f"{rh:.3e}"})

                t0 = time.perf_counter();
                roots_n = newton_baseline(coeffs)
                tn = time.perf_counter() - t0
                rn = max_residual(coeffs, roots_n)
                wr.writerow({"degree":deg,"trial":t,"method":"newton","time_s":f"{tn:.6f}","max_residual":f"{rn:.3e}"})
    print("Saved", args.out)


if __name__ == "__main__":
    main()


