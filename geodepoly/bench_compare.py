import time, random, math, cmath, argparse, csv
import numpy as np
from geodepoly.solver import solve_all


def rand_poly(deg, seed):
    rnd = random.Random(seed)
    roots = [rnd.uniform(0.5,2.0)*cmath.exp(2j*math.pi*rnd.random()) for _ in range(deg)]
    p = np.poly1d([1.0])
    for r in roots:
        p *= np.poly1d([1.0, -r])
    coeffs_low = p.c[::-1]
    return [complex(x) for x in coeffs_low]


def measure(coeffs, method, **kwargs):
    t0 = time.perf_counter()
    if method == "numpy":
        a = [complex(x) for x in coeffs]
        a = [x/a[-1] for x in a]
        n = len(a)-1
        C = np.zeros((n,n), dtype=complex)
        C[1:, :-1] = np.eye(n-1, dtype=complex)
        C[:, -1] = -np.array(a[:-1], dtype=complex)
        roots = np.linalg.eigvals(C)
        roots = [complex(z) for z in roots]
    else:
        roots = solve_all(coeffs, method=method, **kwargs)
    dt = time.perf_counter()-t0
    p = np.poly1d(list(reversed(coeffs)))
    res = float(max(abs(p(r)) for r in roots))
    return dt, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deg", type=int, default=8)
    ap.add_argument("--degrees", type=str, default=None)
    ap.add_argument("--methods", type=str, default="hybrid,aberth,dk,numpy")
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="bench_results.csv")
    ap.add_argument("--resum", type=str, default="auto", choices=["auto","pade","borel","borel-pade", "none"], help="Resummation for series seed")
    args = ap.parse_args()

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    degrees = [args.deg] if not args.degrees else [int(x) for x in args.degrees.split(",") if x.strip()]
    fieldnames = ["trial","degree","method","time_s","max_residual"]
    with open(args.out, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fieldnames)
        wr.writeheader()
        for deg in degrees:
            for t in range(args.trials):
                coeffs = rand_poly(deg, args.seed+t)
                for m in methods:
                    kwargs = {"tol": 1e-12}
                    if args.resum != "none":
                        kwargs["resum"] = args.resum
                    dt, res = measure(coeffs, m, **kwargs)
                    wr.writerow({"trial":t, "degree":deg, "method":m, "time_s":f"{dt:.6f}", "max_residual":f"{res:.3e}"})
    print("Saved", args.out)


if __name__ == "__main__":
    main()


