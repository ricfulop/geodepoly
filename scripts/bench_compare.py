\
import time, random, math, cmath, argparse, csv
import numpy as np
from geodepoly.solver import solve_all
from geodepoly.finishers import durand_kerner, aberth_ehrlich
from geodepoly.util import poly_eval

def rand_poly(deg, seed):
    rnd = random.Random(seed)
    # Random roots on annulus [0.5, 2.0]
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
    elif method == "aberth":
        roots = solve_all(coeffs, method="aberth", **kwargs)
    elif method == "dk":
        roots = solve_all(coeffs, method="dk", **kwargs)
    elif method == "hybrid":
        roots = solve_all(coeffs, method="hybrid", **kwargs)
    else:
        raise ValueError("unknown method")
    dt = time.perf_counter()-t0
    p = np.poly1d(list(reversed(coeffs)))
    res = float(max(abs(p(r)) for r in roots))
    return dt, res

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deg", type=int, default=8, help="Single degree (ignored if --degrees provided)")
    ap.add_argument("--degrees", type=str, default=None, help="Comma-separated degrees list, e.g. 3,5,8,12")
    ap.add_argument("--methods", type=str, default="hybrid,aberth,dk,numpy", help="Comma-separated methods")
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="bench_results.csv")
    ap.add_argument("--resum", type=str, default="auto", choices=["auto","pade","borel","borel-pade", "none"], help="Resummation for series seed")
    ap.add_argument("--agg_out", type=str, default=None, help="Optional path to write aggregate statistics CSV")
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

    # Optional aggregate statistics by (degree,method)
    if args.agg_out:
        import collections
        rows = []
        with open(args.out, "r", newline="") as f:
            rd = csv.DictReader(f)
            rows = list(rd)
        groups = collections.defaultdict(lambda: {"time": [], "res": []})
        for r in rows:
            key = (int(r["degree"]), r["method"]) 
            groups[key]["time"].append(float(r["time_s"]))
            groups[key]["res"].append(float(r["max_residual"]))
        def agg_stats(values):
            a = np.array(values, dtype=float)
            return float(np.mean(a)), float(np.median(a)), float(np.std(a))
        fieldnames2 = ["degree","method","time_mean","time_median","time_std","res_mean","res_median","res_std","success_rate"]
        with open(args.agg_out, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=fieldnames2)
            wr.writeheader()
            for (deg, m), v in sorted(groups.items()):
                tm, tmed, ts = agg_stats(v["time"]) if v["time"] else (0,0,0)
                rm, rmed, rs = agg_stats(v["res"]) if v["res"] else (0,0,0)
                succ = sum(1 for x in v["res"] if x < 1e-8) / max(1, len(v["res"]))
                wr.writerow({
                    "degree": deg,
                    "method": m,
                    "time_mean": f"{tm:.6f}",
                    "time_median": f"{tmed:.6f}",
                    "time_std": f"{ts:.6f}",
                    "res_mean": f"{rm:.3e}",
                    "res_median": f"{rmed:.3e}",
                    "res_std": f"{rs:.3e}",
                    "success_rate": f"{succ:.2f}"
                })
        print("Saved", args.agg_out)

if __name__ == "__main__":
    main()
