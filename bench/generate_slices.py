import csv, argparse, random, math, cmath
from geodepoly.hyper_catalan import evaluate_quadratic_slice


def catalan_slice_rows(degrees, trials, t2=0.1):
    rows = []
    for d in degrees:
        for t in range(trials):
            # For demo: use S[t2] to generate an alpha value as a proxy feature
            alpha = evaluate_quadratic_slice(t2, max_weight=32)
            rows.append({"slice":"catalan", "degree": d, "trial": t, "alpha": f"{alpha:.6g}"})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--degrees", default="3,5,8,12")
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--out", default="bench/geodebench_slices.csv")
    args = ap.parse_args()
    degrees = [int(x) for x in args.degrees.split(",") if x]
    rows = []
    rows += catalan_slice_rows(degrees, args.trials)
    with open(args.out, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["slice","degree","trial","alpha"])
        wr.writeheader()
        for r in rows:
            wr.writerow(r)
    print("Saved", args.out)


if __name__ == "__main__":
    main()


