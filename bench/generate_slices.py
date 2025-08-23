import argparse, csv, random
from geodepoly.hyper_catalan import evaluate_hyper_catalan, evaluate_quadratic_slice


def gen_row(slice_name: str, degree: int, seed: int):
    rnd = random.Random(seed)
    if slice_name == "S":
        t2 = rnd.uniform(-0.2, 0.2)
        alpha = evaluate_quadratic_slice(t2, max_weight=24)
        return {"slice": "S", "degree": degree, "trial": seed, "t2": f"{t2:.8g}", "alpha": f"{alpha.real:.12g}+{alpha.imag:.12g}j"}
    else:
        # G slice: small multivariate setting
        tvals = {2: rnd.uniform(-0.1, 0.1), 3: rnd.uniform(-0.05, 0.05)}
        alpha = evaluate_hyper_catalan(tvals, max_weight=degree)
        return {
            "slice": "G",
            "degree": degree,
            "trial": seed,
            "t2": f"{tvals[2]:.8g}",
            "t3": f"{tvals[3]:.8g}",
            "alpha": f"{alpha.real:.12g}+{alpha.imag:.12g}j",
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--degrees", type=str, default="3,5,8")
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    degrees = [int(x) for x in args.degrees.split(",") if x.strip()]
    fieldnames = ["slice", "degree", "trial", "t2", "t3", "alpha"]
    with open(args.out, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fieldnames)
        wr.writeheader()
        for deg in degrees:
            for tr in range(args.trials):
                wr.writerow(gen_row("S", deg, tr))
                wr.writerow(gen_row("G", deg, tr))


if __name__ == "__main__":
    main()

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


