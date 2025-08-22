import argparse, csv
import numpy as np
import matplotlib.pyplot as plt


def load_agg(path):
    rows = []
    with open(path, "r", newline="") as f:
        rd = csv.DictReader(f)
        rows = list(rd)
    # group by method
    methods = sorted(set(r["method"] for r in rows))
    degrees = sorted(set(int(r["degree"]) for r in rows))
    data = {m: {d: None for d in degrees} for m in methods}
    for r in rows:
        d = int(r["degree"]); m = r["method"]
        data[m][d] = {
            "time_mean": float(r["time_mean"]),
            "time_median": float(r["time_median"]),
            "res_median": float(r["res_median"])
        }
    return degrees, methods, data


def plot(degrees, methods, data, out_dir):
    # Time (median) vs degree
    plt.figure()
    for m in methods:
        y = [data[m][d]["time_median"] if data[m][d] else np.nan for d in degrees]
        plt.plot(degrees, y, marker="o", label=m)
    plt.xlabel("Degree"); plt.ylabel("Time (s) median"); plt.title("Time vs Degree"); plt.legend();
    plt.tight_layout(); plt.savefig(f"{out_dir}/time_vs_degree.png")

    # Residual (median, log) vs degree
    plt.figure()
    for m in methods:
        y = [data[m][d]["res_median"] if data[m][d] else np.nan for d in degrees]
        plt.semilogy(degrees, y, marker="o", label=m)
    plt.xlabel("Degree"); plt.ylabel("Max residual (median)"); plt.title("Residual vs Degree"); plt.legend();
    plt.tight_layout(); plt.savefig(f"{out_dir}/residual_vs_degree.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="Aggregate CSV from bench_compare --agg_out")
    ap.add_argument("--out", dest="out", required=True, help="Output directory for PNGs")
    args = ap.parse_args()
    degrees, methods, data = load_agg(args.inp)
    plot(degrees, methods, data, args.out)
    print("Saved plots to", args.out)


if __name__ == "__main__":
    main()


