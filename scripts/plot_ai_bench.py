import argparse
import csv
from pathlib import Path


def load_rows(csv_path):
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            # normalize
            r["Forward"] = float(r["Forward"])
            r["Backward"] = float(r["Backward"])
            rows.append(r)
    return rows


def plot(rows, out_dir):
    import matplotlib.pyplot as plt  # lazy import

    labels = [f"{r['Backend']}\n{r['Device']}\nB={r['Batch']} D={r['Degree']}" for r in rows]
    fwd = [r["Forward"] for r in rows]
    bwd = [r["Backward"] for r in rows]

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # Forward plot
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(labels, fwd)
    ax.set_title("AI Benchmarks — Forward Time (s)")
    ax.set_ylabel("seconds")
    ax.set_xticklabels(labels, rotation=30, ha="right")
    fig.tight_layout()
    fwd_path = str(Path(out_dir) / "ai_bench_forward.png")
    fig.savefig(fwd_path, dpi=150)
    plt.close(fig)

    # Backward plot
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(labels, bwd)
    ax.set_title("AI Benchmarks — Backward Time (s)")
    ax.set_ylabel("seconds")
    ax.set_xticklabels(labels, rotation=30, ha="right")
    fig.tight_layout()
    bwd_path = str(Path(out_dir) / "ai_bench_backward.png")
    fig.savefig(bwd_path, dpi=150)
    plt.close(fig)

    print(f"Saved plots:\n- {fwd_path}\n- {bwd_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Input CSV with columns Backend,Device,Batch,Degree,Forward,Backward")
    ap.add_argument("--out", required=True, help="Output directory for images")
    args = ap.parse_args()

    rows = load_rows(args.csv)
    plot(rows, args.out)


if __name__ == "__main__":
    main()


