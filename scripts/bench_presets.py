import subprocess, sys, os


PRESETS = {
    "quick": {
        "cmd": [
            [sys.executable, "scripts/bench_compare.py", "--degrees", "3,5,8", "--methods", "hybrid,aberth,dk", "--trials", "5", "--out", "docs/assets/bench.csv", "--agg_out", "docs/assets/bench_agg.csv", "--resum", "auto"],
            [sys.executable, "scripts/plot_bench.py", "--in", "docs/assets/bench_agg.csv", "--out", "docs/assets"],
            [sys.executable, "scripts/bench_newton_vs_hybrid.py", "--degrees", "3,5,8,12,16,20", "--trials", "5", "--out", "docs/assets/newton_vs_hybrid_tuned.csv"],
            [sys.executable, "scripts/bench_edge_cases.py", "--out", "docs/assets/edge_cases.csv"],
        ]
    }
}


def main():
    preset = sys.argv[1] if len(sys.argv) > 1 else "quick"
    spec = PRESETS.get(preset)
    if not spec:
        print("Unknown preset:", preset)
        sys.exit(1)
    for cmd in spec["cmd"]:
        print("Running:", " ".join(map(str, cmd)))
        subprocess.run(cmd, check=True)
    print("Done.")


if __name__ == "__main__":
    main()



