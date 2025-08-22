import os, csv, tempfile, subprocess, sys


def test_bench_smoke_creates_csv():
    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "bench.csv")
        cmd = [sys.executable, "scripts/bench_compare.py", "--deg", "3", "--trials", "1", "--out", out, "--methods", "hybrid", "--resum", "auto"]
        subprocess.run(cmd, check=True)
        assert os.path.exists(out)
        with open(out, "r", newline="") as f:
            rd = csv.DictReader(f)
            cols = rd.fieldnames
        assert set(["trial","degree","method","time_s","max_residual"]).issubset(set(cols))

