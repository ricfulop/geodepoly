import os, csv, tempfile, sys, subprocess


def test_generate_slices_smoke():
    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "slices.csv")
        cmd = [sys.executable, "bench/generate_slices.py", "--degrees", "3,5", "--trials", "1", "--out", out]
        subprocess.run(cmd, check=True)
        assert os.path.exists(out)
        with open(out, "r", newline="") as f:
            rd = csv.DictReader(f)
            cols = rd.fieldnames
        assert set(["slice","degree","trial","alpha"]).issubset(set(cols))

