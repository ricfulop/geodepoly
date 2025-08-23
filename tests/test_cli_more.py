import json
import subprocess
import sys
from pathlib import Path


def test_cli_file_input_with_kwargs_and_resum_none(tmp_path: Path):
    payload = {
        "schemaVersion": 1,
        "coeffs": [-6, 11, -6, 1],
        "kwargs": {"method": "hybrid", "resum": "none"},
    }
    inp = tmp_path / "in.json"
    outp = tmp_path / "out.json"
    inp.write_text(json.dumps(payload))
    code = subprocess.call([
        sys.executable,
        "-m",
        "geodepoly.cli",
        "--input",
        str(inp),
        "--output",
        str(outp),
        "--json",
    ])
    assert code == 0
    obj = json.loads(outp.read_text())
    assert isinstance(obj, list) and len(obj) == 3


def test_cli_coeffs_plain_output(tmp_path: Path):
    # When --json is omitted, roots are printed line-by-line
    proc = subprocess.Popen([
        sys.executable,
        "-m",
        "geodepoly.cli",
        "--coeffs",
        "[-6,11,-6,1]",
        "--resum",
        "none",
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = proc.communicate()
    assert proc.returncode == 0
    lines = [ln for ln in out.decode("utf-8").strip().splitlines() if ln.strip()]
    assert len(lines) == 3


def test_cli_coeffs_with_spaces_and_commas(tmp_path: Path):
    code = subprocess.call([
        sys.executable,
        "-m",
        "geodepoly.cli",
        "--coeffs",
        "-6, 11,  -6 , 1",
        "--json",
    ])
    assert code == 0


def test_cli_bad_resum_exits_nonzero():
    # argparse should reject invalid choice
    proc = subprocess.Popen([
        sys.executable,
        "-m",
        "geodepoly.cli",
        "--coeffs",
        "[-6,11,-6,1]",
        "--resum",
        "bogus",
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = proc.communicate()
    assert proc.returncode != 0


def test_cli_coeffs_with_semicolons(tmp_path: Path):
    code = subprocess.call([
        sys.executable,
        "-m",
        "geodepoly.cli",
        "--coeffs",
        "-6; 11; -6; 1",
        "--json",
    ])
    assert code == 0


