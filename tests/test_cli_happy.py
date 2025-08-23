import json
import subprocess
import sys
from pathlib import Path


def test_cli_coeffs_output(tmp_path: Path):
    outp = tmp_path / "roots.json"
    code = subprocess.call([
        sys.executable,
        "-m",
        "geodepoly.cli",
        "--coeffs",
        "[-6,11,-6,1]",
        "--output",
        str(outp),
        "--json",
    ])
    assert code == 0
    obj = json.loads(outp.read_text())
    assert isinstance(obj, list) and len(obj) == 3



