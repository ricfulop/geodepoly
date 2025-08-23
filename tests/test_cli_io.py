import json
import subprocess
import sys
from pathlib import Path


def run_cmd(args, input_bytes: bytes | None = None) -> tuple[int, bytes, bytes]:
    proc = subprocess.Popen(args, stdin=subprocess.PIPE if input_bytes is not None else None, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = proc.communicate(input=input_bytes)
    return proc.returncode, out, err


def test_bridge_schema_v1_roundtrip():
    payload = {
        "schemaVersion": 1,
        "coeffs": [-6, 11, -6, 1],
        "kwargs": {"method": "hybrid", "resum": "auto"},
    }
    code, out, err = run_cmd([sys.executable, "-m", "geodepoly.bridge_cli"], json.dumps(payload).encode("utf-8"))
    assert code == 0, err.decode()
    obj = json.loads(out.decode("utf-8"))
    assert obj.get("schemaVersion") == 1
    roots = obj.get("roots")
    assert isinstance(roots, list) and all(isinstance(r, list) and len(r) == 2 for r in roots)


def test_bridge_schema_invalid_version_errors():
    payload = {"schemaVersion": 999, "coeffs": [-6, 11, -6, 1]}
    code, out, err = run_cmd([sys.executable, "-m", "geodepoly.bridge_cli"], json.dumps(payload).encode("utf-8"))
    # Bridge returns error JSON with non-zero exit
    assert code != 0
    obj = json.loads(out.decode("utf-8"))
    assert "error" in obj


def test_cli_input_output(tmp_path: Path):
    payload = {
        "schemaVersion": 1,
        "coeffs": [-6, 11, -6, 1],
        "kwargs": {"method": "hybrid", "resum": "auto"},
    }
    inp = tmp_path / "in.json"
    outp = tmp_path / "out.json"
    inp.write_text(json.dumps(payload))

    code, out, err = run_cmd([sys.executable, "-m", "geodepoly.cli", "--input", str(inp), "--output", str(outp), "--json"])
    assert code == 0, err.decode()
    obj = json.loads(outp.read_text())
    assert isinstance(obj, list) and all(isinstance(r, list) and len(r) == 2 for r in obj)


