import json
import subprocess
import sys


def run_cmd(args, input_bytes: bytes | None = None):
    proc = subprocess.Popen(args, stdin=subprocess.PIPE if input_bytes is not None else None, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = proc.communicate(input=input_bytes)
    return proc.returncode, out, err


def test_bridge_bad_payload_errors():
    code, out, err = run_cmd([sys.executable, "-m", "geodepoly.bridge_cli"], b"not json")
    assert code != 0
    obj = json.loads(out.decode("utf-8"))
    assert "error" in obj

    bad = {"schemaVersion": 1, "coeffs": "oops"}
    code, out, err = run_cmd([sys.executable, "-m", "geodepoly.bridge_cli"], json.dumps(bad).encode("utf-8"))
    assert code != 0
    obj = json.loads(out.decode("utf-8"))
    assert "error" in obj



