import subprocess
import sys


def run_cmd(args):
    proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = proc.communicate()
    return proc.returncode, out, err


def test_cli_input_missing_file_exits_nonzero(tmp_path):
    missing = tmp_path / "does_not_exist.json"
    code, out, err = run_cmd([sys.executable, "-m", "geodepoly.cli", "--input", str(missing), "--json"])
    assert code != 0



