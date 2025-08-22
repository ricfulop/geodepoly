import json, subprocess, sys


def main():
    payload = {
        "coeffs": [-6, 11, -6, 1],  # (x-1)(x-2)(x-3) = 0
        "kwargs": {"method": "hybrid", "resum": "auto"}
    }
    s = json.dumps(payload)
    proc = subprocess.run([sys.executable, "bridges/geodepoly_cli.py"], input=s.encode(), capture_output=True)
    if proc.returncode != 0:
        print("bridge failed:", proc.stderr.decode())
        sys.exit(1)
    out = json.loads(proc.stdout.decode())
    print("bridge output:", out)


if __name__ == "__main__":
    main()


