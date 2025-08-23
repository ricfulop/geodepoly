\
#!/usr/bin/env python3
"""
JSON CLI bridge: read {"coeffs":[...], "kwargs":{...}} from stdin; write {"roots":[...]} to stdout.
Usable from Mathematica/Maple via RunProcess/ssystem calls.
"""
import sys, json
from geodepoly.solver import solve_all

def main():
    payload = sys.stdin.read()
    obj = json.loads(payload)
    coeffs = obj["coeffs"]
    kwargs = obj.get("kwargs", {})
    roots = solve_all(coeffs, **kwargs)
    def encode_complex(z):
        return [z.real, z.imag]
    out = {"roots": [encode_complex(complex(z)) for z in roots]}
    print(json.dumps(out))

if __name__ == "__main__":
    main()
