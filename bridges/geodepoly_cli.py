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
    print(json.dumps({"roots":[complex(z).real if abs(z.imag)<1e-15 else complex(z) for z in roots]}, default=lambda o: [o.real, o.imag] if isinstance(o, complex) else o))

if __name__ == "__main__":
    main()
