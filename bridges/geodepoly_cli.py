\
#!/usr/bin/env python3
"""
JSON CLI bridge (schema v1):
stdin:  { "schemaVersion": 1, "coeffs": [...], "kwargs": { ... } }
stdout: { "schemaVersion": 1, "roots": [[re,im], ...] }

Back-compat: if schemaVersion missing, treat as v1.
"""
import sys, json
from geodepoly.solver import solve_all


def _validate_payload(obj: dict):
    version = obj.get("schemaVersion", 1)
    if version != 1:
        raise ValueError("Unsupported schemaVersion; expected 1")
    if "coeffs" not in obj or not isinstance(obj["coeffs"], list):
        raise ValueError("Missing or invalid 'coeffs' (expect list)")
    coeffs = obj["coeffs"]
    kwargs = obj.get("kwargs", {})
    if not isinstance(kwargs, dict):
        raise ValueError("'kwargs' must be an object if provided")
    return version, coeffs, kwargs


def main():
    payload = sys.stdin.read()
    try:
        obj = json.loads(payload)
    except Exception as e:
        print(json.dumps({"error": f"invalid JSON: {e}"}))
        return 1
    try:
        version, coeffs, kwargs = _validate_payload(obj)
        roots = solve_all(coeffs, **kwargs)
        def encode_complex(z):
            zc = complex(z)
            return [zc.real, zc.imag]
        out = {"schemaVersion": version, "roots": [encode_complex(z) for z in roots]}
        print(json.dumps(out))
        return 0
    except Exception as e:
        print(json.dumps({"schemaVersion": 1, "error": str(e)}))
        return 1

if __name__ == "__main__":
    main()
