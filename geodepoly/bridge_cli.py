import sys
import json
from typing import List, Any

from .solver import solve_all


def main() -> int:
    payload = sys.stdin.read()
    try:
        obj = json.loads(payload)
    except Exception as e:
        print(json.dumps({"error": f"invalid JSON: {e}"}))
        return 1

    coeffs = obj.get("coeffs")
    if not isinstance(coeffs, list):
        print(json.dumps({"error": "missing or invalid 'coeffs' (expect list)"}))
        return 1

    kwargs: dict[str, Any] = obj.get("kwargs", {})
    try:
        roots = solve_all(coeffs, **kwargs)
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        return 1

    def encode_complex(z: complex) -> List[float]:
        zc = complex(z)
        return [zc.real, zc.imag]

    out = {"roots": [encode_complex(z) for z in roots]}
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


