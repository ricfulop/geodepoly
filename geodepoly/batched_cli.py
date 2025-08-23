#!/usr/bin/env python3
"""
GPU batched CLI (Torch):
stdin JSON (v1): { "schemaVersion": 1, "coeffs_batch": [[...], ...], "device": "cuda"|"cpu" }
stdout JSON: { "schemaVersion": 1, "roots_batch": [[[re,im], ...], ...] }
"""
import sys, json


def main() -> int:
    try:
        import torch  # type: ignore[import-not-found]
    except Exception as e:
        print(json.dumps({"schemaVersion": 1, "error": "torch not installed"}))
        return 1
    from .batched import torch_batched_roots

    payload = sys.stdin.read()
    try:
        obj = json.loads(payload)
    except Exception as e:
        print(json.dumps({"schemaVersion": 1, "error": f"invalid JSON: {e}"}))
        return 1

    version = obj.get("schemaVersion", 1)
    if version != 1:
        print(json.dumps({"schemaVersion": 1, "error": "Unsupported schemaVersion; expected 1"}))
        return 1
    coeffs_batch = obj.get("coeffs_batch")
    if not isinstance(coeffs_batch, list) or not all(isinstance(row, list) for row in coeffs_batch):
        print(json.dumps({"schemaVersion": 1, "error": "missing or invalid 'coeffs_batch'"}))
        return 1
    device = obj.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    try:
        a = torch.tensor([[complex(x) for x in row] for row in coeffs_batch], dtype=torch.complex64, device=device)
        z = torch_batched_roots(a, iters=int(obj.get("iters", 60)), damping=float(obj.get("damping", 0.8)))
        roots_batch = z.detach().cpu().numpy()
        out = {
            "schemaVersion": 1,
            "roots_batch": [
                [[complex(v).real, complex(v).imag] for v in row] for row in roots_batch
            ],
        }
        print(json.dumps(out))
        return 0
    except Exception as e:
        print(json.dumps({"schemaVersion": 1, "error": str(e)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())



