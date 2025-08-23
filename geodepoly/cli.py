import argparse
import json
import sys
from typing import List

from .solver import solve_all


def _parse_coeffs(arg_values: List[str] | None, arg_string: str | None) -> List[complex]:
    # Priority: explicit string via --coeffs, else positional numbers
    if arg_string:
        s = arg_string.strip()
        # Try JSON first
        try:
            obj = json.loads(s)
            if isinstance(obj, list):
                return [complex(x) for x in obj]
        except Exception:
            pass
        # Try comma/space separated numbers
        tokens = [t for part in s.split(";") for t in part.replace(",", " ").split()]
        return [complex(float(t)) for t in tokens]
    if arg_values:
        return [complex(float(t)) for t in arg_values]
    raise SystemExit("No coefficients provided. Use --coeffs '[a0,a1,...,aN]' or positional numbers.")


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="geodepoly-solve",
        description="Solve a univariate polynomial given coefficients a0..aN (low->high).",
    )
    ap.add_argument(
        "coeffs_vals",
        nargs="*",
        help="Positional coefficients a0 a1 ... aN (overridden by --coeffs)",
    )
    ap.add_argument(
        "-c",
        "--coeffs",
        type=str,
        default=None,
        help="Coefficients as JSON (e.g. '[1,0,-7,6]') or comma/space-separated string",
    )
    ap.add_argument("--method", type=str, default="hybrid", choices=["hybrid", "aberth", "dk", "numpy"], help="Solver method")
    ap.add_argument("--resum", type=str, default="auto", choices=["auto", "pade", "borel", "borel-pade", "none"], help="Resummation for series seed")
    ap.add_argument("--tol", type=float, default=1e-12, help="Target residual tolerance")
    ap.add_argument("--max-order", type=int, default=24, help="Max series order for seed")
    ap.add_argument("--boots", type=int, default=2, help="Bootstrap iterations for seed")
    ap.add_argument("--refine-steps", type=int, default=6, help="Final Halley/Newton refinement steps")
    ap.add_argument("--json", action="store_true", help="Print roots as JSON [[re,im],...] instead of plain text")
    ap.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    args = ap.parse_args(argv)

    coeffs = _parse_coeffs(args.coeffs_vals, args.coeffs)

    kwargs = {
        "method": args.method,
        "tol": args.tol,
        "max_order": args.max_order,
        "boots": args.boots,
        "refine_steps": args.refine_steps,
    }
    if args.resum != "none":
        kwargs["resum"] = args.resum

    roots = solve_all(coeffs, **kwargs)

    if args.json:
        def enc(z: complex):
            zc = complex(z)
            return [zc.real, zc.imag]

        print(json.dumps([enc(z) for z in roots]))
    else:
        for z in roots:
            print(z)

    return 0


if __name__ == "__main__":
    sys.exit(main())


