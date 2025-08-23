import argparse, ast, cmath
import numpy as np


def parse_alpha(s: str) -> complex:
    try:
        return complex(s)
    except Exception:
        # fallback simple parser for "a+bj"
        return complex(ast.literal_eval(s.replace('j', 'j')))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    args = ap.parse_args()

    import pandas as pd

    df = pd.read_csv(args.inp)
    # naive baseline: linear regression on features [t2, t3] to predict Re/Im(alpha)
    # ensure float columns exist
    if "t2" not in df:
        df["t2"] = 0.0
    if "t3" not in df:
        df["t3"] = 0.0
    df["t2"] = df["t2"].astype(float)
    df["t3"] = df["t3"].astype(float)
    t2 = df["t2"].to_numpy()
    t3 = df["t3"].to_numpy()
    y = np.array([parse_alpha(s) for s in df["alpha"]])
    X = np.vstack([np.ones_like(t2), t2, t3]).T
    # Solve least squares for real and imaginary parts
    w_re, *_ = np.linalg.lstsq(X, y.real, rcond=None)
    w_im, *_ = np.linalg.lstsq(X, y.imag, rcond=None)
    preds = (X @ w_re) + 1j * (X @ w_im)
    err = np.mean(np.abs(preds - y))
    print(f"Baseline mean abs error: {err:.3e}")


if __name__ == "__main__":
    main()


