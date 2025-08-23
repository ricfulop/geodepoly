#!/usr/bin/env python3
import argparse
import time


def make_poly(deg: int):
    # (x-1)(x-2)...(x-deg)
    import numpy as np

    coeffs = np.array([1.0], dtype=complex)
    for k in range(1, deg + 1):
        coeffs = np.convolve(coeffs, np.array([-k, 1.0], dtype=complex))
    return coeffs  # low->high


def main():
    ap = argparse.ArgumentParser(description="Bench vectorized Aberth on GPU vs CPU")
    ap.add_argument("--deg", type=int, default=32)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--backend", type=str, default="torch", choices=["torch"])  # future: jax
    args = ap.parse_args()

    if args.backend == "torch":
        import torch
        from geodepoly.batched import torch_aberth_step

        device_cpu = torch.device("cpu")
        device_gpu = torch.device("cuda") if torch.cuda.is_available() else None

        coeffs_np = make_poly(args.deg)
        coeffs = torch.tensor(coeffs_np, dtype=torch.complex64)
        # init roots on a circle
        k = torch.arange(args.deg, dtype=torch.float32)
        theta = 2 * torch.pi * (k / args.deg)
        roots0 = 1.5 * torch.exp(1j * theta)

        # CPU
        z = roots0.to(device_cpu)
        c = coeffs.to(device_cpu)
        t0 = time.perf_counter()
        for _ in range(args.iters):
            z = torch_aberth_step(c, z, damping=0.8)
        dt_cpu = time.perf_counter() - t0

        # GPU
        if device_gpu is not None:
            z2 = roots0.to(device_gpu)
            c2 = coeffs.to(device_gpu)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            for _ in range(args.iters):
                z2 = torch_aberth_step(c2, z2, damping=0.8)
            torch.cuda.synchronize()
            dt_gpu = time.perf_counter() - t1
        else:
            dt_gpu = None

        print(f"deg={args.deg} iters={args.iters} CPU={dt_cpu:.3f}s GPU={dt_gpu:.3f}s")


if __name__ == "__main__":
    main()


