import argparse, time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--degree", type=int, default=16)
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = ap.parse_args()

    import torch
    from geodepoly.ai import root_solve_torch

    B, D = args.batch, args.degree
    torch.manual_seed(0)
    coeffs = torch.randn(B, D + 1, dtype=torch.cdouble, requires_grad=True)
    if args.device == "cuda" and torch.cuda.is_available():
        coeffs = coeffs.cuda()

    # warmup forward
    roots = root_solve_torch(coeffs)
    if coeffs.is_cuda:
        torch.cuda.synchronize()

    # time forward
    t0 = time.perf_counter()
    roots = root_solve_torch(coeffs)
    if coeffs.is_cuda:
        torch.cuda.synchronize()
    t_fwd = time.perf_counter() - t0

    # time backward on simple loss
    loss = (roots.real.clamp_min(0) ** 2).mean()
    t0 = time.perf_counter()
    loss.backward()
    if coeffs.is_cuda:
        torch.cuda.synchronize()
    t_bwd = time.perf_counter() - t0

    print(f"Torch RootLayer: B={B} D={D} device={args.device} forward={t_fwd:.4f}s backward={t_bwd:.4f}s")


if __name__ == "__main__":
    main()


