import argparse
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--degree", type=int, default=16)
    ap.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    args = ap.parse_args()

    import torch
    from typing import cast
    from geodepoly.ai import root_solve_torch
    from geodepoly.batched import batched_poly_eval, batched_newton_step

    B, D = args.batch, args.degree
    torch.manual_seed(0)
    coeffs = torch.randn(B, D + 1, dtype=torch.cdouble, requires_grad=True)
    if args.device == "cuda" and torch.cuda.is_available():
        coeffs = coeffs.cuda()
    elif args.device == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        coeffs = coeffs.to("mps")

    # MPS lacks complex support: run a proxy workload when using MPS
    if args.device == "mps" and coeffs.device.type == "mps":
        creal = torch.randn(B, D + 1, dtype=torch.float32, device=coeffs.device, requires_grad=True)
        xs = torch.randn(B, 8, dtype=torch.float32, device=coeffs.device)
        # warmup
        _ = batched_poly_eval(creal, xs, backend="torch")
        import torch.mps as mps
        mps.synchronize()
        # forward timing (poly eval + 5 Newton steps)
        t0 = time.perf_counter()
        y = batched_poly_eval(creal, xs, backend="torch")
        for _ in range(5):
            xs = batched_newton_step(creal, xs, backend="torch")
        mps.synchronize()
        t_fwd = time.perf_counter() - t0
        # backward
        loss = (y ** 2).mean()
        t0 = time.perf_counter()
        loss.backward()  # type: ignore[attr-defined]
        mps.synchronize()
        t_bwd = time.perf_counter() - t0
        print(f"Torch MPS proxy (poly eval/newton): B={B} D={D} device=mps forward={t_fwd:.4f}s backward={t_bwd:.4f}s")
        return
    # warmup forward (CPU/CUDA complex path)
    roots_t = cast(torch.Tensor, root_solve_torch(coeffs))
    if coeffs.is_cuda:
        torch.cuda.synchronize()

    # time forward
    t0 = time.perf_counter()
    roots_t = cast(torch.Tensor, root_solve_torch(coeffs))
    if coeffs.is_cuda:
        torch.cuda.synchronize()
    if coeffs.device.type == "mps":
        import torch.mps as mps
        mps.synchronize()
    t_fwd = time.perf_counter() - t0

    # time backward on simple loss
    loss = (roots_t.real.clamp_min(0) ** 2).mean()
    t0 = time.perf_counter()
    loss.backward()  # type: ignore[attr-defined]
    if coeffs.is_cuda:
        torch.cuda.synchronize()
    t_bwd = time.perf_counter() - t0

    print(f"Torch RootLayer: B={B} D={D} device={args.device} forward={t_fwd:.4f}s backward={t_bwd:.4f}s")


if __name__ == "__main__":
    main()


