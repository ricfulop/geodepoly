import torch
from geodepoly.ai import root_solve_torch


def main():
    torch.manual_seed(0)
    B, N = 4, 3
    coeffs = torch.randn(B, N + 1, dtype=torch.cdouble, requires_grad=True)
    roots = root_solve_torch(coeffs, method="hybrid", resum="pade")
    loss = (roots.real.clamp_min(0) ** 2).mean()
    loss.backward()
    print("roots shape:", roots.shape)
    print("loss:", loss.item())
    print("grad shape:", coeffs.grad.shape)


if __name__ == "__main__":
    main()


