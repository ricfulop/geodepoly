import torch
from geodepoly.ai import root_solve_torch
from geodepoly.ai.losses import root_set_loss


def main():
    torch.manual_seed(1)
    B, N = 8, 3
    coeffs = torch.randn(B, N + 1, dtype=torch.cdouble, requires_grad=True)
    # fix a target set of roots per batch item
    target = torch.randn(B, N, dtype=torch.cdouble)
    opt = torch.optim.Adam([coeffs], lr=1e-2)
    for step in range(50):
        roots = root_solve_torch(coeffs)
        loss = root_set_loss(roots, target, match="sorted")
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 10 == 0:
            print(f"step={step} loss={loss.item():.6f}")


if __name__ == "__main__":
    main()
