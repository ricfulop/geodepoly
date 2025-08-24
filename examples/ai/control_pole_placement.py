import torch
from geodepoly.ai import root_solve_torch
from geodepoly.ai.losses import pole_placement_loss


def main():
    torch.manual_seed(0)
    B, N = 16, 4
    coeffs = torch.randn(B, N + 1, dtype=torch.cdouble, requires_grad=True)
    opt = torch.optim.Adam([coeffs], lr=1e-2)
    for step in range(50):
        roots = root_solve_torch(coeffs)
        loss = pole_placement_loss(roots, half_plane="left", margin=0.1)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 10 == 0:
            print(f"step={step} loss={loss.item():.6f}")


if __name__ == "__main__":
    main()
