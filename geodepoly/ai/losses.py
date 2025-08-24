from __future__ import annotations


def _ensure_torch():
    try:
        import torch  # type: ignore

        return torch
    except Exception as exc:  # pragma: no cover
        raise ImportError("PyTorch required for ai.losses helpers") from exc


def spectral_radius_loss(roots, target: float):
    """Penalize |r| exceeding target radius.

    roots: torch.Tensor (.., N) complex
    target: float radius
    """
    torch = _ensure_torch()
    r = torch.abs(roots)
    return torch.relu(r - float(target)).pow(2).mean()


def pole_placement_loss(roots, half_plane: str = "left", margin: float = 0.0):
    """Penalize roots outside desired half-plane with margin.

    half_plane: "left" (Re(r) <= -margin) or "right" (Re(r) >= margin)
    """
    torch = _ensure_torch()
    re = roots.real
    if half_plane == "left":
        return torch.relu(re + float(margin)).pow(2).mean()
    if half_plane == "right":
        return torch.relu(float(margin) - re).pow(2).mean()
    raise ValueError("half_plane must be 'left' or 'right'")


def root_set_loss(roots_pred, roots_true, match: str = "sorted"):
    """Compare predicted and true root sets.

    match: "sorted" compares sorted by real, then imag. For robust matching,
    use Hungarian assignment (future option).
    """
    torch = _ensure_torch()
    if match == "sorted":
        def _sort_roots(r):
            key = torch.stack([r.real, r.imag], dim=-1)
            idx = torch.argsort(key[..., 0])
            return r.index_select(-1, idx)
        rp = _sort_roots(roots_pred)
        rt = _sort_roots(roots_true)
        return (rp - rt).abs().pow(2).mean()
    if match == "hungarian":
        # Solve linear sum assignment on pairwise |ri - tj|^2
        # Works on CPU/GPU tensors; loops over batch dimension if present
        def loss_row(rp_row, rt_row):
            n = rp_row.shape[-1]
            # cost matrix
            diff = rp_row.unsqueeze(-1) - rt_row.unsqueeze(-2)  # (n, n)
            C = (diff.real**2 + diff.imag**2)
            # naive Hungarian via CPU SciPy if available; otherwise greedy fallback
            try:
                import numpy as np  # type: ignore
                from scipy.optimize import linear_sum_assignment  # type: ignore
                C_np = C.detach().cpu().numpy()
                ri, ci = linear_sum_assignment(C_np)
                P = torch.zeros_like(C)
                P[ri, ci] = 1.0
            except Exception:
                # Greedy: repeatedly pick min remaining
                P = torch.zeros_like(C)
                C_work = C.clone()
                for _ in range(n):
                    idx = torch.argmin(C_work).item()
                    i = idx // n
                    j = idx % n
                    P[i, j] = 1.0
                    C_work[i, :] = float("inf")
                    C_work[:, j] = float("inf")
            return (C * P).sum() / n

        # support batched
        if roots_pred.dim() == 1:
            return loss_row(roots_pred, roots_true)
        else:
            return torch.stack([loss_row(rp, rt) for rp, rt in zip(roots_pred, roots_true)]).mean()
    if match == "diffsort":
        # Soft sorting via temperature-scaled pairwise weights on real parts
        # Note: this is a simple differentiable surrogate, not a true sort.
        torch = _ensure_torch()
        def soft_order_weights(x, tau: float = 0.1):
            # x: (N,) real scores; weights W_{i,j} ~ exp(-|x_i - x_j|/tau)
            xr = x.real if torch.is_complex(x) else x
            diff = torch.abs(xr[:, None] - xr[None, :])
            W = torch.exp(-diff / tau)
            W = W / (W.sum(dim=1, keepdim=True) + 1e-12)
            return W
        def soft_align(a, b):
            Wa = soft_order_weights(a.real)
            Wb = soft_order_weights(b.real)
            a_soft = Wa @ a
            b_soft = Wb @ b
            return (a_soft - b_soft).abs().pow(2).mean()
        if roots_pred.dim() == 1:
            return soft_align(roots_pred, roots_true)
        return torch.stack([soft_align(rp, rt) for rp, rt in zip(roots_pred, roots_true)]).mean()
    raise ValueError("match must be 'sorted', 'hungarian', or 'diffsort'")


