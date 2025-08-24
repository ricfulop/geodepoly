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
    if match != "sorted":
        raise NotImplementedError("Only 'sorted' matching implemented in MVP")
    def _sort_roots(r):
        key = torch.stack([r.real, r.imag], dim=-1)
        idx = torch.lexsort((key[..., 1], key[..., 0])) if hasattr(torch, 'lexsort') else torch.argsort(key[..., 0])
        return r.index_select(-1, idx)
    rp = _sort_roots(roots_pred)
    rt = _sort_roots(roots_true)
    return (rp - rt).abs().pow(2).mean()


