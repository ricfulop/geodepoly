# AI Overview

This section summarizes how to use GeodePoly in AI workflows.

- Differentiable RootLayer for PyTorch and JAX
- Root-space losses (pole placement, spectral radius, root-set matching)
- Examples for control and spectral fitting

## Install

```bash
pip install geodepoly[ai-torch]
# or
pip install geodepoly[ai-jax]
```

## Differentiable root solving (Torch)

```python
import torch
from geodepoly.ai import root_solve_torch

coeffs = torch.randn(8, 5, dtype=torch.cdouble, requires_grad=True)
roots  = root_solve_torch(coeffs)
loss   = (roots.real.clamp_min(0)**2).mean()
loss.backward()
```

## Losses

- `spectral_radius_loss(roots, target)` — penalizes |r| > target.
- `pole_placement_loss(roots, half_plane="left", margin=0.0)` — pushes poles to desired half-plane.
- `root_set_loss(roots_pred, roots_true, match="sorted")` — compares predicted vs target roots.

Example
```python
from geodepoly.ai.losses import spectral_radius_loss, pole_placement_loss
loss = spectral_radius_loss(roots, target=1.0) + pole_placement_loss(roots, half_plane="left", margin=0.1)
```

## Use cases

See the companion page for six practical patterns.
