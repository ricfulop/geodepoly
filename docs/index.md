# geodepoly Docs

- **API**: `solve_poly`, `solve_all`, `solve_one`
- **Methods**: `hybrid`, `aberth`, `dk`, `numpy`
- **Resummation**: `None`, `pade`, `borel`, `borel-pade`
- **CAS**: SymPy plugin, JSON CLI for Mathematica/Maple

See the README for quickstart and `paper/GeodePoly_MVP.md` for a draft paper.

## Theory and Paper Mapping

- See `docs/paper_guide.md` for how the paper “A Hyper-Catalan Series Solution to Polynomial Equations, and the Geode” maps to the implementation.
- The module `geodepoly.hyper_catalan` provides utilities for the multivariate series `S[t2,t3,...]` described in the paper.

## Benchmarks & Plots

- Run: `python scripts/bench_compare.py --degrees 3,5,8,12 --methods hybrid,aberth,dk --trials 10 --out docs/assets/bench.csv --agg_out docs/assets/bench_agg.csv`
- Plot: `python scripts/plot_bench.py --in docs/assets/bench_agg.csv --out docs/assets`

## CAS Examples

- SymPy comparison: `python examples/sympy_vs_nroots.py`
- JSON bridge round-trip: `python examples/json_bridge_roundtrip.py`
  - Mathematica: `RunProcess[{"geodepoly-bridge"}, "StandardInput"->payloadJSON]`
  - Maple: `ssystem("geodepoly-bridge", payloadJSON)`

Schema (v1):

```json
{
  "schemaVersion": 1,
  "coeffs": [ -6, 11, -6, 1 ],
  "kwargs": { "method": "hybrid", "resum": "auto" }
}
```

## CLI Solver

- Solve from terminal:
  - `geodepoly-solve --coeffs "[-6,11,-6,1]" --method hybrid --resum auto --json`
  - Or file I/O (schema v1): `geodepoly-solve --input payload.json --output roots.json`

## AI Quickstart

- Optional install:
  - `pip install geodepoly[ai-torch]` (PyTorch) or `pip install geodepoly[ai-jax]` (JAX)
- Differentiable root solve (Torch):
  ```python
  import torch
  from geodepoly.ai import root_solve_torch

  coeffs = torch.randn(8, 5, dtype=torch.cdouble, requires_grad=True)
  roots  = root_solve_torch(coeffs)
  loss   = (roots.real.clamp_min(0)**2).mean()
  loss.backward()
  ```
- Losses: `spectral_radius_loss`, `pole_placement_loss`, `root_set_loss` in `geodepoly.ai.losses`.
