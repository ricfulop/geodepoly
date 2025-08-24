# GeodePoly — AI Feature Spec & Roadmap

**Owner:** Ric Fulop (@ricfulop)  
**Repo:** https://github.com/ricfulop/geodepoly  
**Scope:** Differentiable polynomial solving, AI losses, datasets, and performance paths.  
**Status:** v0.1.0 (MVP solver shipped). This document defines the AI extensions.

---

## 1. Goals & Non‑Goals

### 1.1 Goals
- Provide **differentiable polynomial solving** for PyTorch and JAX (RootLayer).
- Add **root‑space losses** and training examples (control & spectral fitting).
- Release **GeodeBench v0** (symmetry generalization tasks) with baselines.
- Enable **eigenvalue workflows** via characteristic polynomials.
- Keep all AI dependencies **optional**; core solver remains lightweight.

### 1.2 Non‑Goals (for now)
- Full GPU kernelization of series/Geode layers (tracked separately).
- Symbolic exact algebra; our AI path is numeric/differentiable.
- Massive‑scale training infrastructure—examples are small and reproducible.

---

## 2. Architecture Overview

```
geodepoly/
  ai/
    __init__.py
    rootlayer_torch.py     # PyTorch autograd Function
    rootlayer_jax.py       # JAX custom_vjp
    losses.py              # spectral/pole placement/root-set losses
  eigs.py                  # Faddeev–LeVerrier + solve_poly
examples/
  ai/
    torch_rootlayer_demo.py
    jax_rootlayer_demo.py
    control_pole_placement.py
    spectral_matching.py
bench/                     # (optional repo) geodebench/ for datasets + baselines
scripts/
  bench_compare.py
```

- **Forward**: `solve_poly(coeffs, method="hybrid", resum="pade")` → roots.  
- **Backward**: analytic Jacobian for simple roots  
  \(∂r_i/∂a_k = − r_i^k / p'(r_i)\) with numerical guards near multiple roots.
- **Batched**: `[B, N+1] → [B, N]`; vectorized Horner for p′(r).
- **Optional deps**: extras `[ai-torch]`, `[ai-jax]`, `[ai-all]` in `pyproject.toml`.

---

## 3. Public APIs

### 3.1 RootLayer (PyTorch)
```python
# geodepoly/ai/rootlayer_torch.py
class RootLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, coeffs: torch.Tensor, *, method="hybrid", resum="pade"):
        # coeffs: [B, N+1] (low→high)
        # returns: roots [B, N] (complex dtype supported via torch.view_as_real if needed)
        ...

    @staticmethod
    def backward(ctx, grad_roots: torch.Tensor):
        # analytic simple-root Jacobian, masked near |p'(r)|<eps
        ...
```

Helper (user‑facing) wrapper:
```python
def root_solve_torch(coeffs, method="hybrid", resum="pade"):
    return RootLayer.apply(coeffs, method=method, resum=resum)
```

### 3.2 RootLayer (JAX)
```python
# geodepoly/ai/rootlayer_jax.py
@jax.custom_vjp
def root_solve_jax(coeffs, method="hybrid", resum="pade"):
    ...  # returns roots [B, N]

def _fwd(coeffs, method="hybrid", resum="pade"):
    roots = host_or_purejax_solve(coeffs, method, resum)  # callable inside jit
    return roots, (coeffs, roots)

def _bwd(res, g):
    coeffs, roots = res
    # build J^T g using dr/da_k = - r^k / p'(r)
    ...
root_solve_jax.defvjp(_fwd, _bwd)
```

### 3.3 Losses (shared)
```python
# geodepoly/ai/losses.py
def spectral_radius_loss(roots, target): ...
def pole_placement_loss(roots, half_plane="left", margin=0.0): ...
def root_set_loss(roots_pred, roots_true, match="sorted"): ...
```

---

## 4. Numerical Stability & Edge Cases

- **Simple‑root assumption in backward.** If \(|p'(r_i)|<ε\), zero or damp gradient for that root and log a warning.  
- **Multiple roots**: optional estimate  
  \( \,\hat{m} = \mathrm{round}\,\Re[p\,p''/p'^2] \) (clamped) → use damped updates in training; add curriculum examples.  
- **Scaling**: encourage pre‑scaling of coefficients (monic normalization already in core).  
- **Resummation** in forward: `resum="pade"` default; later `resum="auto"` selection.

---

## 5. Examples

### 5.1 Torch — Pole Placement
- Learn polynomial coeffs so all roots are in left‑half plane with margin.  
- Script: `examples/ai/control_pole_placement.py`  
- Loss: `pole_placement_loss` + small L2 on coeffs.  
- Plots: root trajectories over epochs.

### 5.2 JAX — Spectral Matching
- Fit coefficients to match a target root set.  
- Script: `examples/ai/spectral_matching.py`  
- Loss: `root_set_loss(sorted)`.

---

## 6. GeodeBench v0 (separate repo or bench/)

### Tasks
- **T1 Coefficient Prediction**: predict masked coefficients from Geode‑layered contexts.  
- **T2 Geode Factor Recovery**: map \(S\) → \(G\).  
- **T3 Invariance**: classify/evaluate invariance under shift/scale.

### Splits
- Easy (Catalan/Fuss), Medium (Bi‑Tri), Hard (mixed).

### Metrics
- Exact‑match %, symmetric‑MAE (up to relabeling), invariance accuracy, OOD gap.

---

## 7. Eigenvalue Path

- `solve_eigs(A)`: get coefficients via **Faddeev–LeVerrier**, then `solve_poly`.  
- Gershgorin shifts, optional block deflation; polish with Halley.  
- Examples: small dense matrices (n ≤ 20).

---

## 8. Performance Plan

- **Batching**: vmap/scan (JAX), torch.vmap or manual batching (Torch).  
- **Hot paths**: vectorized Horner for p′; reuse buffers; avoid Python loops inside jit.  
- **Future**: Geode/face‑layer tiling kernels (JAX/XLA, CUDA) — separate spec.

---

## 9. Testing & CI

- Unit tests for forward accuracy (residuals) and backward gradients (finite‑diff checks).  
- Torch: `tests/test_ai_torch.py`; JAX: `tests/test_ai_jax.py`.  
- Optional CI matrix with `[ai-torch]` and `[ai-jax]` jobs; keep default job lean.

---

## 10. Packaging & Optional Deps

`pyproject.toml`:
```toml
[project.optional-dependencies]
ai-torch = ["torch>=2.2", "numpy"]
ai-jax   = ["jax>=0.4.26", "jaxlib", "numpy"]
ai-all   = ["geodepoly[ai-torch,ai-jax]", "matplotlib", "optax", "flax"]
```

Docs: add an “AI Quickstart” section with pip commands and example runs.

---

## 11. Roadmap (Sprints)

### Sprint 0 — Scaffolding (P0)
- Create `geodepoly/ai/*`, examples folder, extras in pyproject.
- **DoD:** `pip install -e .[ai-torch]` works; readme AI section present.

### Sprint 1 — RootLayer (P0)
- Torch RootLayer (batched), masked backward near small \(|p'|\).  
- JAX RootLayer with `custom_vjp`, `vmap`, `jit`.  
- **DoD:** gradient tests pass (rel‑err < 1e‑3).

### Sprint 2 — Losses & Examples (P0)
- Implement `spectral_radius_loss`, `pole_placement_loss`, `root_set_loss`.  
- Two runnable examples (Torch & JAX).  
- **DoD:** scripts train and converge; screenshots in docs.

### Sprint 3 — GeodeBench v0 (P1)
- Spec, generator, small baselines.  
- **DoD:** dataset artifacts + baseline table committed.

### Sprint 4 — Eigenvalue Path (P1)
- `solve_eigs(A)` + example; unit tests vs NumPy eigvals.  
- **DoD:** residual ≤ 1e−8 on n ≤ 20 random matrices.

### Sprint 5 — Performance & Polish (P2)
- Batch perf passes; `resum="auto"` if ready; docs/Colab.  
- **DoD:** ≥10× speedup on batch=1k (deg 5–8) vs naive loop.

---

## 12. Definition of Done (global)

- Tests green locally and in CI.  
- Residuals and gradient checks meet thresholds.  
- Examples run from a fresh venv with listed extras.  
- Docs updated (README & AI section).  
- Version bump + changelog; tag release (TestPyPI→PyPI optional).
