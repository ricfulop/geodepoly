# GeodePoly — Next‑Gen Polynomial Library (SPEC)

> A Python library for exact/symbolic + high‑precision numeric polynomial algebra, powered by Hyper‑Catalan series and the Geode factorization. Core math follows Wildberger & Rubine (2025) Theorems 3–12 (series roots, Bi‑Tri/Geode arrays, layerings). 【5†source】【5†source】

## 0) Goals
Status (shipped highlights)

- CLI commands: `geodepoly-solve`, `geodepoly-bridge` (JSON schema v1)
- Core solvers: hybrid/aberth/dk/numpy, eigen solver, series bootstrap
- Batched + GPU: vectorized Newton; Torch Aberth step and multi-step; JAX step; solver `method="torch-aberth"`
- Packaging: extras `[sympy] [torch] [jax] [numba]`, `py.typed`
- CI: coverage ≥90% enforced; extras install matrix; docs auto-deploy
- Docs: Performance page (Numba), GPU spec/benches, Colab quickstart and badges


- Be the **top Python library for polynomials**: robust, fast, well‑tested, pleasant API.
- Unify **classical numerics** (Newton, Durand–Kerner, Aberth, companion‑QR) with **new series methods** (Hyper‑Catalan soft formula, Geode factorization) for root finding.
- Offer great **symbolic tools** (exact rationals, algebraic numbers, resultants, Groebner-lite for univariate tasks) with optional SymPy interop.
- Ship with **industrial‑grade** docs, examples, benchmarks, and CI.

---

## 1) User‑Facing Features

### 1.1 Root Finding

**A. Series Roots (Hyper‑Catalan / Soft Polynomial Formula)**
- `series_root(c: Sequence[NumberLike], order: int, variant: Literal["raw","factorized"]="raw") -> FormalSeries`
- `series_bootstrap(f, x0: complex, series_order: int, rounds: int, damping: float=1.0) -> complex`

**B. Classical Methods**
- `newton`, `durand_kerner`, `aberth`, `companion_roots`

**C. Hybrid Strategies**
- `series_seed_then_newton`, `multi_start_hybrid`

### 1.2 Polynomial Types & Arithmetic

- `Polynomial` class: dense/sparse, rational/complex/mpmath.
- Operations, evaluation, resultants, factorization.

### 1.3 Hyper‑Catalan / Geode Combinatorics

- `hyper_catalan`, `bi_tri_array`, `geode_array`
- Layerings: vertex, edge, face
- `geode_factorize`

### 1.4 Formal Series Objects

- `FormalSeries` class with algebra, truncation, substitution, pretty‑print.

### 1.5 Utilities & Interop

- SymPy/numpy/mpmath interop
- IO and plotting helpers

---

## 2) API Sketch

```python
from geodepoly import Polynomial, series_root, series_bootstrap
from geodepoly.series import hyper_catalan, geode_factorize
from geodepoly.numeric import newton, aberth, companion_roots
from geodepoly.formal import FormalSeries
```

---

## 3) Algorithms & Math Notes

- Implement Theorems 3–7 (soft polynomial formula) 【5†source】
- Geode factorization S-1 = S1*G 【5†source】
- Bootstrap via shifts 【5†source】
- Bi‑Tri via factorial quotient 【5†source】

---

## 4) Performance & Engineering

- Memoization of Cm, factorial powers
- Integer first, optional numba
- Sparse dicts for monomials

---

## 5) Project Structure

```
geodepoly/
  __init__.py
  poly.py
  series.py
  formal.py
  numeric.py
  utils.py
  interop.py
  plotting.py
examples/
tests/
pyproject.toml
README.md
SPEC.md
```

---

## 6) Acceptance Tests

1. Catalan sanity
2. Bi‑Tri table matches
3. Geode factorization holds
4. Wallis cubic bootstrap within 1e-15
5. Bring radical terms match Eisenstein 【5†source】
6. Hybrid beats Aberth
7. Determinism

---

## 7) CLI

- `geodepoly solve`
- `geodepoly series`
- `geodepoly arrays`

---

## 8) Docs Plan

- mkdocs site with concepts, how‑to, theory 【5†source】

---

## 9) CI / Quality

- Lint, test, docs workflows
- Coverage >=90%

---

## 10) Roadmap

M1 Foundations → M6 Docs & API hardening

---

## 11) Implementation Notes for Cursor

- Pure Python baseline, numba optional
- Type m as tuple
- Truncation policy
- Cache factorial quotients

---

## 12) License & Attribution

- MIT License
- Cite Wildberger & Rubine (2025) 【5†source】

---

## 13) Stretch Goals

- GPU via JAX
- Algebraic numbers
- Convergence basins
