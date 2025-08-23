# TODO — Roadmap for the IDE

This is a concrete, testable backlog for geodepoly. Each task lists files to touch,
acceptance criteria (DoD), and how to run/verify locally.

---

## Sprint 1 — Core robustness & numerics (P0)

### 1) Adaptive Padé / Borel‑Padé selection
**Files:** `geodepoly/resummation.py`, `geodepoly/series_core.py`, `tests/test_resummation.py`, `geodepoly/scripts/bench_compare.py`  
**Goal:** Choose Padé orders automatically and fall back safely.  
**Tasks:**
- Add `eval_series_auto(g, t, max_order=None)` in `resummation.py`:
  - Try Padé `(m,n)` across a small near‑diagonal grid (e.g., `m≈n≈len(g)//2` down to 4).
  - Score: (i) stability proxy (e.g., denominator magnitude at `t`), (ii) leave‑last‑term residual, (iii) ratio test near `|t|≈1`.
  - If unstable, fall back to Borel‑Padé; then to plain series.
- Wire `resum="auto"` in `series_core.series_seed_step`.
- CLI: add `--resum auto` in `geodepoly/scripts/bench_compare.py`.

**DoD (acceptance):**
- Unit tests pass in `tests/test_resummation.py` with synthetic series where plain diverges at `|t|≈1`.
- On random degrees 5–12 (10 trials), `auto` median residual ≤ `pade` median residual.

---

### 2) Aberth–Ehrlich damping & clustering
**Files:** `geodepoly/finishers.py`, `tests/test_aberth.py`  
**Goal:** Improve convergence on clustered / near‑multiple roots.  
**Tasks:**
- Adaptive damping: update `z_i ← z_i − α_i Δ_i`, where `α_i ∈ (0,1]` chosen by:
  - Backtracking on residual decrease, or
  - `α_i = min(1, c / (1 + |∑_{j≠i} 1/(z_i−z_j)|))` with small `c` (e.g., 0.3–1.0).
- Minimal repulsion: if `|z_i − z_j| < ε` (ε small), jitter orthogonally by ~1e‑8·R (R = Cauchy radius).

**DoD:**
- New `tests/test_aberth.py` includes polynomials with a double root and a clustered pair; residual < 1e‑6 under iteration cap.
- No regressions on random degrees 3–12 vs current implementation.

---

### 3) Multiple‑root detection & handling
**Files:** `geodepoly/finishers.py`, `geodepoly/solver.py`, `tests/test_multiple_roots.py`  
**Goal:** Avoid stalling near repeated roots.  
**Tasks:**
- Multiplicity estimate `m̂` from `p, p', p''`: `m̂ ≈ round( Re( p * p'' / p'^2 ) )`, clamp to [1..3].
- For `m̂ ≥ 2`, switch to multiplicity‑aware Halley update:
  - `z ← z − m * p / ( m*p' − (m−1) * p*p''/p' )`.

**DoD:**
- `(x−1)^2 (x−2)` and `(x−1)^3` (with complex rotations) converge to residual < 1e‑6.

---

### 4) Better series seed center selection
**Files:** `geodepoly/solver.py`, `tests/test_centering.py`  
**Goal:** Pick centers μ that minimize `|t| = |−a0/a1|` with `a1≠0`.  
**Tasks:**
- Grid candidates on radii `[R/8, R/4, R/2, R]` (R = Cauchy bound), 16 angles each + axes.
- Score centers primarily by `|t|`, tie‑break by `|p(μ)|`.
- Try top‑K centers (e.g., K=4) for bootstrap; keep best residual.

**DoD:**
- On random degrees 5–12, median `|t|` of chosen seed < 0.7 for ≥70% of cases.

---

## Sprint 2 — Benchmarks, tests, and docs (P0/P1)

### 5) Benchmark suite & plots
**Files:** `geodepoly/scripts/bench_compare.py`, `scripts/plot_bench.py`, `docs/assets/*`, `tests/test_bench_smoke.py`  
**Goal:** Reproducible timing/accuracy with figures for docs.  
**Tasks:**
- Extend `bench_compare.py`:
  - Add flags: `--degrees 3,5,8,12,16`, `--methods`, `--resum`, `--trials`.
  - Emit CSV with per‑(degree,method) statistics (median/mean/std).
- New `scripts/plot_bench.py` to generate:
  - Residual vs degree (log‑y), Time vs degree (linear/log), Success rate bar chart.
- Smoke test `test_bench_smoke.py` ensures CSV schema & column names only (fast).

**DoD:**
- PNGs saved at `docs/assets/` and referenced from README.

---

### 6) Property‑based tests & edge cases
**Files:** `tests/test_properties.py`  
**Goal:** Catch numerical edge cases.  
**Tasks:**
- Use Hypothesis (optional) or custom sampler for:
  - Very small/large coefficients (10^±8), wide radius spreads, nearly singular leading term, nearly multiple roots.
- Verify `max |p(r_i)| < 1e−8` for degrees 3–10 across random seeds.

**DoD:**
- Property tests pass locally and in CI.

---

### 7) Docs polish & examples
**Files:** `README.md`, `docs/index.md`, `examples/*.py`  
**Goal:** Frictionless adoption.  
**Tasks:**
- Add examples: SymPy comparison to `nroots`, JSON bridge round‑trip, multiple‑root demo.
- Link figures + CSV from `docs/assets/` and cite commands to regenerate.

**DoD:**
- README quickstart and examples run clean in a new venv.

---

## Sprint 3 — Integrations & release polish (P1)

### 8) SymPy “friendly” wrapper and examples
**Files:** `geodepoly/sympy_plugin.py`, `examples/sympy_examples.py`  
**Goal:** Smooth SymPy adoption.  
**Tasks:**
- Add `return='numeric'|'expr'` (keep `numeric`; `'expr'` can warn until symbolic lift lands).
- Notebook: compare `sympy_solve(..., method="hybrid", resum="auto")` vs `nroots`.

**DoD:**
- Examples run; README references the notebook.

---

### 9) Mathematica/Maple bridges + docs
**Files:** `bridges/geodepoly_cli.py`, `docs/index.md`  
**Goal:** Out‑of‑the‑box CAS usage via JSON.  
**Tasks:**
- Document `RunProcess` (Mathematica) and `ssystem` (Maple) snippets with the JSON schema.

**DoD:**
- Copy‑paste snippets solve cubic/quintic correctly.

---

### 10) Release automation to TestPyPI → PyPI
**Files:** `.github/workflows/publish.yml`, `pyproject.toml`, `CHANGELOG.md`  
**Goal:** Safe release flow.  
**Tasks:**
- Add optional TestPyPI job with `TEST_PYPI_API_TOKEN`.
- Document version bump / tag / changelog steps in `CONTRIBUTING.md`.

**DoD:**
- Tagging `v0.1.1` publishes to TestPyPI; toggling to PyPI works.

---

## Sprint 4 — New capabilities (P2)

### 11) Eigenvalue solver path (`solve_eigs`)
**Files:** `geodepoly/eigs.py`, `tests/test_eigs.py`, `README.md`  
**Goal:** Robust small/medium eigenproblems via characteristic polynomial.  
**Tasks:**
- Implement Faddeev–LeVerrier to get characteristic coefficients; call `solve_all`.
- Gershgorin‑guided shifts; optional block deflation.

**DoD:**
- Matches `numpy.linalg.eigvals` on random dense matrices up to n=20 (residual ≤1e−8).

---

### 12) GeodeBench skeleton
**Files:** `bench/geodebench_spec.md`, `bench/generate_slices.py` (new)  
**Goal:** Dataset to probe symmetry generalization (Catalan/Fuss/Geode slices).  
**Tasks:**
- Spec tasks/splits/metrics; generate first CSVs and baselines.

**DoD:**
- Minimal leaderboard CSV + one baseline plot in docs.

---

### 13) GPU roadmap (spec)
**Files:** `docs/geode_gpu_spec.md`  
**Goal:** Plan batched kernels.  
**Tasks:**
- Write a 2–3 page spec with face‑layer tiling, memory layout, JAX prototype outline.

**DoD:**
- Spec committed and linked from README.

---

## How the IDE should run/verify locally

```bash
# fresh venv
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

# unit tests
pytest -q

# quick bench (creates CSV)
python -m geodepoly.scripts.bench_compare --degrees 3,5,8,12 --trials 20 --out docs/assets/bench.csv

# create plots (after you implement scripts/plot_bench.py)
python scripts/plot_bench.py --in docs/assets/bench.csv --out docs/assets

# run examples
python examples/quickstart.py
```

**CI:** Actions are configured to install with `".[dev]"` and run `pytest -q` on `main` pushes and PRs.  
**Publish:** Tag `vX.Y.Z` to trigger the PyPI workflow (requires `PYPI_API_TOKEN` secret).

---

End of file.
