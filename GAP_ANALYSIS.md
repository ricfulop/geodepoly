# GeodePoly — Gap Analysis & To‑Do

This checklist maps the **SPEC.md** to what’s likely in your current repo and outlines concrete tasks to reach “top polynomial library” status. Where the plan depends on the Hyper‑Catalan/Geode theory, I cite the Monthly paper sections you’re implementing so Cursor has exact anchors. Citations use the same marker form you asked for.

---

## Legend

- **Priority**: P0 (now), P1 (next), P2 (later)
- **Effort**: S (small), M (medium), L (large)
- **Owner**: IDE (Cursor‑gen), Human, Both

---

## 1) Repository Snapshot (Assumed)

- `src/` with early Python for **hyper‑Catalan numbers** and a generating‑series sketch.
- Some **unit tests** for Catalan and small hyper‑Catalan slices.
- Early wiring for **series‑based solving**, but not generalized or packaged.

> If anything above is wrong, you won’t lose work: these tasks are additive/organized as PRs.

---

## 2) High‑Level Gaps vs SPEC

### 2.1 Core Math (Series Roots & Geode)
- [ ] **Implement Theorem 4/7 soft polynomial formula** `series_root(...)` (formal series in coefficients) with truncation by total face/weight. **P0 • L • Both** 【21†source】
- [ ] **Memoized `hyper_catalan(m)`** via factorial‑power quotient form and `(E_m, V_m, F_m)` helpers. **P0 • M • IDE** 【21†source】
- [ ] **Build S, S₁, G and verify `S − 1 = S₁ · G`** (`geode_factorize`). **P0 • M • IDE** 【21†source】
- [ ] **Bi‑Tri array generator** + alternating cross‑diagonal checks. **P1 • S • IDE** 【21†source】
- [ ] **Layerings**: vertex/edge/face builders and coefficient extractors. **P1 • M • IDE** 【21†source】

### 2.2 Numeric Root Finding
- [ ] `series_bootstrap(f, x0, series_order, rounds, damping)` (shift, solve, iterate) + Wallis cubic example. **P0 • M • IDE** 【21†source】
- [ ] Classical solvers: `newton`, `durand_kerner`, `aberth`, `companion_roots`. **P0 • M • IDE**
- [ ] Hybrids: `series_seed_then_newton`, `multi_start_hybrid`. **P1 • M • IDE**
- [ ] Robust polishing/deflation for clustered roots. **P2 • M • IDE**

### 2.3 Data Types & Algebra
- [ ] `Polynomial` class (dense/sparse) with exact rational (`fractions.Fraction`) + high‑precision (`mpmath`) paths. **P0 • M • IDE**
- [ ] Ops: add/mul/divmod/pow/compose/diff/integrate/shift/scale; fast Horner/Estrin. **P0 • M • IDE**
- [ ] Resultant, GCD, square‑free factor. **P1 • M • IDE**

### 2.4 Formal Series Object
- [ ] `FormalSeries` with arithmetic, truncation, substitution, composition, LaTeX/MD printing. **P0 • M • IDE**
- [ ] Sparse monomial dict keyed by type `m=(m2,m3,...)` with trailing‑zero trim; degree/weight cutoffs. **P0 • M • IDE**

### 2.5 Interop & Utilities
- [ ] SymPy/Numpy/Mpmath glue; IO (JSON/csv); simple plotting. **P1 • S • IDE**
- [ ] Deterministic seeding; reproducible configs. **P1 • S • IDE**

### 2.6 Docs / Examples / CI
- [ ] `README.md` quickstart + badges. **P0 • S • IDE**
- [ ] **Examples**: Wallis cubic, Eisenstein Bring radical, Geode arrays. **P0 • S • IDE** 【21†source】
- [ ] mkdocs site: Concepts/How‑to/Theory with paper links. **P1 • M • IDE** 【21†source】
- [ ] CI (ruff/mypy/pytest/docs) on 3.9–3.12; coverage targets. **P0 • S • IDE**
- [ ] Packaging (`pyproject.toml`) + CLI (`geodepoly` entrypoints). **P0 • S • IDE**

---

## 3) Detailed To‑Do (File‑Level)

### 3.1 `geodepoly/series.py`
- [ ] `def hyper_catalan(m: Mapping[int,int]) -> int:`
  - Compute \(F_m, E_m, V_m\) and \(C_m = \frac{(E_m-1)!}{(V_m-1)!\,m!}\). Cache results. 【21†source】
- [ ] `def series_root(c: Sequence, order: int, variant="raw") -> FormalSeries:`
  - Implement Theorem 4/7 summation with truncation; support factorial‑power form. 【21†source】
- [ ] `def geode_factorize(order: int, tmax: int=None) -> (FormalSeries, FormalSeries, FormalSeries):`
  - Construct S; compute \(S_1 = \sum_{k\ge2} t_k\); return `(S, S1, G=(S-1)/S1)`; assert coefficient identity. 【21†source】
- [ ] `def bi_tri_array(n2_max, n3_max) -> np.ndarray:`
  - Use \( \frac{(2m_2+3m_3)!}{(1+m_2+2m_3)!\,m_2!\,m_3!} \); add alternating diagonal test. 【21†source】
- [ ] `def layer_vertex/edge/face(...):` build `S_V, S_E, S_F`; expose coefficient queries. 【21†source】

### 3.2 `geodepoly/formal.py`
- [ ] `class FormalSeries:` add `__add__`, `__mul__`, `pow`, `truncate`, `compose`, `subs`, `coeff`, `to_sympy`, `latex()`.

### 3.3 `geodepoly/numeric.py`
- [ ] `newton`, `durand_kerner`, `aberth`, `companion_roots` with tolerances.
- [ ] `series_bootstrap` implementing the shift–solve–add cycle; demo Wallis cubic. 【21†source】

### 3.4 `geodepoly/poly.py`
- [ ] `class Polynomial:` storage (dense list or sparse dict), ops, eval (Horner/Estrin), `shift`, `scale`, `differentiate`, `integrate`.

### 3.5 `geodepoly/utils.py`
- [ ] Factorial cache; factorial‑power quotients; type‑`m` normalization; truncation policy helpers.

### 3.6 `geodepoly/interop.py`
- [ ] Converters to/from SymPy/Numpy/Mpmath.

### 3.7 `geodepoly/plotting.py` (optional)
- [ ] Root plots, residual trajectories, simple convergence basins (small grids).

### 3.8 Project Files
- [ ] `pyproject.toml` (name, deps, console_scripts).
- [ ] `README.md` with theory links and examples. 【21†source】
- [ ] `SPEC.md` (already provided).
- [ ] `examples/01_wallis_cubic.ipynb` reproduces Q(t2,t3) and K(c0,c1,c2,c3) bootstrapping. 【21†source】
- [ ] `examples/02_eisenstein_quintic.ipynb` (Bring radical series). 【21†source】
- [ ] `examples/03_geode_arrays.ipynb` build `G` slice, verify `S-1=S1·G`. 【21†source】
- [ ] `tests/` described below.
- [ ] GitHub Actions workflows: `lint.yml`, `tests.yml`, `docs.yml`.

---

## 4) Tests (Acceptance‑style)

- [ ] **Catalan**: `hyper_catalan({2:n}) == Catalan(n)` for n≤12. **P0**
- [ ] **Bi‑Tri**: small table exact equality to paper numbers. **P0** 【21†source】
- [ ] **Geode factorization**: build S to face degree 4; assert `(S-1) == S1*G` coef‑wise. **P0** 【21†source】
- [ ] **Wallis cubic**: `series_bootstrap(x^3-2x-5, x0=2.0, order=12, rounds=2)` ≈ 2.0945514815423265 (|Δ|<1e-15). **P0** 【21†source】
- [ ] **Bring radical**: match coefficients through \(t^{17}\). **P0** 【21†source】
- [ ] **Hybrid vs Aberth**: random deg‑20 set; hybrid gets lower residual ≤200 iters ≥90% of cases. **P1**
- [ ] **Determinism**: fixed seed → identical output ordering and residuals. **P1**

---

## 5) Prioritized Sprint Plan

**Sprint 1 (P0) — Completed**
1. `pyproject.toml`, `README.md`, CI skeleton.
2. `hyper_catalan`, `(E,V,F)` helpers, factorial cache.
3. `FormalSeries` minimal.
4. `series_root` (raw) + `series_bootstrap` + Wallis cubic example.
5. Tests: Catalan/Bi‑Tri/Wallis.

**Sprint 2 (P0→P1) — Completed**
1. `geode_factorize` + factorization tests.
2. Classical solvers + `Polynomial` type + hybrid glue.
3. Examples: Eisenstein/Geode arrays. Docs quickstart.

**Sprint 3 (P1) — Completed**  
- JSON bridge schema v1 + CLI I/O flags, tests.
- GPU prototype: vectorized Newton/ Aberth (Torch, JAX), benches and docs.
- Coverage raised to ≥90% (with configured omits), CI threshold enforced.

**Sprint 4 (P1) — In progress**
- Docs polish (badges, Colab), SPEC updates.
- GPU batched pipeline integration, benchmarks, and comparisons.
- Packaging wheels and install matrix.

---

## 6) Safety Rails for Cursor (to protect your existing work)

- Add a `.cursor/rules.json` (or equivalent) with:
  - `"never_edit": ["src/**/legacy_*.py", "notebooks/**"]`
  - `"prefer_edit": ["geodepoly/**", "tests/**"]`
- Use PR branches: `feature/series-root`, `feature/geode`, `feature/hybrid`.
- Require tests to pass before merge in branch protection.
- Enable “approve to apply” mode: human OKs file‑adds or big refactors.

---

## 7) Minimal Stubs to Unblock Cursor

Create empty files so Cursor fills them instead of refactoring your early code:
```
geodepoly/__init__.py
geodepoly/series.py
geodepoly/formal.py
geodepoly/numeric.py
geodepoly/poly.py
geodepoly/utils.py
geodepoly/interop.py
tests/test_series_basic.py
tests/test_numeric_roots.py
tests/test_arrays.py
```
Each can just contain `# TODO: Cursor — implement as per SPEC.md`.

---

## 8) Theory Anchors (what each function implements)

- **Soft polynomial formula / series root** — Theorems 3,4,7: construction of series solution and coefficient form with \(V_m, E_m, F_m\). 【21†source】
- **Hyper‑Catalan coefficient** — closed form \(C_m = \frac{(E_m-1)!}{(V_m-1)!\,m!}\). 【21†source】
- **Cubic one‑line Q(t₂,t₃)** and **bootstrapping** demo (Wallis). 【21†source】
- **Geode factorization** \(S-1=S_1\cdot G\) and layerings (vertex/edge/face). 【21†source】
- **Eisenstein/Bring radical** series recovery. 【21†source】

---

## 9) What You Keep (no waste)

- Your existing hyper‑Catalan code and tests are **inputs** to `hyper_catalan`, `bi_tri_array`, and validation for `series_root`.
- Any current polynomial utilities can be kept under `src/` and wrapped by the new `geodepoly/` API.
- The gap plan is additive; nothing is deleted unless you choose to rename for clarity.

---

## 10) Done‑Definition

- Green on **P0 tests**, CI, and examples reproducible.
- `pip install -e .` exposes `geodepoly` and `geodepoly` CLI.
- README has quickstart solving Wallis cubic, and an image/table for Bi‑Tri and Geode slices.
