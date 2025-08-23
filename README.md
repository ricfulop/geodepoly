# geodepoly — Series-Reversion Polynomial Solver (MVP)

[About](ABOUT.md)

[![CI](https://github.com/ricfulop/geodepoly/workflows/CI/badge.svg)](https://github.com/ricfulop/geodepoly/actions)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.16930528.svg)](https://doi.org/10.5281/zenodo.16930528)
[Docs](https://ricfulop.github.io/geodepoly/)
[![PyPI](https://img.shields.io/pypi/v/geodepoly.svg)](https://pypi.org/project/geodepoly/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Problem statement**

Finding the roots of a general polynomial is a workhorse task across CAS, controls, physics, optimization, and ML—but the standard toolchain is brittle:
	•	Newton/Raphson needs good initial guesses and can diverge or crawl near clusters and multiple roots; it also depends on derivatives that amplify conditioning issues.
	•	“All roots” numeric methods (Aberth, Durand–Kerner, Jenkins–Traub) work well but still struggle on ill-scaled coefficients, tight clusters, and near multiplicity, and they offer no symbolic structure to exploit.
	•	Symbolic radicals don’t exist beyond quartics; companion matrix eigenvalue routes trade robustness for heavy linear algebra and can be sensitive to scaling.

In practice, users juggle heuristics (shifts, deflation, scaling) and method switches to get a trustworthy answer—costly in pipelines where failure isn’t an option.

**Why this is a breakthrough**

GeodePoly turns polynomial solving into a series first, structure-aware pipeline:

1.	Analytic series reversion seed (derivative light, no guess):
We recenter the polynomial, invert a truncated series to get a closed-form update $$y=\sum g_m t^m (with t=-a_0/a_1)$$, and bootstrap a nearby root. This removes the fragile “pick a good starting point” step and reduces dependence on higher derivatives.

2.	Resummation for hard regimes:
Padé / Borel–Padé tame borderline radii of convergence (|t|\approx 1), extending where the seed works—exactly where Newton often fails.

3.	Structure → performance:
The Geode/Hyper-Catalan organization of coefficients gives a principled layering (think vertices/edges/faces) that’s cacheable, parallelizable, and eventually GPU-friendly—a path ordinary solvers don’t have.

4.	Hybrid certainty:
After the analytic seed, we switch to a robust simultaneous solver (Aberth–Ehrlich or DK) and Halley polish. You get derivative-light entry + derivative-free global convergence + machine-precision finish—with no user guesswork.

5.	One engine for symbolic + numeric + AI:
The same Geode arrays underpin CAS integration, a benchmark/dataset (GeodeBench) for symmetry generalization, and potential finite-field / coding variants—bridging worlds that have lived apart.

**What it unlocks**

•	A drop-in, robust polynomial solver for SymPy/Mathematica/Maple that “just works” on nasty instances.

•	Eigenvalue pipelines via characteristic polynomials that are more stable on clustered spectra.

•	Scalable kernels (batched solving, differentiable “root layers” in ML) thanks to Geode’s compositional structure.

•	A new benchmark for AI to test whether models truly learn hidden algebraic symmetries.

Bottom line: we replace guess-heavy, case-by-case numerics with a universal, series-driven, structure-aware solver—turning a century-old pain point into a reliable, extensible core primitive.

`geodepoly` is built as a small Python package that finds all roots of a complex polynomial
using a **shift–recenter + truncated series reversion** with optional bootstrap
iterations, and only falls back to classical iterations when strictly necessary.

- Compositional inverse (via Lagrange inversion in coefficient form) around a local
  recentering point to obtain an analytic series for a nearby root.
- **Bootstrap:** update the center by the series estimate and re-expand (typically a few steps).
- **Deflation:** synthetic division to peel off roots one-by-one.
- Safe fallbacks: Halley or Durand–Kerner if series degenerates (multiple root / tiny derivative).

Polynomials sit at the heart of CAS tooling, scientific computing, control, and ML. GeodePoly aims to be a drop-in, open-source engine you can trust in numerically awkward cases (tight clusters, near-multiplicity, wide coefficient scales) while remaining fast and easy to integrate.

> Note: This is a *minimal working scaffold* you can publish and iterate on.
> It is self-contained (numpy optional), tested, and provides a SymPy hook.

## Install (editable)

```bash
pip install -e .
```

## Install (PyPI)

```bash
pip install geodepoly
```

## Quickstart

```python
from geodepoly import solve_all

# Coefficients lowest-degree first: a0 + a1 x + ... + aN x^N
coeffs = [1, 0, -7, 6]
roots = solve_all(coeffs, method="hybrid", resum="auto")
print(roots)
```

## CLI

```bash
python -m geodepoly.scripts.benchmark --deg 8 --seed 123 --trials 100
```

Solve a polynomial from the command line:

```bash
geodepoly-solve --coeffs "[-6,11,-6,1]" --method hybrid --resum auto --json
```

### Examples

- SymPy comparison: `python examples/sympy_vs_nroots.py`
- JSON bridge round‑trip: `python examples/json_bridge_roundtrip.py`
- Multiple root demo: `python examples/multiple_root_demo.py`
- Compare methods: `python examples/compare_methods.py`
- Resummation effect: `python examples/resummation_effect.py`
- Eigenvalues demo: `python examples/eigs_demo.py`

## API

- `series_solve_all(coeffs, max_order=32, boots=3, tol=1e-12, max_deflation=None, verbose=False)`
- `series_one_root(coeffs, center=None, max_order=32, boots=3, tol=1e-14)`
- `sympy_solve(poly)` — lightweight SymPy integration (if SymPy is installed).
- `solve_eigs(A)` — eigenvalues via characteristic polynomial (Faddeev–LeVerrier).

## How it works (short)

Let `p(x)` be a degree-`n` polynomial. We recenter around `x = μ` and expand
`q(y) = p(μ + y) = a0 + a1 y + a2 y^2 + ...`. If `a1 ≠ 0`, solve `q(y)=0` by
compositional inversion of `F(y) = y + β2 y^2 + β3 y^3 + ...` with `βk = ak/a1`.
Lagrange inversion gives the inverse coefficients `{g_m}` of `F`, and the nearby
root is `y ≈ Σ_{m≥1} g_m t^m`, with `t = -a0/a1`. Update `μ ← μ + y` (bootstrap)
and repeat a few times; then deflate and continue.

This repo implements the coefficient formula
`g_m = (1/m) * [y^{m-1}] (1 / F'(y))^m` using truncated series arithmetic.
No derivatives of `p` beyond `a1 = q'(0)` are used.

## Caveats

- Multiple or nearly-multiple roots are ill-conditioned for *any* method.
  We switch to a guarded Durand–Kerner step when `|a1|` is tiny.
- Convergence radius depends on the local analytic structure; bootstrap helps.

## License

MIT

---

## New in this build

### Friendlier API
```python
from geodepoly import solve_poly, solve_all, solve_one
roots = solve_poly(coeffs, method="hybrid", resum="pade")
```

**Methods:** `hybrid` (series seeds + Aberth), `aberth`, `dk`, `numpy` (companion).  
**Resummation:** `None`, `"pade"`, `"borel"`, `"borel-pade"`.

### SymPy integration
```python
from geodepoly.sympy_plugin import sympy_solve
roots = sympy_solve(x**8 - 3*x + 1, method="hybrid", resum="pade")
```

### Mathematica / Maple bridge (JSON CLI)
```bash
geodepoly-bridge <<'JSON'
{"coeffs":[-6,11,-6,1],"kwargs":{"method":"hybrid","resum":"pade"}}
JSON
```
In Mathematica:
```wl
payload = ExportString[<|"coeffs"->{-6,11,-6,1},"kwargs"-><|"method"->"hybrid","resum"->"pade"|>|>,"JSON"];
res = RunProcess[{"geodepoly-bridge"}, "StandardInput"->payload, "StandardOutput"];
ImportString[res, "JSON"]
```

In Maple:
```maple
payload := JSON:-Encode( table([coeffs=[-6,11,-6,1], kwargs=table([method="hybrid", resum="pade"])]) ):
res := ssystem("geodepoly-bridge", payload):
JSON:-Parse(res);
```

### Benchmarks
```bash
geodepoly-bench --degrees 8 --methods hybrid,aberth,dk --trials 50 --out bench_deg8.csv
```

Aggregate and plot (see `docs/assets/`):
```bash
python scripts/bench_presets.py quick
# or run manually:
geodepoly-bench --degrees 3,5,8,12 --methods hybrid,aberth,dk --trials 10 --out docs/assets/bench.csv --agg_out docs/assets/bench_agg.csv --resum auto
python scripts/plot_bench.py --in docs/assets/bench_agg.csv --out docs/assets
```

Previews:

![Time vs Degree](docs/assets/time_vs_degree.png)

![Residual vs Degree](docs/assets/residual_vs_degree.png)

### Paper skeleton
See `paper/GeodePoly_MVP.md`.

## Notebooks

- Quickstart: `notebooks/Quickstart.ipynb`
- Bench summary: `notebooks/BenchSummary.ipynb`

## Bench dataset & GPU roadmap

- GeodeBench spec: `bench/geodebench_spec.md`, generator: `bench/generate_slices.py`
- GPU roadmap: `docs/geode_gpu_spec.md`

## Hyper-Catalan API (S[t2,t3,...])

Utilities based on the paper's multivariate series:

```python
from geodepoly import evaluate_hyper_catalan, evaluate_quadratic_slice, catalan_number

# Quadratic slice (Catalan series)
t2 = 0.05
alpha_approx = evaluate_quadratic_slice(t2, max_weight=20)
catalan_series = sum(catalan_number(n) * (t2**n) for n in range(12))

# Multivariate evaluation (truncated by weighted degree)
alpha_multi = evaluate_hyper_catalan({2: 0.05, 3: 0.01}, max_weight=12)
```

See `docs/paper_guide.md` for how the paper maps onto the codebase.

## Cite

If you use this software, please cite:

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.16930528.svg)](https://doi.org/10.5281/zenodo.16930528)

`CITATION.cff` is included at the repo root.
