# Changelog

## [0.1.0] - 2025-08-21
### Added
- Initial public release of **geodepoly**.
- Hybrid solver: series-reversion seed + Aberth–Ehrlich finisher + Halley polish.
- Resummation options: Padé, Borel, Borel–Padé.
- SymPy plugin and JSON CLI bridge (Mathematica/Maple).
- Benchmarks and paper skeleton.

## [0.1.2] - 2025-08-22
### Added
- Hyper-Catalan S-series API (`geodepoly.hyper_catalan`): coefficients, truncated evaluation, Catalan slice.
- Docs: `docs/paper_guide.md` mapping paper sections to code; README badges (CI, PyPI).
- Example: `examples/hyper_catalan_demo.py`.
### Fixed
- Tests: random polynomial accuracy via stronger polishing; packaging discovery; CI `.[dev]` install.
- Stray character in `geodepoly/sympy_plugin.py`.

## [0.1.3] - 2025-08-22
### Added
- Resummation: adaptive `auto` mode with fallback to Borel–Padé.
- Aberth–Ehrlich: adaptive damping and minimal repulsion for clusters; tests.
- Multiple-root handling: multiplicity-aware Halley; tests; wired into solver polish.
- Seeding: center selection minimizing |t| with top-K candidates; test.
- Benchmarks: `scripts/bench_compare.py` supports `--resum`.

## [0.1.4] - 2025-08-22
### Added
- Bench suite: degrees/methods flags, aggregate CSV; plots and docs previews.
- Property-based edge tests.
- SymPy plugin return mode and CAS examples; stable CLI JSON.
- Eigenvalue solver `solve_eigs` via Faddeev–LeVerrier; tests.
- GeodeBench spec and slice generator; GPU roadmap doc.

## [0.1.5] - 2025-08-22
### Added
- README: inline plot previews; examples and CAS tips.
- Docs: minor bridge/GPU notes.

## [0.1.6] - 2025-08-23
### Added
- Bench presets runner `scripts/bench_presets.py` to regenerate standard CSVs/plots.
- Notebooks: `notebooks/Quickstart.ipynb`, `notebooks/BenchSummary.ipynb`.
- Docs: `docs/bench_presets.md` and nav link in `mkdocs.yml`.
### Fixed
- CI stability: all tests green (20 passed).

## [0.1.7] - 2025-08-23
### Changed
- Code style cleanup across package for CI lint/format: split imports, remove semicolons/unused imports, clarify variable names.
- Minor numeric polish and formatting in examples and benchmarks.
### Fixed
- CI now green with ruff/black scoped to package and code formatted accordingly.

## [0.1.7.post1] - 2025-08-23
### Fixed
- Post-release metadata bump to trigger PyPI publish via tags.

## [0.1.8] - 2025-08-23
### Added
- MVP3 batched kernels: `batched_poly_eval`, `batched_newton_step`, differentiable `torch_root_layer`.
- Batch throughput benchmark `scripts/bench_batched.py` and docs page `docs/batched.md`.
- Export batched APIs in `geodepoly.__init__`.
### Changed
- GeodeBench baseline uses a simple holdout split.

## [0.1.9] - 2025-08-23
### Added
- CLI commands: `geodepoly-solve` (solve from terminal), `geodepoly-bridge` (JSON CLI for CAS).
- Batched solver path: `batched_solve_all` and `method="batched"` integration.
- Numba opt-in acceleration for `poly_eval` and derivatives (set `GEODEPOLY_USE_NUMBA=1`).
- GPU prototype benchmark: `scripts/bench_gpu_prototype.py` with NumPy/Torch/JAX backends.
- Publish workflow for TestPyPI (`test-*` tags) and PyPI (`v*` tags).
### Docs
- README and docs updated with CLI/bridge examples and GPU usage.

## [0.1.10] - 2025-08-23
### Added
- FormalSeries minimal with add/mul/truncate/compose.
- Series scaffolds: `series_root`, `series_bootstrap` (now functional, Wallis cubic demo).
- Geode factorization: constructs S, S1, G with `(S-1) == S1*G` (truncated) identity; tests.
### Tests
- New `tests/test_series_basic.py` covering FormalSeries and series bootstrapping.

## [0.1.11] - 2025-08-23
### Added
- `Polynomial` class with ops (add/sub/mul/divmod/pow), eval, shift/scale, diff/integrate.
- Numeric wrappers: `newton`, `aberth`, `dk`, `companion_roots` delegating to existing implementations.
### Tests
- `tests/test_polynomial.py`, `tests/test_numeric_wrappers.py`.

## [0.1.12] - 2025-08-23
### Changed
- Performance: cache Padé coeffs in `eval_series_auto`, reuse Horner coeffs in Aberth; optional Numba path for finishers.
- CI/docs: Docs deploy workflow; CLI/bridge schema v1; formatting and type checks green.
### Added
- Profiling script `scripts/profile_finishers.py`.

## [0.1.13] - 2025-08-23
### Changed
- Batched CLI refinements and GPU benchmark scripts updates (`bench_aberth_*`).
- Solver fastpath and finishers polish; minor CLI ergonomics.
### CI
- Wheels workflow triggers on `v*` tags; package data includes `py.typed`.
### Misc
- Docs and examples touch-ups.

## [0.1.14] - 2025-08-23
### Added
- `series_root(...)` now returns a proper truncated `FormalSeries` via Lagrange inversion.
- `geode_factorize(...)` builds `S` combinatorially from Hyper‑Catalan coefficients and solves `(S−1)=S1*G` degree‑by‑degree using exact rationals.
### Tests
- All tests green (66/66); factorization identity verified up to truncation.
