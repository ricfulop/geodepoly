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
