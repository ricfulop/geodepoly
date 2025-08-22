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
