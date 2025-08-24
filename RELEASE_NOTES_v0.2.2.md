# geodepoly v0.2.2 — Lint fixes, safer Geode convolution, docs polish

Date: 2025-08-23
Tag: v0.2.2

## Highlights
- Fix lints/types across new modules; safer loops in Geode convolution (no np.nditer writes).
- Tests adjusted for strict typing; full suite green.
- Docs/UX improvements:
  - AI Quickstart Colab notebook (install, Torch demo, optional JAX demo; troubleshooting/GPU notes).
  - Two additional Colab notebooks: Pole Placement, Spectral Matching.
  - AI Benchmarks docs page with how-to run Torch/JAX forward/backward benches.
  - README additions: Geode convolution example and Colab links.
- CI/Packaging:
  - Windows added to wheels build matrix.
  - CI Python matrix expanded to 3.9/3.10/3.11/3.12.
  - Codecov upload + badge.

## Changes since v0.2.1
- Refactor Geode convolution (Python) to use direct index loops for weight-cropping; type-safe dict↔array conversions.
- Minor test updates to use complex literals consistently.

## Installation
```bash
pip install -U geodepoly
# optional AI extras
pip install geodepoly[ai-torch]
# or
pip install geodepoly[ai-jax]
```

## Links
- Changelog: CHANGELOG.md
- Docs: https://ricfulop.github.io/geodepoly/
- AI Quickstart (Colab): notebooks/AI_Quickstart.ipynb
- Demos (Colab): Pole Placement, Spectral Matching linked from README

---
If anything looks off, please open an issue or PR. Thanks!
