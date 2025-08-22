# geodepoly Docs

- **API**: `solve_poly`, `solve_all`, `solve_one`
- **Methods**: `hybrid`, `aberth`, `dk`, `numpy`
- **Resummation**: `None`, `pade`, `borel`, `borel-pade`
- **CAS**: SymPy plugin, JSON CLI for Mathematica/Maple

See the README for quickstart and `paper/GeodePoly_MVP.md` for a draft paper.

## Theory and Paper Mapping

- See `docs/paper_guide.md` for how the paper “A Hyper-Catalan Series Solution to Polynomial Equations, and the Geode” maps to the implementation.
- The module `geodepoly.hyper_catalan` provides utilities for the multivariate series `S[t2,t3,...]` described in the paper.
