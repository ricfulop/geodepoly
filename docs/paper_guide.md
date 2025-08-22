## Mapping the paper to the codebase

This guide cross-references “A Hyper-Catalan Series Solution to Polynomial Equations, and the Geode” with symbols and modules in this repository.

- Geometric polynomial equation and the series solution `S[t2,t3,...]`:
  - Paper: Equation 0 = 1 − α + t2 α^2 + t3 α^3 + ..., with α = S[...].
  - Code: `geodepoly/hyper_catalan.py` implements:
    - `hyper_catalan_coefficient(m_counts)` for the array coefficients.
    - `evaluate_hyper_catalan(t_values, max_weight)` to numerically sum a truncated S.
    - `evaluate_quadratic_slice(t2, ...)` and `catalan_number(n)` for the Catalan slice.

- Lagrange inversion / series reversion for a shifted polynomial:
  - Paper: Sections on Lagrange inversion and series bootstrap.
  - Code: `geodepoly/series_solve.py`:
    - `shift_expand`, `inverseseries_g_coeffs`, `series_step`, `series_one_root`.

- Finishing methods and polishing:
  - Paper: Practical computation beyond the formal series.
  - Code: `durand_kerner`, `halley_refine`, `newton_refine`, composed in `series_solve_all`.

- Resummation and acceleration:
  - Paper: discusses summation/acceleration themes.
  - Code: `geodepoly/resummation.py` supports Padé and Borel(-Padé) options.

- The Geode array and combinatorial structure:
  - Paper: factorization and conjectures about the array.
  - Code: `hyper_catalan_coefficient` and `evaluate_hyper_catalan` expose the array numerics; future work can add factorization utilities once conjectures are finalized.

- Bridges and examples:
  - Paper: Worked examples (e.g., Wallis cubic) and CAS bridges.
  - Code: `bridges/geodepoly_cli.py`, `examples/quickstart.py`, and tests in `tests/`.


