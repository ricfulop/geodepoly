# API

## Core solvers

- `geodepoly.solve_all(coeffs: List[complex], method: str = "hybrid", max_order: int = 24, boots: int = 2, tol: float = 1e-12, resum: Optional[str] = None, refine_steps: int = 3, verbose: bool = False) -> List[complex]`
  - Methods: `"hybrid" | "aberth" | "dk" | "numpy"`
  - Resummation: `None | "pade" | "borel" | "borel-pade" | "auto"`

- `geodepoly.solve_one(coeffs: List[complex], center: complex|None=None, max_order: int=24, boots: int=3, tol: float=1e-14, resum: Optional[str]=None, refine_steps: int=2) -> complex`

- `geodepoly.solve_poly = solve_all` (convenience alias)

## Series reversion (paper-aligned)

- `geodepoly.series_solve_all(coeffs, max_order=24, boots=3, tol=1e-12, max_deflation=None, verbose=False) -> List[complex]`
- `geodepoly.series_one_root(coeffs, center=None, max_order=24, boots=3, tol=1e-14) -> complex`

## SymPy integration

- `geodepoly.sympy_plugin.sympy_solve(poly, return_: str = "numeric", **kwargs) -> List`
  - `poly`: `sympy.Expr` or `sympy.Poly`
  - `return_`: `"numeric"` (Python complex) or `"expr"` (SymPy numbers)

## Eigenvalues

- `geodepoly.eigs.solve_eigs(A) -> List[complex]`
  - Forms characteristic polynomial via Faddeev–LeVerrier and calls `solve_all`.

## Hyper-Catalan utilities

- `geodepoly.hyper_catalan.hyper_catalan_coefficient(m_counts: Dict[int,int]) -> int`
- `geodepoly.hyper_catalan.evaluate_hyper_catalan(t_values: Dict[int,complex], max_weight: int) -> complex`
- `geodepoly.hyper_catalan.evaluate_quadratic_slice(t2: complex, max_weight: int) -> complex`
- `geodepoly.hyper_catalan.catalan_number(n: int) -> int`

## Resummation helpers

- `geodepoly.resummation.eval_series_plain(g: List[complex], t: complex) -> complex`
- `geodepoly.resummation.eval_series_pade(g: List[complex], t: complex) -> complex`
- `geodepoly.resummation.eval_series_borel(g: List[complex], t: complex) -> complex`
- `geodepoly.resummation.eval_series_borel_pade(g: List[complex], t: complex) -> complex`
- `geodepoly.resummation.eval_series_auto(g: List[complex], t: complex) -> complex`

## CLI bridge (JSON)

- `bridges/geodepoly_cli.py`
  - stdin: `{ "coeffs": [...], "kwargs": { ... } }`
  - stdout: `{ "roots": [[re, im], ...] }`
