# API

## Core solvers

- `geodepoly.solve_all(coeffs: List[complex], method: str = "hybrid", max_order: int = 24, boots: int = 2, tol: float = 1e-12, resum: Optional[str] = None, refine_steps: int = 3, verbose: bool = False) -> List[complex]`
  - Methods: `"hybrid" | "aberth" | "dk" | "numpy"`
  - Resummation: `None | "pade" | "borel" | "borel-pade" | "auto"`

Example
```python
from geodepoly import solve_all
coeffs = [1, 0, -7, 6]  # 1 - 7 x^2 + 6 x^3 = 0 (low->high)
roots = solve_all(coeffs, method="hybrid", resum="auto")
```

- `geodepoly.solve_one(coeffs: List[complex], center: complex|None=None, max_order: int=24, boots: int=3, tol: float=1e-14, resum: Optional[str]=None, refine_steps: int=2) -> complex`

- `geodepoly.solve_poly = solve_all` (convenience alias)

## Series reversion (paper-aligned)

- `geodepoly.series_solve_all(coeffs, max_order=24, boots=3, tol=1e-12, max_deflation=None, verbose=False) -> List[complex]`
- `geodepoly.series_one_root(coeffs, center=None, max_order=24, boots=3, tol=1e-14) -> complex`

Example
```python
from geodepoly import series_one_root
coeffs = [1, -1.2, 0.3, 1.0]
root = series_one_root(coeffs, center=0.0, max_order=24, boots=2)
```

## SymPy integration

- `geodepoly.sympy_plugin.sympy_solve(poly, return_: str = "numeric", **kwargs) -> List`
  - `poly`: `sympy.Expr` or `sympy.Poly`
  - `return_`: `"numeric"` (Python complex) or `"expr"` (SymPy numbers)

Example
```python
import sympy as sp
from geodepoly.sympy_plugin import sympy_solve

x = sp.symbols('x')
roots = sympy_solve(x**5 + x - 1, method="hybrid", resum="auto", return_="numeric")
```

## Eigenvalues

- `geodepoly.eigs.solve_eigs(A) -> List[complex]`
  - Forms characteristic polynomial via Faddeev–LeVerrier and calls `solve_all`.

Example
```python
import numpy as np
from geodepoly.eigs import solve_eigs

A = np.array([[0,1],[ -6, 7 ]], dtype=complex)
eigvals = solve_eigs(A)
```

## Hyper-Catalan utilities

- `geodepoly.hyper_catalan.hyper_catalan_coefficient(m_counts: Dict[int,int]) -> int`
- `geodepoly.hyper_catalan.evaluate_hyper_catalan(t_values: Dict[int,complex], max_weight: int) -> complex`
- `geodepoly.hyper_catalan.evaluate_quadratic_slice(t2: complex, max_weight: int) -> complex`
- `geodepoly.hyper_catalan.catalan_number(n: int) -> int`

Example
```python
from geodepoly import evaluate_hyper_catalan, evaluate_quadratic_slice, catalan_number

t2 = 0.05
alpha = evaluate_quadratic_slice(t2, max_weight=20)
c3 = catalan_number(3)
```

## Resummation helpers

- `geodepoly.resummation.eval_series_plain(g: List[complex], t: complex) -> complex`
- `geodepoly.resummation.eval_series_pade(g: List[complex], t: complex) -> complex`
- `geodepoly.resummation.eval_series_borel(g: List[complex], t: complex) -> complex`
- `geodepoly.resummation.eval_series_borel_pade(g: List[complex], t: complex) -> complex`
- `geodepoly.resummation.eval_series_auto(g: List[complex], t: complex) -> complex`

Example
```python
from geodepoly.resummation import eval_series_plain, eval_series_pade, eval_series_borel_pade, eval_series_auto
g = [1, 1, 1, 1, 1]  # toy coefficients for demo
t = 0.9
vals = dict(plain=eval_series_plain(g,t), pade=eval_series_pade(g,t), borel_pade=eval_series_borel_pade(g,t), auto=eval_series_auto(g,t))
```

## CLI bridge (JSON)

- `bridges/geodepoly_cli.py`
  - stdin: `{ "coeffs": [...], "kwargs": { ... } }`
  - stdout: `{ "roots": [[re, im], ...] }`

Example
```bash
python bridges/geodepoly_cli.py <<'JSON'
{"coeffs":[-6,11,-6,1],"kwargs":{"method":"hybrid","resum":"auto"}}
JSON
```
