# Examples

## Solve all roots (hybrid)
```python
from geodepoly import solve_all
coeffs = [1, 0, -7, 6]
roots = solve_all(coeffs, method="hybrid", resum="auto")
```

## Series one root
```python
from geodepoly import series_one_root
coeffs = [1, -1.2, 0.3, 1.0]
root = series_one_root(coeffs, center=0.0, max_order=24, boots=2)
```

## SymPy integration
```python
import sympy as sp
from geodepoly.sympy_plugin import sympy_solve
x = sp.symbols('x')
roots = sympy_solve(x**5 + x - 1, method="hybrid", resum="auto", return_="numeric")
```

## Eigenvalues
```python
import numpy as np
from geodepoly.eigs import solve_eigs
A = np.array([[0,1],[-6,7]], dtype=complex)
vals = solve_eigs(A)
```

## JSON bridge
```bash
python bridges/geodepoly_cli.py <<'JSON'
{"coeffs":[-6,11,-6,1],"kwargs":{"method":"hybrid","resum":"auto"}}
JSON
```

## Geode arrays
```bash
python examples/geode_arrays_demo.py
```

## Eisenstein/Bring quintic
```bash
python examples/eisenstein_quintic_demo.py
```

