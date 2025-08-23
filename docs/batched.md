# Batched kernels (MVP3)

- Backends: NumPy (reference), Torch, JAX.
- Ops:
  - `batched_poly_eval(coeffs, xs, backend)`
  - `batched_newton_step(coeffs, xs, backend)`
  - `torch_root_layer(steps, tol)` differentiable Newton layer
  - `batched_solve_all(coeffs_batch, backend, steps)` (one root per polynomial)

## Integrating with the main solver

- Use `method="batched"` to solve a single polynomial via a vectorized Newton + deflation baseline (CPU/NumPy path today):

```python
from geodepoly import solve_all
roots = solve_all([ -6, 11, -6, 1 ], method="batched")
```

Bench (preview): expect 10–100× throughput vs pure Python on large batches (Torch/JAX on GPU).

Quick bench:
```bash
python scripts/bench_batched.py --backend numpy --batch 8192 --degree 16
# if GPU available
python scripts/bench_batched.py --backend torch --batch 65536 --degree 16
```
