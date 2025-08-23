# Performance Notes

## Numba acceleration (opt-in)

Enable JIT for core polynomial evaluation by setting an environment variable before import:

```bash
export GEODEPOLY_USE_NUMBA=1
python -c "from geodepoly.util import poly_eval; print(poly_eval([1,0,-7,6], 2.0))"
```

Notes:
- Speeds up `poly_eval` and `poly_eval_dp_ddp` (used inside finishers like Aberth/Polish).
- Falls back to pure Python if Numba is not installed.

Solver usage with Numba on:

```bash
export GEODEPOLY_USE_NUMBA=1
python -c "from geodepoly import solve_all; print(solve_all([-6,11,-6,1]))"
```

## Batched and GPU

- See `docs/batched.md` and `scripts/bench_gpu_prototype.py` for vectorized Newton steps.
- Torch/JAX backends can leverage GPU if available for batched kernels.
