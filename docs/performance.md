# Performance Notes

## Numba acceleration (opt-in)

Enable JIT for core polynomial evaluation by setting an environment variable before import:

```bash
export GEODEPOLY_USE_NUMBA=1
python -c "from geodepoly.util import poly_eval; print(poly_eval([1,0,-7,6], 2.0))"
```

Notes:
- Speeds up `poly_eval` and `poly_eval_dp_ddp` when available.
- Falls back to pure Python if Numba is not installed.

## Batched and GPU

- See `docs/batched.md` and `scripts/bench_gpu_prototype.py` for vectorized Newton steps.
- Torch/JAX backends can leverage GPU if available for batched kernels.
