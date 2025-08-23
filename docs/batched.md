# Batched kernels (MVP3)

- Backends: NumPy (reference), Torch, JAX.
- Ops:
  - `batched_poly_eval(coeffs, xs, backend)`
  - `batched_newton_step(coeffs, xs, backend)`
  - `torch_root_layer(steps, tol)` differentiable Newton layer

Bench (preview): expect 10–100× throughput vs pure Python on large batches (Torch/JAX on GPU).
