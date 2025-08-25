# geodepoly v0.2.4

Small version bump to publish recent fixes and improvements:
- Per-backend batched kernels (NumPy/Torch/JAX) with clearer typing and MPS-friendly ops
- JAX tests updated to use `jax.random`; Torch diffsort loss fixed for complex tensors
- JAX geode convolution indexing made portable

No API changes.
