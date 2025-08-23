# Geode GPU Roadmap (draft)

## Goals
- Batched root solves for many polynomials
- Stable kernels for Aberth/Halley updates

## Tiling & Layout
- Face-layer tiling aligned with hyper-Catalan layers
- Structure-of-arrays layout for complex ops

## Prototype
- JAX/NumPy/Torch sketch for vectorized Newton steps

### Usage

```bash
# NumPy (CPU):
python scripts/bench_gpu_prototype.py --backend numpy --batch 4096 --degree 16 --steps 50

# Torch (GPU if available, else CPU):
python scripts/bench_gpu_prototype.py --backend torch --batch 4096 --degree 16 --steps 50 --device cuda

# JAX (CPU/GPU depending on install):
python scripts/bench_gpu_prototype.py --backend jax --batch 4096 --degree 16 --steps 50
```

Outputs throughput (steps/sec) and residual sanity check.

### Aberth (vectorized; Torch)

```bash
# Compare CPU vs GPU for a single Aberth update loop
python scripts/bench_aberth_gpu.py --deg 32 --iters 50
```

Developer note: a single-step `torch_aberth_step(coeffs, roots, damping)` is available in `geodepoly.batched`.

### JAX variant

```bash
python scripts/bench_aberth_jax.py --deg 32 --iters 50
```

Developer note: JAX helpers `jax_aberth_step/solve` are available in `geodepoly.batched`.

## Next steps
- Microbenchmarks and accuracy checks vs CPU
- Memory-bound vs compute-bound analysis
