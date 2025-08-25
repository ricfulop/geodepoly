# AI Benchmarks

This page shows how to run AI-oriented benchmarks for the differentiable RootLayer paths.

## Torch (CPU/CUDA)

```bash
python scripts/bench_ai_torch.py --batch 512 --degree 16 --device cpu
# GPU (if available)
python scripts/bench_ai_torch.py --batch 2048 --degree 24 --device cuda
```

## JAX (CPU/GPU/TPU)

```bash
python scripts/bench_ai_jax.py --batch 512 --degree 16
```

Both scripts report forward and backward times. Adjust batch/degree based on memory.

## Results (placeholders)

| Backend | Device | Batch | Degree | Forward (s) | Backward (s) |
| --- | --- | --- | --- | --- | --- |
| Torch | CPU | 512 | 16 | 23.0536 | 0.0951 |
| Torch | GPU | 2048 | 24 | tbd | tbd |
| JAX | CPU | 512 | 16 | 0.0304 | 1.1731 |
| JAX | GPU | 4096 | 24 | tbd | tbd |

To reproduce: use the commands above. Submit PRs to update these rows with your hardware.
