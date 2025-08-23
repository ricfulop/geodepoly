# Geode GPU Roadmap (draft)

## Goals
- Batched root solves for many polynomials
- Stable kernels for Aberth/Halley updates

## Tiling & Layout
- Face-layer tiling aligned with hyper-Catalan layers
- Structure-of-arrays layout for complex ops

## Prototype
- JAX/NumPy sketch for vectorized series seeds and Aberth steps

## Next steps
- Microbenchmarks and accuracy checks vs CPU
- Memory-bound vs compute-bound analysis
