# Six AI Use Cases with GeodePoly

This page summarizes six practical ways AI can leverage GeodePoly.

## 1) Differentiable Root Layer (inside neural nets)
- Custom autograd op mapping coefficients → roots.
- Gradients for simple roots: ∂r_i/∂a_k = − r_i^k / p'(r_i).
- Tips: mask/damp gradients if |p'(r)| is tiny; polish roots in forward.

## 2) GeodeBench: symmetry-generalization tasks
- Dataset from Geode/Hyper‑Catalan arrays.
- Tasks: coefficient prediction, Geode factor recovery (S→G), invariance under shift/scale.

## 3) Inductive‑bias layers that “speak polynomial”
- Geode convolution (face‑layer accumulation), power‑basis attention, root‑space heads.

## 4) Training objectives in root space
- Pole placement (half‑plane margins), spectral radius, root‑set matching.

## 5) Neuro‑symbolic loops & program synthesis
- Generate → verify with GeodePoly → refine.
- Enforce Viète identities as a regularizer.

## 6) GPU/Compiler target (batched kernels)
- Batched kernels (JAX/XLA, CUDA) and differentiable RootLayer for large batches.

See `examples/ai/` for runnable demos and the AI overview page for install and quickstart.
