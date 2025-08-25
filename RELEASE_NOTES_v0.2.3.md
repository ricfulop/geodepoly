# geodepoly v0.2.3

## Highlights
- Cleanup release to unblock publishing: resolves static typing issues in `geodepoly/batched.py` by splitting NumPy/Torch/JAX execution paths.
- Maintains behavior while improving compatibility with type checkers and Torch MPS.

## Changes
- Per-backend implementations for `batched_poly_eval`, `batched_newton_step`, and `batched_solve_all`.
- Torch paths use `torch.flip` and fused `addcmul`; JAX paths use `lax.scan`.
- Minor documentation/bench polish.

## Notes
- Wheels/CI will pick this up via the v0.2.3 tag. If publishing from tags, ensure GitHub Actions are enabled.
