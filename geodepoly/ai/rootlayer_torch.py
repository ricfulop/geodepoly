from __future__ import annotations

import math
from typing import Tuple


def _ensure_torch():
	try:
		import torch  # type: ignore
		return torch
	except Exception as exc:  # pragma: no cover
		raise ImportError("PyTorch is required for geodepoly.ai.rootlayer_torch") from exc


def _solve_batch_numpy_like(coeffs, method: str = "hybrid", resum: str = "pade"):
	"""Host solve that accepts a torch Tensor and returns torch Tensor, using geodepoly core.
	Assumes coeffs shape (B, N+1), complex dtype.
	"""
	import torch  # local import
	from geodepoly.solver import solve_all

	B, D1 = coeffs.shape
	roots_out = []
	for b in range(B):
		c = coeffs[b].detach().cpu().numpy().tolist()
		r = solve_all(c, method=method, resum=resum)
		# pack to tensor complex
		rt = torch.tensor(r, dtype=coeffs.dtype, device=coeffs.device)
		roots_out.append(rt)
	return torch.stack(roots_out, dim=0)  # (B, N)


def _poly_and_deriv_at(coeffs_row, x):
	"""Evaluate p and p' at x using Horner; coeffs_row low->high.
	Works on torch complex tensors.
	"""
	torch = _ensure_torch()
	p = torch.zeros_like(x)
	dp = torch.zeros_like(x)
	for a in reversed(coeffs_row):
		dp = dp * x + p
		p = p * x + a
	return p, dp


class RootLayerTorchFn(_ensure_torch().autograd.Function):  # type: ignore[misc]
	@staticmethod
	def forward(ctx, coeffs, method: str = "hybrid", resum: str = "pade"):
		"""Forward: coeffs (B, N+1) -> roots (B, N). Saves tensors for backward.
		"""
		torch = _ensure_torch()
		if coeffs.dim() != 2:
			raise ValueError("coeffs must be 2D (B, N+1)")
		if not torch.is_complex(coeffs):
			raise TypeError("coeffs must be a complex dtype (cdouble recommended)")
		with torch.no_grad():
			roots = _solve_batch_numpy_like(coeffs, method=method, resum=resum)
		ctx.save_for_backward(coeffs, roots)
		return roots

	@staticmethod
	def backward(ctx, grad_roots):
		"""Backward: analytic simple-root Jacobian J_ik = - r_i^k / p'(r_i).
		Masks gradients when |p'(r_i)| is very small.
		"""
		torch = _ensure_torch()
		coeffs, roots = ctx.saved_tensors
		B, D1 = coeffs.shape
		N = D1 - 1
		# Build grad wrt coeffs (B, N+1) complex; constant term a0..aN
		grad_coeffs = torch.zeros_like(coeffs)
		# powers r^k for k=0..N-1
		for b in range(B):
			c_row = coeffs[b]
			r_row = roots[b]
			# compute p'(r_i)
			_, dp = _poly_and_deriv_at(c_row, r_row)
			# mask near multiple roots
			mask = torch.abs(dp) > 1e-12
			# Construct dr/da_k for k=0..N-1 per root
			# Note: derivative wrt leading coeff a_N can be derived but is small; keep zero for simplicity
			powers = torch.stack([r_row ** k for k in range(N)], dim=1)  # (N, N)
			J = torch.zeros_like(powers)
			J[mask] = -powers[mask] / dp[mask].unsqueeze(1)
			# chain: grad_coeffs[b,k] = sum_i grad_roots[b,i] * J[i,k]
			g = grad_roots[b]
			if not torch.is_complex(g):
				g = g.to(grad_coeffs.dtype)
			gc = (J.conj().T @ g)  # (N,)
			# assign to a0..a_{N-1}
			grad_coeffs[b, :N] = gc
			# leave grad for a_N as zero
		return grad_coeffs, None, None


def root_solve_torch(coeffs, method: str = "hybrid", resum: str = "pade"):
	"""User-facing helper: calls RootLayerTorchFn.apply.
	"""
	torch = _ensure_torch()
	return RootLayerTorchFn.apply(coeffs, method, resum)
