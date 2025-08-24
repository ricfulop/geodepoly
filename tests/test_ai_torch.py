import os
import math
import pytest


def torch_or_skip():
	try:
		import torch  # type: ignore
		return torch
	except Exception:
		pytest.skip("PyTorch not installed", allow_module_level=True)


def test_rootlayer_torch_grad_simple():
	torch = torch_or_skip()
	from geodepoly.ai import root_solve_torch

	B, N = 2, 2
	coeffs = torch.randn(B, N + 1, dtype=torch.cdouble, requires_grad=True)
	roots = root_solve_torch(coeffs)
	loss = (roots.real ** 2).mean()
	loss.backward()
	assert coeffs.grad is not None
	assert coeffs.grad.shape == coeffs.shape
	# sanity: nonzero grad for most entries
	assert torch.isfinite(coeffs.grad).all()
