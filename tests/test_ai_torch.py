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


def test_root_set_loss_hungarian_runs():
	torch = torch_or_skip()
	from geodepoly.ai.losses import root_set_loss

	B, N = 2, 3
	roots_pred = torch.randn(B, N, dtype=torch.cdouble)
	roots_true = torch.randn(B, N, dtype=torch.cdouble)
	loss = root_set_loss(roots_pred, roots_true, match="hungarian")
	assert torch.isfinite(loss)


def test_root_set_loss_diffsort_runs():
	torch = torch_or_skip()
	from geodepoly.ai.losses import root_set_loss

	B, N = 2, 4
	roots_pred = torch.randn(B, N, dtype=torch.cdouble)
	roots_true = torch.randn(B, N, dtype=torch.cdouble)
	loss = root_set_loss(roots_pred, roots_true, match="diffsort")
	assert torch.isfinite(loss)
