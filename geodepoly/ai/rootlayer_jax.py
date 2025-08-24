from __future__ import annotations

from typing import Tuple


def _ensure_jax():
	try:
		import jax
		import jax.numpy as jnp
		return jax, jnp
	except Exception as exc:  # pragma: no cover
		raise ImportError("JAX is required for geodepoly.ai.rootlayer_jax") from exc


def _pure_jax_solve(coeffs, iters: int = 50, damping: float = 0.8):
	"""Pure-JAX batched Aberth solve.

	coeffs: jnp array (B, N+1) complex128 low->high
	returns: jnp array (B, N) complex128 roots
	"""
	from geodepoly.batched import jax_aberth_solve

	B, D1 = coeffs.shape
	N = D1 - 1
	# Cauchy-like radius per batch: 1 + max |a_k|/|a_n|
	an = jnp.abs(coeffs[:, -1])
	an = jnp.where(an == 0, jnp.ones_like(an), an)
	max_ratio = jnp.max(jnp.abs(coeffs[:, :-1]) / an[:, None], axis=1)
	R = 1.0 + max_ratio
	# initial N points on circle per batch
	k = jnp.arange(N, dtype=jnp.float32)
	theta = 2 * jnp.pi * (k / N)
	unit = jnp.exp(1j * theta.astype(coeffs.dtype))  # (N,)

	def solve_row(c_row, r_scalar):
		z0 = (r_scalar.astype(coeffs.dtype)) * unit  # (N,)
		z = jax_aberth_solve(c_row, z0, iters=iters, damping=damping)
		return z

	roots = jax.vmap(solve_row)(coeffs, R)
	return roots


jax, jnp = _ensure_jax()


@jax.custom_vjp
def root_solve_jax(coeffs, method: str = "hybrid", resum: str = "pade"):
	return _pure_jax_solve(coeffs)



def _fwd(coeffs, method: str = "hybrid", resum: str = "pade"):
	roots = _pure_jax_solve(coeffs)
	return roots, (coeffs, roots)


def _bwd(res, g):
	coeffs, roots = res
	B, D1 = coeffs.shape
	N = D1 - 1
	# Compute p'(r) and powers r^k on the fly
	def poly_and_deriv_at_row(c_row, r_row):
		p = jnp.zeros_like(r_row)
		dp = jnp.zeros_like(r_row)
		for a in c_row[::-1]:
			dp = dp * r_row + p
			p = p * r_row + a
		return p, dp

	def grad_row(c_row, r_row, g_row):
		_, dp = poly_and_deriv_at_row(c_row, r_row)
		mask = jnp.abs(dp) > 1e-12
		k = jnp.arange(N)
		powers = r_row[:, None] ** k[None, :]  # (N, N)
		J = jnp.where(mask[:, None], -powers / dp[:, None], 0.0 + 0.0j)
		gc = J.conj().T @ g_row
		# grad wrt a0..a_{N-1}; a_N left as zero
		return jnp.concatenate([gc, jnp.zeros((1,), dtype=gc.dtype)])

	grad_coeffs = jax.vmap(grad_row)(coeffs, roots, g)
	return (grad_coeffs, None, None)


root_solve_jax.defvjp(_fwd, _bwd)
