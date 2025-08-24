from __future__ import annotations

from typing import Tuple


def _ensure_jax():
	try:
		import jax
		import jax.numpy as jnp
		return jax, jnp
	except Exception as exc:  # pragma: no cover
		raise ImportError("JAX is required for geodepoly.ai.rootlayer_jax") from exc


def _host_solve(coeffs, method: str = "hybrid", resum: str = "pade"):
	"""Call geodepoly host solver; coeffs is jnp array (B, N+1) complex128.
	Returns jnp array (B, N) complex128.
	"""
	import numpy as np
	from geodepoly.solver import solve_all

	B, D1 = coeffs.shape
	roots = []
	arr = np.asarray(coeffs)
	for b in range(B):
		c = arr[b].tolist()
		r = solve_all(c, method=method, resum=resum)
		roots.append(np.asarray(r, dtype=arr.dtype))
	return np.stack(roots, axis=0)


jax, jnp = _ensure_jax()


@jax.custom_vjp
def root_solve_jax(coeffs, method: str = "hybrid", resum: str = "pade"):
	roots = jnp.asarray(_host_solve(coeffs, method=method, resum=resum))
	return roots


def _fwd(coeffs, method: str = "hybrid", resum: str = "pade"):
	roots = root_solve_jax(coeffs, method, resum)
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
