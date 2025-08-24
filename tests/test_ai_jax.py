import pytest

def jax_or_skip():
	try:
		import jax
		import jax.numpy as jnp
		return jax, jnp
	except Exception:
		pytest.skip("JAX not installed", allow_module_level=True)


def test_rootlayer_jax_backward_runs():
	jax, jnp = jax_or_skip()
	from geodepoly.ai import root_solve_jax

	B, N = 2, 2
	coeffs = jnp.asarray(jnp.random.randn(B, N + 1) + 1j * jnp.random.randn(B, N + 1), dtype=jnp.complex128)

	def loss_fn(c):
		roots = root_solve_jax(c)
		return jnp.mean(jnp.square(jnp.real(roots)))

	val, grad = jax.value_and_grad(loss_fn)(coeffs)
	assert jnp.isfinite(val)
	assert grad.shape == coeffs.shape
