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
	key = jax.random.PRNGKey(0)
	r1 = jax.random.normal(key, (B, N + 1))
	key, sub = jax.random.split(key)
	r2 = jax.random.normal(sub, (B, N + 1))
	coeffs = (r1 + 1j * r2).astype(jnp.complex128)

	def loss_fn(c):
		roots = root_solve_jax(c)
		return jnp.mean(jnp.square(jnp.real(roots)))

	val, grad = jax.value_and_grad(loss_fn)(coeffs)
	assert jnp.isfinite(val)
	assert grad.shape == coeffs.shape


def test_rootlayer_jax_jit_and_vmap():
	jax, jnp = jax_or_skip()
	from geodepoly.ai import root_solve_jax

	B, N = 3, 3
	key = jax.random.PRNGKey(1)
	r1 = jax.random.normal(key, (B, N + 1))
	key, sub = jax.random.split(key)
	r2 = jax.random.normal(sub, (B, N + 1))
	coeffs = (r1 + 1j * r2).astype(jnp.complex128)

	@jax.jit
	def f(c):
		return root_solve_jax(c)

	roots = f(coeffs)
	assert roots.shape == (B, N)


def test_rootlayer_jax_jit_with_grad():
	jax, jnp = jax_or_skip()
	from geodepoly.ai import root_solve_jax

	B, N = 2, 3
	key = jax.random.PRNGKey(2)
	r1 = jax.random.normal(key, (B, N + 1))
	key, sub = jax.random.split(key)
	r2 = jax.random.normal(sub, (B, N + 1))
	coeffs = (r1 + 1j * r2).astype(jnp.complex128)

	def loss_fn(c):
		roots = root_solve_jax(c)
		return jnp.mean(jnp.square(jnp.real(roots)))

	jitted = jax.jit(jax.value_and_grad(loss_fn))
	val, grad = jitted(coeffs)
	assert jnp.isfinite(val)
	assert grad.shape == coeffs.shape
