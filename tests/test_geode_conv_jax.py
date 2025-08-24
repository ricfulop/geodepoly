import pytest


def jax_or_skip():
    try:
        import jax  # noqa: F401
        import jax.numpy as jnp  # noqa: F401
        return True
    except Exception:
        pytest.skip("JAX not installed", allow_module_level=True)


def test_geode_convolution_jax_basic():
    assert jax_or_skip()
    from geodepoly.geode_conv_jax import geode_convolution_jax

    A = {(): 1.0 + 0.0j, (1,): 1.0 + 0.0j}
    B = {(): 1.0 + 0.0j, (1,): 2.0 + 0.0j}
    C = geode_convolution_jax(A, B, num_vars=1, max_weight=4)
    assert abs(C.get((), 0) - 1.0) < 1e-12
    assert abs(C.get((1,), 0) - 3.0) < 1e-12
    assert abs(C.get((2,), 0) - 2.0) < 1e-12


