from __future__ import annotations

from typing import Dict, Tuple

import jax
import jax.numpy as jnp

Monomial = Tuple[int, ...]


def weighted_degree(m: Monomial) -> int:
    return sum((i + 2) * e for i, e in enumerate(m))


def shape_for(num_vars: int, max_weight: int) -> Tuple[int, ...]:
    return tuple(max_weight // (i + 2) + 1 for i in range(num_vars))


def dict_to_array_jax(
    coeffs: Dict[Monomial, complex], num_vars: int, max_weight: int
) -> jnp.ndarray:
    shp = shape_for(num_vars, max_weight)
    arr = jnp.zeros(shp, dtype=jnp.complex128)
    # scatter updates (small sizes expected for docs/examples)
    for m, c in coeffs.items():
        exps = list(m) + [0] * (num_vars - len(m))
        if weighted_degree(tuple(exps)) <= max_weight:
            if all(e <= max_weight // (i + 2) for i, e in enumerate(exps)):
                arr = arr.at[tuple(exps)].add(jnp.asarray(c))
    return arr


def array_to_dict_jax(arr: jnp.ndarray, max_weight: int) -> Dict[Monomial, complex]:
    out: Dict[Monomial, complex] = {}
    for idx in jnp.ndindex(arr.shape):
        m = tuple(int(e) for e in idx)
        if weighted_degree(m) <= max_weight:
            val = complex(arr[idx])
            if val != 0:
                exps = list(m)
                while exps and exps[-1] == 0:
                    exps.pop()
                out[tuple(exps)] = out.get(tuple(exps), 0.0 + 0.0j) + val
    return out


def conv_nd_fft_jax(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    out_shape = tuple(da + db - 1 for da, db in zip(a.shape, b.shape))
    # specify axes explicitly for future numpy/jax compat
    axes = tuple(range(len(out_shape)))
    fa = jnp.fft.fftn(a, s=out_shape, axes=axes)
    fb = jnp.fft.fftn(b, s=out_shape, axes=axes)
    fc = fa * fb
    c = jnp.fft.ifftn(fc, axes=axes)
    return c


def crop_by_weight_jax(arr: jnp.ndarray, max_weight: int) -> jnp.ndarray:
    def mask_fn(idx):
        return weighted_degree(tuple(int(i) for i in idx)) <= max_weight
    # Build boolean mask by iterating indices (small arrays expected)
    mask = jnp.zeros_like(arr, dtype=bool)
    for idx in jnp.ndindex(arr.shape):
        keep = mask_fn(idx)
        mask = mask.at[idx].set(keep)
    return jnp.where(mask, arr, 0)


def geode_convolution_jax(
    A: Dict[Monomial, complex],
    B: Dict[Monomial, complex],
    num_vars: int,
    max_weight: int,
) -> Dict[Monomial, complex]:
    a = dict_to_array_jax(A, num_vars=num_vars, max_weight=max_weight)
    b = dict_to_array_jax(B, num_vars=num_vars, max_weight=max_weight)
    c_full = conv_nd_fft_jax(a, b)
    crop_slices = tuple(slice(0, s) for s in a.shape)
    c = c_full[crop_slices]
    c = crop_by_weight_jax(c, max_weight=max_weight)
    return array_to_dict_jax(c, max_weight=max_weight)


