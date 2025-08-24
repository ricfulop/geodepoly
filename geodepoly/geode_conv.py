from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


Monomial = Tuple[int, ...]


def weighted_degree(m: Monomial) -> int:
    return sum((i + 2) * e for i, e in enumerate(m))


def shape_for(num_vars: int, max_weight: int) -> Tuple[int, ...]:
    # axis i corresponds to t_{i+2}
    return tuple(max_weight // (i + 2) + 1 for i in range(num_vars))


def dict_to_array(
    coeffs: Dict[Monomial, complex], num_vars: int, max_weight: int
) -> np.ndarray:
    shp = shape_for(num_vars, max_weight)
    arr = np.zeros(shp, dtype=complex)
    for m, c in coeffs.items():
        if len(m) > num_vars:
            continue
        # pad monomial to num_vars
        exps = list(m) + [0] * (num_vars - len(m))
        if weighted_degree(tuple(exps)) <= max_weight:
            # bounds check per axis
            ok = True
            for i, e in enumerate(exps):
                if e > max_weight // (i + 2):
                    ok = False
                    break
            if ok:
                arr[tuple(exps)] = arr[tuple(exps)] + complex(c)
    return arr


def array_to_dict(arr: np.ndarray, max_weight: int) -> Dict[Monomial, complex]:
    num_vars = arr.ndim
    out: Dict[Monomial, complex] = {}
    for idx in np.ndindex(arr.shape):
        m = tuple(int(e) for e in idx)
        if weighted_degree(m) <= max_weight:
            val = complex(arr[idx])
            if val != 0:
                exps = list(m)
                while exps and exps[-1] == 0:
                    exps.pop()
                out[tuple(exps)] = out.get(tuple(exps), 0.0 + 0.0j) + val
    return out


def conv_nd_fft(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # full convolution using FFT along all axes
    out_shape = tuple(da + db - 1 for da, db in zip(a.shape, b.shape))
    fa = np.fft.fftn(a, s=out_shape)
    fb = np.fft.fftn(b, s=out_shape)
    fc = fa * fb
    c = np.fft.ifftn(fc)
    return c


def crop_by_weight(arr: np.ndarray, max_weight: int) -> np.ndarray:
    # Zero out entries with weighted degree above max_weight; crop to original shape
    out = arr.copy()
    for idx in np.ndindex(out.shape):
        m = tuple(int(e) for e in idx)
        if weighted_degree(m) > max_weight:
            out[idx] = 0
    return out


def geode_convolution_dict(
    A: Dict[Monomial, complex],
    B: Dict[Monomial, complex],
    num_vars: int,
    max_weight: int,
) -> Dict[Monomial, complex]:
    a = dict_to_array(A, num_vars=num_vars, max_weight=max_weight)
    b = dict_to_array(B, num_vars=num_vars, max_weight=max_weight)
    c_full = conv_nd_fft(a, b)
    # crop back to original array shape (same as a/b shapes)
    crop_slices = tuple(slice(0, s) for s in a.shape)
    c = c_full[crop_slices]
    c = crop_by_weight(c, max_weight=max_weight)
    return array_to_dict(c, max_weight=max_weight)


def geode_convolution_sequence(
    seed: Dict[Monomial, complex],
    kernels: Sequence[Dict[Monomial, complex]],
    num_vars: int,
    max_weight: int,
) -> Dict[Monomial, complex]:
    out = dict(seed)
    for k in kernels:
        out = geode_convolution_dict(out, k, num_vars=num_vars, max_weight=max_weight)
    return out


