# Geode factorization

This page summarizes the Geode identity S - 1 = S1 · G and shows how to use the public APIs.

- S: Hyper-Catalan generating function in variables `t2, t3, ...`
- S1: linear part `∑ t_k`
- G: residual factor so that `S-1 = S1·G`

## APIs

- `geodepoly.geode.S_eval(t: Mapping[int,complex], Fmax: int, use_geode: bool=False)`
- `geodepoly.geode.eval_S_via_geode(t, Fmax)`
- `geodepoly.geode.map_t_from_poly(coeffs)`
- `geodepoly.geode.SeriesOptions`
- `geodepoly.geode.solve_series(coeffs, opts)`

## Examples

Evaluate S directly vs. via Geode factoring:

```python
from geodepoly.geode import S_eval, eval_S_via_geode

# only t2,t3 nonzero
vals = {2: 0.1, 3: -0.03}
S_direct = S_eval(vals, Fmax=8, use_geode=False)
S_geode  = eval_S_via_geode(vals, Fmax=8)
```

Map polynomial coefficients to t_k, then evaluate S to get the soft root step variable:

```python
from geodepoly.geode import map_t_from_poly, S_eval

coeffs = [-6, 11, -6, 1]  # a0..a3 (monic cubic with roots 1,2,3)
t = map_t_from_poly(coeffs)
alpha = S_eval(t, Fmax=8)  # alpha(0)=1 branch
```

Series bootstrap to get a single root candidate:

```python
from geodepoly.geode import SeriesOptions, solve_series

opts = SeriesOptions(Fmax=16, bootstrap=True, bootstrap_passes=3)
x = solve_series([-6, 11, -6, 1], opts)
```

Notes
- Larger `Fmax` gives more accurate truncations but higher cost.
- The Geode path is often more stable near the linearization boundary.
