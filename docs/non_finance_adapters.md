# Non-finance Adapters

Domain-oriented convenience wrappers built on top of the core solver.

## Controls: characteristic polynomial roots

```python
from geodepoly.adapters import charpoly_roots
roots = charpoly_roots([-6, 11, -6, 1])
```

## Signals: AR process poles

```python
from geodepoly.adapters import ar_roots
poles = ar_roots([0.7, -0.2, 0.1])  # AR(3): 1 + 0.7 y + -0.2 y^2 + 0.1 y^3
```

## Vision: invert radial distortion

```python
from geodepoly.adapters import invert_radial
ru = invert_radial(rp=0.8, k1=-0.3, k2=0.05)
```

## Geometry: ray-quartic intersections

```python
from geodepoly.adapters import ray_intersect_quartic
hits = ray_intersect_quartic([c0, c1, c2, c3, c4])  # t>=0 real roots
```

Refer to the API docs for argument conventions and return types.
