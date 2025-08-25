# Cubic and Quintic Approximants

## Cubic one-line approximant Q(t2, t3)

The Bi–Tri slice admits a compact cubic approximant:

```python
from geodepoly.geode import Q_cubic
Q = Q_cubic(t2=0.1, t3=-0.03)
```

This provides a fast low-order estimate on the `t2,t3` plane.

## Hybrid-cubic mode

`method="hybrid-cubic"` in `solve_all` uses `Q_cubic` as a warm-start heuristic:

1) Recenter and compute `t=-a0/a1`, map local `a2,a3` to `t2,t3`.  
2) Evaluate `alpha = Q_cubic(t2,t3)` and take `y = t * alpha` as a step.  
3) Repeat briefly across a few centers; pass seeds to Aberth for finishing.

This can reduce time-to-first-root on many instances.

## Quintic (Bring radical) note

The Bring radical/Eisenstein series is part of the theoretical framework. An explicit `bring_radical_series` API may be added in a future release.
