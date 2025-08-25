# Layerings

Layerings provide partial sums of the Hyper-Catalan series by different combinatorial levels.

- Vertex layering: constraint `sum m_k <= L`
- Edge layering: constraint `sum (k-1) m_k <= L`
- Face layering: constraint `1 + sum (k-1) m_k <= 1+L`

## APIs

- `geodepoly.layerings.vertex_layering(t: Mapping[int,complex], Vmax: int) -> list[complex]`
- `geodepoly.layerings.edge_layering(t: Mapping[int,complex], Emax: int) -> list[complex]`
- `geodepoly.layerings.face_layering(t: Mapping[int,complex], Fmax: int) -> list[complex]`

## Example

```python
from geodepoly import vertex_layering, edge_layering, face_layering

vals = {2: 0.1, 3: 0.02}
SV = vertex_layering(vals, Vmax=5)
SE = edge_layering(vals, Emax=5)
SF = face_layering(vals, Fmax=5)
```

`SV[L]`, `SE[L]`, `SF[L]` return the truncated sums at each layering level.

## OEIS note

On the `t2`-only slice, the vertex-layered series recovers the Catalan numbers as coefficients:
`S(t2) = 1 + C1 t2 + C2 t2^2 + ...` with `C_n` Catalan. Our tests confirm the first few by finite differences.
