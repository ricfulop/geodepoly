# Troubleshooting

### Residuals aren’t small enough
- Increase `refine_steps` (Halley polish) or switch finisher (`aberth`/`dk`).
- Use `resum="auto"` for series seeds if |t| is near 1.
- Beware nearly-multiple roots: reduce tolerance or try different centers.

### Divergence or NaNs
- Ensure coefficients are finite and leading coefficient is nonzero.
- Try `method="dk"` (derivative-free) for pathological cases.

### SymPy import errors
- Install: `pip install sympy` and retry `sympy_solve` examples.

### Performance
- Reduce degree/trials in benchmarks; set `boots` lower; use `numpy` method for a baseline.
