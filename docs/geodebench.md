# GeodeBench v0 (MVP)

- S slice (quadratic): `t2` only; target `alpha`.
- G slice (multivariate): small `t2,t3` nonzero; target `alpha`.

Tasks:
- Coefficient prediction: recover truncated coefficients from samples.
- Geode recovery: predict `alpha` from slice inputs.
- Invariance checks: evaluate symmetry/generalization splits.

Generate data:
```bash
python bench/generate_slices.py --degrees 3,5,8 --trials 10 --out docs/assets/geode_slices.csv
```

Starter notebook: `notebooks/GeodeBench_Starter.ipynb`
