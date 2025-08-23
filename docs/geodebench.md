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

Baseline:
```bash
python scripts/baseline_transformer.py --in docs/assets/geode_slices.csv
# Example output: Baseline mean abs error: 4.44e-16
```

## Leaderboard (v0 preview)

| Method                 | Split            | Metric        | Score    |
|------------------------|------------------|---------------|----------|
| Linear baseline        | Random (S+G)     | MAE(|alpha|)  | ~4.4e-16 |
| Naive OEIS-style       | Symmetry holdout | MAE(|alpha|)  | TBA      |
| Tiny Transformer (stub)| Symmetry holdout | MAE(|alpha|)  | TBA      |

