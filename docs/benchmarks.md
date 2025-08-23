# Benchmarks

- Run: `python scripts/bench_compare.py --degrees 3,5,8,12 --methods hybrid,aberth,dk --trials 10 --out docs/assets/bench.csv --agg_out docs/assets/bench_agg.csv`
- Plot: `python scripts/plot_bench.py --in docs/assets/bench_agg.csv --out docs/assets`

![Time vs Degree](assets/time_vs_degree.png)

![Residual vs Degree](assets/residual_vs_degree.png)

## Newton vs hybrid (tuned)

Summary (medians over 5 trials):

| degree | method | time_median(s) | res_median |
| --- | --- | ---: | ---: |
| 3 | hybrid | 0.0027 | 2.48e-16 |
| 3 | newton | 0.0000 | 3.33e-16 |
| 5 | hybrid | 0.0060 | 2.22e-15 |
| 5 | newton | 0.0000 | 1.89e-15 |
| 8 | hybrid | 0.0142 | 1.22e-14 |
| 8 | newton | 0.0001 | 8.30e-15 |
| 12 | hybrid | 0.0314 | 6.17e-13 |
| 12 | newton | 0.0003 | 6.17e-13 |
| 16 | hybrid | 0.0796 | 5.60e-11 |
| 16 | newton | 0.0017 | 2.70e-11 |
| 20 | hybrid | 0.1191 | 1.91e-10 |
| 20 | newton | 0.0027 | 3.19e-10 |

CSV: `docs/assets/newton_vs_hybrid_tuned.csv`

## Edge cases: iteration profile

We measure time, residual, and (for Newton) total iteration count on a few tougher instances (clusters, ill‑scaled, |t|≈1). Hybrid and Aberth reach the target with minimal polish; Newton iteration counts can spike.

| case | hybrid time(s) | aberth time(s) | newton time(s) | newton iters | residual (≈) |
| --- | ---: | ---: | ---: | ---: | ---: |
| double_root | 0.0041 | 0.0005 | 0.0001 | 87 | 1e-16–1e-23 |
| cluster_eps_1e-12 | 0.0038 | 0.0005 | 0.0001 | 96 | 1e-38–1e-25 |
| cluster_eps_1e-10 | 0.0038 | 0.0005 | 0.0001 | 96 | 1e-39–1e-25 |
| cluster_eps_1e-08 | 0.0035 | 0.0005 | 0.0001 | 95 | 1e-16–1e-48 |
| cluster_eps_1e-06 | 0.0034 | 0.0005 | 0.0001 | 81 | 1e-41–1e-60 |
| cluster_eps_1e-4 | 0.0030 | 0.0006 | 0.0001 | 63 | 1e-16–1e-60 |
| ill_scaled | 0.0036 | 0.0007 | 0.0001 | 68 | 1e-14 |
| rotated_cluster | 0.0071 | 0.0012 | 0.0002 | 320 | 1e-16 |
| near_t_one | 0.0032 | 0.0009 | 0.0001 | 72 | 1e-16 |

CSV: `docs/assets/edge_cases.csv`
