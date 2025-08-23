# Benchmarks

- Run: `python scripts/bench_compare.py --degrees 3,5,8,12 --methods hybrid,aberth,dk --trials 10 --out docs/assets/bench.csv --agg_out docs/assets/bench_agg.csv`
- Plot: `python scripts/plot_bench.py --in docs/assets/bench_agg.csv --out docs/assets`

![Time vs Degree](assets/time_vs_degree.png)

![Residual vs Degree](assets/residual_vs_degree.png)
