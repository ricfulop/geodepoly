# GeodeBench (skeleton)

Goal: a small dataset probing symmetry/generalization across Catalan/Fuss/Geode slices, reporting residuals/time for root-finding tasks.

## Slices
- Catalan slice: quadratic-only (t2) coefficients
- Fuss slice: fixed d-gon pattern
- Geode slice: layered hyper-Catalan counts

## Tasks
- Solve polynomial instances per slice and report:
  - max residual, runtime, success (<1e-8)

## Splits
- Degrees: {3,5,8,12}
- Trials per degree: configurable

## Metrics
- Median residual, median time, success rate by (slice, degree)
