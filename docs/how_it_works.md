# How it works (deep-dive)

This section summarizes the core algorithmic choices behind geodepoly and connects them to the paper.

## 1) Recenter and series seed

Given coefficients low-to-high `p(x) = a0 + a1 x + ... + aN x^N`, choose a center `μ` and expand
`q(y) = p(μ + y) = a0 + a1 y + a2 y^2 + ...` via binomial shifting. If `a1 ≠ 0`, set `t = -a0/a1`.

We invert `F(y) = y + β2 y^2 + β3 y^3 + ...` with `βk = ak/a1` using Lagrange inversion to obtain
inverse-series coefficients `{g_m}` for the local root increment `y ≈ Σ g_m t^m`.

## 2) Resummation and auto selection

Truncated series can be fragile near their convergence boundary (|t| ~ 1). We provide:
- Plain: direct Horner evaluation of Σ g_m t^m
- Padé: near-diagonal Padé rational approximation
- Borel–Padé: Pade of the Borel transform followed by Gauss–Laguerre
- Auto: small Padé grid scored for stability, falling back to Borel–Padé or plain

Use `resum="auto"` for robust default behavior.

## 3) Bootstrap and deflation

A few bootstrap steps update the center `μ ← μ + y`. When a root is good, synthetic division (deflation)
reduces degree. The MVP solves all roots with a robust finisher after obtaining one or two good seeds.

## 4) Finishers and polishing

- Aberth–Ehrlich: simultaneous iteration with adaptive damping and minimal repulsion for clustered roots
- Durand–Kerner: derivative-free simultaneous method
- Halley polishing: applied per root; we also provide multiplicity-aware Halley when a root appears repeated

## 5) Multiple/clustered roots

We estimate local multiplicity using `m̂ ≈ Re( p * p'' / p'^2 )` and apply a multiplicity-aware Halley update
when `m̂ ≥ 2`. Clustered roots benefit from Aberth damping and repulsion.

## 6) Hyper-Catalan connection

The generating series `S[t2,t3,...]` with hyper-Catalan coefficients solves a canonical geometric polynomial
and motivates the series-reversion view. We expose utilities to evaluate slices of `S` and to compute
coefficients for exploration and benchmarks.

### Geode factorization

We construct `S`, its linear part `S1 = Σ t_k`, and a series `G` such that the truncated identity
`(S − 1) = S1 · G` holds coefficient-wise up to a chosen total degree. In the code, `S` is assembled from
the hyper‑Catalan coefficient formula and `G` is solved degree‑by‑degree from the convolution structure.

### Geode convolution

We provide convolution utilities over the Geode variables `t2, t3, ...` that operate on sparse
multivariate series represented as monomial dictionaries. Internally, series are converted to
compact N‑D arrays, convolved via FFT, and cropped by a weighted‑degree cutoff. A JAX variant is
also available and designed to be `jit`/`vmap` friendly for ML workflows.

## 7) Eigenvalues via characteristic polynomial

For small/medium matrices, we form the characteristic polynomial using Faddeev–LeVerrier and call the
polynomial solver, then polish. This provides a compact path to `eigvals` without large dependencies.
