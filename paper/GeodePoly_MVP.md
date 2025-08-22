# GeodePoly: Robust Polynomial Root-Finding via Series Reversion and Symmetry-Informed Finishers

**Authors:** Ric Fulop, et al.  
**Affiliation:** MIT Center for Bits and Atoms  
**License:** MIT

## Abstract
We present a practical polynomial solver that combines analytic **series reversion**
(around a recentered origin) with modern simultaneous all-roots iterations
(**Aberth–Ehrlich**) and final **Halley** polishing. The analytic step leverages
Hyper‑Catalan/Geode structure to produce **derivative‑light** seed updates and
admits **Padé** and **Borel–Padé** resummation. The finisher handles clustered
and ill-conditioned spectra without initial guesses. On random degree‑3–20 polynomials,
the method achieves competitive accuracy to companion‑matrix eigenvalue solvers while
retaining derivative‑free robustness and offering a path to GPU tiling via Geode layering.

## 1. Introduction
- Background on Newton fragility and radical limitations.
- Geode/Hyper‑Catalan intuition for compositional inversion.
- Contributions: (i) robust series seed, (ii) Aberth finisher, (iii) resummation toggles,
(iv) simple CAS/API bridges, (v) benchmarks and open dataset plan.

## 2. Method
### 2.1. Series step (recenter → invert → bootstrap)
We write \( q(y)=p(\mu+y)=a_0+a_1y+a_2y^2+\cdots \) and invert
\( F(y)=y+\sum_{k\ge2}\beta_k y^k \), \( \beta_k=a_k/a_1 \).
Using Lagrange inversion, the inverse coefficients \(g_m\) satisfy
\( g_m=\tfrac{1}{m}[y^{m-1}](1/F'(y))^m \). The root increment is
\( y \approx \sum_{m\ge1} g_m t^m \) with \( t=-a_0/a_1 \).

### 2.2. Resummation
We support (i) near-diagonal **Padé**, (ii) **Borel** via Gauss–Laguerre quadrature,
and (iii) **Borel–Padé** (Padé on the Borel transform, then Laplace).

### 2.3. Finishers
**Aberth–Ehrlich**: simultaneous correction
\( z_i \leftarrow z_i - \frac{p(z_i)}{p'(z_i) - p(z_i) \sum_{j\ne i} (z_i-z_j)^{-1}} \).  
**Durand–Kerner** for derivative-free fallback. **Halley** for polishing.

## 3. Results
- Accuracy: max residuals and success rates vs. (a) Hybrid (ours), (b) Aberth, (c) DK,
(d) Companion/QR (`numpy.roots`).
- Runtime: per-degree scaling.
- Stress: clustered and near-multiple roots.

## 4. Discussion & Limitations
- Conditioning near multiple roots.
- Resummation tuning and order selection.
- Potential GPU tiling via Geode layer decomposition.

## 5. Outlook
- Eigenvalue pipelines via characteristic polynomials.
- Finite-field variants and Hensel lifting.
- CAS integration and GPU kernels.
- GeodeBench for symmetry generalization.

## References
*(to be filled)*
