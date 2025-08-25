# GeodePoly — Non‑Finance Gap Analysis & IDE Build Spec

**Scope:** Implement the complete, paper‑faithful Hyper‑Catalan / Geode functionality for general univariate polynomials **outside finance**. This document maps missing or partial features to concrete code, tests, docs, and benchmarks.

**Primary source:** *A Hyper‑Catalan Series Solution to Polynomial Equations, and the Geode* (Wildberger & Rubine, 2025)【159†source】.  
Key anchors you’ll see below:
- Theorem 3 (soft geometric polynomial formula)【159†source】  
- Theorem 4 (soft polynomial formula, the t_k substitution)【159†source】  
- Theorem 6 (explicit hyper‑Catalan coefficients C_m)【159†source】  
- Theorem 10 (Bi‑Tri array & one‑line cubic approximant Q)【159†source】  
- Theorem 11 (Eisenstein / Bring radical quintic series; general d)【159†source】  
- Section 10 (Lagrange reversion and the bridge to S)【159†source】  
- Section 11 + Theorem 12 (layerings and the **Geode** factorization S-1=S1 G)【159†source】

---

## 0) Current Repository State

- **Core series / formal algebra:** `geodepoly/hyper_catalan.py`, `series.py`, `series_core.py`, `series_solve.py`, `formal.py`, `resummation.py`.  
- **Numeric & wrappers:** `solver.py`, `numeric.py`, `poly.py`, `geode_conv.py`, `geode_conv_jax.py`.  
- **Demos/examples:** `examples/hyper_catalan_demo.py`, `examples/geode_arrays_demo.py`, `examples/eisenstein_quintic_demo.py`.  
- **Tests:** `tests/test_hyper_catalan.py`, `tests/test_series_*`, `tests/test_resummation*.py`.  

**Conclusion:** The backbone exists. Missing are:  
(i) paper‑exact primitives and identities exposed as first‑class APIs,  
(ii) layerings & Geode,  
(iii) cubic Q fast path + bootstrap,  
(iv) reversion cross‑check,  
(v) domain adapters.

---

## 1) Gaps (paper → features)

1. General soft polynomial solver (Theorem 4)【159†source】  
2. Geode factorization exposed (Theorem 12)【159†source】  
3. Layerings (Section 11)【159†source】  
4. Bi‑Tri array & cubic Q(t2,t3) (Theorem 10)【159†source】  
5. Eisenstein/Bring radical series (Theorem 11)【159†source】  
6. Lagrange reversion cross‑check (Section 10)【159†source】  
7. Non‑finance adapters (controls, vision, geometry, signals).

---

## 2) File‑Level Plan

### New files
```
geodepoly/geode.py
geodepoly/layerings.py
geodepoly/adapters/controls/charpoly_roots.py
geodepoly/adapters/vision/distortion_invert.py
geodepoly/adapters/geometry/ray_intersect_quartic.py
geodepoly/adapters/signals/ar_roots.py
tests/test_geode_factorization.py
tests/test_layerings.py
tests/test_q_cubic_and_bootstrap.py
tests/test_eisenstein_bring.py
tests/test_lagrange_reversion.py
docs/geode.md
docs/layerings.md
docs/cubic_quintic.md
docs/non_finance_adapters.md
```

### Modified files
```
geodepoly/solver.py
geodepoly/series.py
geodepoly/hyper_catalan.py
docs/index.md, docs/how_it_works.md
```

---

## 3) API Specifications

### SeriesOptions and mapping (Theorem 4)【159†source】
```python
@dataclass
class SeriesOptions:
    Fmax: int = 6
    use_geode: bool = False
    bootstrap: bool = False
    bootstrap_passes: int = 1
    t_guard: float = 0.6

def map_t_from_poly(coeffs: Sequence[complex]) -> Dict[int, complex]:
    c0, c1, *rest = coeffs
    if c1 == 0: raise ValueError
    return {k: (c0**(k-1)*ck)/(c1**k) for k, ck in enumerate(rest, start=2)}
```

### Series evaluator with Geode path (Theorem 12)【159†source】
```python
def S_eval(t: Dict[int, complex], Fmax: int, use_geode: bool=False) -> complex: ...
def eval_S_via_geode(t: Dict[int, complex], Fmax: int) -> complex: ...
```

### Cubic one‑liner Q(t2,t3) (Theorem 10)【159†source】
```python
def Q_cubic(t2: complex, t3: complex) -> complex:
    return 1+(t2+t3)+(2*t2**2+5*t2*t3+3*t3**2)+(5*t3**2+21*t2**2*t3+28*t2*t3**2+12*t3**3)
```

### Bootstrap API (Section 8)【159†source】
```python
def solve_series(coeffs: Sequence[complex], opts: SeriesOptions) -> complex:
    # 1. compute initial x
    # 2. if bootstrap: Horner-shift around x, rebuild t', re-evaluate, update x
    return x
```

### Layerings (Section 11)【159†source】
```python
def vertex_layering(t: Dict[int,complex], Vmax:int)->list[complex]: ...
def edge_layering(t: Dict[int,complex], Emax:int)->list[complex]: ...
def face_layering(t: Dict[int,complex], Fmax:int)->list[complex]: ...
```

### Eisenstein / Bring radical (Theorem 11)【159†source】
```python
def bring_radical_series(t: complex, d:int=5, terms:int=20)->complex: ...
```

### Lagrange reversion (Section 10)【159†source】
```python
def series_reversion_coeffs(a: Dict[int,complex], order:int)->list[complex]: ...
```

### Adapters
- `charpoly_roots(coeffs, opts)` – control roots.  
- `invert_radial(rp, k1,k2,k3, opts)` – vision distortion.  
- `ray_surface_intersections(params, opts)` – geometry.  
- `ar_poles(ar_coeffs, opts)` – signal processing.

---

## 4) Tests

- `test_q_cubic_and_bootstrap.py`: Wallis cubic residual <1e‑12【159†source】.  
- `test_geode_factorization.py`: Geode vs direct parity【159†source】.  
- `test_layerings.py`: OEIS match (little Schröder, Riordan)【159†source】.  
- `test_eisenstein_bring.py`: Eisenstein series residual scaling【159†source】.  
- `test_lagrange_reversion.py`: coefficients match Theorem 6 vs Lagrange【159†source】.

---

## 5) Docs

- `geode.md`: S-1=S1G identity【159†source】.  
- `layerings.md`: SV, SE, SF with OEIS references【159†source】.  
- `cubic_quintic.md`: Q(t2,t3) + Bring radical【159†source】.  
- `non_finance_adapters.md`: usage for controls, vision, geometry, signals.

---

## 6) Acceptance Criteria

- Math fidelity: Theorems 3–12 implemented【159†source】.  
- Residuals <1e‑10 on fixtures.  
- Geode path stable vs direct.  
- Layerings produce OEIS‑matching sequences.  
- Lagrange reversion passes low‑order checks.  
- Docs complete with examples.  
