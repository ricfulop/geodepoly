from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, MutableMapping, Tuple


Monomial = Tuple[int, ...]


def _normalize_monomial(m: Iterable[int], num_vars: int | None = None) -> Monomial:
    exps = list(m)
    if num_vars is not None and len(exps) < num_vars:
        exps = exps + [0] * (num_vars - len(exps))
    # Trim trailing zeros
    while exps and exps[-1] == 0:
        exps.pop()
    return tuple(int(e) for e in exps)


@dataclass(frozen=True)
class SeriesSpec:
    var_names: Tuple[str, ...]


class FormalSeries:
    """Minimal formal power series in several commuting variables with complex coefficients.

    - Internal representation: dict[monomial_tuple, complex]
      where monomial_tuple = (e1, e2, ..., ek) with trailing zeros trimmed.
    - Supported operations: addition, multiplication (Cauchy product), truncation,
      coefficient lookup, simple composition (univariate case), pretty conversion.
    """

    def __init__(self, coeffs: Mapping[Monomial, complex] | None = None, var_names: Iterable[str] = ("t",)):
        names = tuple(var_names)
        if not names:
            names = ("t",)
        self.spec = SeriesSpec(var_names=names)
        self._coeffs: Dict[Monomial, complex] = {}
        if coeffs:
            for m, c in coeffs.items():
                mm = _normalize_monomial(m)
                if c != 0:
                    self._coeffs[mm] = complex(c)

    # --- Basic protocol ---
    def copy(self) -> "FormalSeries":
        return FormalSeries(dict(self._coeffs), self.spec.var_names)

    @property
    def vars(self) -> Tuple[str, ...]:
        return self.spec.var_names

    def coeff(self, m: Iterable[int]) -> complex:
        return self._coeffs.get(_normalize_monomial(m), 0.0 + 0.0j)

    def support(self) -> List[Monomial]:
        return list(self._coeffs.keys())

    # --- Arithmetic ---
    def __add__(self, other: "FormalSeries") -> "FormalSeries":
        self._assert_compatible(other)
        out: MutableMapping[Monomial, complex] = dict(self._coeffs)
        for m, c in other._coeffs.items():
            out[m] = out.get(m, 0.0 + 0.0j) + c
            if out[m] == 0:
                del out[m]
        return FormalSeries(out, self.vars)

    def __mul__(self, other: "FormalSeries") -> "FormalSeries":
        self._assert_compatible(other)
        out: Dict[Monomial, complex] = {}
        for (m1, c1) in self._coeffs.items():
            for (m2, c2) in other._coeffs.items():
                m = _add_monomials(m1, m2)
                out[m] = out.get(m, 0.0 + 0.0j) + c1 * c2
                if out[m] == 0:
                    del out[m]
        return FormalSeries(out, self.vars)

    def scale(self, alpha: complex) -> "FormalSeries":
        if alpha == 0:
            return FormalSeries({}, self.vars)
        return FormalSeries({m: alpha * c for m, c in self._coeffs.items()}, self.vars)

    # --- Truncation ---
    def truncate_total_degree(self, max_deg: int) -> "FormalSeries":
        out = {m: c for m, c in self._coeffs.items() if sum(m) <= max_deg}
        return FormalSeries(out, self.vars)

    # --- Composition (univariate) ---
    def compose_univariate(self, g: "FormalSeries", max_deg: int | None = None) -> "FormalSeries":
        """Return f(g) where f is this series in one variable.

        Only supports one variable; raises if series is multivariate. Truncates by total degree.
        """
        if len(self.vars) != 1:
            raise NotImplementedError("compose_univariate only for univariate series")
        if max_deg is None:
            # Heuristic cutoff: sum of degrees of nonzero coeffs
            max_deg = max((sum(m) for m in self._coeffs.keys()), default=0)
        # f(x) = sum a_k x^k; g as a series with zero constant term assumed
        a_by_k: Dict[int, complex] = {}
        for m, c in self._coeffs.items():
            if len(m) <= 1:
                k = m[0] if m else 0
                a_by_k[k] = a_by_k.get(k, 0.0 + 0.0j) + c
            else:
                raise NotImplementedError("compose_univariate expects univariate f")
        # Powers of g
        one = FormalSeries({(): 1.0 + 0.0j}, g.vars)
        g_tr = g.truncate_total_degree(max_deg)
        pow_cache: List[FormalSeries] = [one]
        for k in range(1, max_deg + 1):
            pow_cache.append((pow_cache[-1] * g_tr).truncate_total_degree(max_deg))
        out = FormalSeries({}, g.vars)
        for k, ak in a_by_k.items():
            if k <= max_deg and ak != 0:
                out = out + pow_cache[k].scale(ak)
        return out.truncate_total_degree(max_deg)

    # --- Conversions ---
    def to_sympy(self):  # type: ignore[override]
        try:
            import sympy as sp  # type: ignore
        except Exception:
            raise ImportError("SymPy not installed")
        sym_vars = sp.symbols(self.vars)
        expr = 0
        for m, c in self._coeffs.items():
            term = c
            for i, e in enumerate(m):
                if e:
                    term *= sym_vars[i] ** e
            expr += term
        return expr

    def __repr__(self) -> str:
        if not self._coeffs:
            return f"FormalSeries(0; vars={self.vars})"
        terms = []
        for m, c in sorted(self._coeffs.items(), key=lambda mc: (sum(mc[0]), mc[0])):
            mon = monomial_to_str(m, self.vars)
            terms.append(f"({c})*{mon}" if mon != "1" else f"({c})")
        return "FormalSeries(" + " + ".join(terms) + "; vars=" + ",".join(self.vars) + ")"

    # --- Helpers ---
    def _assert_compatible(self, other: "FormalSeries") -> None:
        if self.vars != other.vars:
            raise ValueError("series have different variables")


def _add_monomials(a: Monomial, b: Monomial) -> Monomial:
    n = max(len(a), len(b))
    out = [0] * n
    for i in range(n):
        ea = a[i] if i < len(a) else 0
        eb = b[i] if i < len(b) else 0
        out[i] = ea + eb
    return _normalize_monomial(out)


def monomial_to_str(m: Monomial, var_names: Tuple[str, ...]) -> str:
    if not m:
        return "1"
    parts: List[str] = []
    for i, e in enumerate(m):
        if e:
            name = var_names[i] if i < len(var_names) else f"t{i+1}"
            parts.append(f"{name}^{e}" if e != 1 else name)
    return "*".join(parts) if parts else "1"


