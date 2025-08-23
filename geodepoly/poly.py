from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple


def _trim(coeffs: List[complex]) -> List[complex]:
    c = list(coeffs)
    while len(c) > 1 and c[-1] == 0:
        c.pop()
    return c


@dataclass
class Polynomial:
    """Dense univariate polynomial with complex coefficients (low→high).

    - Coefficients stored as list [a0, a1, ..., aN]
    - Supports addition, subtraction, multiplication, divmod, power (non-negative int),
      evaluation (Horner), shift (x→x+mu), scale_x (x→alpha*x), differentiate, integrate.
    """

    coeffs: List[complex]

    def __post_init__(self) -> None:
        self.coeffs = [complex(a) for a in self.coeffs]
        self.coeffs = _trim(self.coeffs)

    @property
    def degree(self) -> int:
        return len(self.coeffs) - 1

    def copy(self) -> "Polynomial":
        return Polynomial(list(self.coeffs))

    # ---- arithmetic ----
    def __add__(self, other: "Polynomial") -> "Polynomial":
        n = max(len(self.coeffs), len(other.coeffs))
        out = [0j] * n
        for i in range(n):
            a = self.coeffs[i] if i < len(self.coeffs) else 0
            b = other.coeffs[i] if i < len(other.coeffs) else 0
            out[i] = a + b
        return Polynomial(_trim(out))

    def __sub__(self, other: "Polynomial") -> "Polynomial":
        n = max(len(self.coeffs), len(other.coeffs))
        out = [0j] * n
        for i in range(n):
            a = self.coeffs[i] if i < len(self.coeffs) else 0
            b = other.coeffs[i] if i < len(other.coeffs) else 0
            out[i] = a - b
        return Polynomial(_trim(out))

    def __mul__(self, other: "Polynomial") -> "Polynomial":
        a, b = self.coeffs, other.coeffs
        out = [0j] * (len(a) + len(b) - 1)
        for i, ai in enumerate(a):
            if ai == 0:
                continue
            for j, bj in enumerate(b):
                if bj == 0:
                    continue
                out[i + j] += ai * bj
        return Polynomial(_trim(out))

    def __pow__(self, k: int) -> "Polynomial":
        if k < 0:
            raise ValueError("Power must be non-negative")
        res = Polynomial([1])
        base = self.copy()
        e = k
        while e > 0:
            if e & 1:
                res = res * base
            base = base * base
            e >>= 1
        return res

    def __divmod__(self, other: "Polynomial") -> Tuple["Polynomial", "Polynomial"]:
        if other.degree < 0 or (len(other.coeffs) == 1 and other.coeffs[0] == 0):
            raise ZeroDivisionError("polynomial division by zero")
        a = list(self.coeffs)
        b = other.coeffs
        m = len(a) - 1
        n = len(b) - 1
        if m < n:
            return Polynomial([0]), self.copy()
        q = [0j] * (m - n + 1)
        r = a[:]
        bn = b[-1]
        for k in range(m - n, -1, -1):
            coeff = r[n + k] / bn
            q[k] = coeff
            for j in range(n + k, k - 1, -1):
                r[j] -= coeff * b[j - k]
        return Polynomial(_trim(q)), Polynomial(_trim(r[:n]))

    # ---- evaluation and transforms ----
    def __call__(self, x: complex) -> complex:
        acc = 0j
        for a in reversed(self.coeffs):
            acc = acc * x + a
        return acc

    def differentiate(self) -> "Polynomial":
        if len(self.coeffs) <= 1:
            return Polynomial([0])
        out = [(i) * self.coeffs[i] for i in range(1, len(self.coeffs))]
        return Polynomial(out)

    def integrate(self, c0: complex = 0) -> "Polynomial":
        out = [complex(c0)] + [
            self.coeffs[i] / (i + 1) for i in range(len(self.coeffs))
        ]
        return Polynomial(out)

    def scale_x(self, alpha: complex) -> "Polynomial":
        # p(alpha x) = sum a_k (alpha^k) x^k
        if alpha == 1:
            return self.copy()
        out = [0j] * len(self.coeffs)
        power = 1.0 + 0.0j
        for k, a in enumerate(self.coeffs):
            out[k] = a * power
            power *= alpha
        return Polynomial(out)

    def shift_x(self, mu: complex) -> "Polynomial":
        # q(y) = p(mu + y)
        # Use binomial expansion: a_k (mu+y)^k
        from math import comb

        n = self.degree
        out = [0j] * (n + 1)
        for k, ak in enumerate(self.coeffs):
            if ak == 0:
                continue
            for j in range(k + 1):
                out[j] += ak * comb(k, j) * (mu ** (k - j))
        return Polynomial(out)
