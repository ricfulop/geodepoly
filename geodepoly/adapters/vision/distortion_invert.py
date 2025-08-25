from __future__ import annotations

from typing import List, Sequence, Tuple

from ...solver import solve_one


def invert_radial(rp: float, k1: float, k2: float = 0.0, k3: float = 0.0) -> float:
    """Invert standard radial distortion r_d = r_u * (1 + k1 r_u^2 + k2 r_u^4 + k3 r_u^6).

    Given distorted radius rp=r_d, find undistorted r_u>=0 by solving
    r_u + k1 r_u^3 + k2 r_u^5 + k3 r_u^7 - r_d = 0.
    Returns the nonnegative root closest to rp.
    """
    # Polynomial coefficients in ascending order: a0 + a1 x + ...
    # p(r) = -rp + 1*r + k1*r^3 + k2*r^5 + k3*r^7
    coeffs: List[complex] = [
        -complex(rp),
        1.0 + 0.0j,
        0.0 + 0.0j,
        complex(k1),
        0.0 + 0.0j,
        complex(k2),
        0.0 + 0.0j,
        complex(k3),
    ]
    x = solve_one(coeffs, center=complex(rp), max_order=24, boots=3)
    return float(abs(x))


__all__ = ["invert_radial"]


