import numpy as np
from geodepoly.eigs import solve_eigs


def test_eigs_random_small():
    rng = np.random.default_rng(0)
    for n in [3, 4, 5]:
        A = rng.normal(size=(n,n)) + 1j*rng.normal(size=(n,n))
        vals_np = np.linalg.eigvals(A)
        vals_gp = solve_eigs(A)
        # Match sets by nearest pairing
        taken = [False]*n
        for v in vals_gp:
            idx = np.argmin([abs(v-w) for w in vals_np])
            taken[idx] = True
            assert abs(v - vals_np[idx]) < 1e-6

