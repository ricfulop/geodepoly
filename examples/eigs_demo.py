import numpy as np
from geodepoly.eigs import solve_eigs


def main():
    rng = np.random.default_rng(1)
    A = rng.normal(size=(4,4)) + 1j*rng.normal(size=(4,4))
    vals = solve_eigs(A)
    print("eigenvalues:", vals)


if __name__ == "__main__":
    main()


