#!/usr/bin/env python3
import argparse
import cProfile
import pstats
import random
import time

from geodepoly import solve_all


def rand_poly(deg: int, seed: int = 0):
    random.seed(seed)
    # ensure nonzero leading coefficient
    coeffs = [random.uniform(-1, 1) for _ in range(deg)] + [1.0]
    return coeffs


def bench(deg: int, trials: int):
    t0 = time.perf_counter()
    for i in range(trials):
        coeffs = rand_poly(deg, seed=1000 + i)
        solve_all(coeffs, method="hybrid", resum="auto", tol=1e-12)
    return time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser(description="Profile finishers and hot paths")
    ap.add_argument("--deg", type=int, default=12)
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--sort", type=str, default="cumtime", choices=["time", "cumtime", "tottime"]) 
    ap.add_argument("--limit", type=int, default=30)
    args = ap.parse_args()

    profiler = cProfile.Profile()
    profiler.enable()
    dt = bench(args.deg, args.trials)
    profiler.disable()

    print(f"deg={args.deg} trials={args.trials} elapsed={dt:.3f}s")
    stats = pstats.Stats(profiler).strip_dirs().sort_stats(args.sort)
    stats.print_stats(args.limit)


if __name__ == "__main__":
    main()


