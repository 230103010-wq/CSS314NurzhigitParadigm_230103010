"""Weak-scaling benchmark: numerical integration of pi.
Usage: python bench.py [--k 20000000] [--runs 5] [--threads 1 2 4 8]
"""
import argparse
import csv
import os
import time
from multiprocessing import Pool

import numpy as np


def work(args):
    """Integrate 4/(1+x^2) over sub-interval i of P, using k midpoints."""
    i, P, k = args
    lo, hi = i / P, (i + 1) / P
    h = (hi - lo) / k
    total, chunk = 0.0, 1_000_000
    for start in range(0, k, chunk):
        end = min(start + chunk, k)
        x = lo + (np.arange(start, end) + 0.5) * h
        total += float(np.sum(4.0 / (1.0 + x * x)))
    return total * h


def noop(_):
    return 0


def default_threads():
    n = os.cpu_count() or 1
    t, p = [], 1
    while p < n:
        t.append(p)
        p *= 2
    t.append(n)
    return sorted(set(t))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=20_000_000,
                    help="intervals per worker (constant)")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--threads", type=int, nargs="+", default=default_threads())
    ap.add_argument("--out", default="../results.csv")
    args = ap.parse_args()

    print(f"Thread counts: {args.threads}, k={args.k}, runs={args.runs}")
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["threads", "run", "time_s", "pi_estimate"])
        for P in args.threads:
            with Pool(P) as pool:
                pool.map(noop, range(P))            # warm-up: start processes
                for run in range(1, args.runs + 1):
                    tasks = [(i, P, args.k) for i in range(P)]
                    t0 = time.perf_counter()
                    parts = pool.map(work, tasks, chunksize=1)
                    pi = sum(parts)                  # serial reduction
                    t = time.perf_counter() - t0
                    w.writerow([P, run, f"{t:.6f}", f"{pi:.12f}"])
                    f.flush()
                    print(f"P={P:3d} run={run} time={t:.4f}s pi={pi:.10f}")
    print(f"Saved: {args.out}")


if __name__ == "__main__":     # required on Windows!
    main()