
import hashlib, json, statistics, time
from pathlib import Path
import numpy as np
from numba import njit

ROOT = Path(__file__).resolve().parent.parent


@njit(cache=False)
def ijk(A, B, C, M):
    for i in range(M):
        for j in range(M):
            acc = 0.0
            for k in range(M):
                acc += A[i * M + k] * B[k * M + j]   # B идёт с шагом M
            C[i * M + j] = acc


@njit(cache=False)
def ikj(A, B, C, M):
    for i in range(M * M):
        C[i] = 0.0
    for i in range(M):
        for k in range(M):
            a = A[i * M + k]                          # в регистр
            for j in range(M):
                C[i * M + j] += a * B[k * M + j]      # stride-1


def timed(fn, A, B, M, runs=3):
    ts = []
    for _ in range(runs):
        C = np.zeros(M * M)
        t0 = time.perf_counter_ns()
        fn(A, B, C, M)
        ts.append(time.perf_counter_ns() - t0)
    return statistics.median(ts), C


def digest(C):
    return hashlib.sha256(np.round(C, 6).tobytes()).hexdigest()


def main():
    p = json.loads((ROOT / "results" / "salted_params.json").read_text())
    M = p["N"] // 2
    rng = np.random.default_rng(p["seed"])
    A = rng.random(M * M)
    B = rng.random(M * M)

    # прогрев компиляции (не входит в замер)
    w = np.zeros(16)
    ijk(A[:16], B[:16], w, 4)
    ikj(A[:16], B[:16], w, 4)

    t_naive, C1 = timed(ijk, A, B, M)
    t_swap, C2 = timed(ikj, A, B, M)
    ref = (A.reshape(M, M) @ B.reshape(M, M)).ravel()

    out = {
        "N": p["N"], "M": M,
        "t_ijk_ns": t_naive, "t_ikj_ns": t_swap,
        "speedup": t_naive / t_swap,
        "max_abs_diff_ijk_vs_ikj": float(np.max(np.abs(C1 - C2))),
        "max_abs_diff_vs_numpy": float(np.max(np.abs(C2 - ref))),
        "sha256_ijk": digest(C1), "sha256_ikj": digest(C2),
    }
    (ROOT / "results" / "loop_interchange.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()