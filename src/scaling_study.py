import csv, json, math, statistics, time
import multiprocessing as mp
from multiprocessing import shared_memory
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ITERS = 100
REPS = 3


def serial_ref(grid, iters):
    a, b = grid.copy(), grid.copy()
    for _ in range(iters):
        b[1:-1, 1:-1] = 0.25 * (a[:-2, 1:-1] + a[2:, 1:-1] + a[1:-1, :-2] + a[1:-1, 2:])
        a, b = b, a
    return a


def worker(names, N, r0, r1, iters, start_bar, step_bar):
    shm = [shared_memory.SharedMemory(name=n) for n in names]
    g = [np.ndarray((N, N), dtype=np.float64, buffer=s.buf) for s in shm]
    start_bar.wait()
    src, dst = 0, 1
    for _ in range(iters):
        a, b = g[src], g[dst]
        b[r0:r1, 1:-1] = 0.25 * (a[r0-1:r1-1, 1:-1] + a[r0+1:r1+1, 1:-1] +
                                 a[r0:r1, :-2] + a[r0:r1, 2:])
        step_bar.wait()           # синхронизация "гало": соседи видят новые строки
        src, dst = dst, src
    for s in shm:
        s.close()


def run_parallel(N, P, init, iters):
    shms = [shared_memory.SharedMemory(create=True, size=N * N * 8) for _ in range(2)]
    try:
        g = [np.ndarray((N, N), dtype=np.float64, buffer=s.buf) for s in shms]
        g[0][:] = init
        g[1][:] = init            # границы фиксированы в обоих буферах
        rows = np.array_split(np.arange(1, N - 1), P)
        start_bar, step_bar = mp.Barrier(P + 1), mp.Barrier(P)
        procs = [mp.Process(target=worker,
                            args=([s.name for s in shms], N, int(r[0]), int(r[-1]) + 1, iters, start_bar, step_bar))
                 for r in rows]
        for p in procs:
            p.start()
        start_bar.wait()
        t0 = time.perf_counter()
        for p in procs:
            p.join()
        t = time.perf_counter() - t0
        result = g[iters % 2].copy()
        return t, result
    finally:
        for s in shms:
            s.close()
            s.unlink()


def make_grid(N, seed):
    return np.random.default_rng(seed).random((N, N))


def measure(N, P, seed):
    init = make_grid(N, seed)
    ts, res = [], None
    for _ in range(REPS):
        t, res = run_parallel(N, P, init, ITERS)
        ts.append(t)
    return statistics.median(ts), res


def main():
    p = json.loads((ROOT / "results" / "salted_params.json").read_text())
    host = json.loads((ROOT / "results" / "host_profile.json").read_text())
    N0, seed = p["N"], p["seed"]
    maxP = host["physical_cores"] or max(1, (mp.cpu_count() // 2))
    Ps = sorted({2**i for i in range(0, 8) if 2**i <= maxP} | {maxP})

    ref = serial_ref(make_grid(N0, seed), ITERS)
    ref_sum = float(ref.sum())

    rows, T1, Tw1 = [], None, None
    s_vals = []
    # strong scaling
    for P in Ps:
        T, res = measure(N0, P, seed)
        assert np.allclose(res, ref, atol=1e-12), "результат не совпал с эталоном"
        if P == 1:
            T1 = T
        S = T1 / T
        E = S / P * 100
        s = ((1 / S - 1 / P) / (1 - 1 / P)) if P > 1 else ""
        if P > 1:
            s_vals.append(s)
        rows.append(["strong", P, N0, f"{T:.4f}", f"{S:.3f}", f"{E:.1f}", s if s == "" else f"{s:.4f}", "", float(res.sum())])
        print("strong", P, T, S, E, s)

    # weak scaling
    for P in Ps:
        NP = math.ceil(N0 * math.sqrt(P))
        T, res = measure(NP, P, seed)
        if P == 1:
            Tw1 = T
        work1, workP = N0 ** 2, NP ** 2
        scaled = P * work1 * Tw1 / (workP * T)
        rows.append(["weak", P, NP, f"{T:.4f}", "", "", "", f"{scaled:.3f}", float(res.sum())])
        print("weak", P, NP, T, scaled)

    with open(ROOT / "results" / "scaling_metrics.tsv", "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["mode", "P", "N", "time_s", "speedup", "efficiency_pct", "serial_fraction_s",
                    "scaled_speedup", "checksum"])
        w.writerows(rows)

    s_avg = statistics.mean(s_vals) if s_vals else None
    print("Amdahl s (mean over P>1):", s_avg, " S_max =", (1 / s_avg) if s_avg and s_avg > 0 else None)

    # Roofline
    flops_per_pt, bytes_per_pt = 4, 16     # 3 add + 1 mul; 8 B read + 8 B write (идеал, без write-allocate)
    AI = flops_per_pt / bytes_per_pt
    stride_rows = list(csv.DictReader(open(ROOT / "results" / "stride_results.tsv"), delimiter="\t"))
    ns1 = float([r for r in stride_rows if r["stride"] == "1"][0]["latency_ns_per_element"])
    bw_gbs = 8 / ns1                      # байт/нс = ГБ/с (однопоточная оценка)
    freq_ghz = (host["base_freq_mhz"] or 3000) / 1000
    flops_per_cycle = 16                  # AVX2 FMA double: подставьте для своего CPU
    peak_gflops = maxP * freq_ghz * flops_per_cycle
    ridge = peak_gflops / bw_gbs
    roof = {
        "kernel": "5-point Jacobi stencil", "flops_per_point": flops_per_pt,
        "dram_bytes_per_point": bytes_per_pt, "arithmetic_intensity_flop_per_byte": AI,
        "measured_stream_bw_gbs_1thread": bw_gbs, "assumed_peak_gflops": peak_gflops,
        "assumed_flops_per_cycle_per_core": flops_per_cycle,
        "ridge_point_flop_per_byte": ridge,
        "working_set_bytes": 2 * N0 * N0 * 8, "l3_bytes": host["caches"].get("l3_bytes"),
        "regime": "memory-bandwidth bound" if AI < ridge else "compute bound",
        "attainable_gflops": min(peak_gflops, AI * bw_gbs),
        "amdahl_serial_fraction_mean": s_avg,
        "amdahl_S_max": (1 / s_avg) if s_avg and s_avg > 0 else None,
    }
    (ROOT / "results" / "roofline_summary.json").write_text(json.dumps(roof, indent=2))
    print(json.dumps(roof, indent=2))


if __name__ == "__main__":
    mp.freeze_support()
    main()