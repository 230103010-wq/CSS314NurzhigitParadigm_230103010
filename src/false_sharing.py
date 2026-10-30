import csv, ctypes, statistics, time
import multiprocessing as mp
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
K = 5_000_000
PS = [1, 2, 4, 8]
REPS = 3


def w_unpadded(arr, base, p, iters, bar):
    idx = base + p            # индексы base..base+7 — одна линия
    bar.wait()
    for _ in range(iters):
        arr[idx] += 1


def w_padded(arr, base, p, iters, bar):
    idx = base + p * 8        # каждая на своей линии
    bar.wait()
    for _ in range(iters):
        arr[idx] += 1


def w_private(arr, base, p, iters, bar):
    bar.wait()
    local = 0
    for _ in range(iters):
        local += 1
    arr[base + p * 8] = local  # одна запись в конце


WORKERS = {"unpadded": w_unpadded, "padded": w_padded, "privatized": w_private}


def run_once(name, P):
    arr = mp.Array(ctypes.c_int64, 80, lock=False)
    # выровнять начало рабочей области по 64 байтам
    addr = ctypes.addressof(arr)
    base = ((64 - addr % 64) % 64) // 8
    bar = mp.Barrier(P + 1)
    procs = [mp.Process(target=WORKERS[name], args=(arr, base, p, K, bar)) for p in range(P)]
    for pr in procs:
        pr.start()
    bar.wait()
    t0 = time.perf_counter()
    for pr in procs:
        pr.join()
    t = time.perf_counter() - t0
    stride = 1 if name == "unpadded" else 8
    total = sum(arr[base + p * stride] for p in range(P))
    assert total == P * K, f"{name} P={P}: {total} != {P*K}"
    return t, total


def main():
    rows = []
    for name in WORKERS:
        t1 = None
        for P in PS:
            ts = []
            for _ in range(REPS):
                t, total = run_once(name, P)
                ts.append(t)
            T = statistics.median(ts)
            if P == 1:
                t1 = T
            e = (P * T / t1 - 1) / (P - 1) if P > 1 else ""
            S = t1 / T
            rows.append([name, P, K, f"{T:.4f}", f"{S:.3f}", e if e == "" else f"{e:.4f}", total])
            print(name, P, f"T={T:.3f}s S={S:.2f} e={e}")
    with open(ROOT / "results" / "false_sharing_karp_flatt.tsv", "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["variant", "P", "K", "median_time_s", "speedup", "karp_flatt_e", "total_count"])
        w.writerows(rows)


if __name__ == "__main__":
    mp.freeze_support()
    main()