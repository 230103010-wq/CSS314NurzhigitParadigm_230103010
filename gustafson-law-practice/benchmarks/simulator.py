"""Amdahl vs Gustafson simulator. Usage: python simulator.py --out-dir ."""
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def amdahl(P, s):
    return 1.0 / (s + (1.0 - s) / P)


def gustafson(P, a):
    return P - a * (P - 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pmax", type=int, default=65536)
    ap.add_argument("--alphas", type=float, nargs="+",
                    default=[0.001, 0.01, 0.04, 0.08, 0.20])
    ap.add_argument("--csv", default="simulator_output.csv")
    ap.add_argument("--plot", default="../plots/simulator_overlay.png")
    args = ap.parse_args()

    if not (1 <= args.pmax <= 65536):
        raise SystemExit("pmax must be in [1, 65536]")
    if any(not (0.001 <= a <= 0.20) for a in args.alphas):
        raise SystemExit("alpha must be in [0.001, 0.20]")

    Ps = 2 ** np.arange(0, int(np.log2(args.pmax)) + 1)
    rows = []
    for a in args.alphas:
        for P in Ps:
            sa, sg = amdahl(P, a), gustafson(P, a)
            rows.append(dict(alpha=a, P=int(P),
                             amdahl_speedup=sa, gustafson_speedup=sg,
                             amdahl_eff=sa / P, gustafson_eff=sg / P))
    df = pd.DataFrame(rows)
    df.to_csv(args.csv, index=False)
    print(f"CSV saved: {args.csv} ({len(df)} rows)")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for a in args.alphas:
        d = df[df.alpha == a]
        axes[0].plot(d.P, d.amdahl_speedup, "--", label=f"Amdahl s={a}")
        axes[1].plot(d.P, d.gustafson_speedup, "-", label=f"Gustafson α={a}")
    for ax, title in zip(axes, ["Strong scaling (Amdahl)", "Weak scaling (Gustafson)"]):
        ax.plot(Ps, Ps, "k:", label="Ideal")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("Processors P")
        ax.set_ylabel("Speedup")
        ax.set_title(title)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.plot)), exist_ok=True)
    fig.savefig(args.plot, dpi=300)
    print(f"Plot saved: {args.plot}")


if __name__ == "__main__":
    main()