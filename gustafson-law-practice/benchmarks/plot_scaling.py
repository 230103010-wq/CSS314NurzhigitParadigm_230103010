"""Plot ideal, Gustafson fit, measured data and Amdahl. Run after fit_gustafson.py"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

g = pd.read_csv("results_summary.csv")
alpha = float(open("alpha_fit.txt").read())
alpha = min(max(alpha, 1e-6), 0.99)   # guard against negative/odd fits

P = np.linspace(1, g.threads.max(), 200)
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(P, P, "k:", label="Ideal linear")
ax.plot(P, P - alpha * (P - 1), "b-", label=f"Gustafson fit (α={alpha:.3f})")
ax.plot(P, 1 / (alpha + (1 - alpha) / P), "r--", label=f"Amdahl (s={alpha:.3f})")
ax.errorbar(g.threads, g.S_mean, yerr=g.S_sem, fmt="o", color="green",
            capsize=4, label="Measured (± std. error)")
ax.set_xlabel("Processes P")
ax.set_ylabel("Scaled speedup")
ax.set_title("Weak scaling: empirical vs theoretical")
ax.grid(True, alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig("../plots/scaling_comparison.png", dpi=300)
fig.savefig("../plots/scaling_comparison.pdf")
print("Saved plots/scaling_comparison.(png|pdf)")