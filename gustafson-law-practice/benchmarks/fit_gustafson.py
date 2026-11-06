"""Fit Gustafson's law to results.csv. Usage: python fit_gustafson.py"""
import numpy as np
import pandas as pd
from scipy import stats

df = pd.read_csv("results.csv")
t1 = df[df.threads == 1].time_s.mean()

# Scaled speedup of every run: S = P * T(1) / T(P)
df["speedup"] = df.threads * t1 / df.time_s

g = df.groupby("threads").agg(
    time_mean=("time_s", "mean"), time_std=("time_s", "std"),
    S_mean=("speedup", "mean"), S_std=("speedup", "std"),
    n=("run", "count")).reset_index()
g["S_sem"] = g.S_std / np.sqrt(g.n)
g.to_csv("results_summary.csv", index=False)
print(g.to_string(index=False))

# Gustafson: S = (1 - alpha) * P + alpha  ->  slope = 1 - alpha
res = stats.linregress(g.threads, g.S_mean)
alpha = 1.0 - res.slope
print(f"\nslope = {res.slope:.4f}, intercept = {res.intercept:.4f}, R^2 = {res.rvalue**2:.4f}")
print(f"alpha_observed = {alpha:.4f}")
with open("alpha_fit.txt", "w") as f:
    f.write(f"{alpha:.6f}\n")