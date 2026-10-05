"""Experiment 1: reproduce CLINE on FHN and test how much data it needs.

A single trajectory (f = 1, dt = 0.1) is truncated to N time steps and CLINE
estimates the u-nullcline from it. Accuracy is the RMSE against the exact
nullcline v = -u^3 + u^2 + u.

Output
  outputs/cline_reproduction/nullcline_N=<N>.csv
  results/cline_reproduction/rmse_vs_n.csv
  figures/cline_rmse_vs_n.png
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from nullsr import config, fhn
from nullsr.cline import estimate_nullcline

N_VALUES = [100, 1_000, 5_000, 10_000, 20_000]
F, DT = 1.0, 0.1
EPOCHS, BATCH = 1000, 64
TMIN = 3  # drop the initial transient, as in the pyCLINE example

out_dir = config.OUTPUT_DIR / "cline_reproduction"
res_dir = config.RESULTS_DIR / "cline_reproduction"
out_dir.mkdir(parents=True, exist_ok=True)
res_dir.mkdir(parents=True, exist_ok=True)
config.FIGURES_DIR.mkdir(exist_ok=True)

full = fhn.generate(F, DT, max(N_VALUES))

rows = []
for n in N_VALUES:
    df = full[full["time"] < n * DT]
    nullcline, _, _ = estimate_nullcline(df, config.CLINE_SEED, EPOCHS, BATCH, tmin=TMIN)
    nullcline.to_csv(out_dir / f"nullcline_N={n}.csv", index=False)
    rmse = np.sqrt(np.mean((nullcline["v"] - fhn.u_nullcline(nullcline["u"])) ** 2))
    rows.append({"N": n, "rmse": rmse})
    print(f"N = {n:>6}: RMSE = {rmse:.4f}")

table = pd.DataFrame(rows)
table.to_csv(res_dir / "rmse_vs_n.csv", index=False)

plt.figure(figsize=(5, 3.5))
plt.plot(table["N"], table["rmse"], "o-")
plt.xscale("log")
plt.xlabel("Number of time steps N")
plt.ylabel("Nullcline RMSE")
plt.title("CLINE accuracy vs. data length (FHN, f = 1)")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(config.FIGURES_DIR / "cline_rmse_vs_n.png", dpi=200)
