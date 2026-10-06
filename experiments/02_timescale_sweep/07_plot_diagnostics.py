"""Diagnostics for the time-scale sweep.

figures/cline_nullcline_examples.png  CLINE nullcline at an easy and a hard setting
figures/derivative_error.png          relative error of the numerical du/dt vs. f * dt

Reads results/timescale_sweep/.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from nullsr import config, fhn

RES = config.RESULTS_DIR / "timescale_sweep"
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3dd"
BLUE = "#2a78d6"
RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]  # f = 1 ... 20
config.FIGURES_DIR.mkdir(exist_ok=True)


def style(ax):
    ax.grid(color=GRID, lw=0.8)
    ax.tick_params(colors=MUTED)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)


# --- CLINE nullcline: easy vs. hard -----------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
for ax, (f, dt) in zip(axes, [(1.0, 0.1), (10.0, 0.1)]):
    nc = pd.read_csv(RES / "cline_nullclines" / f"nullcline_f={f}_dt={dt}.csv")
    exact = fhn.u_nullcline(nc["u"])
    rmse = np.sqrt(np.mean((nc["v"] - exact) ** 2))
    ax.plot(nc["u"], exact, color=MUTED, lw=2, ls="--", label="exact")
    ax.plot(nc["u"], nc["v"], color=BLUE, lw=2, label="CLINE")
    ax.set_title(f"f = {f:g}, dt = {dt:g}   (RMSE {rmse:.2f})", color=INK, fontsize=11)
    ax.set_xlabel("u", color=MUTED)
    style(ax)
axes[0].set_ylabel("v", color=MUTED)
axes[0].legend(frameon=False)
fig.suptitle("u-nullcline estimated by CLINE from one trajectory", color=INK)
fig.tight_layout()
fig.savefig(config.FIGURES_DIR / "cline_nullcline_examples.png", dpi=200)

# --- numerical derivative error ---------------------------------------------
err = pd.read_csv(RES / "errors.csv")
fig, ax = plt.subplots(figsize=(6, 4))
for f, color in zip(config.F_VALUES, RAMP):
    sel = err[err["f"] == f].sort_values("dt")
    ax.plot(sel["f"] * sel["dt"], sel["derivative_rmse_rel"] * 100, "o-",
            color=color, lw=2, ms=8, markeredgecolor="white", markeredgewidth=2,
            label=f"f = {f:g}")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("f · dt", color=MUTED)
ax.set_ylabel("relative RMSE of numerical du/dt (%)", color=MUTED)
ax.set_title("Finite-difference error grows with f · dt", color=INK)
ax.legend(frameon=False, fontsize=9)
style(ax)
fig.tight_layout()
fig.savefig(config.FIGURES_DIR / "derivative_error.png", dpi=200)
print("saved diagnostics")
