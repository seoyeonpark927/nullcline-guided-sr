"""Latest Hybrid SR results (dt = 0.01, seed 0).

Top:    CLINE nullclines estimated from 100 trajectories vs. the exact u-nullcline.
Bottom: largest relative coefficient error of the recovered du/dt equation,
        Direct SR (500 iterations) vs. Hybrid SR (2000 iterations).

Reads results/hybrid/, writes figures/hybrid_latest.png.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from nullsr import config, fhn
from nullsr.evaluate import TARGETS, coefficients, parse

RES = config.RESULTS_DIR / "hybrid"
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3dd"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # fixed categorical order

summary = pd.read_csv(RES / "summary.csv")
main = summary[(summary["stage"] == "main") & (summary["dt"] == 0.01) & (summary["seed"] == 0)]

RUNS = [  # (label, method, lambda, iterations)
    ("Direct SR (500 it.)", "direct_sr", None, 500),
    ("Hybrid SR, λ = 0.1 (2000 it.)", "hybrid_sr", 0.1, 2000),
    ("Hybrid SR, λ = 1 (2000 it.)", "hybrid_sr", 1.0, 2000),
]


def max_coefficient_error(equation, f):
    """Largest |c - c_true| / |c_true| over the four target terms, in %."""
    if not isinstance(equation, str):
        return np.nan
    coef = coefficients(parse(equation))
    return max(abs(coef.get(m, 0.0) - t * f) / abs(t * f)
               for m, t in TARGETS["direct"].items()) * 100


fig = plt.figure(figsize=(11, 6.2))
grid = fig.add_gridspec(2, 5, height_ratios=[1, 1.15], hspace=0.45, wspace=0.08)

for i, f in enumerate(config.F_VALUES):
    ax = fig.add_subplot(grid[0, i])
    nc = pd.read_csv(RES / "cline_nullclines" / f"nullcline_f={f}_dt=0.01.csv")
    rmse = np.sqrt(np.mean((nc["v"] - fhn.u_nullcline(nc["u"])) ** 2))
    ax.plot(nc["u"], fhn.u_nullcline(nc["u"]), color=MUTED, lw=2, ls="--")
    ax.plot(nc["u"], nc["v"], color=SERIES[0], lw=2)
    ax.set_title(f"f = {f:g}   RMSE {rmse:.3f}", fontsize=10, color=INK)
    ax.set_xlabel("u", color=MUTED)
    if i == 0:
        ax.set_ylabel("v", color=MUTED)
        ax.plot([], [], color=MUTED, lw=2, ls="--", label="exact")
        ax.plot([], [], color=SERIES[0], lw=2, label="CLINE")
        ax.legend(frameon=False, fontsize=9, loc="upper left")
    else:
        ax.set_yticklabels([])
    ax.grid(color=GRID, lw=0.8)
    ax.tick_params(colors=MUTED, labelsize=8)

ax = fig.add_subplot(grid[1, 1:4])
for (label, method, lam, iters), color in zip(RUNS, SERIES):
    sel = main[(main["method"] == method) & (main["iter"] == iters)]
    if lam is not None:
        sel = sel[sel["lambda"] == lam]
    sel = sel.sort_values("f")
    err = [max_coefficient_error(e, f) for e, f in zip(sel["equation_strict"], sel["f"])]
    ax.plot(sel["f"], err, "o-", color=color, lw=2, ms=8, label=label,
            markeredgecolor="white", markeredgewidth=2)
ax.set_xticks(config.F_VALUES)
ax.set_xlabel("time-scale separation f", color=MUTED)
ax.set_ylabel("max coefficient error (%)", color=MUTED)
ax.set_ylim(bottom=0)
ax.set_title("Coefficient error of the recovered du/dt (every run passes the strict criterion)",
             fontsize=10, color=INK)
ax.grid(color=GRID, lw=0.8)
ax.tick_params(colors=MUTED)
ax.legend(frameon=False, fontsize=9)
for side in ["top", "right"]:
    ax.spines[side].set_visible(False)

fig.suptitle("Hybrid SR, latest runs (dt = 0.01, seed 0, 100 trajectories)", color=INK)
config.FIGURES_DIR.mkdir(exist_ok=True)
fig.savefig(config.FIGURES_DIR / "hybrid_latest.png", dpi=200, bbox_inches="tight")
print("saved figures/hybrid_latest.png")
