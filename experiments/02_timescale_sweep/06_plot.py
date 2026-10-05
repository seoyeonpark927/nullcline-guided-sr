"""Success-rate heatmaps (f x dt) for Direct and Nullcline-based SR.

Reads results/timescale_sweep/*.csv, writes figures/timescale_success_<criterion>.png.
"""

import matplotlib.pyplot as plt
import pandas as pd

from nullsr import config

RES = config.RESULTS_DIR / "timescale_sweep"
DT_ORDER = [round(dt, 6) for dt in config.DT_VALUES]
F_ORDER = config.F_VALUES


def success_matrix(table, criterion):
    return (table.groupby(["dt", "f"])[f"success_{criterion}"].mean()
            .unstack().reindex(index=DT_ORDER, columns=F_ORDER))


def draw(ax, matrix, title):
    im = ax.imshow(matrix.to_numpy(), cmap="viridis", vmin=0, vmax=1, aspect="auto")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix.iat[i, j]
            if pd.notna(val):
                ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                        color="black" if val > 0.6 else "white", fontsize=10)
    ax.set_xticks(range(len(F_ORDER)), [f"{f:g}" for f in F_ORDER])
    ax.set_yticks(range(len(DT_ORDER)), [f"{dt:.3g}" for dt in DT_ORDER])
    ax.set_xlabel("time-scale separation f")
    ax.set_ylabel("sampling interval dt")
    ax.set_title(title)
    return im


config.FIGURES_DIR.mkdir(exist_ok=True)
for criterion in ["strict", "relaxed"]:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for ax, (name, label) in zip(axes, [("direct_sr", "Direct SR"),
                                        ("nullcline_sr", "Nullcline-based SR")]):
        im = draw(ax, success_matrix(pd.read_csv(RES / f"{name}.csv"), criterion), label)
    fig.colorbar(im, ax=axes, label="success rate (10 seeds)")
    fig.suptitle(f"Equation recovery, {criterion} criterion")
    fig.savefig(config.FIGURES_DIR / f"timescale_success_{criterion}.png", dpi=200)
    print("saved", criterion)
