"""Estimate the u-nullcline from trajectories with CLINE.

CLINE (Prokop et al., 2025) trains a network  v = g(u, du/dt)  and reads the
nullcline off as  g(u, 0).  Network construction and training use the
unmodified pyCLINE package; data preparation is re-implemented here so that
  * several trajectories share one global min-max normalization,
  * derivatives are computed per trajectory,
  * the ground-truth derivative can replace the numerical one (control run).
"""

import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pyCLINE.recovery_methods.nn_training import configure_FFNN_model, train_FFNN_model

VAL_MIN, VAL_MAX = -1.0, 1.0
INPUTS = ["norm u", "d normu/dt"]
TARGET = ["norm v"]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def prepare(df, derivative="numerical", tmin=None):
    """Normalize u, v to [-1, 1] and attach d(norm u)/dt.

    derivative: "numerical" (np.gradient per trajectory) or
                "true" (ground-truth du/dt, rescaled to normalized units).
    """
    df = df.sort_values(["sim", "time"]).reset_index(drop=True)
    if tmin is not None:
        df = df[df["time"] >= tmin].reset_index(drop=True)

    coef = pd.DataFrame(index=["min", "max"])
    for var in ["u", "v"]:
        lo, hi = df[var].min(), df[var].max()
        coef[var] = [lo, hi]
        df[f"norm {var}"] = VAL_MIN + (df[var] - lo) / (hi - lo) * (VAL_MAX - VAL_MIN)

    if derivative == "numerical":
        parts = []
        for _, sim in df.groupby("sim", sort=True):
            dt = sim["time"].iloc[1] - sim["time"].iloc[0]
            sim = sim.copy()
            sim["d normu/dt"] = np.gradient(sim["norm u"].to_numpy(), dt)
            parts.append(sim)
        df = pd.concat(parts, ignore_index=True)
    elif derivative == "true":
        scale = (VAL_MAX - VAL_MIN) / (coef["u"]["max"] - coef["u"]["min"])
        df["d normu/dt"] = df["du_true"] * scale
    else:
        raise ValueError(f"unknown derivative: {derivative}")

    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=INPUTS + TARGET)
    return df.reset_index(drop=True), coef


def split(df, seed, train_frac=0.7, test_frac=0.15):
    """Shuffle, then split 70 / 15 / 15 into train / test / validation."""
    df = df.sample(frac=1, random_state=seed)
    n_train = int(len(df) * train_frac)
    n_test = int(len(df) * test_frac)
    parts = [df.iloc[:n_train], df.iloc[n_train:n_train + n_test], df.iloc[n_train + n_test:]]
    return [(p[INPUTS], p[TARGET]) for p in parts]


def estimate_nullcline(df, seed, epochs, batch_size, derivative="numerical", tmin=None):
    """Train CLINE and return (nullcline DataFrame with columns u, v; losses)."""
    set_seed(seed)
    prepared, coef = prepare(df, derivative=derivative, tmin=tmin)
    (x_tr, y_tr), (x_te, y_te), (x_va, y_va) = split(prepared, seed)

    model, optimizer, loss_fn = configure_FFNN_model(
        Nin=len(INPUTS), Nout=len(TARGET), Nlayers=3, Nnodes=64,
        lr=1e-4, activation=nn.SiLU, summary=False,
    )
    train_loss, val_loss, test_loss, predictions, _, _ = train_FFNN_model(
        model=model, optimizer=optimizer, loss_fn=loss_fn,
        input_train=x_tr, target_train=y_tr, input_test=x_te, target_test=y_te,
        validation_data=(x_va, y_va), epochs=epochs, batch_size=batch_size,
        plot_loss=False, device="cpu", save_evolution=True, method="derivative",
        minimal_value=VAL_MIN, maximal_value=VAL_MAX,
    )

    # predictions[-1]: nullcline after the last epoch on a uniform grid in norm u
    u = np.linspace(coef["u"]["min"], coef["u"]["max"], predictions.shape[1])
    v_lo, v_hi = coef["v"]["min"], coef["v"]["max"]
    v = (predictions[-1] - VAL_MIN) * (v_hi - v_lo) / (VAL_MAX - VAL_MIN) + v_lo

    losses = pd.DataFrame({"train": train_loss, "validation": val_loss})
    return pd.DataFrame({"u": u, "v": v}), losses, test_loss
