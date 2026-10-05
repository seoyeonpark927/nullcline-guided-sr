"""FitzHugh-Nagumo model with an explicit time-scale separation factor f.

Based on the FHN generator in pyCLINE (Prokop, Billen, Frolov & Gelens, 2025;
MIT License), extended with:
  * a time-scale factor f that speeds up the u-dynamics only,
  * RK4 with internal sub-stepping (dt / 10) for stability at large f,
  * ground-truth derivatives saved next to the trajectory.
"""

import numpy as np
import pandas as pd

from .config import A, B, C, D, EPS


def rhs(state, f):
    u, v = state
    du = f * (-u**3 + C * u**2 + D * u - v)
    dv = EPS * (u - B * v + A)
    return np.array([du, dv])


def u_nullcline(u):
    """v on the u-nullcline (du/dt = 0). Independent of f."""
    return -u**3 + C * u**2 + D * u


def _rk4_step(state, h, f):
    k1 = rhs(state, f)
    k2 = rhs(state + h / 2 * k1, f)
    k3 = rhs(state + h / 2 * k2, f)
    k4 = rhs(state + h * k3, f)
    return state + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def step(state, dt, f, n_substeps=10):
    h = dt / n_substeps
    for _ in range(n_substeps):
        state = _rk4_step(state, h, f)
    return state


def initial_conditions(n_per_axis):
    """Grid of n_per_axis x n_per_axis initial conditions (as in pyCLINE)."""
    u0, v0 = np.meshgrid(
        np.linspace(-1.25, 1.75, n_per_axis),
        np.linspace(-0.75, 1.75, n_per_axis),
    )
    return np.stack([u0.ravel(), v0.ravel()])  # shape (2, n_trajectories)


def generate(f, dt, n_steps, n_ic_per_axis=1, n_substeps=10):
    """Simulate all initial conditions and return one long DataFrame.

    Columns: sim (1-based), time, u, v, du_true, dv_true.
    n_substeps = 1 is plain RK4 with step dt (inaccurate when f * dt > ~0.5).
    """
    x0 = initial_conditions(n_ic_per_axis)
    n_traj = x0.shape[1]

    traj = np.empty((n_steps, 2, n_traj))
    traj[0] = x0
    for n in range(1, n_steps):
        traj[n] = step(traj[n - 1], dt, f, n_substeps)

    time = np.arange(n_steps) * dt
    frames = []
    for k in range(n_traj):
        u, v = traj[:, 0, k], traj[:, 1, k]
        du, dv = rhs(np.array([u, v]), f)
        frames.append(pd.DataFrame({
            "sim": k + 1, "time": time, "u": u, "v": v,
            "du_true": du, "dv_true": dv,
        }))
    return pd.concat(frames, ignore_index=True)


def add_numerical_derivative(df, dt, var="u"):
    """Central-difference derivative computed separately for each trajectory."""
    df = df.sort_values(["sim", "time"]).reset_index(drop=True)
    df[f"d{var}_dt"] = df.groupby("sim")[var].transform(lambda x: np.gradient(x, dt))
    return df
