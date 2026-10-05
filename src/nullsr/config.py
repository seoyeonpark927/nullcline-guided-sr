"""Shared experiment settings and paths."""

from pathlib import Path

# Repository layout
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"          # generated, not tracked by git
OUTPUT_DIR = ROOT / "outputs"     # PySR / CLINE runs, not tracked by git
RESULTS_DIR = ROOT / "results"    # small summary tables, tracked by git
FIGURES_DIR = ROOT / "figures"    # figures used in the README, tracked by git

# FitzHugh-Nagumo parameters (Prokop et al., iScience 2024)
#   du/dt = f * (-u^3 + C u^2 + D u - v)
#   dv/dt = EPS * (u - B v + A)
C = 1.0
D = 1.0
EPS = 0.3
B = 0.5
A = 0.0

# Time-scale separation factors and sampling intervals (log-spaced 0.1 -> 0.01)
F_VALUES = [1.0, 5.0, 10.0, 15.0, 20.0]
DT_VALUES = [
    0.1,
    0.05623413251903491,
    0.03162277660168379,
    0.01778279410038923,
    0.01,
]

N_STEPS = 10_000      # time steps per trajectory
SR_SEEDS = range(10)  # PySR random seeds
CLINE_SEED = 42       # seed for the CLINE network and data split


def data_filename(f, dt):
    return f"FHN_eps={EPS}_a={A}_f={f}_dt={dt}.csv"
