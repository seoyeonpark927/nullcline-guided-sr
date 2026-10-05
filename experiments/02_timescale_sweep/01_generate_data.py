"""Generate one FHN trajectory for every (f, dt) pair.

    python 01_generate_data.py           # RK4 with 10 sub-steps per dt (accurate)
    python 01_generate_data.py --legacy  # data used for results/timescale_sweep

--legacy reproduces the archived sweep data exactly: plain RK4 with step dt,
switching to 10 sub-steps only where plain RK4 diverges (f = 15, 20 at dt = 0.1).
Plain RK4 under-estimates the limit-cycle amplitude by ~8-9 % when f * dt > 0.8
(see README, Limitations).

Output: data/timescale_sweep/FHN_eps=0.3_a=0.0_f=<f>_dt=<dt>.csv
"""

import argparse

import numpy as np

from nullsr import config, fhn

parser = argparse.ArgumentParser()
parser.add_argument("--legacy", action="store_true")
args = parser.parse_args()

DATA = config.DATA_DIR / "timescale_sweep"
DATA.mkdir(parents=True, exist_ok=True)

for f in config.F_VALUES:
    for dt in config.DT_VALUES:
        n_sub = 1 if args.legacy else 10
        with np.errstate(over="ignore", invalid="ignore"):
            df = fhn.generate(f, dt, config.N_STEPS, n_ic_per_axis=1, n_substeps=n_sub)
        if not np.isfinite(df[["u", "v"]].to_numpy()).all():
            n_sub = 10
            df = fhn.generate(f, dt, config.N_STEPS, n_ic_per_axis=1, n_substeps=n_sub)
        df.to_csv(DATA / config.data_filename(f, dt), index=False)
        print(f"saved f={f}, dt={dt:.4g} (RK4 sub-steps: {n_sub})")
