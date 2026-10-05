"""Estimate the u-nullcline with CLINE from all 100 trajectories.

Output: outputs/hybrid/cline/nullcline_f=<f>_dt=<dt>.csv (+ training losses)
"""

import argparse

import pandas as pd

from nullsr import config
from nullsr.cline import estimate_nullcline

parser = argparse.ArgumentParser()
parser.add_argument("--dt", type=float, nargs="+", default=[0.01])
args = parser.parse_args()

DATA = config.DATA_DIR / "hybrid"
OUT = config.OUTPUT_DIR / "hybrid" / "cline"
OUT.mkdir(parents=True, exist_ok=True)

for f in config.F_VALUES:
    for dt in args.dt:
        df = pd.read_csv(DATA / config.data_filename(f, dt))
        nullcline, losses, test_loss = estimate_nullcline(
            df, config.CLINE_SEED, epochs=500, batch_size=256)
        nullcline.to_csv(OUT / f"nullcline_f={f}_dt={dt}.csv", index=False)
        losses.to_csv(OUT / f"losses_f={f}_dt={dt}.csv", index=False)
        print(f"f={f}, dt={dt}: test loss {test_loss:.3e}")
