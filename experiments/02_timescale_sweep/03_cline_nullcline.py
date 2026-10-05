"""Estimate the u-nullcline with CLINE for every (f, dt).

    python 03_cline_nullcline.py                    # numerical derivative
    python 03_cline_nullcline.py --true-derivative  # control: exact du/dt

Output: outputs/timescale_sweep/cline[_true_derivative]/nullcline_f=<f>_dt=<dt>.csv
"""

import argparse

import pandas as pd

from nullsr import config
from nullsr.cline import estimate_nullcline

parser = argparse.ArgumentParser()
parser.add_argument("--true-derivative", action="store_true")
parser.add_argument("--epochs", type=int, default=4000)
args = parser.parse_args()

DATA = config.DATA_DIR / "timescale_sweep"
name = "cline_true_derivative" if args.true_derivative else "cline"
OUT = config.OUTPUT_DIR / "timescale_sweep" / name
OUT.mkdir(parents=True, exist_ok=True)

for f in config.F_VALUES:
    for dt in config.DT_VALUES:
        df = pd.read_csv(DATA / config.data_filename(f, dt))
        nullcline, losses, test_loss = estimate_nullcline(
            df, config.CLINE_SEED, epochs=args.epochs, batch_size=64,
            derivative="true" if args.true_derivative else "numerical",
        )
        nullcline.to_csv(OUT / f"nullcline_f={f}_dt={dt}.csv", index=False)
        losses.to_csv(OUT / f"losses_f={f}_dt={dt}.csv", index=False)
        print(f"f={f}, dt={dt:.4g}: test loss {test_loss:.3e}")
