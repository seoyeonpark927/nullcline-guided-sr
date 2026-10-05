"""Nullcline-based SR: fit v = h(u) to the CLINE nullcline.

    python 04_nullcline_sr.py                    # nullclines from 03 (numerical derivative)
    python 04_nullcline_sr.py --true-derivative  # nullclines from the control run

Output: outputs/timescale_sweep/nullcline_sr[_true_derivative]/FHN_f=..._dt=..._seed=.../
"""

import argparse

import pandas as pd

from nullsr import config
from nullsr.sr import make_regressor

parser = argparse.ArgumentParser()
parser.add_argument("--true-derivative", action="store_true")
parser.add_argument("--seeds", type=int, default=len(config.SR_SEEDS))
parser.add_argument("--niterations", type=int, default=500)
args = parser.parse_args()

suffix = "_true_derivative" if args.true_derivative else ""
NULLCLINES = config.OUTPUT_DIR / "timescale_sweep" / f"cline{suffix}"
OUT = config.OUTPUT_DIR / "timescale_sweep" / f"nullcline_sr{suffix}"

for f in config.F_VALUES:
    for dt in config.DT_VALUES:
        df = pd.read_csv(NULLCLINES / f"nullcline_f={f}_dt={dt}.csv")
        X, y = df[["u"]].to_numpy(), df["v"].to_numpy()

        for seed in range(args.seeds):
            print(f"\n=== nullcline SR{suffix}: f={f}, dt={dt:.4g}, seed={seed} ===")
            model = make_regressor(seed, args.niterations, OUT, f"FHN_f={f}_dt={dt}_seed={seed}")
            model.fit(X, y)
            print(model.get_best()["equation"])
