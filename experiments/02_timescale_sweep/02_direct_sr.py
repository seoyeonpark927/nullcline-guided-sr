"""Direct SR: fit du/dt = F(u, v) to the trajectory.

    python 02_direct_sr.py                    # numerical derivative, all (f, dt), 10 seeds
    python 02_direct_sr.py --true-derivative  # control: exact du/dt as the target

Output: outputs/timescale_sweep/direct_sr[_true_derivative]/FHN_f=..._dt=..._seed=.../
"""

import argparse

import pandas as pd

from nullsr import config, fhn
from nullsr.sr import make_regressor

parser = argparse.ArgumentParser()
parser.add_argument("--true-derivative", action="store_true")
parser.add_argument("--seeds", type=int, default=len(config.SR_SEEDS))
parser.add_argument("--niterations", type=int, default=500)
args = parser.parse_args()

DATA = config.DATA_DIR / "timescale_sweep"
name = "direct_sr_true_derivative" if args.true_derivative else "direct_sr"
OUT = config.OUTPUT_DIR / "timescale_sweep" / name

for f in config.F_VALUES:
    for dt in config.DT_VALUES:
        df = fhn.add_numerical_derivative(pd.read_csv(DATA / config.data_filename(f, dt)), dt)
        X = df[["u", "v"]].to_numpy()
        y = df["du_true" if args.true_derivative else "du_dt"].to_numpy()

        for seed in range(args.seeds):
            print(f"\n=== {name}: f={f}, dt={dt:.4g}, seed={seed} ===")
            model = make_regressor(seed, args.niterations, OUT, f"FHN_f={f}_dt={dt}_seed={seed}")
            model.fit(X, y)
            print(model.get_best()["equation"])
