"""Direct SR vs. Hybrid SR on the same trajectory sample.

    python 03_run_sr.py --method direct
    python 03_run_sr.py --method hybrid                     # CLINE nullcline
    python 03_run_sr.py --method hybrid --true-nullcline    # exact nullcline (control)
    python 03_run_sr.py --method hybrid --niterations 2000  # longer search

Every method sees the same N_DATA trajectory points (sampled with the SR seed)
and the same PySR settings. Hybrid SR additionally gets the nullcline points
(500 from CLINE) with target du/dt = 0 and the loss
    L = L_data + lambda * L_nullcline.
The true-nullcline control keeps the same u grid and replaces only v by the
exact value v = -u^3 + u^2 + u.

Output: outputs/hybrid/<run_name>/FHN_f=..._dt=..._[lambda=..._]seed=..._iter=.../
"""

import argparse

import numpy as np
import pandas as pd

from nullsr import config, fhn
from nullsr.sr import hybrid_loss, make_regressor

parser = argparse.ArgumentParser()
parser.add_argument("--method", choices=["direct", "hybrid"], required=True)
parser.add_argument("--true-nullcline", action="store_true")
parser.add_argument("--dt", type=float, nargs="+", default=[0.01])
parser.add_argument("--lambdas", type=float, nargs="+", default=[0.1, 1.0])
parser.add_argument("--seeds", type=int, default=1)
parser.add_argument("--niterations", type=int, default=500)
parser.add_argument("--n-data", type=int, default=100_000)
args = parser.parse_args()

DATA = config.DATA_DIR / "hybrid"
NULLCLINES = config.OUTPUT_DIR / "hybrid" / "cline"
if args.method == "direct":
    run_name = "direct_sr"
else:
    run_name = "hybrid_sr_true_nullcline" if args.true_nullcline else "hybrid_sr"
OUT = config.OUTPUT_DIR / "hybrid" / run_name
lambdas = [None] if args.method == "direct" else args.lambdas

for f in config.F_VALUES:
    for dt in args.dt:
        traj = fhn.add_numerical_derivative(pd.read_csv(DATA / config.data_filename(f, dt)), dt)

        if args.method == "hybrid":
            nc = pd.read_csv(NULLCLINES / f"nullcline_f={f}_dt={dt}.csv")
            if args.true_nullcline:
                nc["v"] = fhn.u_nullcline(nc["u"])
            X_null = np.column_stack([nc["u"], nc["v"], np.ones(len(nc))])
            y_null = np.zeros(len(nc))

        for seed in range(args.seeds):
            sample = traj.sample(n=args.n_data, random_state=seed)
            X = sample[["u", "v"]].to_numpy()
            y = sample["du_dt"].to_numpy()

            for lam in lambdas:
                tag = f"FHN_f={f}_dt={dt}_" + (f"lambda={lam}_" if lam is not None else "")
                tag += f"seed={seed}_iter={args.niterations}"
                print(f"\n=== {run_name}: {tag} ===")

                if lam is None:
                    model = make_regressor(seed, args.niterations, OUT, tag,
                                           variable_names=["u", "v"])
                    model.fit(X, y)
                else:
                    X_all = np.vstack([np.column_stack([X, np.zeros(len(X))]), X_null])
                    y_all = np.concatenate([y, y_null])
                    model = make_regressor(seed, args.niterations, OUT, tag,
                                           variable_names=["u", "v", "marker"],
                                           loss_function=hybrid_loss(lam))
                    model.fit(X_all, y_all)
                print(model.get_best()["equation"])
