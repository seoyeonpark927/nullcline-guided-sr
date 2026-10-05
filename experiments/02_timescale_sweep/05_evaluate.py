"""Score all sweep runs and measure nullcline / derivative errors.

Output (results/timescale_sweep/)
  direct_sr.csv, nullcline_sr.csv            one row per run, strict + relaxed
  *_true_derivative.csv                       same for the control runs (if present)
  errors.csv                                  per (f, dt): CLINE nullcline RMSE and
                                              numerical-vs-exact du/dt RMSE
"""

import numpy as np
import pandas as pd

from nullsr import config, fhn
from nullsr.evaluate import score_runs

RUNS = config.OUTPUT_DIR / "timescale_sweep"
DATA = config.DATA_DIR / "timescale_sweep"
RES = config.RESULTS_DIR / "timescale_sweep"
RES.mkdir(parents=True, exist_ok=True)

for name, kind in [
    ("direct_sr", "direct"),
    ("nullcline_sr", "nullcline"),
    ("direct_sr_true_derivative", "direct"),
    ("nullcline_sr_true_derivative", "nullcline"),
]:
    if not (RUNS / name).exists():
        continue
    table = score_runs(RUNS / name, kind)
    table.to_csv(RES / f"{name}.csv", index=False)
    print(f"{name}: strict {table.success_strict.sum()}/{len(table)}, "
          f"relaxed {table.success_relaxed.sum()}/{len(table)}")

rows = []
for f in config.F_VALUES:
    for dt in config.DT_VALUES:
        row = {"f": f, "dt": round(dt, 6)}
        path = RUNS / "cline" / f"nullcline_f={f}_dt={dt}.csv"
        if path.exists():
            nc = pd.read_csv(path)
            row["nullcline_rmse"] = np.sqrt(np.mean((nc["v"] - fhn.u_nullcline(nc["u"])) ** 2))
        path = DATA / config.data_filename(f, dt)
        if path.exists():
            df = fhn.add_numerical_derivative(pd.read_csv(path), dt)
            row["derivative_rmse"] = np.sqrt(np.mean((df["du_dt"] - df["du_true"]) ** 2))
        rows.append(row)
errors = pd.DataFrame(rows)
errors.to_csv(RES / "errors.csv", index=False)
print(errors.round(4).to_string(index=False))
