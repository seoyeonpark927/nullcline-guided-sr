"""Score Direct, Hybrid and true-nullcline Hybrid runs.

Output: results/hybrid/summary.csv (one row per run, strict + relaxed)
"""

import pandas as pd

from nullsr import config
from nullsr.evaluate import score_runs

RUNS = config.OUTPUT_DIR / "hybrid"
RES = config.RESULTS_DIR / "hybrid"
RES.mkdir(parents=True, exist_ok=True)

tables = []
for method in ["direct_sr", "hybrid_sr", "hybrid_sr_true_nullcline"]:
    if (RUNS / method).exists():
        table = score_runs(RUNS / method, "direct")
        table.insert(0, "method", method)
        tables.append(table)

summary = pd.concat(tables, ignore_index=True)
summary.to_csv(RES / "summary.csv", index=False)
cols = ["method", "f", "dt", "lambda", "iter", "seed", "success_strict", "success_relaxed"]
print(summary[[c for c in cols if c in summary]].to_string(index=False))
