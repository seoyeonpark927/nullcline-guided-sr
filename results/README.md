# Results

Summary tables of the archived experiment runs (July–October 2026). Each row
is one PySR run, re-scored with `src/nullsr/evaluate.py` from its
`hall_of_fame.csv`. Raw runs are not tracked (see `.gitignore`).

| File | Runs | Produced by |
|---|---|---|
| `timescale_sweep/direct_sr.csv` | 5 f × 5 dt × 10 seeds | `experiments/02_timescale_sweep/02_direct_sr.py` |
| `timescale_sweep/nullcline_sr.csv` | 5 f × 5 dt × 10 seeds | `03_cline_nullcline.py` → `04_nullcline_sr.py` |
| `timescale_sweep/*_true_derivative.csv` | 5 f, dt = 0.1, seed 0 | same scripts with `--true-derivative` |
| `timescale_sweep/errors.csv` | 5 f × 5 dt | CLINE nullcline RMSE; numerical vs. exact du/dt RMSE |
| `hybrid/summary.csv` | dt ∈ {0.01, 0.1}, seed 0 (some extra seeds) | `experiments/03_hybrid_sr/03_run_sr.py` |

Columns: `success_strict` / `success_relaxed` (0/1) and the simplest passing
equation and its complexity for each criterion. The criteria are defined in
`src/nullsr/evaluate.py`:

* **strict** – exactly the target monomials (an extra constant is allowed),
  every coefficient within ±15 % of its true value.
* **relaxed** – exactly the target monomials with a negative cubic term;
  coefficient values are not checked.

Notes on provenance

* The time-scale sweep data were generated with plain RK4 (one step per dt),
  except f = 15, 20 at dt = 0.1, where plain RK4 diverges and 10 sub-steps
  were used. `01_generate_data.py --legacy` reproduces these files.
* In `hybrid/summary.csv`, an empty `iter` means 500 PySR iterations.
