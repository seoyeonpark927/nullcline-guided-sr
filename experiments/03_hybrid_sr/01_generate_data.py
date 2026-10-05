"""Generate FHN data from a 10 x 10 grid of initial conditions (100 trajectories,
1,000,000 points per (f, dt)).

Output: data/hybrid/FHN_eps=0.3_a=0.0_f=<f>_dt=<dt>.csv
"""

from nullsr import config, fhn

DT_VALUES = [0.01, 0.1]
DATA = config.DATA_DIR / "hybrid"
DATA.mkdir(parents=True, exist_ok=True)

for f in config.F_VALUES:
    for dt in DT_VALUES:
        df = fhn.generate(f, dt, config.N_STEPS, n_ic_per_axis=10)
        df.to_csv(DATA / config.data_filename(f, dt), index=False)
        print(f"saved f={f}, dt={dt}: {df['sim'].nunique()} trajectories, {len(df):,} points")
