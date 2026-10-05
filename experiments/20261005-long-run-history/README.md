# 2026-10-05 — The long corpus-replay runs: training settings, heads, head health, matched-time comparison

Summaries: [E-0012](../summaries/E-0012_2026-10-04_long-run-recipes.html) (settings and heads),
[E-0014](../summaries/E-0014_2026-10-05_head-health-long-runs.html) (head health),
[E-0015](../summaries/E-0015_2026-10-05_nt8y-equal-time.html) (nt8y at matched wall time).

`long_run_history.py` reads `documentation/dashboards/registry.json` and `data/<run>.csv`, the
session logs the registry names under `~/Library/Logs/DrewsChessMachine` (a missing log is
reported as missing, never guessed), and the R7/R8 probe table. `results.md` is its output on
2026-10-05.

Reproduce, from the repo root on the machine that holds the logs:
`python3 experiments/20261005-long-run-history/long_run_history.py > experiments/20261005-long-run-history/results.md`.
`results.md` was produced with `documentation/dashboards/data/nt8y.csv` as updated by the dashboard tracker on
2026-10-05 01:30 (it added the 291,662 row and two columns); the nt8y rows of section 3 differ slightly from the
version of that file committed before then.
