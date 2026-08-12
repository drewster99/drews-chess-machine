# This directory is SUPERSEDED — do not append to it

As of **2026-08-11** the v5 training record lives in the repo's dashboard tracker:

- **Source of truth:** `documentation/dashboards/data/v5.csv` (856 rows, cum step
  1,000 → 859,769, all 8 segments across both machines).
- **Config:** `documentation/dashboards/registry.json` → `runs.v5`.
- **Full narrative:** `documentation/v5-lineage.md` — read this first.

Everything here was imported from it and is kept as the durable pre-image:

| file | status |
|---|---|
| `new_ckpts*.jsonl` | **imported** (761 probes). Retained as pre-image; do not append. |
| `records.json` | superseded — records are now read off the CSV. |
| `build_dashboard.py`, `build_html.py`, `build_table.py` | superseded by `master.py`. |
| `v5-strength.html` | superseded by `dcm_master.html` (3 axes). |
| `HANDOFF.md` | durable content folded into `v5-lineage.md`; its live-process instructions are obsolete — no run has been active since 2026-08-05 00:20. |
| `checkpoint_inventory.json` | **current** — SHA-256 manifest of all 650 surviving checkpoints. |

⚠️ Cumulative step numbers in the files here are **100,320 lower** than the unified
axis: this monitor numbered run 1 from 0 and omitted the original machine's first three
segments. The pElo record reported here as "cum 538.5k" is cum **638,826** in the CSV.

New marks go through `replay.py track v5` or `replay.py import-probes`.
