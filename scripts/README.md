# scripts/

Helper scripts for the autotrain monitoring loop. Designed to be re-runnable,
side-effect-light, and produce identically-shaped output across ticks so
trends are easy to read.

## tick_status.py

The primary per-tick reporter. Reads the most recent dcm session log, parses
the latest `[STATS]` line plus all `[ARENA]` / `[CHECKPOINT]` / `[ALARM]`
events, and emits a fixed-format block with:

- **Header**: elapsed, steps, pid, log file
- **Per-tick deltas**: Δt and Δsteps since previous invocation
- **Health table**: 11 metrics with absolute value, in-band/watch/OUT label,
  Δ-since-prev, and a one-liner explaining each
- **Throughput / replay / diversity** rollup
- **Arena trajectory** (last 12 arenas with W/D/L, score, elo CI, status)
- **Checkpoints** (last 5 saves, trigger-tagged)
- **Alarms** (last 5, excluding the routine `legal-mass probe ok` line)
- **Hard-reject status** (clear / which criterion tripped)

History persists to `scripts/.tick_history.jsonl` (capped at 500 entries),
which is what makes per-tick deltas work even when the conversation context
is fresh. The file is the source of truth for `trend.py`.

```sh
python3 scripts/tick_status.py            # normal use
python3 scripts/tick_status.py --no-record # don't append to history
python3 scripts/tick_status.py --log <path>
```

Exit codes: `0` healthy, `1` hard-reject tripped, `2` no STATS yet,
`3` app process not running.

## trend.py

Compact one-line-per-tick view of the rolling history.

```sh
python3 scripts/trend.py                   # last 20 ticks, all key metrics
python3 scripts/trend.py --tail 60
python3 scripts/trend.py --metric pEnt     # single metric trajectory
```

## arena_summary.py

Walks every `[ARENA]` line in the most recent session log; prints per-arena
W/D/L/score/elo plus a 5-arena rolling score average.

```sh
python3 scripts/arena_summary.py
```

## sessions_summary.py

Lists `.dcmsession` autosaves under `~/Library/Application Support/DrewsChessMachine/Sessions/`,
parses each `metadata.json`, and emits one line per save with trigger,
ModelID lineage, training time at save, arena count + promotions.

```sh
python3 scripts/sessions_summary.py
python3 scripts/sessions_summary.py --tail 50
python3 scripts/sessions_summary.py --json
```

## analyze_session_log.py

Slices a DCM session log to surface UI-responsiveness and training-throughput
trajectories over the life of a run. Built for one specific question — *"this
run got slow over time, what changed and when?"* — by bucketing `[TICK-SLOW]`
timings, `[STATS]` throughput, RSS memory growth, and `[ALARM]` events into
per-hour trends.

```sh
python3 scripts/analyze_session_log.py            # most-recent dcm_log_*.txt
python3 scripts/analyze_session_log.py <path>     # a specific log
```

## session_failure_analysis.py

Parses one session log and prints a regime-change report focused on *"when did
this start, what changed, and did playing strength actually collapse or just
the trainer numerics?"* — extracts `[STATS]`, `[PARAM]`, `[ALARM]`, and arena
records and looks for the onset of failure.

```sh
python3 scripts/session_failure_analysis.py                 # most-recent log
python3 scripts/session_failure_analysis.py --log <path>
python3 scripts/session_failure_analysis.py --json          # structured output
```

## model_lineage_report.py

Scans **all** session logs for a single model lineage (a trainer / candidate /
champion / model ID) and reports where it appears across files, so you can tell
whether a failure originated in the current session or was inherited from an
earlier one. Prints first/last `[STATS]`, first `[ALARM]`, and arena hits per
log.

```sh
python3 scripts/model_lineage_report.py 20260506-5-aoTz-2
```

## curate_lichess_probes.py

One-time / occasional generator for the fixed Lichess puzzle probe sets. Reads
the Lichess puzzle CSV (`lichess_db_puzzle.csv` from database.lichess.org),
applies quality filters (popularity ≥ 80, ≥ 200 plays, rating-deviation ≤ 90),
samples deterministically per theme, and emits a probe JSON. Two presets: `legacy`
reproduces the bundled `lichess_probes_200.json` exactly (do not change its
config — it's the permanent 200-puzzle yardstick); `wide` builds the broad,
rating-stratified `lichess_probes_wide.json` set in parallel without touching the
legacy file. Unlike the other scripts here, this one needs `python-chess`
installed (imported lazily, only when actually generating a set).

```sh
python3 scripts/curate_lichess_probes.py --preset legacy
python3 scripts/curate_lichess_probes.py --preset wide --csv /tmp/lichess_db_puzzle.csv
python3 scripts/curate_lichess_probes.py --preset wide --out <path> --snapshot <db-date>
```
