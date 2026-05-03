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
