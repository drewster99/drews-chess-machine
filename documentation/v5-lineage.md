# The v5 lineage — consolidated record

_Consolidated 2026-08-11. Supersedes the split tracking that preceded it._

v5 is the 8.45M-param, 5-block 7×7 128ch lc0-style net (SE + ReZero, WDL 3-logit value
head, `basic30` 30-plane encoding, bfloat16, 4864 policy logits, single forward pass —
no search). Its training history is **one continuous weight lineage across 8 segments
and 2 machines**, ending at cumulative step **859,769** with **110.95M corpus games**
consumed (≈5.30 passes over the 20.94M-game corpus).

Until this consolidation that history lived in two systems that shared no axis:
`documentation/dashboards/data/v5.csv` held only the first three segments (the original
machine), while five continuation segments existed solely as probe JSONL in a bundle
under `~/Downloads`. They are now one series of **856 rows** in that same CSV.

---

## 1. Why a raw step number means nothing here

Every resumed segment **restarts its step counter at 1**, and the corpus-replay runner's
`--enumerate-checkpoints` names files from that segment-local counter. Five segments
therefore competed for the same filenames, and later runs overwrote earlier ones in
place: **four distinct files have been named `v5-cont-replay-step1000.safetensors`.**

A filename identifies nothing. A checkpoint's identity is its safetensors
`__metadata__`: `model_id` (minted per segment) + `training_step`. Use
`documentation/dashboards/ckpt_inventory.py` to read that, never `ls`.

The only comparable axis is the cumulative one: `cum_step = segment.cumstep_base + training_step`.

## 2. The segment chain

Verified end to end — each run's `[REPLAY] start-model:` line names the prior segment's
final checkpoint *and* its modelID:

```
v5-checkpoint (20260629-1-Uf4p, step 39419, parent 20260628-9-OdUt)
  -> run1 (20260703-1-Dg5v) -> run2 (20260714-1-h7vI) -> run3 (20260729-1-VZ2j)
  -> run4 (20260802-2-Xuub) -> run5 (20260805-1-0pTW)
```

| seg | label | device | `cumstep_base` | steps | `games_base` | probes | session log |
|---:|---|---|---:|---:|---:|---:|---|
| 0 | wd1e-4 | M5 | 0 | 45,441 | — | *in v5.csv* | `dcm_log_20260627-201127.txt` ✗ |
| 1 | wd5e-4 | M5 | 45,441 | 15,460 | — | *in v5.csv* | `dcm_log_20260628-114648.txt` ✗ |
| 2 | m0.93 | M5 | 60,901 | 39,419 | — | *in v5.csv* | `dcm_log_20260628-183831.txt` ✗ |
| 3 | cont-run1 | M4 Pro | 100,320 | 268,506 | 12,933,495 | 269 | `dcm_log_20260702-201756.txt` |
| 4 | cont-run2 | M4 Pro | 368,826 | 336,610 | 47,582,603 | 337 | `dcm_log_20260713-195047.txt` |
| 5 | cont-run3 | M4 Pro | 705,436 | 106,333 | 90,998,480 | 107 | `dcm_log_20260728-223132.txt` |
| 6 | cont-run4 | M4 Pro | 811,769 | 49,374 † | 104,743,805 | 46 | `dcm_log_20260802-152830.txt` |
| 7 | cont-run5 | M4 Pro | 857,769 | 2,050 | 110,668,628 | 2 | `dcm_log_20260804-221739.txt` |

✗ = log is on the original machine, not this one. † run 4 ran 49,374 steps but only
46,000 are usable — see §5.

Two derivations that are easy to get wrong:

- **Seg 3's base is 100,320** = `60,901 + 39,419`, the entry checkpoint's own
  `training_step` — **not** v5.csv's last row (99,901). The 419-step gap is real: the
  last *frozen* mark was step 39,000 but the run saved at 39,419.
- **Seg 7's base is 857,769** = `811,769 + 46,000`. Run 5 resumed from run 4 at step
  46,000, not from run 4's end (§5). Because run 4 has no probe past 46,000, the series
  stays strictly monotonic with nothing invented.

## 3. Three axes, none of them interchangeable

The lineage spans two machines at very different speeds, so a single "time" axis
conflates duration with work. `data/v5.csv` therefore carries three:

| axis | column | device-independent? | notes |
|---|---|---|---|
| **step** | `cum_step` | yes | exact, but a *work* axis only while batch size and replay ratio hold constant |
| **time** | `elapsed_train_sec` (+ `wall_sec`) | **no** | real cost; see the clamp note below |
| **compute** | `games_fed` | yes | cumulative corpus games, measured from `games=` |

Measured per-step speed:

| segments | device | median s/step | p05→p95 |
|---|---|---:|---|
| 0–2 | M5 | **1.327** | 1.303–1.348 |
| 3–7 | M4 Pro | **3.42–3.45** | 3.39–3.74 |

The **2.59× slope break in the by-time chart at cum 100,320 is hardware** and is left
in deliberately. It is never normalized away, because with batch 4096 and replayRatio
0.48 constant on both machines a "normalized seconds" axis would reduce to
`steps × reference_ms` — a relabelled step axis that carries no new information while
destroying the only record of real cost. Both devices are individually stable and
unimodal; there is **no** high/low-power-mode signature anywhere in this data.

**Why `games_fed` and not steps:** with batch and replay ratio fixed the two are
proportional, but `games_fed` is measured directly and stays correct if either ever
changes. It independently validates the lineage: measured games/step is 129.0 (run1),
129.0 (run2), 129.3 (run3), and segments 0–2 give `12,933,495 / 99,901 = 129.5` — the
two machines agree to 0.4%, confirming a step is the same unit of work on both.

Reading the compute axis:
- Within a segment the slope is 129 ± ~3. `games=` is logged every ~50 steps but
  checkpoints land every 1000, so each reading is up to 50 steps stale (≈±6.5
  games/step). That spread is the logging cadence, not drift.
- At each **segment boundary** the slope jumps (136.8–151.3). That is the replay
  buffer's cold prefill: a restart consumes corpus games *before* step 1. It checks
  out exactly — seg3→4's excess is ~21,300 games against a logged 22,650 prefill, and
  seg5→6's is ~7,800 against run 4's logged `gamesFed=7563` (run 4 prefilled 500K
  positions rather than the usual 1M). Those games are really consumed, so they are
  really counted; they simply arrive with no steps attached.
- **Segments 0–2 have blank `games_fed`, not zero.** Their logs are on the other
  machine, so only one anchor survives (12,933,495 at cum 100,320, from the entry
  checkpoint's `replay_next_game_index`). Interpolating 96 rows at 129.5 games/step
  would look seamless and would be modeled data in a column whose entire point is that
  it is measured. Left blank on purpose.

### `elapsed_train_sec` is sleep-clamped; `wall_sec` is not

`_clamped_timeline` banks `min(real_gap, segment_median_s_per_step × Δsteps + 120s)` per
interval, so machine sleep cannot be counted as training. The same cap also trims
genuine slow stretches, and it is one-directional — faster-than-median intervals pass
through untouched. `wall_sec` is the identical walk with the cap removed, so
`wall_sec − elapsed_train_sec` is exactly what the clamp discarded:

| segment | raw wall | clamped | discarded |
|---|---:|---:|---:|
| 3 run1 | 262.87 h | 262.87 h | 0.00 h |
| 4 run2 | 361.91 h | 341.37 h | **20.54 h (5.7%)** |
| 5 run3 | 104.83 h | 104.83 h | 0.00 h |
| 6 run4 | 48.43 h | 48.43 h | 0.00 h |
| 7 run5 | 2.00 h | 2.00 h | 0.00 h |

One segment, 5.7%, across a 15-day span — consistent with real machine sleep. The clamp
is working; it just needed to be visible. Lineage end: **787.0 h clamped / 807.6 h raw.**

Two bounded caveats on that prefix:

- Segments 0–2 contribute a *pinned* `elapsed_base_sec` of 111,967.7 s taken from v5.csv,
  with no separate raw measurement, so `wall_sec − elapsed_train_sec` measures clamping
  only within segments 3–7 — never inside the prefix.
- That pin is v5.csv's last row (cum 99,901 = seg-2 step 39,000), but segment 3 begins at
  step 39,419. The **419 steps in between (~557 s at 1.33 s/step) are not counted** in the
  time axis. It is ~0.02% of the 787 h total; recorded here rather than silently
  back-filled, since the seg-2 log needed to measure it is on the other machine.

## 4. Records

| | value | where | when |
|---|---|---|---|
| best pElo | **1770.46** | seg 4, raw step 270000 = **cum 638,826** | 2026-07-25 07:09 |
| lowest NLL | **1.8445** | seg 4, raw step 268000 = **cum 636,826** | 2026-07-25 05:12 |

Per-segment bests: seg 3 — 1704.0 · seg 4 — 1770.5 · seg 5 — 1735.4.
Probe battery `wide` = 4435 fixed puzzles, deterministic. v5 pre-training baseline:
pElo 1584.2, top1 47.8%, NLL 2.097.

⚠️ These cum steps are **+100,320** from the figures in the retired bundle monitor,
which numbered run 1 from 0 and so omitted the original machine's 100,320 steps. Old
notes citing "pElo 1770.5 @ cum 538.5k" refer to the same checkpoint.

## 5. Run 4's truncated tail, and why run 5 forked at 46,000

At around step 49,350 macOS revoked read access to corpus shards 14–45 (`[REPLAY]
skipping unreadable shard … you don't have permission to view it`, 32 shards). Run 4
continued on shards 0–13 alone and then recorded a spurious epoch wrap.

- Run 4's checkpoint writes at 47,000 / 48,000 / 49,000 were **denied** and do not
  exist. (Three run-3 files survived under those names and are renamed
  `run3-step*-MISNAMED` in `quarantine/`.)
- Its final save, step 49,374, is quarantined: metadata claims `replay_epoch=1,
  replay_next_game_index=0` — a completed pass — when only 6,360,368 of ~20,935,171
  games were fed. **Never use it with `--resume-exact`.**
- So step 46,000 was the newest checkpoint with honest resume metadata
  (`epoch=0, nextGame=5,924,823`). Run 5 resumed exactly there; its step-2000 save reads
  `nextGame=6,183,024`. Independently corroborated: the session log's `games=` at run-4
  step 46,000 is **5,924,823**, matching that checkpoint's header exactly.

Run 4's abandoned 46,000→49,374 tail is recorded here and deliberately not charted.

## 6. Surviving weights — 650 files, 616 distinct

Full manifest with SHA-256 per file:
`~/Downloads/v5-continue-bundle/monitor/checkpoint_inventory.json`.

| seg | model_id | files | step range | gap |
|---:|---|---:|---|---|
| entry | `20260629-1-Uf4p` | 1 | 39,419 | — |
| 3 | `20260703-1-Dg5v` | 268 | 2,000–268,506 | step 1000 only |
| 4 | `20260714-1-h7vI` | 243 | 107,000–336,610 | 1,000–106,000 overwritten by run 3 |
| 5 | `20260729-1-VZ2j` | 81 | 9,000–106,333 | 1,000–46,000 overwritten by runs 4/5 |
| 6 | `20260802-2-Xuub` | 54 | 2,000–49,374 | — |
| 7 | `20260805-1-0pTW` | 3 | 1,000–2,000 | — |

34 of the 650 are exact duplicate content (1.1 GB) where `preserved-best/` re-copied a
root file that happened to survive. Nothing deleted.

**This corrects `archive/MANIFEST.md` §3**, which stated that `preserved-best/` held the
only surviving weights. Run 1's enumerated checkpoints survive complete in `run1/`, and
231 of run 2's survive at the bundle root.

Every root checkpoint is now **hardlinked** into `run2/ run3/ run4/ run5/` beside the
existing `run1/`, named `run<N>-step<M>.safetensors` by verified `model_id` — 335 links,
zero extra bytes, nothing renamed or removed. Note a hardlink only survives a future
overwrite if the writer replaces atomically rather than truncating in place; that is
unverified, so durability rests on the copies, not the links.

## 7. Where the data lives

| what | where |
|---|---|
| **the unified series (source of truth)** | `documentation/dashboards/data/v5.csv` — 856 rows |
| segment/axis config | `documentation/dashboards/registry.json` → `runs.v5` |
| dashboard | `dcm_master.html` (+ hosted copy), 3 axes: by step / by time / by compute |
| probe JSONL (pre-image) | `~/Downloads/v5-continue-bundle/monitor/new_ckpts*.jsonl` |
| checkpoint manifest | `…/monitor/checkpoint_inventory.json` |
| full per-step session logs | `~/Library/Logs/DrewsChessMachine/` **and** gzipped in `…/archive/logs/` (all 5 segments, integrity + line counts verified) |
| weights | `…/{run1..run5,preserved-best,quarantine}/` |
| 4 MB checksummed telemetry copy | `…/telemetry-snapshot-20260811/` (`SHA256SUMS.txt`) |
| verified full backup | `~/v5-consolidation-backup-20260811/` |

`~/Library/Logs/DrewsChessMachine/` is **excluded from Time Machine**. The gzipped
copies inside the bundle are what make those logs recoverable — keep them there, and
compress any future run's log the same way.

## 8. Retired by this consolidation

Left in place, superseded — do not add to them:

- `monitor/build_dashboard.py`, `build_html.py`, `build_table.py`, `v5-strength.html`,
  `records.json` — the bundle's separate strength tracker. `master.py` now renders v5
  from the CSV.
- `monitor/HANDOFF.md` — the monitor-loop runbook. Its durable content is here; its
  live-process instructions are obsolete (no run is active).
- `monitor/new_ckpts*.jsonl` — retained as the imported pre-image, not appended to.
  New marks go through `replay.py track v5` / `import-probes`.

## 9. Reproducing the import

```bash
cd documentation/dashboards
B=~/Downloads/v5-continue-bundle
# NOTE: replay.py must run WITHOUT -I. Unlike master.py/build_html.py, it does
# `from _schema import FIELDS`, and -I (which implies -P and -E) strips the script's
# own directory from sys.path and ignores PYTHONPATH, so the import cannot resolve.
python3 replay.py import-probes v5 --segment 3 --jsonl $B/monitor/new_ckpts.jsonl \
  --ckpt-dir $B --ckpt-dir $B/run1 --ckpt-dir $B/preserved-best --ckpt-dir $B/quarantine
# ...segments 4-7 with new_ckpts_run{2,3,4,5}.jsonl
python3 -I master.py && python3 -I make_hosted.py
```

Set `DCM_DASH_ROOT` to a scratch copy of `registry.json` + `data/` to rehearse any of
this without touching the real CSV.

Idempotent on `cum_step`. Prefer one process for all five segments so the 6.4 GB of logs
is parsed once. Each probe's `modelID` must equal its segment's registry `model_id` or
the row is refused — a monitor that trusted filenames once re-probed nine month-old
files and published a fabricated curve, so that gate is not optional. The single
exception is run 1's step-1000 probe, which carries `recovered: true` (its weights were
overwritten before archiving; values were read back off a published chart, rounded and
un-reprobeable). It imports with its provenance note in the `note` column and is the
only non-measured row in the file.

**Do not** point `probe_backfill()`'s enumerated scan at the bundle: it maps filenames
to the latest segment's base and would mis-assign every colliding name.

### Two traps when regenerating on this machine

1. **`SegTime.seg_for()` cannot be trusted for a mark on a segment boundary.** Each
   segment's final step equals the next segment's `cumstep_base`, and segment 7's base
   additionally sits *inside* segment 6's span (the fork at step 46,000). Pass the known
   segment explicitly — `elapsed_and_clock(cum, segment)` / `wall_at(cum, segment)` —
   as `import_probes` and `recompute_elapsed` now do. Left to `seg_for`, the last row of
   every segment takes its `wallclock_iso` from the *next* run's log (off by 21 s to 8 h).
   `elapsed`/`wall` happen to survive it only because the pins are defined to agree at
   exactly those points; do not rely on that.

2. **`master.py` needs the corpus manifests, which are per-machine — not in the repo.**
   It reads ply counts from
   `~/Library/Application Support/DrewsChessMachine/Corpora/<id>/corpus.json`. That
   directory did **not** exist on this machine until 2026-08-11 (this Mac never recorded
   a corpus; training read the bundle's shards via `--replay-corpus ./corpus`, which
   registers nothing). Regenerating without it silently drops `epochs` for all 14
   corpus-backed runs and renders the corpus summary table empty — no warning, because
   `_load_corpora` deliberately skips a corpus whose directory is gone.

   Resolved by copying the three manifests across:

   | corpus | id | games | plies |
   |---|---|---:|---:|
   | std-2026-05 | `20260624-192615-w3aA5b` | 20,935,171 | 1,386,486,078 |
   | elite-2025-05_11 | `20260704-001142-lLOrLj` | 2,023,146 | 181,908,093 |
   | elite-2021_2025 | `20260704-215145-op24Gp` | 14,619,153 | 1,271,862,796 |

   Only `corpus.json` is read — the `.dcmgames` shards are not. The bundle's own
   `corpus/corpus.json` cannot substitute: never finalized (`state: recording`,
   `gamesAdded: 0`).

   **Design note:** run config lives in the repo (`registry.json`) but corpus config lives
   in per-machine Application Support, so the same repo renders differently on different
   machines with no warning. Caching these manifests beside `registry.json` would remove
   the whole failure mode.

   Sanity check after any such move — v5 should read **5.291 epochs**, which independently
   agrees with the compute axis: 110,949,479 / 20,935,171 = **5.300**.
