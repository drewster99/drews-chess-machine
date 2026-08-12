# Plan — retire `v5-continue-bundle`, merge v5's runs into normal storage

_Written 2026-08-12. **Executed 2026-08-12** — see `documentation/v5-lineage.md` §10 for the
outcome. Retained as the record of what was decided and why; the one item deliberately
left undone is noted at the end._

## Goal

v5's five continuation runs currently live in a one-off container,
`~/Downloads/v5-continue-bundle/`, that was built to ship training to another machine
and was never meant to be permanent. Every other run in this project stores its
artifacts in three standard places. The end state is that v5 does too, and the bundle is
deleted.

Nothing about v5's *metrics* is at stake — those are already consolidated and committed
(`documentation/dashboards/data/v5.csv`, 856 rows, 8 segments; see
`documentation/v5-lineage.md`). What remains unhomed is **650 checkpoint files, 22.0 GB**,
plus a handful of small manifests.

| artifact | normal home | v5 status |
|---|---|---|
| metrics | `documentation/dashboards/data/<run>.csv` + `registry.json` (git) | ✅ done |
| session logs | `~/Library/Logs/DrewsChessMachine/` | ⏳ copy in progress |
| corpora | `…/Application Support/DrewsChessMachine/Corpora/` | ✅ already canonical on M5 |
| provenance narrative | `documentation/v5-lineage.md` | ✅ done |
| **checkpoints** | `…/Application Support/DrewsChessMachine/Models/` | ❌ **this plan** |

## The naming convention we are targeting

Observed on the M5 across 3,142 files. Runs are stored **per segment**, each segment
getting its own filename prefix, with step numbers staying **segment-local** inside it:

```
20260702-Qeu8-replay-step<N>.safetensors            <- first segment
20260702-Qeu8-resume2-replay-step<N>.safetensors    <- a later segment
20260702-Qeu8-resume3-replay-step<N>.safetensors
```

Two suffix forms exist and mean different things — **use `-replay-step<N>`**, because
these files were written by the app's `--enumerate-checkpoints`, not minted by the
tracker's `freeze()`:

- `<date>-<name>-replay-step<N>.safetensors` — app-enumerated (what we have)
- `<date>-<name>-step<N>-frozen.safetensors` — tracker-made copies (legacy runs)

This convention is **why the merge is a prefixing job, not a renumbering one**: the
bundle's step numbers are already segment-local. They collide only because all five runs
shared one flat namespace.

### Proposed names

| seg | run | model_id | date | target prefix | files |
|---:|---|---|---|---|---:|
| 3 | run 1 | `20260703-1-Dg5v` | 20260702 | `20260702-v5cont-replay-step<N>` | 268 |
| 4 | run 2 | `20260714-1-h7vI` | 20260713 | `20260713-v5cont-resume-replay-step<N>` | 243 |
| 5 | run 3 | `20260729-1-VZ2j` | 20260728 | `20260728-v5cont-resume2-replay-step<N>` | 81 |
| 6 | run 4 | `20260802-2-Xuub` | 20260802 | `20260802-v5cont-resume3-replay-step<N>` | 54 |
| 7 | run 5 | `20260805-1-0pTW` | 20260804 | `20260804-v5cont-resume4-replay-step<N>` | 3 |
| — | entry point | `20260629-1-Uf4p` | — | *skip* — already on M5 as `…-wd2.5e4-m93-replay-latest`, sha-verified identical | 1 |

**Assignment is by `model_id` from the safetensors header, never by current filename.**
Four distinct files have been named `v5-cont-replay-step1000.safetensors`.

Naming decision to confirm before starting: `v5cont` vs continuing the segment-0–2 family
name (`v5_5block_7x7_lnout-cont1` etc.). The former is shorter; the latter makes the
8-segment lineage obvious from `ls`.

## Blocker: the tracker cannot see per-segment families

`registry.json` gives each run a **single** run-level `frozen_glob`, and
`probe_backfill()` resolves enumerated checkpoints with `enum_glob(cfg)` (derived from
the run-level `out_model`) against **`cfg["segments"][-1]["cumstep_base"]` only**. So it
can find at most one segment's files, and maps every filename to the *latest* segment's
base.

This is not a v5 quirk — qeu8 has four families and the same latent bug; it simply has
not been exercised. v5 forced it into the open, which is why segments 0–2 already carry a
per-segment `frozen_prefix` key (added 2026-08-12, currently documentation only — nothing
reads it).

**Without fixing this, merged checkpoints are invisible to the tracker** and the merge
achieves nothing beyond tidier filenames.

### Required change — `documentation/dashboards/replay.py`

1. Add `enum_prefix_for(cfg, si)` returning the segment's own prefix: `segments[si]`'s
   `frozen_prefix` if present, else fall back to the run-level `enum_glob(cfg)` so every
   existing run behaves exactly as today.
2. Rewrite `probe_backfill()`'s enumerated branch to **iterate segments**, globbing each
   segment's prefix and mapping `cum = segments[si].cumstep_base + N`. Today it does one
   glob against the last segment.
3. Leave the legacy `-frozen` branch untouched.

Guard: every other run's `probe_backfill` output must be unchanged. Verify by running it
across all 29 runs before and after and diffing the CSVs — expect zero differences,
since none of them define `frozen_prefix`.

## Phases

### Phase 0 — preconditions
- `~/Library` copy to the M5 complete and verified.
- `monitor/checkpoint_inventory.json` regenerated (650 entries, SHA-256) so the manifest
  matches the bytes about to move.
- M5 free space ≥ 40 GB (was 425 GB).

### Phase 1 — code (no data moved)
Implement the `replay.py` change above. Add `frozen_prefix` to segments 3–7 in
`registry.json`. Commit before touching any file, so the tooling change is bisectable
independently of the data move.

### Phase 2 — copy checkpoints to `Models/` under new names
Copy — **never move** — from the bundle into
`~/Library/Application Support/DrewsChessMachine/Models/` on the M5, resolving each
file's segment from its header `model_id` and its step from `training_step` (not the
filename). Sources, in order:

| source | files | note |
|---|---:|---|
| `run1/` | 268 | sole copies |
| root `v5-cont-replay-step*` | 335 | runs 2–5; `run2/`–`run5/` are hardlinks to these — copy **once** |
| `preserved-best/` | 40 | **8 are sole copies** (7× run 3 at steps 9k/10k/12k/20k/30k/40k/42k, 1× run 4 at step 2k); the other 32 duplicate surviving root files and dedupe away by sha |
| `quarantine/` | 5 | see below |
| `v5-cont-replay-latest.safetensors` | 1 | run 5 step 2000 — dedupes against `run5` step2000 |

Expected result: **650 sources → 649 distinct files written** (the entry point is skipped
as already present). Dedupe by SHA-256 against `checkpoint_inventory.json`; a collision
where two sources share a sha is expected and must resolve to one file, not an error.

### Phase 3 — quarantine handling
The normal scheme has no quarantine folder, and these files are actively dangerous if
they land under ordinary names. Preserve the warning **in the filename**:

- `run3-step{47000,48000,49000}-MISNAMED` → `20260728-v5cont-resume2-replay-step<N>.safetensors`.
  These are legitimate run-3 checkpoints; the misnaming was relative to the bundle's flat
  namespace and disappears under per-segment prefixes. No marker needed.
- run 4 step 49374 (two copies, same content) → `20260802-v5cont-resume3-replay-step49374-DO-NOT-RESUME.safetensors`.
  Its metadata falsely claims a completed epoch (`replay_epoch=1, replay_next_game_index=0`)
  when only 6,360,368 of ~20,935,171 games were fed. **`--resume-exact` from it would
  resume from a corpus position never reached.** The marker must survive the merge; the
  rationale is in `v5-lineage.md` §5.

### Phase 4 — verification (this is what licenses deletion)
1. Every one of the 649 targets exists in `Models/` and its SHA-256 matches
   `checkpoint_inventory.json`. Zero mismatches, zero missing.
2. Header `model_id` + `training_step` of each written file agrees with the segment
   implied by its new prefix — i.e. the rename didn't scramble anything.
3. `probe_backfill("v5")` finds and probes them; `data/v5.csv` gains
   `bn1Mean`/`sae2`/`eff_alpha` on rows that lacked them, and **changes no existing
   value**. Diff before/after: only fills, never edits.
4. All 29 runs' CSVs unchanged apart from v5's fills.
5. `master.py` renders; v5 still 856 rows / 792.7 h / 5.291 epochs / peak 1770.46.

### Phase 5 — retire the bundle
Only after Phase 4 is clean, and only with explicit approval per the standing
"nothing deleted" rule on this bundle.

Move to the repo first (small, and they have no other home):
- `monitor/checkpoint_inventory.json` — the manifest that made this merge verifiable
- `monitor/new_ckpts*.jsonl` — the 761-probe pre-image behind v5.csv's rows
- `parameters.json` — the hyperparameters all five runs used

Then, in order: confirm `~/v5-consolidation-backup-20260811/` is still intact and
checksum-clean → delete `~/Downloads/v5-continue-bundle/` → keep the backup until the
next Time Machine cycle has captured the merged `Models/`.

Redundant by this point, needing no home: `corpus/` (Corpora is canonical),
`archive/logs/*.gz` and `run*.out` (raw logs now in `~/Library/Logs`),
`DrewsChessMachine.app`, `telemetry-snapshot-20260811/`, and the `README`/`MANIFEST`/
`HANDOFF` narrative (folded into `v5-lineage.md`).

## Risks

- **`rsync -H` on any bundle copy.** Without it the 335 hardlinked paths become 335 extra
  independent files — 33.3 GB instead of 22.0 GB, and the sharing is lost.
- **Filename-derived segment assignment.** Any step that reads the step number from the
  name instead of the header will mis-assign silently. The manifest is the check.
- **Deleting before Phase 4 passes.** The 268 `run1/` files and the 8 sole
  `preserved-best/` files exist in exactly one place plus the backup.
- **`probe_backfill` regression.** The segment-iteration change touches a path shared by
  all 29 runs.

## Decisions taken

1. **Prefix wording:** `v5cont`, matching the qeu8 shape
   (`20260702-v5cont`, `20260713-v5cont-resume`, …).
2. **All checkpoints, not a subset** — every other run keeps everything. 650 sources
   became 615 targets (1 entry point already present on the M5, 34 duplicate contents).
3. **Probe JSONL committed** to `documentation/dashboards/data/v5-source/` (~1 MB). Each
   probe holds more than the four fields the tracker keeps — top-1/top-5 accuracy,
   `avgProb`, `avgRank`, and a 13-category tactical `themes` breakdown — none of it
   recoverable from the CSV, and most of the checkpoints can no longer be re-probed.

### Discovered while executing

- `enum_glob()` raised on any run with an explicit `"out_model": null` (v5, t97x),
  because `.get(k, "")` returns the stored `None`. Latent — `probe_backfill` is not
  CLI-wired, so it had never been called for those runs. Fixed.
- The five other under-finding runs were left alone. `enum_specs()` supports them, but
  declaring their stems is a separate opt-in and a separate verification.

### Still outstanding

Deleting `~/Downloads/v5-continue-bundle/` — held back pending explicit approval, per the
standing "nothing deleted from this bundle" rule.
