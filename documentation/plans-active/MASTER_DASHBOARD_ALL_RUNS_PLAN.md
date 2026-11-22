# Master Dashboard: every run (plan)

Owner instruction (2026-11-22): every run goes on the DCM Master Dashboard except very short tests, and the
dashboard stays updated, both the repo copy (`documentation/artifacts/dcm-master-dashboard.html`) and the
published artifact (https://claude.ai/artifact/LfMS23HF3QEVo1r2N4GYaU).

The published page (and its repo copy, snapshot 2026-11-16) shows 29 runs; its newest data is 2026-09-17 (Ejp0
run 2). The SE A/B/C runs (2026-09-29/30) are in `registry.json` but were never rendered onto it. Nothing since
2026-10-01 is on it.

## Inventory

"Include" = added to the dashboard by this plan. Step ranges are the dashboard's `cum_step` axis (see
"Step axis" below). Logs are `~/Library/Logs/DrewsChessMachine/dcm_log_<stamp>.txt`; files are under
`~/Library/Application Support/DrewsChessMachine/Models/` unless noted.

### Corpus replay (`registry.json`, tracked by `replay.py`)

| key | run | lineage / files | logs (segments) | steps | corpus | include |
|---|---|---|---|---|---|---|
| `ejp0hf` | Ejp0 head-fix phase 2: branch of qeu8 resume3 step 681,000 (model `20260727-1-Ejp0`), bf16 head-offset research | pre-lineage; `20261001-headfix-phase2-Ejp0-replay-step*` (20, model `20261001-18-oeNy`) | 20261001-015706 | 0–20,000 | std-2026-05 | yes |
| `fc1leaky` | R4: SE scale+bias, leaky ReLU in SE FC1 | pre-lineage; `20261001-test_SE_scale+bias-fc1leaky-replay-step*` (33) | 20261001-151822 | 0–33,000 | std | yes |
| `nose1` | R7: no SE, no ReZero, seed 1 | pre-lineage; `20261002-bench_v5s3_noSE_noReZero-replay-step*` | 20261002-011124 | 0–33,000 | std | yes |
| `nose2` | R8: no SE, no ReZero, seed 2 | pre-lineage; `…noSE_noReZero-seed2-replay-step*` | 20261002-035513 | 0–33,000 | std | yes |
| `lsC` | R10: label smoothing C (policy ε 0.03) | pre-lineage; `20261002-label-smoothing-C-replay-step*` | 20261002-061425 | 0–33,000 | std | yes |
| `rz0` | R9: zero-init ReZero (no SE) | pre-lineage; `20261002-rezero-zero-init-replay-step*` | 20261002-202430 | 0–33,000 | std | yes |
| `lsD` | R12: label smoothing D (value ε 0) | pre-lineage; `20261002-label-smoothing-D-replay-step*` | 20261003-002816 | 0–33,000 | std | yes |
| `lsC2` | R11: label smoothing C seed 2 (stopped by the owner) | pre-lineage; `…label-smoothing-C-seed2-replay-step*` | 20261003-033152 | 0–31,906 | std | yes |
| `fatty216` | R13: fatty, 1 block 7×7 @216 | pre-lineage; `20261003-fatty216-b2275-replay-step*` | 20261003-204412 | 0–33,000 | std | yes |
| `fatty224` | R15: slim-neck fatty, 3×3 stem, 1 block 7×7 @224 | pre-lineage; `20261003-fatty224s3-b2275-replay-step*` | 20261003-231825 | 0–33,000 | std | yes |
| `fatconv98` | R16: fatconv, 15×15 stem, 1 block 15×15 @98, + LR-cycle continuation | seg 0 pre-lineage `20261004-fatconv98-b2275-replay-step*`; seg 1 lineage `FAE5AB97` (`…fatconv98-cont-replay-step*`) | 20261004-015804, 20261004-210810 | 0–39,093 | std | yes |
| `lrA` | LR arm A: constant 0.01 | lineage `2C66BFD0`, 2 segments | 20261005-013220-2, 20261005-171451 | 0–40,000 | std | yes |
| `lrB` | LR arm B: cycle 1.0 ↔ 0.001 | lineage `9249756B`, 2 segments | 20261005-013235, 20261005-171541 | 0–40,000 | std | yes |
| `lrC` | LR arm C: cycle 10 ↔ 0.01 (diverged ~300, resumed) | lineage `E67A96F2`, 2 segments | 20261005-090417, 20261005-121841 | 0–6,116 | std | yes |
| `lrBleaky` | B-leaky: leaky ReLU in the value head | lineage `783BF744` | 20261005-204108 | 0–40,000 | std | yes |
| `lrBleakyall` | B-leakyall: leaky ReLU everywhere | lineage `A41BEB12` | 20261005-234434 | 0–40,000 | std | yes |
| `lrBsilu` | B-silu: SiLU tower, leaky heads, cap 15 (blew up at 20,600) | lineage `085EF48A` segment 0 | 20261005-234437 | 0–36,066 | std | yes |
| `lrBsilu_clip1` | B-silu-clip1: fork of B-silu at 18,000, clip 1.0 | `085EF48A` fork segment | 20261006-170000 | 18,000–40,000 | std | yes |
| `lrBsilu_clip2` | B-silu-clip2: fork at 18,000, clip 2.0 | `085EF48A` fork segment | 20261007-035017-2 | 18,000–23,000 | std | yes |
| `lrBsilu_clip5` | B-silu-clip5: fork at 18,000, clip 5.0 | `085EF48A` fork segment | 20261007-035017 | 18,000–23,000 | std | yes |
| `bsiluall` | B-siluall: SiLU everywhere, relative cap | lineage `66006003`, 7 segments | 20261008-125718, 20261009-163612, 20261009-213842, 20261010-082147, 20261010-111314, 20261010-211711, 20261014-214445 | 0–100,000 | std | yes |
| `rreplay` | R-replay (zlra) | lineage `0EFE2DA3`, 3 segments | 20261008-094622, 20261010-122003, 20261010-211714 | 0–86,398 | std | yes |
| `relite` | R-elite: branch of R-replay at 86,398, fixed LR 0.002 | lineage `ACC058E9` | 20261011-130546 | 0–21,006 | elite-2021_2025 | yes |
| `relitecyc` | R-elitecyc: branch of R-elite at 21,006, gentle LR cycle, then decay | lineage `E709F4EB`, segments 0–19 | 20261011-213940 … 20261106-011055 (20 logs, list in `registry.json`) | 0–883,503 | elite | yes |
| `relitecyc_anneal` | R-elitecyc-anneal: branch of R-elitecyc at 384,964, LR annealed | lineage `23DF16CB` | 20261025-142424 | 0–30,000 | elite | yes |
| `relitefresh` | R-elitefresh: R-replay's start/params/seed on the elite corpus | lineage `F79C7568`, segments 0–12 | 20261012-153151 … 20261101-193002 (13 logs) | 0–353,933 | elite | yes |
| `relitecyc_hilr` | R-elitecyc-hilr: fork of R-elitecyc at 384,964 keeping the original cycle | `E709F4EB` fork segment | 20261109-234535 | 384,964–528,527 | elite | yes |
| `relitecyc_fp32` | R-elitecyc-fp32: fork at 384,964 in fp32 (MPSGraph fp32 weight-gradient bug) | `E709F4EB` fork segment | 20261112-234508 | 384,964–413,778 | elite | yes |
| `relitecyc_fp32fix` | R-elitecyc-fp32fix: fork at 384,964 in fp32 with the workaround (LIVE) | `E709F4EB` fork, segments 14–18 | 20261114-003038, 20261114-154717, 20261116-011436, 20261119-191618, 20261121-164043 | 384,964–live | elite | yes |
| — | R14 skinny 22×7×7 @48 | `20261003-skinny48-b2275` | 20261003-204412-2 | 0–1,605 | std | no: stopped by the owner at 1,605 (one probe) |
| — | LR arm C-leaky | `E62E827B` | 20261005-231335 | 0–1,175 | std | no: ran away at 1,175 |
| — | B-silu-ctl15 | `085EF48A` fork | 20261006-181510 | 18,000–22,030 | std | no: control that reproduces B-silu bit for bit (E-0020); its curve is B-silu's |
| — | V-1 (`lrBsilu-relcapV1`) | `085EF48A` fork | 20261007-060735 | 18,000–21,000 | std | no: log-only rerun, byte-identical to B-silu (E-0024) |
| — | V-3 (`lrB-relcapV3`) | `1AFCF318` | 20261007-060832 | 0–3,000 | std | no: 3,000 steps, byte-identical to B (E-0024) |
| — | `seg-resume-check` | `C378E0D2` | 20261003-111036, 20261003-120520 | 0–2,000 | std | no: resume test |
| — | `tsr-replay` | `598D5E73` | 20261007-211128 | 0–2,000 | std | no: test-set-results test |
| — | `dry-replay`, `fatty216` (non-b2275), `r7-mixedB`, `mx*`, `old*`/`new*`/`fix*`/`parent*`, `cap_*` | various | 2026-09-30 … 10-04 | ≤ 600 | — | no: smoke / speed / numerics tests |
| — | determinism tests (`first`, `control-straight`, `wrap-first`, `out-*`, `S-replay`, `A`–`E`, `straight`, `epochs-straight`) | various, written outside `Models/` | 2026-10-03 … 10-07 | ≤ 300 | test corpora | no: determinism / resume tests |
| — | fp32 speed tests (`out-zlra9x9-*`, `out-v4_5block_7x7-*`, `out-v3_16block_3x3-*`) | various | 20261114-1407 … 1427 | ≤ 120 | elite | no: speed tests |

### GUI self-play (`selfplay_registry.json`, tracked by `selfplay.py`)

| key | run | logs | steps | include |
|---|---|---|---|---|
| `AgG3` | AgG3: GUI self-play, fresh SiLU network, 4 promotions | 20261007-084445 | 1–5,257 | yes |
| `ZlrA` | ZlrA: GUI self-play on the 0.001–1.0 LR cycle (E-0025), 19 promotions | 20261007-164741 | 1–51,136 | yes |
| `Ew5t` | R-fixedlr: GUI self-play, LR 0.01 / momentum 0.90 fixed, model `20261008-4-Ew5t` (lineage `A2599216`) | 20261008-094623, 20261010-113351 (cut at 66,602), 20261010-224411, 20261012-161833 (cut at 87,218), 20261024-023842, 20261102-101606 | 1–286,235 | yes |
| — | `20261014-140421` (R-fixedlr, minutes, nothing saved), `20261015-102200` (R-fixedlr, 565 steps never saved) | — | — | no: abandoned branches (`excluded_logs`) |
| — | GUI tests 2026-10-01 (`VFTY`, `2DWq`, `DCDV`, ~350 steps), 2026-10-03 (`OPMQ`, `TRNR`), 2026-10-06 (`TRNR`/`CHMP`) | — | ≤ 372 | no: short tests |

### Train-vs-UCI

None since `sf100sl100` (already on the dashboard).

### Older runs not on the dashboard

| run | files / logs | steps | include |
|---|---|---|---|
| `20260627-v3_8block_3x3` replay | 25 `-frozen` files; log gone | 0–25,000 | no: predates the v5 run the dashboard starts from, its log is gone, and its frozen files were cum-named by a tracker no registry entry describes, so their step meaning cannot be checked |
| `20260627-freshNet` | 4 `-frozen` files; log gone | 0–4,000 | no: short |
| `20260712-qeu8init-sf200-vsuci` and `-sloppy20`, `-sf10`, `-sf100` | `…-vsuci-step*` | ≤ 15,000 | no: train-vs-UCI pilots started the same day as `sf100sl100` (D-8) |

## Step axis

- A fresh run starts at 0. A **branch** (new lineage run from another run's save: R-elite, R-elitecyc,
  R-elitecyc-anneal, `ejp0hf`) starts at 0 too, as `qeu8e` does; its label names where it branched.
- A **fork** (an exact resume that starts a second chain of the same lineage run: B-silu-clip1/2/5, R-elitecyc-hilr,
  -fp32, -fp32fix) shares its parent's axis, so it is directly comparable with the parent row for row: clip arms start
  at 18,000, the R-elitecyc forks at 384,964.
- From architecture format v11, `cum_step` = the segment's `cumstep_base` + the record's `segment_local_step`
  (= the trainer step for the zlra lineage).

## pElo where the 1000-step file is gone

R-elitecyc's 1000-step files from 720k to 790k (and others) were thinned to every 10k. Every save since
2026-10-07 also logs its test-set results in the writer's own session log:
`[CHECKPOINT] test sets trainer step N (<file>) (… ms): … lichess-wide n=4435 … nll=X pElo=Y`. That is the same
evaluation as `--probe-model --probe-set wide` (checked: 720,000 logs 1807.7 / 1.8361, the probe gives 1807.66 /
1.83605). The log names the trainer step and is the segment's own log, so the row is filed by the segment the
registry already holds, with no filename involved. Such rows carry the note `writer test sets (log)` and no
`probe_build`; `pLogit_*` and the internals stay blank (they need the file). `wide_pelo_record.json` is not used:
it records values by file-name step without a model ID, which the dashboard's provenance rule does not accept.
Rows whose files exist are probed as before.

## Code changes

1. `scripts/dcm_lineage.derive_runs`: a lineage run may fork (two chains resumed from one save): segments are keyed
   by `segment_id` (a segment index can repeat across chains), each records its `parent_segment_id`, history
   segments are read from every chain. A `--derive-model` output that carries its source's trainer state (and so its
   record, under a new model ID) is a copy, not a checkpoint of that segment: it is set aside (`carried_copies`).
2. `_lineage_registry.plan`: a lineage run may feed several registry runs when each starts at a fork of the lineage
   (its first segment's parent has two or more children); a split anywhere else is still a contradiction. The step
   offset comes from the registry run holding the lineage's segment 0 and applies to all of them. Registry-level
   `excluded_lineage_runs` / `excluded_segments` ({id: reason}) record deliberate exclusions, reported, not proposed.
3. `replay.py`:
   - `lineage_checkpoints` skips carried copies.
   - New `log_backfill(run)`: the per-1000-step log rows `tick.py` wrote, moved here so the refresh shares it, with
     marks on the trainer-step thousands for v11 segments (the old segment-step marks fall between saves on a resumed
     segment), plus the writer's test-set results for saves whose file is gone.
4. `refresh_dashboard.py` (replaces the retired static-PNG v5 script): one command that, under the cron tick's lock,
   runs `probe-backfill` + `log_backfill` for every registry run whose rolling out-model is being written by a live
   process (or every run with `--all`), rebuilds the self-play CSVs, appends the live GUI runs' probe marks, renders
   `master.py`, makes the hosted body (`make_hosted.py`) and copies it to `documentation/artifacts/dcm-master-dashboard.html`.
5. `tick.py` calls `replay.log_backfill` instead of its own loop.

## Refresh command

```
DCM_BIN="$HOME/Library/Application Support/DrewsChessMachine/FrozenBuilds/DCM-2687-d8afa524.app/Contents/MacOS/DrewsChessMachine" \
  /usr/bin/python3 documentation/dashboards/refresh_dashboard.py
```

Then publish `documentation/artifacts/dcm-master-dashboard.html` to the artifact and commit the changed CSVs and the
copy. A live run that starts a new segment (a resume) shows as `NEW SEGMENT` in the refresh output: add it to the run
in `registry.json` with its log, then refresh again.

## Validation

- `documentation/dashboards/tests` pass (unittest, `/usr/bin/python3`); new tests cover forked lineage runs,
  carried copies, the fork rule in `plan`, exclusions and the log test-set rows.
- `replay.py derive-registry` reports no conflicts, refused files, NEW RUN or NEW SEGMENT lines.
- Every included run has a CSV whose last `cum_step` matches the run's last checkpoint; pElo present on every
  1000-step (or 10k) row whose file exists; thinned rows filled from the log or blank.
- Spot checks: a probed row equals the file's embedded lichess-wide result; R-elitecyc forks start at 384,964+.
- `master.py` renders every run; the hosted copy is byte-identical to the rendered body.

## Decisions

- D-1: "very short test" = under ~2,000 steps, or a smoke / speed / determinism / resume test, or a validation rerun
  that reproduces an included curve bit for bit (B-silu-ctl15, V-1, V-3).
- D-2: branches start at 0 (the `qeu8e` convention); forks share the parent's axis.
- D-3: thinned checkpoints take the writer's logged test-set result (same evaluation); `wide_pelo_record.json` is not
  used (no model ID).
- D-4: files that exist are probed with DCM-2687 (pLogit and the internals need the file; one build across the rows).
- D-5: the abandoned launches (elitefresh `20261017-182142`, R-fixedlr `20261014-140421` / `20261015-102200`) are left
  out: later launches resumed from earlier saves.
- D-6: `refresh_dashboard.py` was a retired script for the pre-master static dashboard (paths under
  `~/Documents/drews-chess-machine`); it becomes the refresh entry point.
- D-7: cron (`cron_tick.sh`) is unchanged: it keeps tracking the live replay run's rows every 5 minutes; it does not
  render the master page or touch the artifact copy, so the tree is not dirtied every 5 minutes.
- D-8: pre-September runs that were never on the dashboard (2026-06-27 `v3_8block_3x3` / `freshNet`, the 2026-07-12
  train-vs-UCI pilots) stay off: the earlier dashboard work left them off, and the owner's instruction concerns runs
  since then; adding them is a registry entry each if wanted.
