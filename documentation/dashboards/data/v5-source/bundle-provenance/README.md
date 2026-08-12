# v5-continue-bundle — everything that existed nowhere else

These are the files that were unique to `~/Downloads/v5-continue-bundle/`. Every other
part of that bundle is duplicated elsewhere and was verified so by hash: all **650
checkpoints** exist on the master-record machine's `Models/`, all **6 session logs**
match its `Logs/` byte-for-byte, the **47 corpus files** match its `Corpora/`, and the
probes, checkpoint manifest and run parameters are committed one directory up.

This folder is what makes the bundle safe to delete without losing anything.

## Retired monitor tooling

The v5 continuation runs were watched by a standalone loop that lived in the bundle, not
by the repo's tracker. It has been superseded by `replay.py` + `master.py`, but these are
the only record of how it worked and how the published artifact was produced.

| file | role |
|---|---|
| `build_dashboard.py` | regenerated `v5-strength.html` from the probe JSONL |
| `build_html.py` | earlier version of the same |
| `build_table.py` | the stitched per-checkpoint markdown table (`OFFSET=268506`) |
| `check_record.py` | tested a new probe against the standing records |
| `v5-strength.html`, `.bak` | the published strength artifact as it last stood |
| `table.md` | a generated per-checkpoint table — superseded by `../v5.csv` |
| `records.json` | the standing records: pElo 1770.461, NLL 1.8445 |

⚠️ Anything here reporting a **cumulative step is 100,320 too low.** That monitor numbered
run 1 from 0 and knew nothing of the three earlier segments. Its "pElo 1770.5 @ cum
538.5k" is cum **638,826** on the unified axis. See `documentation/v5-lineage.md` §4.

## Forensic

`new_ckpts_run3.jsonl.corrupt.bak` — **the fabricated rows.** On 2026-07-29 a watcher
treated *file existence* as "new checkpoint". Because every run reused the same step
numbers, it instantly "found" step2000, step3000, … and probed nine July-13 files within
minutes, producing a fake run-3 curve (1594→1626→1656→1669…) and a fake "6× logit
collapse" that reached both this repo's notes and a published artifact.

Three independent tells would each have caught it: the files' mtimes were two weeks old;
their `modelID` was the previous run's; and checkpoints take ~59 min, so probes arriving
minutes apart were themselves the alarm.

This is why `replay.py import-probes` refuses any probe whose `modelID` does not match its
segment's registry `model_id`. The bad data is kept beside the rule it produced.

## Contemporaneous narrative

Written while the runs were happening, so they record intent and uncertainty that the
tidied-up account cannot. `documentation/v5-lineage.md` supersedes them and folds in the
durable content, but paraphrased — these are the primary sources.

| file | what it is |
|---|---|
| `MANIFEST.md` | the 2026-08-02 inventory, written *for* the eventual consolidation |
| `HANDOFF.md` | the monitor loop's resume runbook |
| `bundle-README.md` | the original "how to run this on another Mac" instructions (2026-07-02) |
| `quarantine-README.md` | why three run-3 files sat under run-4 names, and why run 4's final save is poisoned |
| `SUPERSEDED.md`, `bundle-STATUS.md` | written during consolidation, marking the bundle retired |

Two known errors in `MANIFEST.md`, both corrected in `v5-lineage.md`: §3 claims
`preserved-best/` holds the only surviving weights (run 1's 268 enumerated checkpoints
survived complete, plus 231 of run 2's), and §5 calls the pre-move history unrecoverable
(it was in `../v5.csv` all along).

## Sampled console output

`run.out`, `run-resume.out`, `run4.out`, `run5.out`, `run4-failed-resume-exact.out` —
stdout from each run, the `[REPLAY]` line every 50 steps.

**Redundant in content, not in bytes.** The full session logs on the master machine carry
the same lines plus timestamps and everything else; these are kept as the exact view the
operator actually watched. Note `run.out` holds **two concatenated runs** (segments 3 and
4), both numbering from step 1 — split where the step number decreases.
`run4-failed-resume-exact.out` corresponds to `dcm_log_20260802-151950.txt`.

`telemetry-snapshot-SHA256SUMS.txt` is the checksum manifest of the 2026-08-11 telemetry
snapshot taken during consolidation.

## Deliberately not kept

`.DS_Store` (Finder metadata), `default.profraw` (709 KB profiler artifact),
`monitor/checkpoint_inventory.log` (stdout of `ckpt_inventory.py`, regenerable) — no
informational content.
