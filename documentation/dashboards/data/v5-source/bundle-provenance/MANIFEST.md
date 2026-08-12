# v5 training — complete data manifest

Written 2026-08-02. Inventory of **every artifact** for the v5 network's training
history, for eventual consolidation back on the original machine.

The v5 lineage is one continuous corpus-replay training chain, split into segments
by machine moves and reboots. Each segment restarts its own step counter at 1, so
**a raw step number is meaningless without knowing its segment.** The cumulative
axis is the only comparable one.

---

## 1. The segment chain

| seg | where | steps | games fed | epoch | model_id | probe data |
|---|---|---:|---:|---|---|---|
| pre-move | ORIGINAL machine | ≥39,419 (see §5) | 12,933,495 | 0 → 0.62 | `20260629-1-Uf4p` | **none — on the other machine** |
| run 1 | this machine | 268,506 | 34,649,108 | 0 → 2 | — | `new_ckpts.jsonl` |
| run 2 | this machine | 336,610 | 43,415,877 | 2 → 4 | `20260714-1-h7vI` | `new_ckpts_run2.jsonl` |
| run 3 | this machine | 106,333 | 13,745,325 | 4 → 5 | `20260729-1-VZ2j` | `new_ckpts_run3.jsonl` |
| run 4 | this machine | in progress | — | 5 → 6 | (minted at start) | `new_ckpts_run4.jsonl` |

**Total through run 3: 104,743,805 games = 5.00 epochs** over a ~20.94M-game corpus.

### Cumulative-step offsets (`cum = offset + raw step`)

| segment | offset | derivation |
|---|---:|---|
| run 1 | 0 | — |
| run 2 | 268,506 | run 1's final step |
| run 3 | 605,116 | 268,506 + 336,610 |
| run 4 | 711,449 | 605,116 + 106,333 |

---

## 2. Records (all 713 checkpoints through run 3)

| | value | where | when |
|---|---|---|---|
| best pElo | **1770.5** | run 2 raw `step270000` = cum 538.5k | 2026-07-25 07:09 |
| lowest NLL | **1.8445** | run 2 raw `step268000` = cum 536.5k | 2026-07-25 05:12 |

Per-run bests: run 1 — 1704.0 @ cum 129.0k · run 2 — 1770.5 @ cum 538.5k ·
run 3 — 1735.4 @ cum 614.1k.

Probe battery: `wide`, 4435 fixed puzzles, deterministic. `v5 baseline`
(pre-training reference) = pElo 1584.2, top1 47.8%, NLL 2.097.

---

## 3. What is stored, and where

### Complete — every checkpoint, no gaps
- `monitor/new_ckpts.jsonl` — run 1, 269 probes
- `monitor/new_ckpts_run2.jsonl` — run 2, 337 probes
- `monitor/new_ckpts_run3.jsonl` — run 3, 107 probes
- `monitor/records.json` — standing records
- `monitor/build_dashboard.py` — regenerates `v5-strength.html` from the jsonl
  files alone (reproducible from data, not from its own prior output)

### Training logs
- `run.out` — **run 1 AND run 2 concatenated**, both numbering from step 1.
  Split where the step number decreases (run 2 begins at line 5926).
- `run-resume.out` — run 3
- `run4.out` — run 4
- `archive/logs/*.gz` — **full per-step session logs**, gzipped. Far more detailed
  than the `run*.out` files (which sample every 50 steps). Originals live at
  `~/Library/Logs/DrewsChessMachine/` and are NOT part of the bundle:

  | file | run | span | raw | gz | lines verified |
  |---|---|---|---:|---:|---|
  | `dcm_log_20260702-201756.txt.gz` | run 1 | Jul 2 → Jul 13 | 2.4 GB | 551 MB | 86,488 ✓ |
  | `dcm_log_20260713-195047.txt.gz` | run 2 | Jul 13 → Jul 28 | 3.0 GB | 691 MB | 108,418 ✓ |
  | `dcm_log_20260728-223132.txt.gz` | run 3 | Jul 28 → Aug 2 | 947 MB | 218 MB | 34,267 ✓ |
  | `dcm_log_20260802-151950.txt` | run 4 | Aug 2 → | growing | — | still open |

  6.35 GB → 1.46 GB. Each archive passed `gzip -t` and its decompressed line
  count matches the original exactly. **Originals were NOT deleted** — they remain
  at `~/Library/Logs/DrewsChessMachine/`. Compress run 4's log the same way once
  that run ends.

### Weights
- `v5-checkpoint.safetensors` — pre-move endpoint, the chain's entry point here
- `preserved-best/` — best checkpoints by pElo and NLL from runs 2 and 3, plus
  each run's final save. **These are the only weight files that survive.**
- `v5-cont-replay-step*.safetensors` — enumerated saves of the CURRENT run only.
  Each run restarts numbering at 1 and **overwrites the previous run's files**.
  A filename therefore does not identify a checkpoint; see §4.

### Config
- `parameters.json` — hyperparameters. **Byte-identical across runs 1–4**
  (lr 0.01, batch 4096, wd 2.5e-4, momentum 0.93, gradClip 30, replayRatio 0.48).
- `corpus/corpus.json` — corpus manifest. Note `state: recording`,
  `gamesAdded: 0` — never finalized, so the game count comes from the runner
  counting shards, not from this file.

---

## 4. Traps for whoever consolidates this

1. **Step numbers repeat across segments.** Always carry the segment label or the
   cumulative offset. Four different files have been named `step1000.safetensors`.
2. **`run.out` holds two runs.** Parsing it as one sequence produces impossible
   epoch wraps (a 2→3 wrap at step 118k *after* a 1→2 wrap at 225k).
3. **Old weight files are not evidence of the current run.** Verify by mtime AND
   by the `model_id` in the safetensors `__metadata__`. A monitor that trusted
   filenames alone once produced nine fabricated data points from July-13 files.
4. **The epoch counter is cumulative across the resume chain**
   (`epochsCompleted = startEpoch`, seeded from `replay_epoch`). It is NOT
   per-run, and `--epochs N` is an absolute ceiling, not "N more passes."

   **…but ONLY under `--resume-exact`.** `startEpoch` is assigned in that branch
   alone; the `--start-shard` / `--start-game-index` paths leave it at 0. So a
   segment started with `--start-game-index` restarts the epoch label at 0 and
   `--epochs N` then means N full passes. **Run 4 is such a segment** — its
   `replay_epoch` counts from 0 while the true cumulative total continues from 5.
   True epochs = 5 + run 4's reported epoch.

5. **A cleanly-finished run cannot be resumed with `--resume-exact`.** A run that
   exhausts its epoch budget saves at `(nextGame=0, shard=0, epoch=N)`, and
   `--resume-exact` rejects exactly that shape: rebuilding the replay buffer would
   need the PRIOR epoch's tail, which is unsupported. It exits 2 with an
   actionable message. Use `--start-game-index 0` instead — approximate
   cold-refill resume — and set `--epochs` accordingly per trap 4. This bit run 4's
   first launch attempt (`run4-failed-resume-exact.out`).
5. **`~/Library/Logs/DrewsChessMachine/` holds ~1,493 zero-byte files** — one per
   `--probe-model` invocation. Noise, but they inflate the file count.

---

## 5. Known gap — the pre-move history

`v5-checkpoint.safetensors` metadata:

```
model_id                20260629-1-Uf4p
parent_model_id         20260628-9-OdUt     <- at least one earlier ancestor
training_step           39419
replay_epoch            0
replay_next_game_index  12933495
built_by_git            4d60ca2
```

The bundle README describes the pre-move work as "~99.9k steps," but this
checkpoint reports `training_step = 39419`. Since `parent_model_id` points to a
further ancestor, 39,419 is almost certainly that *final segment's* counter, not
the cumulative total — so the pre-move phase was at least two segments.

**Exactly known:** the corpus position at handoff (game 12,933,495 of ~20.94M,
epoch 0) and the endpoint strength (pElo ~1583, peak ~1604).
**Not recoverable here:** per-checkpoint telemetry, step counts, and session logs
for anything before `20260629-1-Uf4p`. Those artifacts are on the original
machine and are what consolidation should merge in.
