# 2026-07-10 — wxil: deep-narrow fat-conv (8 blocks × 16 ch, 15×15)

**Status:** abandoned (terminated by the user at step 23,350 / ~3.2 h as underperforming; commit 324bf6f)

## Question

- Does trading nt8y's width for depth help, with the same 15×15 fat convs and a smaller budget?
  - nt8y: 3 blocks @ 32 channels.
  - wxil: 8 blocks @ 16 channels, ~1.05M params.

## Setup

- **Architecture** (embedded JSON of `20260711-014615-20260711-1-wXIL-manual.safetensors`, same as the log's `[REPLAY-ARCH]`):
  - v5 · basic30(30) → stem 5×5 → 16 · **8×[15×15+15×15 @16**, SE scale+bias /4, ReLU/pre, clean_add, **ReZero α init 0.125 (cap 0.125)**, out: layer_norm, dropout×1] · policy intermediate_conv (pre-conv 512) → 4864 · value WDL (16 → FC64) · bfloat16 · **1,052,183 params** (161 tensors).
- **Differences from nt8y's architecture JSON:** `block_groups[0]` has channels 32→16, count 3→8, rezero_alpha_init 0.333→0.125. The stem (5×5) and heads are identical.
- **Recipe** (`[REPLAY-HPARAMS]`): lr 0.01, batch 4096, wd 0.0005, momentum 0.9, gradClip 30, pLabelSmooth 0.1, vLabelSmooth 0.013, lrWarmup 500, bufCap 1,000,000, replayRatio 0.48, minPrefill 500,000, complementCE on, sqrtBatchLR on, `--epochs 12 --enumerate-checkpoints`.
  - These are the same settings the nt8y stem arms used later ([stem-kernel write-up](../20260710-nt8y-stem-kernel-and-seed-variance/README.md)).
- **Corpus:** std `20260624-192615-w3aA5b`, with the same game order as every other arm. `games=` at step 1k/5k/20k/23k = 136,804 / 655,048 / 2,584,454 / 2,967,684, identical to the stem arms.
- **Build / host:** build 2072, git 400f63d, this Mac (corpus path `~/Library/Application Support/DrewsChessMachine/Corpora/…`).

## Runs

- Registry key `wxil`.
- Fresh model: GUI-built, saved as `20260711-014615-20260711-1-wXIL-manual.safetensors` (model_id 20260711-1-wXIL, creator manual).
- Trainer lineage: 20260711-2-b5TY (parent wXIL).
- Log: `dcm_log_20260710-210320.txt`, 2026-07-10 21:03:20 → 2026-07-11 00:19:28, last logged step 23,350.
- Checkpoints: `20260711-wXIL-std2026_05-replay-step{1000..23000}` (last = step 23000).
- Ran alone. It ended before nt8y3x3 started (00:26:49); only zero-byte logs overlap it.

## Results

pElo and nll are from `documentation/dashboards/data/wxil.csv` (replay-era probe scale), compared at matched step, which here is also matched games_fed.

| step | wxil pElo | nll | nt8y 5×5 pElo | nt8y3x3 pElo | nt8y15x15 pElo |
|---:|---:|---:|---:|---:|---:|
| 1000 | 769.6 | 3.5342 | 769.6 | 740.0 | 769.6 |
| 5000 | 831.4 | 3.2967 | 992.5 | 962.2 | 961.1 |
| 10000 | 907.2 | 3.2162 | 1091.2 | 1050.6 | 1077.6 |
| 14000 | 986.6 | 2.9933 | 1146.7 | 1105.4 | 1084.4 |
| 15000 | 918.9 | 3.1018 | 1135.2 | 1102.8 | 1093.4 |
| 20000 | 1001.1 | 2.9317 | 1242.8 | 1189.4 | 1157.2 |
| 23000 | 1036.3 | 2.9238 | 1267.1 | | 1173.3 |

| pair (A − B) | steps | n | pElo mean | max \|Δ\| (step) | A ahead | nll mean |
|---|---|---:|---:|---:|---:|---:|
| 15×15 − wxil | 1k–23k | 23 | +138.5 | 202.7 (16k) | 22/23 | −0.3120 |
| 3×3 − wxil | 1k–20k | 20 | +136.0 | 203.5 (17k) | 19/20 | −0.3201 |
| nt8y 5×5 − wxil | 1k–23k | 23 | +180.5 | 259.7 (19k) | 22/23 | −0.3816 |

- **Speed:** ~503 s per 1,000 steps (log ~452–457 ms/step), the slowest of the nt8y-family arms despite the fewest params. Eight sequential 15×15 blocks cost more per step than three wider ones.
- **Instability:** one sharp dip at 15k (986.6 → 918.9 pElo, nll 2.9933 → 3.1018), which it had not made back by 18k.

## Conclusion

- At this budget, deep-narrow is clearly worse:
  - It trails every 3-block @32 arm by ~136–180 pElo on average and ~0.31–0.38 nll at the same games seen.
  - It is ~6–12% slower per step.
- The gap is 7–10× the 3×3 seed spread (18.6 pElo) and far outside the probe jitter, so it is not noise.
- The comparison against the two new stem arms has the same recipe and host (only the build differs), so that part of the result is clean.

## Caveats

- **Single seed**, and it ran only to 23k steps (early training, epoch 0). The ranking could change later, but the gap was still widening at 20k (+188 vs 3×3).
- **Width, depth and ReZero init change together** (α 0.125 = 1/8 for 8 blocks vs 1/3 for 3 blocks), so this does not isolate depth.
- The nt8y 5×5 comparator is confounded (host, build, wd/momentum). See the stem-kernel write-up.
- Probe pElo is quantized and single-sample per checkpoint.

## Follow-ups

- If depth is revisited, hold width at 32 and vary only the count (3 vs 6), with α = 1/count, on the current build.

## Audit notes

- **Verified:**
  - Param count 1,052,183 (tensor-element sum of the fresh and step-23000 checkpoints; `[REPLAY-ARCH]` line; registry `params`).
  - The architecture fields listed above were diffed against nt8y's JSON. The step-23000 checkpoint has the same architecture hash as the fresh model (153aeb8fa0ba).
  - Hyperparameters and timing come from the log. Termination reason from commit 324bf6f ("wXIL terminated by user (underperforming)").
- **Correction (data record):** the `wallclock_iso` column in `wxil.csv` and the registry segment `date: "20260711"` → the run actually started **2026-07-10 21:03**.
  - Evidence: log filename `dcm_log_20260710-210320.txt`; log mtime 2026-07-11 00:19; registry commit 16947f6 at 2026-07-10 21:08.
  - The CSV's step-1000 row reads `2026-07-11T21:11:57`, one day late. Its time-of-day values are consistent with the log.
  - **Fixed 2026-09-29:** registry segment `date` → `20260710`, and the CSV's `wallclock_iso` re-derived from it with `replay.py recompute wxil` (step 1000 now reads `2026-07-10T21:11:57`). That recompute also filled the `wall_sec` column, which this CSV's older schema lacked; `games_fed` stays blank.
- **Unverified:** nothing further.

## Reproduce

**Status: partial** — start model, build, corpus, hparams known; parameters file and exact command not recorded.

- **Commit / build:** 2072 / `400f63d` (`built_by_build`/`built_by_git` of `20260711-wXIL-std2026_05-replay-step1000.safetensors`). The replay CLI writes no `[APP]` line.
- **Corpus:** [`20260624-192615-w3aA5b`](../corpora/20260624-192615-w3aA5b.md).
- **Starting point:** `~/Library/Application Support/DrewsChessMachine/Models/20260711-014615-20260711-1-wXIL-manual.safetensors` (model_id 20260711-1-wXIL, GUI-built; still present). No preset file for this arch exists; its architecture is embedded in that checkpoint.
- **Parameters:** no parameters file recorded. `[REPLAY-HPARAMS]` in `dcm_log_20260710-210320.txt`: `lr=0.01 batch=4096 wd=0.0005 momentum=0.9 gradClip=30 pLabelSmooth=0.1 vLabelSmooth=0.013 lrWarmup=500 bufCap=1000000 replayRatio=0.48 minPrefill=500000 complementCE=on sqrtBatchLR=on` (others 0/1 defaults as logged).
- **Command** (reconstructed): `"$BIN" --replay-corpus 20260624-192615-w3aA5b --start-model "$M/20260711-014615-20260711-1-wXIL-manual.safetensors" --out-model "$M/20260711-wXIL-std2026_05-replay-latest.safetensors" --parameters <file with the values above> --epochs 12 --enumerate-checkpoints`; stop at step 23,350 to match.
- **Probe / analysis:** `documentation/dashboards/replay.py`, registry key `wxil` (wide-set `--probe-model` per enumerated checkpoint).
- **Expected exactness:** statistical only, not bit-exact. Replay minibatch sampling uses unseeded `Int.random` (`ReplayBuffer.sample`), fresh nets use a random init with no seed flag, and bf16 GPU execution is not guaranteed deterministic. Corpus game *order* is deterministic, so `games=` at a given step matches exactly.
- **Missing:** the parameters file; the literal command line; the `-latest` out-model name is inferred from the checkpoint stem.
