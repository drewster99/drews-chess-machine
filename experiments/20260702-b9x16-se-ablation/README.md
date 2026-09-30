# 2026-07-02 — b9x16: SE vs no-SE on a 9-block, 16-channel 15×15 tower

**Status:** abandoned. Both arms were stopped early (no-SE at 45,416 steps, SE at 74,225). Only the SE arm has probe data and surviving checkpoints.

## Question

Does adding squeeze-excitation (SE scale+bias, reduction ratio 4) to a deep, very narrow fat-conv tower (9 blocks @16 channels, 15×15 kernels, 9×9 stem) change what it learns from the std corpus? The SE adds 2,052 parameters, about 0.17% of the net.

## Setup

- **Architecture, both arms.** Checked against the embedded architecture JSON in `20260702-9blk16se-9x9stem-seed.safetensors`, and against the `[REPLAY-ARCH]` log line for the no-SE arm:
  - basic30 → stem 9×9→16
  - 9×[15×15+15×15 @16, ReLU/pre, clean_add, ReZero 0.333·tanh, LN-out]
  - policy intermediate_conv (pre-conv 512) · value WDL(16→FC64) · bf16
  - **b9x16 (no SE):** `no-SE`, **1,192,600 params** (from its `[REPLAY-ARCH]` line; no checkpoint survives).
  - **b9x16se:** SE scale+bias/4, **1,194,652 params** (tensor-element sum of the seed and final checkpoint). The difference is 2,052 = 9 blocks × (16·4+4 + 4·32+32), which is the SE FC weights and biases.
- **Seeds:** both are hand-crafted headless seeds (`creator = handcraft` on the SE seed). Seed notes: "9-block uniform-16 9x9-stem 15x15 +SE(scale_and_bias/4) ReZero; value draw-prior; SE fc2 Glorot". The no-SE seed file `20260701-9blk16-9x9stem-seed.safetensors` (modelID `20260701-1-B9x16`) no longer exists.
- **Corpus:** `20260624-192615-w3aA5b` (std), read from `/Volumes/20260624-192615-w3aA5b`.
- **Hyperparameters:** identical `[REPLAY-HPARAMS]` in both logs: lr=0.01, batch=4096, wd=0.00025, momentum=0.93, gradClip=30, pLabelSmooth=0.1, vLabelSmooth=0.013, lrWarmup=500, bufCap=1,000,000, replayRatio=0.48, minPrefill=500000, complementCE=on, sqrtBatchLR=on.
- **Build:** SE arm build 2011 / git `e7c52d9` (header). The no-SE arm's build is unverified: no `[APP]` line is in the log, and no checkpoint survives.

## Runs

| arm | registry key | log | seed modelID → final modelID | steps | games fed | stop |
|---|---|---|---|---:|---:|---|
| b9x16 (no SE) | none | `dcm_log_20260701-223131.txt` (404 MB) | 20260701-1-B9x16 → (unknown; file gone) | 45,416 | 5,849,620 | SIGINT 2026-07-02 02:44 |
| b9x16se | `b9x16se` | `dcm_log_20260702-024634.txt` (660 MB) | 20260702-1-B9SE → **20260702-4-DEQi** | 74,225 | 9,551,945 | SIGINT 2026-07-02 09:49 |

Earlier, abandoned starts from the same evening:
- `dcm_log_20260701-222338.txt`: b9x16 no-SE, 1,137 steps, SIGINT, then restarted from the same seed as the run above.
- `dcm_log_20260701-221652.txt`: `uni16-9x9stem`, a 3-block @16 no-SE net with 500,434 params, 1,125 steps, SIGINT. Its checkpoint `20260702-1-CPZm` was rated 156.0 in the 2026-07-08 arena.

**Surviving checkpoints:**
- b9x16se: 76 files matching `20260702-9blk16se-9x9stem-*`, including the seed, `-replay-latest` (DEQi, step 74,225) and frozen step files.
- b9x16 no-SE: **none**. No `20260701-9blk16-*` file exists in `Models/`, so the no-SE arm cannot be puzzle-probed or played.

## Results

The no-SE arm has no probe data, so the comparison uses the per-step trainer telemetry that both logs print every 50 steps. Each cell is the mean over the 20 logged steps in (S−1000, S]:

| step | loss (no SE / SE) | pLoss | vLoss | pIllM | playedP | pD (batch) |
|---:|---|---|---|---|---|---|
| 5,000 | 3.9398 / 3.9430 | 3.0531 / 3.0531 | 0.8191 / 0.8199 | 0.0683 / 0.0703 | 0.1247 / 0.1246 | 0.062 / 0.064 |
| 10,000 | 3.8328 / 3.8445 | 2.9641 / 2.9750 | 0.8217 / 0.8193 | 0.0476 / 0.0491 | 0.1345 / 0.1338 | 0.065 / 0.066 |
| 20,000 | 3.7563 / 3.7578 | 2.9062 / 2.9055 | 0.8187 / 0.8193 | 0.0306 / 0.0315 | 0.1407 / 0.1403 | 0.064 / 0.064 |
| 30,000 | 3.6953 / 3.7031 | 2.8610 / 2.8594 | 0.8139 / 0.8195 | 0.0227 / 0.0233 | 0.1448 / 0.1443 | 0.065 / 0.065 |
| 40,000 | 3.6797 / 3.6766 | 2.8445 / 2.8445 | 0.8174 / 0.8125 | 0.0183 / 0.0191 | 0.1470 / 0.1486 | 0.068 / 0.067 |
| 45,000 | 3.6609 / 3.6586 | 2.8336 / 2.8313 | 0.8098 / 0.8102 | 0.0169 / 0.0168 | 0.1469 / 0.1490 | 0.062 / 0.065 |
| 60,000 | — / 3.6203 | — / 2.7953 | — / 0.8104 | — / 0.0129 | — / 0.1522 | — / 0.068 |
| 74,000 | — / 3.6188 | — / 2.7758 | — / 0.8336 | — / 0.0111 | — / 0.1538 | — / 0.065 |

**Throughput:**
- Wall time from step 1 to step 45,000 was 4 h 10 m 27 s without SE and 4 h 15 m 8 s with SE, about 1.9% slower.
- Logged `ms` per step was ~252–257 without SE and ~258–261 with SE through step 45k.

**b9x16se puzzle probe** (`data/b9x16se.csv`; there is no no-SE counterpart):

| cum step | elapsed (h) | pElo | nll |
|---:|---:|---:|---:|
| 3,000 | 0.28 | 810.8 | 3.297 |
| 20,000 | 1.89 | 1055 | 2.828 |
| 45,000 | 4.25 | 1180.6 | 2.654 |
| 59,000 | 5.58 | 1232.4 | 2.583 |
| 69,000 (peak pElo, best nll) | 6.52 | 1288.2 | 2.529 |
| 73,000 (last row) | 6.90 | 1281.5 | 2.546 |

**Arena** (2026-07-08, 38 engines): `9blk16se-9x9stem` (DEQi, step 74,225) rated **334.4 ± 9.9**, rank 25 of 37 rated. It placed below every std-corpus parity line at comparable or earlier times: coxw at 5.6 h rated 462.8, nt8y seg0 at cum 65,883 rated 447.7. No-SE: not rated, because no checkpoint exists.

## Conclusion

- **SE made no measurable difference** on this tower. For 45k steps the two arms' losses agree to within ±0.01 (pLoss within 0.011), which is inside step-to-step noise. The SE arm is ~2% slower per step.
- **The architecture itself is weak for its time.** At 5.5 h b9x16se's pElo is 1261 / nll 2.572, against 1330–1380 / 2.36–2.40 for nt8y, mini2b, coxw and qeu8 at the same time (see `20260629-replay-arch-parity`). A 16-channel width appears to bottleneck the net even with 9 blocks and 15×15 receptive fields. Both arms were dropped in favour of qeu8, which launched 2026-07-02 09:51, minutes after the SE arm was stopped.

## Caveats

- There is no probe or arena data for the no-SE arm. The comparison relies only on training-batch telemetry: in-distribution loss on the batch being trained, not a held-out measurement.
- Single seed per arm, and the arms have different run lengths.
- The no-SE arm's build is unverified. The SE arm's step-count axis matches its CSV.

## Follow-ups

None planned. A decisive answer would need the no-SE arm re-run to a matched step count with frozen checkpoints and probes. Given how weak the base architecture is, this is low value.

## Audit notes

- Verified the param counts: SE from the seed and DEQi headers (1,194,652, matching registry `params`); no-SE from `dcm_log_20260701-223131.txt` `[REPLAY-ARCH]` (1,192,600). The 2,052 difference matches the SE parameter formula.
- Verified steps and games from the `done:` lines in both logs and the DEQi header (`training_step` 74,225, `replay_next_game_index` 9,551,945, epoch 0).
- Verified that the hyperparameters are identical, from the `[REPLAY-HPARAMS]` lines.
- The registry label says "SE ablation of b9x16 (identical + SE)". This is confirmed apart from the SE blocks.
- **Unverified:**
  - No-SE build and final ModelID: no `[APP]` line in the log, and the output file was deleted or overwritten.
  - The reason both arms were stopped: no commit, CHANGELOG or memory note records it. That qeu8 superseded them is inferred from timing only (qeu8 `dcm_log_20260702-095124.txt` started 2 minutes after the SE arm's SIGINT).

## Reproduce

**Status: partial** — SE arm reproducible from its surviving seed; the no-SE arm's seed and build are gone.

- **Commit / build:** SE arm build 2011, `e7c52d9` (DEQi `__metadata__`); dirty state unrecorded. No-SE arm: unknown (no `[APP]` line, no checkpoint).
- **Corpus:** [`20260624-192615-w3aA5b`](../corpora/20260624-192615-w3aA5b.md) (`replay_corpus_id` in DEQi headers).
- **Starting point:** SE seed `20260702-1-B9SE` = `Models/20260702-9blk16se-9x9stem-seed.safetensors` (`creator handcraft`, present). No-SE seed `20260701-1-B9x16` (`20260701-9blk16-9x9stem-seed.safetensors`) is **not** in `Models/` or `Sessions/`. No preset file exists; architecture is embedded in the seed.
- **Parameters:** no file preserved. Both logs' `[REPLAY-HPARAMS]`: `lr=0.01 batch=4096 wd=0.00025 momentum=0.93 gradClip=30 pLabelSmooth=0.1 vLabelSmooth=0.013 lrWarmup=500 bufCap=1000000 replayRatio=0.48 minPrefill=500000 complementCE=on sqrtBatchLR=on`, `epochLimit=1`.
- **Commands:** not recorded. From the log: `--replay-corpus <w3aA5b> --start-model 20260702-9blk16se-9x9stem-seed.safetensors --epochs 1` (the corpus was read from `/Volumes/20260624-192615-w3aA5b`); the `--out-model` stem was `20260702-9blk16se-9x9stem`.
- **Probe / analysis:** `documentation/dashboards/replay.py` registry key `b9x16se` (wide probe); the no-SE arm has log metrics only. Arena rating from the 2026-07-08 arena.
- **Expected exactness:** statistical only (unseeded `Int.random` minibatch sampling, GPU nondeterminism). A no-SE rerun needs a freshly built seed, so it would also differ in init.
- **Missing:** no-SE seed file and build; command lines; parameters files.
