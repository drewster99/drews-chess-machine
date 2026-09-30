# 2026-07-11 — qeu8 width vs depth: 2 blocks @64 vs 1 block @128

**Status:** abandoned at 25.96 h of the planned 31.3 h. It was terminated on 2026-07-12 12:17 to free the machine for train-vs-UCI (commit `e74596c`).

## Question

Keep qeu8's recipe (15×15 fat convs, 7×7 stem, SE scale+bias, ReZero 0.5, LN-out, and the same heads) but trade depth for width: 1 block @128 channels instead of 2 blocks @64. That roughly doubles the parameters (3.93M → 7.75M). Does a wider, shallower tower learn more per hour, per step or per game from the std corpus?

## Setup

- **qeu8 (baseline), 3,929,393 params:** basic30 → stem 7×7→64 · **2×**[15×15+15×15 **@64**, SE scale+bias/4, ReLU/pre, clean_add, ReZero 0.5·tanh, LN-out] · policy intermediate_conv (pre-conv 512) · value WDL(16→FC64) · bf16. Seed `20260702-7-Qeu8` (GUI-built).
- **qeu8-1blk128 (registry `qeu8b1128`), 7,750,320 params:** the same recipe with **1×**[… **@128**]. Preset `~/Library/Application Support/DrewsChessMachine/Presets/qeu8_1blk_128.json`. Fresh seed `20260711-qeu8-1blk128-fresh.safetensors` (modelID **20260711-16-VRR4**, `creator = new-model`, notes "fresh qeu8_1blk_128 net (untrained), arch v5").
- Both architectures were checked against the embedded architecture JSON. The two JSONs differ only in `channels` (64 → 128) and `count` (2 → 1).
- **Corpus:** std `20260624-192615-w3aA5b` for both runs.
- **Launch:** qeu8b1128 was launched with `--epochs 12 --enumerate-checkpoints`. Log `dcm_log_20260711-101346.txt` (1.0 GB), build 2080 / git `29ba72e` (header). qeu8's builds were 2013 / `48fd638`. Both ran on the current Apple M5 Max host; qeu8's segments 0–2 predate the 2026-07-07 migration (see the parity write-up).
- **Initialization confound:** qeu8 started from a GUI-built seed; qeu8b1128 from a CLI `new-model` seed.

**Parameter-scaling formula check** (from memory `dcm-parity-goal.md`): `params = 2006·C + B·(450.75·C² + 12.25·C + 1) + 106895` for a 7×7 stem.
- C=64, B=2: 128,384 + 2·(1,846,272 + 784 + 1) + 106,895 = **3,929,393**. This matches qeu8 exactly.
- C=128, B=1: 256,768 + (7,385,088 + 1,568 + 1) + 106,895 = **7,750,320**. This matches VRR4 exactly.
- The per-block term was checked against the tensor shapes in the VRR4 header:
  - 2 conv weights: 2·225·C².
  - SE fc1 `[C/4, C]` + bias C/4 and fc2 `[2C, C/4]` + bias 2C: together 0.75·C² + 2.25·C.
  - bn1 + bn2, each with weight/bias/running_mean/running_var: 8·C.
  - `res_ln` weight/bias: 2·C.
  - `rezero_alpha`: 1.
  - Sum: 450.75·C² + 12.25·C + 1.
- Note that the "params" figure counts BatchNorm running statistics, which are not trainable.
- Blocks are 94.0% (qeu8) and 95.3% (1blk128) of the total.

## Runs

| run | seed → trained ModelID(s) | segments | log(s) | end |
|---|---|---|---|---|
| qeu8 | 20260702-7-Qeu8 → GLu5 → Lnji → PVZp (→ Ejp0 in resume3) | 4 (see registry `qeu8`) | all present | parity at cum 175,915 / 31.33 h; kept going to cum 1,572,915 |
| qeu8b1128 | 20260711-16-VRR4 → **20260711-17-pycz** | 1 | `dcm_log_20260711-101346.txt` | last save step 120,000 (header: epoch 0, next game 15,461,577). The log runs to step 120,430 with no `done:` line (hard termination). |

qeu8b1128 checkpoints: 122 files matching `20260711-qeu8-1blk128-*` (fresh, `-std-replay-latest` = pycz @120,000, and enumerated `-std-replay-stepN`).

## Results

Both runs trained on the same corpus from game 0, so steps map to games almost 1:1 (batch 4096, replay ratio 0.48).

**Matched step** (last probed row at or before S; pElo / nll, elapsed):

| cum step | qeu8 | qeu8b1128 |
|---:|---|---|
| 10,000 | 1175.4 / 2.569 (1.78 h) | 1155.1 / 2.561 (2.19 h) |
| 30,000 | 1351.6 / 2.360 (5.34 h) | 1330.5 / 2.373 (6.69 h) |
| 60,000 | 1484.6 / 2.227 (10.47 h) | 1393.2 / 2.311 (13.33 h) |
| 90,000 | 1510.8 / 2.191 (15.49 h) | 1428.7 / 2.270 (19.65 h) |
| 106,000 | 1543.1 / 2.153 (18.37 h) | 1487.7 / 2.203 (23.00 h) — qeu8b1128 peak |
| 120,000 | 1543.7 / 2.153 (21.00 h) | 1460.5 / 2.233 (25.96 h) — qeu8b1128 end |

**Matched time:**

| T (h) | qeu8 | qeu8b1128 |
|---:|---|---|
| 2 | 1211.7 / 2.531 (cum 11,000) | 1142.6 / 2.606 (cum 9,000) |
| 5 | 1349.0 / 2.376 (28,000) | 1242.8 / 2.478 (22,000) |
| 10 | 1459.0 / 2.217 (56,407) | 1406.6 / 2.303 (44,000) |
| 15 | 1515.9 / 2.159 (86,407) | 1392.7 / 2.292 (67,000) |
| 20 | 1517.5 / 2.164 (113,915) | 1422.0 / 2.266 (91,000) |
| 25.9 | 1607.8 / 2.067 (146,915) | 1453.8 / 2.255 (119,000) |

**Summary:**

| | qeu8 | qeu8b1128 |
|---|---:|---:|
| params | 3,929,393 | 7,750,320 |
| median ms/step (CSV) | 560.0 (segs 0–2) | 684.3 |
| steps at 25.96 h | ~147k | 120,000 |
| games fed at end of the compared window | 14,023,256 at cum 108,915 (segs 0+1 `done:`) | 15,461,577 at 120,000 |
| pElo / nll at ~14–15.5M games | 1546.2 / 2.158 (cum 108,915) | 1460.5 / 2.233 (cum 120,000) |
| peak pElo ≤ 25.96 h | 1627.3 (cum 142,915, 25.14 h) | 1487.7 (cum 106,000, 23.00 h) |
| best nll ≤ 25.96 h | 2.0613 (cum 142,915) | 2.1997 (cum 118,000) |

## Conclusion

- **Depth beats width here.** At equal steps and games the 2-block @64 net leads by ~50–90 pElo and ~0.05–0.08 nll from 60k steps on. Only up to ~30k steps are the two within probe noise.
- **Per hour the gap is larger.** The 1-block @128 net's steps are ~22% slower (684 vs 560 ms median), so by 25.9 h qeu8 leads by ~150 pElo and 0.19 nll.
- **Doubling the parameters bought nothing.** 7.75M params in one wide block learned less than 3.93M params in two blocks. The formula shows why: parameters (and compute) grow with C² for width but only linearly for depth, and the second block adds another nonlinearity and another 15×15 receptive field on top of the first.
- The run stopped at 25.96 h, so there is no 31.3 h parity point. The gap was widening, not closing, when it stopped.

## Caveats

- Single seed each, and the seeds came from different builders (GUI vs CLI `new-model`).
- qeu8's first 108,915 steps span two segments with a resume at cum 41,407 that restarted the corpus at game 0 and zeroed optimizer velocity. qeu8b1128 ran as a single uninterrupted segment. If anything this favours qeu8b1128.
- Builds differ (2013 vs 2080). No training-math change between them is known, but none was checked for this write-up.
- There are no arena games for qeu8b1128: it did not exist at the 2026-07-08 arena.
- Probe noise is ±30–60 pElo per row.

## Follow-ups

- If width is revisited, test 2 blocks @96 (≈ 8.5M total by the formula) against 3–4 blocks @64 at a matched parameter or compute budget, instead of trading a block for width.
- A 2-block @128 net would cost ~15M params and ~2× qeu8b1128's step time. It was priced in memory, but no data exists for it.

## Audit notes

- Verified: 7,750,320 params (tensor sum of VRR4 and pycz, and `[REPLAY-ARCH]` in `dcm_log_20260711-101346.txt`); 3,929,393 for qeu8 (PVZp, Lnji headers).
- Verified: the parameter formula is exact for both points, and its per-block term is structurally derived from the header tensor shapes (above), not just curve-fitted.
- Verified: the fresh ModelID `20260711-16-VRR4` and trained ModelID `20260711-17-pycz` with parent VRR4 (header).
- **Correction / clarification:** memory says the run was set "to 31.3h". It actually ended at **25.958 h** (cum 120,000, last CSV row, pElo 1460.51). Commit `e74596c` ("qeu8b1128 final snapshot before terminate (switching to train-vs-uci)", 2026-07-12 12:17) records a deliberate early stop.
- qeu8 games: 5,338,859 (seg0 `done:`, `dcm_log_20260702-095124.txt`) + 8,684,397 (seg1 `done:`, `dcm_log_20260702-182922.txt`) = 14,023,256 at cum 108,915.
- **Unverified:**
  - Whether the preset JSON on disk still matches what was used. It was not re-read, but the embedded architecture in VRR4 is authoritative anyway.
  - The exact termination mechanism: the log has no SIGINT or `done:` line.

## Reproduce

**Status: partial** — qeu8b1128 arm recorded; qeu8 baseline is a multi-segment run on older builds.

- **Commit / build:** qeu8b1128 2080 / `29ba72e` (checkpoint `built_by_build`/`built_by_git`). qeu8 baseline 2013 / `48fd638` for its early segments (see the parity write-up).
- **Corpus:** [`20260624-192615-w3aA5b`](../corpora/20260624-192615-w3aA5b.md) for both.
- **Starting point:** qeu8b1128 `~/Library/Application Support/DrewsChessMachine/Models/20260711-qeu8-1blk128-fresh.safetensors` (20260711-16-VRR4, still present); preset `~/Library/Application Support/DrewsChessMachine/Presets/qeu8_1blk_128.json` (still present; the embedded architecture in VRR4 is authoritative). qeu8 `20260702-164826-20260702-7-Qeu8-manual.safetensors` (20260702-7-Qeu8, still present).
- **Parameters:** no parameters file recorded. qeu8b1128 `[REPLAY-HPARAMS]` (`dcm_log_20260711-101346.txt`): `lr=0.01 batch=4096 wd=0.0005 momentum=0.9 gradClip=30 pLabelSmooth=0.1 vLabelSmooth=0.013 lrWarmup=500 bufCap=1000000 replayRatio=0.48 minPrefill=500000 complementCE=on sqrtBatchLR=on`.
- **Command** (reconstructed): `"$BIN" --replay-corpus 20260624-192615-w3aA5b --start-model "$M/20260711-qeu8-1blk128-fresh.safetensors" --out-model "$M/20260711-qeu8-1blk128-std-replay-latest.safetensors" --parameters <file with the values above> --epochs 12 --enumerate-checkpoints`; stop at step 120,000 (25.96 h) to match. The baseline's commands are per segment; see registry `qeu8` and the parity write-up.
- **Probe / analysis:** `documentation/dashboards/replay.py`, registry keys `qeu8b1128` and `qeu8`.
- **Expected exactness:** statistical only, not bit-exact. Replay minibatch sampling uses unseeded `Int.random` (`ReplayBuffer.sample`), fresh nets use a random init with no seed flag, and bf16 GPU execution is not guaranteed deterministic. Corpus game *order* is deterministic, so `games=` at a given step matches exactly.
- **Missing:** parameters files for both runs; literal command lines; the qeu8 baseline cannot be rerun as one command (4 warm-restart segments on 2 hosts).
