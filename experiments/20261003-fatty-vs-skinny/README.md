# 2026-10-03 — Depth at a fixed budget: fatty (1 block @ 216) vs skinny (22 blocks @ 48)

**Status:** result to date. Skinny (R14) was stopped by the owner at step 1,605; fatty (R13)
continues to its 33,000-step limit and is reported here through step 5,000. Per-run detail:
`20261003-fatty-1x7x7-216/` and `20261003-skinny-22x7x7-48/`. The follow-up, slim-neck fatty
(R15), is in `20261003-fatty224-3x3stem/`.

## Question

The no-SE / no-ReZero tower — 3 blocks × [7×7 + 7×7] @ 128 (R7/R8,
`20261002-noSE-noReZero/`) — is the strongest shape of the R1–R12 series. Spend the same
parameters and compute on the two ends of a depth sweep: one wide block, or many narrow
ones. Which uses the budget best?

## Arms

| | fatty (R13) | baseline R7/R8 | skinny (R14) |
|---|---:|---:|---:|
| tower | 1 × [7×7 + 7×7] @ 216 | 3 × [7×7 + 7×7] @ 128 | 22 × [7×7 + 7×7] @ 48 |
| conv layers after the stem | 2 | 6 | 44 |
| stem | 7×7, 30 → 216 | 7×7, 30 → 128 | 7×7, 30 → 48 |
| trainable parameters | 5,064,751 | 5,167,983 | 5,193,103 |
| conv/FC MACs per position | 315.7M | 322.3M | 323.6M |
| training FLOPs per step (batch 4096) | 7.767 T (0.979×) | 7.931 T | 7.978 T (1.006×) |
| activation elements per position | 189k (0.75×) | 253k | 576k (2.28×) |
| start net (ModelID) | `20261004-8-2Sao` | `20261002-1-bh2u` / `20261002-3-x4gI` | `20261004-9-czp5` |
| run ModelID | `20261004-10-YpxP` | `20261002-2-5tKN` / `20261002-4-T79u` | `20261004-11-e7dM` |

Everything else is shared: no SE, no ReZero, ReLU pre-activation, clean_add skip,
LayerNorm block output, policy intermediate_conv (K = 128), value WDL (16 → FC128), bf16,
standard init (policy / value final layers He, draw prior 0.75). FLOPs and activation
counts come from `experiments/arch_flops.py`, which reads each start net's own header;
equal steps are equal compute to within 2.1%.

## Design

- **Identical training to R7/R8:** frozen build 2275, a byte-identical copy of
  `20261002-noSE-noReZero/parameters.json`, corpus `20260624-192615-w3aA5b`, 12 epochs,
  step limit 33,000, `--enumerate-checkpoints`, `--policy-tail-precision fp32_from_pre_bn`.
  Each run's `[REPLAY-HPARAMS]` and batch lines are identical to R7's; its
  `[REPLAY-CYCLE]` line differs only in the start net's model ID.
- **Matched wall time:** both launched 2026-10-03 20:44:59 CDT, the moment label
  smoothing D ended (fatty pid 72276, log `dcm_log_20261003-204412.txt`; skinny pid 72279,
  log `dcm_log_20261003-204412-2.txt`).
- **Baseline:** Avg(R7,R8), the mean of the two seeds' probes at each step.
- **Probes:** pElo / NLL every 1,000 steps (`--probe-set wide`, build 2275). Step 0 is the
  untrained start net (`experiments/probe_step0.sh` → `step0-probes.jsonl`).

## Results to date

`python3 experiments/20261003-skinny-22x7x7-48/table.py`:

| step | pElo fatty | pElo skinny | pElo Avg(R7,R8) | NLL fatty | NLL skinny | NLL Avg(R7,R8) |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 601.3 | 463.2 | 526.7 | 3.6079 | 3.7466 | 3.6257 |
| 1,000 | 994.1 | 668.9 | 925.5 | 2.8736 | 3.6027 | 3.0751 |
| 2,000 | 1121.1 |  | 1051.9 | 2.6971 |  | 2.8089 |
| 3,000 | 1126.4 |  | 1148.3 | 2.6415 |  | 2.6253 |
| 4,000 | 1174.3 |  | 1229.1 | 2.5640 |  | 2.5440 |
| 5,000 | 1177.5 |  | 1268.9 | 2.5542 |  | 2.4872 |

Skinny's last checkpoint, step 1,605 (written by the stop's abort save): pElo 867.8,
NLL 3.2008.

Gap to the baseline:

| step | fatty pElo | fatty NLL | skinny pElo | skinny NLL |
|---:|---:|---:|---:|---:|
| 1,000 | +68.6 | −0.2015 | −256.6 | +0.5276 |
| 2,000 | +69.2 | −0.1118 | | |
| 3,000 | −21.9 | +0.0162 | | |
| 4,000 | −54.8 | +0.0200 | | |
| 5,000 | −91.4 | +0.0670 | | |

(NLL: lower is better, so a negative gap is ahead of the baseline.)

Training-loss lines at the same steps (`[REPLAY]`):

| step | run | loss | pLoss | vLoss | pIllM |
|---:|---|---:|---:|---:|---:|
| 1,000 | fatty | 3.9207 | 3.0320 | 0.8347 | 0.0540 |
| 1,000 | skinny | 4.1759 | 3.2336 | 0.8243 | 0.1180 |
| 1,600 | fatty | 3.8032 | 2.9492 | 0.8202 | 0.0338 |
| 1,600 | skinny | 3.9029 | 3.0060 | 0.8320 | 0.0649 |

## Findings

1. **Skinny learns slowest per step.** At 1,000 steps it is 256.6 pElo behind the
   baseline; its step-1,605 probe (867.8) is still below the baseline's 1,000-step probe
   (925.5). Its policy loss trails fatty's at every logged step and it puts about twice
   fatty's probability mass on illegal moves (pIllM). Its gradient norm fell smoothly
   from 3.4 at step 250 to 1.0 at step 1,600 — slow, not unstable.
2. **Skinny is also the most expensive per step.** Both runs shared the GPU, so these are
   relative, not solo, times:

   | period | fatty s/step | skinny s/step | skinny ÷ fatty |
   |---|---:|---:|---:|
   | 3 trainers (with R11), memory overcommitted | 1.94 | 5.95 | 3.1× |
   | 2 trainers, after R11 stopped 22:54:58 | 1.13 | 4.67 | 4.1× |

   For reference, R7 and R8 ran at a median 2.17 / 2.19 s/step, three trainers sharing
   the GPU — a different mix, so not directly comparable. At equal wall time the gap is
   larger still: skinny reached step 1,600 at 23:17:39, 2 h 33 min after launch; fatty
   reached step 1,600 at 21:36:29 (52 min) and step 5,000 at 23:13:34.
3. **Fatty leads early, then stalls.** It is 68.6–69.2 pElo ahead at 1k and 2k, behind
   from 3k on, and the gap grows to −91.4 at 5k: from 4k to 5k fatty gained +3.2 pElo
   against the baseline's +39.8. One seed and 5k of 33k steps — provisional, but the trend
   is consistent across three probes.
4. **Memory.** Activation storage scales with channels × layers, not parameters. Measured
   with `footprint` at 22:51 with three trainers running:

   | run | process footprint | GPU memory | of which compressed / swapped |
   |---|---:|---:|---:|
   | R11 (3 × @128 + SE, ReZero) | 21.06 GB | 14.94 GB | 5.83 GB |
   | fatty | 18.50 GB | 12.54 GB | 4.42 GB |
   | skinny | 31.25 GB | 25.58 GB | 8.72 GB |

   The three totalled 70.8 GB on a 64 GB machine, with the memory compressor working
   ~0.87 GB/s each way. After R11 stopped, compressor activity fell to near zero and
   memory was 64–69% free. A two-point fit (fatty, skinny) puts per-step interim storage
   at ≈ 8.8 bytes per activation element plus a fixed ≈ 6.2 GB of GPU memory: ≈ 6.4 GB per
   step for fatty, 8.5 GB for R7/R8, 19.4 GB for skinny. Checked against R11 the fit
   over-predicts by 13%, so treat it as approximate.
5. **Why skinny is slow (inference, not yet measured directly).** Memory paging cost
   all runs while it lasted, but it is not the main cause: with paging gone skinny is
   still 4.1× slower. The elementwise passes (norms, ReLUs, adds) and kernel launches
   are each estimated at ≤ 0.1–0.35 s per step. The likeliest cause is convolution
   efficiency: a conv layer's FLOPs grow with C² but its data movement with C, so at
   equal FLOPs the data moved per FLOP scales as 1/width. A "time ∝ FLOPs ÷ width" model
   predicts skinny at 216/48 = 4.5× fatty; observed 4.1×. It predicts R7/R8's shape at
   ≈ 1.7× fatty. A solo timing run, or a GPU trace of a running trainer, would confirm it.

## Conclusion so far

- **Many narrow blocks lose on every axis at this budget:** slower to learn per step,
  ~4× the cost per step, and ~2.3× the activation memory of R7/R8. Skinny was stopped at
  step 1,605 to give its slot to the next variant.
- **One wide block learns fastest at first but stalls by 3k–5k.** The three-block
  baseline has passed it and is pulling away. So far the middle of the sweep — R7/R8's
  3 blocks — is the best use of the budget.
- Open: whether fatty's stall comes from too little depth (2 convs after the stem) or
  from where its parameters sit. The follow-up, slim-neck fatty (R15), moves 259,200
  parameters from fatty's 7×7 stem into tower width (3×3 stem, 224 channels).
- A middle depth point (e.g. 6 blocks @ ~90) would separate depth from width more
  cleanly than either end.

## Reproduce

```
BIN="$HOME/Library/Application Support/DrewsChessMachine/FrozenBuilds/DCM-2275-de0f22b.app/Contents/MacOS/DrewsChessMachine"
M="$HOME/Library/Application Support/DrewsChessMachine/Models"
# start nets
"$BIN" --new-model --architecture experiments/20261003-fatty-1x7x7-216/test_1_fatty_216-v5.json \
  --out-model "$M/20261003-fatty216-b2275-fresh.safetensors"
"$BIN" --new-model --architecture experiments/20261003-skinny-22x7x7-48/test_22_skinny_48-v5.json \
  --out-model "$M/20261003-skinny48-b2275-fresh.safetensors"
# training (one per arm; stem = 20261003-fatty216-b2275 or 20261003-skinny48-b2275)
"$BIN" --replay-corpus 20260624-192615-w3aA5b --start-model "$M/<stem>-fresh.safetensors" \
  --out-model "$M/<stem>-replay-latest.safetensors" --parameters <experiment>/parameters.json \
  --epochs 12 --training-step-limit 33000 --enumerate-checkpoints --policy-tail-precision fp32_from_pre_bn
# probes
PROBE_BIN="$BIN" TRAINER_PID=<trainer pid> experiments/probe_loop.sh <stem> <experiment>/probes.jsonl
experiments/probe_step0.sh
# tables
python3 experiments/20261003-skinny-22x7x7-48/table.py
python3 experiments/arch_flops.py
python3 experiments/rchart.py
```

Build 2275 has no `--init-seed`, so the start weights are not reproducible from a seed (as
for R7/R8); the start nets' ModelIDs are listed above. Skinny was stopped with SIGINT at
23:17:58 CDT.

## Summary: R1–R15

pElo at selected steps (`python3 experiments/rchart.py`, as of 2026-10-03 23:30 CDT; best
per column in bold). R13–R15 have not reached these steps yet, or never will (R14).

| Max step | Run | 33k | 32k | 31k | 30k | 21k | 7k |
|---:|---|---:|---:|---:|---:|---:|---:|
| 33,000 | **R1** SE scale+bias, seed 1 (label-smoothing baseline, ε 0.1/0.013) | 1461.5 | 1469.8 | 1438.4 | 1446.7 | 1308.8 | 1268.1 |
| 7,000 | **R2** SE scale+bias, seed 2 (baseline seed 2) |  |  |  |  |  | 1261.4 |
| 33,000 | **R3** SE attenuate-only | 1484.1 | 1474.4 | 1463.6 | 1478.0 | 1316.1 | 1265.0 |
| 33,000 | **R4** SE scale+bias, leaky ReLU in SE FC1 | 1492.8 | 1503.1 | 1495.9 | 1474.9 | 1308.8 | 1290.8 |
| 32,000 | **R5** no SE + ReZero, seed 1 |  | 1477.5 | 1469.8 | 1483.6 | 1313.0 | 1297.5 |
| 7,000 | **R6** no SE + ReZero, seed 2 |  |  |  |  |  | 1299.6 |
| 33,000 | **R7** no SE, no ReZero, seed 1 | 1493.4 | 1504.1 | 1471.8 | 1492.8 | **1358.3** | 1290.3 |
| 33,000 | **R8** no SE, no ReZero, seed 2 | **1515.9** | **1517.0** | **1500.0** | **1511.8** | 1300.6 | **1323.8** |
| 33,000 | **R9** zero-init ReZero (no SE) — *same start weights as R5 (only ReZero α init/cap changed)* | 1483.1 | 1467.7 | 1466.2 | 1471.8 | 1304.7 | 1250.0 |
| 33,000 | **R10** C: SE scale+bias, policy ε 0.03, seed 1 — *same start weights as R1* | 1480.0 | 1491.3 | 1459.0 | 1461.5 | 1345.4 | 1280.5 |
| 31,000 | **R11** C: SE scale+bias, policy ε 0.03, seed 2 — *same start weights as R2; stopped by the owner at 31,906* |  |  | 1471.3 | 1456.4 | 1302.1 | 1294.9 |
| 33,000 | **R12** D: SE scale+bias, value ε 0 — *same start weights as R1* | 1466.7 | 1479.0 | 1460.5 | 1469.2 | 1340.3 | 1252.6 |
| 5,000 (running) | **R13** fatty: no SE, no ReZero, 1 block × 7×7 @216 — *same budget as R7/R8* |  |  |  |  |  |  |
| 1,605 | **R14** skinny: no SE, no ReZero, 22 blocks × 7×7 @48 — *same budget as R7/R8; stopped by the owner at 1,605* |  |  |  |  |  |  |
| running, no probe yet | **R15** slim-neck fatty: fatty with a 3×3 stem, 1 block × 7×7 @224 — *same budget as R7/R8* |  |  |  |  |  |  |

R11's last probe is step 31,906 (pElo 1477.5); the chart's max-step column counts whole
thousands only.
