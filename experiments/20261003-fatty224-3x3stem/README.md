# 2026-10-03 — Slim-neck fatty: fatty with a 3×3 stem, 1 × [7×7 + 7×7] @ 224, no SE, no ReZero (R15)

**Status:** ended at its 33,000-step limit (2026-10-04 18:13:07 CDT). Final probe: pElo 1348.0, NLL 2.3630. Launched 2026-10-03 23:18:25 CDT in the slot skinny's stop freed.

## Question

Fatty (`20261003-fatty-1x7x7-216/`, R13) spends 317,520 parameters — 6.3% of its budget —
on a 7×7 stem from the 30 input planes. Shrink the stem (the "neck") to 3×3 and put the
saved parameters back into the tower's width: 216 → 224 channels. Does a wider single
block with a local stem do better than fatty's whole-board stem?

## Design

- **Shape:** basic30 → stem 224 (**3×3**) → 1 × [7×7 + 7×7 @ 224, no SE, ReLU pre-act,
  clean_add, no ReZero, LayerNorm out] → policy intermediate_conv (K = 128) · value WDL
  (16 → FC128) · bf16. 5,155,983 parameters (logged count, which includes batch-norm
  running statistics; 5,153,903 trainable), −0.3% vs R7/R8's 5,170,319 logged; 321.4M
  MACs per position vs 322.3M (`experiments/arch_flops.py`). Standard init: policy and
  value final layers He, draw prior 0.75.
- **Differs from fatty only in** stem kernel 3 (was 7) and channels 224 (was 216); the
  presets differ in exactly those two fields and the label.
- **Everything but the network identical to R7/R8 and fatty:** frozen build 2275, the same
  `parameters.json` (byte-identical copy of `20261002-noSE-noReZero/parameters.json`),
  corpus `20260624-192615-w3aA5b`, 12 epochs, step limit 33,000, `--enumerate-checkpoints`,
  `--policy-tail-precision fp32_from_pre_bn`. Its `[REPLAY-HPARAMS]` and batch lines are
  identical to fatty's; its `[REPLAY-CYCLE]` line differs only in the start net's model ID.
- **Start net:** `20261003-fatty224s3-b2275-fresh.safetensors`, ModelID `20261004-12-QsqZ`,
  minted on build 2275 from `test_1_fatty_224_3x3stem-v5.json` (`test_1_fatty_224_3x3stem.json`
  without the format-8 init fields, which equal 2275's built-in standard init; the full
  preset is also in the app's `Presets/`). No `--init-seed` on 2275. Step-0 probe: pElo
  480.1, NLL 3.7283 (`step0-probes.jsonl`).
- **Probes:** pElo / NLL every 1,000 steps (`--probe-set wide`) with build 2275.
- **Table:** `python3 experiments/20261003-fatty224-3x3stem/table.py` — fatty, slim-neck
  fatty and Avg(R7,R8), pElo then NLL.

## What would count as an answer

- One seed; the two comparator seeds differ by ~22 pElo, so a gap under ~25 pElo over the
  last 5k steps is a tie.

## Launch record

- **Launched** 2026-10-03 23:18:25 CDT (pid 7076), session log
  `dcm_log_20261003-231825.txt`, build 2275, beside fatty.
- **Commands**

```
BIN="$HOME/Library/Application Support/DrewsChessMachine/FrozenBuilds/DCM-2275-de0f22b.app/Contents/MacOS/DrewsChessMachine"
M="$HOME/Library/Application Support/DrewsChessMachine/Models"
"$BIN" --new-model --architecture experiments/20261003-fatty224-3x3stem/test_1_fatty_224_3x3stem-v5.json \
  --out-model "$M/20261003-fatty224s3-b2275-fresh.safetensors"
"$BIN" --replay-corpus 20260624-192615-w3aA5b --start-model "$M/20261003-fatty224s3-b2275-fresh.safetensors" \
  --out-model "$M/20261003-fatty224s3-b2275-replay-latest.safetensors" \
  --parameters experiments/20261003-fatty224-3x3stem/parameters.json \
  --epochs 12 --training-step-limit 33000 --enumerate-checkpoints --policy-tail-precision fp32_from_pre_bn
PROBE_BIN="$BIN" TRAINER_PID=<trainer pid> experiments/probe_loop.sh 20261003-fatty224s3-b2275 \
  experiments/20261003-fatty224-3x3stem/probes.jsonl
```
