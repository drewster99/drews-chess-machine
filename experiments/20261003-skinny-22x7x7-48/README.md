# 2026-10-03 — Deep and narrow: 22 × [7×7 + 7×7] @ 48, no SE, no ReZero ("skinny")

**Status:** queued — launches together with the fatty run (`../20261003-fatty-1x7x7-216/`)
the moment label smoothing D ends (~20:45 CDT), so the two get matched wall time.

## Question

The deep end of a depth sweep at the no-SE / no-ReZero budget (R7/R8: 3 blocks × two 7×7
convs @ 128): the same parameters and compute spent on 22 blocks @ 48 channels — 44 conv
layers instead of 6, but only 48 features per square between blocks. Each 7×7 conv
already sees the whole 8×8 board, so the extra depth buys nonlinear steps, not context;
the narrow width may be the bottleneck. The fatty run (1 block @ 216) is the shallow end.

## Design

- **Shape:** basic30 → stem 48 (7×7) → 22 × [7×7 + 7×7 @ 48, no SE, ReLU pre-act,
  clean_add, no ReZero, LayerNorm out] → policy intermediate_conv (K = 128) · value WDL
  (16 → FC128) · bf16. 5,197,807 parameters (logged count), +0.5% vs R7/R8's 5,170,319;
  ≈ 323M MACs per position vs ≈ 322M. Standard init: policy and value final layers He,
  draw prior 0.75.
- **Everything but the tower identical to R7/R8:** the same frozen build 2275, the same
  `parameters.json` (byte-identical copy of `20261002-noSE-noReZero/parameters.json`),
  corpus `20260624-192615-w3aA5b`, 12 epochs, step limit 33,000, `--enumerate-checkpoints`,
  `--policy-tail-precision fp32_from_pre_bn`.
- **Start net:** `20261003-skinny48-b2275-fresh.safetensors`, ModelID `20261004-9-czp5`,
  minted on build 2275 from `test_22_skinny_48-v5.json` (the saved preset
  `test_22_skinny_48.json`, also copied here, without the format-8 init fields, which equal
  2275's built-in standard init). No `--init-seed` on 2275, as for R7/R8.
- **Probes:** pElo / NLL every 1,000 steps (`--probe-set wide`) with build 2275.

## What to watch

- Steps per hour against R7/R8 and fatty: 44 small convs plus their norms use the GPU less
  efficiently than 6 large ones, so equal MACs need not mean equal step time.
- Early gNorm and `[LAYER-HEALTH]`: 22 plain residual blocks with no ReZero; each block's
  output LayerNorm keeps the residual stream bounded.

## What would count as an answer

- One seed; a gap under ~25 pElo over the last 5k steps is a tie. A clear loss against both
  R7/R8 and fatty would not by itself separate depth from narrowness — a middle point (e.g.
  6 blocks @ ~90) would.

## Launch record

- **Launcher:** waits for label smoothing D's trainer to end, then starts this run and the
  fatty run at the same moment, each with its probe loop. Launch times and session logs are
  added here after launch.
- **Commands**

```
BIN="$HOME/Library/Application Support/DrewsChessMachine/FrozenBuilds/DCM-2275-de0f22b.app/Contents/MacOS/DrewsChessMachine"
M="$HOME/Library/Application Support/DrewsChessMachine/Models"
"$BIN" --new-model --architecture experiments/20261003-skinny-22x7x7-48/test_22_skinny_48-v5.json \
  --out-model "$M/20261003-skinny48-b2275-fresh.safetensors"
"$BIN" --replay-corpus 20260624-192615-w3aA5b --start-model "$M/20261003-skinny48-b2275-fresh.safetensors" \
  --out-model "$M/20261003-skinny48-b2275-replay-latest.safetensors" \
  --parameters experiments/20261003-skinny-22x7x7-48/parameters.json \
  --epochs 12 --training-step-limit 33000 --enumerate-checkpoints --policy-tail-precision fp32_from_pre_bn
PROBE_BIN="$BIN" TRAINER_PID=<trainer pid> experiments/probe_loop.sh 20261003-skinny48-b2275 \
  experiments/20261003-skinny-22x7x7-48/probes.jsonl
```
