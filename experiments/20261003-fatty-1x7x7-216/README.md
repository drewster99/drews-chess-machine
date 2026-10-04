# 2026-10-03 — One wide block: 1 × [7×7 + 7×7] @ 216, no SE, no ReZero ("fatty")

**Status:** running since 2026-10-03 20:13 CDT.

## Question

The no-SE / no-ReZero tower (3 blocks × two 7×7 convs @ 128; `20261002-noSE-noReZero/`,
seeds 1 and 2) is the strongest shape so far. Does the same budget do as well spent on
one wide block (two convs @ 216) instead of three narrow ones (six convs @ 128)? Same
parameter count and compute, one third of the depth.

## Design

- **Shape:** basic30 → stem 216 (7×7) → 1 × [7×7 + 7×7 @ 216, no SE, ReLU pre-act,
  clean_add, no ReZero, LayerNorm out] → policy intermediate_conv (K = 128) · value WDL
  (16 → FC128) · bf16. 5,066,767 parameters (logged count), −2.0% vs the comparators'
  5,170,319; ≈ 318M MACs per position vs ≈ 322M. Preset `test_1_fatty_216.json` (copied
  here). Init options standard: policy and value final layers He, draw prior 0.75.
- **Comparators (not re-run):** no SE, no ReZero seeds 1 and 2 (R7, R8 in the summary
  chart), same everything but the tower shape.
- **Start net:** `20261003-fatty216-fresh.safetensors`, ModelID `20261004-4-XS2w`, minted on
  build 2320 with `--init-seed 4683348161003864489` (reproducible).
- **Corpus / numerics / schedule:** corpus `20260624-192615-w3aA5b`, 12 epochs, step limit
  33,000, `--enumerate-checkpoints`, `--policy-tail-precision fp32_from_pre_bn`.
- **Parameters:** `parameters.json` here is the comparators' file with two keys changed so
  that build 2320 samples as their build did. Build 2275 (the comparators') ignored the
  batch-composition parameters in corpus replay and drew uniformly; 2320 applies them.
  `max_plies_from_any_one_game` 10 → 400 (its declared maximum; cannot bind at batch
  4096) and `target_sampled_game_length_plies` 999 → 0 (no length tilt), the nearest the
  declarations allow to the uniform draw. The run logs
  `sampling=(maxPerGame=400 maxDrawPct=100 targetLen=0 stratify=off applied=on)`.
- **Other build differences vs 2275:** corpus replay now feeds games played past an
  unclaimed threefold instead of dropping them (0.045% of this corpus's games); probes run
  with build 2320.

## What would count as an answer

- One seed; the two comparator seeds differ by ~22 pElo, so a gap smaller than ~25 pElo
  over the last 5k steps is a tie.
- Faster or slower per step is reported separately (one wide block runs fewer, larger
  convs).

## Launch record

- **Launched** 2026-10-03 20:13:27 CDT, session log `dcm_log_20261003-201327.txt`, build 2320
  (`FrozenBuilds/DCM-2320-1ab52554.app`), run seed drawn and logged on the `[RUN]` line.
- **Commands**

```
BIN="$HOME/Library/Application Support/DrewsChessMachine/FrozenBuilds/DCM-2320-1ab52554.app/Contents/MacOS/DrewsChessMachine"
M="$HOME/Library/Application Support/DrewsChessMachine/Models"
"$BIN" --new-model --architecture test_1_fatty_216 --init-seed 4683348161003864489 \
  --out-model "$M/20261003-fatty216-fresh.safetensors"
"$BIN" --replay-corpus 20260624-192615-w3aA5b --start-model "$M/20261003-fatty216-fresh.safetensors" \
  --out-model "$M/20261003-fatty216-replay-latest.safetensors" \
  --parameters experiments/20261003-fatty-1x7x7-216/parameters.json \
  --epochs 12 --training-step-limit 33000 --enumerate-checkpoints --policy-tail-precision fp32_from_pre_bn
PROBE_BIN="$BIN" TRAINER_PID=<trainer pid> experiments/probe_loop.sh 20261003-fatty216 \
  experiments/20261003-fatty-1x7x7-216/probes.jsonl
```
