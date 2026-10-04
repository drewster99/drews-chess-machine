# 2026-10-03 — One wide block: 1 × [7×7 + 7×7] @ 216, no SE, no ReZero ("fatty")

**Status:** running since 2026-10-03 20:44:59 CDT, launched together with the skinny run when label smoothing D ended.

## Question

The no-SE / no-ReZero tower (3 blocks × two 7×7 convs @ 128; `20261002-noSE-noReZero/`,
seeds 1 and 2, R7 and R8 in the summary chart) is the strongest shape so far. Spend the
same budget on one wide block (two convs @ 216) instead of three narrow ones (six convs
@ 128): same parameters and compute, one third of the depth. With the skinny run (22
blocks @ 48) this makes a depth sweep at constant budget: 1 / 3 / 22 blocks.

## Design

- **Shape:** basic30 → stem 216 (7×7) → 1 × [7×7 + 7×7 @ 216, no SE, ReLU pre-act,
  clean_add, no ReZero, LayerNorm out] → policy intermediate_conv (K = 128) · value WDL
  (16 → FC128) · bf16. 5,066,767 parameters (logged count), −2.0% vs R7/R8's 5,170,319;
  ≈ 316M MACs per position vs ≈ 322M. Standard init: policy and value final layers He,
  draw prior 0.75.
- **Everything but the tower identical to R7/R8:** the same frozen build 2275 (`de0f22b`
  code, stamped `f6fdd88`), the same `parameters.json` (byte-identical copy of
  `20261002-noSE-noReZero/parameters.json`), corpus `20260624-192615-w3aA5b`, 12 epochs,
  step limit 33,000, `--enumerate-checkpoints`, `--policy-tail-precision fp32_from_pre_bn`.
  Build 2275 samples batches uniformly in corpus replay, as it did for R7/R8.
- **Start net:** `20261003-fatty216-b2275-fresh.safetensors`, ModelID `20261004-8-2Sao`,
  minted on build 2275 from `test_1_fatty_216-v5.json` (the saved preset without the
  format-8 init fields, which 2275 predates and which equal its built-in standard init).
  Build 2275 has no `--init-seed`, so the start weights are not reproducible from a seed —
  the same as R7/R8's.
- **Probes:** pElo / NLL every 1,000 steps (`--probe-set wide`) with build 2275.

## What would count as an answer

- One seed; the two comparator seeds differ by ~22 pElo, so a gap under ~25 pElo over the
  last 5k steps is a tie.

## History

- A first launch (2026-10-03 20:13, build 2320, `test_1_fatty_216.json` minted with
  `--init-seed 4683348161003864489`, sampler limits loosened to approximate build 2275's
  uniform draw) was stopped by the owner at step 42 so that every training setting could
  match R7/R8 exactly; its checkpoints and probe output were deleted. Its start net
  `20261003-fatty216-fresh.safetensors` (ModelID `20261004-4-XS2w`) is kept; `mint.txt`
  records that mint.

- **Suspended** by the owner on 2026-10-03 at 23:46:03 CDT (SIGSTOP, pid 72276) at step
  ~6,350, until slim-neck fatty (`20261003-fatty224-3x3stem/`) posts its 6k probe; a
  watcher then resumes it with SIGCONT. The suspension shows as one long gap between two
  `[REPLAY]` lines: wall time across it is not training time.
  Resumed 2026-10-04 00:55:28 CDT (SIGCONT, by the watcher) after slim-neck fatty's 6k probe.

## Launch record

- **Launched** 2026-10-03 20:44:59 CDT (pid 72276), session log `dcm_log_20261003-204412.txt`, build 2275, the
  moment label smoothing D's trainer ended; the other sweep run started in the same second.
  Its `[REPLAY-HPARAMS]` line and batch line are identical to R7's; its `[REPLAY-CYCLE]` line
  differs only in the start net's model ID.
- **Commands**

```
BIN="$HOME/Library/Application Support/DrewsChessMachine/FrozenBuilds/DCM-2275-de0f22b.app/Contents/MacOS/DrewsChessMachine"
M="$HOME/Library/Application Support/DrewsChessMachine/Models"
"$BIN" --new-model --architecture experiments/20261003-fatty-1x7x7-216/test_1_fatty_216-v5.json \
  --out-model "$M/20261003-fatty216-b2275-fresh.safetensors"
"$BIN" --replay-corpus 20260624-192615-w3aA5b --start-model "$M/20261003-fatty216-b2275-fresh.safetensors" \
  --out-model "$M/20261003-fatty216-b2275-replay-latest.safetensors" \
  --parameters experiments/20261003-fatty-1x7x7-216/parameters.json \
  --epochs 12 --training-step-limit 33000 --enumerate-checkpoints --policy-tail-precision fp32_from_pre_bn
PROBE_BIN="$BIN" TRAINER_PID=<trainer pid> experiments/probe_loop.sh 20261003-fatty216-b2275 \
  experiments/20261003-fatty-1x7x7-216/probes.jsonl
```
