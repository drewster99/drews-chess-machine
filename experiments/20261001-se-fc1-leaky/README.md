# 2026-10-01 — Leaky ReLU in the SE bottleneck (FC1 only) vs ReLU

**Status:** running (launched 2026-10-01 15:18 CDT). Results are added at the 5k and 10k reviews.

## Question

Does leaky ReLU in the squeeze-and-excitation bottleneck (FC1) revive the dead SE
units measured in the SE style experiment, and does it change training quality?

The SE style experiment (`experiments/20260929-se-style-ab/`, TENSOR-STATS.md
finding 4) found 2–13 of each block's 32 FC1 units dead (zero optimizer velocity),
up to 41% of a block, and nothing dead anywhere else in the network. FC1 is the
only ReLU in the net that sees pooled `[batch, C/r]` vectors, so it is the one
place a dead unit cannot recover.

Leaky ReLU everywhere costs roughly 4–7% training throughput (2026-10-01
benchmark, `leaky_abba.sh`). Leaky ReLU in FC1 alone runs on 4096 × 32 values per
block and should cost nothing measurable.

## Design

- **Only variable:** the SE FC1 activation — `relu` (baseline) vs `leaky_relu`
  (negative slope 0.01, `ActivationFunction.leakyReLUNegativeSlope`). Everything
  else in the tower and heads stays ReLU.
- **Starting net:** the SE experiment's scale+bias seed-1 fresh net
  (`20260929-test_SE_scale+bias-fresh.safetensors`, ModelID `20260929-12-JZOe`),
  copied bit-exact with only `block_groups[0].se_activation` set to `leaky_relu`
  via `--derive-model --set-se-activation leaky_relu`. Same weights, same init.
- **Baseline:** the existing ReLU scale+bias seed-1 arm (`se_sb`, 33,014 steps),
  trained from the same fresh net. It is not re-run.
- **Architecture:** v5-style — basic30 input, 7×7 stem → 3×[7×7+7×7 @128, SE+/4,
  ReLU pre-act, ReZero (α init 0.447), clean_add, LayerNorm out] · policy
  intermediate_conv (128) · value WDL (16ch → FC128) · bf16 · 5,208,050 params.
- **Corpus / parameters:** corpus `20260624-192615-w3aA5b`, the SE experiment's
  `parameters.json` (all 78 keys pinned; copied here at launch).
- **Numerics:** `--policy-tail-precision fp32_from_pre_bn`, matching the baseline's
  build (the default became `mixed_final_projection` on 2026-10-01). Without it the
  policy tail precision would be a second difference.
- **Length:** to 33,000 steps (`--training-step-limit 33000`), matching the
  baseline. Reviewed at 5k and 10k; may stop early.
- **Concurrency:** this run trains alone; the baseline shared the GPU three ways.
  Compare on **step** (and `games_fed`), never on time. Step time is reported for
  its own interest, not as a comparison.

## Measurements at each review point

- pElo and NLL (enumerated-checkpoint probes, same probe set as the SE experiment).
- Training loss, policy loss, value loss, policy entropy from the `[REPLAY]` lines.
- Dead FC1 units per block: units whose FC1 weight row and bias have zero
  optimizer velocity (this build saves velocity in enumerated checkpoints), and
  units whose weights never moved from init (decay-only), as in TENSOR-STATS.md.
- Step time.

## Remaining differences from the baseline

Build: the baseline ran build 2255 (`cbc1894`/`7a434ea` code). This run uses the
build of the commit recorded in the launch record. Engine changes in between that
touch training are listed there; the intent is that the training step math for a
fresh start is identical apart from the FC1 activation.

## Reproduce

```
BIN=<DrewsChessMachine binary of the launch-record commit>
M="$HOME/Library/Application Support/DrewsChessMachine/Models"
"$BIN" --derive-model --from "$M/20260929-test_SE_scale+bias-fresh.safetensors" \
  --set-se-activation leaky_relu \
  --out "$M/20261001-test_SE_scale+bias-fc1leaky-fresh.safetensors"
"$BIN" --replay-corpus 20260624-192615-w3aA5b \
  --start-model "$M/20261001-test_SE_scale+bias-fc1leaky-fresh.safetensors" \
  --out-model "$M/20261001-test_SE_scale+bias-fc1leaky-replay-latest.safetensors" \
  --parameters experiments/20261001-se-fc1-leaky/parameters.json \
  --epochs 12 --training-step-limit 33000 --enumerate-checkpoints \
  --policy-tail-precision fp32_from_pre_bn
```

## Launch record

| field | value |
|---|---|
| launched | 2026-10-01 15:18:22 CDT |
| build | commit `de0f22b` (Release, frozen copy) |
| fresh net | `20261001-test_SE_scale+bias-fc1leaky-fresh.safetensors`, ModelID `20261001-42-2q0Q`, derived from `20260929-12-JZOe` (source sha256 `2c4b779b…df693c89`) |
| out model | `20261001-test_SE_scale+bias-fc1leaky-replay-latest.safetensors` (+ enumerated `…-fc1leaky-replay-step<N>`) |
| log | `~/Library/Logs/DrewsChessMachine/dcm_log_20261001-151822.txt` |
| baseline | ReLU scale+bias seed 1 (`se_sb`), log `dcm_log_20260929-150727.txt`, 33,014 steps |

Startup lines match the baseline exactly: `[REPLAY-HPARAMS]` (lr 0.001, batch 4096,
wd 3e-4, grad clip 15, warmup 1000, buffer 500k / prefill 250k, replay ratio
0.48, KL probe 100) and `[REPLAY-CYCLE]` (LR 1e-1 peak → 1e-3 trough over 20k,
decay horizon 1M, momentum following the cycle). `[REPLAY] trainer policy tail
precision: fp32_from_pre_bn`.

## Progress tracking

- `probe_loop.sh` probes every enumerated checkpoint (every 1,000 steps) with
  `--probe-set wide` into `probes.jsonl`; `table.py` renders the per-1000 table
  against the SE experiment's seed-1 arms (leaky FC1 next to ReLU scale+bias);
  `review.py <step> <binary>` is the 5k-interval analysis (both arms probed with
  the same binary, loss windows, FC1 units unmoved from init).
- **Step-time caveat:** the full test suite ran on the same machine 16:42–17:12
  CDT (about steps 6,000–7,800); step times in that window are slowed by it and
  are excluded from speed figures. Training itself is unaffected.
