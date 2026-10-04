# 2026-10-04 — Fatconv: 15×15 stem and 1 × [15×15 + 15×15] @ 98, no SE, no ReZero (R16)

**Status:** running since 2026-10-04 01:58:04 CDT, beside fatty and slim-neck fatty.

## Question

On an 8×8 board a 7×7 kernel reaches only part of the board from an edge square; a 15×15
kernel reaches every square from every square. Fatty (R13) and slim-neck fatty (R15) put
the R7/R8 budget into one wide block of 7×7 convs. Fatconv spends the same budget on one
block of whole-board 15×15 convs (and a 15×15 stem), which at this budget forces the width
down to 98. Is a whole-board kernel at narrow width a better use of one block than a
partial-board kernel at wide width?

## Design

- **Shape:** basic30 → stem 98 (**15×15**) → 1 × [**15×15 + 15×15** @ **98**, no SE, ReLU
  pre-act, clean_add, no ReZero, LayerNorm out] → policy intermediate_conv (K = 128) ·
  value WDL (16 → FC128) · bf16. 5,141,143 parameters (logged count, including batch-norm
  running statistics; 5,140,071 trainable), −0.6% vs R7/R8's 5,170,319 logged; 320.6M MACs
  per position vs 322.3M (`experiments/arch_flops.py`). Width 98 is the closest to R7/R8's
  parameter count (97: −2.4%, 99: +1.3%). Standard init: policy and value final layers He,
  draw prior 0.75.
- **Padding:** with a 15×15 kernel on 8×8 squares only 64 of the 225 taps of any output
  land on the board, so 72% of the nominal MACs multiply padding (38% for the 7×7 nets);
  useful MACs per position are 92.4M vs R7/R8's 199.7M. Activation elements per position
  are 99k, 0.39× R7/R8's.
- **Input encoding:** basic30, not basic24. Build 2275 predates basic24, and every
  comparison run (R7/R8, fatty, slim-neck fatty) trained on 2275 with basic30.
- **Differs from fatty only in** the three kernel sizes (stem, conv1, conv2: 7 → 15) and
  channels (216 → 98); the presets differ in exactly those four fields and the label.
- **Everything else identical to R7/R8, fatty and slim-neck fatty:** frozen build 2275,
  the same `parameters.json` (byte-identical copy of `20261002-noSE-noReZero/parameters.json`),
  corpus `20260624-192615-w3aA5b`, 12 epochs, step limit 33,000, `--enumerate-checkpoints`,
  `--policy-tail-precision fp32_from_pre_bn`. Its `[REPLAY-HPARAMS]`, `[REPLAY-CYCLE]` and
  batch lines are identical to fatty's apart from the start net's model ID.
- **Start net:** `20261004-fatconv98-b2275-fresh.safetensors`, ModelID `20261004-14-3Rkc`,
  minted on build 2275 from `test_1_15x15_98-v5.json` (`test_1_15x15_98.json` without the
  format-8 init fields, which equal 2275's built-in standard init; the full preset is also
  in the app's `Presets/`). It was minted as `20261004-15x15-98-b2275-fresh.safetensors`
  (the name in `mint-b2275.txt`) and renamed before training when the run was named. No
  `--init-seed` on 2275. Step-0 probe: pElo 449.0, NLL 3.8204 (`step0-probes.jsonl`).
- **Probes:** pElo / NLL every 1,000 steps (`--probe-set wide`) with build 2275.
- **Table:** `python3 experiments/20261004-fatconv-1x15x15-98/table.py` — fatty, slim-neck
  fatty, fatconv and Avg(R7,R8), pElo then NLL.

## What to watch

- Step time: at width 98 each conv input value feeds only 98 outputs per load, so the GPU
  is likely to run it less efficiently per FLOP than fatty's 216 (see
  `20261003-fatty-vs-skinny/README.md`, finding 5); the 15×15 kernels may also take a
  slower convolution path.

## What would count as an answer

- One seed; the two comparator seeds differ by ~22 pElo, so a gap under ~25 pElo over the
  last 5k steps is a tie.

## Continuation past 33,000 (queued)

The owner asked (2026-10-04) for fatconv not to stop at its 33,000-step limit. The running
build-2275 process cannot change its limit, so a watcher (session scratch
`fatconv_continue.sh`, the recipe prepared and dry-run-verified for label smoothing C seed 2)
waits for it to save step 33,000 and end, then resumes `20261004-fatconv98-b2275-replay-step33000`
with `--resume-exact` on build 2320 (`DCM-2320-1ab52554`) and no step limit (`--epochs 12`).

- **Carried over exactly:** fp32 master weights, momentum velocity, the trainer step clock,
  the LR/momentum cycle and its decay envelope, and the corpus position.
- **Not exact** (build 2275 wrote no lineage record): `rng_sampler`, `dropout_state`,
  `feed_carry`, `params`, `lineage`, `policy_tail`, accepted with `--accept-inexact`.
  Dropout is 0, so its stream has no effect; `--policy-tail-precision fp32_from_pre_bn` is
  passed as before.
- **Sampling:** build 2275's corpus replay ignored the batch-composition parameters and drew
  uniformly; build 2320 applies them. `parameters-continue.json` is `parameters.json` with
  `max_plies_from_any_one_game` 10 → 400 and `target_sampled_game_length_plies` 999 → 0, the
  nearest the declarations allow to the uniform draw. Build 2320 also carries the
  adjudication fix (0.045% of games).
- **Names:** the resumed run numbers its own steps, so it writes
  `20261004-fatconv98-cont-replay-step<N>` (real step 33,000 + N), probed with build 2320
  into `probes-cont.jsonl`.
- **No comparator past 33,000:** R7/R8, fatty and slim-neck fatty all ended at 33,000.

## Launch record

- **Launched** 2026-10-04 01:58:04 CDT (pid 36047), session log
  `dcm_log_20261004-015804.txt`, build 2275, the third trainer beside fatty and slim-neck
  fatty.
- **Commands**

```
BIN="$HOME/Library/Application Support/DrewsChessMachine/FrozenBuilds/DCM-2275-de0f22b.app/Contents/MacOS/DrewsChessMachine"
M="$HOME/Library/Application Support/DrewsChessMachine/Models"
"$BIN" --new-model --architecture experiments/20261004-fatconv-1x15x15-98/test_1_15x15_98-v5.json \
  --out-model "$M/20261004-fatconv98-b2275-fresh.safetensors"
"$BIN" --replay-corpus 20260624-192615-w3aA5b --start-model "$M/20261004-fatconv98-b2275-fresh.safetensors" \
  --out-model "$M/20261004-fatconv98-b2275-replay-latest.safetensors" \
  --parameters experiments/20261004-fatconv-1x15x15-98/parameters.json \
  --epochs 12 --training-step-limit 33000 --enumerate-checkpoints --policy-tail-precision fp32_from_pre_bn
PROBE_BIN="$BIN" TRAINER_PID=<trainer pid> experiments/probe_loop.sh 20261004-fatconv98-b2275 \
  experiments/20261004-fatconv-1x15x15-98/probes.jsonl
```
