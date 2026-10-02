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
| build | Release build 2275 (frozen copy), stamped `f6fdd88` — built from the working tree committed minutes later as `de0f22b` (only tests and docs changed in between) |
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

## Review at 5,000 and 10,000 steps (owner decision at 10k: continue to 33k)

| | ReLU scale+bias (5k) | leaky FC1 (5k) | ReLU scale+bias (10k) | leaky FC1 (10k) |
|---|---:|---:|---:|---:|
| pElo (probed with the `de0f22b` build) | 1239.2 | 1221.6 | 1277.9 | 1270.2 |
| NLL | 2.4899 | 2.5137 | 2.4621 | 2.4609 |
| training loss (last 10 logged steps) | 3.639 | 3.649 | 3.613 | 3.612 |
| FC1 units unmoved from init (b0 / b1 / b2) | 2 / 9 / 0 | 2 / 5 / 0 | 2 / 9 / 0 | 1 / 6 / 0 |
| FC1 units with zero velocity | not saved | 0 / 0 / 0 | not saved | 0 / 0 / 0 |
| FC1 units < 5% of block median velocity | not saved | 0 / 0 / 0 | not saved | 0 / 0 / 0 |

Velocity comparator (ReLU scale+bias **seed 2**, same architecture, different
init, a build that saves velocity): zero-velocity units 4 / 11 / 4 at 5k and
3 / 11 / 5 at 7k; below 5% of the block median 12 / 15 / 11 and 10 / 15 / 8.

- **Dead units:** leaky ReLU keeps every FC1 unit receiving gradient at every
  checkpoint (1k–10k); under ReLU about a third are near-dead. A few leaky units
  barely change direction (weakly active), none is dead.
- **Strength:** no difference through 10k — the arms trade places within seed
  noise (7–25 pElo); losses agree to three decimals.
- **Cost:** median 818 ms/step alone on the GPU (excluding the test-suite window).

## Review at 15,000 steps

| | ReLU scale+bias | leaky FC1 |
|---|---:|---:|
| pElo (probed with the `de0f22b` build) | 1283.6 | 1307.8 |
| NLL | 2.4534 | 2.4252 |
| training loss / policy / value (last 10 logged steps) | 3.604 / 2.779 / 0.816 | 3.603 / 2.779 / 0.815 |
| FC1 units unmoved from init (b0 / b1 / b2) | 2 / 9 / 0 | 1 / 4 / 0 |
| FC1 units < 5% of block median velocity | not saved | 0 / 0 / 0 |

- From 14k to 16k leaky FC1 leads its comparator at three consecutive
  checkpoints (+23 to +28 pElo, −0.022 to −0.028 NLL), the first sustained gap;
  still near the edge of seed-to-seed noise (7–25 pElo).
- Training losses are identical; the difference appears only on the held-out
  probe set.
- Block-0 FC1 units occasionally dip below 5% of the median velocity (4 / 2 / 3
  at 11k / 12k / 13k) and recover; none since 14k.

## Review at 20,000 steps

| | ReLU scale+bias | leaky FC1 |
|---|---:|---:|
| pElo (probed with the `de0f22b` build) | 1287.2 | 1268.6 |
| NLL | 2.4410 | 2.4465 |
| training loss / policy / value (last 10 logged steps) | 3.631 / 2.808 / 0.813 | 3.638 / 2.819 / 0.809 |
| FC1 units unmoved from init (b0 / b1 / b2) | 2 / 6 / 0 | 1 / 3 / 0 |
| FC1 units < 5% of block median velocity | not saved | 0 / 0 / 0 |

- Leaky FC1 led its comparator at six consecutive checkpoints (14k–19k; at 19k
  1338.2 vs 1222.1 pElo, NLL 2.3679 vs 2.4972 — best of all four arms) and is
  behind at 20k. Single checkpoints swing by tens of pElo in every arm (the ReLU
  attenuate-only arm reads 1226.7 at 20k after 1327.9 at 19k).
- **Interruption:** the Mac slept (lid closed on battery) from 20:00:56 on
  2026-10-01 at step ~17,020; training resumed intermittently on wake and fully
  on AC power around 00:07 on 2026-10-02. Step times from 20:00 to ~00:08 are
  excluded from speed figures; training is unaffected.

## Correction (2026-10-02): what leaky ReLU did to the SE bottleneck

The full-model analysis (`full-model-analysis/REPORT.md`, step 23,000) corrects the
dead-unit reading in the 5k–20k reviews above:

- The "below 5% of block median velocity: 0 / 0 / 0" counts are an artifact. More
  than half of each block's FC1 units are weak, so the block *median* is itself a
  weak unit. Against the block's 90th percentile, 15–23 units per block sit below
  5% at every checkpoint.
- Leaky ReLU removed **exact** deadness (no zero-velocity FC1 unit at any of 23
  checkpoints, vs 4 / 13 / 4 for ReLU seed 2), but the units it "revived" mostly
  just trickle: 18 / 17 / 16 units per block stay on the negative side, receiving
  only the 0.01 slope's share of gradient (0.5–1.5% of an active unit's).
- Units actually used (FC2 input column moved ≥ 5° from init) at 23k: leaky
  14 / 9 / 11 vs ReLU seed 1 14 / 10 / 7 — 29 of leaky's 34 are the same units ReLU
  uses; the net gain is ~4 units in block 2.
- ReLU-dead SE units also revive on their own on this architecture (their input
  keeps drifting), so "a dead SE unit can never recover" was wrong here.
- Everything else in the net (stem, tower, LayerNorms, tower end, both heads) has
  no dead, stuck or always-on units in either arm; the only exactly dead weights
  are the 7 always-zero input planes and one never-read en-passant kernel row.

## Review at 25,000 steps

| | ReLU scale+bias | leaky FC1 |
|---|---:|---:|
| pElo (probed with build 2275) | 1437.9 | 1446.7 |
| NLL | 2.2813 | 2.3082 |
| training loss / policy / value (last 10 logged steps) | 3.580 / 2.780 / 0.795 | 3.581 / 2.782 / 0.794 |

- The LR-cycle trough jump (~+90–130 pElo between 23k and 25k) happened in every
  arm; at 25k all four arms sit within 25 pElo (1422.0–1446.7). Leaky FC1 leads
  its comparator on pElo but trails on NLL — no separation.
- Running tally against ReLU scale+bias (pElo, 1k–25k): **ahead at 17 of 25
  checkpoints, behind at 7, tied at 1.** A sign test on the 24 decided checkpoints
  gives p ≈ 0.064 (two-sided), and successive checkpoints of one run are
  correlated, so the effective sample is smaller — a lean toward leaky FC1, not an
  established difference.

## Review at 30,000 steps

| | ReLU scale+bias | leaky FC1 |
|---|---:|---:|
| pElo (probed with build 2275) | 1448.7 | 1474.9 |
| NLL | 2.2532 | 2.2473 |
| training loss / policy / value (last 10 logged steps) | 3.560 / 2.740 / 0.816 | 3.558 / 2.744 / 0.810 |
| FC1 units unmoved from init (per block, of 32) | 2 / 4 / 0 | 1 / 1 / 0 |
| FC1 units with zero velocity | not saved | 0 / 0 / 0 |
| FC1 units below 5% of block p90 velocity | not saved | 18 / 16 / 17 (25k: 17 / 16 / 15) |

- `review.py` now measures low velocity against the block's 90th percentile, not its
  median (the correction above); the 5k–25k "0 / 0 / 0" lines were the median
  artifact.
- Leaky FC1 leads its comparator on both pElo (+26.2) and NLL at 30k. Training
  losses are indistinguishable.
- Tallies, 1k–30k (pElo; NLL in brackets):

  | leaky FC1 vs | ahead / behind / tied | sign test p (two-sided) | mean pElo difference | leaky lower NLL |
  |---|---|---:|---:|---:|
  | ReLU scale+bias (same init) | 22 / 7 / 1 | 0.008 | +16.9 | 19 / 30 |
  | ReLU attenuate-only | 16 / 13 / 1 | 0.71 | +2.7 | 17 / 30 |
  | ReLU no SE | 7 / 22 / 1 | 0.008 | −13.8 | 6 / 30 |

  Successive checkpoints of one run are correlated, so the sign tests overstate the
  evidence, and each arm is one seed (seed spread 7–25 pElo on this setup). Leaky
  FC1 sits consistently above its ReLU twin and consistently below the no-SE net.
