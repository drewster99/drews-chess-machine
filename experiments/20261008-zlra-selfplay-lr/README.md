# ZlrA self-play run and its follow-ups (2026-10-07 → )

Owner request (2026-10-08): stop the GUI self-play run ZlrA, record it as a failure with open
questions, then run two follow-ups at the same time:

- **R-replay** — corpus replay with ZlrA's settings (same architecture, same parameter snapshot).
- **R-fixedlr** — GUI self-play with ZlrA's settings except a fixed learning rate of 0.01 and a
  fixed momentum of 0.90 (LR cycle and momentum cycle off).

## 1. ZlrA (stopped; failure, cause not established)

### Setup

- GUI Play-and-Train, build 2415 (git `dac3a541`, Release), launched 2026-10-07 16:47:40,
  log `dcm_log_20261007-164741.txt`, lineage run `716F996A-7000-4FB7-AF08-10B53710B6F6`.
- Champion model ID `20261007-60-ZlrA` (built in the GUI, fresh init); final trainer
  `20261007-60-ZlrA-20`, final champion `20261007-60-ZlrA-19`.
- Architecture (`zlra_arch.json`): basic24 input; stem 128 channels 5×5; 3 blocks of 9×9 + 9×9 at
  128 channels, no SE, SiLU pre-activation, clean add, no ReZero, `out:layer_norm`; SiLU at the tower
  end, the policy head (intermediate_conv) and the value head (WDL, 24 conv channels → FC 128);
  bf16 compute, policy tail `fp32_from_pre_bn`; 8,271,279 parameters.
- Parameters (`parameters-zlra.json`, the lineage snapshot at the final save):
  - LR cycle on: geometric (log-space) cosine between 0.001 and 1.0, period 10,000 steps, inverted
    (starts at the peak), 1,000 warmup steps; `sqrt_batch_scaling_lr` on, batch 4,096 (scale 1).
  - Momentum follows the LR cycle (0.85–0.95, low momentum at high LR).
  - Weight decay 3e-4, gradient clip 15, relative gradient clip mode 2 (k = 3).
  - Policy label smoothing ε 0.1, value ε 0.013.
  - Self-play τ 1.0 → 0.3, decay 0.02/ply; arena τ 0.6 → 0.2; SPRT elo0 0 / elo1 10, α = β = 0.05.
  - Replay buffer 1M, min 500k; replay ratio target 0.48 auto.
  - `arena_auto_interval_sec` changed by the owner from 900 to 400 at trainer step 1,872.
- Stopped 2026-10-08 09:41:38 by SIGUSR2 at trainer step 51,135 (`[SEGMENT] close (save)`); final
  session `20261008-144140-20261007-61-FNz0-sigusr2.dcmsession` (no replay buffer).
- 134 arenas, 19 promotions.

### Findings

**Puzzle rating (pElo) on the wide set stayed flat from about step 20k.** Wide set: 4,435 Lichess
puzzles, first move only, maximum-likelihood puzzle rating. Mean of the probes within ±500 steps:

| Run | 10k | 20k | 30k | 40k | 50k | best (step) |
|---|---:|---:|---:|---:|---:|---|
| 2Gd1 (2026-06-06) | 587 | 624 | 636 | 657 | 671 | 713 @48.6k |
| eaRt (2026-06-07) | 650 | 663 | 653 | 681 | 679 | 723 @53.8k |
| JhJQ (2026-06-10) | 472 | 558 | 607 | 624 | 645 | 719 @51.2k |
| eBNC (2026-06-12) | 560 | 583 | 606 | 652 | 631 | 734 @46.5k |
| mQl9 (2026-06-18) | 549 | 675 | | | | 738 @25.7k |
| LMGh (2026-06-09) | 557 | 590 | 561 | 579 | 540 | 740 @77.7k |
| wTp3 (2026-06-14) | 561 | 567 | 571 | | | 618 @23.8k |
| zEFi (2026-06-23) | 446 | 503 | 500 | 430 | 460 | 685 @29.3k |
| **ZlrA** | 482 | 536 | 549 | 542 | 544 | 594 @12.6k |

The June runs are every GUI self-play run from a fresh init with wide-set probes past 20k steps
(the same 4,435-puzzle set and pElo method throughout). A blank cell: the run never reached that
step. They differ from ZlrA in architecture and settings (flat LR 0.01, self-play τ 1.0 → 0.5 at
0.007/ply), so this is not a controlled comparison.

**Training losses were far below every other run.** pLoss / vLoss, mean of `[STATS]` lines within
±1,000 steps:

| Run | pLoss 10k | 30k | 50k | vLoss 10k | 30k | 50k |
|---|---:|---:|---:|---:|---:|---:|
| 2Gd1 | 1.90 | 1.67 | 1.93 | 0.67 | 0.58 | 0.72 |
| eaRt | 1.89 | 1.91 | 2.08 | 0.65 | 0.66 | 0.77 |
| JhJQ | 1.82 | 1.94 | 1.90 | 0.59 | 0.67 | 0.68 |
| eBNC | 1.79 | 1.67 | 1.81 | 0.60 | 0.54 | 0.69 |
| **ZlrA** | 0.89 | 0.88 | 0.72 | 0.18 | 0.19 | 0.16 |

They start in the normal range (step 1: vLoss 0.744 vs 2Gd1 0.750) and separate within the first
1,000 steps (ZlrA vLoss 0.315 at 500, 0.198 at 1,000; 2Gd1 0.497 at 500). The first batches were
alike: ZlrA 80.2% draws / 9.7% wins / 10.1% losses, 2Gd1 82.0% / 9.5% / 8.5%, similar game
lengths, same batch size and per-game cap. The learning rate was not: ZlrA warms up to 1.0 by step
1,000; the June runs used a flat 0.01.

**At the LR troughs** (lowest-LR step line of each trough; wide probe nearest that step):

| Trough | Step | LR | pElo | NLL | pLoss | vLoss |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 5,995 | 0.00097 | 494 | 3.957 | 1.305 | 0.423 |
| 2 | 15,973 | 0.00090 | 539 | 3.822 | 1.389 | 0.497 |
| 3 | 25,985 | 0.00084 | 555 | 3.719 | 1.448 | 0.508 |
| 4 | 35,911 | 0.00079 | 538 | 3.732 | 1.280 | 0.427 |

**Layer health ratcheted at each LR peak.** No channel was ever parked (SiLU pass-through below
Φ(−3)), no non-finite values, no gradient spikes, 0 clips. But:

- BN running-variance max/median rose at every peak and held at the troughs: about 16 (trough 1) →
  79 (trough 2) → 529–589 (trough 3) → 1,008 at 42,600 (`bn_running_variance_runaway` raised) →
  1,654.7 at the end (`blocks.1.bn1[25]`, 75 channels ≥ 10× their site median).
- `policy.pre_bn[122]` (feeds short queen-style moves) β/|γ| 0.84 → 31.0, |γ| under 5% of the site
  median: a channel pinned linear with nearly constant output.
- Value FC1 units 16 and 121 at 2.4% / 3.4% of the p90 velocity at 43,016, outgoing W/D/L weight
  about 0.07 vs median 0.77; unit 66 heading the same way.
- `policy_offset_drift` warning active from 10,600 (median |policy logit mean| up to about 7.3).

### Interpretation (open questions)

- The leading hypothesis is that the LR cycle's peak of 1.0 (100× the June runs' flat 0.01) let the
  network fit its own replay buffer: positions of one game share its result, and a fast-fitting
  value head can learn the game rather than the position. That would explain losses far below every
  other run while the puzzle rating stays flat. **Not proven:** the losses are measured on the
  buffer the network trains on, so they cannot show it.
- Colder self-play (τ floor 0.3 reached by ply 35, vs 0.5 at ply 71) also makes the buffer easier
  to predict; it does not explain the separation in the first 1,000 steps, when the buffer was still
  the initial games.
- The same LR cycle on corpus replay (`20261005-lr-schedule-ab`, arms B / B-silu-clip*) reached
  about 1,630 pElo, so the cycle alone is not fatal; what differs is the data source.
- Open: (1) loss on fresh, never-sampled games vs the training loss (a gap would confirm fitting
  the buffer); (2) whether the variance ratchet and the pinned policy channel cost pElo or only
  coincide with it; (3) how much of the gap to the June runs is architecture rather than schedule.

The two follow-ups separate the first two causes: R-replay keeps the schedule and removes self-play;
R-fixedlr keeps self-play and removes the schedule.

## 2. Follow-up runs (launched 2026-10-08 09:46, running side by side)

Shared:

- Binary `~/Library/Application Support/DrewsChessMachine/FrozenBuilds/DCM-2440-9c6a463c.app`
  (build 2440, git `9c6a463c`, clean, Release; executable sha256 prefix `d45267a1ad05`). Both runs
  record their weights' wide- and 200-set results in every model file they write
  (`dcm_test_set_results`), so no separate probe loop.
- Start model `Models/20261008-zlra-ab-fresh.safetensors`, model ID `20261008-4-Ew5t`, minted from
  `zlra_arch.json` with `--init-seed 20261008` (`mint.txt`); its architecture metadata is identical
  to ZlrA's final trainer file (checked by JSON comparison). ZlrA's own initial weights were never
  saved, so the init differs from ZlrA's.
- `--seed 20261008`.

| | R-replay | R-fixedlr |
|---|---|---|
| launch | `launch/replay_launch.sh` | `launch/fixedlr_launch.sh` |
| path | `--replay-corpus 20260624-192615-w3aA5b` | GUI `--train` (Play-and-Train, arenas, SPRT) |
| parameters | `parameters-zlra.json` (ZlrA's snapshot, unchanged) | `parameters-selfplay-fixedlr.json` |
| LR / momentum | cycle 0.001–1.0 / follows LR 0.85–0.95 | fixed 0.01 / fixed 0.90 (1,000 warmup steps kept) |
| budget | `--training-step-limit 60000`, `--enumerate-checkpoints` | until stopped |
| pid / log | 20981, `dcm_log_20261008-094622.txt` | 20985, `dcm_log_20261008-094623.txt` |
| lineage run | `0EFE2DA3-070A-4EDF-954A-DB8EB92309B9` | `A2599216-F200-4555-84AD-0CC1BF546011` |
| outputs | `Models/20261008-zlra-replay-{latest,step<T>}.safetensors`, `results-replay.json`, `train-replay.stdout` | session folders under `Sessions/`, `results-fixedlr.json` on stop, `train-fixedlr.stdout` |

`parameters-selfplay-fixedlr.json` differs from `parameters-zlra.json` in exactly four keys:
`lr_cycle_enabled` false, `learning_rate` 0.01, `momentum_cycle_enabled` false,
`momentum_follows_lr_cycle` false (`momentum_coeff` was already 0.9). With both cycles off the
trainer feeds the static `learning_rate` and `momentum_coeff` (`LRMomentumCycle.swift`, the `Fed`
computation); the first step line shows `lr=1.0e-02·√b·warmup(1/1000)` and `μ=0.900`.

Decisions made without the owner (2026-10-08):

- Kept the 1,000-step LR warmup in R-fixedlr: "fixed" was read as no cycle, and warmup is part of
  ZlrA's settings.
- Corpus `20260624-192615-w3aA5b`, the corpus of every earlier LR-schedule replay arm.
- R-replay step budget 60,000 (ZlrA stopped at 51,135).
- Shared fresh init for both runs (comparable to each other, not to ZlrA's init).

## Reproduce

```sh
BIN="$HOME/Library/Application Support/DrewsChessMachine/FrozenBuilds/DCM-2440-9c6a463c.app/Contents/MacOS/DrewsChessMachine"
M="$HOME/Library/Application Support/DrewsChessMachine/Models"
E=experiments/20261008-zlra-selfplay-lr
"$BIN" --new-model --architecture $PWD/$E/zlra_arch.json --init-seed 20261008 --name zlra-ab-fresh \
  --out-model "$M/20261008-zlra-ab-fresh.safetensors"
$E/launch/replay_launch.sh &
$E/launch/fixedlr_launch.sh &   # then `open -g <the .app>` if the window never appears (auto-train fires on first appear)
```
