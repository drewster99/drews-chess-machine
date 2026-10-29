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

## 3. R-elitecyc and R-elitefresh (2026-10-11 → ; running)

Status (2026-10-23): both runs stopped by hand together on 2026-10-20 07:30 (R-elitecyc at trainer step 314,767,
R-elitefresh at 196,491) and were **resumed on 2026-10-23** (owner: "continue both runs"); both are **running**.
The results below cover everything through the 2026-10-20 07:30 stop.

- R-elitecyc lineage segment 12: `launch/elitecyc_resume11_launch.sh` → `train-elitecyc-seg11.stdout`,
  `results-elitecyc-seg11.json`; `[RESUME] EXACT` from `20261020-22-Irow` at 314,767, build 2523, `--epochs 3`.
- R-elitefresh lineage segment 8: `launch/elitefresh_resume9_launch.sh` → `train-elitefresh-seg9.stdout`,
  `results-elitefresh-seg9.json`; `[RESUME] EXACT` from `20261020-23-PA3e` at 196,491, build 2523.

### What it is

A corpus-replay continuation of R-replay on the Lichess Elite corpus with a gentle LR cycle, and a from-scratch control
on the same corpus. Two steps led to R-elitecyc:

- **R-elite** (owner 2026-10-11: does the May corpus's ~1,640-mean teaching cap R-replay at ~1,610–1,650 pElo?):
  a branch of R-replay's SIGINT save at trainer step 86,398 (`20261008-zlra-replay-step86398.safetensors`, model ID
  `20261011-2-abM2`, wide pElo 1,639.7) onto the elite corpus at a fixed LR of 0.002, momentum 0.90
  (`launch/replay_elite_fork_launch.sh`, build 2501, lineage run `ACC058E9-15A5-49EA-8F60-9D1F254EAA4C`). It stayed
  flat: wide pElo 1,628–1,645 at every 1,000-step probe from 1k to 21k. Stopped by SIGINT at trainer step 21,006
  (`20261011-zlra-elite-latest-step21006.safetensors`, model ID `20261011-4-btGW`, 1,637.6), corpus position
  `nextGame=2162743`.
- **R-elitecyc** (owner 2026-10-11: "create a less intense cycle"): a branch (not an exact resume, which would have
  restored the checkpoint's fixed LR) of R-elite's step-21,006 save with `parameters-zlra-elite-gentlecycle.json`.
  It differs from `parameters-zlra-elite-fixedlr.json` (R-elite's) in four keys: `lr_cycle_enabled` true,
  `lr_cycle_invert` false, `lr_cycle_max` 0.02, `lr_cycle_decay_horizon_steps` 0. So: geometric cosine
  0.002 ↔ 0.02, period 10,000, not inverted, no decay; momentum fixed 0.90; no √batch scaling; 1,000-step warmup to
  the 0.002 trough; LR peaks at trainer steps 6,000, 16,000, … Everything else is R-replay's (ZlrA's) snapshot:
  batch 4,096, weight decay 3e-4, gradient clip 15 plus the relative cap in clip mode (k = 3), policy ε 0.1, value
  ε 0.013, buffer 1M (500k before training), replay ratio 0.48, at most 10 positions per game. The branch restarts
  the trainer step at 0 with zero velocity and continues the corpus where R-elite stopped
  (`--start-game-index 2162743`), so R-elite + R-elitecyc's epoch 0 together fed the whole corpus once.
- **R-elitefresh** (owner 2026-10-12: "an elite corpus run with same params as R-replay", from scratch, beside
  R-elitecyc): R-replay's start file (`20261008-zlra-ab-fresh.safetensors`, model ID `20261008-4-Ew5t`, untrained),
  parameter file (`parameters-zlra.json`) and seed (`--seed 20261008`) on the elite corpus from game 0. Its LR cycle
  is R-replay's: inverted geometric cosine, period 10,000, 1,000 warmup steps, √batch scaling, momentum following
  the cycle, and both ends decaying geometrically toward step 1,000,000 (`lr_cycle_peak_end` 0.001,
  `lr_cycle_trough_end` 1e-6): the cycle LR (`lr_effective_base` in the step rows, before warmup) is 1.0 at the
  start, 0.575 at the 81,000 peak and 0.269 at the 191,000 peak; the troughs (steps 6,000, 16,000, …) fall from
  0.00097 to 0.00026 at 196,000.
  Lineage run `F79C7568-4DDB-443E-BFB4-8131DAB27F4A`.

Every R-elitecyc step number below is its own trainer step. At its step 0 the weights already carried 107,404 steps
(86,398 R-replay + 21,006 R-elite).

| segment | launch | build | trainer steps | wall (CDT) | ended | stdout / results |
|---|---|---|---|---|---|---|
| 0 | `launch/elite_gentlecycle_branch_launch.sh` (branch from `20261011-4-btGW`) | 2503, git `7d945ec7` | 0 → 45,467 | 10-11 21:39 → 10-12 15:25 | SIGINT abort save (rebuild for LR keyframes) | `train-elitecyc.stdout`, `results-elitecyc.json` |
| 1 | `launch/elitecyc_resume1_launch.sh` (`--resume-exact`) | 2523, git `a242b7ad` | 45,467 → 126,222 | 10-12 15:30 → 10-14 19:57 | SIGINT abort save (macOS update) | `train-elitecyc-seg1.stdout`, `results-elitecyc-seg1.json` |
| 2 | `launch/elitecyc_resume2_launch.sh` (`--resume-exact`) | 2523, git `a242b7ad` | 126,222 → 127,931 | 10-14 20:11 → 21:24 | end of epoch 0 (final save; the counter then reads `nextGame=0 epoch=1`) | `train-elitecyc-seg2.stdout`, `results-elitecyc-seg2.json` |
| 3 | `launch/elitecyc_resume3_launch.sh` (`--resume-exact`, `--epochs 3`) | 2523, git `a242b7ad` | 127,931 → 134,247 | 10-15 13:16 → 18:25 | SIGINT abort save (owner: stop everything but the Lichess bot) | overwritten by segment 4 (same file names, `--overwrite-output`); this segment's versions are in git at `8747714d` |
| 4 | `launch/elitecyc_resume3b_launch.sh` | 2523, git `a242b7ad` | 134,247 → 194,481 | 10-15 23:08 → 10-16 20:01 | SIGINT abort save (owner: "pause chess trainer, restart in 5 hours") | `train-elitecyc-seg3.stdout`, `results-elitecyc-seg3.json` |
| 5 | `launch/elitecyc_resume4_launch.sh` | 2523, git `a242b7ad` | 194,481 → 234,240 | 10-17 01:03 → 17:50 | SIGINT abort save; reason not recorded (R-elitefresh stopped the same minute) | `train-elitecyc-seg4.stdout`, `results-elitecyc-seg4.json` |
| 6 | `launch/elitecyc_resume5_launch.sh` | 2523, git `a242b7ad` | 234,240 → 234,700 | 10-17 18:21 → 19:12 | SIGINT abort save; reason not recorded (3 min after R-elitefresh's GPU out-of-memory fault) | `train-elitecyc-seg5.stdout`, `results-elitecyc-seg5.json` |
| 7 | `launch/elitecyc_resume6_launch.sh` | 2523, git `a242b7ad` | 234,700 → 239,101 | 10-17 20:12 → 23:51 | SIGINT abort save; reason not recorded | `train-elitecyc-seg6.stdout`, `results-elitecyc-seg6.json` |
| 8 | `launch/elitecyc_resume7_launch.sh` | 2523, git `a242b7ad` | 239,101 → 260,410 | 10-17 23:57 → 10-18 13:24 | SIGINT abort save; reason not recorded | `train-elitecyc-seg7.stdout`, `results-elitecyc-seg7.json` |
| 9 | `launch/elitecyc_resume8_launch.sh` | 2523, git `a242b7ad` | 260,410 → 262,044 | 10-18 14:25 → 16:17 | SIGINT abort save; reason not recorded | `train-elitecyc-seg8.stdout`, `results-elitecyc-seg8.json` |
| 10 | `launch/elitecyc_resume9_launch.sh` | 2523, git `a242b7ad` | 262,044 → 300,300 | 10-18 19:00 → 10-19 19:56 | SIGINT abort save; reason not recorded | `train-elitecyc-seg9.stdout`, `results-elitecyc-seg9.json` |
| 11 | `launch/elitecyc_resume10_launch.sh` | 2523, git `a242b7ad` | 300,300 → 314,767 | 10-19 23:01 → 10-20 07:30 | SIGINT abort save (`manual_stop`) | `train-elitecyc-seg10.stdout`, `results-elitecyc-seg10.json` |
| 12 | `launch/elitecyc_resume11_launch.sh` | 2523, git `a242b7ad` | 314,767 → (running) | 10-23 → | running | `train-elitecyc-seg11.stdout`, `results-elitecyc-seg11.json` |

Segment numbers are the lineage `segment_index`; file suffixes lag it by one from segment 4 on (segment 3's resume
`3b` reused the `seg3` names). Times are CDT, from each segment's lineage record (`segment_started_unix`,
`recorded_unix`). Every segment from 3 on ended with `termination_reason: manual_stop` after a SIGINT abort save
(`[REPLAY] SIGINT received … abort requested`). Whatever triggered the SIGINTs after 10-16 was not written down;
from 10-17 on, R-elitecyc and R-elitefresh were stopped and resumed together (same minutes).

- Lineage run `E709F4EB-1D95-4166-90FA-F0D31002BB89`; seed `16308727193092630423` (drawn at segment 0, continued by
  both resumes). Both resumes logged `[RESUME] EXACT` (segment 1: build changed, behavior fingerprint matches;
  segment 2: OS 26B5091g → 26B5112a, fingerprint matches; no `--accept-inexact` needed).
- Final file `20261011-zlra-elitecyc-latest-step127931.safetensors` (also the rolling `…-latest`), model ID
  `20261015-1-KXNZ`. 130 enumerated checkpoints (every 1,000 steps plus 45,467, 126,222, 127,931).
- Trainer time 246,502 s (68.5 h), wall 257,091 s (71.4 h) summed over the segments (lineage `time`). Games fed
  2,162,743 → 14,619,153 (end of corpus), 12,456,410 games.
- `results-elitecyc-seg2.json` records `termination_reason: step_limit_reached` although no step limit was set
  (`stepLimit=none epochLimit=1`); the log shows the run ended at the end of epoch 0 (`[REPLAY] done: … epochs=1`,
  the count of epochs finished).

Continuation, segments 3–11 (owner 2026-10-15: "another round of elite corpus through elitecyc ... let's actually do
2 more"):

- `--epochs 3`, counted from the lineage start: epoch 0 was the partial pass from game 2,162,743, so the budget is
  two more full passes (epochs 1 and 2), ending at the end of epoch 2.
- All nine resumes logged `[RESUME] EXACT` (`not_exact_items` empty, no `--accept-inexact`): build 2523 (git
  `a242b7ad`, clean), macOS 27.2 (26B5112a), Apple M5 Max throughout.
- Epoch 1 (the first full pass, 14,619,153 games) ran trainer steps 127,931 → about 276,950: the step lines at
  276,910 and 276,990 read `epoch=1` and `epoch=2`; the 277,000 save reads `nextGame=1706 epoch=2`.
- At the 2026-10-20 07:30 stop: trainer step 314,767, epoch 2, next game 3,812,602 (26.1% of the pass).
- Final file `20261011-zlra-elitecyc-latest-step314767.safetensors` (also the rolling `…-latest`), model ID
  `20261020-22-Irow`, content sha256 `70272cd6ec7036587c1c5b8945926c55f979b8ede014625783fe52f1fb514e08` (lineage
  `checkpoint_sha256`). The Lichess bot plays these weights.
- 326 enumerated checkpoints in all (196 since 127,931: every 1,000 steps plus each segment's end), every one with
  recorded test-set results.
- Totals over all 12 segments (lineage `time` / `fed`): trainer time 576,694 s (160.2 h), wall 602,846 s (167.5 h);
  games fed 30,888,165 = 12,456,410 (epoch 0) + 14,619,153 (epoch 1) + 3,812,602 (epoch 2).
- Throughput (trained positions = steps × 4,096 per segment trainer time): 12.5M positions/h (1,179 ms/step) in
  segment 4, while R-elitefresh was paused; 10.3M (1,436 ms/step) in segment 5, alone until R-elitefresh resumed at
  14:51; 5.2M (2,830 ms/step) in segment 3 and 6.5M–7.2M (2,040–2,260 ms/step) in segments 8, 10, 11, beside
  R-elitefresh; 2.5M / 5.1M / 4.0M positions/h in the short segments 6, 7, 9 (cause not recorded).

R-elitefresh segments (build 2523, git `a242b7ad`, clean, for all):

| segment | launch | trainer steps | wall (CDT) | ended | stdout / results |
|---|---|---|---|---|---|
| 0 | `launch/elitefresh_launch.sh` (branch from `20261008-4-Ew5t`, `--seed 20261008`, `--epochs 1`) | 0 → 80,518 | 10-12 15:31 → 10-14 19:57 | SIGINT abort save (macOS update) | `train-elitefresh.stdout`, `results-elitefresh.json` |
| 1 | `launch/elitefresh_resume1_launch.sh` (`--resume-exact`, `--epochs 1`) | 80,518 → 111,327 | 10-14 20:11 → 10-15 18:25 | SIGINT abort save (owner: stop everything but the Lichess bot) | `train-elitefresh-seg1.stdout`, `results-elitefresh-seg1.json` |
| 2 | `launch/elitefresh_resume2_launch.sh` (`--epochs 3`, from here on) | 111,327 → 116,369 | 10-17 14:51 → 17:50 | SIGINT abort save; reason not recorded | `train-elitefresh-seg2.stdout`, `results-elitefresh-seg2.json` |
| — | `launch/elitefresh_resume3_launch.sh` | 116,369 → 116,820, not saved | → 10-17 19:09 | **GPU fault** (out of memory) during trainer step 116,821; no save, 451 steps lost | `train-elitefresh-seg3.stdout`, `results-elitefresh-seg3.json` (`termination_reason: gpu_fault`) |
| 3 | `launch/elitefresh_resume4_launch.sh` (from 116,369 again) | 116,369 → 120,754 | 10-17 20:12 → 23:50 | SIGINT abort save; reason not recorded | `train-elitefresh-seg4.stdout`, `results-elitefresh-seg4.json` |
| 4 | `launch/elitefresh_resume5_launch.sh` | 120,754 → 142,084 | 10-17 23:57 → 10-18 13:24 | SIGINT abort save; reason not recorded | `train-elitefresh-seg5.stdout`, `results-elitefresh-seg5.json` |
| 5 | `launch/elitefresh_resume6_launch.sh` | 142,084 → 143,756 | 10-18 14:25 → 16:17 | SIGINT abort save; reason not recorded | `train-elitefresh-seg6.stdout`, `results-elitefresh-seg6.json` |
| 6 | `launch/elitefresh_resume7_launch.sh` | 143,756 → 182,041 | 10-18 19:00 → 10-19 19:56 | SIGINT abort save; reason not recorded | `train-elitefresh-seg7.stdout`, `results-elitefresh-seg7.json` |
| 7 | `launch/elitefresh_resume8_launch.sh` | 182,041 → 196,491 | 10-19 23:01 → 10-20 07:30 | SIGINT abort save (`manual_stop`) | `train-elitefresh-seg8.stdout`, `results-elitefresh-seg8.json` |
| 8 | `launch/elitefresh_resume9_launch.sh` | 196,491 → 221,457 | 10-23 07:26 → 10-24 02:06 | SIGINT abort save | `train-elitefresh-seg9.stdout`, `results-elitefresh-seg9.json` |
| 9 | `launch/elitefresh_resume10_launch.sh` | 221,457 → 245,158 | 10-24 22:48 → 10-25 14:23 | SIGINT abort save (owner; GPU for R-elitecyc-anneal) | `train-elitefresh-seg10.stdout`, `results-elitefresh-seg10.json` |
| 10 | `launch/elitefresh_resume11_launch.sh` (`--epochs 5`, step limit to trainer step 700,000, owner 10-28) | 245,158 → (running) | 10-28 21:37 → | running | `train-elitefresh-seg11.stdout`, `results-elitefresh-seg11.json` |

- Every resume logged `[RESUME] EXACT`; segment 1 also `[RESUME] OS changed (Version 27.2 (Build 26B5091g) →
  Version 27.2 (Build 26B5112a)), behavior fingerprint matches`.
- The faulted attempt saved nothing (`[REPLAY] GPU fault during trainer step 116821; stopping training, no further
  saves`), so it has no lineage segment index; segment 3 resumed from segment 2's 116,369 save and retrained those
  steps.
- Epoch 0 (the first full pass) ended between trainer steps 148,940 and 149,000 (the 149,000 save reads
  `nextGame=547 epoch=1`). At the 2026-10-20 stop: trainer step 196,491, epoch 1, next game 4,763,961 (32.6% of the
  pass).
- Final file `20261012-zlra-elitefresh-latest-step196491.safetensors` (also the rolling `…-latest`), model ID
  `20261020-23-PA3e`. 204 enumerated checkpoints, every one with recorded test-set results.
- Totals over the eight saved segments: trainer time 449,570 s (124.9 h), wall 468,036 s (130.0 h); games fed
  19,383,114 = 14,619,153 (epoch 0) + 4,763,961 (epoch 1).
- Throughput: 6.0M–7.2M positions/h beside R-elitecyc in its long segments (2,042–2,465 ms/step).

### Results

Wide set (4,435 Lichess puzzles, first move, maximum-likelihood rating), from each checkpoint's own
`dcm_test_set_results` (read with `--describe-model`, build 2614; checkpoints identified by `model_id` and trainer
step from the file metadata). Every R-elitecyc, R-elitefresh, R-elite and R-replay checkpoint has recorded results.

**R-elitecyc, epoch 0 (segments 0–2):**

- **Peak 1,726.6 at step 120,000** (NLL 1.9449, top-1 2,399 / 4,435). Lowest NLL 1.9361 at 126,222 (1,718.4).
- **End of epoch 0 (127,931): 1,716.3, NLL 1.9634**, top-1 2,379, top-5 3,832; 200-set 1,414.9, NLL 1.7988.
- Start file (R-elite 21,006): 1,637.6 / 2.0654. 127,931 − start +78.7 pElo, NLL −0.102. Mean of the last 10 probes
  (118k–127k) 1,712.6; of the first 10 (1k–10k) 1,647.8.
- The LR peaks cost little: pElo at each peak step (6k, 16k, …, 126k) was on average 18 below the best probe of the
  same cycle (range 1–40). R-replay's 1.0 peaks: 194 on average (119–284; peaks at 11k … 81k).

**R-elitecyc, epochs 1–2 (segments 3–11, to the 2026-10-20 stop):**

- **Peak 1,786.0 at step 291,000** (NLL 1.8695, top-1 2,514 / 4,435), in epoch 2. Lowest NLL **1.8653 at 296,000**
  (1,772.0). Best 200-set 1,502.4 at 296,000.
- **Final (314,767): 1,776.7, NLL 1.8683**, top-1 2,496, top-5 3,923; 200-set 1,474.4, NLL 1.6351 (model
  `20261020-22-Irow`).
- Final − end of epoch 0 (127,931): +60.4 pElo, NLL −0.0951. Final − start file: +139.1 pElo, NLL −0.1971.
- LR-peak cost (pElo at each peak step 136k, 146k, …, 306k below the best probe of the same 10,000-step cycle, the
  definition that reproduces epoch 0's 18): mean 29.6 over 18 peaks (range 0.0–53.1).
- Best probe per 10,000-step window (ending at the step shown): 130k 1,735.9 · 140k 1,739.0 · 150k 1,730.2 ·
  160k 1,739.5 · 170k 1,732.3 · 180k 1,731.3 · 190k 1,735.9 · 200k 1,737.4 · 210k 1,747.8 · 220k 1,741.1 ·
  230k 1,749.3 · 240k 1,766.8 · 250k 1,753.4 · 260k 1,758.6 · 270k 1,774.1 · 280k 1,767.9 · 290k 1,769.4 ·
  300k 1,786.0 · 310k 1,778.7.

Per epoch (probes assigned by trainer step; epoch 1 → 2 at about 276,950):

| epoch | trainer steps | probes | best pElo (step) | lowest NLL (step) | mean pElo | mean NLL | mean of last 10 probes |
|---:|---|---:|---|---|---:|---:|---:|
| 0 | 1,000–127,931 | 130 | 1,726.6 (120,000) | 1.9361 (126,222) | 1,683.2 | 1.9990 | 1,715.1 |
| 1 | 128,000–276,000 | 156 | 1,774.1 (269,000) | 1.8702 (269,000) | 1,731.1 | 1.9257 | 1,750.9 |
| 2 (26.1% of the pass) | 277,000–314,767 | 40 | 1,786.0 (291,000) | 1.8653 (296,000) | 1,759.9 | 1.8922 | 1,760.5 |

**R-elitefresh (segments 0–7, to the 2026-10-20 stop):**

- **Peak 1,638.6 at step 159,000**, also its lowest NLL (2.0639), top-1 2,228 / 4,435; in epoch 1. Best 200-set
  1,388.5 at 172,000.
- **Final (196,491): 1,614.5, NLL 2.0745**, top-1 2,181, top-5 3,711; 200-set 1,357.1, NLL 1.9375 (model
  `20261020-23-PA3e`).
- Below R-replay at every one of the 86 shared probe steps (1,000–86,000; mean −72.2 pElo). Its peak (1,638.6, step
  159,000) is level with R-replay's last save (1,639.7 at 86,398) and below R-replay's best (1,649.9 at 48,000).
- LR-peak cost (pElo at each 1.0-cycle peak below the best probe within ±5,000 steps, R-replay's definition): mean
  200.1 over the peaks 11k–81k, 156.4 over 91k–191k as the peak LR decays from 0.537 to 0.269.

| epoch | trainer steps | probes | best pElo (step) | lowest NLL (step) | mean pElo | mean NLL | mean of last 10 probes |
|---:|---|---:|---|---|---:|---:|---:|
| 0 | 1,000–148,000 | 154 | 1,593.4 (138,000) | 2.1143 (99,000) | 1,506.0 | 2.2217 | 1,566.6 |
| 1 (32.6% of the pass) | 149,000–196,491 | 50 | 1,638.6 (159,000) | 2.0639 (159,000) | 1,583.1 | 2.1225 | 1,583.4 |

**Comparison at each run's own trainer step** (value at the step, and the best probe in the 10,000 steps ending
there). R-replay and R-elitefresh started from the untrained `20261008-zlra-ab-fresh` network with the 0.001–1.0
cycle (`parameters-zlra.json`): R-replay on `w3aA5b` (stopped at 86,398), R-elitefresh on the elite corpus. R-elite
(fixed LR 0.002, elite corpus) branched from R-replay at 86,398 and stopped at 21,006; R-elitecyc branched from
R-elite's 21,006 save. This is not an equal-history comparison: R-elitecyc began from a network with 107,404 steps
behind it, R-elite from one with 86,398. A blank cell: the run never reached that step.

| step | R-elitecyc | best | R-elite | best | R-replay | best | R-elitefresh | best |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 10,000 | 1658.7 | 1658.7 | 1635.0 | 1638.6 | 1392.7 | 1460.5 | 1364.4 | 1396.8 |
| 20,000 | 1660.2 | 1666.9 | 1636.6 | 1645.3 | 1424.6 | 1564.2 | 1396.3 | 1501.1 |
| 30,000 | 1674.1 | 1682.3 |  |  | 1520.6 | 1615.0 | 1439.0 | 1547.8 |
| 40,000 | 1678.2 | 1678.2 |  |  | 1504.6 | 1611.4 | 1461.5 | 1529.3 |
| 50,000 | 1685.9 | 1685.9 |  |  | 1502.1 | 1649.9 | 1460.0 | 1550.3 |
| 60,000 | 1699.8 | 1699.8 |  |  | 1589.8 | 1629.4 | 1499.0 | 1573.9 |
| 70,000 | 1701.9 | 1701.9 |  |  | 1554.9 | 1638.1 | 1497.5 | 1574.4 |
| 80,000 | 1691.1 | 1700.9 |  |  | 1568.8 | 1620.1 | 1465.1 | 1570.3 |
| 90,000 | 1703.4 | 1707.0 |  |  |  |  | 1464.6 | 1559.6 |
| 100,000 | 1707.0 | 1707.0 |  |  |  |  | 1492.3 | 1586.2 |
| 110,000 | 1705.0 | 1708.1 |  |  |  |  | 1546.2 | 1590.9 |
| 120,000 | 1726.6 | 1726.6 |  |  |  |  | 1528.8 | 1586.2 |
| 130,000 | 1722.5 | 1735.9 |  |  |  |  | 1514.9 | 1581.6 |
| 140,000 | 1726.1 | 1739.0 |  |  |  |  | 1510.8 | 1593.4 |
| 150,000 | 1727.1 | 1730.2 |  |  |  |  | 1564.2 | 1600.1 |
| 160,000 | 1739.5 | 1739.5 |  |  |  |  | 1563.7 | 1638.6 |
| 170,000 | 1722.0 | 1732.3 |  |  |  |  | 1602.7 | 1613.5 |
| 180,000 | 1717.3 | 1731.3 |  |  |  |  | 1570.8 | 1628.4 |
| 190,000 | 1718.9 | 1735.9 |  |  |  |  | 1594.5 | 1625.8 |
| 200,000 | 1735.4 | 1737.4 |  |  |  |  |  |  |
| 210,000 | 1744.1 | 1747.8 |  |  |  |  |  |  |
| 220,000 | 1740.5 | 1741.1 |  |  |  |  |  |  |
| 230,000 | 1745.2 | 1749.3 |  |  |  |  |  |  |
| 240,000 | 1740.5 | 1766.8 |  |  |  |  |  |  |
| 250,000 | 1753.4 | 1753.4 |  |  |  |  |  |  |
| 260,000 | 1758.6 | 1758.6 |  |  |  |  |  |  |
| 270,000 | 1763.2 | 1774.1 |  |  |  |  |  |  |
| 280,000 | 1767.9 | 1767.9 |  |  |  |  |  |  |
| 290,000 | 1762.2 | 1769.4 |  |  |  |  |  |  |
| 300,000 | 1757.6 | 1786.0 |  |  |  |  |  |  |
| 310,000 | 1767.4 | 1778.7 |  |  |  |  |  |  |

Best per run: R-elitecyc 1,786.0 @291,000 (1,776.7 at its last save, 314,767); R-replay 1,649.9 @48,000 (1,639.7 at
its last save, 86,398); R-elitefresh 1,638.6 @159,000 (1,614.5 at its last save, 196,491); R-elite 1,645.3 @19,000
(1,637.6 at its last save, 21,006).

<details>
<summary>Every probe: wide-set pElo and NLL at each checkpoint, all four runs (336 rows)</summary>

| step | R-elitecyc pElo | NLL | R-elite pElo | NLL | R-replay pElo | NLL | R-elitefresh pElo | NLL |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1,000 | 1645.8 | 2.0678 | 1637.6 | 2.0894 | 1109.6 | 2.6606 | 1069.7 | 2.7607 |
| 2,000 | 1652.0 | 2.0634 | 1633.0 | 2.0869 | 1258.8 | 2.5451 | 1216.9 | 2.6049 |
| 3,000 | 1648.9 | 2.0672 | 1638.6 | 2.0823 | 1415.3 | 2.3376 | 1343.9 | 2.4107 |
| 4,000 | 1649.9 | 2.0700 | 1628.4 | 2.0896 | 1443.1 | 2.2911 | 1388.6 | 2.3727 |
| 5,000 | 1646.9 | 2.0730 | 1634.0 | 2.0831 | 1457.4 | 2.2736 | 1392.2 | 2.3638 |
| 6,000 | 1632.0 | 2.0816 | 1633.0 | 2.0845 | 1456.4 | 2.2723 | 1396.8 | 2.3660 |
| 7,000 | 1646.3 | 2.0548 | 1632.0 | 2.0864 | 1454.4 | 2.2712 | 1396.8 | 2.3635 |
| 8,000 | 1640.2 | 2.0547 | 1632.5 | 2.0790 | 1460.5 | 2.2590 | 1390.7 | 2.3636 |
| 9,000 | 1657.7 | 2.0440 | 1632.5 | 2.0801 | 1406.6 | 2.3165 | 1371.6 | 2.3960 |
| 10,000 | 1658.7 | 2.0395 | 1635.0 | 2.0747 | 1392.7 | 2.3355 | 1364.4 | 2.4035 |
| 11,000 | 1660.7 | 2.0372 | 1640.2 | 2.0737 | 1375.2 | 2.3646 | 1284.1 | 2.4926 |
| 12,000 | 1666.9 | 2.0372 | 1641.2 | 2.0750 | 1481.0 | 2.2662 | 1304.7 | 2.4290 |
| 13,000 | 1656.1 | 2.0519 | 1637.6 | 2.0740 | 1512.9 | 2.2041 | 1429.7 | 2.2953 |
| 14,000 | 1660.2 | 2.0533 | 1640.2 | 2.0682 | 1544.2 | 2.1647 | 1491.8 | 2.2406 |
| 15,000 | 1652.5 | 2.0515 | 1631.4 | 2.0771 | 1554.9 | 2.1587 | 1491.3 | 2.2342 |
| 16,000 | 1653.5 | 2.0424 | 1630.4 | 2.0771 | 1559.0 | 2.1451 | 1494.4 | 2.2372 |
| 17,000 | 1641.7 | 2.0585 | 1641.2 | 2.0767 | 1563.2 | 2.1372 | 1492.8 | 2.2366 |
| 18,000 | 1657.1 | 2.0398 | 1641.2 | 2.0675 | 1547.2 | 2.1496 | 1487.2 | 2.2389 |
| 19,000 | 1664.3 | 2.0203 | 1645.3 | 2.0671 | 1564.2 | 2.1415 | 1501.1 | 2.2209 |
| 20,000 | 1660.2 | 2.0323 | 1636.6 | 2.0730 | 1424.6 | 2.3300 | 1396.3 | 2.3509 |
| 21,000 | 1657.1 | 2.0405 | 1637.6 | 2.0674 | 1371.1 | 2.3888 | 1316.1 | 2.5018 |
| 21,006 |  |  | 1637.6 | 2.0654 |  |  |  |  |
| 22,000 | 1656.1 | 2.0377 |  |  | 1470.8 | 2.2710 | 1415.3 | 2.3491 |
| 23,000 | 1666.9 | 2.0240 |  |  | 1580.1 | 2.1439 | 1511.8 | 2.2216 |
| 24,000 | 1652.5 | 2.0315 |  |  | 1603.7 | 2.1265 | 1530.8 | 2.2022 |
| 25,000 | 1667.4 | 2.0242 |  |  | 1606.3 | 2.1140 | 1533.4 | 2.1996 |
| 26,000 | 1672.1 | 2.0277 |  |  | 1615.0 | 2.1183 | 1540.6 | 2.1914 |
| 27,000 | 1671.0 | 2.0217 |  |  | 1607.8 | 2.1119 | 1538.5 | 2.1900 |
| 28,000 | 1682.3 | 2.0047 |  |  | 1609.9 | 2.1110 | 1547.8 | 2.1868 |
| 29,000 | 1673.6 | 2.0158 |  |  | 1601.1 | 2.1136 | 1512.9 | 2.2264 |
| 30,000 | 1674.1 | 2.0137 |  |  | 1520.6 | 2.1977 | 1439.0 | 2.3287 |
| 31,000 | 1672.6 | 2.0167 |  |  | 1398.9 | 2.3600 | 1343.9 | 2.4109 |
| 32,000 | 1674.6 | 2.0125 |  |  | 1509.3 | 2.1979 | 1414.3 | 2.3335 |
| 33,000 | 1675.1 | 2.0150 |  |  | 1594.0 | 2.1545 | 1481.0 | 2.2526 |
| 34,000 | 1676.7 | 2.0075 |  |  | 1602.7 | 2.1033 | 1512.9 | 2.2121 |
| 35,000 | 1659.7 | 2.0368 |  |  | 1608.8 | 2.0942 | 1529.3 | 2.1955 |
| 36,000 | 1672.1 | 2.0104 |  |  | 1601.1 | 2.1003 | 1526.7 | 2.1871 |
| 37,000 | 1660.7 | 2.0389 |  |  | 1611.4 | 2.0898 | 1520.0 | 2.1958 |
| 38,000 | 1650.5 | 2.0236 |  |  | 1604.7 | 2.0897 | 1528.3 | 2.1833 |
| 39,000 | 1666.4 | 2.0180 |  |  | 1604.2 | 2.0955 | 1481.6 | 2.2317 |
| 40,000 | 1678.2 | 2.0052 |  |  | 1504.6 | 2.2289 | 1461.5 | 2.2836 |
| 41,000 | 1674.1 | 2.0053 |  |  | 1361.9 | 2.4000 | 1358.3 | 2.3552 |
| 42,000 | 1682.9 | 2.0029 |  |  | 1490.8 | 2.2696 | 1403.0 | 2.3347 |
| 43,000 | 1679.3 | 1.9947 |  |  | 1596.5 | 2.1454 | 1508.2 | 2.2266 |
| 44,000 | 1681.3 | 2.0018 |  |  | 1637.6 | 2.0817 | 1536.5 | 2.1866 |
| 45,000 | 1671.0 | 2.0121 |  |  | 1640.7 | 2.0814 | 1534.9 | 2.1848 |
| 45,467 | 1653.5 | 2.0224 |  |  |  |  |  |  |
| 46,000 | 1671.5 | 1.9968 |  |  | 1645.8 | 2.0744 | 1542.1 | 2.1855 |
| 47,000 | 1669.5 | 2.0103 |  |  | 1645.8 | 2.0794 | 1550.3 | 2.1720 |
| 48,000 | 1671.0 | 2.0063 |  |  | 1649.9 | 2.0758 | 1539.5 | 2.1806 |
| 49,000 | 1685.4 | 1.9964 |  |  | 1624.3 | 2.0783 | 1524.7 | 2.1748 |
| 50,000 | 1685.9 | 1.9992 |  |  | 1502.1 | 2.2103 | 1460.0 | 2.2279 |
| 51,000 | 1681.3 | 1.9968 |  |  | 1424.1 | 2.3102 | 1363.9 | 2.3276 |
| 52,000 | 1684.4 | 1.9966 |  |  | 1563.7 | 2.1827 | 1451.8 | 2.2741 |
| 53,000 | 1678.7 | 2.0003 |  |  | 1607.3 | 2.1199 | 1551.3 | 2.1895 |
| 54,000 | 1694.7 | 1.9889 |  |  | 1618.6 | 2.0892 | 1558.0 | 2.1656 |
| 55,000 | 1694.7 | 2.0002 |  |  | 1625.3 | 2.0777 | 1563.2 | 2.1579 |
| 56,000 | 1681.3 | 2.0001 |  |  | 1623.7 | 2.0802 | 1560.1 | 2.1612 |
| 57,000 | 1672.6 | 2.0023 |  |  | 1629.4 | 2.0780 | 1573.9 | 2.1401 |
| 58,000 | 1666.9 | 2.0147 |  |  | 1629.4 | 2.0735 | 1569.3 | 2.1452 |
| 59,000 | 1688.5 | 2.0016 |  |  | 1623.2 | 2.0730 | 1550.3 | 2.1774 |
| 60,000 | 1699.8 | 1.9839 |  |  | 1589.8 | 2.1073 | 1499.0 | 2.2267 |
| 61,000 | 1682.9 | 1.9920 |  |  | 1474.4 | 2.2712 | 1360.3 | 2.3616 |
| 62,000 | 1687.5 | 1.9919 |  |  | 1479.5 | 2.2509 | 1391.2 | 2.3674 |
| 63,000 | 1680.8 | 2.0009 |  |  | 1588.3 | 2.1028 | 1534.9 | 2.1762 |
| 64,000 | 1684.4 | 2.0069 |  |  | 1620.7 | 2.0760 | 1557.0 | 2.1600 |
| 65,000 | 1666.4 | 2.0048 |  |  | 1628.9 | 2.0749 | 1570.3 | 2.1471 |
| 66,000 | 1680.8 | 2.0062 |  |  | 1628.9 | 2.0758 | 1574.4 | 2.1467 |
| 67,000 | 1680.8 | 2.0112 |  |  | 1627.3 | 2.0830 | 1564.7 | 2.1463 |
| 68,000 | 1682.3 | 1.9916 |  |  | 1638.1 | 2.0703 | 1560.1 | 2.1529 |
| 69,000 | 1688.0 | 1.9866 |  |  | 1613.0 | 2.0833 | 1561.1 | 2.1516 |
| 70,000 | 1701.9 | 1.9772 |  |  | 1554.9 | 2.1642 | 1497.5 | 2.2229 |
| 71,000 | 1696.2 | 1.9873 |  |  | 1510.8 | 2.2171 | 1425.6 | 2.3190 |
| 72,000 | 1692.1 | 1.9894 |  |  | 1470.3 | 2.2451 | 1453.8 | 2.2599 |
| 73,000 | 1695.7 | 1.9840 |  |  | 1589.8 | 2.1241 | 1537.0 | 2.1799 |
| 74,000 | 1700.4 | 1.9769 |  |  | 1611.9 | 2.0945 | 1562.6 | 2.1581 |
| 75,000 | 1684.4 | 1.9827 |  |  | 1615.0 | 2.0872 | 1564.7 | 2.1543 |
| 76,000 | 1677.7 | 1.9829 |  |  | 1620.1 | 2.0818 | 1570.3 | 2.1545 |
| 77,000 | 1688.0 | 1.9902 |  |  | 1619.1 | 2.0844 | 1565.7 | 2.1576 |
| 78,000 | 1700.4 | 1.9757 |  |  | 1612.4 | 2.0827 | 1565.2 | 2.1595 |
| 79,000 | 1700.9 | 1.9724 |  |  | 1619.1 | 2.0779 | 1539.0 | 2.1668 |
| 80,000 | 1691.1 | 1.9821 |  |  | 1568.8 | 2.1706 | 1465.1 | 2.2733 |
| 80,518 |  |  |  |  |  |  | 1457.4 | 2.2741 |
| 81,000 | 1689.0 | 1.9754 |  |  | 1519.5 | 2.2294 | 1345.4 | 2.4084 |
| 82,000 | 1701.9 | 1.9711 |  |  | 1542.1 | 2.2070 | 1443.6 | 2.2604 |
| 83,000 | 1707.0 | 1.9696 |  |  | 1623.7 | 2.0914 | 1557.5 | 2.1565 |
| 84,000 | 1687.0 | 1.9881 |  |  | 1628.9 | 2.0804 | 1548.3 | 2.1560 |
| 85,000 | 1687.5 | 2.0015 |  |  | 1638.1 | 2.0729 | 1557.0 | 2.1536 |
| 86,000 | 1691.6 | 1.9707 |  |  | 1639.2 | 2.0723 | 1550.8 | 2.1567 |
| 86,398 |  |  |  |  | 1639.7 | 2.0739 |  |  |
| 87,000 | 1695.2 | 1.9743 |  |  |  |  | 1559.6 | 2.1517 |
| 88,000 | 1688.0 | 1.9935 |  |  |  |  | 1551.3 | 2.1591 |
| 89,000 | 1691.6 | 1.9727 |  |  |  |  | 1544.7 | 2.1805 |
| 90,000 | 1703.4 | 1.9677 |  |  |  |  | 1464.6 | 2.2968 |
| 91,000 | 1702.4 | 1.9680 |  |  |  |  | 1418.9 | 2.3099 |
| 92,000 | 1699.8 | 1.9698 |  |  |  |  | 1444.1 | 2.2890 |
| 93,000 | 1707.0 | 1.9680 |  |  |  |  | 1567.3 | 2.1533 |
| 94,000 | 1704.5 | 1.9685 |  |  |  |  | 1579.6 | 2.1393 |
| 95,000 | 1690.6 | 1.9858 |  |  |  |  | 1581.1 | 2.1386 |
| 96,000 | 1686.5 | 1.9902 |  |  |  |  | 1579.6 | 2.1377 |
| 97,000 | 1675.1 | 1.9851 |  |  |  |  | 1579.6 | 2.1328 |
| 98,000 | 1695.2 | 1.9712 |  |  |  |  | 1572.9 | 2.1386 |
| 99,000 | 1698.3 | 1.9623 |  |  |  |  | 1586.2 | 2.1143 |
| 100,000 | 1707.0 | 1.9658 |  |  |  |  | 1492.3 | 2.2229 |
| 101,000 | 1708.1 | 1.9635 |  |  |  |  | 1428.7 | 2.3111 |
| 102,000 | 1708.1 | 1.9657 |  |  |  |  | 1438.4 | 2.2747 |
| 103,000 | 1702.9 | 1.9648 |  |  |  |  | 1564.7 | 2.1425 |
| 104,000 | 1696.8 | 1.9897 |  |  |  |  | 1581.6 | 2.1379 |
| 105,000 | 1701.9 | 1.9651 |  |  |  |  | 1590.9 | 2.1269 |
| 106,000 | 1706.5 | 1.9752 |  |  |  |  | 1584.2 | 2.1281 |
| 107,000 | 1687.0 | 1.9981 |  |  |  |  | 1589.8 | 2.1257 |
| 108,000 | 1687.5 | 1.9932 |  |  |  |  | 1585.2 | 2.1247 |
| 109,000 | 1701.4 | 1.9644 |  |  |  |  | 1574.4 | 2.1410 |
| 110,000 | 1705.0 | 1.9716 |  |  |  |  | 1546.2 | 2.1779 |
| 111,000 | 1701.9 | 1.9747 |  |  |  |  | 1435.4 | 2.2442 |
| 111,327 |  |  |  |  |  |  | 1431.3 | 2.3216 |
| 112,000 | 1700.4 | 1.9676 |  |  |  |  | 1451.8 | 2.2788 |
| 113,000 | 1717.9 | 1.9563 |  |  |  |  | 1563.7 | 2.1602 |
| 114,000 | 1694.2 | 1.9625 |  |  |  |  | 1577.0 | 2.1534 |
| 115,000 | 1699.3 | 1.9675 |  |  |  |  | 1573.4 | 2.1357 |
| 116,000 | 1687.0 | 1.9949 |  |  |  |  | 1581.1 | 2.1313 |
| 116,369 |  |  |  |  |  |  | 1580.1 | 2.1285 |
| 117,000 | 1702.9 | 1.9774 |  |  |  |  | 1584.2 | 2.1302 |
| 118,000 | 1708.1 | 1.9662 |  |  |  |  | 1586.2 | 2.1210 |
| 119,000 | 1701.4 | 1.9811 |  |  |  |  | 1582.7 | 2.1276 |
| 120,000 | 1726.6 | 1.9449 |  |  |  |  | 1528.8 | 2.2077 |
| 120,754 |  |  |  |  |  |  | 1473.3 | 2.2375 |
| 121,000 | 1721.5 | 1.9541 |  |  |  |  | 1407.6 | 2.3592 |
| 122,000 | 1723.5 | 1.9500 |  |  |  |  | 1448.2 | 2.3031 |
| 123,000 | 1722.0 | 1.9465 |  |  |  |  | 1543.1 | 2.1664 |
| 124,000 | 1709.6 | 1.9605 |  |  |  |  | 1568.3 | 2.1375 |
| 125,000 | 1706.0 | 1.9576 |  |  |  |  | 1564.7 | 2.1346 |
| 126,000 | 1708.6 | 1.9573 |  |  |  |  | 1570.8 | 2.1284 |
| 126,222 | 1718.4 | 1.9361 |  |  |  |  |  |  |
| 127,000 | 1698.3 | 1.9810 |  |  |  |  | 1575.5 | 2.1273 |
| 127,931 | 1716.3 | 1.9634 |  |  |  |  |  |  |
| 128,000 | 1709.1 | 1.9622 |  |  |  |  | 1581.6 | 2.1278 |
| 129,000 | 1735.9 | 1.9305 |  |  |  |  | 1566.2 | 2.1507 |
| 130,000 | 1722.5 | 1.9387 |  |  |  |  | 1514.9 | 2.2362 |
| 131,000 | 1732.3 | 1.9359 |  |  |  |  | 1404.0 | 2.3663 |
| 132,000 | 1720.4 | 1.9414 |  |  |  |  | 1469.8 | 2.2738 |
| 133,000 | 1725.6 | 1.9470 |  |  |  |  | 1585.2 | 2.1363 |
| 134,000 | 1739.0 | 1.9338 |  |  |  |  | 1585.2 | 2.1357 |
| 134,247 | 1713.2 | 1.9567 |  |  |  |  |  |  |
| 135,000 | 1737.4 | 1.9361 |  |  |  |  | 1580.1 | 2.1354 |
| 136,000 | 1710.7 | 1.9565 |  |  |  |  | 1584.2 | 2.1341 |
| 137,000 | 1719.4 | 1.9233 |  |  |  |  | 1583.7 | 2.1357 |
| 138,000 | 1702.4 | 1.9581 |  |  |  |  | 1593.4 | 2.1316 |
| 139,000 | 1728.2 | 1.9393 |  |  |  |  | 1590.4 | 2.1430 |
| 140,000 | 1726.1 | 1.9290 |  |  |  |  | 1510.8 | 2.2288 |
| 141,000 | 1719.9 | 1.9415 |  |  |  |  | 1488.7 | 2.2568 |
| 142,000 | 1719.4 | 1.9353 |  |  |  |  | 1564.7 | 2.1534 |
| 142,084 |  |  |  |  |  |  | 1495.4 | 2.2368 |
| 143,000 | 1724.0 | 1.9385 |  |  |  |  | 1576.0 | 2.1519 |
| 143,756 |  |  |  |  |  |  | 1592.4 | 2.1308 |
| 144,000 | 1730.2 | 1.9433 |  |  |  |  | 1584.7 | 2.1301 |
| 145,000 | 1719.9 | 1.9349 |  |  |  |  | 1591.4 | 2.1312 |
| 146,000 | 1730.2 | 1.9520 |  |  |  |  | 1591.4 | 2.1279 |
| 147,000 | 1708.6 | 1.9610 |  |  |  |  | 1590.4 | 2.1281 |
| 148,000 | 1711.7 | 1.9534 |  |  |  |  | 1590.9 | 2.1239 |
| 149,000 | 1727.1 | 1.9252 |  |  |  |  | 1600.1 | 2.1072 |
| 150,000 | 1727.1 | 1.9355 |  |  |  |  | 1564.2 | 2.1140 |
| 151,000 | 1728.7 | 1.9254 |  |  |  |  | 1486.2 | 2.2976 |
| 152,000 | 1722.0 | 1.9313 |  |  |  |  | 1495.9 | 2.2394 |
| 153,000 | 1726.6 | 1.9434 |  |  |  |  | 1581.6 | 2.1153 |
| 154,000 | 1719.4 | 1.9310 |  |  |  |  | 1613.0 | 2.0906 |
| 155,000 | 1706.5 | 1.9391 |  |  |  |  | 1616.0 | 2.0871 |
| 156,000 | 1699.8 | 1.9635 |  |  |  |  | 1611.4 | 2.0875 |
| 157,000 | 1718.9 | 1.9420 |  |  |  |  | 1619.6 | 2.0874 |
| 158,000 | 1723.0 | 1.9199 |  |  |  |  | 1618.6 | 2.0883 |
| 159,000 | 1729.7 | 1.9298 |  |  |  |  | 1638.6 | 2.0639 |
| 160,000 | 1739.5 | 1.9094 |  |  |  |  | 1563.7 | 2.1315 |
| 161,000 | 1732.3 | 1.9198 |  |  |  |  | 1470.3 | 2.2659 |
| 162,000 | 1726.6 | 1.9426 |  |  |  |  | 1525.7 | 2.1705 |
| 163,000 | 1724.0 | 1.9398 |  |  |  |  | 1602.7 | 2.0821 |
| 164,000 | 1724.0 | 1.9400 |  |  |  |  | 1604.2 | 2.0893 |
| 165,000 | 1731.3 | 1.9288 |  |  |  |  | 1608.3 | 2.0838 |
| 166,000 | 1710.1 | 1.9556 |  |  |  |  | 1613.0 | 2.0846 |
| 167,000 | 1713.2 | 1.9540 |  |  |  |  | 1613.5 | 2.0831 |
| 168,000 | 1727.7 | 1.9439 |  |  |  |  | 1608.8 | 2.0842 |
| 169,000 | 1721.5 | 1.9378 |  |  |  |  | 1598.6 | 2.0863 |
| 170,000 | 1722.0 | 1.9365 |  |  |  |  | 1602.7 | 2.0993 |
| 171,000 | 1718.4 | 1.9431 |  |  |  |  | 1478.5 | 2.2599 |
| 172,000 | 1720.4 | 1.9320 |  |  |  |  | 1515.9 | 2.1680 |
| 173,000 | 1715.3 | 1.9233 |  |  |  |  | 1602.7 | 2.0951 |
| 174,000 | 1719.4 | 1.9224 |  |  |  |  | 1618.1 | 2.0837 |
| 175,000 | 1731.3 | 1.9042 |  |  |  |  | 1622.7 | 2.0771 |
| 176,000 | 1704.5 | 1.9589 |  |  |  |  | 1627.3 | 2.0784 |
| 177,000 | 1713.2 | 1.9352 |  |  |  |  | 1628.4 | 2.0734 |
| 178,000 | 1712.2 | 1.9263 |  |  |  |  | 1627.8 | 2.0768 |
| 179,000 | 1715.3 | 1.9328 |  |  |  |  | 1600.6 | 2.0785 |
| 180,000 | 1717.3 | 1.9297 |  |  |  |  | 1570.8 | 2.1414 |
| 181,000 | 1723.5 | 1.9169 |  |  |  |  | 1455.4 | 2.3264 |
| 182,000 | 1735.9 | 1.9191 |  |  |  |  | 1527.2 | 2.1925 |
| 182,041 |  |  |  |  |  |  | 1510.3 | 2.1991 |
| 183,000 | 1728.7 | 1.9135 |  |  |  |  | 1599.6 | 2.1133 |
| 184,000 | 1722.5 | 1.9348 |  |  |  |  | 1614.0 | 2.0900 |
| 185,000 | 1723.0 | 1.9394 |  |  |  |  | 1625.8 | 2.0830 |
| 186,000 | 1730.7 | 1.9340 |  |  |  |  | 1621.7 | 2.0861 |
| 187,000 | 1717.9 | 1.9427 |  |  |  |  | 1617.6 | 2.0947 |
| 188,000 | 1719.9 | 1.9339 |  |  |  |  | 1616.5 | 2.0873 |
| 189,000 | 1732.3 | 1.9198 |  |  |  |  | 1610.4 | 2.0970 |
| 190,000 | 1718.9 | 1.9281 |  |  |  |  | 1594.5 | 2.1126 |
| 191,000 | 1737.4 | 1.9137 |  |  |  |  | 1461.0 | 2.2782 |
| 192,000 | 1728.7 | 1.9216 |  |  |  |  | 1491.8 | 2.2220 |
| 193,000 | 1719.4 | 1.9289 |  |  |  |  | 1604.2 | 2.0767 |
| 194,000 | 1724.6 | 1.9342 |  |  |  |  | 1611.9 | 2.0756 |
| 194,481 | 1712.7 | 1.9546 |  |  |  |  |  |  |
| 195,000 | 1717.3 | 1.9433 |  |  |  |  | 1613.0 | 2.0758 |
| 196,000 | 1724.0 | 1.9383 |  |  |  |  | 1616.5 | 2.0711 |
| 196,491 |  |  |  |  |  |  | 1614.5 | 2.0745 |
| 197,000 | 1725.1 | 1.9372 |  |  |  |  |  |  |
| 198,000 | 1714.3 | 1.9424 |  |  |  |  |  |  |
| 199,000 | 1715.3 | 1.9374 |  |  |  |  |  |  |
| 200,000 | 1735.4 | 1.9155 |  |  |  |  |  |  |
| 201,000 | 1735.9 | 1.9126 |  |  |  |  |  |  |
| 202,000 | 1747.8 | 1.9059 |  |  |  |  |  |  |
| 203,000 | 1739.0 | 1.9053 |  |  |  |  |  |  |
| 204,000 | 1731.3 | 1.9313 |  |  |  |  |  |  |
| 205,000 | 1730.2 | 1.9268 |  |  |  |  |  |  |
| 206,000 | 1710.1 | 1.9447 |  |  |  |  |  |  |
| 207,000 | 1730.7 | 1.9306 |  |  |  |  |  |  |
| 208,000 | 1723.0 | 1.9375 |  |  |  |  |  |  |
| 209,000 | 1738.5 | 1.9277 |  |  |  |  |  |  |
| 210,000 | 1744.1 | 1.9149 |  |  |  |  |  |  |
| 211,000 | 1720.4 | 1.9335 |  |  |  |  |  |  |
| 212,000 | 1734.4 | 1.9154 |  |  |  |  |  |  |
| 213,000 | 1735.4 | 1.9279 |  |  |  |  |  |  |
| 214,000 | 1741.1 | 1.9009 |  |  |  |  |  |  |
| 215,000 | 1741.1 | 1.9204 |  |  |  |  |  |  |
| 216,000 | 1692.1 | 1.9873 |  |  |  |  |  |  |
| 217,000 | 1712.7 | 1.9508 |  |  |  |  |  |  |
| 218,000 | 1729.2 | 1.9360 |  |  |  |  |  |  |
| 219,000 | 1733.3 | 1.9349 |  |  |  |  |  |  |
| 220,000 | 1740.5 | 1.9221 |  |  |  |  |  |  |
| 221,000 | 1737.4 | 1.9250 |  |  |  |  |  |  |
| 222,000 | 1737.4 | 1.9302 |  |  |  |  |  |  |
| 223,000 | 1738.0 | 1.9298 |  |  |  |  |  |  |
| 224,000 | 1749.3 | 1.8955 |  |  |  |  |  |  |
| 225,000 | 1718.4 | 1.9194 |  |  |  |  |  |  |
| 226,000 | 1728.7 | 1.9175 |  |  |  |  |  |  |
| 227,000 | 1732.8 | 1.9298 |  |  |  |  |  |  |
| 228,000 | 1749.3 | 1.9017 |  |  |  |  |  |  |
| 229,000 | 1740.0 | 1.9059 |  |  |  |  |  |  |
| 230,000 | 1745.2 | 1.9124 |  |  |  |  |  |  |
| 231,000 | 1743.1 | 1.9077 |  |  |  |  |  |  |
| 232,000 | 1743.1 | 1.9051 |  |  |  |  |  |  |
| 233,000 | 1746.7 | 1.9072 |  |  |  |  |  |  |
| 234,000 | 1740.5 | 1.9110 |  |  |  |  |  |  |
| 234,240 | 1766.8 | 1.8909 |  |  |  |  |  |  |
| 234,700 | 1717.3 | 1.9395 |  |  |  |  |  |  |
| 235,000 | 1758.6 | 1.8943 |  |  |  |  |  |  |
| 236,000 | 1722.5 | 1.9259 |  |  |  |  |  |  |
| 237,000 | 1709.6 | 1.9623 |  |  |  |  |  |  |
| 238,000 | 1723.5 | 1.9273 |  |  |  |  |  |  |
| 239,000 | 1745.2 | 1.8957 |  |  |  |  |  |  |
| 239,101 | 1740.0 | 1.9165 |  |  |  |  |  |  |
| 240,000 | 1740.5 | 1.9129 |  |  |  |  |  |  |
| 241,000 | 1740.0 | 1.9139 |  |  |  |  |  |  |
| 242,000 | 1745.7 | 1.9094 |  |  |  |  |  |  |
| 243,000 | 1747.2 | 1.9069 |  |  |  |  |  |  |
| 244,000 | 1753.4 | 1.8893 |  |  |  |  |  |  |
| 245,000 | 1728.7 | 1.9305 |  |  |  |  |  |  |
| 246,000 | 1726.6 | 1.9183 |  |  |  |  |  |  |
| 247,000 | 1734.9 | 1.9150 |  |  |  |  |  |  |
| 248,000 | 1726.6 | 1.9245 |  |  |  |  |  |  |
| 249,000 | 1750.3 | 1.9107 |  |  |  |  |  |  |
| 250,000 | 1753.4 | 1.8889 |  |  |  |  |  |  |
| 251,000 | 1743.1 | 1.9077 |  |  |  |  |  |  |
| 252,000 | 1751.9 | 1.9029 |  |  |  |  |  |  |
| 253,000 | 1747.2 | 1.9098 |  |  |  |  |  |  |
| 254,000 | 1721.0 | 1.9266 |  |  |  |  |  |  |
| 255,000 | 1722.5 | 1.9373 |  |  |  |  |  |  |
| 256,000 | 1705.5 | 1.9538 |  |  |  |  |  |  |
| 257,000 | 1746.2 | 1.9106 |  |  |  |  |  |  |
| 258,000 | 1729.2 | 1.9279 |  |  |  |  |  |  |
| 259,000 | 1743.6 | 1.9239 |  |  |  |  |  |  |
| 260,000 | 1758.6 | 1.8974 |  |  |  |  |  |  |
| 260,410 | 1746.7 | 1.9187 |  |  |  |  |  |  |
| 261,000 | 1759.6 | 1.8945 |  |  |  |  |  |  |
| 262,000 | 1756.0 | 1.9039 |  |  |  |  |  |  |
| 262,044 | 1760.1 | 1.9026 |  |  |  |  |  |  |
| 263,000 | 1759.1 | 1.8951 |  |  |  |  |  |  |
| 264,000 | 1749.3 | 1.9043 |  |  |  |  |  |  |
| 265,000 | 1747.2 | 1.8994 |  |  |  |  |  |  |
| 266,000 | 1749.3 | 1.8997 |  |  |  |  |  |  |
| 267,000 | 1753.4 | 1.8891 |  |  |  |  |  |  |
| 268,000 | 1738.5 | 1.9241 |  |  |  |  |  |  |
| 269,000 | 1774.1 | 1.8702 |  |  |  |  |  |  |
| 270,000 | 1763.2 | 1.8979 |  |  |  |  |  |  |
| 271,000 | 1757.0 | 1.8975 |  |  |  |  |  |  |
| 272,000 | 1755.5 | 1.9022 |  |  |  |  |  |  |
| 273,000 | 1752.9 | 1.8909 |  |  |  |  |  |  |
| 274,000 | 1756.0 | 1.9110 |  |  |  |  |  |  |
| 275,000 | 1732.8 | 1.9311 |  |  |  |  |  |  |
| 276,000 | 1725.1 | 1.9407 |  |  |  |  |  |  |
| 277,000 | 1725.6 | 1.9359 |  |  |  |  |  |  |
| 278,000 | 1752.4 | 1.8921 |  |  |  |  |  |  |
| 279,000 | 1758.1 | 1.8763 |  |  |  |  |  |  |
| 280,000 | 1767.9 | 1.8863 |  |  |  |  |  |  |
| 281,000 | 1752.4 | 1.8995 |  |  |  |  |  |  |
| 282,000 | 1745.7 | 1.8967 |  |  |  |  |  |  |
| 283,000 | 1761.2 | 1.8834 |  |  |  |  |  |  |
| 284,000 | 1763.2 | 1.8908 |  |  |  |  |  |  |
| 285,000 | 1737.4 | 1.9050 |  |  |  |  |  |  |
| 286,000 | 1725.1 | 1.9031 |  |  |  |  |  |  |
| 287,000 | 1749.8 | 1.8953 |  |  |  |  |  |  |
| 288,000 | 1767.4 | 1.8693 |  |  |  |  |  |  |
| 289,000 | 1769.4 | 1.8890 |  |  |  |  |  |  |
| 290,000 | 1762.2 | 1.8829 |  |  |  |  |  |  |
| 291,000 | 1786.0 | 1.8695 |  |  |  |  |  |  |
| 292,000 | 1775.1 | 1.8803 |  |  |  |  |  |  |
| 293,000 | 1771.0 | 1.8765 |  |  |  |  |  |  |
| 294,000 | 1771.0 | 1.8844 |  |  |  |  |  |  |
| 295,000 | 1732.3 | 1.9227 |  |  |  |  |  |  |
| 296,000 | 1772.0 | 1.8653 |  |  |  |  |  |  |
| 297,000 | 1732.8 | 1.9308 |  |  |  |  |  |  |
| 298,000 | 1779.2 | 1.8803 |  |  |  |  |  |  |
| 299,000 | 1758.6 | 1.8855 |  |  |  |  |  |  |
| 300,000 | 1757.6 | 1.8992 |  |  |  |  |  |  |
| 300,300 | 1772.5 | 1.8827 |  |  |  |  |  |  |
| 301,000 | 1760.1 | 1.8888 |  |  |  |  |  |  |
| 302,000 | 1777.2 | 1.8772 |  |  |  |  |  |  |
| 303,000 | 1761.7 | 1.8907 |  |  |  |  |  |  |
| 304,000 | 1765.8 | 1.8906 |  |  |  |  |  |  |
| 305,000 | 1778.7 | 1.8937 |  |  |  |  |  |  |
| 306,000 | 1739.5 | 1.9199 |  |  |  |  |  |  |
| 307,000 | 1716.3 | 1.9257 |  |  |  |  |  |  |
| 308,000 | 1761.2 | 1.8987 |  |  |  |  |  |  |
| 309,000 | 1776.7 | 1.8809 |  |  |  |  |  |  |
| 310,000 | 1767.4 | 1.8891 |  |  |  |  |  |  |
| 311,000 | 1768.9 | 1.8937 |  |  |  |  |  |  |
| 312,000 | 1772.0 | 1.8939 |  |  |  |  |  |  |
| 313,000 | 1775.1 | 1.8810 |  |  |  |  |  |  |
| 314,000 | 1751.4 | 1.9147 |  |  |  |  |  |  |
| 314,767 | 1776.7 | 1.8683 |  |  |  |  |  |  |

</details>

Training losses (mean of the step lines in the last 10,000 steps before each point; pD = mean predicted draw
probability):

| run, window | step lines | pLoss | vLoss | pEnt | pD |
|---|---:|---:|---:|---:|---:|
| R-elitecyc, end of epoch 0 (117,931–127,931) | 133 | 2.457 | 0.826 | 2.803 | 0.180 |
| R-elitecyc, end of epoch 1 (266,950–276,950) | 131 | 2.444 | 0.819 | 2.791 | 0.177 |
| R-elitecyc, last 10,000 (304,767–314,767; epoch 2, before corpus game 3,812,602) | 119 | 2.505 | 0.820 | 2.784 | 0.141 |
| R-elitefresh, end of epoch 0 (138,970–148,970) | 141 | 2.474 | 0.816 | 2.831 | 0.176 |
| R-elitefresh, last 10,000 (186,491–196,491) | 119 | 2.534 | 0.817 | 2.825 | 0.132 |
| R-replay, last 10,000 (76,398–86,398; `w3aA5b`) | 158 | 2.689 | 0.747 | 2.858 | 0.043 |

The last-10,000 windows sample a different part of the corpus (the early shards) than the epoch ends (the late
shards), so the epoch-end rows are the like-for-like pair.

Training losses in epoch 0 (step 1,000 → 120,000): pLoss 2.624 → 2.506, vLoss 0.861 → 0.841, pEnt 2.868 → 2.804.

### Health

Every segment of both runs ran the health monitor with every rule's action set to log (`alarm_config`), so no alarm
stopped either run.

R-elitecyc, epoch 0 (segments 0–2):

- Gradient: 0 clips in 1,438 step lines; largest `gNormMax` 0.742.
- No GPU faults (`gpu_faults.faults` empty in all three results files), no non-finite values, 0 parked or mostly-off
  channels and 0 of 128 value FC1 units at zero velocity at all 130 checkpoints; 0–7 value FC1 units at low velocity.
- BN running-variance max/median: 278.2 at step 1,000 (inherited; `blocks.2.bn1[13]`, 37 channels ≥ 10× their site
  median) → 415.3 at the final checkpoint (same channel, 40 channels); range over the checkpoints 271.9–424.4.
  `bn_running_variance_runaway` was not raised.
- Only alarm: `policy_offset_drift` (warning, action log), raised at the first check of each segment (trainer steps
  100, 45,550, 126,300) and active throughout. Its median |policy logit mean| was 10.40 at step 100, so it came in
  with the start weights; 9.36 at 127,000.

R-elitecyc, segments 3–11 (127,931 → 314,767; segment 3 from its results file in git at `8747714d`):

- Gradient: 0 clips in 1,903 step lines; largest `gNormMax` 0.898 (step 256,160).
- No GPU faults (`gpu_faults.faults` and `crash_dumps` empty in all nine results files, monitor available).
- 196 checkpoints (128,000 → 314,767): 0 non-finite values, 0 parked or mostly-off channels, 0 of 128 value FC1
  units at zero velocity; 1–10 value FC1 units at low velocity (3 at the final checkpoint).
- BN running-variance max/median: always `blocks.2.bn1[13]` (the channel of section 4); 417.2 at 128,000, range
  394.4 (166,000) – 512.6 (284,000), 465.9 at 314,767. Channels ≥ 10× their site median 35–44 (41 at the end).
  `bn_running_variance_runaway` (threshold 1,000) was not raised.
- Only alarm: `policy_offset_drift` (warning), raised at the first check of each segment and active throughout;
  median |policy logit mean| 9.354 at 128,000 → 9.038 at 314,000.

R-elitefresh, segments 0–7 (0 → 196,491):

- Gradient: 3 clips, all in the first 50 steps (`gNormMax` 24.78 at step 1, 26.10 in the line at step 50); after
  step 1,000 the largest `gNormMax` was 1.148 (step 161,910, during the variance jump below). 2,528 step lines.
- **One GPU fault**, in the attempt after segment 2 (`results-elitefresh-seg3.json`, `termination_reason:
  gpu_fault`): a training-step submission failed with `Insufficient Memory
  (00000008:kIOGPUCommandBufferCallbackErrorOutOfMemory)` (queue `ChessTrainer.net(init)`, batch 4,096, 37,439 ms on
  the GPU) at trainer step 116,821, 2026-10-17 19:09:33 CDT; the system log had the same error 38 s earlier. Training
  stopped with no further save; crash dump
  `CrashDumps/20261017-190933-20261017-3-it0A-step116820-gpu-fault.dcmcrash`. No weights written after the fault
  entered the lineage (segment 3 resumed from the 116,369 save). No other segment of either run had a fault.
- 204 checkpoints: 0 non-finite values, 0 parked or mostly-off channels, 0 value FC1 units at zero velocity; 0–9 at
  low velocity (6 at the end).
- **BN running-variance max/median ratcheted for the whole run**: 47.9 at 1,000 (`stem.bn`), 131.0 at 40,000
  (`blocks.1.bn1[13]`, the worst site from then on), 299.6 at 80,000, 803.5 at 120,000, 1,553.5 at 160,000,
  1,969.6 at 196,491 (ZlrA ended at 1,654.7). Channels ≥ 10× their site median: 2 → 88.
- `bn_running_variance_runaway` (≥ 1,000): raised at 132,500 (1,009.6, `blocks.1.bn1`), re-raised at the start of
  each later segment, active to the end (1,968.0 at 196,000).
- `bn_running_variance_jump` (one channel's running variance up ≥ 10×; critical at ratio ≥ 100), three episodes:
  - 110,200 warning → 110,550 critical: `blocks.2.bn1[58]` 0.96 → 20.7, then `blocks.2.bn1[22]` 0.28 → 135.9;
    active (critical) at 111,000 when segment 1 stopped; not raised again in segment 2.
  - 131,100 warning: `blocks.2.bn1[23]` 0.86 → 16.1; cleared 132,100.
  - 161,900 warning → 161,950 critical: `blocks.2.bn1[30]` 1.08 → 122.5 (221.4 at 162,750); cleared 162,950.
- `policy_offset_drift` (warning): raised at 1,300 (median |policy logit mean| 3.28), active throughout; 10.73 at
  80,000, 11.65 at 196,000.

### Interpretation

- Feeding the elite corpus alone did not move R-replay's network (R-elite flat at ~1,635 for 21k steps at a fixed
  0.002). The same corpus with a 0.002–0.02 cycle gained about 80 pElo over 128k steps and was still rising slowly at
  the end of the epoch (cycle bests 1,659 → 1,682 → 1,700 → 1,707 → 1,727). One run, one seed: the LR cycle and the
  extra 128k steps are not separated (R-elite at a fixed LR was stopped at 21k).
- The 10× cycle cost ~18 pElo at its peaks; the 1,000× cycle (0.001–1.0) cost ~194.
- From scratch, the elite corpus trained slower than `w3aA5b` on the same 0.001–1.0 schedule, seed and start:
  R-elitefresh was below R-replay at all 86 shared probe steps (mean −72.2).

What the extra epochs bought (segments 3–11):

- R-elitecyc's second pass (epoch 1) lifted every summary: best 1,726.6 → 1,774.1, lowest NLL 1.9361 → 1.8702,
  mean of the last 10 probes 1,715.1 → 1,750.9. The first 26% of epoch 2 added a little more: best 1,786.0 at
  291,000, lowest NLL 1.8653 at 296,000, last-10 mean 1,760.5. The 10,000-step window bests rose from ~1,736 (130k–
  200k) to 1,786.0 (300k); the rise is slowing, not stopped.
- The gain per pass shrank: epoch 0 +78.7 over the start file (end-of-epoch save vs start), epoch 1 +~47 (best to
  best), epoch 2 so far +~12 (best to best). One run, one seed.
- The 0.002–0.02 peaks cost more later: 29.6 pElo on average over epochs 1–2 vs 18 in epoch 0 (range 0–53).
- R-elitefresh also gained in its second pass (best 1,593.4 → 1,638.6), but stayed at R-replay's level
  (~1,640) after 196k steps — about 2.3× R-replay's 86k steps; its final is 162.2 pElo below R-elitecyc's. Its 1.0-peak
  cycle still cost 156 pElo at the peaks as the peak LR decayed to 0.27.
- R-elitefresh's BN running-variance ratio climbed for the whole run to 1,969.6 (`blocks.1.bn1[13]`), past ZlrA's
  1,654.7, with three single-channel variance jumps; R-elitecyc, on the 0.002–0.02 cycle, held 394–513 on the same
  channel index at `blocks.2.bn1`. Neither run parked a channel or produced a non-finite value. Whether the ratchet
  costs pElo is not shown (no controlled comparison).
- Ranking at the 2026-10-20 stop: R-elitecyc 1,776.7 (best 1,786.0) ≫ R-replay 1,639.7 (best 1,649.9) ≈
  R-elitefresh 1,614.5 (best 1,638.6) ≈ R-elite 1,637.6. The trained-network-plus-gentle-cycle path is the only
  one that moved past ~1,650; the cycle, the extra steps and the starting network are not separated.

### Reproduce

Corpus `20260704-215145-op24Gp` (Lichess Elite Database Dec 2021–Nov 2025, 14,619,153 games, 42 shards; imported
from the PGNs, not in the repo). Parameters `parameters-zlra-elite-gentlecycle.json`. Builds: frozen
`FrozenBuilds/DCM-2503-7d945ec7.app` (R-elitecyc segment 0) and `FrozenBuilds/DCM-2523-a242b7ad.app` (R-elitecyc
segments 1–12, every R-elitefresh segment). R-elitefresh: parameters `parameters-zlra.json`, start
`20261008-zlra-ab-fresh.safetensors` (section 2's Reproduce mints it), `--seed 20261008`. Test-set numbers are read
from the checkpoints with `DrewsChessMachine --describe-model <file>…` (build 2614, `DCM-2614-89ca9b6b.app`, header
only).

The start file is R-elite's SIGINT save, `20261011-zlra-elite-latest-step21006.safetensors` (model ID
`20261011-4-btGW`, content sha256 `63747de55e8316286ae883b966bc9a0d13f983c58baef9e421543e232965127e`), itself a
branch of R-replay's SIGINT save at 86,398 (`20261011-2-abM2`, content sha256
`5795606d94b05c2265e5aab6702dab3b8820ffa89eae0b9881909396627a6abd`). Both ends were set by SIGINT, not a step
limit. Regenerating them: R-replay's `launch/replay_launch.sh` (0 → 60,000), `replay_resume1_launch.sh` (from
60,000; its last save before the 2026-10-10 GPU reset was 67,000) and `replay_resume2_launch.sh` (from 67,000) with
`--training-step-limit 19398` in place of 33000; then `launch/replay_elite_fork_launch.sh` with
`--training-step-limit 21006`.

```sh
E=experiments/20261008-zlra-selfplay-lr
$E/launch/elite_gentlecycle_branch_launch.sh     # segment 0 (stopped by SIGINT at 45,467)
$E/launch/elitecyc_resume1_launch.sh             # segment 1, exact resume (stopped by SIGINT at 126,222)
$E/launch/elitecyc_resume2_launch.sh             # segment 2, exact resume to the end of epoch 0
$E/launch/elitecyc_resume3_launch.sh             # segment 3, --epochs 3 (stopped by SIGINT at 134,247)
$E/launch/elitecyc_resume3b_launch.sh            # segment 4, from -step134247 (SIGINT at 194,481)
$E/launch/elitecyc_resume4_launch.sh             # segment 5 (SIGINT at 234,240)
$E/launch/elitecyc_resume5_launch.sh             # segment 6 (SIGINT at 234,700)
$E/launch/elitecyc_resume6_launch.sh             # segment 7 (SIGINT at 239,101)
$E/launch/elitecyc_resume7_launch.sh             # segment 8 (SIGINT at 260,410)
$E/launch/elitecyc_resume8_launch.sh             # segment 9 (SIGINT at 262,044)
$E/launch/elitecyc_resume9_launch.sh             # segment 10 (SIGINT at 300,300)
$E/launch/elitecyc_resume10_launch.sh            # segment 11 (SIGINT at 314,767, 2026-10-20 07:30)
$E/launch/elitecyc_resume11_launch.sh            # segment 12 (running since 2026-10-23)

$E/launch/elitefresh_launch.sh                   # R-elitefresh segment 0 (SIGINT at 80,518)
$E/launch/elitefresh_resume1_launch.sh           # segment 1 (SIGINT at 111,327)
$E/launch/elitefresh_resume2_launch.sh           # segment 2, --epochs 3 from here on (SIGINT at 116,369)
$E/launch/elitefresh_resume3_launch.sh           # GPU out-of-memory fault at 116,821, nothing saved
$E/launch/elitefresh_resume4_launch.sh           # segment 3, from 116,369 again (SIGINT at 120,754)
$E/launch/elitefresh_resume5_launch.sh           # segment 4 (SIGINT at 142,084)
$E/launch/elitefresh_resume6_launch.sh           # segment 5 (SIGINT at 143,756)
$E/launch/elitefresh_resume7_launch.sh           # segment 6 (SIGINT at 182,041)
$E/launch/elitefresh_resume8_launch.sh           # segment 7 (SIGINT at 196,491, 2026-10-20 07:30)
$E/launch/elitefresh_resume9_launch.sh           # segment 8 (running since 2026-10-23)
```

The resume launchers from `elitecyc_resume4` / `elitefresh_resume2` on pick the run's highest-step enumerated
checkpoint, so each must run in order (a later script run first would resume from whatever is newest). Their header
comments were copied from one segment to the next and are stale: every `elitecyc_resume4`–`10` comment describes
the 2026-10-16 pause, every `elitefresh_resume2`–`8` comment the 111,327 pause, and the elitefresh ones call
`parameters-zlra.json` "fixed LR" (it is the 0.001–1.0 cycle). The commands themselves are correct.

Without the interruptions, segment 0 alone runs to the end of epoch 0 (`--epochs 1`), and one `--epochs 3` resume
from 127,931 runs to the end of epoch 2; the exact resumes reproduce
the uninterrupted run's feed and random streams (determinism plan D-1), weights bit for bit only where MPSGraph is
deterministic.

## 4. Numerics: single-channel residual-stream concentration (2026-10-16)

A full static-numerics + puzzle-probe analysis of the live R-elitecyc net, with an architecture-matched
independent-seed control (ZlrA), to understand why one or two channels carry far more magnitude (and gradient)
than the rest.

### Nets compared

Identical architecture (8,271,279 parameters; basic24; stem 128×5×5; 3 blocks 9×9+9×9 @128; no SE; SiLU
pre-activation; clean-add skip; `out:layer_norm`; intermediate_conv policy head; WDL value head; bf16 compute,
`fp32_from_pre_bn` tail), two lineages with **no shared history**:

- **R-elitecyc** (`E709F4EB`), trainer step 167,000, model `20261016-2-qGIj` — the strong run.
- **ZlrA** (`716F996A`), trainer step 51,154 (its final `-sigusr2` trainer file) — the failed GUI seed from §1.

A scan of all 5,067 model files in `Models/` and 160 in `Sessions/` found exactly one matched lineage independent
of R-elitecyc: ZlrA. Every other same-architecture run (R-replay, R-elite, R-elitefresh, R-fixedlr/peGM) branches
from the `FCFD86A9` zlra-ab-fresh seed, so none of those is an independent control.

| | R-elitecyc (167k) | ZlrA (51k) |
|---|---|---|
| Wide-set pElo | **1713** (peak 1739 @160k) | **506** (seed ≈546) |
| lichess-200 pElo | 1425 | 727 (n=200, noisy) |
| Wide top-1 | 2373/4435 = 54% | 240/4435 = 5.4% |
| Wide avg rank | 2.48 | 15.4 (≈ uniform) |
| HangingPiece | 88% | 16% |
| Policy logit abs-max | ~29 | ~47 |
| rv max/median (worst bn1 site) | 398× | **1656×** |
| rv ≥10×median channels | 39 | 75 |
| policy final offset ÷ init | 4.6× | 5.05× |
| `policy.pre_bn` max β/|γ| | +2.0 | **+101** |

ZlrA never learned a policy (avg rank ≈ uniform; fails even hanging-piece). It promoted 19 times climbing 546→~506;
the promotions were noise.

### Finding: this architecture elects one trunk channel as a high-amplitude material carrier

Both independent seeds spontaneously funnel magnitude into **one trunk channel** — channel **13** in R-elitecyc,
channel **25** in ZlrA. Because the channel index differs by seed, it is an emergent training outcome, not a fixed
wiring defect. In R-elitecyc the hot channel shows runaway batch-norm running-variance at every block input
(rv max/median 13× → 206× → 398× at `blocks.0/1/2.bn1[13]`, 32× at `tower_final_bn[13]`).

The mechanism is the clean-add skip, shown directly (not inferred) from per-position activations of the net's own
forward pass over the 4,635 probe positions (numpy fp32 forward validated against the app: top-1 match 98.83%,
valueWin MAE 0.006). Channel-13 trunk value (mean over the 64 squares), per position:

- **block0 injects it:** +0.96 → +8.62 (the block-0 residual branch writes the large DC onto ch13).
- **the skip carries it:** block1→block2 Pearson **0.926**, net increment −0.02 (std 0.32) on a carried level
  of +8.6 — block 1 passes ch13 to block 2 essentially untouched, which is exactly the identity term in
  `clean_add`; `res_ln` (channel-wise LayerNorm on the sum) does not rein it in.
- **tower_final_bn re-centers it:** +8.6 → +4.8.

Channel 13 co-varies with board "fullness"/material (max-abs over squares vs total_pawns Pearson +0.38,
piece_total +0.33, legal_count +0.33, in_check −0.32, endgame-phase −0.33); it is modulated, not collapsed (the
modulation rides on a large constant offset, so its coefficient of variation is small, ~0.047 deep in the trunk).

What game-state attribute confers the amplitude was settled by a single-attribute **intervention** on 350 quiet probe
positions (perturb one thing, hold the rest; `analysis/intervene_ch13.py`), because the observational correlates are
collinear and a standardized regression misleadingly credited `total_pawns` (β +0.45). Causally:

- **Per-unit material is a weak, non-specific driver.** Removing a pawn (Δ −0.063) and removing a minor piece
  (Δ −0.073, piece-count drop matched) move ch13 by the same tiny amount (<1% of base, statistically
  indistinguishable). So the pawn/piece counts are collinear proxies for aggregate board fullness/phase, not a
  per-pawn detector; each scalar count explains only ~19% of ch13's across-position variance (one-way η²), and
  rearranging a pawn at fixed count moves ch13 as much as removing one — ~80% of the variance is configuration.
- **Side-to-move in check is the one strong, clean causal lever, and it suppresses ch13.** Imposing check with
  material and pawn structure held identical drops ch13 by −0.54 (~7% of base, 90% consistent), confirmed on an
  independent material-matched not-in-check subset (−0.48, 91%) — ~8× any material perturbation.

So ch13's high amplitude is conferred mainly by aggregate board configuration/phase ("fullness", which the pawn and
piece counts proxy) and is causally suppressed when the side to move is in check; it is not a per-pawn counter.

### The `|mean|/std` "degraded" flags are the opposite — near-constant channels, not gradient hogs

R-elitecyc's other two flagged channels (`blocks.1.bn1[25]` |mean|/std 16.6, `blocks.2.bn1[48]` 49.0) are
low-amplitude (5–10× below ch13) and weakly feature-correlated (|r| ≤ 0.33). They do **not** carry forward
(block1→block2 Pearson −0.02 / +0.20); each block's branch bias sets their level (ch48's −2.06 increment pins it at
the −2.573 constant, std 0.013 across all 4,635 positions). These are saturated bias-like channels — low gradient,
a cosmetic `|mean|/std` flag, a different mechanism than ch13's residual carry.

### Takeaway

The residual single-channel magnitude concentration is a **universal behaviour of this clean-add 3-block
architecture**, present in both independent seeds. It is benign at ~398× in the net that learned and pathological at
~1656× (plus a +101 always-on policy-head channel and logits ~47) in the net that blew up. So it is a
learning-rate / optimisation / seed sensitivity, not a design defect: this topology *will* pick one channel to carry
trunk magnitude, and keeping that concentration moderate is part of what "training worked" means here. If a future
change wants to damp it, the levers are the skip-merge style (clean-add vs a normalised/projected merge), per-channel
output normalisation, or the LR schedule — not removing the feature, which is doing real work (material sensing).

### Reproduce

Model files (immutable checkpoints): R-elitecyc `Models/20261011-zlra-elitecyc-latest-step167000.safetensors`
(`20261016-2-qGIj`); ZlrA `Sessions/20261008-144140-20261007-61-FNz0-sigusr2.dcmsession/trainer.safetensors`
(`20261007-60-ZlrA-20`). Build `FrozenBuilds/DCM-2566-8747714d.app`.

```sh
APP=".../FrozenBuilds/DCM-2566-8747714d.app/Contents/MacOS/DrewsChessMachine"
"$APP" --analyze-numerics <file> --numerics-static-only --numerics-out <dir>
"$APP" --probe-model <file> --probe-set both --probe-out <positions.jsonl>
```

`analysis/analyze_channels.py` reimplements the forward pass in numpy (fp32), reconstructs the probe FENs from the
bundled `lichess_probes_200.json` / `lichess_probes_wide.json`, self-checks against the app's recorded top-1 and
W/D/L, then captures channel 13/25/48 activations at the four taps and correlates them with board features
(needs python-chess). Its outputs: `analysis/channel_feature_correlations.json` (self-check, per-tap stds, full
Pearson+Spearman tables) and `analysis/residual_carry_forward.json` (the carry-forward correlations above).
`analysis/intervene_ch13.py` runs the causal intervention (remove pawn / remove minor / add pawn / impose check /
move pawn at fixed count) and writes `analysis/intervention_results.json`.
