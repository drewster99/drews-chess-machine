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

## 3. R-elitecyc (2026-10-11 → 2026-10-14; finished its epoch)

### What it is

A corpus-replay continuation of R-replay on the Lichess Elite corpus with a gentle LR cycle. Two steps led to it:

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
  (`--start-game-index 2162743`), so R-elite + R-elitecyc together fed the whole corpus once.

Every R-elitecyc step number below is its own trainer step. At its step 0 the weights already carried 107,404 steps
(86,398 R-replay + 21,006 R-elite).

| segment | launch | build | trainer steps | wall (CDT) | ended | stdout / results |
|---|---|---|---|---|---|---|
| 0 | `launch/elite_gentlecycle_branch_launch.sh` (branch from `20261011-4-btGW`) | 2503, git `7d945ec7` | 0 → 45,467 | 10-11 21:39 → 10-12 15:25 | SIGINT abort save (rebuild for LR keyframes) | `train-elitecyc.stdout`, `results-elitecyc.json` |
| 1 | `launch/elitecyc_resume1_launch.sh` (`--resume-exact`) | 2523, git `a242b7ad` | 45,467 → 126,222 | 10-12 15:30 → 10-14 19:57 | SIGINT abort save (macOS update) | `train-elitecyc-seg1.stdout`, `results-elitecyc-seg1.json` |
| 2 | `launch/elitecyc_resume2_launch.sh` (`--resume-exact`) | 2523, git `a242b7ad` | 126,222 → 127,931 | 10-14 20:11 → 21:24 | end of epoch 1 (final save, `nextGame=0 epoch=1`) | `train-elitecyc-seg2.stdout`, `results-elitecyc-seg2.json` |

- Lineage run `E709F4EB-1D95-4166-90FA-F0D31002BB89`; seed `16308727193092630423` (drawn at segment 0, continued by
  both resumes). Both resumes logged `[RESUME] EXACT` (segment 1: build changed, behavior fingerprint matches;
  segment 2: OS 26B5091g → 26B5112a, fingerprint matches; no `--accept-inexact` needed).
- Final file `20261011-zlra-elitecyc-latest-step127931.safetensors` (also the rolling `…-latest`), model ID
  `20261015-1-KXNZ`. 130 enumerated checkpoints (every 1,000 steps plus 45,467, 126,222, 127,931).
- Trainer time 246,502 s (68.5 h), wall 257,091 s (71.4 h) summed over the segments (lineage `time`). Games fed
  2,162,743 → 14,619,153 (end of corpus), 12,456,410 games.
- `results-elitecyc-seg2.json` records `termination_reason: step_limit_reached` although no step limit was set
  (`stepLimit=none epochLimit=1`); the log shows the run ended at the end of epoch 1 (`[REPLAY] done: … epochs=1`).

### Results

Wide set (4,435 Lichess puzzles, first move, maximum-likelihood rating), from each checkpoint's own
`dcm_test_set_results`:

- **Peak 1,726.6 at step 120,000** (NLL 1.9449, top-1 2,399 / 4,435). Lowest NLL 1.9361 at 126,222 (1,718.4).
- **Final (127,931): 1,716.3, NLL 1.9634**, top-1 2,379, top-5 3,832; 200-set 1,414.9, NLL 1.7988.
- Start file (R-elite 21,006): 1,637.6 / 2.0654. Final − start +78.7 pElo, NLL −0.102. Mean of the last 10 probes
  (118k–127k) 1,712.6; of the first 10 (1k–10k) 1,647.8.
- The LR peaks cost little: pElo at each peak step (6k, 16k, …, 126k) was on average 18 below the best probe of the
  same cycle (range 1–40). R-replay's 1.0 peaks: 194 on average (119–284; peaks at 11k … 81k).

Comparison at each run's own trainer step (value at the step, and the best probe in the 10,000 steps ending there).
R-replay and R-elitefresh started from the untrained `20261008-zlra-ab-fresh` network with the 0.001–1.0 cycle
(`parameters-zlra.json`): R-replay on `w3aA5b` (stopped at 86,398), R-elitefresh on the elite corpus (running;
probes through 82,000 at this writing). This is not an equal-history comparison: R-elitecyc began from a network
with 107,404 steps behind it. A blank cell: the run never reached that step.

| step | R-elitecyc | best | R-replay | best | R-elitefresh | best |
|---:|---:|---:|---:|---:|---:|---:|
| 10,000 | 1658.7 | 1658.7 | 1392.7 | 1460.5 | 1364.4 | 1396.8 |
| 20,000 | 1660.2 | 1666.9 | 1424.6 | 1564.2 | 1396.3 | 1501.1 |
| 30,000 | 1674.1 | 1682.3 | 1520.6 | 1615.0 | 1439.0 | 1547.8 |
| 40,000 | 1678.2 | 1678.2 | 1504.6 | 1611.4 | 1461.5 | 1529.3 |
| 50,000 | 1685.9 | 1685.9 | 1502.1 | 1649.9 | 1460.0 | 1550.3 |
| 60,000 | 1699.8 | 1699.8 | 1589.8 | 1629.4 | 1499.0 | 1573.9 |
| 70,000 | 1701.9 | 1701.9 | 1554.9 | 1638.1 | 1497.5 | 1574.4 |
| 80,000 | 1691.1 | 1700.9 | 1568.8 | 1620.1 | 1465.1 | 1570.3 |
| 90,000 | 1703.4 | 1707.0 | | | | |
| 100,000 | 1707.0 | 1707.0 | | | | |
| 110,000 | 1705.0 | 1708.1 | | | | |
| 120,000 | 1726.6 | 1726.6 | | | | |

Best per run: R-elitecyc 1,726.6 @120,000; R-replay 1,649.9 @48,000 (1,639.7 at its last save, 86,398); R-elitefresh
1,574.4 @66,000 (so far).

Training losses moved little (step 1,000 → 120,000): pLoss 2.624 → 2.506, vLoss 0.861 → 0.841, pEnt 2.868 → 2.804.

### Health

- Gradient: 0 clips in 1,438 step lines; largest `gNormMax` 0.742.
- No GPU faults (`gpu_faults.faults` empty in all three results files), no non-finite values, 0 parked or mostly-off
  channels and 0 of 128 value FC1 units at zero velocity at all 130 checkpoints; 0–7 value FC1 units at low velocity.
- BN running-variance max/median: 278.2 at step 1,000 (inherited; `blocks.2.bn1[13]`, 37 channels ≥ 10× their site
  median) → 415.3 at the final checkpoint (same channel, 40 channels); range over the checkpoints 271.9–424.4.
  `bn_running_variance_runaway` was not raised.
- Only alarm: `policy_offset_drift` (warning, action log), raised at the first check of each segment (trainer steps
  100, 45,550, 126,300) and active throughout. Its median |policy logit mean| was 10.40 at step 100, so it came in
  with the start weights; 9.36 at 127,000.

### Interpretation

- Feeding the elite corpus alone did not move R-replay's network (R-elite flat at ~1,635 for 21k steps at a fixed
  0.002). The same corpus with a 0.002–0.02 cycle gained about 80 pElo over 128k steps and was still rising slowly at
  the end of the epoch (cycle bests 1,659 → 1,682 → 1,700 → 1,707 → 1,727). One run, one seed: the LR cycle and the
  extra 128k steps are not separated (R-elite at a fixed LR was stopped at 21k).
- The 10× cycle cost ~18 pElo at its peaks; the 1,000× cycle (0.001–1.0) cost ~194.
- From scratch, the elite corpus has so far trained slower than `w3aA5b` (R-elitefresh below R-replay at every
  10k step through 80k) on the same 0.001–1.0 schedule, seed and start.

### Reproduce

Corpus `20260704-215145-op24Gp` (Lichess Elite Database Dec 2021–Nov 2025, 14,619,153 games, 42 shards; imported
from the PGNs, not in the repo). Parameters `parameters-zlra-elite-gentlecycle.json`. Builds: frozen
`FrozenBuilds/DCM-2503-7d945ec7.app` (segment 0) and `FrozenBuilds/DCM-2523-a242b7ad.app` (segments 1–2).

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
$E/launch/elitecyc_resume2_launch.sh             # segment 2, exact resume to the end of the epoch
```

Without the interruptions, segment 0 alone runs to the end of the epoch (`--epochs 1`); the exact resumes reproduce
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

Channel 13 genuinely co-varies with board "fullness"/material (max-abs over squares vs total_pawns Pearson +0.38,
piece_total +0.33, legal_count +0.33, in_check −0.32, endgame-phase −0.33); it is modulated, not collapsed (the
modulation rides on a large constant offset, so its coefficient of variation is small, ~0.047 deep in the trunk).

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
