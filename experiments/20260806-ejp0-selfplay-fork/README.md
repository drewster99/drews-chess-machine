# 2026-08-06 — Ejp0: self-play from a strong corpus-distilled seed

**Status:** abandoned. Run 2 was stopped on 2026-08-30 for a macOS update and later resumed only for ~2h (09-17) plus a ~20 s launch (09-21). The decisive champion-vs-seed head-to-head was never run.

## Question

Can real self-play (Play-and-Train, arena-gated promotion) improve on a strong network distilled from a human-game corpus? In other words, can it break the ~1640–1740 wide-pElo corpus ceiling that qeu8 reached?

## Setup

- **Seed:** `Models/20260806-Qeu8sp-seed-from-step1300000.safetensors`. Its `__metadata__`:
  - model_id `20260727-1-Ejp0`, parent `20260706-1-PVZp`
  - training_step 1300000, creator `replay`, replay_epoch 8 (0-based)
  - corpus `20260624-192615-w3aA5b`
  - built_by_build 2089, git 085356f
  - content_sha256 `dad6e4a4…`
  - F32 storage, 3,929,393 elements

  The seed's pElo was 1731.3 / nll 1.9033 on the replay-era `--probe-model` scale (per memory; not re-probed here).
- **Architecture** (from the `[ARCH] loaded model` line, `dcm_log_20260808-174024.txt`): `v5 . in basic30(30) -> stem 64 (7x7) . 2x[15x15+15x15 @64, SE+/4, relu/pre, clean_add, ReZero(0.5·tanh≤0.5), out:layer_norm, drop*1] . act relu . policy intermediate_conv(4864) . value WDL(16->FC64) . bfloat16 . 3,929,393 params`
- **Common settings:** GUI Play-and-Train launched as `open <app> --args --train --start-model <seed>` plus a reopen event. Batch 4096, auto-arena every 900 s, 400 games, promote ≥ 0.53, 170–175 self-play workers, buffer 1M.
- **Builds:**

  | phase | build | git |
  |---|---|---|
  | first attempt, run 1 part a | 2090 | 085356f\* |
  | run 1 part b, run 2 launch 1 | 2091 | c756864\* |
  | run 2 launch 2 | 2092 | c756864\* |
  | run 2 launch 3 | 2102 | 58876cc\* |
  | 09-17 resume | 2105 | 9f88eb3\* |
  | 09-21 launch | 2114 | 0b3e5a6\* |

- **Machine:** the local M4 Pro (GUI session).

## Runs

Runs 1 and 2 are **separate forks of the same seed** and must never be concatenated. Both mint `Ejp0-1…` IDs, so those IDs are ambiguous across runs.

### Phase 0: the broken-looking first attempt (soft self-play τ), build 2090

- `dcm_log_20260806-215333.txt`, 08-06 21:55 → 08-07 01:42:
  - sp.tau 1.00/0.50/0.007, ar.tau 0.60/0.20/0.020
  - lr 1e-2, μ 0.90, wd 5e-4, **dropout 0.30** (leaked from June via UserDefaults)
  - 21,017 steps, 14 arenas, **0 promotions**, arena Elo −330 … −211 (mean −264)
  - SIGUSR2 save: `Sessions/20260807-064237-20260807-2-OSWK-sigusr2.dcmsession`
- `dcm_log_20260807-014550.txt`, 01:46 → 02:13: identical except **dropout 0**. 2,532 steps, 1 arena at **−317** (13.9%), so dropout is ruled out. SIGUSR2 save `…-FeUB-sigusr2.dcmsession`.
- `dcm_log_20260807-021701.txt`, 02:17 → 13:09:
  - **sp.tau = arena schedule 0.60/0.20/0.020**, dropout 0, lr 1e-2
  - 59,717 steps, 42 arenas, **0 promotions**. Arena #1 was **−48 (43.1%)**; the range was −282 … +12 (mean −72).
  - In its last ~2 minutes the lr was set to 1.0 (13:07:21) and then 0.1 (13:08:22). The probe CSV's 442 pElo / NLL 16.3 at step 59,578 is that damage.
  - This session (`orSA`) is the start of run 1.
- **Diagnosis** (repo `TODO.md` #1 at commit 089383f^, now folded into ROADMAP):
  - Soft self-play τ diffuses the trainer's policy.
  - The arena *samples* that diffuse policy and loses.
  - The wide probe scores argmax top-1 and cannot see diffuseness.
  - The arena `pol` column confirms it: soft 0.20 → 0.68 vs sharp 0.48 → 0.83.
  - Not a code bug.

### Run 1: session `orSA`, registry key `Ejp0r1`

- It resumed the 10:17 `orSA` periodic autosave at step 44,113 under build 2091: `dcm_log_20260807-131013.txt`, 08-07 13:10 → 08-08 17:24.
- Settings on resume:
  - lr set 1e-2 → 1e-3 (13:10:46)
  - wd 5e-3
  - **lr/momentum cycling turned ON at 13:15:36**: lr [1e-5, 2e-2] period 4000; μ [0.82, 0.96] period 1000
  - the lr range narrowed to **[5e-6, 1e-3] at 19:06:18**
  - wd → 1e-3 and lr range [1e-6, 5e-4] at 17:13 on 08-08, after the last promotion
- 112 arenas (#32–#143), **9 promotions**, all under the [5e-6, 1e-3] cycle. They were arenas #59, 65, 71, 75, 78, 89, 90, 114 and 116, at +24, +26, +23, +34, +25, +31, +49, +27 and +23 Elo.
- Final: champion `Ejp0-9` / trainer `Ejp0-10` @ 197,340 (safetensors metadata, `Sessions/20260808-222418-20260807-6-orSA-sigusr2.dcmsession`).
- Ended by the user after 27 non-promoting arenas (#117–#143).

### Run 2: session `sjIy`, registry key `Ejp0`

A fresh fork from the seed, 08-08 17:40 → 08-30 14:11, plus later resumes:

| launch | log | build | steps | arenas | promotions |
|---|---|---|---|---|---|
| 1 | `dcm_log_20260808-174024.txt` | 2091 | 1 → 82,039 | #1–#60 | 3 |
| 2 | `dcm_log_20260809-101517.txt` | 2092 | 82,033 → 570,685 | #61–#442 | 17 |
| 3 | `dcm_log_20260825-110210.txt` | 2102 | 569,192 → **1,184,817** | #442–#928 | 46 |
| post-stop | `dcm_log_20260917-170217.txt` | 2105 | 1,183,795 → 1,192,765 | #929–#935 | 1 (#930 @ 1,186,322, +21) |
| post-stop | `dcm_log_20260921-000813.txt` | 2114 | one `[STATS]` line at 1,186,322 | — | — |

- **Settings throughout:** sp.tau = ar.tau 0.60/0.20/0.020 until the τ change; dropout 0; wd 1e-3; lr cycling [1e-6, 5e-4] and μ cycling [0.82, 0.96] from step 1.
- **Checkpoints** (safetensors `__metadata__`):
  - `…-sjIy-manual` (08-09): champion `Ejp0-3` @ 82,033
  - `20260830-185905-…-sjIy-promote`: champion `Ejp0-66` / trainer `-67` @ 1,183,795
  - `20260917-223317-…-sjIy-promote`: champion **`Ejp0-67`** / trainer `-68` @ 1,186,322. This save has no replay_buffer.bin.
  - `20260830-062000-…-sjIy-periodic`

### The near-greedy τ experiment (inside run 2, launch 3)

- The `[PARAM]` lines at **2026-08-28 19:12:13–19:12:26**, at step **959,497**, set ar.tau 0.60/0.20 → 0.20/0.01 and sp.tau 0.60/0.20 → 0.20/0.01. Decay stayed 0.02/ply.
- It lasted until the 08-30 14:11 stop. The 09-17 resume still ran with 0.20/0.01.

## Results

The pElo below is the in-app WIDE (4,435-position) probe on the recording build's scale (`selfplay_probe/Ejp0*.csv`). The 200-position probe in the same logs reads higher and is not used. The seed's 1731.3 is on the `--probe-model` scale, so compare in-app values only with each other.

| phase | steps | arenas | promotions | arena Elo range | wide pElo | notes |
|---|---|---|---|---|---|---|
| Phase 0 soft τ, dropout 0.3 | 1 → 21,017 | 14 | 0 | −330 … −211 | — | trainer diffused |
| Phase 0 soft τ, dropout 0 | 1 → 2,532 | 1 | 0 | −317 | — | dropout ruled out |
| Phase 0 sp.tau = arena, lr 1e-2 | 1 → 59,717 | 42 | 0 | −282 … +12 | — | no promotion at constant lr 1e-2 |
| Run 1 (cycled lr ≤ 1e-3) | 44,113 → 197,342 | 112 | **9** | −234 … +49 | 1769 @ 82 (≈ seed), 1704 @ 89,052, **1671** end @ 197,353 | 27-arena stall at the end |
| Run 2 before the τ change | 1 → 959,497 | #1–#758 | **41** (last #757 @ 957,823) | −70 … +48 | peak **1761 @ 6,403** (1760 in the 1000-step bucket @ 6,967); 1743 @ 88,878; 1723 @ 219,953; 1696 @ 499,978; **1638** @ 929,953 and @ 959,376 | draws 48–92 / 400; unique 399–400 / 400; avgDiverge 1.7–2.8 |
| Run 2 near-greedy τ | 959,497 → 1,184,817 | #759–#928 (170) | **25** (#759 … #928) | **−182 … +126** | 1652 max just after the change; min **1576 @ 1,079,751**; median 1607 at ≥ 1.15M; end **1613** @ 1,184,801 | draws **8 – 307** / 400; unique down to **64 / 400** (#851); avgDiverge up to **124.2** |
| Run 2, 09-17 resume | 1,183,795 → 1,192,765 | 7 | 1 | −45 … +21 | 1595–1616 (35 marks); end **1609** @ 1,192,678 | still near-greedy τ |

Other run 2 facts:

- **Totals:** 1,184,817 steps at the stop, 66 promotions (3 + 17 + 46) through 08-30. Adding 09-17 gives 67 promotions and 1,192,765 steps. `data/Ejp0.csv` now includes the 09-17 resume: **237.1 h** of summed elapsed time (was 235.6 h to 08-30, which also counted the 08-30 log's 1,183,795–1,184,817 tail that the resume discarded).
- **Promotion rate:** 41 in 20.1 days before the τ change vs 25 in 43 h after it, ~6.8× faster.
- **Arena swings under near-greedy τ:**
  - #813 −182 → #814 +80 (262 points)
  - #825 −151 → #826 +124 (275)
  - #849 +126 → #850 −35
- **Policy sharpening over run 2 while the probe fell:**
  - pEnt: 2.6250 at step 10 → 2.4001 at 930,013 → 2.3643 at 1,184,817
  - playedMoveProb: 0.2314 → 0.3473 → 0.3763
  - pD: 0.037 → 0.164 → 0.199. The value head stays decisive; pD is nowhere near 1.

## Conclusion

- **Self-play did not improve the distilled seed on any absolute measure available.**
  - The arena promoted 66 times, but it only ever scores a candidate against its immediate predecessor.
  - The only fixed-reference instrument, the wide puzzle probe, fell steadily in run 2: ~1760 at ~6.4k → 1638 by 930k → 1613 at the stop.
  - The policy got *sharper* over the same span (pEnt ↓, playedMoveProb ↑). "The argmax probe is blind to diffuseness" does not explain this decline; the net became more confident while agreeing less with puzzle solutions.
  - Whether the 66 promotions are transitive gains in play or non-transitive churn is **unresolved. The champion-vs-seed head-to-head was not run.**
- **Soft self-play τ is incompatible with a sharp seed.** τ 1.0 → 0.5 self-play diffused the trainer so much that it could not beat the frozen seed (−211 to −330 Elo, 0/15 arenas). Matching self-play τ to the arena schedule got arena #1 to −48.
- **Constant lr 1e-2 never promoted (42 arenas). Cycled lr with a ≤ 1e-3 ceiling did** (run 1: 9, run 2: 66). This is correlation across two configs, not a controlled test.
- **Near-greedy τ (0.20 → 0.01) on both self-play and arena is a negative result.**
  - It collapsed arena game diversity (64/400 unique, avgDiverge 124), which turned the arena into a coin-flip over a handful of long lines.
  - That manufactured 25 promotions in 43 h, and each promotion wrote the arena-picked snapshot into the champion.
  - The probe dropped to 1576 and ended at 1613, never recovering above its pre-change 1638.
  - If this is retried, sharpen self-play only, partially, and leave ar.tau at 0.60/0.20/0.020 so the instrument survives.

## Caveats

- One seed, one run per config. Runs 1 and 2 differ in more than one knob: run 1 used wd 5e-3 and cycle [5e-6, 1e-3] at its promotions; run 2 used wd 1e-3 and [1e-6, 5e-4].
- In-app pElo reads ±~40 between consecutive probes (for example 1679 @ step 82 vs 1761 @ 6,403 in run 2, and 1769 @ 82 in run 1). Only multi-hundred-k-step trends are meaningful.
- Run 1's summed elapsed was 43.3 h while it double-counted the rewound segment 2 tail; with that tail cut (registry `log_kept_to`, 2026-09-29) `data/Ejp0r1.csv` reads 40.5 h. Segments 0 and 1 are separate fresh starts from the seed, concatenated on the step axis. Use steps.
- The 25 near-greedy promotions (Ejp0-42…-66, plus -67 on 09-17) are **suspect** as strength evidence.
- Elo in this file is arena Elo relative to the immediate predecessor, never absolute.

## Follow-ups

- **Champion vs seed head-to-head: not run.** No log, experiment folder, arena file (`experiments/20260708-arena-38engines/` has no Ejp0 entry) or memory note records one. Suggested set: Ejp0-67 (or -41, the last pre-τ-change champion) vs the 1300000 seed, and vs run 1's Ejp0-9, over the cutechess harness with an opening book.
- A sampled-play probe, or entropy reported alongside pElo, so probe and arena stop disagreeing by construction (TODO #1 follow-up (b)).
- Before any rerun, set a retention cap for `-promote` saves. Each save is ~7.2–7.5 GB.

## Audit notes

- **Verified against logs** (per-log streaming scans; `[STATS]` filtered to the Ejp0 lineage; `[ARENA] #N kv` lines; `[PARAM]` lines):
  - builds, τ, lr, μ, wd and dropout per phase
  - step ranges, arena counts and ranges
  - promotion counts: run 1 = 9; run 2 = 3 + 17 + 46 = 66, split 41 before and 25 after the τ change
  - the τ-change timestamp and step
  - diversity and draw ranges, swings, pEnt / playedMoveProb / pD
- **Verified against dashboards:**
  - `selfplay_probe/Ejp0.csv` and `Ejp0r1.csv` (4,398 and 911 probes)
  - `data/Ejp0.csv`: 1,184,817 max step; 235.6 h; peak 1760 @ 6,967
  - `data/Ejp0r1.csv`: 43.3 h (40.5 h after the 2026-09-29 fix that cuts segment 2's abandoned tail; `data/Ejp0.csv` is 1,192,765 max step / 237.1 h after adding the 09-17 resume)
- **Verified against headers:** the seed and the session champion/trainer `__metadata__` above. `du -h` shows sjIy/orSA/OSWK sessions of 7.2–7.5 GB each (base-2), matching "~7–8 GiB per promote save".
- **Corrections:**
  - Old claim (registry `Ejp0r1` note; memory): "run 1 … lr-cycling and momentum-cycling OFF"; "run 2 differs from run 1 by cycling ON". New: run 1 had cycling ON from 13:15:36 on 08-07, before its first promotion (20:13). 1,669 of its 1,796 `[STATS]` lines carry `lr=…·cyc` / `μ=…·cyc`. Evidence: `dcm_log_20260807-131013.txt` `[PARAM] lr_momentum_cycle` lines at 13:15:36, 19:06:18, 17:13:39.
  - Old claim (memory): "the old lr-0.1 sp.τ=arena run went 42 arenas / 0 promotions"; "session restored lr=0.01". New: that run (`dcm_log_20260807-021701`) ran lr 1.0e-2 for all but its last ~100 s (1.0 at 13:07:21, 0.1 at 13:08:22). On resume the lr was set to 1e-3 (13:10:46), then cycled.
  - Old claim (registry and memory): "run 1 ended after a 28-arena parity stall." New: 27 arenas (#117–#143) after the last promotion (#116). Evidence: `[ARENA]` lines in `dcm_log_20260807-131013.txt`.
  - Old claim (memory): "unique fell … as low as 74/400 (18%)." New: the minimum is **64/400 (16%)** at arena #851 (avgDiverge 108.1). 74/400 also occurred. The registry already says 64.
  - Old claim (memory): "consecutive arenas swung +126 then −151." New: the pairs are #825 −151 → #826 +124 (275 points) and #849 +126 → #850 −35. Evidence: `[ARENA] #825/#826/#849/#850 kv` lines.
  - Old claim (memory): "wide pElo fell 1626 → ~1605" during the τ experiment. New: 1638 at the change (probe @ 959,376) → min 1576 @ 1,079,751 → median 1607 at ≥ 1.15M → 1613 at the end. 1625–1626 is the reading near 1.00M. Evidence: `selfplay_probe/Ejp0.csv`.
  - Old claim (registry `Ejp0`): "final champion Ejp0-66; 66 promotions; a resume would append a fourth log." New: the run *was* resumed on 2026-09-17 (build 2105, still near-greedy τ). It added arenas #929–#935 and one promotion (#930, +21, step 1,186,322 → champion **Ejp0-67**), saved as `20260917-223317-…-sjIy-promote.dcmsession`. A 09-21 launch (build 2114) logged one `[STATS]` line. Neither log is in the registry. 66 stands for the 08-08 → 08-30 span. **Fixed 2026-09-29:** `selfplay_registry.json` `Ejp0` now lists the 09-17 log (with the 08-30 log cut at 1,183,795, where it resumed), records the 09-21 one-line launch under `excluded_logs`, and reads 67 promotions, final trainer `Ejp0-68`, endpoint 1609 / 2.216 (last wide probe, @ 1,192,678), label "67 promo, 237h, ended 09-17". The `Ejp0r1` note's cycling and stall-length claims were corrected as above.
- **Unverifiable:**
  - Disk figures "148 GiB → 13 GiB free", "Sessions/ 656 GB / 89 saves, 74 promote", and "15 GiB free at stop": `Sessions/` has since been pruned to 21 entries, and no log line records free space. Run 2's logs show 96 `Saved session` events across all triggers (6 + 30 + 60), which is consistent with that magnitude but does not match the save count.
  - The seed pElo of 1731.3 and the OSWK champion 1731.3 / trainer 1641.2 probes (memory): re-probing is barred while training is live.
  - "~21.2M plies/hr" (memory): not recomputed.

## Reproduce

**Status: partial** — seed, builds and session parameters recorded; self-play is inherently unrepeatable and several builds were dirty.

- **Commit / build:** from `[APP] launched` lines (table in Setup): 2090 `085356f*`, 2091 `c756864*`, 2092 `c756864*`, 2102 `58876cc*`, 2105 `9f88eb3*`, 2114 `0b3e5a6*`. Every build was a **dirty tree** (`*`), so no commit reproduces them exactly.
- **Corpus:** none (self-play). The seed itself was trained on [`20260624-192615-w3aA5b`](../corpora/20260624-192615-w3aA5b.md).
- **Starting point:** `~/Library/Application Support/DrewsChessMachine/Models/20260806-Qeu8sp-seed-from-step1300000.safetensors` (model_id 20260727-1-Ejp0, training_step 1300000; still present).
- **Parameters:** no parameters file; the GUI used saved app settings. Each `.dcmsession/session.json` records them, e.g. `20260830-185905-20260808-2-sjIy-promote`: learningRate 0.001, momentumCoeff 0.9, gradClipMaxNorm 30, lrWarmupSteps 500, batchSize 4096, replayBufferCapacity 1000000, minPositionsBeforeTraining 500000, replayRatioTarget 0.48 (autoAdjust off), selfPlayWorkerCount 175, arenaGames 400, arenaAutoIntervalSec 900, promoteThreshold 0.53, dropoutRate 0. τ settings changed between phases (see Runs).
- **Command:** `open <DrewsChessMachine.app> --args --train --start-model "$M/20260806-Qeu8sp-seed-from-step1300000.safetensors"`, then a reopen event (`open <app>` again) so the window materializes and auto-train fires. Parameters must be set in the app (Training Settings) beforehand.
- **Probe / analysis:** `documentation/dashboards/selfplay.py` + `selfplay_registry.json` keys `Ejp0r1`, `Ejp0`; in-app wide probe series `documentation/dashboards/selfplay_probe/Ejp0r1.csv`, `Ejp0.csv`.
- **Expected exactness:** statistical only. Self-play move sampling (`Float.random` in `MoveSampler`), Dirichlet noise, minibatch sampling (`Int.random`), arena outcomes, and GPU nondeterminism are all unseeded; promotion timing diverges almost immediately.
- **Missing:** exact source for the dirty builds; a parameters file per phase (only session snapshots at save points, none for the phase-0 attempts besides their SIGUSR2 sessions); the literal τ settings per launch are only as recorded in Runs.
