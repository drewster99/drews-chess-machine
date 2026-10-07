# Up next (written 2026-10-07 ~08:15)

State: `main` is pushed and clean. The last full suite passed: 3,234 passed, 0 failed, 1 skipped, at b4845088. Nothing is running: no training runs, worktrees or background jobs. Yesterday's work, decisions and blockers are in `DAILY_BRIEFING_2026-10-07.md`.

## 1. Relative gradient cap: try it in the GUI (you)

- Rebuild from `main` and relaunch. The app that was running had been built before the cap existed.
- Find the cap under Settings ▸ Optimizer, directly under "Clip:".

  | Field | Meaning | Default |
  |---|---|---|
  | Relative clip | Off / Log only / Clip | Log only |
  | k × median | multiple of the trailing median | 3.0 |
  | Floor | lowest the relative cap can go | 0.5 |
  | Window N | steps in the median lookback | 1000 |
  | Min history W | steps before the relative cap applies | 100 |

- Set Relative clip to Clip, run Play-and-Train for about 30 minutes, then check:
  - each `[STATS]` line carries `gNormMax=`, `clips=` and `gCap=`;
  - every clip logs a `[GRAD-CLIP]` line, and each start logs one `[GRAD-CLIP] config …` line;
  - File ▸ Save Session, quit and resume, and the `[RESUME]` lines show no `grad_norm_history` gap;
  - after Promote Trainee Now, training continues with no discontinuity error.
- These are the relative cap plan's check V-5 (`documentation/plans-active/RELATIVE_GRADIENT_CAP_PLAN.md`). In-process tests cover it, but nobody has tried it by hand.

## 2. Relative cap default: still log only (decision D-15)

- Validation runs V-1 and V-3 both passed (E-0024), so the plan's step P5 would switch the default to clip with k = 3.
- The session's permission check refused that edit, so the default is still log only.
- To change it, set `relative_grad_clip_mode` default `1 → 2` in `Training/TrainingParameters.swift` (around line 474). Then update the one test that pins it, `RelativeGradientCapParameterTests.test_declarations` (`encode(1)` → `encode(2)`), and run the parameter and cap test classes.
- Until then, set `"relative_grad_clip_mode": 2` per run (parameters file or popover).

## 3. Proposed experiments (not started)

- **B-silu from scratch with the relative cap, to 40k:** the real test of the cap.
  - B-silu is the run that blew up at about 20,600, and it has never been trained from step 0 with the cap on. V-3 was the ReLU tower, and only 3,000 steps.
  - Recipe: B-silu's own (README, Arm B-silu), plus `relative_grad_clip_mode` 2, k 3, N 1000, W 100, floor 0.5.
  - Expect about 5–6 h to 20k with the GPU to itself.
- **B (ReLU) with the cap, to 40k, alongside it:** does the cap cost anything on a healthy run?
- **V-2:** a k = 3 clip run from B-silu's 18k checkpoint to 23k, against the fixed 1 / 2 / 5 caps. The parameter file is ready: `experiments/20261005-lr-schedule-ab/parameters-B-relcap-v2.json`. The command is in the README's V-1/V-3 section; add `grad_norm_history` to `--accept-inexact`.
- **Frozen build:** `FrozenBuilds/DCM-2390-b4845088-relcap.app` is main with the cap. Launch scripts are in `experiments/20261005-lr-schedule-ab/launch/`, and `launch/pelo_table.py` prints the pElo comparison table.

## 4. Optional

- **Training-health alarms on the real corpus (alarms plan V-2..V-5):**
  - These are 7 command-line runs, about 4,100 steps in all.
  - Run them with `experiments/alarms-validation/run_v.sh <binary> <output folder>`, which needs a build of main.
  - They were not run because the permission check refused the launches. In-process tests cover their logic.
- **Lichess bot live checks:** the follow-lineage plan §6 1–12 and the challenge-log plan §6.5–6.7 run on their own the next time the bot goes online. Look at the `[LICHESS-BOT]` lines and `Challenges/`.
- **Type-check timing warnings:** the 31 that remain come from expanding the `@TrainingParameter` macro (`TrainingParameters`, `LichessBotController`). Removing them means restructuring those files, which is your call.
- **Untracked leftovers:** `experiments/20261005-lr-schedule-ab/train-*.stdout` and the `probes-*.errors/` folders, the probe loop's stderr. They were never committed. Keep or delete as you like.
