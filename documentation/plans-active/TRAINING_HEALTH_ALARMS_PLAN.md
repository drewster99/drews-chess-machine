# Training health alarms plan: every training path checks itself and says so in the log

Status (2026-10-06): **P1–P4 implemented** (P1: pure core and incident tests; P2: parameters, CLI paths, `results.json`, `--replay-health-log`, the value-FC1 read; P3: GUI; P4: docs — see the **Implementation notes** sections at the end). P5 (OD-9) is not implemented. V-1 is done (P2 notes); V-2 to V-7 need live training or GUI runs and are not done. The owner's decisions of 2026-10-06 (OD-18 to OD-22: rule 2 covers every activation through parked channels, rule 4's relative regression arm, the new rule 9 `gradient_spike`, live evaluations decoupled from log lines every 50 trainer steps, every affected site named) are folded into the text below; where an older paragraph and an OD-18 to OD-22 entry disagree, the entry rules.
- **Sequencing: P2 is gated on the owner's approval of the redesigned `documentation/plans-active/STATS_LINE_RESUME_CADENCE_FIX_PLAN.md` and on that plan landing.** That redesign (committed on `main` at `ed534733`, reviewed) moves CLI step lines to a time cadence (about 180 s; dense every 50 trainer steps for the first 1,000) with lines and saves at overall trainer-step multiples of 1,000, and already assumes this plan's decoupled evaluation (OD-21): live alarm evaluations run every 50 trainer steps on every path, independent of the step line, with the per-step order **step-line block (when due) → live alarm evaluation (when its 50-step tick is due) → save block**, and one live `[LAYER-HEALTH]` read serving both the logged readout and the evaluation on steps where both are due (its OD-15, adopted here). This plan's requirements on it are in Part P. P1 can proceed. This plan's P1–P3 land before HPARAM_RECORDING_PLAN P4 (its O-22).
- Independently reviewed against the code and the logs (2026-10-05). The review's corrections are folded in; the design changes it caused are marked in the text where they matter (worker parking during a GUI suspension, checkpoint ordering, ring sizing, the sparse offline-replay semantics).
- Every `file:line` was checked against `main` at `62dc8559`.
- Paths are relative to `DrewsChessMachine/DrewsChessMachine/` unless they start with `DrewsChessMachine/` (project folder), `DrewsChessMachineTests/` (= `DrewsChessMachine/DrewsChessMachineTests/`), `documentation/`, `experiments/` or `scripts/`.
- Session logs are under `~/Library/Logs/DrewsChessMachine/`. Every number quoted from a log was re-measured for this plan from the log itself. Nothing was taken from a summary.

**The request (owner, 2026-10-05).** "Our normal training path needs to periodically check for these things, even if every 1000 steps... implement alarms, running in the UI but also when not in UI... I'd normally want log-only as I may not wish to stop the run immediately."

**What this plan does.**
- Adds one pure, testable evaluator, `TrainingHealthEvaluator`. All three training paths use it: GUI Play-and-Train (including GUI `--train`), `--replay-corpus` and `--train-vs-uci`.
- Feeds it data the paths already compute:
  - the per-step `TrainStepTiming` that `trainStep` returns;
  - the `LayerHealthSummary` that the live and checkpoint `[LAYER-HEALTH]` passes already build;
  - and one new, small read (OD-15): the optimizer velocity of the single tensor `value.fc1.weight`, every 1,000 trainer steps, on the paths whose saves do not already provide it (D6).
- Adds no random draws and no change to trainer, optimizer or replay-buffer state. Its only GPU work is that one read, on the trainer's own queue between SGD steps.
- Defines nine rules (two more — value loss above ln 3 and a one-sided value head — were dropped by the owner, OD-7; rule 9, `gradient_spike`, was added by the owner on 2026-10-06, OD-20). Their thresholds were measured against three incident runs, four healthy replay runs, a long GUI self-play run, and a survey of every log since `[LAYER-HEALTH]` shipped; the 2026-10-06 changes were checked against the same runs plus arm B-silu (`dcm_log_20261005-234437.txt`).
- Reports each alarm as one `[ALARM] health …` line in a fixed key=value format. By default every rule only logs.
- Lets the owner set any rule to stop the run. The stop goes through each path's existing clean exit: on the CLI paths, the loop exit and final save; in the GUI, a training suspension or `AutoTrainTermination`.
- Surfaces alarms in the GUI as a list under the existing banner, and in `results.json` as an `alarms` array.
- Adds an offline replay mode, `--replay-health-log`. It runs the real evaluator over saved session logs, so validation never needs a second copy of the rules.

Rules this plan follows (CLAUDE.md files and the owner's standing rules):
- one source of truth: thresholds live in one Swift declaration, and every path and the offline replay call the same evaluator;
- one code path shared by GUI, corpus replay and train-vs-UCI;
- no silent defaults and no fallbacks: a rule with no data in a window holds its state and says so, and never reads "missing" as "healthy";
- no `try?` and no force unwraps;
- the full CLAUDE.md parameter checklist for every new parameter;
- probe isolation: the monitor is an observer;
- tests are never modified or deleted without the owner's approval (each needed edit is under **Owner decisions**);
- one SwiftUI `View` per file and no helper `some View` properties. New visible content stays in the hierarchy and is hidden with opacity 0 and a zero frame, never removed with `if`;
- no app, test or CLI runs while training runs are live (validation waits for an idle machine).

---

## Summary

| # | Item | Phase |
|---|---|---|
| 1 | `TrainingHealthEvaluator` (pure value type): nine rules, two severities, hysteresis, sustain, gates, reminders; `BatchNormPassThrough` (the activation-aware parked classification rule 2 counts, OD-18) | P1 |
| 2 | `TrainingHealthMonitor` (lock-protected class): per-step pending window, loss-reference history, window statistics, serialized evaluations, stale-checkpoint rejection, trainer-clock rewind with generations | P1 |
| 3 | `TrainingHealthLog`: the `[ALARM] health …` and `[HEALTH] …` line formats, shared by every path | P1 |
| 4 | `TrainingHealthLogReplay`: parses `[REPLAY]` / `[VS-UCI]` / `[LAYER-HEALTH]` lines into evaluator observations (used by the incident tests and `--replay-health-log`) | P1 |
| 5 | `LayerHealthLog.live(trainer:)` returns the summary as well as the lines; `liveLines` is removed and its four callers converted | P1 |
| 6 | 12 new `TrainingParameters` (enable, check interval, learning grace, one action per rule — nine rules), full checklist | P2 |
| 7 | Corpus replay and train-vs-UCI: record every step, evaluate at every stats tick and every checkpoint pass, stop through the loop exit and final save | P2 |
| 8 | `results.json`: `alarms` array, `alarm_config` object, `termination_reason: "training_health_alarm"` | P2 |
| 9 | `--replay-health-log <log>…` CLI (no GUI, no GPU) | P2 |
| 10 | GUI: monitor wired to the trainer worker, the `[STATS]` ticker and the checkpoint passes. Promotion rewind handled. Alarm list view with its own Silence; one beep loop for both alarm sources. Health tab in the settings popover. Stop = training suspension with the worker parked (interactive) or `AutoTrainTermination` (`--train`) | P3 |
| 11 | Docs: `documentation/training-health-alarms.md`, `--help`, CLAUDE.md tag list (OD-11), CHANGELOG | P4 |
| 12 | Move the GUI-only detectors' conditions into the shared evaluator (OD-9, P5). Provide the per-segment alarm summary API that HPARAM_RECORDING_PLAN P4 stores as `configuration.health_alarms` (OD-10; this plan's API in P1–P3, the record field in HPARAM P4) | P1–P3 (API), P5 (OD-9) |
| 13 | Rule 3's value-FC1 velocity check at most 1,000 trainer steps apart on every path (OD-15, D6): reused from a save's checkpoint pass where one covers it, otherwise one dedicated read of that tensor | P1 (pure scheduling), P2 (trainer read, train-vs-UCI), P3 (GUI) |

## Every check, its trigger and its interval

The owner's rule (OD-15): every check runs on a set interval or trigger. "Step" is the trainer step unless it says segment step.

| Check | GUI Play-and-Train / `--train` | `--replay-corpus` | `--train-vs-uci` |
|---|---|---|---|
| Record the step (rules 1 and 4–7's inputs) | every SGD step | every SGD step | every SGD step |
| Live evaluation: live `[LAYER-HEALTH]` read (BN state, ReZero α) + rules 1, 2, 4–9 | every 50 trainer steps, on overall trainer-step multiples (`TrainingHealthCadence.isLiveEvaluationStep`), independent of the `[STATS]` emits (OD-21) | every 50 trainer steps on overall trainer-step multiples, independent of the step lines (OD-21); per-step order: step-line block → live evaluation → save block, so at every save step (a multiple of 50) the live evaluation precedes the save's checkpoint pass (R0). Where a step line and an evaluation fall on the same step, one live read serves both (cadence plan OD-15) | same as replay |
| Checkpoint evaluation: full-tensor `[LAYER-HEALTH]` pass + rules 1, 2, 3, 8 | every session save: periodic (default 6 h), promotion, Promote Trainee Now, manual, SIGUSR2 | every rolling save, plus the final save. Save cadence per the cadence plan: every 1,000 segment steps today, at overall trainer-step multiples of 1,000 after its redesign; consecutive saves are 1,000 trainer steps apart either way | every session save (periodic, time-based; final; abort), plus each enumerated checkpoint (same 1,000-step cadence as replay's saves) with `--enumerate-checkpoints` |
| Rule 3 value-FC1 velocity (OD-15, D6) | every 1,000 steps: a dedicated read whenever 1,000 steps have passed since the run started or since the last rule-3 observation (a save's pass also counts) | every 1,000 steps, from the autosave's checkpoint pass (every 1,000 segment steps); a dedicated read only when that save or its pass produced no observation (a first, non-fatal save failure, `CLI/CorpusReplayRunner.swift:1628-1633`, or a failed health pass) | every 1,000 steps: from the enumerated checkpoint's pass with `--enumerate-checkpoints`, otherwise a dedicated read on the same deadline as the GUI |
| `[HEALTH] check` line and `active` reminders | first live evaluation at or after each multiple of `training_health_check_interval_steps` (1,000), which with 50-step evaluations is the multiple itself whenever the interval is a multiple of 50 | same | same |
| Stop | decided on the main actor after every delivered evaluation; the worker parks at its next loop top | `healthStop` checked at the loop top before every step | same as replay |

---

## Evidence: the incidents and the healthy baselines

### The runs

| Run | What | Log(s) | Rows |
|---|---|---|---|
| A | LR-schedule A/B arm A: constant LR 0.01, healthy | `dcm_log_20261005-013220-2.txt` | 656 `[REPLAY]` (to trainer step 32,750), 654 live, 32 checkpoint |
| B | Arm B: LR cycle 1.0 ↔ 0.001 | `dcm_log_20261005-013235.txt` | 655 `[REPLAY]` (to trainer step 32,700), 653 live (to 32,600), 32 checkpoint |
| C | Arm C: LR cycle 10 ↔ 0.01. Segment 0 to step 513, then `--resume-exact` segment 1 to trainer step 6,116 | `dcm_log_20261005-090417.txt`, `dcm_log_20261005-121841.txt` | 11 + 113 `[REPLAY]`, 11 + 113 live, 1 + 6 checkpoint |
| R7, R8 | noSE/noReZero seeds 1 and 2: the Avg(R7,R8) comparator, healthy | `dcm_log_20261002-011124.txt`, `dcm_log_20261002-035513.txt` | 661 `[REPLAY]` each. **No `[LAYER-HEALTH]` lines**: these logs predate it. |
| GUI | Ejp0 self-play fork (a long GUI run from a trained seed) | `dcm_log_20260806-215333.txt` | 720 `[STATS] elapsed=` lines, `steps=` 1 → 21,017 (the session's step count); steady state a median of 94 steps per 60 s emit at ≈ 0.65 s/step. **Caveat:** this run's arena was audited as broken (candidate ≈ −250 Elo against the frozen champion from the first arena). Its trainer metrics are used here as one GUI data point, not as proof of a healthy run |
| GUI-F1 | Fresh GUI run: Build Network, then Play-and-Train | `dcm_log_20260921-000925.txt` (build 2114) | 858 `[STATS] elapsed=` lines, `steps=` 1 → 14,781. No `[LAYER-HEALTH]` (predates it) |
| GUI-F2 | Fresh GUI `--train` run | `dcm_log_20260927-211635.txt` (build 2131) | 532 `[STATS] elapsed=` lines, `steps=` 1 → 2,041. No `[LAYER-HEALTH]` |

**Evidence cutoff.** A and B were still training when this plan was measured. Every A and B number here stops at the row counts above; later rows are not in them. (B kept drifting after the cutoff: `valueFC1ZeroVel=27/128` at trainer step 33,000.) The incident tests freeze the same cutoff by keeping verbatim excerpts with their source hashes (Part X).

Run definitions: `experiments/20261005-lr-schedule-ab/README.md` (A, B, C) and `experiments/20261002-noSE-noReZero/README.md:52,70` (R7, R8 log names).

### Arm C: diverged at LR ≈ 3, then stayed broken without a single non-finite value

From `dcm_log_20261005-090417.txt` (segment 0):

| trainerStep | lr | loss | pLoss | pIllM | gNorm | pLogitMean | live dead / 1,040 | worst site |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 0.01 | 11.8138 | 8.6299 | 0.9938 | 31.172 | -- | 0 | none |
| 50 | 0.5 | 6.4783 | 5.0650 | 0.5131 | 1.235 | -0.0173 | 12 | value.bn (12 dead / 16) |
| 100 | 1.0 | 5.7369 | 4.5472 | 0.3179 | 1.502 | -1.3832 | 14 | value.bn |
| 150 | 1.5 | 5.1343 | 3.9693 | 0.2925 | 0.608 | -4.5106 | 24 | value.bn |
| 200 | 2.0 | 5.1376 | 3.9890 | 0.2893 | 0.621 | -5.6759 | 25 | value.bn |
| 250 | 2.5 | 4.7847 | 3.4011 | 0.2360 | 0.899 | -6.8423 | 28 | value.bn |
| 300 | 3.0 | 36.3806 | 34.1114 | 0.9971 | 14.325 | -6.2725 | 322 | policy.pre_bn (92) |
| 350 | 3.5 | 8.7093 | 6.8958 | 0.9458 | 0.059 | -0.8083 | 340 | policy.pre_bn |
| 400 | 4.0 | 8.6401 | 6.8210 | 0.9476 | 0.018 | -0.7581 | 340 | policy.pre_bn |
| 500 | 5.0 | 8.6092 | 6.7816 | 0.9446 | 0.057 | -1.1248 | 339 | policy.pre_bn |

Segment 1 (`dcm_log_20261005-121841.txt`, trainer steps 514–6,113):
- `pIllM` 0.928–0.968 throughout;
- `gNorm` 0.009–0.613: up to 0.6 only while the LR was near its peak of 10 (trainer steps 863–1,513); median 0.015 from 2,000 on, maximum 0.108;
- `vLoss` median 0.889;
- live dead channels 339 → 351 of 1,040; `nonFinite=0` on every line.

Checkpoint passes:
- at trainer step 513 (the abort save): `valueFC1ZeroVel=0/128`, `value.bn` 14 of 16 dead, `policy.pre_bn` 91 of 128;
- from trainer step 1,513 on: `valueFC1ZeroVel=128/128`, through trainer step 6,116.

`rvMaxOverMedian` reached 466,888.2 at `value.bn[12]` in the checkpoint at 513, and 548,521.0 (same channel) on the live line at trainer step 4,663 (`dcm_log_20261005-121841.txt:1579`), C's maximum.

What C shows:
- **The first damage came long before the divergence.** At step 50, still in warmup at LR 0.5, `value.bn` already had 12 of 16 channels dead, while the loss was falling normally. The README's "trained normally to LR 2" is true of the loss, not of the value head.
- The policy offset `pLogitMean` fell past −4.5 by step 150. That is well outside the healthy range (below) and also before the divergence.
- `nonFinite` was 0 on every live and checkpoint pass. The trainer's existing non-finite-loss halt (`Training/ChessTrainer.swift:7089-7103`) never had anything to catch.
- **The resumed segment never logged a diagnostic field.** All 113 `[REPLAY]` lines in segment 1 have `pEnt=--`, `pLogitMean=--` and `pD=--`.
  - The trainer computes the diagnostic outputs only when the next trainer step is a multiple of `batch_stats_interval` (`Training/ChessTrainer.swift:4649-4657`; interval 10 in `experiments/20261005-lr-schedule-ab/parameters-C.json`).
  - The `[REPLAY]` line is written at segment step 1 and when `segmentStep % 50 == 0` (`CLI/CorpusReplayRunner.swift:1840,1904`). Train-vs-UCI has the same cadence (`CLI/TrainVsUciRunner.swift:744,764`).
  - The resume starts at trainer step 513, so the rows fall on trainer steps 514, 563, 613, …, 6,113. None is a multiple of 10, while the segment's own `[BATCH-STATS]` lines (written on diagnostic steps) sit at 520, 530, 540, …. The diagnostics were computed; the log rows just never landed on them.
  - In general a row carries diagnostics only when `(resumeOffset + segmentStep) % batch_stats_interval == 0`. With an interval that divides 50 and an offset that is not a multiple of it, no multiple-of-50 row ever aligns; the segment-step-1 row aligns only when `resumeOffset + 1` is a multiple of the interval (e.g. offset 9 with interval 10). Intervals that do not divide 50 align on some rows.
  - The monitor in this plan records every step, so it sees every diagnostic step whatever the log cadence. The log-line cadence itself is a separate bug, fixed by `documentation/plans-active/STATS_LINE_RESUME_CADENCE_FIX_PLAN.md` (OD-12).

### Arm B: lost value-BN channels early, drifted its policy offset all run

From `dcm_log_20261005-013235.txt`:
- Live dead channels went 0 → 5 between trainer steps 250 and 300, all in `value.bn` (16 channels), and 5 → 6 between 1,250 and 1,300. They stayed at 6 to the end (32,600). Mostly-off channels flickered between 0 and 6.
- The `valueFC1ZeroVel` checkpoints read 10/128 at 1,000, 6/128 at 2,000, 9–15/128 through 20,000, 17–20/128 from 23,000, and 24/128 at 32,000.
- `pLogitMean` crossed −3 between steps 800 (−2.9331) and 850 (−3.4569). It reached −4.503 at 1,000, −6.8871 at 4,950, −8.9277 at 19,950 and −10.8165 at 32,150 (its minimum).
- No loss spike: the largest logged loss over the median of the ten previous logged values was 1.026.

Correction to the brief: B lost **5** of its 16 value-BN channels between steps 250 and 300 and a **6th** between 1,250 and 1,300. It did not lose all six at 250–300.

### Healthy baselines

| Metric (window) | A | B | R7 | R8 | GUI Ejp0 | Incident value |
|---|---:|---:|---:|---:|---:|---|
| `pIllM` max, trainer step ≥ 2,000 | 0.0649 | 0.0113 | 0.0263 | 0.0279 | 0.0015 | C 0.9682 |
| first trainer step with `pIllM` < 0.1 | 1,100 | 300 | 500 | 500 | — | C never (minimum 0.236 at 250) |
| `gNorm` min, ≥ 500 | 0.899 | 0.235 | 0.568 | 0.572 | 2.648 (≥ 2,000) | C median 0.015 from 2,000 |
| largest single logged loss ÷ median of the 10 before it | 1.021 | 1.026 | 1.023 | 1.029 | — | C 36.38 ÷ 5.44 = 6.69 at 300 |
| `pLogitMean` range | 0.017 … 0.581 | 0.327 … −10.817 | −0.044 … 0.757 | −0.005 … 0.760 | not logged | C −6.842 at 250 |
| live dead channels, max | 0 | 6 | no data | no data | no data | C 351 |
| `valueFC1ZeroVel` max | 0/128 | 24/128 | no data | no data | no data | C 128/128 |

**Fresh GUI runs (GUI-F1, GUI-F2).** Values are the `[STATS]` line's rolling means, not per-step samples, so they bound the per-step medians the monitor will see only loosely:
- `pIllM`: F1 0.9961 at step 1, 0.6212 at 500, 0.2262 at 1,004, 0.2180 at 2,033, 0.1115 at 6,021, 0.0420 at 14,028; maximum 0.2189 from 2,000 on. F2 0.6464 at 500, 0.2211 at 2,041. So rule 4's not-learned form (≥ 0.5 past warmup + grace = 2,000) has a margin of ≈ 2.3× on fresh GUI nets.
- `gNorm` from 2,000 on: F1 1.154–1.375, F2 1.051.

**Survey of every log since `[LAYER-HEALTH]` shipped** (2026-10-02 → 2026-10-05; 60 logs carry live lines, 73 carry any `[LAYER-HEALTH]` line):
- **Live dead channels.** Maximum 0 in every log except B (6), C (351) and `dcm_log_20261003-001701.txt` (17, a 3-step run branched from the grafted v5 model `20261003-18-AkMs`: already 17 at its first live evaluation, after one training step, and unchanged through its step-3 checkpoint. That fixes the process's baseline; it does not by itself prove the channels came in with the model, nor that they are damage, rule 2 note).
- **`valueFC1ZeroVel`.** At most 1/128 in a healthy 128-unit head (`dcm_log_20261002-202430.txt`, 33,000 steps). Short 16-unit test runs (corpus-replay `replay-final` saves at 3–62 trainer steps, e.g. `dcm_log_20261003-135725.txt`) showed 0–3/16. The branch run `dcm_log_20261003-001701.txt` showed 49/128 at its `replay-final` save after 3 steps (with so few trained steps, mostly the velocity the branch did not continue; below, and rule 3's gate).
- **16/16 where nothing was trained.** In 15 logs from 2026-10-03 (test processes), specific GUI `session-manual` / `session-promote` saves show `valueFC1ZeroVel=16/16` with the headline's trainer step anywhere from 0 to **274** (`dcm_log_20261003-153145.txt:108` is the 274). Those saves come from the test harness that advances the trainer clock without calling `trainStep` (`DrewsChessMachineTests/GuiSaveHarness.swift:163-177`): session ID `unknown`, every BN β/|γ| exactly 0. No velocity had accumulated, so every unit read exactly zero. (The same logs hold other tests' output too, including `[STATS]` lines and, in `dcm_log_20261003-134853.txt` and `dcm_log_20261003-135725.txt`, corpus-replay training; the claim is about those saves, not the whole logs.) Rule 3's minimum-history gate therefore counts steps **this process trained**, never the trainer clock (R1–R8).
- **`rvMaxOverMedian`.** Below 80 everywhere except the two runs that loaded older models (231 and 332), and C (548,521).

A condition of the GUI path that the replay incidents do not show: a fresh GUI net learns legality more slowly per step (in `dcm_log_20261001-011107.txt`, `pIllM` was 0.996 at step 1 and still 0.50 at step 326). Rule 4's not-learned form is gated with this in mind.

### Prototype replay of the rules below over these logs

The rules in Part R were prototyped in a scratch Python script, and re-derived independently by a second script during review with the same results. Neither is committed: validation uses the real evaluator through `--replay-health-log` instead (Part V). Both scripts use the sparse offline semantics of D4: each logged row is one evaluation with a one-record window, and site channel counts come from the logs' checkpoint tables. The table was re-run after the owner's decisions (OD-1 dead-channel critical at 5% overall / 20% per site; OD-7 value-loss rule dropped): only B's `dead_channels` changed, from a warning to **critical at 300**; A, R7 and R8 still raise nothing, and C's events are unchanged. It was re-run again after the owner also dropped the one-sided value-head rule (OD-7): the only change is that C no longer has an event at step 250; every other event of A, B, C, R7, R8 and the survey is unchanged, and B and C still raise their critical alarms (B `dead_channels` 300; C `dead_channels` 50, `illegal_mass` 350, `gradient_collapse` 400, `value_fc1_zero_velocity` 1,513). Results:

| Run | Event (rule, severity, trainer step) |
|---|---|
| A | none |
| R7, R8 | none (scalar rules only; these logs have no `[LAYER-HEALTH]`) |
| B | `dead_channels` **critical 300** (`value.bn` 5/16 = 31% ≥ 20%; owner OD-1 thresholds); worsened to 6 at **1,300**. `policy_offset_drift` warning **900** (−3.80). `value_fc1_zero_velocity` warning **1,000** (10/128, checkpoint). All three still active at the end. |
| C, one process | `dead_channels` **critical 50** (`value.bn` 12/16); worsened from 12 to 14 dead overall at 100 (worsen lines then rate-limited to one per check interval as the count keeps rising). `policy_offset_drift` warning 200 (cleared 400: −0.81 at 350 and −0.76 at 400 are the two clear evaluations). `bn_running_variance_runaway` warning 200 (1,415.5). `loss_spike` warning 300 (36.38 against a reference of 5.44; cleared 500). `illegal_mass` **critical 350** (regression: 0.9458 after a minimum of 0.236). `gradient_collapse` **critical 400** (0.018); cleared 913 while the LR was near 10, re-raised 1,713. `value_fc1_zero_velocity` **critical 1,513** (128/128). |
| C segment 1 as its own process | `dead_channels` critical 514 (339/1,040). `bn_running_variance_runaway` warning 514. `value_fc1_zero_velocity` critical 1,513. `gradient_collapse` critical 1,713. `illegal_mass` critical 2,063 ("not learned": the resumed process never saw the pre-divergence minimum). |

### Long runs (reported, not judged)

Every baseline above stops by ≈ 35,000 trainer steps. Two older long corpus-replay lines, replayed under the rules with the sparse semantics of D4 (rows only). These logs carry no `trainerStep=` and no `pLogitMean` / `vLogitMean`, so they replay with D4's legacy-row handling: `--segment-step-as-trainer-step`, and rules without their fields report no data. The logs do not record the trainer clock the start model carried, so the segment step may be offset from the true trainer step by an unknown constant. That cannot change the rule reported here: rule 6's reference uses only differences of trainer steps (spans), never an absolute step. Both started from an already trained model, ran mixed precision (bf16 working weights, fp32 masters) and predate the head-numerics fix (`da15920`, 2026-09-28), so these events may be true signals rather than false positives:

| Log | Rows (steps) | Rule 6 `loss_spike` |
|---|---|---|
| `dcm_log_20260702-201756.txt` (v5 line, 268,500 steps) | 5,371 | **21 raises** (first at 26,600; ratios 1.50–2.36) |
| `dcm_log_20260727-094049.txt` (qeu8 line, 1,397,600 steps) | 27,953 | 2 raises (125,350 ×1.50; 330,550 ×1.51) |

(The long-run replay also measured the dropped value-loss rule; its numbers were removed with the rule, OD-7.)

Whether these change the thresholds is OD-17. V-1 replays both and reports their events without a pass/fail verdict.

C's `dead_channels` critical at 50 needs `value.bn`'s channel count (16), which no live line carries: the offline replay takes it from the checkpoint table at trainer step 513 in the same log, read in a pre-scan (D4). Without the pre-scan the step-50 evaluation is a warning (12 of 1,040 is 1.2% overall).

---

# Part R — The rules

## R0. Definitions shared by every rule

- **Trainer step.** `ChessTrainer.completedTrainSteps` (`Training/ChessTrainer.swift:4357-4360`), the lineage-continuous clock, never the segment step. Gates and sustain spans are measured on it, so a resumed run is judged by how far its weights have trained.
- **Step record.** One per `trainStep`, built from `TrainStepTiming`:
  - Lean fields, valid on every step, that a rule reads: `loss` (rule 6), `illegalMassPenalty` (rule 4), `gradGlobalNorm` (rule 5) (`Training/ChessTrainer.swift:106,127,144`), plus `totalMs` for the `train_ms` field (D2). `policyLoss`, `valueLoss` and `sampledBatchDrawFraction` are not recorded: no rule reads them since OD-7 dropped the value-loss and one-sided value-head rules.
  - Diagnostic field, only when `hasDiagnostics` (`:308`): `policyLogitMean` (`:318`; rules 1 and 7).
  - `illegalMassPenalty` is in the trainer's lean readback targets (`Training/ChessTrainer.swift:6747-6750`), so it is measured on every step. The log agrees: every segment-1 `[REPLAY]` line of C carries `pIllM` while its diagnostic fields read `--`. (The comment above `dg` at `CLI/CorpusReplayRunner.swift:1829-1833` still lists illegal mass in the diagnostic bundle; it is stale.)
- **Evaluation.** A call to the evaluator with the window and, where available, a layer-health digest.
  - CLI paths: every 50 trainer steps, on overall trainer-step multiples (`TrainingHealthCadence.isLiveEvaluationStep`; OD-21), each right after its own live `[LAYER-HEALTH]` read — or the step line's read when a step line falls on the same step (cadence plan OD-15) — and after every checkpoint pass. The evaluation cadence is independent of the step-line cadence (`documentation/plans-active/STATS_LINE_RESUME_CADENCE_FIX_PLAN.md`), whose step lines are logging only. **The per-step order is: step-line block (when due) → live evaluation (when due) → save block**, so at every save step — every 1,000-step save is a multiple of 50 — the step's live evaluation runs before the save's checkpoint pass (the requirement R0 places on the loop; it holds by loop position, not by riding the line). The final save, at an arbitrary step, has a live evaluation before it only when that step is itself a multiple of 50; P2 adds one explicitly before the final save's checkpoint pass so the last partial window is judged. The monitor records every step, so a window is the 50 steps since the previous evaluation (a resumed run's first window is shorter: 514…550 after a resume at 513), every window holds diagnostic steps at any `batch_stats_interval` ≤ 50, and a process's first evaluation is at its first multiple of 50 or its first save, whichever comes first.
  - GUI: every 50 trainer steps on overall trainer-step multiples, from the trainer worker's step count (OD-21; P3 wires it), independent of the `[STATS]` emits (every 25 session steps for the first 500, then every 60 s, `App/SessionController+Training.swift:1435,2119-2124,1419,2177-2179`, which stay logging only); and after every checkpoint pass. Cost: one live read (a `graph.run` on the trainer queue reading BN γ/β/running stats and ReZero α, ≈ 1 ms) plus the summary on a GCD queue — about 0.1 ms for a ReLU tower, and for a SiLU / GELU tower one numerical integral per channel (estimated tens of milliseconds for 1,040 channels, measured by V-5's `cost_ms` and the summary's own timing); at the Ejp0 GUI pace (≈ 0.65 s/step, 32.5 s per 50 steps) well under 0.5% either way.
  - A **live evaluation** carries a window and the live digest. A **checkpoint evaluation** carries only the checkpoint digest: it never consumes the step window and never takes part in rewind detection.
- **Window `W`.** The step records since the previous live evaluation, in trainer steps `(previous, now]`.
  - `median_W(x)` and `max_W(x)` are over the records in `W` that carry `x`.
  - For a diagnostic field, `W` has data only if it holds at least one diagnostic step.
  - **A rule whose inputs have no data in `W` holds its state**: no raise, no clear, no sustain progress. The `[HEALTH] check` line counts these "no data" evaluations, so a silent rule is visible.
  - `W` is held whole, however long it is (D2). The monitor never truncates a window silently; if a bound is ever hit, the evaluation says so (`truncated=` on the `[HEALTH] check` line).
- **Steps trained by this process.** The number of step records this process's monitor has recorded since it started, less any span a trainer-clock rewind took back (below). Rule 3 gates on it. It is deliberately not the trainer clock: a clock can be set without training (the 16/16 survey case).
- **Layer-health digest.** `LayerHealthDigest`, built in one place from a `LayerHealthSummary` (or, for the offline replay, from the logged lines):
  - classified channels and the dead count (`LayerHealthSummary.deadChannelCount`, `Training/LayerHealth.swift:919`);
  - per classified site, its dead count and channel count;
  - `nonFiniteValueCount`;
  - the largest `runningVarianceMaxOverMedian`;
  - `valueFC1` zero-velocity units and total units (checkpoint tier only, `:883`).
- **Sustain `n` / span `s`.** The condition must hold on `n` consecutive evaluations that have data, spanning at least `s` trainer steps. Measuring both makes the rule independent of cadence: live evaluations are 50 trainer steps apart on every path (OD-21), checkpoint evaluations add more at saves, and the offline replay (D4) evaluates once per logged row, whatever the logged cadence was.
- **Learning gate.** `trainerStep ≥ lr_warmup_steps + training_health_learning_grace_steps` (defaults 1,000 + 1,000). It applies only to the "has not learned yet" form of rule 4. Every damage rule runs from the first evaluation.
- **Warmup and LR-cycle awareness: how and why.**
  - Warmup is handled by the learning gate above, which reads the run's own `lr_warmup_steps`.
  - LR-cycle peaks are **not** suppressed. Every incident here started at a rising LR: C's damage began in warmup at LR 0.5 and its divergence came at LR 3. Suppressing alarms near peaks would have hidden exactly these.
  - The cost of not suppressing was measured: B's 1.0 peaks produced no false loss-spike, gradient-collapse or illegal-mass event. Its three alarms are real damage.
  - Every raise and escalate line records the effective LR (`lr=`), so a reader sees the schedule context.
- **Hysteresis.** Raise thresholds and clear thresholds are separate, with a gap between them. Clear needs its own sustain.
- **Events.**
  - `raise`: the rule enters warning or critical.
  - `escalate`: warning → critical.
  - `worsen`: the measured count rose while the alarm was active (rate-limited to one line per check interval per rule).
  - `active`: a reminder every `training_health_check_interval_steps` while the alarm is active.
  - `clear`: the clear condition held for its sustain.
  - `stop`: the rule's action requested a stop.
  - A critical alarm never de-escalates to warning; it clears or stays critical.
- **Trainer-clock rewind.** A GUI promotion rewinds the trainer's weights and clock to the arena-start snapshot (`App/SessionController+Arena.swift:463`). Training runs during the tournament, so the clock really goes back.
  - The promotion calls `monitor.noteTrainerClockRewind(to:)` explicitly, under both gate pauses, next to `trainingBox?.resetRollingWindows()` (`App/SessionController+Arena.swift:490`), which is what the existing code does with its own rolling windows.
  - As a safety net, `recordStep` also treats a record whose trainer step is at or below the last recorded one as an unannounced rewind (D2). Checkpoint evaluations never detect a rewind (their trainer step may be old), but every evaluation, live or checkpoint, applies one that `recordStep` detected before it does anything else (D2).
  - On a rewind the monitor:
    - increments its **generation**;
    - discards the pending window and the loss-reference history;
    - takes the rewound span off "steps trained by this process". `a` is the last recorded trainer step and `b` the restored clock (the trainer's `completedTrainSteps` after the rewind; the next record will be `b + 1`). The count becomes `max(0, trained − (a − b))`, the steps this process trained into the restored snapshot (whose velocity the promotion restores, `App/SessionController+Arena.swift:455`). Example: 100 trained, last record 100, restored clock 50 → 50; the next record (51) makes it 51;
    - resets every rule's pending sustain progress (raise and clear), rule 4's regression running minimum, and every rule's newest applied trainer step (D2's freshness check; otherwise every post-rewind observation would look stale), because they described weights that no longer exist;
    - keeps active alarms, so they clear only by recovery;
    - logs `[HEALTH] trainer clock rewound <a> -> <b>; generation <g>; windows reset`.

## R1–R8. The rule table

The thresholds are **decided** (OD-1, both rounds; declared constants per OD-14). `Healthy` is the most extreme value seen in a healthy run (the tables above). `Fires on` is the prototype result.

| # | id | Data | Raise | Clear | Sustain | Healthy | Fires on |
|---|---|---|---|---|---|---|---|
| 1 | `non_finite` | live + checkpoint digest `nonFiniteValueCount`. A non-finite `pLogitMean` in `W` (the trainer's own halt does not check it) | critical: count > 0 | never auto-clears (non-finite weights do not heal) | none: raises on first sight | 0 in all 73 logs | none of the incidents (C stayed finite) |
| 2 | `dead_channels` | digest: **parked** channels (`BatchNormPassThrough`, OD-18) over every BN site an activation consumes — for relu / leaky_relu exactly the dead count (β/\|γ\| < −3, `Training/LayerHealth.swift:73`), for silu / gelu the same excess-pass-through line Φ(−3) — and per-site counts | warning: parked > 0 (absolute; OD-16). Critical: parked ÷ classified ≥ 0.05, or any site's parked ÷ channels ≥ 0.2 (OD-1). Every raise, escalate, worsen and reminder names **every** affected site with its counts (`sites=value.bn(5/16),…`, OD-22) | parked = 0 for 2 evaluations spanning ≥ 1 trainer step — two observed states, not the live and checkpoint reads of one save step (which also ends both critical arms) | none (dead is near-permanent) | 0 in every healthy run surveyed | B critical 300 (`value.bn` 5/16 = 31%). C critical 50. B-silu: 0 parked at 20,000; `policy.pre_bn` (leaky_relu) 20 at 21,000 and 21 at 22,000, `value.bn` 1 (checkpoint pass, OD-18 test) |
| 3 | `value_fc1_zero_velocity` | `valueFC1` zero/units, from a checkpoint digest or the dedicated value-FC1 read (D6), at least every 1,000 trainer steps on every path. Evaluated only once **steps trained by this process** (R0) ≥ 200 when the state was read (otherwise no data) | warning: ≥ 0.05. Critical: ≥ 0.5 | < 0.025 at one checkpoint | none | ≤ 1/128 (0.8%) | B warning 1,000 (10/128). C critical 1,513 (128/128) |
| 4 | `illegal_mass` | `median_W(illegalMassPenalty)` | critical, **regression form**, either arm (OD-19): (i) the run's own running minimum of window medians has been < 0.5, and now ≥ 0.8; (ii) now ≥ 0.3 **and** ≥ 10 × that running minimum. Critical, **not-learned form**: past the learning gate and ≥ 0.5 | < 0.15 for 2 evaluations (OD-19) | 2 evaluations, span ≥ 50 | replay ≤ 0.0649 from 2,000; fresh GUI ≤ 0.2189 from 2,000 (rolling mean, GUI-F1) | C critical 350 (arm i). B-silu critical 20,700 (arm ii: 0.7817 against a minimum of 0.0023). C segment 1: 2,063 (not learned) |
| 5 | `gradient_collapse` | `median_W(gradGlobalNorm)` | critical: < 0.1 | ≥ 0.2 for 2 evaluations | 2 evaluations, span ≥ 50 | min 0.235 (B); GUI ≥ 2.648 | C critical 400 (cleared 913 near LR 10, re-raised 1,713) |
| 6 | `loss_spike` | `max_W(loss)` and `median_W(loss)` against `ref` = median of the loss records in the 1,000 trainer steps before `W`. `ref` has data only when those records span ≥ 200 trainer steps (first to last) and number ≥ 5 — a span, not a record count, so the in-app per-step history and the sparse offline rows (D4) use one definition | warning: `median_W ≥ 1.5 × ref` or `max_W ≥ 3 × ref` | `median_W < 1.2 × ref` | none | logged ratio ≤ 1.029 | C warning 300 (×6.69) |
| 7 | `policy_offset_drift` | `median_W(\|policyLogitMean\|)` (diagnostic) | warning: ≥ 3.0 | < 2.0 for 2 evaluations | 2 evaluations, span ≥ 50 | ≤ 0.760 | B warning 900. C warning 200 |
| 8 | `bn_running_variance_runaway` | digest: largest `runningVarianceMaxOverMedian` | warning: ≥ 1,000 (OD-1, decided) | < 300 for 2 evaluations spanning ≥ 1 trainer step | none | < 80 in healthy runs (231, 332 in runs from older models) | C warning 200 (1,415.5) |
| 9 | `gradient_spike` (OD-20) | `max_W(gradGlobalNorm)` against `gref` = median of the gradient-norm records in the 1,000 trainer steps before `W`, with rule 6's reference definition (≥ 5 records spanning ≥ 200 trainer steps) | warning: `max_W ≥ 5 × gref` | `max_W < 2.5 × gref` | none | largest logged ratio 1.25 (A), 1.40 (B), 1.29 (R7), 1.30 (R8) | B-silu warning 20,600 (2.566 against 0.359, ×7.15). C warning 300 (×13.4) |

**Notes per rule (the measurements behind them):**

- **Rule 1.** The trainer already throws `nonFiniteLoss` on a non-finite loss, gradient norm, or (on diagnostic steps) `valueMean` or entropy (`Training/ChessTrainer.swift:7089-7103`). Every path treats that as fatal: the GUI suspends (`App/SessionController+Training.swift:1248-1268`), and the CLI paths propagate the error. This plan leaves that behavior alone. Rule 1 covers the tensors (BN state and ReZero α live; every tensor at checkpoints), which can go non-finite while the losses are still finite.
- **Rule 2.**
  - **Warning on any dead channel (absolute form; OD-16, decided).** No healthy run in the survey had one. A line that already carries a few dead channels therefore holds a permanent warning, with a reminder every check interval and a row in the GUI list: `dcm_log_20261003-001701.txt`, a 3-step branch of the grafted v5 model `20261003-18-AkMs`, shows 17 of 1,808 dead, 1–4 per site across 8 sites, `value.bn` 2 of 16. The owner accepted that for now and will revisit it; the alternative ("new-damage" form) is written down, not built, under **Deferred (OD-16 revisit)** after Part R.
  - The critical arms are absolute: C (and C's segment 1 alone, 339 of 1,040 at its first evaluation) is critical, and so is B at 300 (`value.bn` 5 of 16, 31% ≥ 20%).
  - The `[ALARM] health` lines name every affected site with its counts (`sites=…`, largest fraction first; OD-22), so the damage's location is visible.
  - Mostly-off and always-on are **not** alarms. B's mostly-off count flickered 0–6 for the whole run with no other symptom. The `dcm_log_20261005-012743.txt` benchmark (an older model) had 12 always-on channels and was otherwise healthy.
  - The per-site critical threshold (0.2, owner OD-1) is reached first in C, by `value.bn` (12 of 16 at step 50). B's `value.bn` reached 5 of 16 (31%) at 300 and 6 (38%) at 1,300, so B is **critical from 300**. Under the earlier proposal (0.5) B was a warning; the owner chose the stricter threshold. The overall arm (5%) is reached by C at 300 (322 of 1,040) and by C's segment 1 at its first evaluation (339 of 1,040); B's 6 of 1,040 (0.6%) is far below it.
  - In-app the digest has every site's counts. Offline, between checkpoints, only the live line's `worst=` site is known, and `worst` is the site with the most dead + mostly-off + always-on channels (`Training/LayerHealth.swift:923-925`), not necessarily the most-dead site. C's step-513 checkpoint shows the difference: `worst=policy.pre_bn` (91 of 128 dead) while `value.bn` had 14 of 16. The offline per-site arm is therefore a lower bound between checkpoints (D4).
  - Classification applies only to `relu` / `leaky_relu` BN sites (`Training/LayerHealth.swift:274-283`). A tower whose sites are all SiLU/GELU has no classified sites, and the rule logs that it does not apply. It never reads that as "0 dead".
- **Rule 3.**
  - Exact-zero velocity lags death. Momentum shrinks a dead unit's velocity by μ per step, so exact fp32 zero arrives hundreds to a couple of thousand steps after the unit dies, depending on μ (0.85–0.95 in these runs) and on whether the GPU flushes subnormals.
  - C shows the lag: 0/128 at the step-513 abort save, 128/128 at 1,513.
  - Velocity that has not accumulated reads as zero: a save before any training in the process, or soon after a branch whose velocity was not continued, shows every unit at exactly zero — including saves whose trainer clock reads up to 274 because a test set it (the 16/16 survey case). Hence the gate on **steps trained by this process** (R0), never on the trainer clock. C's step-513 abort save (513 trained) and every 1,000-step checkpoint pass it; every 16/16 save in the survey trained 0.
  - The velocity is `v ← μ·v + clipped gradient` (`Training/ChessTrainer.swift:4195-4199`). With μ = 0 it is the last step's gradient, so exact zero then means the unit had no gradient on that one batch: still a strong signal at batch 4,096, but with no lag. The `[HEALTH] config` line records the run's base μ, and every live raise and escalate line its effective μ (`mom=`), since A–C cycle momentum. A checkpoint evaluation's lines carry `mom=--` (no live value at hand).
  - The live tier deliberately reads no velocity (`Training/LayerHealth.swift:113-114`), and GUI saves are hours apart. So that the rule runs on a fixed interval everywhere (OD-15, decided), the value-FC1 velocity is read every 1,000 trainer steps: from a save's checkpoint pass where one covers the interval (every corpus-replay autosave; train-vs-UCI enumerated checkpoints), otherwise by one dedicated read of that tensor (D6).
  - **Applies only to a ReLU value hidden layer.** Exact-zero velocity marks a dead ReLU unit; with leaky ReLU, SiLU or GELU the gradient is almost never exactly zero, so the rule would never fire and would mean nothing. Rule 3 therefore evaluates only when the value FC1 layer's activation, as `LayerHealth.valueFC1Layer(for:)` reports it, is `relu`. Today that function takes the tower-level `arch.activationFunction` (`Training/LayerHealth.swift:245-252`) and ignores any value-head-specific setting. Otherwise the rule is `not_applicable(activation=<fn>)`, stated once on the `[HEALTH] config` line, never raised, and D6 schedules no dedicated reads. `HEAD_ACTIVATIONS_PLAN.md` adds a value-head hidden activation; whichever plan lands second makes `valueFC1Layer(for:)` return it, so rule 3 follows it from one source (Risks). Test: `testValueFC1RuleNotApplicableForNonReluActivation`.
  - SE FC1 zero velocity is reported, not alarmed: SE bottlenecks with many weak units were measured in healthy runs (`Training/LayerHealth.swift:9-17`).
- **Rule 4.**
  - The regression form catches C's divergence at the first evaluation after it (the prototype fires at 350; the in-app per-step window would fire at the first evaluation whose window median is ≥ 0.8).
  - Arm (ii) (OD-19) catches a regression from a well-learned state that stops short of 0.8: B-silu learned to 0.0023 (trainer step 19,000), then jumped to 0.7988 at 20,650 and 0.7817 at 20,700 with the LR near its 0.98 peak, and decayed back over the next 700 steps (0.4014 at 21,000, 0.1422 at 21,350). Arm (i) alone never fires on it (0.7988 < 0.8). Arm (ii) alone would miss C (its minimum 0.236 × 10 = 2.36 is out of reach), so both arms stay (the owner first proposed replacing arm (i); corrected the same day, OD-19). The prototype finds neither arm firing on A, B, R7 or R8.
  - Clear < 0.15 (was 0.3): a cleared alarm must sit clearly below the lowest level that raises it, which arm (ii) lowered to 0.3; 0.15 is half of it, the 2:1 hysteresis gradient_collapse uses (0.1 / 0.2). B-silu clears at 21,400 (0.1422 at 21,350, 0.1187 at 21,400). It also applies to the not-learned form (raise 0.5): a fresh GUI net's rolling `pIllM` was 0.1115 at 6,021 (GUI-F1), so such an alarm clears a few thousand steps later than under 0.3; the default is log-only.
  - The not-learned form waits for the learning gate because a fresh GUI net is still at 0.50 at step 326. On the replay runs the slowest healthy value at 2,000 was 0.065.
  - It overlaps the GUI's legal-mass probe (`App/SessionController+Training.swift:2282-2423`, 60 s probe cadence, aborts only GUI `--train`), which uses an inference copy of the network. Both stay until OD-9.
- **Rule 5.**
  - The floor 0.1 is 2.35× below the lowest healthy logged `gNorm` (B at its LR peaks) and 26× below the GUI run's minimum.
  - It is an absolute number in loss-gradient units, so an architecture or loss-weight change can move it. It is a declared constant, revisited per OD-1.
  - C's clear at 913 is the honest result of a symmetric rule: with the LR near 10, a broken net still produced `gNorm` 0.2–0.6. The other critical rules (2, 3, 4) stayed active through that window.
- **Rule 6.**
  - The logged ratios are single steps against a median of ten logged steps. Per-step maxima over a 50-step window will be larger.
  - Offline, the reference holds at most 20 logged rows (one per 50 steps in the prior 1,000), and needs 5 spanning ≥ 200 steps. C's step-300 spike has exactly 6 (steps 1–250, median 5.44), so the offline replay reproduces it under the same definition the app uses.
  - The `max_W ≥ 3 × ref` arm was set before per-step data existed. The owner decided 1.5× / 3× (OD-1) and keeping the thresholds (OD-17); V-2 measures the per-step distribution on a healthy run and **reports** it. It never changes the threshold: a change needs a new owner decision.
- **Rule 7.** In a model with an fp32 policy tail, `pLogitMean` is invisible to the softmax and to the loss. Drift means the loss path is not centered, or the head has a large shared bias riding on always-on channels (`Training/LayerHealth.swift:18-23`). It is a warning signal, never critical. A model trained before the head-numerics fix (`da15920`) can carry a large offset from its first step: the nT8Y-line benchmark `dcm_log_20261005-012743.txt` reads −14.26, and rule 7 warns on it (V-1).
- **Rule 8.** The threshold (1,000) was decided in OD-1's second round. Measured `rvMaxOverMedian`: A max 7.2 up to the Evidence cutoff (7.4 at trainer step 34,250, after the cutoff; the owner's note quotes 7.4); B max 78.6 at 1,400 and at most 36.3 after 5,000; C 461,870–548,521 on its live lines from trainer step 500 on (first above 1,000 at 200: 1,415.5); the two older-model runs 231 and 332. The healthy maximum was 78.6 (B). The runs from older models reached 231–332 without other symptoms. 1,000 is 3× the highest non-incident value.
- **Rule 9 (OD-20).** A one-step gradient blow-up that the pre-clip clip absorbs leaves no trace in the loss for a while; B-silu's illegal-mass regression at 20,650 was preceded by `gNorm` 2.566 at 20,600 against a reference near 0.36 (LR 0.88 on the way to its 0.98 peak). The reference is built like rule 6's (same span definition, so the in-app per-step history and the sparse offline rows share it). Healthy runs' largest logged ratio (sparse rows, prototype over the Evidence cutoff) is 1.40 (B), so 5× has a 3.5× margin; per-step maxima over a 50-step window will be larger than the logged single steps, which V-2 measures (`gradMaxRatio=` on the `[HEALTH] check` line) and reports without moving the threshold. Warning only, no sustain, clears immediately below 2.5×.

## Deferred (OD-16 revisit) — the new-damage warning form, not built

The owner decided the absolute form for now (OD-16) and will revisit it. Nothing in this section is implemented, tested or wired; it is kept so the revisit starts from a worked design.

- *Absolute form:* warn on any dead channel, because no healthy run in the survey had one.
- *New-damage form:* the first digest this process sees, live or checkpoint (whichever comes first; both carry every site in-app), fixes each site's dead count as its **baseline**. It is logged once, in full, as `[HEALTH] baseline dead_channels=<dead>/<classified> at trainerStep=<s> sites=<site>:<dead>/<ch>,…` (every classified site with a nonzero count; `sites=none` when all are 0), so every later comparison is auditable. The rule warns when any site's count rises above its baseline. The baseline is per process (offline: per evaluator, D4) and survives a trainer-clock rewind (the restored snapshot is from the same process).
- Offline the baseline is only partly known, because a live line names only its worst site (D4). The offline replay compares the **total** dead count against the first observation's total (exact), and a site's count against its baseline only when that site's count is known at the first observation (the first live line's worst site, or every site when the first observation is a checkpoint table). A site first seen later has an unknown baseline: its per-site comparison is no data for the rest of the evaluator, never a backdated or zero baseline. The output header lists those sites.
- Why both: a long-trained line may simply carry a few dead channels. `dcm_log_20261003-001701.txt` is a 3-step branch of the grafted v5 model `20261003-18-AkMs` and shows 17 of 1,808 dead, 1–4 per site across 8 sites, `value.bn` 2 of 16. Nothing in the logs says whether that is damage or the normal state of a long-trained net. Under the absolute form every resume of that line holds a permanent warning, with a reminder every 1,000 steps and a permanent row in the GUI list; under the new-damage form it warns only if the count grows.
- If adopted, the form would be a declared constant like the thresholds (OD-14), not a parameter.
- The critical arms are absolute in both forms, so C (and C's segment 1 alone, 339 of 1,040 at its first evaluation) is critical either way, and B's critical alarm at 300 (`value.bn` 5 of 16, 31% ≥ 20%) is the same under both (its first evaluation had 0).
- Its clear condition: every site at or below its baseline **and** neither critical arm holding, so a critically damaged baseline never clears by staying unchanged.
- Its tests (not written): `testDeadChannelsNewDamageFormIgnoresTheBaseline`, `testDeadChannelsNewDamageFormWarnsWhenASiteGrows`, `testDeadChannelsBaselineSurvivesARewind`, `testDeadChannelsCriticalBaselineNeverClearsByStayingUnchanged`, `testDeadChannelsBaselineFromAFirstCheckpointDigest`, `testOfflineBaselineUnknownForASiteFirstSeenLater`.
- Under it, `dcm_log_20261003-001701.txt` would give nothing (17 dead at its first live line and at its step-3 checkpoint).

## R2. Actions

Each rule has one action parameter (Int):

| Value | Name | Meaning |
|---|---|---|
| 0 | `log` | log, record and show only (**default for every rule**, per the owner's request) |
| 1 | `stop_on_critical` | additionally stop the run while the rule is active at critical |
| 2 | `stop_on_any` | additionally stop the run while the rule is active at any severity |

- **When a stop is requested.** One pure function decides, for every path: `TrainingHealthStopPolicy.firstQualifying(active:actions:)` returns the first active alarm (rule order) whose severity qualifies under its rule's **current** action, or nil.
  - CLI: the evaluator applies it at the end of every evaluation with the run's actions, which never change during a run; a non-nil result is the evaluation's `stopRequest`, and a `stop` event is logged.
  - GUI: the decision is made on the main actor, after **every** evaluation's hop (live and checkpoint, whether or not the evaluation itself asked for a stop), from the monitor's current active set and the actions read from `TrainingParameters.shared` at that moment (R3). So the action in force when the result arrives governs in both directions: changing an active rule's action to a stop action stops at the next delivered evaluation, including a detached checkpoint pass that was evaluated under an older `log` action; changing it to `log` before delivery means no stop.
- Severity levels: rules 1, 4 and 5 are critical only; 2 and 3 have both; 6–9 are warning only. For 6–9, `stop_on_critical` never stops; the Health tab says so next to those rules.
- A stop is requested once per monitor. Later events are still logged.

## R3. What stop does, per path

- **Corpus replay.**
  - The stats tick or checkpoint pass that requests the stop sets a run-local `healthStop` value. A dedicated value, not the SIGINT `ReplayAbortFlag`: reusing that flag would make the next Ctrl-C force-kill the process (`CLI/CorpusReplayRunner.swift:926-937`).
  - The loop checks it next to `abort.isRequested` (`:1849-1853`) and breaks before the next step. The saved state is therefore the state at the step whose evaluation requested the stop (or at the checkpoint whose pass requested it). It logs `[REPLAY] training health alarm <rule> requested a stop — stopping at step N`.
  - A stop requested by the **final** save's own checkpoint pass is logged and recorded in `alarms`, but changes neither the termination reason nor the exit status: the run was already ending.
  - The final save runs with reason `health-stop` (the `finalReason` choice at `:2008`) and goes through the existing failure handling (`rollingSaveFailures.requireLastSaveSucceeded`, `:2018`).
  - `results.json` gets `termination_reason: "training_health_alarm"` (the choice at `:2032`).
  - Exit status: per OD-4. The recommendation is a new status **35**, "stopped by a training-health alarm after a successful final save", added to the doc at `:911-912` and to `--help`. A chain script must not mistake an alarm stop for a completed run. (4 is taken: `--show-default-parameters` misuse exits 4, `App/DrewsChessMachineApp.swift:831`. 35 appears nowhere in the source: not as an `exit(…)` literal, not among the `fail(_:_:)` helpers' codes, and not as `--validate-corpus`'s computed `worstExit`, which is 0 or 1.)
- **Train-vs-UCI.**
  - The same flag is checked at the loop top (`CLI/TrainVsUciRunner.swift:746`).
  - The final session save uses a new `TrainVsUciSession.SaveKind.healthStop` (raw value `health-stop`, disk tag `vsuci-health-stop`; `CLI/TrainVsUciSession.swift:33-42`), so the folder name says why the run ended.
  - The termination reason and exit status are as for corpus replay (`CLI/TrainVsUciRunner.swift:871-889`).
- **GUI, interactive.**
  - The run is **suspended, not torn down**, as the GUI already does for a divergence (`suspendTrainingOnDivergence`, `App/SessionController+Training.swift:2489-2509`): the banner stays up, and self-play, heartbeat and stats keep running. Like that path, `suspendTrainingForHealthAlarm` closes the active training segment (`checkpoint?.closeActiveTrainingSegment(reason: "health-suspend")`, as `:2507` does with `"diverge-suspend"`) and logs one `[HEALTH] training suspended …` line.
  - `trainingSuspendedByDivergence: Bool` (`App/SessionController.swift:289`) becomes one `trainingSuspension: TrainingSuspension?` with cases `.divergence(reason)` and `.healthAlarm(rule, detail)` (single source of truth; OD-5). The gates read the case:

    | Gate | `.divergence` | `.healthAlarm` |
    |---|---|---|
    | arena skip (`:1370-1373`; the log line names the case: `[ARENA] skipped — training suspended (divergence)` / `(health alarm <rule>)`) | yes | yes: a damaged trainer must not become a candidate |
    | periodic autosave skip (`App/SessionController+Heartbeat.swift:376`) | yes | **no**: the weights are finite and a save is useful for forensics |
    | heartbeat alarm evaluation skip (`:645`) | yes | no |
    | legal-mass banner guard (`App/SessionController+Training.swift:2394`) | yes | yes |
    | Train ▸ Promote Trainee Now (`App/SessionController+ManualPromote.swift:35-52`; menu `App/DrewsChessMachineApp.swift:683-684`) | **yes** (OD-5, decided; not gated today, Risks) | yes |

    - Promote Trainee Now, both cases: `promoteTrainerNow` refuses through `onRefuseMenuAction` naming the suspension (divergence reason or health rule), and the menu item is disabled while `trainingSuspension` is set — the arena's reason. Without this a NaN or damaged trainer could become the champion (for `.healthAlarm` the parked worker would even acknowledge the promotion's training pause).

  - On every evaluation hop the main actor runs `TrainingHealthStopPolicy.firstQualifying` against the current actions (R2). When it returns an alarm and the run is not already suspended, it calls `suspendTrainingForHealthAlarm(rule:)`, which sets `trainingSuspension`, logs the `stop` event, and asks the monitor to park the worker (`monitor.requestPark()`, a flag in the step store). The trainer worker checks that flag at the top of each iteration, next to the pause gate (`App/SessionController+Training.swift:1199-1210`), and then **parks**; it does not return.
    - Why it must not return as the divergence path does (`:1267-1268`): every session save pauses training and waits for the worker to acknowledge (`App/SessionController+Checkpoint.swift:432-437`, `Training/WorkerPauseGate.swift:72-79`; the worker acknowledges at `App/SessionController+Training.swift:1201-1210`). With the worker gone, every save — the periodic autosave this suspension allows, a promotion save, File ▸ Save Session — would time out and abort.
    - Parked = a loop that takes no training step: `while !Task.isCancelled` it services the pause gate exactly as the loop top does (`markWaiting` while a pause is requested, `markRunning` after), and otherwise sleeps 100 ms in `do { try await Task.sleep(for: .milliseconds(100)) } catch { return }` (the only error is cancellation; no `try?`). Stop cancels the task group, which ends the parked loop.
    - The divergence path keeps returning (its weights are non-finite and its saves are gated off); this plan does not change it. Observed while reviewing: a File ▸ Save Session during a divergence suspension therefore times out at the training pause today. That is existing behavior, outside this plan, recorded under Risks.
  - An arena already running when the suspension begins finishes normally (its candidate is the arena-start snapshot, `App/SessionController+Arena.swift:164-188`); its pauses are acknowledged by the parked worker. If it promotes, the promotion rewinds the trainer as usual (R0) and the suspension stays until Stop.
  - Stop, then Start (continue after Stop), starts a fresh monitor (T8) and clears the suspension (as `:524` does today). If the damage is still there and the rule's action still stops, the first evaluation re-raises it and suspends again; to keep training a damaged trainer, set that rule's action to `log` first.
- **GUI `--train`.**
  - The stop takes the run's termination claim and calls `AutoTrainTermination.writeResultsAndExit(reason: .trainingHealthAlarm, trigger: "training health alarm <rule> at trainerStep=N", elapsed:)`. This is exactly the legal-mass-collapse path (`App/SessionController+Training.swift:2418-2421`; `App/AutoTrainTermination.swift:51-58,106-112`).
  - Like that path it writes no session save before exiting (OD-13).
  - Like that path it exits with status 0 (`Darwin._exit(0)`, `App/AutoTrainTermination.swift:111`) — every GUI `--train` termination does. Whether GUI `--train` should adopt OD-4's status 35 is part of OD-4.

---

# Part D — Design

## D1. Types (new files)

`Training/TrainingHealth.swift` — pure, no GPU, no files, no logging:

```swift
/// One training-health rule. The raw value is the stable id used in log
/// lines, results.json and parameter ids.
enum TrainingHealthRule: String, CaseIterable, Codable, Sendable {
    case nonFinite = "non_finite"
    case deadChannels = "dead_channels"
    case valueFC1ZeroVelocity = "value_fc1_zero_velocity"
    case illegalMass = "illegal_mass"
    case gradientCollapse = "gradient_collapse"
    case lossSpike = "loss_spike"
    case policyOffsetDrift = "policy_offset_drift"
    case batchNormRunningVarianceRunaway = "bn_running_variance_runaway"
    case gradientSpike = "gradient_spike"                       // OD-20
}

enum TrainingHealthAction: Int, CaseIterable, Codable, Sendable {
    case log = 0, stopOnCritical = 1, stopOnAny = 2
}

/// Resolved from a parameter snapshot, in one place for every path:
/// `TrainingHealthConfig(_ snapshot: TrainingParametersSnapshot)` (added in
/// P2 with the parameters; P1 uses the memberwise initializer in tests).
/// Encoded as results.json's `alarm_config` with explicit snake_case
/// CodingKeys (`enabled`, `check_interval_steps`, `learning_grace_steps`,
/// `lr_warmup_steps`, `momentum_coefficient`, `actions` as
/// `{rule id: action name}`).
struct TrainingHealthConfig: Sendable, Equatable, Encodable {
    let enabled: Bool
    let checkIntervalSteps: Int
    let learningGraceSteps: Int
    let lrWarmupSteps: Int
    let momentumCoefficient: Double                           // the base μ, recorded for rule 3's reading, never a condition
    let actions: [TrainingHealthRule: TrainingHealthAction]   // total over allCases
}

/// The thresholds — the one declaration (`enum TrainingHealthThresholds`,
/// `static let` per value, as `LayerHealth` and `TrainingAlarmController`
/// declare theirs).

struct TrainingHealthStepRecord: Sendable { /* trainerStep + lean + optional diagnostics */
    init(timing: TrainStepTiming, trainerStep: Int)
}

struct LayerHealthDigest: Sendable, Equatable {
    enum Tier: String, Sendable { case live, checkpoint }
    init(summary: LayerHealthSummary)            // the in-app source
    // the offline replay builds it from parsed lines (D4)
}

struct TrainingHealthObservation: Sendable {
    let trainerStep: Int
    let window: TrainingHealthWindowStatistics?   // nil for a checkpoint-only evaluation
    let layerHealth: LayerHealthDigest?
    let stamp: TrainingHealthStamp                // D2: run, generation, steps trained (rule 3's gate), taken before the data was read
    let effectiveLearningRate: Double?            // recorded on lines, never a condition
    let effectiveMomentum: Double?                // recorded on lines (`mom=`), never a condition; nil for a checkpoint evaluation
}

struct TrainingHealthEvent: Encodable, Sendable, Equatable {   // snake_case CodingKeys: trainer_step, learning_rate, …
    enum Kind: String, Codable, Sendable { case raise, escalate, worsen, active, clear, stop }
    let kind: Kind
    let rule: TrainingHealthRule
    let severity: TrainingAlarm.Severity           // reuse: Training/TrainingAlarm.swift:4-7 (gains Codable, below)
    let trainerStep: Int
    let since: Int?                                // trainer step the alarm was raised
    let value: String                              // the measured value, rendered
    let threshold: String                          // the crossed threshold, rendered
    let detail: String                             // e.g. worst site
    let action: TrainingHealthAction
    let learningRate: Double?
    let momentum: Double?                          // `mom=` on the line; nil (`--`) for a checkpoint evaluation
}

struct TrainingHealthEvaluator: Sendable {
    private(set) var state: State                  // per-rule sustain counters, running min/max, active set
    mutating func evaluate(_ observation: TrainingHealthObservation,
                           config: TrainingHealthConfig) -> TrainingHealthEvaluation
}
struct TrainingHealthActiveAlarm: Sendable, Equatable {
    let rule: TrainingHealthRule
    let severity: TrainingAlarm.Severity
    let since: Int                                 // trainer step it was raised
    let value: String                              // latest measured value, rendered
    let detail: String
    let action: TrainingHealthAction
}
struct TrainingHealthEvaluation: Sendable {
    let events: [TrainingHealthEvent]              // deterministic order: rule order, then kind
    let stopRequest: TrainingHealthEvent?          // R2: first qualifying active alarm, once per monitor
    let active: [TrainingHealthActiveAlarm]
    let noDataRules: [TrainingHealthRule]
}
```

- `TrainingAlarm.Severity` is reused, so there is one severity type. It is a `String` raw-value enum declared `Sendable` only (`Training/TrainingAlarm.swift:4-7`); `TrainingHealth.swift` adds `extension TrainingAlarm.Severity: Codable {}` (raw-value coding: `"warning"` / `"critical"`), so `TrainingHealthEvent` can synthesize its encoding. `TrainingAlarm` itself is unchanged.
- `TrainingHealthRule` and `TrainingHealthAction` encode as their raw values; in JSON an action is written by name (`log`, `stop_on_critical`, `stop_on_any`) through an explicit `encode(to:)`, because its persisted parameter value is the Int raw value.
- `TrainingHealthWindowStatistics` holds the medians and maxima of R0, computed by the monitor (D2). The evaluator never sees raw rings.

## D2. The monitor

`Training/TrainingHealthMonitor.swift`: `final class TrainingHealthMonitor: @unchecked Sendable`. One monitor per run (CLI process; GUI Play-and-Train start), with a `runID: UUID`. Two `SyncBox`es (`Utils/SyncBox.swift`, the project standard over `OSAllocatedUnfairLock`; `SyncBox` runs its closure under the lock and has no lock upgrade, `:25-37`). Lock order is always evaluation → steps. Nothing ever takes the evaluation lock while holding the steps lock.

**Stamps.** `observationStamp() -> TrainingHealthStamp` returns `{ runID, generation, stepsTrainedByThisProcess, lastRecordedTrainerStep }`, read under the steps lock alone. The **generation lives in the step store** (its one home): a rewind increments it under the steps lock, so a stamp taken after a rewind always carries the new generation. Every observation carries the stamp taken **before** its data was read:
- a live evaluation takes it before the live `[LAYER-HEALTH]` read;
- a checkpoint evaluation takes it when the save exports its trainer state (GUI: under the save's training pause; for the promotion save, after the rewind in the same pause; CLI: right after the export, inline). The detached GUI pass carries it to `evaluateCheckpoint`. Rule 3's gate uses the stamp's `stepsTrainedByThisProcess`, never the count when the pass finishes.

**`steps: SyncBox<StepStore>` — the hot path.**
- `recordStep(_ timing: TrainStepTiming, trainerStep: Int)` appends one record (trainer step, the three lean floats, the step's `totalMs`, and `policyLogitMean` when `hasDiagnostics`; about 32 B) to the **pending window** and increments "steps trained by this process". It holds only this lock, for one append: one uncontended lock per SGD step. It never takes the evaluation lock.
- **Rewind safety net.** If the record's trainer step `r` is at or below the last recorded step `a`, `recordStep` performs the step-store half of R0's rewind right there, under the steps lock it already holds: generation + 1; pending window discarded; with restored clock `b = r − 1`, steps trained becomes `max(0, trained − (a − b))`, then the record is appended (+1). (Example: last 100, record 51 → restored 50 → 50 trained → 51 after the append.) It also stores `unannouncedRewind = (a, b)`. It never takes the evaluation lock.
  - The evaluation-side half (loss-reference history, pending sustain, regression extrema, per-rule freshness steps) belongs to the evaluation state, which keeps `appliedGeneration`. **Every** evaluation, live or checkpoint, starts by comparing it with the step store's generation (evaluation → steps, in order); if it is behind, it applies the evaluation-side reset, logs `[HEALTH] trainer clock rewind detected by recordStep (not announced): <a> -> <b>; generation <g>` from `unannouncedRewind`, and only then validates its own stamp. So a delayed checkpoint stamped before the rewind is rejected by the generation check whichever evaluation runs first, and no evaluation ever judges discarded weights.
  - In the GUI the announced path below always runs first (under the training pause, before any post-rewind step), so the safety net is for a future rewind path that forgets to announce.
- The pending window grows until a live evaluation drains it: the 50 steps between two live evaluations on every path (OD-21). A hard cap of 65,536 records (≈ 2–2.5 MB at a 32–40 B record) guards against evaluations that never come (a GUI suspension, a stalled stats path); past it the oldest records are dropped and counted, and the next `[HEALTH] check` line reports `truncated=<n>`. Nothing is dropped silently.

**`evaluation: SyncBox<EvaluationState>` — serializes every evaluation end to end.** It holds the evaluator (with, per rule, the newest trainer step it has applied), the loss-reference history, the counters for the `[HEALTH] check` line, and the stop-request flag.
- `noteTrainerClockRewind(to:)` takes evaluation → steps and applies both halves of R0's rewind at once, with `a` = the last recorded step and `b` = `to` (generation + 1, pending window and loss-reference history discarded, steps trained reduced by the rewound span, pending sustain, regression extrema and per-rule newest applied steps reset). Observations stamped with the old generation are then rejected by the generation check, so resetting the freshness steps cannot let them in.
- `evaluateLive(stamp:layerHealth:digestTrainerStep:learningRate:momentum:config:log:)`, all under the evaluation lock (`learningRate` and `momentum` are the live effective values the caller already has; they go into the observation and onto the event lines, never into a condition):
  1. apply the evaluation-side half of an unannounced rewind, if the step store's generation is ahead of `appliedGeneration` (above);
  2. under one brief inner steps lock: if `stamp.generation` is not the step store's current generation, the observation describes weights that no longer exist — log `[HEALTH] stale live observation ignored: …` after the lock, count it, change nothing (the pending records stay for the next evaluation); otherwise, in the same lock section, do step 3. Validation and extraction are therefore one atomic step against `recordStep`;
  3. take out of the pending window exactly the records with trainer step ≤ the **boundary** and leave newer ones pending. The boundary is the live digest's own trainer step (`LayerHealthLiveState.completedTrainSteps`, read on the trainer queue with the tensors, `Training/ChessTrainer.swift:5830`), or, when the live read failed, the stamp's `lastRecordedTrainerStep`. That case is explicit, not silent: the failed read is already logged (`[LAYER-HEALTH] live read failed: …`, `Training/LayerHealthLog.swift:33-34`), the evaluation logs `[HEALTH] live read failed at trainerStep=<s>; window boundary = last recorded step <b>; layer-health rules no data`, and the `[HEALTH] check` line counts it (`liveReadFailed=<n>`). So the window and the digest describe the same weights, and records the worker appends meanwhile belong to the next window;
  4. compute the window statistics and the loss reference. Normally that is sorting ≈ 50–100 window values and ≤ 1,024 reference values (expected well under a millisecond); at the 65,536-record cap it is larger (expected a few milliseconds). These are estimates; the measured figure is the `cost_ms` field below, checked in V-5. Either way it holds only the evaluation lock, so the trainer's `recordStep` never waits on it;
  5. run the evaluator transition **on a copy** of the evaluator (it is a value type, D1), applying each rule only if the observation's trainer step is at or above that rule's newest applied step (below). Nothing is logged, recorded or published yet;
  6. **commit**, under one brief inner steps lock: if the step store's generation still equals the stamp's, replace the evaluator with the copy, set the stop-request flag if the copy requested one (used by the CLI; the GUI decides on the main actor, R2), and append the window's `(trainerStep, loss)` pairs to the **loss-reference history**, a ring of 1,024 entries that therefore always covers the 1,000 trainer steps before the next window — kept apart from the window, so a long window can never push the reference out. If the generation moved (an unannounced rewind happened in `recordStep` while statistics were being computed), the copy and the window are discarded, the evaluation is counted stale and logged as such, and nothing else changes. While the commit holds the steps lock no rewind can start, and a rewind that already happened is seen, so an evaluation never commits against a generation it did not validate;
  7. only after a successful commit: render the event lines and pass each, with its event kind, to the caller's `log` sink before releasing the lock, so log order is evaluation order. The GUI sink is `SessionLogger.shared.log` (a non-blocking enqueue, `Logging/SessionLogger.swift:198-203`); the CLI sink is the runner's `emit` (stdout + session log) plus stderr for raise, escalate and stop lines (D3), the same pattern as `AutoTrainTermination.writeResults(log:)` (`App/AutoTrainTermination.swift:75-76`). On the CLI a slow stdout only delays the one task that both trains and evaluates, as every `emit` does today.
- `evaluateCheckpoint(stamp:layerHealth:digestTrainerStep:config:log:)` takes the same lock and no window. Its `config`: on the CLI, the run-start config every evaluation uses; in the GUI, resolved by the save from `TrainingParameters.shared` at the moment it takes the stamp (both save paths run on the main actor: `SessionController` is `@MainActor`, `App/SessionController.swift:26-28`) and carried to the detached pass with the stamp, so a checkpoint's **data** is judged under the settings in force when its state was exported (learning grace, warmup, enable). **Stop decisions are not**: R2 says a stop follows the current action, so in the GUI the stop decision is made on the main actor when the evaluation is delivered, from the current actions, for every delivered evaluation (R2, R3). A detached checkpoint pass evaluated under an older action therefore neither stops under an action the owner has since set to `log`, nor fails to stop under one the owner has since set to a stop action. On the CLI the actions never change during a run. It follows the same sequence without a window: apply an unannounced rewind (step 1); validate the stamp's generation under the steps lock (a stale one is ignored and logged, `[HEALTH] stale checkpoint observation ignored: …`); transition a copy; commit under the steps lock only if the generation is unchanged; then log.
- **Freshness across tiers.** Each rule remembers the newest trainer step whose observation it applied, from either tier. An observation older than that is not applied **to that rule** and is counted `stale=` on the `[HEALTH] check` line. Consequences: a slow, older checkpoint can never supply clears for `dead_channels` (or rules 1 and 8) after a newer live observation raised it; a checkpoint still updates rule 3, which only checkpoints feed; and a checkpoint at the same trainer step as the live evaluation that preceded it applies to both (the CLI order at a save step, which R0 requires of the cadence plan).
- **Self-measured cost.** `recordStep` and both evaluations time themselves with `ContinuousClock` (a clock read, not a random draw). `recordStep` adds its own time and the step's `totalMs` to counters in the step store (it holds only the steps lock); evaluations add their time to a counter in the evaluation state and, when writing a `[HEALTH] check` line, move the step-store totals over under evaluation → steps; the `[HEALTH] check` line reports `cost_ms=` since the previous check, next to `train_ms=` (the sum of `TrainStepTiming.totalMs` over the steps recorded in the same interval), so the monitor's whole cost — including the work outside `trainStep`'s `ms` — is visible as a ratio in every run (V-5). A final `[HEALTH] check … final=true` line is written when the run ends — on the CLI after the final save's checkpoint evaluation, so the last partial interval, including that last evaluation's cost, is never lost; in the GUI at Stop. A GUI checkpoint pass is detached and can finish after that flush; its evaluation is still logged and applied, and its cost appears on its own `[HEALTH] check … late=true` line, so the GUI guarantee is "nothing lost", not "all in the final line".

**Who calls it.** On the CLI, one task calls everything in sequence. In the GUI there are three callers — the trainer worker (`recordStep`), the stats task (`observationStamp`, then `evaluateLive`), and detached checkpoint passes (`evaluateCheckpoint`, `App/SessionController+Checkpoint.swift:622-643`) — and the evaluation lock serializes the last two.

**Publishing to the UI.** The evaluation's events are logged and recorded by the evaluating task. Every main-actor hop (after a live or a checkpoint evaluation) carries the monitor it used; the main actor ignores the hop unless `monitor === trainingHealthMonitor` (so a delayed hop from an earlier run changes nothing), and otherwise has the controller re-read that monitor's **current** active set (`activeAlarmsSnapshot()`, under the evaluation lock), so two hops arriving out of order cannot show an older state. `activeAlarmsSnapshot()` makes the main actor wait for the evaluation lock, which is held while an evaluation sorts its window and hands lines to the logger: normally well under a millisecond, at the 65,536-record cap a few milliseconds (estimates; `cost_ms` measures it). That is accepted; the main actor never waits on the steps lock's hot path.

**Stop decisions** belong to their monitor. In the GUI they are made on the main actor (R2) only for a hop whose monitor is the current `trainingHealthMonitor`; a hop from an earlier run's monitor is dropped and logged. The trainer worker polls the park flag of the monitor its own run created. On the CLI the evaluator's `stopRequest` sets the run-local `healthStop` directly.

No GPU, no random draws, no access to the trainer, buffer or optimizer (D6's value-FC1 read is made by the path through `ChessTrainer`; the monitor only decides when it is due and evaluates its result). Probe isolation holds by construction: the monitor's only inputs are values the paths already hold.

## D3. Rendering and recording

`Training/TrainingHealthLog.swift`, one renderer shared by every path (like `Training/LayerHealthLog.swift`). Fixed key=value formats:

```
[HEALTH] config enabled=true interval=1000 grace=1000 warmup=1000 momentum=0.85 path=replay actions=non_finite:log,dead_channels:log,…,gradient_spike:log value_fc1_zero_velocity=applies
[HEALTH] check trainerStep=2000 generation=0 evaluations=21 live=20 checkpoint=1 stale=0 truncated=0 cost_ms=3.1 train_ms=651400.0 liveReadFailed=0 lossMaxRatio=1.14 lossMedianRatio=1.02 gradMaxRatio=-- nodata=policy_offset_drift:3 active=dead_channels:critical,policy_offset_drift:warning dead_channels_sites=value.bn(6/16),blocks.0.bn1(1/128)
[ALARM] health raise rule=dead_channels severity=critical trainerStep=300 value=dead=5/1040 sites=value.bn(5/16) threshold=site>=0.2 action=log lr=0.3 mom=0.85
[ALARM] health raise rule=dead_channels severity=warning trainerStep=… value=dead=1/1040 sites=… threshold=dead>0 action=log lr=… mom=…
[ALARM] health escalate rule=dead_channels severity=critical trainerStep=1300 value=dead=5/1040 sites=value.bn(5/16) threshold=site>=0.2 action=log lr=0.3 mom=0.85
[ALARM] health worsen rule=dead_channels severity=critical trainerStep=1300 value=dead=6/1040 was=5 sites=value.bn(6/16)
[ALARM] health active rule=dead_channels severity=critical since=300 trainerStep=2000 value=dead=6/1040 sites=value.bn(6/16)
[ALARM] health clear rule=loss_spike severity=warning since=300 trainerStep=500 value=median/ref=1.04 max/ref=1.04 ref=7.5600
[ALARM] health stop rule=illegal_mass severity=critical trainerStep=350 action=stop_on_critical
[HEALTH] trainer clock rewound 5200 -> 4900; generation 1; windows reset
[HEALTH] trainer clock rewind detected by recordStep (not announced): 100 -> 50; generation 2
[HEALTH] stale checkpoint observation ignored: trainerStep=4800 generation=0 (current generation 1, newest applied 4900)
[HEALTH] live read failed at trainerStep=750; window boundary = last recorded step 750; layer-health rules no data
[HEALTH] rule value_fc1_zero_velocity does not apply: value FC1 activation is silu, not relu
[LAYER-HEALTH] value-fc1 trainerStep=3000 trained=2500 valueFC1ZeroVel=2/128 lowVel=9 readMs=1.23 summaryMs=0.50
```

- The tag stays `[ALARM]`. Every health line is `[ALARM] health <kind> rule=…`, so `grep '\[ALARM\] health'` selects exactly these and nothing else.
- On the CLI paths, raise, escalate and stop lines also go to stderr, as the existing replay `[ALARM]` lines do (`CLI/CorpusReplayRunner.swift:897-900,1878-1881`).
- `[HEALTH] check` is written on the first live evaluation at or after each multiple of `training_health_check_interval_steps` — with 50-step evaluations (OD-21), at the multiple itself on every path when the interval is a multiple of 50. The `active` reminders come on the same cadence. A run with no alarms therefore still writes one line per interval: positive evidence that the checks ran. The line also reports `gradMaxRatio=` (rule 9's largest `max_W(gradGlobalNorm) / ref` since the previous check) and, while `dead_channels` is active, `dead_channels_sites=` naming every affected site with its counts (OD-22).
- `CliTrainingRecorder` gains:
  - `appendAlarmEvent(_:)`;
  - top-level `alarms` (array of `TrainingHealthEvent`, always present, empty when none, like `arena_results`);
  - `alarm_config` (the resolved `TrainingHealthConfig`, D1's coding keys);
  - `TerminationReason.trainingHealthAlarm = "training_health_alarm"` (`CLI/CliTrainingRecorder.swift:247-283`).

## D4. Offline replay

- `Training/TrainingHealthLogReplay.swift` (pure) parses, from one or more session logs:
  - `[REPLAY] step=` / `[VS-UCI] step=` lines: `--` means not measured, never 0;
  - `[LAYER-HEALTH] live` lines;
  - `[LAYER-HEALTH] checkpoint` headlines and their per-site table rows;
  - `[LAYER-HEALTH] value-fc1` lines (D6), as rule-3-only observations;
- **Pre-scan.** Before evaluating, it reads every checkpoint table in the logs passed together and builds `site → channel count` **per run**: a run starts at each `[RUN]` line (lines before a log's first `[RUN]` form a run of their own), so a test-process log holding several runs (and architectures) gets one map per run, and a live line uses its own run's map. Two different counts for one site within one run are a malformed input: exit 2, naming both lines. A live line carries the worst site's dead count but not its channel count (`Training/LayerHealth.swift:985-990`), so this is the only offline source of that denominator (C's step-50 `value.bn` 12/16 needs the table at trainer step 513). A site with no known count makes the per-site arm "no data" for that evaluation; it is never guessed.
- **Sparse semantics — the offline replay feeds the real evaluator, and these are the only differences from the app:**
  - each stats row is one live evaluation whose window holds one record (that row);
  - the loss reference is the rows in the 1,000 trainer steps before the row, under rule 6's span definition (≥ 5 records spanning ≥ 200 steps);
  - "steps trained by this process" (rule 3's gate) is the stats row's own `step=` (segment steps), summed over the segments passed together; a checkpoint headline from a CLI path (`replay-…`, `vsuci-…`) uses its own `step=`. GUI `session-…` checkpoints carry no such count (their `step=` is the session's clock), so offline rule 3 has no data for them;
  - per-site dead counts between checkpoints: the live line's worst site only, which is the site with the most dead + mostly-off + always-on, not necessarily the most-dead (rule 2 note);
  - field coverage: `[REPLAY]` rows carry every rule input. `[VS-UCI]` rows carry no `pIllM` (`CLI/TrainVsUciRunner.swift:771-779`), so offline rule 4 has no data for train-vs-UCI logs; the in-app monitor has it;
  - logs without `[REPLAY]` / `[VS-UCI]` rows (GUI logs, whose `[STATS]` values are rolling means, not step samples) are replayed for the layer-health rules only (1, 2, 3, 8);
  - the output header lists these limitations for the logs given, so a reader knows which rules could have fired.
- `App/TrainingHealthReplayCLI.swift`: `DrewsChessMachine --replay-health-log <log> [<log> …] [--learning-grace-steps N] [--lr-warmup-steps N]`.
  - A pre-flight like the other no-GUI modes: a `handleReplayHealthLogIfPresent(rawArgs:)` in `App/DrewsChessMachineApp.swift`, called with the others (`:185`; pattern at `:1903-1938`), which hands over to `TrainingHealthReplayCLI.runAndExit`. It runs synchronously on the launching thread before any GUI or Swift task exists; it never parses inside a `Task`.
  - Prints the events in the live format, then a summary table.
  - **Field format is a contract.** `TrainingHealthLogReplay` parses `[REPLAY]` / `[VS-UCI]` rows by `key=value` field, needs `trainerStep=` and reads `--` as "not measured". The cadence redesign changes which steps are logged, which is fine, but must keep that field format; this is stated as a requirement on the cadence plan (Part P). Row density will change with a time-based cadence, so offline rule 6 results on new logs are not comparable row-for-row with the old 50-step logs (its reference rule, ≥ 5 records spanning ≥ 200 steps, still applies).
  - **Legacy rows.** A step row from an older build may lack fields newer builds print (`pLogitMean`, `vLogitMean`, `trainerStep`, …). A missing rule-input field is treated exactly like `--`: not measured, so the rules that need it have no data. The output header lists, per log, every field absent from its rows and the rules left without data because of it (e.g. `pLogitMean absent in 5,371 rows: policy_offset_drift no data`). Nothing is inferred or filled in.
  - `trainerStep=` is the exception: without it there is no trainer clock, and the run refuses that log unless `--segment-step-as-trainer-step` is given. That flag uses the row's `step=` as the trainer step. Explicit only, never a fallback; refused unless every row in the log is strictly increasing; stated in the output header. Monotonic segment steps do not establish the clock's starting point: the header says the trainer steps may be offset by the start model's clock, which leaves span-based conditions (sustain spans, rule 6's reference) exact and makes the absolute ones (the learning gate of rule 4) unreliable for that log.
  - Exits 0. Exits 2 on an unreadable log; on a malformed line (named with its file and line number); on a row lacking `step=` or `loss=`, or lacking `trainerStep=` without the flag (reported as `unsupported log format: <file> (build <N> from its [APP] line): step rows carry no <field>`, not as malformed); or on a log with neither stats rows nor `[LAYER-HEALTH]` lines.
  - Evaluator state: within one log, each `[RUN]` line after the log's first starts a fresh evaluator, as the app starts a fresh monitor per run (a test-process log holding several runs is several runs). The first `[RUN]` of the first log starts the evaluator; the first `[RUN]` of each later log passed together does **not** reset it, so logs passed together are one continuing evaluator across the log boundary — the "C, one process" comparison in Evidence — and the output header says so; that is an offline convenience, since in the app each segment gets its own monitor (the "C segment 1 as its own process" row).
- A Python mirror of the rules is deliberately not written: it would be a second source of truth for the thresholds.

## D5. `LayerHealthLog.live`

- `LayerHealthLog.liveLines(trainer:)` (`Training/LayerHealthLog.swift:28-36`) becomes `live(trainer:) async -> LiveOutcome { lines, summary?, trainerStep? }`, mirroring `CheckpointOutcome` (`:46-49`).
- Its four call sites — `App/SessionController+Training.swift:2120,2177`, `CLI/CorpusReplayRunner.swift:1927`, `CLI/TrainVsUciRunner.swift:783` — are converted in **P1** to `live(trainer:).lines` with no behavior change, so P1 builds on its own with `liveLines` removed. P2 (CLI) and P3 (GUI) then hand `summary` to the monitor.
- `liveLine(summary:trainerStep:)` stays; `LayerHealthTests.swift:616` uses it.
- The checkpoint passes already return the summary (`CheckpointOutcome.summary`). Their call sites hand it to the monitor: `CLI/CorpusReplayRunner.swift:1614-1623`, `CLI/TrainVsUciRunner.swift:562-572`, and `App/SessionController+Checkpoint.swift:622-643` (which gains `monitor` and `stamp` arguments and a main-actor delivery callback; callers `:598` and `App/SessionController+Arena.swift:821`).

## D6. The value-FC1 velocity check (OD-15)

**One schedule for every path: a deadline, not a grid.** (Only while rule 3 applies — a ReLU value FC1 layer, rule 3 note; otherwise no read is ever due.) `TrainingHealthMonitor.valueFC1ReadDue(trainerStep:) -> Bool` (pure, under the evaluation lock) is true when `trainerStep − anchor ≥ TrainingHealthThresholds.valueFC1CheckIntervalSteps` (1,000; a declared constant like the thresholds, OD-14). The **anchor** is the trainer step of the newest rule-3 observation in this generation (from any source, by its stamp's trainer step), or, when there is none yet, the trainer clock at which this process's monitor started recording (the first record's step − 1) or the restored clock after a rewind. So two consecutive rule-3 observations are never more than 1,000 trainer steps apart, whatever falls in between, and a save's pass simply moves the deadline. Every path asks after the step's work and after any checkpoint pass that step ran:
- **Corpus replay** asks after the autosave block (`CLI/CorpusReplayRunner.swift:1979-1988`, pre-cadence-fix). Its rolling saves are 1,000 trainer steps apart and the first falls at most 1,000 steps after the process's starting clock — today every 1,000th segment step (`:1110`), after the cadence plan's redesign at overall trainer-step multiples of 1,000 — and its checkpoint pass feeds rule 3 before the question is asked. The deadline therefore lands on a save step that has already reset it, fresh or resumed, on either save grid (resumed at 513, start anchor 513: today's segment grid saves at trainer steps 1,513, 2,513, …; the redesigned overall grid saves at 1,000, 2,000, …; in both the first save comes at or before the 1,513 deadline and each later one exactly 1,000 after the previous), and no dedicated read happens — **as long as each save and its checkpoint pass succeed**. A first save failure is non-fatal (`:1628-1633`, the run continues) and a failed health pass returns no summary (`Training/LayerHealthLog.swift:69-70`); either leaves the anchor where it was, and the dedicated read at that step keeps the interval. That is the schedule working, not a double read: it reads only when no observation exists.
- **Train-vs-UCI** asks after its enumerated-checkpoint / periodic-save blocks (`CLI/TrainVsUciRunner.swift:847-853`). With `--enumerate-checkpoints` its pass at every enumerated checkpoint covers the interval — those checkpoints are 1,000 trainer steps apart and the first falls within 1,000 of the start whether they sit on segment multiples (today) or overall trainer-step multiples (the redesign under discussion), so the corpus-replay reasoning above applies unchanged; without it (the default; `writeEnumeratedCheckpoint` returns at once when no writer, `:646-647`) the read is dedicated.
- **GUI** asks in the trainer worker right after `recordStep` (T8). Saves are hours apart, so the read is dedicated almost every time: 1,000 trainer steps after the session's start, and every 1,000 after that or after the latest save.

**The read.** `ChessTrainer.readTrainableVelocity(named: "value.fc1.weight") async throws -> (velocity: [Float], completedTrainSteps: Int)`, new. It follows the existing read pattern exactly — `enqueue` onto the trainer's serial `executionQueue`, one `network.graph.run` whose only target is that tensor's velocity variable, fed the dummy inference input, read back as fp32 — as `internalReadLayerHealthLiveState` (`Training/ChessTrainer.swift:5784-5830`) and `readVelocityValues` (`:5562-5587`) do. Because it runs on `executionQueue`, it falls between SGD steps and never overlaps one. Its result goes through the existing pure `LayerHealth.hiddenUnitVelocityHealth(layer: LayerHealth.valueFC1Layer(for: arch), velocity:)` (`Training/LayerHealth.swift:245-252,614`), on the calling task after the read returns (a single pass over the tensor).

**Probe isolation.** The graph run targets one variable and no operation, so nothing is assigned, no dropout op is encoded (no RNG advance), and the optimizer, weights, BN statistics and replay buffer are untouched — the same guarantee the live read has today. A test pins it (X1).

**Observation and log.** The result is a checkpoint-tier observation whose digest carries only `valueFC1` (every other field absent, so other rules have no data from it), stamped with `observationStamp()` taken before the read, and evaluated through `evaluateCheckpoint`. It is logged as one line, which the offline replay also parses (D4):
`[LAYER-HEALTH] value-fc1 trainerStep=<s> trained=<stamp.stepsTrainedByThisProcess> valueFC1ZeroVel=<zero>/<units> lowVel=<n> readMs=<ms> summaryMs=<ms>`. `trained=` is rule 3's gate (R0): a GUI log has no step rows, so without it the offline replay could never judge rule 3 there; the replay refuses a value-fc1 line without it.

**Cost.**
- Size: `64 × valueHeadConvChannels × valueHeadHiddenUnits` fp32 values. 131,072 values = 512 KB for the A/B/C architecture (16 × 128); 16 KB for a preset with 1 × 64.
- Time (estimated, not yet measured): one single-target `graph.run` plus a copy of at most 512 KB from unified memory, ≈ 1–3 ms, and one pass over the values on the CPU, well under 1 ms. Once per 1,000 steps at the Ejp0 GUI pace (≈ 0.65 s/step, ≈ 650 s per 1,000 steps), that is about 0.0005% of training time; the trainer queue is held only for the read itself.
- `readMs` and `summaryMs` are logged on every read and counted in `cost_ms`; V-5 checks them.

---

# Part K — Parameters (12), with the full CLAUDE.md checklist

All of them:
- category `"Health"` (new);
- `absentValue: .currentSetting` (operational knobs that do not change training math);
- `liveTunable: true`.

The GUI re-reads them at every evaluation (D2 takes the config per call). The CLI paths read them once from their run-start snapshot, as every other CLI parameter is read.

| id | Type | Default | Range | Meaning |
|---|---|---:|---|---|
| `training_health_alarms_enabled` | Bool | `true` | — | off: no evaluation, one `[HEALTH] alarms disabled` line at run start |
| `training_health_check_interval_steps` | Int | 1000 | 50…100000 | `[HEALTH] check` line and `active` reminder cadence (trainer steps) |
| `training_health_learning_grace_steps` | Int | 1000 | 0…100000 | added to `lr_warmup_steps` for the not-learned form of rule 4 |
| `training_health_action_<rule>` × 9 | Int | 0 (`log`) | 0…2 | `TrainingHealthAction` raw value, one per `TrainingHealthRule` |

**Checklist walk** (CLAUDE.md "Adding / removing / renaming a parameter"):
1. **Declare.** 12 `@TrainingParameter` declarations in `Training/TrainingParameters.swift`, added to `allKeys` (`:2479`). The action keys are Int-coded like `RandomSeedModeParameter` (`:1331-1341`), with `TrainingHealthAction(persistedRawValue:)`. A test pins the range against `TrainingHealthAction.allCases`, as `test_arenaPromotionCriterion_rangeMatchesEnumCases` does.
2. **Singleton.** Stored properties, `collectValues` / `applyOne`, and snapshot accessors (pattern at `:1490-1491`, `:1634-1635`, `:1743-1744`, `:1839-1840`).
3. **`parameters.json`.** Confirm all 12 appear in `--show-default-parameters`, and that `--create-parameters-file` → edit → reload round-trips. Write to a scratch folder first, never `--force` into the repo.
4. **Session save/load.** Optional fields on `SessionCheckpointState` (`Persistence/SessionCheckpointFile.swift`), passed through `buildCurrentSessionState`, and one `restore(…)` line each in `SessionParameterResume.applyGuiSession` (`App/SessionParameterResume.swift:129`, next to the `BatchStatsInterval` / `KLProbeInterval` lines at `:144-145`).
5. **`results.json`.** `alarm_config` (D3).
6. **Runtime log.** The `[HEALTH] config …` line at every run start (all paths, beside the `[RUN]` line). Live GUI edits are logged by the popover model as `[PARAM]` lines (pattern at `App/UpperContentView/TrainingSettingsPopoverModel.swift:1531-1535`).
7. **UI (P3).** A new "Health" tab in `TrainingSettingsPopover` (the `Tab` enum at `App/UpperContentView/TrainingSettingsPopover.swift:36-42`), as a new internal `View` in its own file, `App/UpperContentView/TrainingHealthTab.swift`, with bindings and validation in `TrainingSettingsPopoverModel.swift`. Layout: one aligned row per rule (name, one-line meaning, an action picker of Log / Stop on critical / Stop on any, with Stop on critical disabled and labelled "no critical level" for rules 6–8). Then enable, check interval and grace, with monospaced, padded digits.
8. **Live tunability.** The GUI stats task resolves `TrainingHealthConfig` from `TrainingParameters.shared` in the same `MainActor.run` hop that already reads the live parameters for each `[STATS]` emit (`App/SessionController+Training.swift:1458-1468`).
9. **Renames.** Not applicable.

**Consequences.**
- The lineage record's `parameters` snapshot is built from every key (`ReplayParams.init`, `CLI/CorpusReplayRunner.swift:48-57`, `LineageRecord.Parameters(values: parameters.rawValueMap())`). So the 12 keys enter every model file written after this lands, and `params_sha` on every `[RUN]` line changes from that build on.
- An exact resume from an older file finds them absent and applies `.currentSetting`: no `params` gap.
- This is the intended single source (HPARAM_RECORDING_PLAN.md S3); nothing else is needed for recording.
- **Shared edit sites with `HPARAM_RECORDING_PLAN.md`** (coordinate; line numbers shift with whichever lands first):
  - the session save right after `trainingGate.pauseAndWait` (`App/SessionController+Checkpoint.swift:432-447`): this plan's export stamp and config; HPARAM's configuration cut, which also carries the merged alarm summary;
  - the promotion rewind (`App/SessionController+Arena.swift:455-490`): this plan's `noteTrainerClockRewind`; HPARAM's journal re-stamp;
  - `startRealTraining`: this plan's new monitor per start; HPARAM's run-start capture and segment start;
  - the `TrainingParameters` stored-property `didSet`s: this plan adds 11; HPARAM passes `oldValue` to `commitAssignment` — whichever plan lands second wires the other's;
  - the CLI step blocks: the cadence plan replaces the tick and the save cadence; this plan's P1 converts `liveLines` and P2 adds record/evaluate; HPARAM touches other runner regions.

---

# Part T — Touch points

### T1. New pure core (P1)
- `Training/TrainingHealth.swift` (D1)
- `Training/TrainingHealthMonitor.swift` (D2)
- `Training/TrainingHealthLog.swift` (D3)
- `Training/TrainingHealthLogReplay.swift` (D4)
- `Training/BatchNormPassThrough.swift` (OD-18): the activation-aware parked classification, single source of the model that `experiments/20261005-lr-schedule-ab/bn_liveness.py` mirrors; `LayerHealth` gains per-site `parked` / `parkedMostlyOff` counts and min / median pass-through, and the compact line gains `parkedSites= parkedCh= parked= parkedOff= parkedBy=` after `nonFinite=` (the relu / leaky_relu fields keep their meaning)

### T2. `Training/LayerHealthLog.swift` (P1)
`live(trainer:)` (D5).

### T3. `CLI/CorpusReplayRunner.swift` (P2)
Line numbers in T3 and T4 are **pre-cadence-fix** (`main` at `62dc8559`). `STATS_LINE_RESUME_CADENCE_FIX_PLAN.md` lands first and edits the same stats-tick blocks (`CorpusReplayRunner.swift:1839-1840,1904-1928`, `TrainVsUciRunner.swift:744,764-786`), so P2 re-cites them against the fixed code before implementing. The wiring is anchored to code landmarks (the step-line tick, the live read, the checkpoint pass), not to the numbers.
- Construct the monitor and resolve the config from `params` before the loop.
- Log `[HEALTH] config` right after the `[RUN]` line.
- `monitor.recordStep(timing, trainerStep: trainer.completedTrainSteps)` after `lineageTracker.recordTrainingStep` (`:1902`).
- At the stats tick (`:1904-1928`): `observationStamp()` before the live read and `evaluateLive` after it (`:1927`), passing `liveLR` (`:1910`), `liveMomentum` (`:1911`) and the runner's `emit` as the log sink. Append the events to the recorder.
- `evaluateCheckpoint` after the checkpoint pass (`:1614-1623`), inside `saveTrainerModel`; the monitor and the `healthStop` value are captured by that closure like `recorder` is.
- After the autosave block (`:1979-1988`): `if monitor.valueFC1ReadDue(…)`, the dedicated read (D6). With the rolling save every 1,000 segment steps it is due on this path only when a save or its health pass produced no observation (D6); the call stays so all three paths share one schedule.
- Stop: `healthStop` checked at the loop top (`:1849`); final save reason `health-stop` (`:2008`); termination reason (`:2032`); exit status (`:911-912,969`, OD-4).

### T4. `CLI/TrainVsUciRunner.swift` and `CLI/TrainVsUciSession.swift` (P2)
- Same wiring at `:758` (record), `:764-786` (the step-line tick block; evaluate after the live read at `:783`), `:562-572` (checkpoint), and `:746` (stop check).
- After the enumerated-checkpoint and periodic-save blocks (`:847-853`): `if monitor.valueFC1ReadDue(…)`, the dedicated read (D6).
- `SaveKind.healthStop` (`CLI/TrainVsUciSession.swift:33-42`); final save at `:871-878` (the enumerated copy at `:873-875` takes `finalKind.rawValue`, so its layer-health context reads `vsuci-health-stop`); termination reason at `:887-889`; exit status at `:112,157`.

### T5. `CLI/CliTrainingRecorder.swift` (P2)
`alarms`, `alarm_config`, `TerminationReason.trainingHealthAlarm`, `appendAlarmEvent`. Encoding added to `encodedJSONData` (`:136-162`) and `Snapshot` (`:345-400`).

### T6. Parameters (P2)
Part K.

### T7. `App/TrainingHealthReplayCLI.swift` + argument dispatch + `App/CommandLineHelp.swift` (P2)
- D4; the pre-flight `handleReplayHealthLogIfPresent` in `App/DrewsChessMachineApp.swift`.
- `--help`: the new flag; the alarm exit status (OD-4, 35) next to the existing "0 / 2 / 33" text of both CLI paths; and the `-vsuci-health-stop` tag in the train-vs-UCI session-folder list (`App/CommandLineHelp.swift:258`).

### T8. GUI wiring (P3)
- `App/SessionController.swift`: `trainingHealthMonitor`, a new monitor at every `startRealTraining` (including a continue after Stop). `trainingSuspension` replaces `trainingSuspendedByDivergence` (`:280-289`), and every reader is updated (the `Bool` is removed, so a missed reader fails to compile):
  - `App/SessionController+Training.swift:524,1370,2394,2490,2491,2521`;
  - `App/SessionController+Heartbeat.swift:376,645`.
- Trainer worker (`App/SessionController+Training.swift:1187-1273`): `recordStep(timing, trainerStep: trainer.completedTrainSteps)` after `box.recordStep(timing)` (`:1271`); then `if monitor.valueFC1ReadDue(…)`, the dedicated read (D6), awaited inline so the next SGD step waits for it, then its main-actor delivery like any checkpoint evaluation; the park-flag check at the loop top (`:1199`); the parked loop (R3).
- Stats task (`:1404-2180`): `observationStamp()` before each live read, `evaluateLive` after it (`:2119-2124`, `:2177-2179`), with the config resolved in the existing `MainActor.run` hop (Part K item 8). Then hop to the main actor so `TrainingAlarmController` re-reads the monitor's active set and the stop decision is made from the current actions (R2, R3).
- Checkpoint passes: `logSavedTrainerLayerHealth` gains the monitor, the stamp (D2) read with the save's trainer export (for the promotion save, after the rewind, in the same pause), and a main-actor delivery callback (`App/SessionController+Checkpoint.swift:622-643`; callers `:598`, `App/SessionController+Arena.swift:821-827`). It calls `evaluateCheckpoint`, then delivers to the main actor like the stats task: identity check, active-set refresh, stop decision from the current actions (D2, R2).
- Promotion: `monitor.noteTrainerClockRewind(to: trainerSnapshotCompletedSteps)` next to `trainingBox?.resetRollingWindows()` (`App/SessionController+Arena.swift:490`), under both pauses. `trainingAlarm?.clear()` (`:492`) clears the existing banner only; health alarms stay (they clear by recovery).
- Promote Trainee Now (OD-5): `promoteTrainerNow` (`App/SessionController+ManualPromote.swift:35-52`) gains a guard next to its `isArenaRunning` refusal: while `trainingSuspension` is set (either case) it calls `onRefuseMenuAction` with the suspension's reason and returns. The menu item's `.disabled(...)` (`App/DrewsChessMachineApp.swift:683-684`) adds the suspension, through a `trainingSuspended` flag mirrored onto `AppCommandHub` the way `realTraining` is (`App/AppCommandHub.swift:41`). Test (new file, P3): `PromoteTrainerNowSuspensionTests` — with the save harness the existing promote tests use, a `.divergence` and a `.healthAlarm` suspension each make `promoteTrainerNow` refuse with that reason, and the champion's weights and ID are unchanged.
- `--train`: `AutoTrainTermination` with `.trainingHealthAlarm` (R3), through the shared claim that `collapseTermination` uses (`App/SessionController+Training.swift:2271`). Every evaluation's events are also appended to the run's `cliRecorder`, so a GUI `--train` `results.json` carries `alarms` and `alarm_config` like the CLI paths.

### T9. GUI surfaces (P3)
- `App/UpperContentView/TrainingAlarmController.swift`:
  - adds `private(set) var healthAlarms: [TrainingHealthActiveAlarm]`, replaced wholesale from the monitor's current active set on each main-actor hop (the monitor is the source; the controller mirrors it for the view);
  - adds `refreshHealth(from monitor:)`.
  - **Sound, one beep loop for both sources.** Today the loop starts only when the banner alarm is set (`:443`) and each burst stops when it is not (`:459`), so a health-only critical alarm would never beep. Both guards become one private predicate, `shouldSound` = (`active != nil` or a critical health alarm is present) and not `silenced`; a private `reconcileSound()` starts or cancels the loop and is called after `raise`, `clear`, `silence`, `dismiss` and `refreshHealth`. So `clear()` / `dismiss()` on the banner (e.g. at promotion, `App/SessionController+Arena.swift:492`) still reset `silenced`, and the loop restarts if a critical health alarm remains. Warnings alone never beep (OD-8).
  - `silence()` is reachable from the health list too (below), since the banner's Silence button exists only while the banner is shown (`App/UpperContentView/UpperContentView.swift:1141-1146`).
  - The existing detectors, their titles, `active`, `Streaks`, and every public method's observable effect on them stay unchanged. `TrainingAlarmControllerTests` exercises `evaluate`, `dismiss`, `clear` and the streaks (`:76-95`), not the sound loop, so it stays valid unmodified; new tests cover `shouldSound` (X1).
- New `App/UpperContentView/TrainingHealthAlarmList.swift` (one `View`):
  - mounted in `UpperContentView.body` directly under the existing banner (`App/UpperContentView/UpperContentView.swift:1141-1148`);
  - always in the hierarchy; when the list is empty it is hidden with `.opacity(0)` and a zero frame in both dimensions, never with an `if` (the owner's view-stability rule);
  - one aligned row per active alarm: an SF Symbol for severity (`exclamationmark.triangle.fill` warning, `xmark.octagon.fill` critical), the rule's display name, the measured value in monospaced, padded digits, "since trainer step N", and the action ("log" / "stops run");
  - semantic colors that work in light and dark mode (`.orange` / `.red` symbols on `.primary` text over `.background.secondary`), not the banner's fixed yellow;
  - each row has one combined `accessibilityLabel` ("Critical: dead channels, 339 of 1,040, since trainer step 514");
  - when the run is suspended by a health alarm, a header row says so and names the rule;
  - a Silence button (the same `trainingAlarm.silence()`), shown while any critical health alarm is present and the sound is not silenced, and kept in the hierarchy with opacity 0 and a zero frame otherwise.
- Existing banner: unchanged. (It still uses `if let alarm` at `:1141`; changing that is outside this plan.)
- Health tab: Part K item 7.

### T10. Docs (P4)
- New `documentation/training-health-alarms.md`: rules, evidence, formats, actions, offline replay.
- `--help`; CHANGELOG.
- The other lists of train-vs-UCI save tags gain `vsuci-health-stop`: the docstrings of `scripts/sessions_summary.py:15-17` and `documentation/dashboards/vsuci.py:13-15`.
- CLAUDE.md tag list: `[HEALTH]` and the `[ALARM] health` form (OD-11). CLAUDE.md and `documentation/UCI.md` ("Output: session folders") also list the train-vs-UCI save tags; both gain `vsuci-health-stop` (OD-11).

### T11. Verified untouched
- `ChessTrainer`'s graph, step and per-step readbacks. Its only change is the new read-only method `readTrainableVelocity(named:)` (D6), which adds no graph operation.
- `LayerHealth` (the analysis and thresholds; only `LayerHealthLog` gains `live`).
- The non-finite-loss halt.
- The GUI legal-mass probe, the divergence / value-saturation / pD heartbeat detectors and the entropy `[ALARM]` line (`App/SessionController+Training.swift:2063-2069`), until OD-9.
- Model files, session format apart from the parameter fields, lineage schema, and the replay buffer.

---

# Part X — Tests

### X1. New test files (`DrewsChessMachineTests/`)

- **`TrainingHealthEvaluatorTests.swift`** (synthetic observations), at least:
  - Per rule: `test<Rule>RaisesAtThreshold`, `…DoesNotRaiseJustBelow`, `…ClearsOnlyAfterClearSustain`; for rules 2 and 3 (the only rules with both levels) `…EscalatesWarningToCritical` and `…NeverDeescalates`.
  - `testNoDataHoldsStateAndCountsIt`: diagnostic rules with windows that hold no diagnostic step neither clear nor progress.
  - `testSustainNeedsBothCountAndSpan`: 2 evaluations 10 steps apart do not satisfy span 50.
  - `testLearningGateUsesTrainerClockAndWarmup`: rule 4's not-learned form is silent below `warmup + grace` and active above. A resumed run whose trainer clock is already past the gate evaluates it at its first window.
  - `testRegressionFormIgnoresTheGate`: rule 4.
  - `testDeadChannelsWarnsOnAnyDeadChannel`, `testDeadChannelsCriticalOverallFivePercent`, `testDeadChannelsCriticalSiteTwentyPercent`, `testDeadChannelsCriticalNeverClearsWhileDeadRemains`; in `TrainingHealthLogReplayTests`, `testLegacyRowsReportAbsentFieldsAsNoData`. (The new-damage form's tests are listed under **Deferred (OD-16 revisit)** and are not written.)
  - `TrainingHealthStopPolicy` tests: `testFirstQualifyingFollowsRuleOrder`, `testActionChangedToLogMeansNoStop`, `testActionChangedToStopOnAnyStopsAnAlreadyActiveWarning` (both directions of a change made while a detached checkpoint pass was running: the pass's evaluation used the old actions, the decision uses the new).
  - `testStopFollowsCurrentActionForAnAlreadyActiveAlarm`: an active critical rule whose action changes from `log` to `stop_on_critical` requests a stop at the next evaluation (R2).
  - `testReminderEveryCheckInterval` and `testWorsenRateLimitedToOncePerInterval`.
  - `testActionLogNeverRequestsStop`; `testStopOnCriticalIgnoresWarnings`; `testStopOnCriticalNeverFiresForRulesWithoutCritical`; `testStopRequestedOnce`.
  - `testValueFC1RuleNeedsStepsTrainedByThisProcess`: a 16/16 checkpoint with the trainer clock at 274 and 0 steps trained (the survey case) is no data; the same digest after 200 trained steps raises critical.
  - `testDisabledConfigEmitsNothing`.
  - `testEventOrderIsDeterministic`.
- **`TrainingHealthMonitorTests.swift`**:
  - `testWindowMediansAndMaxima`; `testDiagnosticFieldsOnlyFromDiagnosticSteps`;
  - `testRingCapacityDropsOldest`; `testLossReferenceNeedsMinimumHistory`;
  - `testTrainerClockRewindResetsWindowsKeepsActiveAlarms` (also: pending sustain and regression extrema reset, generation incremented, steps trained reduced by exactly the rewound span);
  - `testOlderCheckpointArrivingLateIsIgnored`: a damaged checkpoint at 2,000 raises rule 3; a healthy one at 1,000 applied afterwards changes nothing and is counted `stale`;
  - `testOlderCheckpointsCannotClearALiveRaise`: a live observation at 3,000 raises `dead_channels`; two healthy checkpoints at 2,000 and 2,500 arriving later do not clear it;
  - `testCheckpointFromPreviousGenerationIsIgnored` and `testLiveObservationFromPreviousGenerationIsIgnored` (stamp taken, rewind announced, then `evaluateLive`);
  - `testWindowStopsAtTheDigestsTrainerStep`: records appended after the live read stay pending for the next window;
  - `testUnannouncedRewindCountsExactly` (last 100, record 51 → 51 trained) and `testAnnouncedRewindCountsExactly`;
  - `testCheckpointArrivingFirstAfterAnUnannouncedRewindIsRejected` (pre-rewind stamp, rewind detected by `recordStep`, checkpoint evaluated before any live evaluation: evaluation-side reset applied, observation rejected, nothing raised);
  - `testConcurrentRewindAndEvaluation` (an announced rewind racing an evaluation; no deadlock, state consistent);
  - `testUnannouncedRewindDuringLiveEvaluationDiscardsIt` and `testUnannouncedRewindDuringCheckpointEvaluationDiscardsIt`: a test hook pauses the evaluation between validation and commit, `recordStep` rewinds, the evaluation resumes; it must not commit, log an event or request a stop;
  - `testFinalCheckLineFlushesThePartialInterval`;
  - `testValueFC1ReadDueAfterOneThousandStepsFromStart`, `testValueFC1ObservationsNeverMoreThanOneThousandApart` (a checkpoint at 1,001 moves the deadline to 2,001; nothing at 2,000 and nothing waits until 3,000), `testValueFC1ReadNeverDueOnCorpusReplaySaveCadence` (successful saves on both grids — every 1,000 segment steps, and at overall trainer-step multiples of 1,000 — each fresh and resumed at 513, so the test holds whichever cadence the redesigned plan lands with), `testValueFC1ReadDueWhenASaveProducedNoObservation`, `testValueFC1ReadDueAnchorsOnTheRestoredClockAfterARewind`;
  - `testRule3UsesTheStampedTrainedCount`: a checkpoint stamped at 199 trained steps and evaluated after 200 is no data;
  - `testStaleHopFromAnEarlierMonitorIsIgnored` (P3, controller side);
  - `testCheckpointEvaluationNeverConsumesTheWindowOrDetectsRewind`;
  - `testLossReferenceSurvivesALongWindow`: a 3,000-record window still finds its reference in the 1,000 steps before it;
  - `testPendingWindowCapCountsTruncation`;
  - `testConcurrentRecordAndEvaluate`: one task records 10,000 steps while two others run live and checkpoint evaluations; every recorded step lands in exactly one window.
- **`TrainingHealthIncidentReplayTests.swift`**: the evidence as executable tests, through `TrainingHealthLogReplay` and the real evaluator. Added 2026-10-06: `testArmCRaisesGradientSpikeAt300`, `testArmCRaisesRunningVarianceRunawayAt200`, `testBSiluIllegalMassRegressionRaisesAt20700`, `testBSiluGradientSpikeRaisesAt20600` (B-silu's excerpt: every line through trainer step 22,000 except `[BATCH-STATS]`).
  - The inputs are log excerpts shipped as **bundled test resources**: `DrewsChessMachineTests/Resources/TrainingHealthIncidents/<run>.log`, one plain-text file per run, in the test target's Copy Bundle Resources, read through `Bundle(for:)`. Not Swift source: the excerpts are hundreds of KB of log text (the checkpoint tables included), which belongs in resource files rather than string literals in an already slow-to-build test target. (`[BATCH-STATS]` lines, 61,592–79,930 B each in B, C and R7, are not included at all; next bullet.) They are generated once from the logs named in **Evidence**.
  - Every selected line is verbatim. `[BATCH-STATS]` lines (61–80 KB each) are left out: since the one-sided value-head rule was dropped (OD-7) the replay reads nothing from them. Each file's header (lines starting with `#`) states each source log's SHA-256 and line count at extraction (A and B were still growing; the cutoff is the Evidence cutoff) and the line-selection rule; `TrainingHealthLogReplay` skips `#` header lines only in a file whose first line is the excerpt header, never in a session log.
  - Apart from the omitted `[BATCH-STATS]` lines, selection is contiguous, never thinned, so windows, loss references and sustain spans are exactly what the full log gives: every C stats row, live line and checkpoint block (both segments); B's through trainer step 1,500 plus every B checkpoint block to the cutoff; A's, R7's and R8's through trainer step 5,000 plus every checkpoint block. Full-length behavior is V-1's job.
  - `testArmCRaisesDeadChannelsCriticalAt50` (through the checkpoint-table pre-scan), `…IllegalMassCriticalAt350`, `…GradientCollapseAt400`, `…ValueFC1CriticalAt1513`, `…LossSpikeAt300`, `…PolicyOffsetDriftRaisedAt200ClearedAt400`, `…NoRaiseEscalateOrClearAt250` (no raise, escalation or clear at step 250; C's dead_channels worsen events, the first at 100, are unaffected), `…NeverRaisesNonFinite`.
  - `testArmCSegmentOneAloneRaisesDeadChannelsAtFirstEvaluation`.
  - `testArmBRaisesValueBNDeadChannelCriticalAt300`, `…PolicyOffsetDriftWarningAt900`, `…ValueFC1WarningAt1000`, `…OnlyDeadChannelsIsCritical`.
  - `testArmARaisesNothing`, `testR7RaisesNothing`, `testR8RaisesNothing`.
  - The expected steps are the prototype's (Evidence table). If the Swift log replay differs in a step, the difference is reported to the owner and the expected value is not quietly changed.
- **`ValueFC1VelocityReadTests.swift`** (needs Metal; P2): `readTrainableVelocity(named: "value.fc1.weight")` equals the matching slice of `exportVelocitySnapshot()` at the same step; and **probe isolation** — two trainers built from the same seed and weights, trained the same k steps with the same streams, one reading the velocity after every step, end with bit-identical weights, velocity and dropout Philox state.
- **`TrainingHealthLogTests.swift`**: exact strings for every line kind (the grep contract), including `--`, signed values and site rendering.
- **`BatchNormPassThroughTests.swift`** (OD-18, P1): pass-through, excess pass-through and band against 120 reference values `bn_liveness.py` produced (`Resources/TrainingHealthIncidents/BatchNormPassThroughReference.json`; relu, leaky_relu, silu, gelu, γ = 0, near both lines, \|γ\| to 1,000); the derivative roots and the Gauss–Legendre rule; parked = dead and parked-mostly-off = mostly-off exactly for relu / leaky_relu on 20,004 generated channels plus the band edges; B-silu's real γ/β at 20,000 / 21,000 / 22,000 (`TrainingHealthBSiluBatchNorm.json`, source files and `content_sha256` recorded) against `bn_liveness.py`'s per-site parked / mostly-off / min and median pass-through; the compact line's new fields beside unchanged relu fields.
- **`TrainingHealthLogReplayTests.swift`**:
  - parsing of `[REPLAY]`, `[VS-UCI]`, live and checkpoint lines;
  - `--` read as not measured;
  - a segment-1 log whose diagnostic fields are all `--`;
  - a checkpoint table with `n/a` columns (`stem.bn` with no activation);
  - malformed lines rejected with an error naming the line, never skipped silently.
- **`TrainingHealthParameterTests.swift`**:
  - defaults, ranges and `absentValue` for the 12 keys;
  - the action range equals `TrainingHealthAction.allCases`;
  - every `TrainingHealthRule` has exactly one action key;
  - JSON round trip through the parameters-file encoder and decoder;
  - session restore through `SessionParameterResume` (take defaults from `makeTemporaryDefaultsSuite()`, never a named suite).
- **`TrainingAlarmControllerHealthTests.swift`** (P3): `shouldSound` for banner-only, health-critical-only, health-warning-only and silenced cases; `clear()` at promotion keeps `healthAlarms` and resumes sound for a remaining critical; `refreshHealth` mirrors the monitor's set.
- **`CliTrainingRecorderAlarmTests.swift`**:
  - `alarms` present and empty by default;
  - event encoding (`trainer_step`, `rule`, `kind`, `severity`, `value`, `threshold`, `action`);
  - `alarm_config` encoding (snake_case keys, actions by name);
  - `TrainingAlarm.Severity` encodes as `"warning"` / `"critical"`;
  - `TerminationReason.trainingHealthAlarm.rawValue == "training_health_alarm"`.

### X2. Existing tests that must change (OD-6)
- `TrainingParametersTests.test_registry_size` (`DrewsChessMachineTests/TrainingParametersTests.swift:19-25`) pins `allKeys.count` at 85. It must become **97** (85 + the 12 new keys: enable, check interval, learning grace, and nine action keys). The test's message ("requires intentionally updating this count") says the edit is expected maintenance; it is **not** approval. **Approved by the owner (OD-6, 2026-10-05)** for the edit the plan needs; the approval was given when the plan said 98; dropping the action parameters of `value_loss_above_ln3` and `value_head_one_sided` (OD-7) made it 96, and the owner's 2026-10-06 rule 9 (`gradient_spike`, its own action parameter; OD-20, whose approval covers the edit the plan needs) makes it **97**. If the cadence plan's own +1 (its TE-1) lands first, the count is one more than whatever it is then.
- No other existing test is expected to change:
  - `CliTrainingRecorderTests` checks keys it names, not the absence of others (`:39-81`);
  - `TrainingAlarmControllerTests` exercises `evaluate`, `dismiss`, `clear` and the streaks, whose observable behavior this plan keeps (T9);
  - `LayerHealthTests` uses `liveLine`, which stays.

  Anything else found during implementation goes to the owner first.

### X3. Existing tests that guard the change
- `LayerHealthTests`, `TrainingAlarmControllerTests`, `CliTrainingRecorderTests`, `AutoTrainTerminationTests`, `TrainVsUciSessionTests`, `TrainingParametersTests`.
- `ResumeEquivalenceTests`: the observer must not change a corpus-replay resume.
- `DropoutRNGStateTests`: no RNG draw.

---

# Part V — Validation

Every step runs only when no training run is live (owner rule), or with the owner's explicit OK.

- **V-1 — Offline replay of the incidents.**
  - Run `--replay-health-log` on each run's log(s), named in **Evidence** (C's two logs together, then segment 1 alone). Use the frozen build of the implementing commit.
  - **Pass:** A, R7 and R8 print no `[ALARM] health` line. B's and C's events match the Evidence table in rule, severity and trainer step. For A and B (still growing when measured) the comparison stops at the Evidence cutoff; later events are reported, not judged.
  - Any mismatch is explained (e.g. a window-semantics difference) and shown to the owner before thresholds or expectations move.
  - Also run it on the two long lines (`dcm_log_20260702-201756.txt`, `dcm_log_20260727-094049.txt`) with `--segment-step-as-trainer-step` (D4). **Reported, not judged:** the events are recorded in the implementing commit's notes and compared with the Long-runs table; OD-17 decides what they mean.
  - The fresh GUI runs (GUI-F1, GUI-F2) are not replayed: their `[STATS]` values are rolling means, and feeding them to the evaluator as samples would judge a different statistic. Their numbers are in Evidence; V-6 is the in-app check.
  - Also run it on the 73-log survey set (each log alone; GUI-only logs get the layer-health rules, D4). **Pass:** no event in any log other than B, C and the two older-model runs: `dcm_log_20261003-001701.txt` gives a `dead_channels` warning (absolute form, OD-16: 17 of 1,808 dead, below 5%; worst site `value.bn` 2 of 16, below 20%), and no rule 3 event for its 49/128 after 3 trained steps; `dcm_log_20261005-012743.txt` gives a `policy_offset_drift` warning at trainer step 100 and nothing else: it is a 500-step benchmark of the older model `20260708-4-kEiZ` (nT8Y line, loaded from `20260701-nT8Y-resume4-replay-latest.safetensors`), whose `pLogitMean` sits at −14.26 from its first diagnostic row — a shared policy offset carried in from the mixed-precision era, which rule 7 correctly reports (always-on channels are not alarmed; `rvMaxOverMedian` 332 < 1,000). This corrects the earlier expectation of "none", found by re-running the survey under the decided thresholds. In particular, none of the 16/16 `valueFC1ZeroVel` saves in the 15 test-process logs (trainer clock 0–274, nothing trained) may raise rule 3.
- **V-1c — Parked counts on SiLU runs (OD-18).** Run `--replay-health-log` on a log written after P1 by a SiLU-tower run (B-silu resumed, or a new one) and check its `parkedBy=` against `bn_liveness.py` on the same steps' checkpoints. **Pass:** equal per site. Also check that `[LAYER-HEALTH]` greps on `dead=` / `off=` / `alwaysOn=` give the same numbers as before for relu / leaky_relu runs (the fields are unchanged).
- **V-1b — R7/R8 layer health.** R7/R8's logs predate `[LAYER-HEALTH]`, so the brief's "R7/R8 have 0 dead channels" is not in their logs. Run `--analyze-numerics <file> --numerics-static-only --numerics-out <scratch folder>` on R7's and R8's final checkpoints (stems in `experiments/20261002-noSE-noReZero/README.md`). Layer health is part of the static checks (`Network/NumericsAudit.swift:209`): no forward passes, read-only on the model, though it builds one MPSGraph to read variable names (`App/NumericsAuditCLI.swift:113-124`). **Pass:** 0 dead channels and `valueFC1ZeroVel` ≤ 1/128. If a file carries no optimizer velocity, the audit reports `valueFC1ZeroVel` unavailable; that half is then reported as not checked, never as passed.
- **V-2 — Per-step loss distribution (rule 6).**
  - A 2,000-step corpus replay on a healthy configuration (R7's parameters, `--seed`), with a temporary `--output`. Read the largest per-window ratios from the `[HEALTH] check` lines, which report them permanently (`lossMaxRatio=` = the largest `max_W(loss) / ref`, and `lossMedianRatio=` = the largest `median_W(loss) / ref`, over the evaluations since the previous check; `--` when no evaluation had a reference). No temporary debug code.
  - **Pass:** the largest `lossMaxRatio` over the run is below 2.0 and the largest `lossMedianRatio` below 1.2, so the decided 3× and 1.5× arms keep a margin. **Otherwise V-2 fails and the measured ratios go to the owner**; the thresholds stay as decided (OD-1, OD-14, OD-17) unless the owner makes a new decision. The plan never adjusts a threshold from a measurement on its own.
- **V-3 — Live CLI run, log-only.** Run C's recipe (`experiments/20261005-lr-schedule-ab/README.md` Arm C) for 600 steps with `--output`.
  - **Pass:** `[ALARM] health raise rule=dead_channels severity=critical` at or before trainer step 100. `illegal_mass` critical within 100 steps of the divergence. `alarms` in `results.json` matches the log line for line. The run continues to its step limit (log-only).
- **V-4 — Stop action, CLI.** Same recipe with `training_health_action_illegal_mass=1`.
  - **Pass:** the `[ALARM] health stop` line is followed by `[REPLAY] training health alarm illegal_mass requested a stop — stopping at step N`, where N is the segment step of the evaluation that raised it (no further step runs), and the `health-stop` final save records trainer step = that evaluation's trainer step. `termination_reason` is `training_health_alarm`, the exit status is per OD-4, and a `--resume-exact` from the saved file reports `EXACT`.
  - Repeat on train-vs-UCI with a 2-opponent pool. **Pass:** a `vsuci-health-stop` session folder exists.
- **V-5 — Observer neutrality and cost.**
  - Three 300-step corpus replays with the same `--seed`, each a **fresh** run (trainer clock starting at 0, so segment and trainer steps coincide): two with `training_health_alarms_enabled=false` (D1, D2), one with it on (E).
  - **Pass, neutrality:** compare the `[REPLAY]` lines field by field, ignoring `ms` and the line timestamp. If D1 and D2 are identical, E must be identical to D1. If D1 and D2 differ (the GPU is not deterministic on this machine), compute each numeric field's largest relative difference between D1 and D2 over all rows, with relative difference `|x − y| / max(|x|, |y|)` and 0 when both are 0; E against D1 must stay within that per field, a field that reads `--` must read `--` in all three, and the report says the exact comparison was not possible.
  - **Pass, cost:** two measurements, because `ms` covers only `trainStep` (`Training/ChessTrainer.swift:7243-7269`, printed at `CLI/CorpusReplayRunner.swift:1919`) and the monitor's work runs after it.
    - E runs with `training_health_check_interval_steps=50`, so its `[HEALTH] check` lines (plus the `final=true` one) cover the whole run. Σ`cost_ms` ≤ 0.5% of Σ`train_ms` over those lines (the same steps, from the same lines).
    - Wall time per trainer step, from each run's own step lines, without assuming which steps are logged: take the first and the last `[REPLAY]` step line whose `trainerStep` lies in 50…300, and divide the time between their timestamps by the difference of their `trainerStep`s (no save falls in that range on either save grid). |E − D1| ≤ max(1% of D1, |D1 − D2|). E's own figure is cross-checked against its `[HEALTH] check` lines' `train_ms + cost_ms` over the same steps.
    - **Value-FC1 read (D6).** Corpus replay does no dedicated read while each save and its checkpoint pass succeed, so this is measured on a train-vs-UCI run without `--enumerate-checkpoints` of at least 3,000 steps (and again in V-6's GUI run): every `[LAYER-HEALTH] value-fc1` line's `readMs + summaryMs` ≤ 0.5% of the `train_ms` of the 1,000 steps it covers (in practice ≤ 5 × the median step `ms`), and Σ`cost_ms` still ≤ 0.5% of Σ`train_ms`. A corpus-replay run of 3,000 steps in which every save and its checkpoint pass succeed logs **no** `value-fc1` line, fresh and resumed from a checkpoint whose step is not a multiple of 1,000 (no double read).
- **V-6 — GUI.**
  - Build New Model, then Play-and-Train for 3,000 trainer steps with defaults. **Pass:**
    - a `[HEALTH] check` line at the first `[STATS]` emit at or after each multiple of 1,000 trainer steps;
    - one `[LAYER-HEALTH] value-fc1` line every 1,000 trainer steps from the session's start (no save fell in the run), with `readMs` within V-5's bound;
    - no `[ALARM] health` line;
    - the alarm list stays hidden;
    - the Health tab shows 12 settings and edits log `[PARAM]` lines;
    - `illegal_mass` not-learned did not fire at 2,000. This is the fresh-GUI-net case the replay data cannot cover. If it fires, the owner decides the grace before merge.
  - Then load C's step-513 abort save as a trainer (or any damaged model) and start, all actions `log`. **Pass:** `dead_channels` critical at the first live read; the list shows it in light and dark mode; VoiceOver reads the row label; the beep sounds although the banner shows nothing; the list's Silence silences it.
  - While it is active, set `dead_channels` to `stop_on_critical`. **Pass:** training suspends at the next evaluation (R2) with the list's suspension header and `[HEALTH] training suspended …`; arenas log `[ARENA] skipped — training suspended (health alarm dead_channels)`; a periodic save (interval temporarily shortened) and a File ▸ Save Session both **complete** with a `[CHECKPOINT] Saved session` line, not a pause timeout; the step count does not advance.
  - Stop, then Start with the action still `stop_on_critical`. **Pass:** the first evaluation re-raises and suspends again (R3). Set the action to `log`, Stop, Start. **Pass:** training continues and the alarm stays listed.
- **V-7 — GUI `--train`.** A 10-minute `--train` run from the damaged model with `dead_channels` set to `stop_on_critical` before start. **Pass:** the process exits through `AutoTrainTermination` with `termination_reason: "training_health_alarm"`, an `alarms` array matching the log's `[ALARM] health` lines, the exit status OD-4 settles for GUI `--train`, and a drained log.
- **V-8 — Full test suite** before merge (this touches persistence and parameters). Every test passes, and the only edited test is OD-6's (`test_registry_size` 85 → 97, plus the cadence plan's own +1 if it lands first).

---

# Part P — Phasing

Each phase builds once at its end, then is committed (owner's standing rule for approved multi-phase plans).

**Order with the adjacent plans.**
- **P2 is gated on the owner's approval of the *redesigned* `STATS_LINE_RESUME_CADENCE_FIX_PLAN.md`, and on that plan landing.** The redesign under discussion: time-based step lines (about 180 s, emitted on the first diagnostics step after each deadline; dense every 50 trainer steps for the first 1,000), lines and saves at overall trainer-step multiples of 1,000, possibly enumerated checkpoints named by overall step. This plan's requirements on it are three: the per-step order step-line block → live alarm evaluation (its own 50-step tick, OD-21) → save block, so at a save step the live evaluation runs before the save's checkpoint pass (R0; the committed redesign states it and notes it holds because every 1,000-step save is a multiple of 50); consecutive saves stay 1,000 trainer steps apart with the first within 1,000 of the start (D6; it holds); and the `[REPLAY]` / `[VS-UCI]` field format is kept (`trainerStep=`, `--` = not measured; D4; it holds). The step line is logging only: nothing here rides it. Where a step line and an evaluation fall on the same step, one live `[LAYER-HEALTH]` read serves both (the cadence plan's OD-15, adopted by this plan; P2 implements it by handing the step line's `LayerHealthLog.LiveOutcome` to the evaluation). If the approved redesign changes any of them, R0, D6 or D4 is revised before P2.
- **This plan's P1–P3 land before `HPARAM_RECORDING_PLAN.md` P4** (its O-22, decided yes). HPARAM P4 builds the required `configuration.health_alarms = {evaluations, raised}` from this plan's `segmentSummary()` (OD-10).
- `HEAD_ACTIVATIONS_PLAN.md` has landed (`LayerHealth.valueFC1Layer(for:)` reports `valueHeadFC1HiddenActivation`, so rule 3 follows the value head's own activation). The cadence plan's step-line schedule (`TrainingStepLineSchedule.lineDue`) is logging only; P2 does **not** wire evaluations into it: the CLI loop calls the live evaluation on `TrainingHealthCadence.isLiveEvaluationStep(trainerStep:)` between the step-line block and the save block (OD-21), sharing the step line's live read when both are due (cadence plan OD-15). P1 touched the runner blocks only for the mechanical `liveLines` → `live(trainer:).lines` conversion; P2 re-cites T3/T4 against the fixed code before implementing.

- **P1 — Pure core and incident tests.** T1; T2 including the conversion of all four `liveLines` callers to `live(trainer:).lines` (D5), so `liveLines` can go and P1 builds alone; `TrainingHealthConfig` with its memberwise initializer only; the `TrainingAlarm.Severity` `Codable` extension. X1's evaluator, monitor, log, log-replay and incident-replay tests; `valueFC1ReadDue` (D6) and its tests. No behavior change in any run.
- **P2 — Parameters, CLI paths, results, offline replay.** Part K items 1–6 and 8 (CLI); `TrainingHealthConfig(_ snapshot:)`; T3–T7; `ChessTrainer.readTrainableVelocity(named:)` and `ValueFC1VelocityReadTests` (D6); X1's parameter and recorder tests; X2 (after OD-6). Validation V-1, V-1b, V-2, V-3, V-4, V-5.
- **P3 — GUI.** T8, T9, Part K item 7 (the Health tab) and item 8's GUI side; X1's alarm-controller tests. V-6, V-7.
- **P4 — Documentation.** T10.
- **P5 — Later work.** OD-9 (shared conditions for the GUI-only detectors; decided yes, each `TrainingAlarmControllerTests` edit listed for approval when planned), OD-10's record field is not in this plan: `configuration.health_alarms` is implemented by HPARAM_RECORDING_PLAN P4 from this plan's `segmentSummary()` API, which lands in P1. (OD-15, decided: the value-FC1 velocity check on a fixed 1,000-step interval, is not P5 — it lands in P1–P3 with D6.)

---

# Owner decisions needed

Decided by the owner on 2026-10-05: every decision below (OD-1 in two rounds, OD-2 to OD-11, OD-13 to OD-17). OD-12 is superseded by its own plan. OD-16's decision carries an owner "revisit later" note. Each decision is recorded after its original text.
- **Sequencing, not a decision:** P2 is blocked on the cadence plan, which is being redesigned (Part P).

- **OD-1 — Thresholds.** Accept Part R's defaults? Specifically:
  - `dead_channels` warning in the form OD-16 picks; critical at ≥ 10% overall or ≥ 50% of one site (B is then a warning; C is critical at step 50); *(the owner's decision below changes this to 5% / 20%)*
  - `value_fc1_zero_velocity` warning at 5%;
  - `gradient_collapse` at `gNorm` < 0.1;
  - `policy_offset_drift` at |pLogitMean| ≥ 3;
  - `loss_spike` 1.5× median / 3× max (to be confirmed by V-2);
  - `bn_running_variance_runaway` at 1,000.

  **Decided (owner, 2026-10-05):** `dead_channels`: warning on any dead channel (the absolute form; OD-16 was then still open, and was later decided: absolute, revisit later), critical at **≥ 5% overall or ≥ 20% of one site** (changed from 10% / 50%). With these, B is **critical at 300** (`value.bn` 5 of 16 = 31%) and C is still critical at 50; the event tables and tests are updated. All other OD-1 thresholds accepted, **except `bn_running_variance_runaway` at 1,000, which stayed open** (owner unsure; measured `rvMaxOverMedian`: A max 7.4 at trainer step 34,250, after the Evidence cutoff — 7.2 within it; B max 78.6 at trainer step 1,400 and ≤ 36.3 after 5,000; C 461,870–548,521 on its live lines from trainer step 500 on; the two older-model runs 231 and 332).

  **Decided (owner, 2026-10-05):** (second round, same day) `bn_running_variance_runaway` at 1,000 — decided. The owner also agreed that B's `value.bn` at 5 of 16 at step 300 is critical, as the 20% arm says (no change).
- **OD-2 — Which rules may stop runs.**
  - All defaults are `log`, as requested.
  - Recommendation for when the owner wants unattended protection: `stop_on_critical` for `non_finite`, `illegal_mass`, `gradient_collapse` and `dead_channels`. Of these, only C reached a critical level (and no run reached `non_finite`).
  - `log` for the rest: their evidence is warnings, and B ran to its end with them.

  **Decided (owner, 2026-10-05):** accepted: every rule defaults to `log`; the recommended `stop_on_critical` set is documented (Health tab help text and `documentation/training-health-alarms.md`), not applied. Note under OD-1's new thresholds: `stop_on_critical` on `dead_channels` would have stopped B at trainer step 300.
- **OD-3 — Action granularity.** One Int per rule with three values (Part K), versus one global action plus per-rule overrides. The recommendation is per rule, as requested. **Decided (owner, 2026-10-05):** accepted: one action per rule.
- **OD-4 — Exit status on an alarm stop.** CLI paths: recommendation is a new status 35 after a successful final save (0 would let a chain script treat the stop as completion). 4 is not available: `--show-default-parameters` misuse already exits 4 (`App/DrewsChessMachineApp.swift:831`). 35 is unused anywhere in the source and sits next to the runners' own 33 (run failed). The alternative is 0, with the reason in `results.json` only. GUI `--train`: every termination exits 0 today through `AutoTrainTermination` (`App/AutoTrainTermination.swift:106-112`), legal-mass collapse included; recommendation is to keep 0 there (the reason is in `results.json`) unless the owner wants 35 on both, which would mean giving `writeResultsAndExit` a status argument. **Decided (owner, 2026-10-05):** accepted with an unused exit code: 35 on the CLI paths; GUI `--train` keeps 0.
- **OD-5 — GUI stop semantics.** Also: gate Train ▸ Promote Trainee Now during a `.divergence` suspension as well (recommended; today it is not gated, Risks) — this plan gates it for `.healthAlarm` either way. Recommendation: suspend training with the worker parked (R3), with `trainingSuspension` replacing `trainingSuspendedByDivergence`. One consequence to accept or veto: during a health suspension the periodic autosave runs, and every successful save moves `LastSessionPointer` (`App/SessionController+Checkpoint.swift:590-594`), so the launch-time "Resume Training" offer would then point at the damaged state. Alternatives: a full Stop (it clears the banner, `App/SessionController+Training.swift:2520-2522`, so the reason would vanish), or holding the training pause gate (but the gate is not counted, and an arena's resume would release it: `Training/WorkerPauseGate.swift:108-116`). **Decided (owner, 2026-10-05):** accepted as recommended, including the `LastSessionPointer` consequence and gating Promote Trainee Now during a divergence suspension too.
- **OD-6 — Test edit.** `test_registry_size` 85 → 98. Pending the owner's explicit approval; P2 cannot pass the suite without it. **Decided (owner, 2026-10-05):** approved for the edit the plan needs. With `value_loss_above_ln3` and `value_head_one_sided` dropped (OD-7) the plan adds **11** parameters, not 13, so the edit is **85 → 96**, not 98.
- **OD-7 — Keep `value_loss_above_ln3`?** It is not supported by the incident data, and it is near-firing on healthy GUI self-play. The recommendation is to keep it as log-only with the long sustain, or to drop it. **Decided (owner, 2026-10-05):** **drop `value_loss_above_ln3` entirely** — its rule, its action parameter, its tests and its evidence rows are removed, and the rules are renumbered (the former rule 10, `bn_running_variance_runaway`, is now rule 9). The plan now has nine rules and twelve parameters. (Superseded by the next decision.)
  - **Decided (owner, 2026-10-05): drop both `value_loss_above_ln3` and `value_head_one_sided`.** The owner's "drop it" covered the one-sided value-head rule too; it had been read as covering only the value-loss rule. The one-sided rule, its enum case, its action parameter, its tests, its evidence and event-table entries, its notes and its draw-gate discussion are removed, and the rules are renumbered: `bn_running_variance_runaway` is now rule 8. **The plan has eight rules and eleven parameters**, so the registry edit is 85 → 96.
- **OD-8 — Sound.** Beep on critical health raises and escalations only (recommended), or never. Optional macOS notification (`UNUserNotificationCenter`) for critical: off unless requested. **Decided (owner, 2026-10-05):** accepted: beep on critical only; no notification. As T9 implements it: the beep loop runs while any critical alarm is active and not silenced — including a critical already present at the first evaluation — and never for warnings alone.
- **OD-9 — One evaluator for the old GUI detectors too.** Move the divergence, value-saturation and pD detectors' conditions and the legal-mass probe's condition into `TrainingHealthEvaluator`, so the CLI paths get them. This changes `TrainingAlarmControllerTests` (each edit listed for approval then). Deferred to P5. **Decided (owner, 2026-10-05):** yes, in P5; each `TrainingAlarmControllerTests` edit is still listed for approval when P5 is planned.
- **OD-10 — Lineage.** Record a per-segment alarm summary (rule, first trainer step, highest severity) in the lineage record's segment summary when HPARAM_RECORDING_PLAN P4 introduces schema 3? The recommendation is yes, as one optional field written by `LineageTracker`. Not before schema 3, because this plan does not change the model-file format. **Decided (owner, 2026-10-05):** yes, provided the lineage JSON cannot be flooded. Bounded design: one entry per rule that raised in the segment — `{rule, first_trainer_step, highest_severity, raise_count}` — and no per-event list; rules that never raised have no entry. Worst case per segment, in compact JSON with keys `rule` / `first_trainer_step` / `highest_severity` / `raise_count`: the longest entry (`bn_running_variance_runaway`, both integers at the `Int` maximum, 19 digits; `"critical"`) is 143 B, and all eight ids at those maxima make a 1,068 B array (≈ 1 KB, before the enclosing field name and any pretty-printing whitespace; restated for eight rules after OD-7 dropped the second rule). Realistic values (a 7-digit first step, a 4-digit raise count) give ≈ 850 B. A record carries a summary of every earlier segment of its run, so the total grows by at most ≈ 1.1 KB per segment, never with the number of events. Lands with HPARAM_RECORDING_PLAN P4's schema 3 (P5 here).
  - **Superseded in shape and owner by `HPARAM_RECORDING_PLAN.md`** (its S2 and "Interaction with adjacent plans"; owner decision O-22 there: this plan's P1–P3 land before HPARAM P4). At schema 3 every key is required, so the summary is not an optional field: it is the required key **`configuration.health_alarms = {"evaluations": N, "raised": [{rule, first_trainer_step, highest_severity, raise_count}]}`**, written by HPARAM P4's `LineageTracker`, which also merges the summaries of a segment's several GUI monitors (Continue, keep-trainer starts). This plan provides only the per-segment summary API: `TrainingHealthMonitor.segmentSummary() -> TrainingHealthSegmentSummary` (`Codable`, exactly that shape and those keys), where `evaluations` counts the committed evaluations (live and checkpoint; stale ones excluded; 0 when alarms are disabled) and `raised` has one entry per rule that raised, never per event. It is added in P1 with tests (`testSegmentSummaryOneEntryPerRaisedRule`, `testSegmentSummaryCountsCommittedEvaluationsOnly`, `testSegmentSummaryZeroWhenDisabled`); this plan writes no lineage field itself.
  - Size bound, with `evaluations` added: the object at all eight ids and every integer at the `Int` maximum is 1,113 B in compact JSON (1,068 B array + `{"evaluations":<19 digits>,"raised":` + `}`; ≈ 880 B with realistic values), so the per-segment growth stays at most ≈ 1.1 KB (8,904 B for the eight-segment v5 line), never with the number of events.
- **OD-11 — CLAUDE.md.** Add `[HEALTH]` and `[ALARM] health …` to the tag list in "Where to look for runtime state", a sentence on the alarms under "Training observability", and `vsuci-health-stop` beside `vsuci-periodic` / `vsuci-final` / `vsuci-abort` in "Saved model state". **Decided (owner, 2026-10-05):** yes.
- **OD-12 — Stats-line cadence after a resume.** Superseded by `documentation/plans-active/STATS_LINE_RESUME_CADENCE_FIX_PLAN.md`.
- **OD-13 — GUI `--train` alarm stop: save first?** The legal-mass precedent exits without a session save. The recommendation is to follow it, for consistency. **Decided (owner, 2026-10-05):** yes.
- **OD-14 — Thresholds as parameters?** The recommendation is no: declared constants, like `LayerHealth`'s and `TrainingAlarmController`'s. 8 rules with 1–3 thresholds each would add about 18 knobs. (V-2 reads its measurements from the permanent `[HEALTH] check` fields; there is no temporary debug switch.) **Decided (owner, 2026-10-05):** yes: declared constants.
- **OD-15 — Live value-FC1 velocity in the GUI.** Rule 3 is checkpoint-only, and GUI checkpoints are hours apart. Reading `value.fc1.weight`'s velocity on the live tier (128 × 1,024 floats = 512 KB per read) would be new GPU readback on the trainer's queue. The recommendation is no.
  - **In plain terms** (the owner asked what it means): rule 3 spots a dead value head by finding hidden units in the value head's first fully connected layer (`value.fc1`) whose optimizer momentum has decayed to exactly zero — units that stopped learning. That momentum is only read when a checkpoint is saved, because reading it costs an extra 512 KB copy from the GPU. In the GUI, saves happen only every few hours (periodic) or at promotions, so the GUI would notice this failure hours late. OD-15 asks whether to also read that one tensor's momentum every `[STATS]` emit (about once a minute), so the GUI notices within minutes — at the cost of a 512 KB GPU read per minute, scheduled between training steps on the trainer's queue.
  - **Decided (owner, 2026-10-05):** every check runs on a set interval or trigger. Rule 3's value-FC1 velocity read runs every 1,000 trainer steps on every path: corpus replay reuses its 1,000-step saves (no double read); train-vs-UCI reuses its enumerated checkpoints when they are on; otherwise — the GUI, whose saves are hours apart, and train-vs-UCI without `--enumerate-checkpoints` — one dedicated read of that tensor on the trainer's `executionQueue` between SGD steps, under the probe-isolation rules (D6). Cost in D6; checked in V-5. The "Every check" table near the top lists every check's trigger per path.
- **OD-16 — Rule 2's warning form.** Absolute (warn on any dead channel) or new-damage (warn only when a site's dead count rises above its count at the process's first evaluation); the critical arms are absolute in both. Recommendation: new-damage, so a long-trained line with a few dead channels (17 of 1,808 on the grafted v5 model) does not hold a permanent warning, with the baseline logged once so it is never hidden. Until it is decided the plan implements the absolute form (OD-1's decision). **Decided (owner, 2026-10-05):** keep the absolute warning form for now. **Revisit later** (owner): a long-trained line that carries a few dead channels holds a permanent warning under it; the new-damage form, its tests and its offline semantics are kept, not built, in the **Deferred (OD-16 revisit)** section after Part R. Note that with OD-1's 20% per-site critical arm, a line whose small `value.bn` (16 channels) already carries 4 or more dead channels is critical at its first evaluation under either form.
- **OD-17 — Long-run evidence.** On two older long mixed-precision lines from before the head-numerics fix, rule 6 raised 21 times on the 268,500-step v5 line (ratios up to 2.36) and the since-dropped value-loss rule raised from step 58,050 on the 1.4M-step qeu8 line (Long runs). Keep rule 6's thresholds as proposed and treat those raises as true signals (they predate `da15920`), or loosen rule 6's median arm before merge? Recommendation: keep, since the defaults are log-only, and revisit after the first long post-fix run. (Its value-loss half is moot since OD-7 dropped that rule.) **Decided (owner, 2026-10-05):** keep the thresholds.
- **OD-18 — Rule 2 covers SiLU and GELU (owner, 2026-10-06).** **Decided:** `LayerHealth` classified only relu / leaky_relu BN sites, so a SiLU tower (B-silu) was unchecked. Rule 2 now counts **parked** channels over every BN site an activation consumes: per channel the excess pass-through X = (E\|f'(γz+β)\| − floor) / (1 − floor) with z ~ N(0, 1), parked when X < Φ(−3), mostly off when X < Φ(−2). For relu / leaky_relu this equals today's β/\|γ\| < −3 / < −2 exactly (decided from β/\|γ\| itself, so the counts agree to the channel); for silu it is the Gauss–Legendre panel integral of `experiments/20261005-lr-schedule-ab/bn_liveness.py`, and for (exact, erf) gelu the same integral with gelu' = Φ(y) + yφ(y). Single source in Swift (`Training/BatchNormPassThrough.swift`); the Python tool mirrors it (gelu added there too). Existing `[LAYER-HEALTH]` field meanings for relu / leaky_relu are unchanged; the new counts are appended (`parkedSites= parkedCh= parked= parkedOff= parkedBy=`) and a `parked channels` section follows the BN table in the checkpoint block. Thresholds unchanged. Tested against B-silu's real γ/β at trainer steps 20,000 / 21,000 / 22,000 (read-only from its checkpoints into a committed test resource): `policy.pre_bn` 0 parked at 20k, 20 parked / 7 mostly off at 21k, 21 / 8 at 22k, matching `bn_liveness.py`.
- **OD-19 — Rule 4's relative regression arm (owner, 2026-10-06).** **Decided:** add arm (ii) — window median ≥ 0.3 **and** ≥ 10 × the run's own running minimum of window medians — so regressions like B-silu's (0.005 → 0.78–0.80 at LR 0.88, `dcm_log_20261005-234437.txt`, trainer steps 20,600–21,000) fire. The owner first approved replacing arm (i) with it; the implementer found that loses arm C (minimum 0.236, so 10 × min = 2.36 is unreachable), and the owner confirmed keeping **both arms (OR)** the same day. Clear < 0.15 for 2 evaluations, span 50 (Rule 4 note). Not-learned form unchanged. B-silu's excerpt is an incident fixture (`testBSiluIllegalMassRegressionRaisesAt20700`).
- **OD-20 — Rule 9 `gradient_spike` (owner, 2026-10-06).** **Decided:** warning when the window's largest `gradGlobalNorm` ≥ 5 × the median `gradGlobalNorm` over the previous 1,000 trainer steps (reference built like `loss_spike`'s: ≥ 5 records spanning ≥ 200 steps); clear when `max_W` < 2.5 × ref; no sustain. B-silu at 20,600 (2.566 vs ≈ 0.36) fires; the prototype's largest healthy ratio is 1.40 (B; A 1.25, R7 1.29, R8 1.30). Its own action parameter: 12 parameters, registry 85 → 97 (the owner's approval covers the edit the plan needs).
- **OD-21 — Evaluation cadence decoupled from log lines (owner, 2026-10-06).** **Decided:** live evaluations run every 50 trainer steps (overall trainer-step multiples) on every path — GUI, corpus replay, train-vs-UCI — independent of when `[STATS]` / `[REPLAY]` / `[VS-UCI]` lines are written (the cadence redesign moves those to ≈ 180 s). Per-step order on the CLI: step-line block → live evaluation → save block (R0 holds by position: every save step is a multiple of 50). One live read serves both on a shared step (cadence plan OD-15, adopted). Cost bounded: one ≈ 1 ms live read per 50 steps plus the summary (D2's `cost_ms`; SiLU / GELU summaries add a per-channel integral, measured in V-5). The offline replay is unchanged: it evaluates once per logged row (sparse semantics, D4).
- **OD-22 — Name every affected site (owner, 2026-10-06).** **Decided:** the `dead_channels` alarm (and the parked counts) name every affected site with its counts — tower, policy head, value head, any site — not just the worst one: `sites=<site>(<parked>/<channels>),…` (largest fraction first) on raise, escalate, worsen and `active` lines and in `results.json`'s event `detail`, `dead_channels_sites=` on the `[HEALTH] check` line while the rule is active, and `parkedBy=` on every `[LAYER-HEALTH]` compact line. Offline, logs written before the parked counts name only the live line's `worst=` site between checkpoints (`coverage=relu_leaky_relu_only` on the line; header notes it).

---

# Risks

- **Thresholds come from one corpus, one architecture family and one batch size.**
  - `gNorm` and the loss-spike ratio are scale-dependent. A different architecture or loss weighting can move them.
  - Mitigations: declared constants; the `[HEALTH] check` line shows each rule's no-data count; V-1 reruns the survey; defaults are log-only.
- **Fresh GUI runs are only partly in the evidence.** Two fresh GUI runs (GUI-F1 to 14,781 steps, GUI-F2 to 2,041) cover the scalar rules through their rolling `[STATS]` means; none since `[LAYER-HEALTH]` shipped covers the layer-health rules. V-6 is the in-app check.
- **Long runs are outside the main evidence (OD-17).** Every baseline stops by about 35,000 trainer steps; the Long-runs replay in Evidence shows rule 6 firing on two older long lines.
- **`HEAD_ACTIVATIONS_PLAN.md` (in flight) changes what some rules see.** It adds per-site activations, including smooth ones on the head hidden layers. Rule 2 already treats SiLU / GELU BN sites as not classified, but its per-site channel map gains and loses sites with it. Rule 3's premise is a dead ReLU unit in `value.fc1`: with a leaky or smooth value hidden activation the gradient is rarely exactly zero. This plan decides what rule 3 does then: it is `not_applicable` unless `LayerHealth.valueFC1Layer(for:)` reports `relu`, and D6 reads nothing (rule 3 note). Today that function ignores any value-head activation (`Training/LayerHealth.swift:245-252`, tower activation); whichever plan lands second (this one or `HEAD_ACTIVATIONS_PLAN.md`, in either order) makes it return the value-head hidden activation and re-checks rules 2 and 3 against the other plan's sites.
- **Promote Trainee Now is not gated during a divergence suspension today** (`App/SessionController+ManualPromote.swift:35-52` checks Play-and-Train, a running arena and a save in flight, not the suspension), so a NaN trainer can be promoted from the menu. OD-5's decision closes it: this plan gates it for both `.divergence` and `.healthAlarm` (R3, T8).
- **Out-of-order GUI checkpoint evaluations and promotion races.** Handled by the evaluation lock, the generation and the stale-checkpoint rule (D2); `TrainingHealthMonitorTests` covers each (X1).
- **The `trainingSuspension` refactor touches the divergence path.** Its gates are listed one by one (R3); the divergence column keeps today's behavior and its worker still returns. V-6 exercises the health case only. A missed reader would fail to compile, because the `Bool` is removed, not kept beside the enum.
- **Existing behavior seen during review, not changed here.** In a divergence suspension the trainer worker has returned, so any session save (File ▸ Save Session, despite the banner's "you can still save manually" intent at `App/SessionController+Heartbeat.swift:372-375`) waits for a pause acknowledgement that never comes and aborts at the training-pause timeout (`App/SessionController+Checkpoint.swift:432-437`). The health suspension avoids this by parking the worker (R3). Whether to park the divergence worker too is for the owner, separately.
- **Log volume.** At most one `[HEALTH] check` line per 1,000 trainer steps, plus event lines. C would add about a dozen event lines and up to 6 reminder lines per 1,000 steps at worst.
- **`SaveKind` gains a case.** Nothing iterates `TrainVsUciSession.SaveKind.allCases` in the tests today (checked). `TrainVsUciSessionTests.swift:204` uses `.final` only.

# Non-goals

- Changing the trainer's non-finite-loss halt or any training math.
- Extra forward passes or probes, and any GPU read beyond D6's one value-FC1 velocity read per 1,000 steps (OD-15, decided). Otherwise the monitor only reads values the paths already compute.
- Alarm suppression around LR-cycle peaks (R0 explains why not).
- Persisting alarm state in session files or model files. Every process starts its own evaluator. The not-learned gates use the lineage-continuous trainer clock, and the damage rules' critical arms read absolute counts, so a resumed process still raises critical damage at its first evaluation. A warning for damage already present at start depends on rule 2's form (OD-16).
- Rewriting the existing GUI detectors or banner (OD-9 is the separate decision).
- Automatic remediation (LR reduction, rollback): an alarm logs or stops; it never changes the run's parameters.
- A Python copy of the rules.

---

# Implementation notes (P1)

P1 landed on branch `worktree-agent-a5aae13e873e3a8a4` (2026-10-06): the pure core, the parked classification (OD-18), the offline replay core, `LayerHealthLog.live`, and the tests. No run's behavior changes except the `[LAYER-HEALTH]` lines, which gain the parked fields (OD-18). Decisions made while implementing, each a refinement of the text above that the text alone did not settle:

- **Files.** `Training/TrainingHealth.swift` (rules, actions, config, thresholds, cadence, inputs, outputs, evaluator, stop policy, segment summary), `Training/TrainingHealthMonitor.swift`, `Training/TrainingHealthLog.swift`, `Training/TrainingHealthLogReplay.swift`, `Training/BatchNormPassThrough.swift` (OD-18), and the `LayerHealth` / `LayerHealthLog` changes. Tests: `TrainingHealthEvaluatorTests`, `TrainingHealthMonitorTests`, `TrainingHealthLogTests`, `TrainingHealthLogReplayTests`, `TrainingHealthIncidentReplayTests`, `BatchNormPassThroughTests`, shared builders in `TrainingHealthTestSupport.swift`. Resources in `DrewsChessMachineTests/Resources/TrainingHealthIncidents/` (the test target is a file-system-synchronized group, so they are bundle resources without a project-file edit; they land flattened in the bundle's `Resources/`, hence the `TrainingHealthIncident-` / `TrainingHealthBSilu…` prefixes).
- **`TrainingHealthActions` instead of a dictionary.** D1 sketched `actions: [TrainingHealthRule: TrainingHealthAction]` "total over allCases". It is a struct with one stored field per rule and a `subscript(rule)`, so it is total by construction and a lookup can never miss (no optional, no fallback). It encodes as `{rule id: action name}` exactly as D1 asks.
- **`TrainingHealthConfig.init` throws.** It keeps the memberwise shape but refuses a check interval below 1 (a division by zero) and negative step counts, so no evaluation can run under an unusable config. P2's `init(_ snapshot:)` sits on top of it.
- **`evaluateLive`'s layer-health argument is one enum**, `TrainingHealthLiveLayerHealth`: `.read(digest, trainerStep:)`, `.readFailed`, `.absentFromLog` (offline only). D2's signature had an optional digest plus an optional digest step, which can disagree; the enum cannot. `.absentFromLog` lets the offline replay evaluate a row with no live line without counting it as a failed read.
- **Rule state is an array indexed by `ruleOrder`** (total by construction, no optional lookups or preconditions). `ruleOrder` equals the `allCases` index, pinned by `testEveryRuleHasExactlyOneAction`.
- **Rule 1 also counts non-finite lean values in the window** (loss, illegal mass, gradient norm), not only a non-finite `pLogitMean`. In-app they cannot occur (the trainer halts first), but a window statistic that silently skipped a NaN would hide it; they are excluded from the medians and counted instead.
- **Rule 6 and rule 9 need a positive reference.** A reference median ≤ 0 (the policy loss can be negative) makes the ratio meaningless, so the rule has no data on that evaluation rather than a nonsense ratio.
- **`worsen` applies to the counted rules** (non_finite, dead_channels, value_fc1_zero_velocity): the count rose above its peak while active, at most one line per rule per check-interval bucket. Its detail is `was=<peak>` plus the current `sites=…` (OD-22).
- **The `[HEALTH] check` line's `evaluations=`** is `live + checkpoint` committed since the previous check (D3's example had `evaluations=20 live=20 checkpoint=1`, which did not add up; the example is corrected). `stale=` counts both observation-level rejections (generation) and rule-level ones (older than a rule's newest applied step). It also carries `gradMaxRatio=` (rule 9) and, while `dead_channels` is active, `dead_channels_sites=` (OD-22).
- **The D6 anchor uses the observation's trainer step** (the digest's), which equals the stamp's last recorded step on every path that stamps at export.
- **The check-interval bucket starts at the process's first live window** (`(first record's step − 1) / interval`), so a resumed run writes its first check at the first multiple of the interval it reaches, not at once.
- **A disabled config drains the window** (it must not grow to the cap) and keeps the spike reference current, but judges, counts and logs nothing; `segmentSummary().evaluations` stays 0.
- **Live summaries run off the cooperative pool.** `LayerHealthLog.live` now hops to a GCD queue for the summary (as the checkpoint pass already did): a SiLU / GELU site's parked count is a numerical integral per channel.
- **Offline replay specifics.** Lines are tokenized on spaces outside parentheses and brackets (`worst=value.bn(dead 12 off 2 on 0)` is one field); a token without `=` is malformed except a parenthesized annotation (`(no relu/leaky_relu BN sites)`); `[RUN]` / `[APP]` lines are read only for `build=`. The monitor is created lazily per evaluator with the run's value-FC1 activation from the pre-scan (the `value.fc1.weight <act>` velocity row of a checkpoint table; a run that only has `[LAYER-HEALTH] value-fc1` lines is relu by D6; otherwise `.unknownFromLog`, rule 3 no data). A live line whose step goes backward (a GUI promotion) is replayed as an announced rewind. A checkpoint headline whose context is not `replay-…` / `vsuci-…` (GUI `session-…`) has its value-FC1 reading dropped (D4: no trained-step count). A live line without the parked fields gives relu / leaky_relu-only dead counts (`coverage=relu_leaky_relu_only` in the event detail, and a header note); `reluSites=0/n` on such a line is no data, not "does not apply", because its SiLU / GELU sites were never checked.
- **Incident excerpts** were generated from the logs named in Evidence (each file's header records the source's SHA-256 and line count at extraction, 2026-10-06, and the selection rule): C both segments whole; A and R7 / R8 through trainer step 5,000; B through 1,500; A and B then every checkpoint block up to the Evidence cutoff (32,750 / 32,700); B-silu through 22,000. `[BATCH-STATS]` lines are left out everywhere. A and B had stopped growing by extraction (their logs now run past the cutoff; the excerpts stop at it).
- **`bn_liveness.py`** mirrors the Swift model and gained GELU (exact erf), with its self-test extended (Monte Carlo and a dense-integral check for GELU, finite-difference check of `gelu'`); `--selftest` passes. `BatchNormPassThroughReference.json` (120 cases) was produced from its functions; `TrainingHealthBSiluBatchNorm.json` holds B-silu's γ / β at trainer steps 20,000 / 21,000 / 22,000, read once and read-only from `20261005-lrBsilu-cyc1-replay-step{20000,21000,22000}.safetensors` (their `content_sha256` and whole-file SHA-256 recorded), with `bn_liveness.analyze`'s per-site results as the expectations. No test reads `~/Library`.
- **B-silu's SiLU tower at those steps (Swift, equal to `bn_liveness.py`)**: no SiLU site parked at 20,000, 21,000 or 22,000; one mostly-off channel in `blocks.0.bn1` at 22,000; the lowest SiLU pass-through fell from 0.0775 (`blocks.2.bn1`, 20,000) to 0.0256 (`blocks.1.bn2`, 21,000) and 0.0215 (`blocks.0.bn1`, 22,000). The parked channels are in the leaky heads: `policy.pre_bn` 0 → 20 (7 mostly off) → 21 (8), `value.bn` 0 → 1 (4) → 1 (4).
- **Prototype numbers for the 2026-10-06 rules** (sparse rows, Evidence cutoff): rule 9's largest healthy ratio is 1.246 (A, 14,750), 1.404 (B, 29,950), 1.291 (R7), 1.295 (R8); it raises on B-silu at 20,600 (×7.15) and C at 300 (×13.4), both clearing at the next row. Rule 4's arm (ii) raises on B-silu at 20,700 and on none of A, B, R7, R8.
- **Not in P1** (left for P2 / P3 as the plan says): the parameters and `TrainingHealthConfig(_ snapshot:)`; any path wiring (record, evaluate, the 50-step tick, the shared live read of the cadence plan's OD-15, stop, `results.json`); `--replay-health-log` (the pure replay it calls is here); `ChessTrainer.readTrainableVelocity(named:)` and the dedicated value-FC1 read (its schedule `valueFC1ReadDue` and its line renderer are here); the GUI park flag (`requestPark`), alarm list, Health tab and sound.
- **OD-10's size bound with rule 9.** The ninth id adds one entry (`gradient_spike`, 130 B at the `Int` maxima): the `raised` array at all nine ids is 1,199 B and the whole `{"evaluations":…,"raised":…}` object 1,244 B in compact JSON, so the per-segment growth stays at most ≈ 1.2 KB (1,244 B), never with the number of events.

## Review fixes (P1, 2026-10-06)

An independent review of the P1 merge (`4cc60ab5`) found four defects and ten smaller points; each defect has a regression test that failed before its fix.

- **M1 — offline overall arm.** A log line written before the parked counts counts relu / leaky_relu channels only; dividing those by the same relu-only `ch=` overstated rule 2's overall fraction on a mixed tower (B-silu: 20/144, a false critical; the app would say 20/1040, a warning). The overall arm's denominator is now always every channel an activation consumes: in-app the summary's pass-through channels, offline the `parkedCh=` field or, for older lines, the run's checkpoint table (rows whose `act` is not `-`, read by the pre-scan, which now also refuses two activations for one site in one run). With no table in the run the overall arm has no data and the value reads `dead=<n>/--`. `LayerHealthDigest.DeadChannels` names its counts `modeledSiteCount` / `modeledChannelCount` / `parkedChannelCount` (minor 7), so they no longer share names with `LayerHealthSummary`'s relu-only `classifiedChannelCount` / `deadChannelCount`. `testBSiluDeadChannelsIsAWarningNotACritical` pins B-silu at a warning at 20,650; `testBSiluIllegalMassClearsAt21400` pins its rule-4 clear.
- **M2 — `valueFC1ReadDue(trainerStep:config:)`.** A disabled config commits nothing, so the D6 anchor never moved and every step past 1,000 was due (2,001 reads in 3,000 steps). The due test now takes the config and is never due while alarms are disabled.
- **M3 — `trained=` on the value-fc1 line** (format above; the offline replay uses it, offset like a step row's `step=` for logs passed together). The replay header no longer claims rule 3 for a log without step rows; it names the value-fc1 lines (and CLI checkpoints) as rule 3's only source and counts them.
- **M4 — clear sustain of rules 2 and 8** is two evaluations spanning ≥ 1 trainer step (`TrainingHealthThresholds.layerHealthRuleClearSustain`). A save step's live and checkpoint evaluations read the same weights, so "2 evaluations" alone could clear on one recovered state and flap. Span rather than "count only steps above the streak's last": it is the measure every other sustain uses, and with non-decreasing steps the two are equivalent.
- **Minors.** (1) A pending `dead_channels` clear's detail is `sites=none` (current state, not the last affected sites, so value and sites agree), and the check line prefixes every detail field (`dead_channels_sites=`, `dead_channels_coverage=`). (2) An announced rewind logs any unannounced rewind still pending before its own line. (3) A disabled config logs and counts nothing — failed reads, stale stamps and detected rewinds included; a detected rewind's line waits for the first enabled evaluation or announced rewind. (4) `nodata=` counts a rule only on evaluations whose source is meant to carry it (`LayerHealthDigest.Tier.feeds`; the dedicated read is its own tier, `value-fc1`), so a healthy run's check line reads `nodata=none`. (5) No `?? 0`: the final check line names the newest recorded or applied step, `none` when there is neither, and a stale line without a step says `none`. (6) Stale comments corrected. (8) `NumericsAudit.run` runs `runStatic` and the layer-health summary on a GCD queue. (9) The monitor reads step-store scalars inside the lock (`SyncBox.read`) instead of copying the store and its pending buffer out. (10) The tests above, plus `testReluAndLeakyParkedEqualsDeadExactly` now checks LayerHealth's dead / mostly-off counts against an independent Φ(β/|γ|) classification instead of against the parked counts that share its comparisons. A synthetic near-parked SiLU case already exists among the 120 reference cases (β = −9.06 at |γ| = 1, X = 1.001 × Φ(−3); β = −16.64 at |γ| = 5, 0.992 ×), so none was added.

# Implementation notes (P2)

P2 landed on branch `worktree-agent-a13d06e3f01f924f5` (2026-10-06), on top of the step-line cadence plan's branch (`worktree-agent-a3b6ce12b50064967`, merged in first; the wiring is cited against that code). Decisions made while implementing:

- **Files.** `CLI/CliTrainingHealth.swift` (the per-run wiring both command-line paths share: record → stamp → step-line block → live evaluation → save block → value-FC1 read, the stop latch, `results.json` events, stderr routing), `Training/TrainingHealthReads.swift` (the live-read adapter and the dedicated value-FC1 read, shared with the GUI in P3, plus `TrainingHealthValueFC1ReadRetry`), `App/TrainingHealthReplayCLI.swift` (`--replay-health-log`), `ChessTrainer.readTrainableVelocity(named:)`. Tests: `TrainingHealthParameterTests`, `CliTrainingRecorderAlarmTests`, `ValueFC1VelocityReadTests`, `TrainingHealthReplayCLITests`, `CorpusReplayHealthStopTests` (an in-process V-3 / V-4 on the real replay loop).
- **Action parameters are stored as `TrainingHealthAction` on the singleton** (as `randomSeedMode` is), Int-coded at the persistence boundary through `TrainingHealthAction(persistedRawValue:)` (traps on a value outside the declared range, which `testActionRangeMatchesTheEnumCases` pins). The enum is therefore `public`. One rule → key mapping per side: `TrainingParametersSnapshot.trainingHealthAction(for:)` and `TrainingParameters.trainingHealthActionKeyPath(for:)`.
- **Session fields store action names** (`"stop_on_critical"`, like `arenaPromotionCriterion`'s log token), so a session file stays readable. An unknown name is one `invalidSavedSettings` finding for the whole set (`training_health_actions`); accepting it keeps the current actions with one `[RESUME-DIFF]` line per key. Both session writers (GUI `buildCurrentSessionState`, train-vs-UCI `sessionState`) go through `SessionCheckpointState.withTrainingHealthSettings`.
- **Who decides a stop is now explicit** (`TrainingHealthStopDecision`): `.byEvaluator` for the command-line paths and the offline replay (the existing initializers keep that meaning, so no P1 test changed), `.byCaller` for the GUI (P3): its evaluator never writes a `stop` line or requests a stop, and the main actor decides from the actions in force when each evaluation arrives (R2) and writes the `stop` line when it acts. Without this, an action changed between a detached checkpoint pass and its delivery would leave a `stop` line with no stop, or a stop with no line.
- **The evaluation's own live read is not logged** unless it failed (the failure's reason is then logged). The logged live readout rides the step line; logging the 50-step evaluation reads would undo the cadence plan's log thinning. Where the step line falls on the evaluation step its read is shared (cadence plan OD-15).
- **Alarms disabled on the CLI** records nothing (no evaluation would drain the window) and creates no evaluation; `segmentSummary().evaluations` stays 0.
- **The stamp for a live evaluation is taken right after `recordStep`, before the step-line block,** so it precedes whichever live read the evaluation uses.
- **Final save.** The explicit live evaluation before it runs only when a step was recorded since the last live evaluation and that evaluation was not at the current trainer step. Final reasons, first match: `capture-failed`, `abort`, `health-stop`, `final` (replay); `abort`, `health-stop`, `final` (train-vs-UCI). `results.json`'s termination reason follows the same order (a Ctrl-C during a health-stopped run is still `manual_stop`).
- **A failed dedicated value-FC1 read** logs `[LAYER-HEALTH] value-fc1 read failed at trainerStep=N: …; next attempt after 1000 trainer steps` and waits one interval (`TrainingHealthValueFC1ReadRetry`): the deadline moves only with a committed observation, so without the wait a persistently failing read would be retried on every step.
- **`--replay-health-log`** runs under the declared defaults (never the user's saved settings) with `--learning-grace-steps` / `--lr-warmup-steps` overriding; it prints the header lines prefixed `# `, every monitor line, then a per-rule summary table (raises, first raise, highest severity, clears, severity active at the end). Pre-flight right after the defaults flags. Exit 0 or 2.
- **Registry count.** `TrainingParametersTests.test_registry_size` 86 → 98 (the cadence plan's +1 landed first, OD-6 / OD-20). The only existing-test edit in P2.
- **Parameter tests use the `GuiSessionResumeTests` pattern** (snapshot `TrainingParameters.shared`, `suppressPersistence`, restore in tear-down) rather than `makeTemporaryDefaultsSuite()`: the singleton reads and writes the standard domain only, and with persistence suppressed nothing is written anywhere.
- **`ValueFC1VelocityReadTests`' isolation test** trains on identically seeded real-data replay buffers (the synthetic `trainStep(batchSize:)` draws its batch from the system generator, so two trainers never agree) with a third trainer as the GPU-determinism control: bit-identical when the two plain trainers are; otherwise the reading trainer may differ from a plain one by no more than the plain ones differ. The dropout Philox state must match exactly either way.

### V-1 (P2), read-only, built binary of this branch

`--replay-health-log <log(s)> --learning-grace-steps 1000 --lr-warmup-steps 1000` on the full logs (not the excerpts), 2026-10-06:

| Run | Raise / escalate / clear events |
|---|---|
| A (`dcm_log_20261005-013220-2.txt`, to 36,000) | none |
| R7, R8 | none |
| B (`dcm_log_20261005-013235.txt`) | `dead_channels` critical 300 (value.bn 5/16; worsen to 6 at 1,300), `policy_offset_drift` warning 900, `value_fc1_zero_velocity` warning 1,000 (10/128; worsen 11 at 5,000, 12 at 8,000, …). All active at the end. Matches Evidence. |
| C, one process (both logs) | `dead_channels` critical 50 (value.bn 12/16), `policy_offset_drift` 200 → clear 400, `bn_running_variance_runaway` 200 (1,415.5), `loss_spike` 300 (×6.69) → clear 500, `gradient_spike` 300 (×13.43) → clear 350, `illegal_mass` critical 350 (0.9458 after 0.236), `gradient_collapse` critical 400 → clear 913 → raise 1,713, `value_fc1_zero_velocity` critical 1,513 (128/128). Matches Evidence. |
| C segment 1 alone | `dead_channels` critical 514 (339/1,040), `bn_running_variance_runaway` 514, `value_fc1_zero_velocity` critical 1,513, `gradient_collapse` critical 1,713, `illegal_mass` critical 2,063 (not learned). Matches Evidence. |
| B-silu (`dcm_log_20261005-234437.txt`, live; read at 21:27) | `gradient_spike` warning **20,600** (×7.15: 2.566 vs 0.359) → clear 20,650; `illegal_mass` critical **20,700** (arm ii: 0.7817 vs min 0.0020) → clear 21,400; `dead_channels` **warning** 20,650 (20/1,040, `policy.pre_bn` 19/128 — below both critical arms). Also, not in the Evidence table and reported here: `policy_offset_drift` warning 900 → clear 20,700 → raise 21,100 (B-silu shares B's LR cycle; B raises at 900 too), `bn_running_variance_runaway` warning 20,600 (1,031.9 at `blocks.2.bn1`), `loss_spike` warning 20,650 (×2.20) → clear 21,150. |

# Implementation notes (P3)

P3 landed on the same branch (2026-10-06). Decisions made while implementing:

- **The GUI's live evaluation runs in the trainer worker, not the `[STATS]` ticker.** OD-21 puts it on overall trainer-step multiples of 50 "from the trainer worker's step count"; the ticker polls on its own time-based cadence (the cadence plan's D8) and cannot land on exact multiples. `GuiTrainingHealthWorker.afterStep` (in `App/SessionController+TrainingHealth.swift`) records the step, and on a multiple of 50 stamps, resolves the config from `TrainingParameters.shared` in one `MainActor.run` hop (Part K item 8), takes its own live read (its line is logged only when it failed — the logged readout stays on the ticker), evaluates and delivers; then the value-FC1 read when due (with the config of the newest live evaluation, so there is no main-actor hop per step). Awaited inline, so the next SGD step waits; the cost is one ≈ 1 ms read per 50 steps plus the summary. T8's "stats task" wording is superseded by this.
- **Delivery is fire-and-forget** (`Task { @MainActor in … }`): the evaluating task never waits for the main actor. `deliverTrainingHealth(from:)` checks `monitor === trainingHealthMonitor` (a hop from an earlier start is logged and dropped), refreshes the list from the monitor's current set, and — only while the run is live, not already suspended and alarms are enabled — runs `TrainingHealthStopPolicy.firstQualifying` with the actions read from the singleton at that moment.
- **Stops are decided only on the main actor** (`TrainingHealthStopDecision.byCaller`, P2): the GUI monitor's evaluator writes no `stop` line; the main actor writes it when it acts, and appends it to `results.json`'s `alarms` (GUI `--train`).
- **`trainingSuspension: TrainingSuspension?`** (`Training/TrainingSuspension.swift`) replaces the Bool; the gate table is on the enum (`skipsPeriodicAutosave`, `skipsHeartbeatAlarmEvaluation`, `arenaSkipLabel`, `refusalReason`). A divergence while a health suspension is in force replaces it (the parked worker cannot diverge, so this is a race at most); a divergence suspension is never replaced by a health one.
- **Parking.** The park flag lives on the monitor (`requestPark()` / `parkRequested`), so a later start's worker never sees an earlier run's request. The parked loop (`GuiTrainingHealthWorker.park(at:)`) acknowledges pause requests exactly as the loop top does and ends only on cancellation.
- **Promote Trainee Now** refuses with the suspension's reason for both cases; the menu item is disabled through `AppCommandHub.trainingSuspended`, mirrored through `MenuHubSignature` like `isArenaRunning`.
- **Checkpoint passes** take a `GuiTrainingHealthCheckpoint` (monitor, stamp, config, delivery) made by `makeTrainingHealthCheckpoint()` right after the save's trainer export under its training pause, and for the promotion save right after `noteTrainerClockRewind` under both pauses.
- **`LastSessionPointer`**: unchanged, as decided (OD-5): a periodic autosave during a health suspension moves it like any successful save.
- **Sound.** One `reconcileSound()` decides the loop after every change (raise, clear, dismiss, silence, health refresh) from `shouldSound` = not silenced and (banner alarm or a critical health alarm). `clear()` / `dismiss()` still reset `silenced`, so the loop restarts for a remaining critical health alarm.
- **Alarm list** (`App/UpperContentView/TrainingHealthAlarmList.swift`, with its header and row as child views in the same file): always mounted under the banner, hidden with opacity 0 and a zero frame when empty. A row's "stops run" / "log" is computed from the rule's **current** action at the alarm's severity (R2), not the action recorded at its raise. Display names and one-line meanings live in one extension (`TrainingHealthRule+Display.swift`) shared with the Health tab.
- **Health tab** (`App/UpperContentView/TrainingHealthTab.swift`): one row per rule with a menu picker (Stop on critical is disabled for the four warning-only rules), then enable, check interval and grace as monospaced fields with their declared ranges. The popover's private `PopoverRow` / steppers are file-private to the popover file, so the tab has its own small field view rather than reaching into them.
- **HPARAM_RECORDING_PLAN P4 coordination (OD-10).** This plan's side is the typed value `TrainingHealthSegmentSummary` (`{evaluations, raised:[{rule, first_trainer_step, highest_severity, raise_count}]}`, `Codable`, the exact recorded shape, pinned by `TrainingHealthSegmentSummaryMergeTests.testEncodesInTheRecordedShape`) and its pure, order-independent `merging(_:)` (counts summed, earliest first step, highest severity; `.empty` is the identity). How HPARAM P4 consumes it:
  - **CLI paths** (one monitor per process = one segment): at each save's record, `trainingHealth.monitor.segmentSummary()` (`CliTrainingHealth.monitor`), read in the same sequential turn as the save's export.
  - **GUI**: the tracker keeps a stored merged summary for the segment. At every `startRealTraining` it merges the outgoing `trainingHealthMonitor?.segmentSummary()` into the stored one **before** `beginTrainingHealthRun` replaces the monitor (a segment-ending start resets the stored value to `.empty` instead); `takeConfigurationCut()` records `stored.merging(trainingHealthMonitor.segmentSummary())` without storing it, so nothing is counted twice. All on the main actor; `segmentSummary()` takes only the monitor's evaluation lock.
  - Alarms disabled: `evaluations` stays 0 (P1), so `{"evaluations":0,"raised":[]}` honestly says "never evaluated".
- **Existing-test edits in P3:** none.
- **Not done (needs the GUI, which this work must not launch):** V-6 and V-7. Covered instead by `TrainingAlarmControllerHealthTests`, `PromoteTrainerNowSuspensionTests`, `TrainingHealthGuiDeliveryTests` (stale hop, stop from the action in force at delivery in both directions, disabled alarms never stop, the parked worker acknowledging a pause and ending on cancellation) and the P2 runner tests.

# Implementation notes (P4)

- `documentation/training-health-alarms.md` (rules, cadence, actions per path, parameters, formats, the app's surfaces, offline replay); CHANGELOG entry; CLAUDE.md: `[HEALTH]` / `[ALARM] health` / `[LAYER-HEALTH] value-fc1` in the tag list, a "Training health alarms" section under observability, `vsuci-health-stop` in "Saved model state" (OD-11); `documentation/UCI.md` "Output: session folders", `scripts/sessions_summary.py` and `documentation/dashboards/vsuci.py` docstrings gain `vsuci-health-stop`. `--help` was done in P2.

### V-1 long runs (reported, not judged; OD-17)

`--replay-health-log <log> --segment-step-as-trainer-step`, declared defaults, read only:

| Log | `loss_spike` | `gradient_spike` (rule 9, new since the Long-runs table) | other rules |
|---|---|---|---|
| `dcm_log_20260702-201756.txt` (v5 line, 5,371 rows) | 21 raises, first 26,600 — **identical to the Long-runs table** | **247 raises** (first 1,850), each cleared at the next row; ratio min 5.04, median 8.23, max 46.59, 87 at ≥ 10× | none (`pLogitMean` absent, so `policy_offset_drift` has no data) |
| `dcm_log_20260727-094049.txt` (qeu8 line, 27,953 rows) | 2 raises (125,350; 330,550) — **identical** | **173 raises** (first 700); ratio min 5.01, median 5.74, max 9.63 | none |

**For the owner (not changed here, OD-14 / OD-17):** rule 9's 5× threshold was set against the post-fix runs (largest healthy ratio 1.40). On these two older bf16-era lines (before `da15920`) single logged rows reach 5–47× their reference median often enough to keep rule 9 near-permanently flickering as a warning (raise, then clear at the next row). The default action is `log` and the rule is warning-only, so nothing stops; whether these are real gradient spikes of that era or the threshold needs a long post-fix run to judge is the same question OD-17 asked of rule 6.
