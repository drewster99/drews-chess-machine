# Training health alarms plan: every training path checks itself and says so in the log

Status (2026-10-05): **PLAN ONLY.** Nothing here is implemented.
- Independently reviewed against the code and the logs (2026-10-05). The review's corrections are folded in; the design changes it caused are marked in the text where they matter (worker parking during a GUI suspension, checkpoint ordering, ring sizing, the sparse offline-replay semantics, rule 8's statistic).
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
- Defines nine rules (a tenth, value loss above ln 3, was dropped by the owner, OD-7). Their thresholds were measured against three incident runs, four healthy replay runs, a long GUI self-play run, and a survey of every log since `[LAYER-HEALTH]` shipped.
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
| 1 | `TrainingHealthEvaluator` (pure value type): nine rules, two severities, hysteresis, sustain, gates, reminders | P1 |
| 2 | `TrainingHealthMonitor` (lock-protected class): per-step pending window, loss-reference history, window statistics, serialized evaluations, stale-checkpoint rejection, trainer-clock rewind with generations | P1 |
| 3 | `TrainingHealthLog`: the `[ALARM] health …` and `[HEALTH] …` line formats, shared by every path | P1 |
| 4 | `TrainingHealthLogReplay`: parses `[REPLAY]` / `[VS-UCI]` / `[LAYER-HEALTH]` lines into evaluator observations (used by the incident tests and `--replay-health-log`) | P1 |
| 5 | `LayerHealthLog.live(trainer:)` returns the summary as well as the lines; `liveLines` is removed and its four callers converted | P1 |
| 6 | 12 new `TrainingParameters` (enable, check interval, learning grace, one action per rule), full checklist | P2 |
| 7 | Corpus replay and train-vs-UCI: record every step, evaluate at every stats tick and every checkpoint pass, stop through the loop exit and final save | P2 |
| 8 | `results.json`: `alarms` array, `alarm_config` object, `termination_reason: "training_health_alarm"` | P2 |
| 9 | `--replay-health-log <log>…` CLI (no GUI, no GPU) | P2 |
| 10 | GUI: monitor wired to the trainer worker, the `[STATS]` ticker and the checkpoint passes. Promotion rewind handled. Alarm list view with its own Silence; one beep loop for both alarm sources. Health tab in the settings popover. Stop = training suspension with the worker parked (interactive) or `AutoTrainTermination` (`--train`) | P3 |
| 11 | Docs: `documentation/training-health-alarms.md`, `--help`, CLAUDE.md tag list (OD-11), CHANGELOG | P4 |
| 12 | Optional: move the GUI-only detectors' conditions into the shared evaluator (OD-9); lineage segment summary (OD-10, with HPARAM P4) | P5 |
| 13 | Rule 3's value-FC1 velocity check at most 1,000 trainer steps apart on every path (OD-15, D6): reused from a save's checkpoint pass where one covers it, otherwise one dedicated read of that tensor | P1 (pure scheduling), P2 (trainer read, train-vs-UCI), P3 (GUI) |

## Every check, its trigger and its interval

The owner's rule (OD-15): every check runs on a set interval or trigger. "Step" is the trainer step unless it says segment step.

| Check | GUI Play-and-Train / `--train` | `--replay-corpus` | `--train-vs-uci` |
|---|---|---|---|
| Record the step (rules 4–8's inputs) | every SGD step | every SGD step | every SGD step |
| Live evaluation: live `[LAYER-HEALTH]` read (BN state, ReZero α) + rules 1, 2, 4–9 | every 25 session steps for the first 500, then every 60 s `[STATS]` emit (a median of 94 steps in the Ejp0 run) | every step-line tick: segment step 1, every step-line interval (50 at today's settings), every autosave segment step (cadence plan) | same as replay |
| Checkpoint evaluation: full-tensor `[LAYER-HEALTH]` pass + rules 1, 2, 3, 9 | every session save: periodic (default 6 h), promotion, Promote Trainee Now, manual, SIGUSR2 | every rolling save: every 1,000 segment steps, plus the final save | every session save (periodic, time-based; final; abort), plus each enumerated checkpoint (every 1,000 segment steps) with `--enumerate-checkpoints` |
| Rule 3 value-FC1 velocity (OD-15, D6) | every 1,000 steps: a dedicated read whenever 1,000 steps have passed since the run started or since the last rule-3 observation (a save's pass also counts) | every 1,000 steps, from the autosave's checkpoint pass (every 1,000 segment steps); a dedicated read only when that save or its pass produced no observation (a first, non-fatal save failure, `CLI/CorpusReplayRunner.swift:1628-1633`, or a failed health pass) | every 1,000 steps: from the enumerated checkpoint's pass with `--enumerate-checkpoints`, otherwise a dedicated read on the same deadline as the GUI |
| `[HEALTH] check` line and `active` reminders | first live evaluation at or after each multiple of `training_health_check_interval_steps` (1,000) | same | same |
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

| trainerStep | lr | loss | pLoss | pIllM | gNorm | pLogitMean | vAbs − \|pW−pL\| | live dead / 1,040 | worst site |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 0.01 | 11.8138 | 8.6299 | 0.9938 | 31.172 | -- | -- | 0 | none |
| 50 | 0.5 | 6.4783 | 5.0650 | 0.5131 | 1.235 | -0.0173 | 0.005 | 12 | value.bn (12 dead / 16) |
| 100 | 1.0 | 5.7369 | 4.5472 | 0.3179 | 1.502 | -1.3832 | 0.068 | 14 | value.bn |
| 150 | 1.5 | 5.1343 | 3.9693 | 0.2925 | 0.608 | -4.5106 | 0.054 | 24 | value.bn |
| 200 | 2.0 | 5.1376 | 3.9890 | 0.2893 | 0.621 | -5.6759 | 0.019 | 25 | value.bn |
| 250 | 2.5 | 4.7847 | 3.4011 | 0.2360 | 0.899 | -6.8423 | 0.009 | 28 | value.bn |
| 300 | 3.0 | 36.3806 | 34.1114 | 0.9971 | 14.325 | -6.2725 | -0.003 | 322 | policy.pre_bn (92) |
| 350 | 3.5 | 8.7093 | 6.8958 | 0.9458 | 0.059 | -0.8083 | 0.003 | 340 | policy.pre_bn |
| 400 | 4.0 | 8.6401 | 6.8210 | 0.9476 | 0.018 | -0.7581 | -0.002 | 340 | policy.pre_bn |
| 500 | 5.0 | 8.6092 | 6.7816 | 0.9446 | 0.057 | -1.1248 | -0.003 | 339 | policy.pre_bn |

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
- The value head stopped telling positions apart.
  - `vAbs − |vMean|` was ≈ 0 from step 250 on. That quantity is 0 exactly when every position's `v = p_win − p_loss` in the batch has the same sign (a constant output is the extreme case); it does not by itself prove the output is constant.
  - The loss shows the rest: `vLoss` sat at ≈ 0.889, about the entropy of the batch label mix (W 0.4629 / D 0.0596 / L 0.4775 in the first segment-1 `[BATCH-STATS]` line gives 0.878). That is what a head scores when its output ignores the position. It is **below ln 3**, so a "value loss above ln 3" rule would never have fired for C. (Such a rule was proposed at the owner's request and dropped by OD-7.)
- `nonFinite` was 0 on every live and checkpoint pass. The trainer's existing non-finite-loss halt (`Training/ChessTrainer.swift:7088-7102`) never had anything to catch.
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
- `vAbs − |pW−pL|` dipped to 0.004 at step 250 (one logged sample), the step at which the value-BN channels died, and was ≥ 0.115 at every logged step from 500 on.
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
| `vAbs − |pW−pL|` min, ≥ 500 | 0.134 | 0.115 | 0.114 | 0.116 | 0.470 (`vAbs − |vMean|`) | C ≈ 0 from 250 |
| live dead channels, max | 0 | 6 | no data | no data | no data | C 351 |
| `valueFC1ZeroVel` max | 0/128 | 24/128 | no data | no data | no data | C 128/128 |

**Fresh GUI runs (GUI-F1, GUI-F2).** Values are the `[STATS]` line's rolling means, not per-step samples, so they bound the per-step medians the monitor will see only loosely:
- `pIllM`: F1 0.9961 at step 1, 0.6212 at 500, 0.2262 at 1,004, 0.2180 at 2,033, 0.1115 at 6,021, 0.0420 at 14,028; maximum 0.2189 from 2,000 on. F2 0.6464 at 500, 0.2211 at 2,041. So rule 4's not-learned form (≥ 0.5 past warmup + grace = 2,000) has a margin of ≈ 2.3× on fresh GUI nets.
- `gNorm` from 2,000 on: F1 1.154–1.375, F2 1.051.
- `vAbs − |vMean|` from 2,000 on: F1 minimum 0.0225, F2 0.0292 — **below** rule 8's 0.03. Rule 8 does not apply to them: their buffers were 83–96% draws (`comp=… D=0.825` at step 100 rising to 0.948 by 12,503 in F1), above the 0.5 draw gate. This is exactly the case the gate exists for; without it rule 8's not-learned form would fire on a healthy fresh GUI run. It also means rule 8 is inactive on draw-heavy GUI self-play.

**Survey of every log since `[LAYER-HEALTH]` shipped** (2026-10-02 → 2026-10-05; 60 logs carry live lines, 73 carry any `[LAYER-HEALTH]` line):
- **Live dead channels.** Maximum 0 in every log except B (6), C (351) and `dcm_log_20261003-001701.txt` (17, a 3-step run branched from the grafted v5 model `20261003-18-AkMs`: already 17 at its first live evaluation, after one training step, and unchanged through its step-3 checkpoint. That fixes the process's baseline; it does not by itself prove the channels came in with the model, nor that they are damage, rule 2 note).
- **`valueFC1ZeroVel`.** At most 1/128 in a healthy 128-unit head (`dcm_log_20261002-202430.txt`, 33,000 steps). Short 16-unit test runs (corpus-replay `replay-final` saves at 3–62 trainer steps, e.g. `dcm_log_20261003-135725.txt`) showed 0–3/16. The branch run `dcm_log_20261003-001701.txt` showed 49/128 at its `replay-final` save after 3 steps (with so few trained steps, mostly the velocity the branch did not continue; below, and rule 3's gate).
- **16/16 where nothing was trained.** In 15 logs from 2026-10-03 (test processes), specific GUI `session-manual` / `session-promote` saves show `valueFC1ZeroVel=16/16` with the headline's trainer step anywhere from 0 to **274** (`dcm_log_20261003-153145.txt:108` is the 274). Those saves come from the test harness that advances the trainer clock without calling `trainStep` (`DrewsChessMachineTests/GuiSaveHarness.swift:163-177`): session ID `unknown`, every BN β/|γ| exactly 0. No velocity had accumulated, so every unit read exactly zero. (The same logs hold other tests' output too, including `[STATS]` lines and, in `dcm_log_20261003-134853.txt` and `dcm_log_20261003-135725.txt`, corpus-replay training; the claim is about those saves, not the whole logs.) Rule 3's minimum-history gate therefore counts steps **this process trained**, never the trainer clock (R1–R10).
- **`rvMaxOverMedian`.** Below 80 everywhere except the two runs that loaded older models (231 and 332), and C (548,521).

Two conditions of the GUI path that the replay incidents do not show:
- Self-play buffers can be almost all draws: `comp=… D=0.964` in `dcm_log_20261003-150458.txt`, D ≈ 0.97 in `dcm_log_20261001-011107.txt`.
- A fresh GUI net learns legality more slowly per step: in `dcm_log_20261001-011107.txt`, `pIllM` was 0.996 at step 1 and still 0.50 at step 326.

Rules 4 and 8 are gated with both of these in mind.

### Prototype replay of the rules below over these logs

The rules in Part R were prototyped in a scratch Python script, and re-derived independently by a second script during review with the same results. Neither is committed: validation uses the real evaluator through `--replay-health-log` instead (Part V). Both scripts use the sparse offline semantics of D4: each logged row is one evaluation with a one-record window, and site channel counts come from the logs' checkpoint tables. The table was re-run after the owner's decisions (OD-1 dead-channel critical at 5% overall / 20% per site; OD-7 value-loss rule dropped): only B's `dead_channels` changed, from a warning to **critical at 300**; A, R7 and R8 still raise nothing, and C's events are unchanged. Results:

| Run | Event (rule, severity, trainer step) |
|---|---|
| A | none |
| R7, R8 | none (scalar rules only; these logs have no `[LAYER-HEALTH]`) |
| B | `dead_channels` **critical 300** (`value.bn` 5/16 = 31% ≥ 20%; owner OD-1 thresholds); worsened to 6 at **1,300**. `policy_offset_drift` warning **900** (−3.80). `value_fc1_zero_velocity` warning **1,000** (10/128, checkpoint). All three still active at the end. |
| C, one process | `dead_channels` **critical 50** (`value.bn` 12/16). `policy_offset_drift` warning 200 (cleared 400: −0.81 at 350 and −0.76 at 400 are the two clear evaluations). `bn_running_variance_runaway` warning 200 (1,415.5). `value_head_one_sided` warning 250. `loss_spike` warning 300 (36.38 against a reference of 5.44; cleared 500). `illegal_mass` **critical 350** (regression: 0.9458 after a minimum of 0.236). `gradient_collapse` **critical 400** (0.018); cleared 913 while the LR was near 10, re-raised 1,713. `value_fc1_zero_velocity` **critical 1,513** (128/128). |
| C segment 1 as its own process | `dead_channels` critical 514 (339/1,040). `bn_running_variance_runaway` warning 514. `value_fc1_zero_velocity` critical 1,513. `gradient_collapse` critical 1,713. `illegal_mass` critical 2,063 ("not learned": the resumed process never saw the pre-divergence minimum). |

C's `value_head_one_sided` at 250 comes from its regression form (rule 8). In segment 1, no `[REPLAY]` row carries the diagnostic fields, so the offline replay cannot evaluate that rule there. The in-app monitor sees every tenth step.

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
  - Lean fields, valid on every step, that a rule reads: `loss` (rule 6), `illegalMassPenalty` (rule 4), `gradGlobalNorm` (rule 5) (`Training/ChessTrainer.swift:106,127,144`); `sampledBatchDrawFraction` (`:297`, valid every step; rule 8's draw gate), plus `totalMs` for the `train_ms` field (D2). `policyLoss` and `valueLoss` are not recorded: no rule reads them since OD-7 dropped the value-loss rule.
  - Diagnostic fields, only when `hasDiagnostics` (`:308`): `policyLogitMean` (`:318`), `valueAbsMean` (`:158`), `valueMean` (`:151`).
  - `illegalMassPenalty` is in the trainer's lean readback targets (`Training/ChessTrainer.swift:6747-6750`), so it is measured on every step. The log agrees: every segment-1 `[REPLAY]` line of C carries `pIllM` while its diagnostic fields read `--`. (The comment above `dg` at `CLI/CorpusReplayRunner.swift:1829-1833` still lists illegal mass in the diagnostic bundle; it is stale.)
- **Evaluation.** A call to the evaluator with the window and, where available, a layer-health digest.
  - CLI paths: at every step-line tick, right after the live `[LAYER-HEALTH]` read, and after every checkpoint pass. The tick is whatever `TrainingStepLogCadence.isStepLineTick(trainerStep:segmentStep:batchStatsInterval:)` from `documentation/plans-active/STATS_LINE_RESUME_CADENCE_FIX_PLAN.md` returns; that plan lands first (Part P) and owns the contract (today: segment step 1, every trainer step on its diagnostics-aligned step-line interval, and every autosave segment step). This plan relies on only two properties of it: it ticks at segment step 1, and it ticks at every autosave step (where the checkpoint pass runs). Before that fix the tick is `step == 1 || step % 50 == 0` on the segment step (`CLI/CorpusReplayRunner.swift:1904`, `CLI/TrainVsUciRunner.swift:764`; pre-cadence-fix line numbers). The monitor does not depend on which: it records every step, and a tick whose window holds no diagnostic step (an autosave tick off the diagnostics cadence, or a short window) leaves the diagnostic rules holding their state (no data). After a resume at trainer step 513 the first window is step 514 alone (segment step 1) and the next ends at the first step-line tick; sustain spans are measured in trainer steps, so short windows are harmless.
  - GUI: after every live read (every 25 steps for the first 500, `App/SessionController+Training.swift:1435,2119-2124`; then every 60 s `[STATS]` emit, `:1419,2177-2179`), and after every checkpoint pass.
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
- **Sustain `n` / span `s`.** The condition must hold on `n` consecutive evaluations that have data, spanning at least `s` trainer steps. Measuring both makes the rule independent of cadence: 25 steps (GUI bootstrap), 50 (CLI), and in GUI steady state whatever one 60 s `[STATS]` interval holds (a median of 94 steps in the Ejp0 run at ≈ 0.65 s/step; more at smaller batches, fewer at larger).
- **Learning gate.** `trainerStep ≥ lr_warmup_steps + training_health_learning_grace_steps` (defaults 1,000 + 1,000). It applies only to the "has not learned yet" forms of rules 4 and 8. Every damage rule runs from the first evaluation.
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
    - resets every rule's pending sustain progress (raise and clear), the regression forms' running minimum / maximum (rules 4 and 8), and every rule's newest applied trainer step (D2's freshness check; otherwise every post-rewind observation would look stale), because they described weights that no longer exist;
    - keeps active alarms, so they clear only by recovery;
    - logs `[HEALTH] trainer clock rewound <a> -> <b>; generation <g>; windows reset`.

## R1–R9. The rule table

Defaults for the thresholds are **proposed** (OD-1). `Healthy` is the most extreme value seen in a healthy run (the tables above). `Fires on` is the prototype result.

| # | id | Data | Raise | Clear | Sustain | Healthy | Fires on |
|---|---|---|---|---|---|---|---|
| 1 | `non_finite` | live + checkpoint digest `nonFiniteValueCount`. Any non-finite diagnostic field in `W` (`pLogitMean`, `valueAbsMean`, `valueMean`; the trainer's own halt checks only `valueMean` of these) | critical: count > 0 | never auto-clears (non-finite weights do not heal) | none: raises on first sight | 0 in all 73 logs | none of the incidents (C stayed finite) |
| 2 | `dead_channels` | digest: dead count (β/\|γ\| < −3, `Training/LayerHealth.swift:73`) and per-site counts | warning, by OD-16: **absolute form** dead > 0, or **new-damage form** any site's dead count above its baseline (below). Critical (both forms): dead ÷ classified ≥ 0.05, or any site's dead ÷ channels ≥ 0.2 (OD-1, decided) | only when **no** raise condition of any level holds for 2 evaluations: absolute form, dead = 0 (which also ends both critical arms); new-damage form, every site at or below its baseline **and** neither critical arm holding. So a critically damaged baseline (C's segment 1 alone) never clears by staying unchanged | none (dead is near-permanent) | 0 in every healthy run surveyed | B critical 300 (`value.bn` 5/16 = 31%). C critical 50 |
| 3 | `value_fc1_zero_velocity` | `valueFC1` zero/units, from a checkpoint digest or the dedicated value-FC1 read (D6), at least every 1,000 trainer steps on every path. Evaluated only once **steps trained by this process** (R0) ≥ 200 when the state was read (otherwise no data) | warning: ≥ 0.05. Critical: ≥ 0.5 | < 0.025 at one checkpoint | none | ≤ 1/128 (0.8%) | B warning 1,000 (10/128). C critical 1,513 (128/128) |
| 4 | `illegal_mass` | `median_W(illegalMassPenalty)` | critical, **regression form**: the run's own running minimum of window medians has been < 0.5, and now ≥ 0.8. Critical, **not-learned form**: past the learning gate and ≥ 0.5 | < 0.3 for 2 evaluations | 2 evaluations, span ≥ 50 | replay ≤ 0.0649 from 2,000; fresh GUI ≤ 0.2189 from 2,000 (rolling mean, GUI-F1) | C critical 350 (regression). C segment 1: 2,063 (not learned) |
| 5 | `gradient_collapse` | `median_W(gradGlobalNorm)` | critical: < 0.1 | ≥ 0.2 for 2 evaluations | 2 evaluations, span ≥ 50 | min 0.235 (B); GUI ≥ 2.648 | C critical 400 (cleared 913 near LR 10, re-raised 1,713) |
| 6 | `loss_spike` | `max_W(loss)` and `median_W(loss)` against `ref` = median of the loss records in the 1,000 trainer steps before `W`. `ref` has data only when those records span ≥ 200 trainer steps (first to last) and number ≥ 5 — a span, not a record count, so the in-app per-step history and the sparse offline rows (D4) use one definition | warning: `median_W ≥ 1.5 × ref` or `max_W ≥ 3 × ref` | `median_W < 1.2 × ref` | none | logged ratio ≤ 1.029 | C warning 300 (×6.69) |
| 7 | `policy_offset_drift` | `median_W(\|policyLogitMean\|)` (diagnostic) | warning: ≥ 3.0 | < 2.0 for 2 evaluations | 2 evaluations, span ≥ 50 | ≤ 0.760 | B warning 900. C warning 200 |
| 8 | `value_head_one_sided` | `median_W(valueAbsMean − \|valueMean\|)` (diagnostic): 0 exactly when every position's `v` in the batch has the same sign. Applies only when `median_W(sampledBatchDrawFraction) ≤ 0.5` | warning, **regression form**: the run's running maximum has been ≥ 0.06, and now < 0.03. Warning, **not-learned form**: past the learning gate and < 0.03 | ≥ 0.06 for 2 evaluations | 2 evaluations, span ≥ 50 | ≥ 0.114 from 500; GUI ≥ 0.470 | C warning 250. B's single 0.004 at 250 does not sustain |
| 9 | `bn_running_variance_runaway` | digest: largest `runningVarianceMaxOverMedian` | warning: ≥ 1,000 (OD-1, decided) | < 300 for 2 evaluations | none | < 80 in healthy runs (231, 332 in runs from older models) | C warning 200 (1,415.5) |

**Notes per rule (the measurements behind them):**

- **Rule 1.** The trainer already throws `nonFiniteLoss` on a non-finite loss, gradient norm, or (on diagnostic steps) `valueMean` or entropy (`Training/ChessTrainer.swift:7088-7102`). Every path treats that as fatal: the GUI suspends (`App/SessionController+Training.swift:1248-1268`), and the CLI paths propagate the error. This plan leaves that behavior alone. Rule 1 covers the tensors (BN state and ReZero α live; every tensor at checkpoints), which can go non-finite while the losses are still finite.
- **Rule 2.**
  - **Two warning forms; the choice is OD-16.**
    - *Absolute form:* warn on any dead channel, because no healthy run in the survey had one.
    - *New-damage form:* the first digest this process sees, live or checkpoint (whichever comes first; both carry every site in-app), fixes each site's dead count as its **baseline**. It is logged once, in full, as `[HEALTH] baseline dead_channels=<dead>/<classified> at trainerStep=<s> sites=<site>:<dead>/<ch>,…` (every classified site with a nonzero count; `sites=none` when all are 0), so every later comparison is auditable. The rule warns when any site's count rises above its baseline. The baseline is per process (offline: per evaluator, D4) and survives a trainer-clock rewind (the restored snapshot is from the same process).
    - Offline the baseline is only partly known, because a live line names only its worst site (D4). The offline replay compares the **total** dead count against the first observation's total (exact), and a site's count against its baseline only when that site's count is known at the first observation (the first live line's worst site, or every site when the first observation is a checkpoint table). A site first seen later has an unknown baseline: its per-site comparison is no data for the rest of the evaluator, never a backdated or zero baseline. The output header lists those sites.
    - Why both: a long-trained line may simply carry a few dead channels. `dcm_log_20261003-001701.txt` is a 3-step branch of the grafted v5 model `20261003-18-AkMs` and shows 17 of 1,808 dead, 1–4 per site across 8 sites, `value.bn` 2 of 16. Nothing in the logs says whether that is damage or the normal state of a long-trained net. Under the absolute form every resume of that line holds a permanent warning, with a reminder every 1,000 steps and a permanent row in the GUI list; under the new-damage form it warns only if the count grows.
    - The chosen form is a declared constant like the thresholds (OD-14), not a parameter.
    - The critical arms are absolute in both forms, so C (and C's segment 1 alone, 339 of 1,040 at its first evaluation) is critical either way, and B's critical alarm at 300 (`value.bn` 5 of 16, 31% ≥ 20%) is the same under both (its first evaluation had 0).
  - The `[ALARM] health raise` line names the worst site, so the damage's location is visible.
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
  - SE FC1 zero velocity is reported, not alarmed: SE bottlenecks with many weak units were measured in healthy runs (`Training/LayerHealth.swift:9-17`).
- **Rule 4.**
  - The regression form catches C's divergence at the first evaluation after it (the prototype fires at 350; the in-app per-step window would fire at the first evaluation whose window median is ≥ 0.8).
  - The not-learned form waits for the learning gate because a fresh GUI net is still at 0.50 at step 326. On the replay runs the slowest healthy value at 2,000 was 0.065.
  - It overlaps the GUI's legal-mass probe (`App/SessionController+Training.swift:2282-2423`, 60 s probe cadence, aborts only GUI `--train`), which uses an inference copy of the network. Both stay until OD-9.
- **Rule 5.**
  - The floor 0.1 is 2.35× below the lowest healthy logged `gNorm` (B at its LR peaks) and 26× below the GUI run's minimum.
  - It is an absolute number in loss-gradient units, so an architecture or loss-weight change can move it. It is a declared constant, revisited per OD-1.
  - C's clear at 913 is the honest result of a symmetric rule: with the LR near 10, a broken net still produced `gNorm` 0.2–0.6. The other critical rules (2, 3, 4) stayed active through that window.
- **Rule 6.**
  - The logged ratios are single steps against a median of ten logged steps. Per-step maxima over a 50-step window will be larger.
  - Offline, the reference holds at most 20 logged rows (one per 50 steps in the prior 1,000), and needs 5 spanning ≥ 200 steps. C's step-300 spike has exactly 6 (steps 1–250, median 5.44), so the offline replay reproduces it under the same definition the app uses.
  - The `max_W ≥ 3 × ref` arm is a guess at the per-step noise. Validation V-2 measures the per-step distribution on a healthy run before the default is confirmed (OD-1).
- **Rule 7.** In a model with an fp32 policy tail, `pLogitMean` is invisible to the softmax and to the loss. Drift means the loss path is not centered, or the head has a large shared bias riding on always-on channels (`Training/LayerHealth.swift:18-23`). It is a warning signal, never critical. A model trained before the head-numerics fix (`da15920`) can carry a large offset from its first step: the nT8Y-line benchmark `dcm_log_20261005-012743.txt` reads −14.26, and rule 7 warns on it (V-1).
- **Rule 8.**
  - `valueAbsMean − |valueMean|` = `mean|v| − |mean v|` (`Training/ChessTrainer.swift:151,158`, computed at `:3294-3303`) is 0 exactly when every position's `v = p_win − p_loss` in the batch has the same sign. A constant output is one such case; an output that varies but never changes sign (e.g. `[0.1, 0.9]`) is another. The two readbacks cannot tell those apart, so the rule is named for what it measures: a **one-sided** value head.
  - Why one-sided is usually a failure: `v` is from the side to move's perspective, so every decisive game contributes both W-labelled positions (the winner's moves) and L-labelled ones (the loser's), in roughly equal numbers. Batches sampled from many games therefore hold W and L labels in about equal shares (C's: W 0.46 / L 0.48). That balance comes from the game structure, not from the draw fraction; the ≤ 50% draw gate only removes batches where a near-constant output can be correct. A head that tells winning from losing positions on a balanced batch outputs both signs.
  - **False positives it can still raise** (it is a heuristic, so it is a warning, never critical, and `log` by default):
    - a head that discriminates but carries a large shared bias, so every `v` lands on one side (`[0.1, 0.9]` is the textbook case). Early in training this is common, which is one reason the not-learned form waits for the learning gate; later it is itself worth a look;
    - a batch whose decisive labels are unbalanced by chance or by a sampling constraint. The evaluator does not see the batch's W/L split (only `sampledBatchDrawFraction` is in `TrainStepTiming`), so it cannot rule this out; the 2-evaluation sustain over ≥ 50 steps makes a chance imbalance unlikely to persist.
  - Tests pin both: a varying-but-one-sided synthetic window is flagged (as designed), and a single one-sided window does not sustain (X1).
  - Corroboration in C comes from the loss, not this statistic: `vLoss` sat at the label-mix entropy (Evidence), which is what a head that ignores the position scores.
  - It is the cheapest live symptom of a dead value head. In C it fired about 1,250 steps before rule 3 could.
  - It applies only to batches with ≤ 50% draws, because a correct value head on an almost-all-draw self-play batch may legitimately output nearly the same thing everywhere.
  - A fresh head with the draw prior (`[0, ln 6, 0]`) outputs `p_win ≈ p_loss` everywhere, so the statistic starts near 0. That is why the not-learned form waits for the learning gate.
  - Offline precision: the `[REPLAY]` line prints `pW` and `pL` to 2 decimals and `vAbs` to 3 (`CLI/CorpusReplayRunner.swift:1917`), so the offline statistic `vAbs − |pW − pL|` is good to about ±0.01 against thresholds of 0.03 / 0.06. In-app values are exact.
- **Rule 9.** The threshold is the one OD-1 item the owner left open. Measured `rvMaxOverMedian`: A max 7.2 up to the Evidence cutoff (7.4 at trainer step 34,250, after the cutoff; the owner's note quotes 7.4); B max 78.6 at 1,400 and at most 36.3 after 5,000; C 461,870–548,521 on its live lines from trainer step 500 on (first above 1,000 at 200: 1,415.5); the two older-model runs 231 and 332. The healthy maximum was 78.6 (B). The runs from older models reached 231–332 without other symptoms. 1,000 is 3× the highest non-incident value.

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
    | Train ▸ Promote Trainee Now (`App/SessionController+ManualPromote.swift:35-52`; menu `App/DrewsChessMachineApp.swift:683-684`) | **not gated today** (Risks; gating it is part of OD-5) | yes: `promoteTrainerNow` refuses through `onRefuseMenuAction`, naming the rule, and the menu item is disabled while suspended — the arena's reason. Without this the parked worker would acknowledge the promotion's training pause, and a damaged trainer would become the champion |

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
    case valueHeadOneSided = "value_head_one_sided"
    case batchNormRunningVarianceRunaway = "bn_running_variance_runaway"
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
- `recordStep(_ timing: TrainStepTiming, trainerStep: Int)` appends one record (trainer step, the three lean floats, the batch draw fraction, the step's `totalMs`, and the diagnostic fields when `hasDiagnostics`; about 48 B) to the **pending window** and increments "steps trained by this process". It holds only this lock, for one append: one uncontended lock per SGD step. It never takes the evaluation lock.
- **Rewind safety net.** If the record's trainer step `r` is at or below the last recorded step `a`, `recordStep` performs the step-store half of R0's rewind right there, under the steps lock it already holds: generation + 1; pending window discarded; with restored clock `b = r − 1`, steps trained becomes `max(0, trained − (a − b))`, then the record is appended (+1). (Example: last 100, record 51 → restored 50 → 50 trained → 51 after the append.) It also stores `unannouncedRewind = (a, b)`. It never takes the evaluation lock.
  - The evaluation-side half (loss-reference history, pending sustain, regression extrema, per-rule freshness steps) belongs to the evaluation state, which keeps `appliedGeneration`. **Every** evaluation, live or checkpoint, starts by comparing it with the step store's generation (evaluation → steps, in order); if it is behind, it applies the evaluation-side reset, logs `[HEALTH] trainer clock rewind detected by recordStep (not announced): <a> -> <b>; generation <g>` from `unannouncedRewind`, and only then validates its own stamp. So a delayed checkpoint stamped before the rewind is rejected by the generation check whichever evaluation runs first, and no evaluation ever judges discarded weights.
  - In the GUI the announced path below always runs first (under the training pause, before any post-rewind step), so the safety net is for a future rewind path that forgets to announce.
- The pending window grows until a live evaluation drains it: 50 records on the CLI, one 60 s `[STATS]` interval in the GUI (a median of 94 in the Ejp0 run). A hard cap of 65,536 records (≈ 3 MB) guards against evaluations that never come; past it the oldest records are dropped and counted, and the next `[HEALTH] check` line reports `truncated=<n>`. Nothing is dropped silently.

**`evaluation: SyncBox<EvaluationState>` — serializes every evaluation end to end.** It holds the evaluator (with, per rule, the newest trainer step it has applied), the loss-reference history, the counters for the `[HEALTH] check` line, and the stop-request flag.
- `noteTrainerClockRewind(to:)` takes evaluation → steps and applies both halves of R0's rewind at once, with `a` = the last recorded step and `b` = `to` (generation + 1, pending window and loss-reference history discarded, steps trained reduced by the rewound span, pending sustain, regression extrema and per-rule newest applied steps reset). Observations stamped with the old generation are then rejected by the generation check, so resetting the freshness steps cannot let them in.
- `evaluateLive(stamp:layerHealth:digestTrainerStep:learningRate:momentum:config:log:)`, all under the evaluation lock (`learningRate` and `momentum` are the live effective values the caller already has; they go into the observation and onto the event lines, never into a condition):
  1. apply the evaluation-side half of an unannounced rewind, if the step store's generation is ahead of `appliedGeneration` (above);
  2. under one brief inner steps lock: if `stamp.generation` is not the step store's current generation, the observation describes weights that no longer exist — log `[HEALTH] stale live observation ignored: …` after the lock, count it, change nothing (the pending records stay for the next evaluation); otherwise, in the same lock section, do step 3. Validation and extraction are therefore one atomic step against `recordStep`;
  3. take out of the pending window exactly the records with trainer step ≤ the **boundary** and leave newer ones pending. The boundary is the live digest's own trainer step (`LayerHealthLiveState.completedTrainSteps`, read on the trainer queue with the tensors, `Training/ChessTrainer.swift:5830`), or the stamp's `lastRecordedTrainerStep` when the live read failed. So the window and the digest describe the same weights, and records the worker appends meanwhile belong to the next window;
  4. compute the window statistics and the loss reference. Normally that is sorting ≈ 50–100 window values and ≤ 1,024 reference values (expected well under a millisecond); at the 65,536-record cap it is larger (expected a few milliseconds). These are estimates; the measured figure is the `cost_ms` field below, checked in V-5. Either way it holds only the evaluation lock, so the trainer's `recordStep` never waits on it;
  5. run the evaluator transition **on a copy** of the evaluator (it is a value type, D1), applying each rule only if the observation's trainer step is at or above that rule's newest applied step (below). Nothing is logged, recorded or published yet;
  6. **commit**, under one brief inner steps lock: if the step store's generation still equals the stamp's, replace the evaluator with the copy, set the stop-request flag if the copy requested one (used by the CLI; the GUI decides on the main actor, R2), and append the window's `(trainerStep, loss)` pairs to the **loss-reference history**, a ring of 1,024 entries that therefore always covers the 1,000 trainer steps before the next window — kept apart from the window, so a long window can never push the reference out. If the generation moved (an unannounced rewind happened in `recordStep` while statistics were being computed), the copy and the window are discarded, the evaluation is counted stale and logged as such, and nothing else changes. While the commit holds the steps lock no rewind can start, and a rewind that already happened is seen, so an evaluation never commits against a generation it did not validate;
  7. only after a successful commit: render the event lines and pass each, with its event kind, to the caller's `log` sink before releasing the lock, so log order is evaluation order. The GUI sink is `SessionLogger.shared.log` (a non-blocking enqueue, `Logging/SessionLogger.swift:198-203`); the CLI sink is the runner's `emit` (stdout + session log) plus stderr for raise, escalate and stop lines (D3), the same pattern as `AutoTrainTermination.writeResults(log:)` (`App/AutoTrainTermination.swift:75-76`). On the CLI a slow stdout only delays the one task that both trains and evaluates, as every `emit` does today.
- `evaluateCheckpoint(stamp:layerHealth:digestTrainerStep:config:log:)` takes the same lock and no window. Its `config`: on the CLI, the run-start config every evaluation uses; in the GUI, resolved by the save from `TrainingParameters.shared` at the moment it takes the stamp (both save paths run on the main actor: `SessionController` is `@MainActor`, `App/SessionController.swift:26-28`) and carried to the detached pass with the stamp, so a checkpoint's **data** is judged under the settings in force when its state was exported (learning grace, warmup, enable). **Stop decisions are not**: R2 says a stop follows the current action, so in the GUI the stop decision is made on the main actor when the evaluation is delivered, from the current actions, for every delivered evaluation (R2, R3). A detached checkpoint pass evaluated under an older action therefore neither stops under an action the owner has since set to `log`, nor fails to stop under one the owner has since set to a stop action. On the CLI the actions never change during a run. It follows the same sequence without a window: apply an unannounced rewind (step 1); validate the stamp's generation under the steps lock (a stale one is ignored and logged, `[HEALTH] stale checkpoint observation ignored: …`); transition a copy; commit under the steps lock only if the generation is unchanged; then log.
- **Freshness across tiers.** Each rule remembers the newest trainer step whose observation it applied, from either tier. An observation older than that is not applied **to that rule** and is counted `stale=` on the `[HEALTH] check` line. Consequences: a slow, older checkpoint can never supply clears for `dead_channels` (or rules 1 and 9) after a newer live observation raised it; a checkpoint still updates rule 3, which only checkpoints feed; and a checkpoint at the same trainer step as the live evaluation that preceded it (the CLI order at an autosave) applies to both.
- **Self-measured cost.** `recordStep` and both evaluations time themselves with `ContinuousClock` (a clock read, not a random draw). `recordStep` adds its own time and the step's `totalMs` to counters in the step store (it holds only the steps lock); evaluations add their time to a counter in the evaluation state and, when writing a `[HEALTH] check` line, move the step-store totals over under evaluation → steps; the `[HEALTH] check` line reports `cost_ms=` since the previous check, next to `train_ms=` (the sum of `TrainStepTiming.totalMs` over the steps recorded in the same interval), so the monitor's whole cost — including the work outside `trainStep`'s `ms` — is visible as a ratio in every run (V-5). A final `[HEALTH] check … final=true` line is written when the run ends — on the CLI after the final save's checkpoint evaluation, so the last partial interval, including that last evaluation's cost, is never lost; in the GUI at Stop. A GUI checkpoint pass is detached and can finish after that flush; its evaluation is still logged and applied, and its cost appears on its own `[HEALTH] check … late=true` line, so the GUI guarantee is "nothing lost", not "all in the final line".

**Who calls it.** On the CLI, one task calls everything in sequence. In the GUI there are three callers — the trainer worker (`recordStep`), the stats task (`observationStamp`, then `evaluateLive`), and detached checkpoint passes (`evaluateCheckpoint`, `App/SessionController+Checkpoint.swift:622-643`) — and the evaluation lock serializes the last two.

**Publishing to the UI.** The evaluation's events are logged and recorded by the evaluating task. Every main-actor hop (after a live or a checkpoint evaluation) carries the monitor it used; the main actor ignores the hop unless `monitor === trainingHealthMonitor` (so a delayed hop from an earlier run changes nothing), and otherwise has the controller re-read that monitor's **current** active set (`activeAlarmsSnapshot()`, under the evaluation lock), so two hops arriving out of order cannot show an older state. `activeAlarmsSnapshot()` makes the main actor wait for the evaluation lock, which is held while an evaluation sorts its window and hands lines to the logger: normally well under a millisecond, at the 65,536-record cap a few milliseconds (estimates; `cost_ms` measures it). That is accepted; the main actor never waits on the steps lock's hot path.

**Stop decisions** belong to their monitor. In the GUI they are made on the main actor (R2) only for a hop whose monitor is the current `trainingHealthMonitor`; a hop from an earlier run's monitor is dropped and logged. The trainer worker polls the park flag of the monitor its own run created. On the CLI the evaluator's `stopRequest` sets the run-local `healthStop` directly.

No GPU, no random draws, no access to the trainer, buffer or optimizer (D6's value-FC1 read is made by the path through `ChessTrainer`; the monitor only decides when it is due and evaluates its result). Probe isolation holds by construction: the monitor's only inputs are values the paths already hold.

## D3. Rendering and recording

`Training/TrainingHealthLog.swift`, one renderer shared by every path (like `Training/LayerHealthLog.swift`). Fixed key=value formats:

```
[HEALTH] config enabled=true interval=1000 grace=1000 warmup=1000 momentum=0.85 path=replay actions=non_finite:log,dead_channels:log,…
[HEALTH] check trainerStep=2000 generation=0 evaluations=20 live=20 checkpoint=1 stale=0 truncated=0 cost_ms=3.1 train_ms=651400.0 lossMaxRatio=1.14 lossMedianRatio=1.02 nodata=value_head_one_sided:3 active=dead_channels:critical,policy_offset_drift:warning
[ALARM] health raise rule=dead_channels severity=critical trainerStep=300 value=dead=5/1040 worst=value.bn(5/16) threshold=site>=0.2 action=log lr=0.3 mom=0.85
[ALARM] health raise rule=dead_channels severity=warning trainerStep=… value=dead=1/1040 worst=… threshold=dead>0 action=log lr=… mom=…
[ALARM] health escalate rule=dead_channels severity=critical trainerStep=… value=… threshold=site>=0.2 action=… lr=… mom=…
[ALARM] health worsen rule=dead_channels severity=critical trainerStep=1300 value=dead=6/1040 was=5
[ALARM] health active rule=dead_channels severity=critical since=300 trainerStep=2000 value=dead=6/1040
[ALARM] health clear rule=loss_spike severity=warning since=300 trainerStep=500 value=median/ref=1.04
[ALARM] health stop rule=illegal_mass severity=critical trainerStep=350 action=stop_on_critical
[HEALTH] trainer clock rewound 5200 -> 4900; generation 1; windows reset
[HEALTH] stale checkpoint observation ignored: trainerStep=4800 generation=0 (current generation 1, newest applied 4900)
```

- The tag stays `[ALARM]`. Every health line is `[ALARM] health <kind> rule=…`, so `grep '\[ALARM\] health'` selects exactly these and nothing else.
- On the CLI paths, raise, escalate and stop lines also go to stderr, as the existing replay `[ALARM]` lines do (`CLI/CorpusReplayRunner.swift:897-900,1878-1881`).
- `[HEALTH] check` is written on the first live evaluation at or after each multiple of `training_health_check_interval_steps` (so at trainer step 1,000 on the CLI, and at the first `[STATS]` emit past it in the GUI). The `active` reminders come on the same cadence. A run with no alarms therefore still writes one line per interval: positive evidence that the checks ran.
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
  - `[BATCH-STATS]` JSON lines, for the batch draw fraction (`outcome_pct.D`, the most recent at or before each stats row) that rule 8's draw gate needs. The `[REPLAY]` line does not carry it; on these corpus runs it was about 0.066 (`dcm_log_20261005-090417.txt`, trainer step 10: 0.0659).
- **Pre-scan.** Before evaluating, it reads every checkpoint table in the logs passed together and builds `site → channel count` **per run**: a run starts at each `[RUN]` line (lines before a log's first `[RUN]` form a run of their own), so a test-process log holding several runs (and architectures) gets one map per run, and a live line uses its own run's map. Two different counts for one site within one run are a malformed input: exit 2, naming both lines. A live line carries the worst site's dead count but not its channel count (`Training/LayerHealth.swift:985-990`), so this is the only offline source of that denominator (C's step-50 `value.bn` 12/16 needs the table at trainer step 513). A site with no known count makes the per-site arm "no data" for that evaluation; it is never guessed.
- **Sparse semantics — the offline replay feeds the real evaluator, and these are the only differences from the app:**
  - each stats row is one live evaluation whose window holds one record (that row);
  - the loss reference is the rows in the 1,000 trainer steps before the row, under rule 6's span definition (≥ 5 records spanning ≥ 200 steps);
  - "steps trained by this process" (rule 3's gate) is the stats row's own `step=` (segment steps), summed over the segments passed together; a checkpoint headline from a CLI path (`replay-…`, `vsuci-…`) uses its own `step=`. GUI `session-…` checkpoints carry no such count (their `step=` is the session's clock), so offline rule 3 has no data for them;
  - per-site dead counts between checkpoints: the live line's worst site only, which is the site with the most dead + mostly-off + always-on, not necessarily the most-dead (rule 2 note);
  - field coverage: `[REPLAY]` rows carry every rule input, with `pW` / `pL` at 2 decimals (rule 8 note). `[VS-UCI]` rows carry no `pIllM`, `vAbs`, `pW` or `pL` (`CLI/TrainVsUciRunner.swift:771-779`), so offline rules 4 and 8 have no data for train-vs-UCI logs; the in-app monitor has them;
  - logs without `[REPLAY]` / `[VS-UCI]` rows (GUI logs, whose `[STATS]` values are rolling means, not step samples) are replayed for the layer-health rules only (1, 2, 3, 9);
  - the output header lists these limitations for the logs given, so a reader knows which rules could have fired.
- `App/TrainingHealthReplayCLI.swift`: `DrewsChessMachine --replay-health-log <log> [<log> …] [--learning-grace-steps N] [--lr-warmup-steps N]`.
  - A pre-flight like the other no-GUI modes: a `handleReplayHealthLogIfPresent(rawArgs:)` in `App/DrewsChessMachineApp.swift`, called with the others (`:185`; pattern at `:1903-1938`), which hands over to `TrainingHealthReplayCLI.runAndExit`. It runs synchronously on the launching thread before any GUI or Swift task exists; it never parses inside a `Task`.
  - Prints the events in the live format, then a summary table.
  - **Legacy rows.** A step row from an older build may lack fields newer builds print (`pLogitMean`, `vLogitMean`, `trainerStep`, …). A missing rule-input field is treated exactly like `--`: not measured, so the rules that need it have no data. The output header lists, per log, every field absent from its rows and the rules left without data because of it (e.g. `pLogitMean absent in 5,371 rows: policy_offset_drift no data`). Nothing is inferred or filled in.
  - `trainerStep=` is the exception: without it there is no trainer clock, and the run refuses that log unless `--segment-step-as-trainer-step` is given. That flag uses the row's `step=` as the trainer step. Explicit only, never a fallback; refused unless every row in the log is strictly increasing; stated in the output header. Monotonic segment steps do not establish the clock's starting point: the header says the trainer steps may be offset by the start model's clock, which leaves span-based conditions (sustain spans, rule 6's reference) exact and makes the absolute ones (the learning gate of rules 4 and 8) unreliable for that log.
  - Exits 0. Exits 2 on an unreadable log; on a malformed line (named with its file and line number); on a row lacking `step=` or `loss=`, or lacking `trainerStep=` without the flag (reported as `unsupported log format: <file> (build <N> from its [APP] line): step rows carry no <field>`, not as malformed); or on a log with neither stats rows nor `[LAYER-HEALTH]` lines.
  - Evaluator state: within one log, each `[RUN]` line after the log's first starts a fresh evaluator, as the app starts a fresh monitor per run (a test-process log holding several runs is several runs). The first `[RUN]` of the first log starts the evaluator; the first `[RUN]` of each later log passed together does **not** reset it, so logs passed together are one continuing evaluator across the log boundary — the "C, one process" comparison in Evidence — and the output header says so; that is an offline convenience, since in the app each segment gets its own monitor (the "C segment 1 as its own process" row).
- A Python mirror of the rules is deliberately not written: it would be a second source of truth for the thresholds.

## D5. `LayerHealthLog.live`

- `LayerHealthLog.liveLines(trainer:)` (`Training/LayerHealthLog.swift:28-36`) becomes `live(trainer:) async -> LiveOutcome { lines, summary?, trainerStep? }`, mirroring `CheckpointOutcome` (`:46-49`).
- Its four call sites — `App/SessionController+Training.swift:2120,2177`, `CLI/CorpusReplayRunner.swift:1927`, `CLI/TrainVsUciRunner.swift:783` — are converted in **P1** to `live(trainer:).lines` with no behavior change, so P1 builds on its own with `liveLines` removed. P2 (CLI) and P3 (GUI) then hand `summary` to the monitor.
- `liveLine(summary:trainerStep:)` stays; `LayerHealthTests.swift:616` uses it.
- The checkpoint passes already return the summary (`CheckpointOutcome.summary`). Their call sites hand it to the monitor: `CLI/CorpusReplayRunner.swift:1614-1623`, `CLI/TrainVsUciRunner.swift:562-572`, and `App/SessionController+Checkpoint.swift:622-643` (which gains `monitor` and `stamp` arguments and a main-actor delivery callback; callers `:598` and `App/SessionController+Arena.swift:821`).

## D6. The value-FC1 velocity check (OD-15)

**One schedule for every path: a deadline, not a grid.** `TrainingHealthMonitor.valueFC1ReadDue(trainerStep:) -> Bool` (pure, under the evaluation lock) is true when `trainerStep − anchor ≥ TrainingHealthThresholds.valueFC1CheckIntervalSteps` (1,000; a declared constant like the thresholds, OD-14). The **anchor** is the trainer step of the newest rule-3 observation in this generation (from any source, by its stamp's trainer step), or, when there is none yet, the trainer clock at which this process's monitor started recording (the first record's step − 1) or the restored clock after a rewind. So two consecutive rule-3 observations are never more than 1,000 trainer steps apart, whatever falls in between, and a save's pass simply moves the deadline. Every path asks after the step's work and after any checkpoint pass that step ran:
- **Corpus replay** asks after the autosave block (`CLI/CorpusReplayRunner.swift:1979-1988`, pre-cadence-fix). Its rolling save runs at every 1,000th segment step (`:1110`), i.e. exactly 1,000 trainer steps after the process's starting clock and after each previous save, and its checkpoint pass feeds rule 3 before the question is asked. The deadline therefore lands on a save step that has already reset it, fresh or resumed (resumed at 513: start anchor 513, first save at trainer step 1,513, and so on), and no dedicated read happens — **as long as each save and its checkpoint pass succeed**. A first save failure is non-fatal (`:1628-1633`, the run continues) and a failed health pass returns no summary (`Training/LayerHealthLog.swift:69-70`); either leaves the anchor where it was, and the dedicated read at that step keeps the interval. That is the schedule working, not a double read: it reads only when no observation exists.
- **Train-vs-UCI** asks after its enumerated-checkpoint / periodic-save blocks (`CLI/TrainVsUciRunner.swift:847-853`). With `--enumerate-checkpoints` its pass every 1,000 segment steps covers the interval; without it (the default; `writeEnumeratedCheckpoint` returns at once when no writer, `:646-647`) the read is dedicated.
- **GUI** asks in the trainer worker right after `recordStep` (T8). Saves are hours apart, so the read is dedicated almost every time: 1,000 trainer steps after the session's start, and every 1,000 after that or after the latest save.

**The read.** `ChessTrainer.readTrainableVelocity(named: "value.fc1.weight") async throws -> (velocity: [Float], completedTrainSteps: Int)`, new. It follows the existing read pattern exactly — `enqueue` onto the trainer's serial `executionQueue`, one `network.graph.run` whose only target is that tensor's velocity variable, fed the dummy inference input, read back as fp32 — as `internalReadLayerHealthLiveState` (`Training/ChessTrainer.swift:5784-5830`) and `readVelocityValues` (`:5562-5587`) do. Because it runs on `executionQueue`, it falls between SGD steps and never overlaps one. Its result goes through the existing pure `LayerHealth.hiddenUnitVelocityHealth(layer: LayerHealth.valueFC1Layer(for: arch), velocity:)` (`Training/LayerHealth.swift:245-252,614`), on the calling task after the read returns (a single pass over the tensor).

**Probe isolation.** The graph run targets one variable and no operation, so nothing is assigned, no dropout op is encoded (no RNG advance), and the optimizer, weights, BN statistics and replay buffer are untouched — the same guarantee the live read has today. A test pins it (X1).

**Observation and log.** The result is a checkpoint-tier observation whose digest carries only `valueFC1` (every other field absent, so other rules have no data from it), stamped with `observationStamp()` taken before the read, and evaluated through `evaluateCheckpoint`. It is logged as one line, which the offline replay also parses (D4):
`[LAYER-HEALTH] value-fc1 trainerStep=<s> valueFC1ZeroVel=<zero>/<units> lowVel=<n> readMs=<ms> summaryMs=<ms>`.

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
| `training_health_learning_grace_steps` | Int | 1000 | 0…100000 | added to `lr_warmup_steps` for the not-learned forms of rules 4 and 8 |
| `training_health_action_<rule>` × 9 | Int | 0 (`log`) | 0…2 | `TrainingHealthAction` raw value, one per `TrainingHealthRule` |

**Checklist walk** (CLAUDE.md "Adding / removing / renaming a parameter"):
1. **Declare.** 12 `@TrainingParameter` declarations in `Training/TrainingParameters.swift`, added to `allKeys` (`:2479`). The action keys are Int-coded like `RandomSeedModeParameter` (`:1331-1341`), with `TrainingHealthAction(persistedRawValue:)`. A test pins the range against `TrainingHealthAction.allCases`, as `test_arenaPromotionCriterion_rangeMatchesEnumCases` does.
2. **Singleton.** Stored properties, `collectValues` / `applyOne`, and snapshot accessors (pattern at `:1490-1491`, `:1634-1635`, `:1743-1744`, `:1839-1840`).
3. **`parameters.json`.** Confirm all 12 appear in `--show-default-parameters`, and that `--create-parameters-file` → edit → reload round-trips. Write to a scratch folder first, never `--force` into the repo.
4. **Session save/load.** Optional fields on `SessionCheckpointState` (`Persistence/SessionCheckpointFile.swift`), passed through `buildCurrentSessionState`, and one `restore(…)` line each in `SessionParameterResume.applyGuiSession` (`App/SessionParameterResume.swift:129`, next to the `BatchStatsInterval` / `KLProbeInterval` lines at `:144-145`).
5. **`results.json`.** `alarm_config` (D3).
6. **Runtime log.** The `[HEALTH] config …` line at every run start (all paths, beside the `[RUN]` line). Live GUI edits are logged by the popover model as `[PARAM]` lines (pattern at `App/UpperContentView/TrainingSettingsPopoverModel.swift:1531-1535`).
7. **UI (P3).** A new "Health" tab in `TrainingSettingsPopover` (the `Tab` enum at `App/UpperContentView/TrainingSettingsPopover.swift:36-42`), as a new internal `View` in its own file, `App/UpperContentView/TrainingHealthTab.swift`, with bindings and validation in `TrainingSettingsPopoverModel.swift`. Layout: one aligned row per rule (name, one-line meaning, an action picker of Log / Stop on critical / Stop on any, with Stop on critical disabled and labelled "no critical level" for rules 6–9). Then enable, check interval and grace, with monospaced, padded digits.
8. **Live tunability.** The GUI stats task resolves `TrainingHealthConfig` from `TrainingParameters.shared` in the same `MainActor.run` hop that already reads the live parameters for each `[STATS]` emit (`App/SessionController+Training.swift:1458-1468`).
9. **Renames.** Not applicable.

**Consequences.**
- The lineage record's `parameters` snapshot is built from every key (`ReplayParams.init`, `CLI/CorpusReplayRunner.swift:48-57`, `LineageRecord.Parameters(values: parameters.rawValueMap())`). So the 12 keys enter every model file written after this lands, and `params_sha` on every `[RUN]` line changes from that build on.
- An exact resume from an older file finds them absent and applies `.currentSetting`: no `params` gap.
- This is the intended single source (HPARAM_RECORDING_PLAN.md S3); nothing else is needed for recording.

---

# Part T — Touch points

### T1. New pure core (P1)
- `Training/TrainingHealth.swift` (D1)
- `Training/TrainingHealthMonitor.swift` (D2)
- `Training/TrainingHealthLog.swift` (D3)
- `Training/TrainingHealthLogReplay.swift` (D4)

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
  - `testLearningGateUsesTrainerClockAndWarmup`: not-learned forms are silent below `warmup + grace` and active above. A resumed run whose trainer clock is already past the gate evaluates them at its first window.
  - `testRegressionFormsIgnoreTheGate`: rules 4 and 8.
  - `testDeadChannelsNewDamageFormIgnoresTheBaseline`, `testDeadChannelsNewDamageFormWarnsWhenASiteGrows` (a site rising while another falls still warns), `testDeadChannelsBaselineSurvivesARewind`, `testDeadChannelsCriticalArmsAreAbsoluteInBothForms`, `testDeadChannelsCriticalBaselineNeverClearsByStayingUnchanged`, `testDeadChannelsBaselineFromAFirstCheckpointDigest`; in `TrainingHealthLogReplayTests`, `testOfflineBaselineUnknownForASiteFirstSeenLater` and `testLegacyRowsReportAbsentFieldsAsNoData`.
  - `TrainingHealthStopPolicy` tests: `testFirstQualifyingFollowsRuleOrder`, `testActionChangedToLogMeansNoStop`, `testActionChangedToStopOnAnyStopsAnAlreadyActiveWarning` (both directions of a change made while a detached checkpoint pass was running: the pass's evaluation used the old actions, the decision uses the new).
  - `testValueHeadOneSidedSkipsDrawHeavyBatches`, `testValueHeadOneSidedFlagsVaryingButOneSidedOutput`, `testValueHeadOneSidedChanceImbalanceInOneWindowDoesNotSustain` (the stand-in for a batch whose W/L split is off by chance: the evaluator never sees labels, so the fixture is one one-sided window between two-sided ones).
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
  - `testValueFC1ReadDueAfterOneThousandStepsFromStart`, `testValueFC1ObservationsNeverMoreThanOneThousandApart` (a checkpoint at 1,001 moves the deadline to 2,001; nothing at 2,000 and nothing waits until 3,000), `testValueFC1ReadNeverDueOnCorpusReplaySaveCadence` (successful saves every 1,000 segment steps, fresh and resumed at 513), `testValueFC1ReadDueWhenASaveProducedNoObservation`, `testValueFC1ReadDueAnchorsOnTheRestoredClockAfterARewind`;
  - `testRule3UsesTheStampedTrainedCount`: a checkpoint stamped at 199 trained steps and evaluated after 200 is no data;
  - `testStaleHopFromAnEarlierMonitorIsIgnored` (P3, controller side);
  - `testCheckpointEvaluationNeverConsumesTheWindowOrDetectsRewind`;
  - `testLossReferenceSurvivesALongWindow`: a 3,000-record window still finds its reference in the 1,000 steps before it;
  - `testPendingWindowCapCountsTruncation`;
  - `testConcurrentRecordAndEvaluate`: one task records 10,000 steps while two others run live and checkpoint evaluations; every recorded step lands in exactly one window.
- **`TrainingHealthIncidentReplayTests.swift`**: the evidence as executable tests, through `TrainingHealthLogReplay` and the real evaluator.
  - The inputs are log excerpts shipped as **bundled test resources**: `DrewsChessMachineTests/Resources/TrainingHealthIncidents/<run>.log`, one plain-text file per run, in the test target's Copy Bundle Resources, read through `Bundle(for:)`. Not Swift source: a `[BATCH-STATS]` line is 61–80 KB (measured 61,592–79,930 B in B, C and R7), so verbatim excerpts as string literals would be tens of MB of Swift in an already slow-to-build test target. They are generated once from the logs named in **Evidence**.
  - Every line is verbatim **except `[BATCH-STATS]` lines**, which are reduced to the two fields the replay reads, keeping the original timestamp and tag: `<time>  [BATCH-STATS] {"step":N,"outcome_pct":{"D":…,"L":…,"W":…}}`. Each file's header (lines starting with `#`) states this transform, each source log's SHA-256 and line count at extraction (A and B were still growing; the cutoff is the Evidence cutoff), and the line-selection rule. `TrainingHealthLogReplay` accepts the reduced form because it parses `[BATCH-STATS]` as JSON and reads only `outcome_pct.D` (D4); it skips `#` header lines only in a file whose first line is the excerpt header, never in a session log.
  - Selection is contiguous, never thinned, so windows, loss references and sustain spans are exactly what the full log gives: every C stats row, live line and checkpoint block (both segments); B's through trainer step 1,500 plus every B checkpoint block to the cutoff; A's, R7's and R8's through trainer step 5,000 plus every checkpoint block. Each stats row is preceded by the `[BATCH-STATS]` line the replay would use for it (rule 8's draw gate). Full-length behavior is V-1's job.
  - `testArmCRaisesDeadChannelsCriticalAt50` (through the checkpoint-table pre-scan), `…IllegalMassCriticalAt350`, `…GradientCollapseAt400`, `…ValueFC1CriticalAt1513`, `…LossSpikeAt300`, `…ValueHeadOneSidedAt250`, `…PolicyOffsetDriftRaisedAt200ClearedAt400`, `…NeverRaisesNonFinite`.
  - `testArmCSegmentOneAloneRaisesDeadChannelsAtFirstEvaluation`.
  - `testArmBRaisesValueBNDeadChannelCriticalAt300`, `…PolicyOffsetDriftWarningAt900`, `…ValueFC1WarningAt1000`, `…OnlyDeadChannelsIsCritical`.
  - `testArmARaisesNothing`, `testR7RaisesNothing`, `testR8RaisesNothing`.
  - The expected steps are the prototype's (Evidence table). If the Swift log replay differs in a step, the difference is reported to the owner and the expected value is not quietly changed.
- **`ValueFC1VelocityReadTests.swift`** (needs Metal; P2): `readTrainableVelocity(named: "value.fc1.weight")` equals the matching slice of `exportVelocitySnapshot()` at the same step; and **probe isolation** — two trainers built from the same seed and weights, trained the same k steps with the same streams, one reading the velocity after every step, end with bit-identical weights, velocity and dropout Philox state.
- **`TrainingHealthLogTests.swift`**: exact strings for every line kind (the grep contract), including `--`, signed values and site rendering.
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
- `TrainingParametersTests.test_registry_size` (`DrewsChessMachineTests/TrainingParametersTests.swift:19-25`) pins `allKeys.count` at 85. It must become **97** (85 + the 12 new keys). The test's message ("requires intentionally updating this count") says the edit is expected maintenance; it is **not** approval. **Approved by the owner (OD-6, 2026-10-05)** for the edit the plan needs; the approval was given when the plan said 98, and dropping rule 9's action parameter (OD-7) makes the needed count 97.
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
  - Also run it on the 73-log survey set (each log alone; GUI-only logs get the layer-health rules, D4). **Pass:** no event in any log other than B, C and the two older-model runs: `dcm_log_20261003-001701.txt` gives a `dead_channels` warning under the absolute form and nothing under the new-damage form (17 of 1,808 dead at the first live line and at the step-3 checkpoint, so the total comparison is exact; the per-site comparison is evaluated for `value.bn` only, the first live line's worst site at 2 of 16, and the header lists the other sites as unknown-baseline, D4; OD-16), and no rule 3 event for its 49/128 after 3 trained steps; `dcm_log_20261005-012743.txt` gives a `policy_offset_drift` warning at trainer step 100 and nothing else: it is a 500-step benchmark of the older model `20260708-4-kEiZ` (nT8Y line, loaded from `20260701-nT8Y-resume4-replay-latest.safetensors`), whose `pLogitMean` sits at −14.26 from its first diagnostic row — a shared policy offset carried in from the mixed-precision era, which rule 7 correctly reports (always-on channels are not alarmed; `rvMaxOverMedian` 332 < 1,000). This corrects the earlier expectation of "none", found by re-running the survey under the decided thresholds. In particular, none of the 16/16 `valueFC1ZeroVel` saves in the 15 test-process logs (trainer clock 0–274, nothing trained) may raise rule 3.
- **V-1b — R7/R8 layer health.** R7/R8's logs predate `[LAYER-HEALTH]`, so the brief's "R7/R8 have 0 dead channels" is not in their logs. Run `--analyze-numerics <file> --numerics-static-only --numerics-out <scratch folder>` on R7's and R8's final checkpoints (stems in `experiments/20261002-noSE-noReZero/README.md`). Layer health is part of the static checks (`Network/NumericsAudit.swift:209`): no forward passes, read-only on the model, though it builds one MPSGraph to read variable names (`App/NumericsAuditCLI.swift:113-124`). **Pass:** 0 dead channels and `valueFC1ZeroVel` ≤ 1/128. If a file carries no optimizer velocity, the audit reports `valueFC1ZeroVel` unavailable; that half is then reported as not checked, never as passed.
- **V-2 — Per-step loss distribution (rule 6).**
  - A 2,000-step corpus replay on a healthy configuration (R7's parameters, `--seed`), with a temporary `--output`. Read the largest per-window ratios from the `[HEALTH] check` lines, which report them permanently (`lossMaxRatio=` = the largest `max_W(loss) / ref`, and `lossMedianRatio=` = the largest `median_W(loss) / ref`, over the evaluations since the previous check; `--` when no evaluation had a reference). No temporary debug code.
  - **Pass:** the largest `lossMaxRatio` over the run is below 2.0 and the largest `lossMedianRatio` below 1.2, so the 3× and 1.5× arms keep a margin. Otherwise raise the arm to the measured maximum × 1.5 and tell the owner.
- **V-3 — Live CLI run, log-only.** Run C's recipe (`experiments/20261005-lr-schedule-ab/README.md` Arm C) for 600 steps with `--output`.
  - **Pass:** `[ALARM] health raise rule=dead_channels severity=critical` at or before trainer step 100. `illegal_mass` critical within 100 steps of the divergence. `alarms` in `results.json` matches the log line for line. The run continues to its step limit (log-only).
- **V-4 — Stop action, CLI.** Same recipe with `training_health_action_illegal_mass=1`.
  - **Pass:** the `[ALARM] health stop` line is followed by `[REPLAY] training health alarm illegal_mass requested a stop — stopping at step N`, where N is the segment step of the evaluation that raised it (no further step runs), and the `health-stop` final save records trainer step = that evaluation's trainer step. `termination_reason` is `training_health_alarm`, the exit status is per OD-4, and a `--resume-exact` from the saved file reports `EXACT`.
  - Repeat on train-vs-UCI with a 2-opponent pool. **Pass:** a `vsuci-health-stop` session folder exists.
- **V-5 — Observer neutrality and cost.**
  - Three 300-step corpus replays with the same `--seed`: two with `training_health_alarms_enabled=false` (D1, D2), one with it on (E).
  - **Pass, neutrality:** compare the `[REPLAY]` lines field by field, ignoring `ms` and the line timestamp. If D1 and D2 are identical, E must be identical to D1. If D1 and D2 differ (the GPU is not deterministic on this machine), compute each numeric field's largest relative difference between D1 and D2 over all rows, with relative difference `|x − y| / max(|x|, |y|)` and 0 when both are 0; E against D1 must stay within that per field, a field that reads `--` must read `--` in all three, and the report says the exact comparison was not possible.
  - **Pass, cost:** two measurements, because `ms` covers only `trainStep` (`Training/ChessTrainer.swift:7243-7269`, printed at `CLI/CorpusReplayRunner.swift:1919`) and the monitor's work runs after it.
    - E runs with `training_health_check_interval_steps=50`, so its `[HEALTH] check` lines (plus the `final=true` one) cover the whole run. Σ`cost_ms` ≤ 0.5% of Σ`train_ms` over those lines (the same steps, from the same lines).
    - Wall time from the `[REPLAY] step=50` line to the `[REPLAY] step=300` line (timestamps; the three runs use an autosave interval above 300 so no save falls between them): |E − D1| ≤ max(1% of D1, |D1 − D2|).
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
- **V-8 — Full test suite** before merge (this touches persistence and parameters). Every test passes, and the only edited test is OD-6's (`test_registry_size` 85 → 97).

---

# Part P — Phasing

Each phase builds once at its end, then is committed (owner's standing rule for approved multi-phase plans).

**Dependency: `STATS_LINE_RESUME_CADENCE_FIX_PLAN.md` lands before P2.** P2 wires the CLI evaluations into the step-line tick that plan replaces (`TrainingStepLogCadence.isStepLineTick`), and both edit the same runner blocks. P1 touches those blocks only for the mechanical `liveLines` → `live(trainer:).lines` conversion; if P1 goes first, the cadence fix rebases over a one-line change, and if the cadence fix goes first, P1 converts the post-fix block. P2 re-cites T3/T4 against the fixed code before implementing.

- **P1 — Pure core and incident tests.** T1; T2 including the conversion of all four `liveLines` callers to `live(trainer:).lines` (D5), so `liveLines` can go and P1 builds alone; `TrainingHealthConfig` with its memberwise initializer only; the `TrainingAlarm.Severity` `Codable` extension. X1's evaluator, monitor, log, log-replay and incident-replay tests; `valueFC1ReadDue` (D6) and its tests. No behavior change in any run.
- **P2 — Parameters, CLI paths, results, offline replay.** Part K items 1–6 and 8 (CLI); `TrainingHealthConfig(_ snapshot:)`; T3–T7; `ChessTrainer.readTrainableVelocity(named:)` and `ValueFC1VelocityReadTests` (D6); X1's parameter and recorder tests; X2 (after OD-6). Validation V-1, V-1b, V-2, V-3, V-4, V-5.
- **P3 — GUI.** T8, T9, Part K item 7 (the Health tab) and item 8's GUI side; X1's alarm-controller tests. V-6, V-7.
- **P4 — Documentation.** T10.
- **P5 — Later work.** OD-9 (shared conditions for the GUI-only detectors; decided yes, each `TrainingAlarmControllerTests` edit listed for approval when planned), OD-10 (bounded per-segment lineage alarm summary; decided yes, lands with HPARAM_RECORDING_PLAN P4's schema 3). (OD-15, decided: the value-FC1 velocity check on a fixed 1,000-step interval, is not P5 — it lands in P1–P3 with D6.)

---

# Owner decisions needed

Decided by the owner on 2026-10-05: every decision below (OD-1 in two rounds, OD-2 to OD-11, OD-13 to OD-17). OD-12 is superseded by its own plan. **None is open.** OD-16's decision carries an owner "revisit later" note. Each decision is recorded after its original text.

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
- **OD-6 — Test edit.** `test_registry_size` 85 → 98. Pending the owner's explicit approval; P2 cannot pass the suite without it. **Decided (owner, 2026-10-05):** approved for the edit the plan needs. With rule 9 dropped (OD-7) the plan adds **12** parameters, not 13, so the edit is **85 → 97**, not 98.
- **OD-7 — Keep `value_loss_above_ln3`?** It is not supported by the incident data, and it is near-firing on healthy GUI self-play. The recommendation is to keep it as log-only with the long sustain, or to drop it. (Related: rule 8 is named `value_head_one_sided` for what its statistic measures; the owner may prefer another name before the id is fixed in logs and `results.json`.) **Decided (owner, 2026-10-05):** **drop `value_loss_above_ln3` entirely** — its rule, its action parameter, its tests and its evidence rows are removed, and the rules are renumbered (the former rule 10, `bn_running_variance_runaway`, is now rule 9). The plan now has nine rules and twelve parameters.
- **OD-8 — Sound.** Beep on critical health raises and escalations only (recommended), or never. Optional macOS notification (`UNUserNotificationCenter`) for critical: off unless requested. **Decided (owner, 2026-10-05):** accepted: beep on critical raises and escalations only; no notification.
- **OD-9 — One evaluator for the old GUI detectors too.** Move the divergence, value-saturation and pD detectors' conditions and the legal-mass probe's condition into `TrainingHealthEvaluator`, so the CLI paths get them. This changes `TrainingAlarmControllerTests` (each edit listed for approval then). Deferred to P5. **Decided (owner, 2026-10-05):** yes, in P5; each `TrainingAlarmControllerTests` edit is still listed for approval when P5 is planned.
- **OD-10 — Lineage.** Record a per-segment alarm summary (rule, first trainer step, highest severity) in the lineage record's segment summary when HPARAM_RECORDING_PLAN P4 introduces schema 3? The recommendation is yes, as one optional field written by `LineageTracker`. Not before schema 3, because this plan does not change the model-file format. **Decided (owner, 2026-10-05):** yes, provided the lineage JSON cannot be flooded. Bounded design: one entry per rule that raised in the segment — `{rule, first_trainer_step, highest_severity, raise_count}` — and no per-event list; rules that never raised have no entry. Worst case per segment, in compact JSON with keys `rule` / `first_trainer_step` / `highest_severity` / `raise_count`: the longest entry (`bn_running_variance_runaway`, both integers at the `Int` maximum, 19 digits; `"critical"`) is 143 B, and all nine ids at those maxima make a 1,205 B array (≈ 1.2 KB, before the enclosing field name and any pretty-printing whitespace). Realistic values (a 7-digit first step, a 4-digit raise count) give ≈ 960 B. A record carries a summary of every earlier segment of its run, so the total grows by at most ≈ 1.2 KB per segment (9,640 B for the eight-segment v5 line), never with the number of events. Lands with HPARAM_RECORDING_PLAN P4's schema 3 (P5 here).
- **OD-11 — CLAUDE.md.** Add `[HEALTH]` and `[ALARM] health …` to the tag list in "Where to look for runtime state", a sentence on the alarms under "Training observability", and `vsuci-health-stop` beside `vsuci-periodic` / `vsuci-final` / `vsuci-abort` in "Saved model state". **Decided (owner, 2026-10-05):** yes.
- **OD-12 — Stats-line cadence after a resume.** Superseded by `documentation/plans-active/STATS_LINE_RESUME_CADENCE_FIX_PLAN.md`.
- **OD-13 — GUI `--train` alarm stop: save first?** The legal-mass precedent exits without a session save. The recommendation is to follow it, for consistency. **Decided (owner, 2026-10-05):** yes.
- **OD-14 — Thresholds as parameters?** The recommendation is no: declared constants, like `LayerHealth`'s and `TrainingAlarmController`'s. 9 rules with 1–3 thresholds each would add about 20 knobs. (V-2 reads its measurements from the permanent `[HEALTH] check` fields; there is no temporary debug switch.) **Decided (owner, 2026-10-05):** yes: declared constants.
- **OD-15 — Live value-FC1 velocity in the GUI.** Rule 3 is checkpoint-only, and GUI checkpoints are hours apart. Reading `value.fc1.weight`'s velocity on the live tier (128 × 1,024 floats = 512 KB per read) would be new GPU readback on the trainer's queue. Rule 8 already covers the same failure live. The recommendation is no.
  - **In plain terms** (the owner asked what it means): rule 3 spots a dead value head by finding hidden units in the value head's first fully connected layer (`value.fc1`) whose optimizer momentum has decayed to exactly zero — units that stopped learning. That momentum is only read when a checkpoint is saved, because reading it costs an extra 512 KB copy from the GPU. In the GUI, saves happen only every few hours (periodic) or at promotions, so the GUI would notice this failure hours late. OD-15 asks whether to also read that one tensor's momentum every `[STATS]` emit (about once a minute), so the GUI notices within minutes — at the cost of a 512 KB GPU read per minute, scheduled between training steps on the trainer's queue. The recommendation is no, because rule 8 (a value head whose outputs never change sign) catches the same failure live from numbers the trainer already reports, and caught C about 1,250 steps before rule 3 could.
  - **Decided (owner, 2026-10-05):** every check runs on a set interval or trigger. Rule 3's value-FC1 velocity read runs every 1,000 trainer steps on every path: corpus replay reuses its 1,000-step saves (no double read); train-vs-UCI reuses its enumerated checkpoints when they are on; otherwise — the GUI, whose saves are hours apart, and train-vs-UCI without `--enumerate-checkpoints` — one dedicated read of that tensor on the trainer's `executionQueue` between SGD steps, under the probe-isolation rules (D6). Cost in D6; checked in V-5. The "Every check" table near the top lists every check's trigger per path.
- **OD-16 — Rule 2's warning form.** Absolute (warn on any dead channel) or new-damage (warn only when a site's dead count rises above its count at the process's first evaluation); the critical arms are absolute in both. Recommendation: new-damage, so a long-trained line with a few dead channels (17 of 1,808 on the grafted v5 model) does not hold a permanent warning, with the baseline logged once so it is never hidden. Until it is decided the plan implements the absolute form (OD-1's decision). **Decided (owner, 2026-10-05):** keep the absolute warning form for now. **Revisit later** (owner): a long-trained line that carries a few dead channels holds a permanent warning under it; the new-damage form, its tests and its offline semantics stay written down here for that revisit, but are not implemented. Note that with OD-1's 20% per-site critical arm, a line whose small `value.bn` (16 channels) already carries 4 or more dead channels is critical at its first evaluation under either form.
- **OD-17 — Long-run evidence.** On two older long mixed-precision lines from before the head-numerics fix, rule 6 raised 21 times on the 268,500-step v5 line (ratios up to 2.36) and the since-dropped value-loss rule raised from step 58,050 on the 1.4M-step qeu8 line (Long runs). Keep rule 6's thresholds as proposed and treat those raises as true signals (they predate `da15920`), or loosen rule 6's median arm before merge? Recommendation: keep, since the defaults are log-only, and revisit after the first long post-fix run. (Its value-loss half is moot since OD-7 dropped that rule.) **Decided (owner, 2026-10-05):** keep the thresholds.

---

# Risks

- **Thresholds come from one corpus, one architecture family and one batch size.**
  - `gNorm` and the loss-spike ratio are scale-dependent. A different architecture or loss weighting can move them.
  - Mitigations: declared constants; the `[HEALTH] check` line shows each rule's no-data count; V-1 reruns the survey; defaults are log-only.
- **Fresh GUI runs are only partly in the evidence.** Two fresh GUI runs (GUI-F1 to 14,781 steps, GUI-F2 to 2,041) cover the scalar rules through their rolling `[STATS]` means; none since `[LAYER-HEALTH]` shipped covers the layer-health rules. V-6 is the in-app check.
- **Long runs are outside the main evidence (OD-17).** Every baseline stops by about 35,000 trainer steps; the Long-runs replay in Evidence shows rule 6 firing on two older long lines.
- **`HEAD_ACTIVATIONS_PLAN.md` (in flight) changes what some rules see.** It adds per-site activations, including smooth ones on the head hidden layers. Rule 2 already treats SiLU / GELU BN sites as not classified, but its per-site channel map gains and loses sites with it. Rule 3's premise is a dead ReLU unit in `value.fc1`: with a leaky or smooth value hidden activation the gradient is rarely exactly zero, so the rule loses most of its meaning, and whether `LayerHealth` still reports `valueFC1ZeroVel` there is that plan's decision. Whichever lands second re-checks rules 2 and 3 against the other.
- **Promote Trainee Now is not gated during a divergence suspension today** (`App/SessionController+ManualPromote.swift:35-52` checks Play-and-Train, a running arena and a save in flight, not the suspension), so a NaN trainer can be promoted from the menu. This plan gates it for `.healthAlarm`; gating it for `.divergence` too is part of OD-5.
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
