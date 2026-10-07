# Training health alarms

Every training path checks itself while it trains and says so in the log: GUI Play-and-Train (including GUI `--train`), `--replay-corpus` and `--train-vs-uci`. One evaluator does the checking for all of them, and `--replay-health-log` runs the same evaluator over saved logs. By default every rule only logs. Any rule can be set to stop the run.

Design, evidence and decisions: `documentation/plans-active/TRAINING_HEALTH_ALARMS_PLAN.md`.

## What is checked, and when

| Check | Every path |
|---|---|
| Record the step (loss, illegal-move mass, pre-clip gradient norm, step time; the policy logit mean on diagnostic steps) | every SGD step |
| Live evaluation: a live `[LAYER-HEALTH]` read (BN state, ReZero α) plus rules 1, 2, 4–9 | every 50 trainer steps, on overall trainer-step multiples, independent of the step lines. On the CLI, one read serves both when a step line falls on the same step |
| Checkpoint evaluation: the save's full-tensor `[LAYER-HEALTH]` pass plus rules 1, 2, 3, 8 | every save (CLI rolling and final saves, train-vs-UCI session folders and enumerated checkpoints, every GUI session save) |
| Rule 3's value-FC1 velocity | at most 1,000 trainer steps apart. A save's pass covers it where there is one (corpus replay's 1,000-step saves; train-vs-UCI with `--enumerate-checkpoints`); otherwise one dedicated read of `value.fc1.weight`'s velocity on the trainer queue, logged as `[LAYER-HEALTH] value-fc1 …` |
| `[HEALTH] check` line and `active` reminders | at each multiple of `training_health_check_interval_steps` (default 1,000) |

The monitor only observes. It makes no random draws and changes no trainer, optimizer or replay-buffer state. Its one GPU read beyond the paths' own is the value-FC1 velocity read: one variable, no operation, between SGD steps. Its whole cost is reported on every `[HEALTH] check` line (`cost_ms=` against `train_ms=`).

## The rules

"Window" means the steps since the previous live evaluation. "Reference" means the records of the 1,000 trainer steps before the window; a reference counts only with at least 5 records spanning at least 200 steps.

| # | Rule | Raise | Clear |
|---|---|---|---|
| 1 | `non_finite` | critical: any NaN or Inf in BN state, ReZero α, a checkpointed tensor, or a recorded step value | never (non-finite weights do not heal) |
| 2 | `dead_channels` | parked BN channels (dead for relu / leaky_relu; the pass-through equivalent for silu / gelu). Warning: any. Critical: ≥ 5% of all such channels, or ≥ 20% of one site's. Every line names every affected site | none parked on 2 evaluations ≥ 1 step apart |
| 3 | `value_fc1_zero_velocity` | value FC1 units whose velocity is exactly zero. Warning ≥ 5%, critical ≥ 50%. Judged only after this process trained ≥ 200 steps; applies only to a ReLU value hidden layer | < 2.5% |
| 4 | `illegal_mass` | critical: a regression (the run's minimum window median was < 0.5 and the median is now ≥ 0.8; or the median is ≥ 0.3 and ≥ 10× that minimum), or not learned (median ≥ 0.5 past `lr_warmup_steps + training_health_learning_grace_steps`) | < 0.15 on 2 evaluations ≥ 50 steps apart |
| 5 | `gradient_collapse` | critical: window median gradient norm < 0.1 (2 evaluations, ≥ 50 steps) | ≥ 0.2 on 2 evaluations |
| 6 | `loss_spike` | warning: window median ≥ 1.5× or window max ≥ 3× the reference median | median < 1.2× |
| 7 | `policy_offset_drift` | warning: window median \|policy logit mean\| ≥ 3 (2 evaluations, ≥ 50 steps) | < 2 on 2 evaluations |
| 8 | `bn_running_variance_runaway` | warning: largest BN running-variance max/median ≥ 1,000 | < 300 on 2 evaluations ≥ 1 step apart |
| 9 | `gradient_spike` | warning: window max gradient norm ≥ 5× the reference median | < 2.5× |
| 10 | `divergence` | window medians of policy entropy and gradient norm. Critical: entropy < 0.5 or gNorm > 500. Warning: entropy < 1.0 and gNorm > 50 (2 evaluations, ≥ 50 steps) | neither, on 2 evaluations |
| 11 | `value_saturation` | window median of the value head's mean \|p_win − p_loss\|: warning ≥ 0.97, critical ≥ 0.995 (2 evaluations) | below 0.97 on 2 evaluations |
| 12 | `value_draw_saturation` | window median of the value head's mean p_draw: warning ≥ 0.92, critical ≥ 0.97 (a fresh head starts at 0.75) | below 0.92 on 2 evaluations |
| 13 | `legal_mass_stall` | critical: window median illegal mass above `legal_mass_collapse_threshold` (0.99) on `legal_mass_collapse_no_improvement_probes` (8) consecutive evaluations with no improvement, past warmup + learning grace | at or below the threshold on 2 evaluations |

Rules 10–13 are the conditions of the GUI's training-alarm banner detectors and its legal-mass probe (owner decision OD-9). One set of functions, `TrainingHealthDetectorConditions`, decides their levels for both: the banner applies them to its heartbeat's rolling means with its own streaks (unchanged), the evaluator to the step window's medians, so the command-line paths judge them too. In the app both report: the banner as before, the alarm list as a rule. Their inputs are diagnostic-step fields; `[VS-UCI]` rows carry no `vAbs` / `pD`, so offline rules 11–12 have no data for train-vs-UCI logs.

Rule 9's reference median is the same function the relative gradient cap uses (`TrainingHealthReference.make(_:windowStart:policy:)`; the rules call it with `TrailingReferencePolicy.spikeRules`, the cap with its own window N and warm-up W; `documentation/plans-active/RELATIVE_GRADIENT_CAP_PLAN.md`). The data is not shared: the monitor keeps its own observer history, the trainer keeps the cap's per-step history as training state, and both are fed pre-clip norms, so a clip never hides a spike from rules 5 and 9. Every clip is logged as `[GRAD-CLIP]`, and each step line carries `gNormMax=` / `clips=` / `gCap=` for the steps since the previous line.

Thresholds are declared constants (`TrainingHealthThresholds`), not parameters. Each was measured against the incident runs (arms B and C of the 2026-10-05 LR-schedule A/B, B-silu) and the healthy baselines (arm A, R7, R8, a long GUI run).

A rule whose input has no data in a window holds its state: it neither raises nor clears, and the check line counts it under `nodata=`. LR-cycle peaks are not suppressed, because every incident began at a rising LR. A GUI promotion rewinds the trainer clock; the monitor then discards its windows and references but keeps active alarms, which clear only by recovery.

## Actions and what a stop does

One parameter per rule, `training_health_action_<rule>`: `0` log (the default), `1` stop while active at critical, `2` stop while active at any severity. Rules 6–9 have no critical level, so `1` never stops them.

For unattended runs, the plan recommends `1` for `non_finite`, `illegal_mass`, `gradient_collapse` and `dead_channels` (documented, not applied).

| Path | Stop |
|---|---|
| `--replay-corpus` | stops before the next step; final save with reason `health-stop`; `results.json` `termination_reason: "training_health_alarm"`; **exit status 35** |
| `--train-vs-uci` | the same, with a `vsuci-health-stop` session folder; exit 35 |
| GUI, interactive | training is **suspended**, not torn down. The trainer worker parks but still acknowledges pause requests, so saves complete. Self-play and the periodic autosave continue. Arenas and Train ▸ Promote Trainee Now are refused. Stop, then Start clears it; to keep training a damaged trainer, set the rule to Log first |
| GUI `--train` | ends through `AutoTrainTermination` with `training_health_alarm`, no session save, exit 0 (as the legal-mass collapse does) |

A stop requested by a CLI run's final save changes neither the termination reason nor the exit status, because the run was already ending. In the GUI the stop decision uses the actions in force when each evaluation arrives, so a change on the Health tab applies at the next evaluation.

## Parameters (Health category, all live-tunable)

| id | default | range |
|---|---:|---|
| `training_health_alarms_enabled` | `true` | — |
| `training_health_check_interval_steps` | 1000 | 50…100000 |
| `training_health_learning_grace_steps` | 1000 | 0…100000 |
| `training_health_action_<rule>` × 13 | 0 | 0…2 |

- **Command-line paths:** read once from the run-start snapshot.
- **GUI:** the config is resolved at every live evaluation, and the stop decision reads the actions when the result arrives.
- **Sessions:** the settings are saved in `session.json`, with actions by name; a resume restores them.
- **Older sessions:** a session without them keeps the current settings.

## Log formats

```
[HEALTH] config enabled=true interval=1000 grace=1000 warmup=1000 momentum=0.85 path=replay actions=non_finite:log,… value_fc1_zero_velocity=applies
[HEALTH] check trainerStep=2000 generation=0 evaluations=21 live=20 checkpoint=1 stale=0 truncated=0 cost_ms=3.1 train_ms=651400.0 liveReadFailed=0 lossMaxRatio=1.14 lossMedianRatio=1.02 gradMaxRatio=1.10 nodata=none active=dead_channels:critical dead_channels_sites=value.bn(6/16)
[ALARM] health raise rule=dead_channels severity=critical trainerStep=300 value=dead=5/1040 sites=value.bn(5/16) threshold=site>=0.2 action=log lr=0.3 mom=0.85
[ALARM] health clear rule=loss_spike severity=warning since=300 trainerStep=500 value=median/ref=1.14 max/ref=1.14 ref=7.5471
[ALARM] health stop rule=illegal_mass severity=critical trainerStep=350 action=stop_on_critical
[LAYER-HEALTH] value-fc1 trainerStep=3000 trained=2500 valueFC1ZeroVel=2/128 lowVel=9 readMs=1.23 summaryMs=0.50
[REPLAY] training health alarm illegal_mass requested a stop — stopping at step 350
[HEALTH] training suspended: rule=dead_channels severity=critical trainerStep=514 …   (GUI)
```

- **Grep:** `grep '\[ALARM\] health'` selects exactly the event lines.
- **Event kinds:** `raise`, `escalate` (warning → critical), `worsen` (the count rose while active; at most one per check interval), `active` (a reminder each check interval), `clear`, `stop`.
- **stderr:** on the command-line paths, raise, escalate and stop lines also go to stderr.
- **results.json:** `alarms` holds every event in log order and `alarm_config` the resolved settings. GUI `--train` records both too.

## In the app

- **Alarm list:** active health alarms show in a list under the training-alarm banner. Each row shows the severity symbol, the rule, the measured value, the step it was raised at, and whether its current action stops the run.
- **Suspension header:** when a health stop has suspended training, a header row names the rule.
- **Sound:** a critical health alarm beeps even when the banner shows nothing; warnings never beep. The list has its own Silence button.
- **Health tab:** the training settings popover's Health tab holds the 16 settings. Each edit is logged as a `[PARAM]` line.

## Offline replay

```
DrewsChessMachine --replay-health-log <log> [<log> …] [--learning-grace-steps N] [--lr-warmup-steps N] [--segment-step-as-trainer-step]
```

This runs the real monitor and evaluator over saved session logs. It reads only: no GUI, no GPU, no writes. The settings are the declared defaults, never your saved ones, and every action is `log`. Pass the run's own warmup and grace to judge its not-learned gate.

**Output:**
- a header (lines starting with `# `) listing what the logs cannot show;
- every line the monitor would have written;
- a per-rule summary (raises, first raise, highest severity, clears, active at the end).

**Exit status:** 0, or 2 on an unreadable or malformed log.

**How offline differs from the app (sparse semantics):**
- Each `[REPLAY]` / `[VS-UCI]` row is one evaluation whose window is that row.
- Spike references are the rows in the previous 1,000 trainer steps.
- Rule 3's gate uses the row's `step=` (segment steps) and the value-fc1 lines' `trained=`.
- `[VS-UCI]` rows carry no `pIllM`, so rule 4 has no data for train-vs-UCI logs.
- GUI logs (no step rows) get the layer-health rules only.
- Logs passed together are one continuing evaluator; each later `[RUN]` within a log starts a fresh one.
- Rows without `trainerStep=` (older builds) need `--segment-step-as-trainer-step`.

**Validation (2026-10-06):** run over the full evidence logs, it reproduces every incident in the plan's Evidence table. Arms A, R7 and R8 raise nothing. The detail is in the plan's "Implementation notes (P2)".
