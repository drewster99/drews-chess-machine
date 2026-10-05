# Stats-line cadence after a resume — bug-fix plan

Status: proposed 2026-10-05; revised after an independent review (M1 checkpoint-step consumers, M2 runner-level test). Owner asked for the fix ("fix that bug"); this is the plan it follows.
Found while drafting `TRAINING_HEALTH_ALARMS_PLAN.md` (its OD-12, which this plan supersedes).

## The bug

After an exact resume, every `[REPLAY]` (corpus replay) and `[VS-UCI]` (train-vs-UCI) step line,
and the matching `results.json` stats row, is missing the diagnostic fields (`pEnt`, `playedP`, `pW`
/ `pD` / `pL`, `vAbs`, `pLogitMean`, `vLogitMean` print as `--`; the row's fields are `null`).

Evidence: arm C's resumed segment (`~/Library/Logs/DrewsChessMachine/dcm_log_20261005-121841.txt`,
exact resume from trainer step 513): 113 of 113 step lines have `pEnt=--`. Its first segment
(`dcm_log_20261005-090417.txt`) and the uninterrupted B run (`dcm_log_20261005-013235.txt`, 683 lines;
the only `--` line is step 1) are fine.

## Cause

Two cadences that are meant to coincide count different things:

- The step line is written when the **segment-local** step counter says so:
  `if step == 1 || step % logEvery == 0` with `logEvery = 50`, where `step` starts at 0 in every process
  (`CLI/CorpusReplayRunner.swift:1839-1840,1904`; `CLI/TrainVsUciRunner.swift:744,764`).
- The trainer computes the diagnostic reductions when the **trainer's cumulative** next step is a
  multiple of the diagnostics interval: `includeDiagnostics = nextStep % diagnosticsInterval == 0`, with
  `nextStep = _completedTrainSteps + 1` and `diagnosticsInterval = batchStatsInterval > 0 ?
  batchStatsInterval : diagnosticsFallbackInterval` (`Training/ChessTrainer.swift:4649-4657`, fallback
  constant at `:1201`).

In a first segment the two counters are equal, so with `batch_stats_interval` 10 every multiple of 50
is a diagnostic step. A resumed segment's trainer step is `resumeOffset + step`; unless
`resumeOffset` is a multiple of the interval, no logged step is ever a diagnostic step (C: offset 513,
lines at trainer steps 514, 563, 613, …; its `[BATCH-STATS]` lines sit at 520, 530, …).

A second, smaller way to hit the same symptom exists without any resume: a `batch_stats_interval`
that does not divide 50 (e.g. 7, or 100) leaves some or all step lines without diagnostics.

## Hypotheses considered

1. **Log cadence keyed on the wrong counter (chosen).** Key the step line on the trainer's cumulative
   step. Then a resumed segment logs at the same trainer steps the uninterrupted run would have
   (550, 600, … after a resume at 513), which also makes a resumed run's log line up row-for-row with
   the uninterrupted run's for comparison. Diagnostics stay on the trainer's own cadence, so which
   steps run the diagnostic graph is unchanged — no effect on training numerics or exact-resume
   determinism.
2. **Ask the trainer for diagnostics on the steps the runner logs** (a `trainStep(…, includeDiagnostics:)`
   request OR'ed with the trainer's own cadence). Rejected: it makes which steps run the diagnostic
   executable depend on where a process started, so a resumed run and the uninterrupted run would run
   different graphs on the same trainer steps — a possible bit-level divergence on the one path (corpus
   replay) whose resume is meant to continue exactly. It also adds a second source for "is this a
   diagnostic step".

## Fix

1. A pure, shared predicate for the step-line tick, used by both runners (single source):
   `TrainingStepLogCadence.isStepLineTick(trainerStep:segmentStep:batchStatsInterval:)` (new file
   `Training/TrainingStepLogCadence.swift`), true when any of:
   - `segmentStep == 1` (the segment's first step, kept as today);
   - `trainerStep % TrainingStepLogCadence.stepLineInterval(batchStatsInterval:) == 0` — the
     diagnostics-carrying cadence, on the uninterrupted run's trainer steps. Per OD-A (d), the interval
     is the smallest multiple of the trainer's diagnostics interval
     (`ChessTrainer.diagnosticsInterval(batchStatsInterval:)`, the same function `isDiagnosticsStep`
     uses) that is ≥ `TrainingStepLogCadence.minimumStepLineInterval` (50, replacing both local
     `logEvery` constants): 50 for today's interval of 10 or the fallback, 56 for 7, 100 for 100. Each
     runner logs the interval in use once at start;
   - `segmentStep % TrainingStepLogCadence.autosaveEvery == 0` — one shared constant (1000) replacing
     both runners' local `autosaveEvery` (`CLI/CorpusReplayRunner.swift:1110`,
     `CLI/TrainVsUciRunner.swift:306`), which their autosave / enumerated-checkpoint code then uses
     too, so the save cadence and the checkpoint tick cannot drift apart — a line on every
     autosave / enumerated-checkpoint step, as today. Consumers rely on a line landing exactly on each
     checkpoint step: `experiments/table_common.py:87` (`if step % 1000: continue`, the buffer
     plies/game column), `documentation/dashboards/replay.py:283-311` (`_metrics_at` / `_games_at`,
     nearest line at or below the checkpoint, so a line up to 49 steps early would understate
     `games_fed`) and `documentation/dashboards/vsuci.py:105-108` (`nearest_at`). Without this tick a
     resumed segment would have no line on any checkpoint step. After a resume these checkpoint lines
     carry diagnostics only when they happen to fall on a diagnostics step; the trainer-cadence lines
     always do (owner decision OD-C).
2. The trainer's diagnostic-step rule becomes two pure static functions,
   `ChessTrainer.diagnosticsInterval(batchStatsInterval:)` and
   `ChessTrainer.isDiagnosticsStep(trainerStep:batchStatsInterval:)`, called from the existing site at
   `ChessTrainer.swift:4656-4657` (same result there), so the tests exercise the trainer's actual rule.
3. Both runners read `trainer.completedTrainSteps` into `observedSteps` once, **before** the tick test
   (today it is read inside the block), and pass it to the predicate; the block keeps using the same
   value for LR, momentum and the cycle values. The `results.json` row keeps `steps: step` (segment step) and its existing
   `cum_trainer_step`; only which steps get rows changes after a resume.
4. A `batch_stats_interval` that does not divide 50: handled by the interval rule in 1 (OD-A (d)).

**OD-A and OD-C are settled before any code is written** (the tests below assert the chosen behavior
and may not be edited after they are shown failing). The design above assumes the recommended
answers, OD-A (d) and OD-C "keep the checkpoint tick"; a different answer is written into this plan
first.

The regression test is written and run first against step 1–2 with the **current** semantics
(`segmentStep % 50`), shown failing, and then the predicate is changed to the trainer step; the test
is not modified after it fails.

## Regression tests

Pure tests, `DrewsChessMachineTests/TrainingStepLogCadenceTests.swift` (new):

- `testEveryTrainerCadenceLineAfterAResumeIsADiagnosticsStep` — for resume offsets {0, 513, 36000, 7}
  and `batch_stats_interval` {0 (fallback), 7, 10, 25, 50, 100}, for segment steps 2…3,000: every step
  that is a tick **and is not a checkpoint tick** (`segmentStep % TrainingStepLogCadence.autosaveEvery
  != 0`) satisfies `ChessTrainer.isDiagnosticsStep(trainerStep: offset + segmentStep, …)`, and such
  steps occur at least once per `stepLineInterval(batchStatsInterval:)` trainer steps. Fails with the
  pre-fix predicate at offsets 513 and 7, and at interval 7 and 100 at every offset.
- `testStepLinesFallOnTheUninterruptedRunsTrainerSteps` — comparing only trainer-cadence ticks
  (excluding segment step 1 and every `segmentStep % TrainingStepLogCadence.autosaveEvery == 0`): the
  trainer steps logged by a run resumed at offset k equal the uninterrupted run's trainer-cadence
  ticks restricted to trainer steps > k + 1, over segment steps 2…3,000, at every offset and interval
  above.
- `testFirstSegmentStepIsAlwaysLogged` — segment step 1 is a tick at every offset.
- `testEveryCheckpointStepHasALine` — every `segmentStep % TrainingStepLogCadence.autosaveEvery == 0` is a tick at every offset.

Runner-level test (M2 of the review: the pure tests alone would pass a runner that kept passing the
segment step). Added as a **new** method in `DrewsChessMachineTests/ResumeEquivalenceTests.swift`,
whose corpus / start-model / config helpers are file-private (no existing test in it changes):

- `testAResumedRunsStatsRowsCarryDiagnosticsOnTheTrainerCadence` — train 13 steps, save, then
  `--resume-exact` from that file for 120 steps with `batch_stats_interval` 10 and a temporary
  `results.json` output (`CorpusReplayConfig.output`); every stats row after the segment's first that
  is not on a checkpoint step has a non-null `policy_entropy` and `cum_trainer_step % 50 == 0`, and
  such rows exist. Fails before the fix (rows at trainer steps 63, 113: `policy_entropy` null); after
  it, rows at 14 (the segment's first step), 50, 100.

Pure-function tests alone cannot be shown failing before the fix unless the predicate exists with the
current semantics first; the sequence is therefore: add the predicate and `isDiagnosticsStep` with
today's behavior (segment-step tick every 50, regardless of the interval) and wire both runners to it — a pure refactor, no behavior
change — add all tests, run them and record the failures here, then change the predicate.

## Validation

- While training runs are live: only the Xcode build of the app and test targets (`build_project`,
  compile only); no app launch, no test execution.
- After the live runs end: run `TrainingStepLogCadenceTests` and the new `ResumeEquivalenceTests`
  method on the refactored, pre-fix predicate (shown failing; output recorded here), apply the fix,
  run them again unchanged (pass), then `ResumeEquivalenceTests`, `ExactResumeTests`,
  `ExactResumeCompletionTests`, `RunObservabilityResumeTests`, `CorpusReplayFeederTests`,
  `CorpusReplayFailLoudTests`, `ReplayRunnerPreflightTests`, `TrainVsUciSessionTests` and every class
  under `DrewsChessMachineTests` whose name contains `TrainVsUci`, then the full suite before merging.
- Live check: from a saved checkpoint whose step is not a multiple of 10, a `--replay-corpus …
  --resume-exact` with a scratch `--out-model` stem (the enumerated-checkpoint guard refuses a stem
  that already holds reachable step files) and `--training-step-limit 1100`: every step line after
  the first that is not on segment step 1000 carries `pEnt=` values, at trainer steps that are
  multiples of 50, and there is a line at segment step 1000.
- Consumers: run `experiments/table_common.py`'s plies/game helper and the replay dashboard's
  `_metrics_at` / `_games_at` on that log; checkpoint steps resolve to the line on the step itself.

## Owner decisions

- **OD-A — `batch_stats_interval` that does not divide 50.** (a) refuse at run start on the CLI paths,
  naming both values; (b) log one `[REPLAY]` / `[VS-UCI]` warning at start and run; (c) leave as is;
  (d) make the trainer-cadence interval the smallest multiple of the diagnostics interval that is
  ≥ 50 (e.g. 100 for an interval of 100, 56 for 7), so every trainer-cadence line carries diagnostics,
  and log the interval used at start. Recommendation: (d) — it removes the symptom rather than
  reporting it, and with today's interval of 10 it is exactly 50.
- **OD-C — checkpoint-step lines.** Keep a line on every checkpoint step (recommended, above), or
  instead change the three consumers to key on `trainerStep`. The first keeps every existing log
  reader working; its checkpoint lines may lack diagnostics after a resume.
- **OD-B — no existing test edits are expected.** If any existing test pins the segment-step cadence,
  it is listed for approval before it is touched.

## Non-goals

- The GUI `[STATS]` cadence (time-based; unaffected).
- Changing which steps compute diagnostics, or the `[BATCH-STATS]` cadence.
- Re-logging the missing fields for C's segment-1 log (they were never computed).
