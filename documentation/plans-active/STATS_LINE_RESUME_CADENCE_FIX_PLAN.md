# Step-line cadence, save points and checkpoint names on the trainer step — plan

Status:
- 2026-10-05: proposed as a bug-fix plan; revised after an independent review (M1 checkpoint-step consumers, M2 runner-level test). Owner asked for the fix ("fix that bug").
- 2026-10-06: **redesigned on the owner's approved direction** (below). The bug analysis (The bug, Cause, Hypotheses) is unchanged from the reviewed version; the fix design, tests, validation and decisions are replaced. Not implemented.
- 2026-10-06: independent review of the redesign; fixes applied in place and listed under Review at the end. Reflects the owner's decision that alarm evaluations are decoupled from log lines. Not implemented.

Found while drafting `TRAINING_HEALTH_ALARMS_PLAN.md` (its OD-12, which this plan supersedes). That plan's P2 is gated on this one (its Part P).

---

## The bug

After an exact resume, every `[REPLAY]` (corpus replay) and `[VS-UCI]` (train-vs-UCI) step line,
and the matching `results.json` stats row, is missing the diagnostic fields (`pEnt`, `playedP`, `pW`
/ `pD` / `pL`, `vAbs`, `pLogitMean`, `vLogitMean` print as `--`; the row's fields are `null`).

Evidence: arm C's resumed segment (`~/Library/Logs/DrewsChessMachine/dcm_log_20261005-121841.txt`,
exact resume from trainer step 513, `[RESUME] EXACT`): 113 of 113 step lines have `pEnt=--`. Its first segment
(`dcm_log_20261005-090417.txt`) and the uninterrupted B run (`dcm_log_20261005-013235.txt`, 721 lines
through step 36,000; the only `--` line is step 1) are fine.

## Cause

Two cadences that are meant to coincide count different things:

- The step line is written when the **segment-local** step counter says so:
  `if step == 1 || step % logEvery == 0` with `logEvery = 50`, where `step` starts at 0 in every process
  (`CLI/CorpusReplayRunner.swift:1840,1904`; `CLI/TrainVsUciRunner.swift:744,764`).
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
   step. Then a resumed segment logs at the same trainer steps the uninterrupted run would have,
   which also makes a resumed run's log line up row-for-row with the uninterrupted run's. Which steps
   run the diagnostic graph stays a function of the trainer step alone, so a resumed run and the
   uninterrupted run run the same graph on every step — no effect on exact-resume determinism.
2. **Ask the trainer for diagnostics on the steps the runner logs** (a `trainStep(…, includeDiagnostics:)`
   request OR'ed with the trainer's own cadence). Rejected: it makes which steps run the diagnostic
   executable depend on where a process started, so a resumed run and the uninterrupted run would run
   different graphs on the same trainer steps — a possible bit-level divergence on the one path (corpus
   replay) whose resume is meant to continue exactly. It also adds a second source for "is this a
   diagnostic step".

The redesign keeps hypothesis 1. Its D2 below also forces diagnostics on a fixed set of trainer
steps; that set is a function of the trainer step only (never of where a process started), so it
is not hypothesis 2.

---

## The approved direction (owner, 2026-10-06)

1. **Step lines** — `[REPLAY]`, `[VS-UCI]` and the GUI's `[STATS]` — follow one shared cadence with one new training parameter for the interval (default 180 s, full parameter checklist):
   - dense at the start: every 50 trainer steps through trainer step 1,000;
   - then time-based: the line is written on the first diagnostics step after each deadline, so every step line after the first carries diagnostics;
   - plus a line at every overall trainer-step multiple of 1,000;
   - a segment's first step still gets a line;
   - the live `[LAYER-HEALTH]` readout keeps riding the step line.
2. **Saves** (corpus-replay autosave and enumerated checkpoints; train-vs-UCI enumerated checkpoints) at overall trainer-step multiples of 1,000, not segment multiples.
3. **Enumerated checkpoint files are named by the overall trainer step** (option a). Old files keep their names; nothing is renamed on disk; old files stay readable.
4. **`[BATCH-STATS]` decoupled from the diagnostics interval:** diagnostics stay every `batch_stats_interval` steps (cheap; alarms use them); the 72–75 KB `[BATCH-STATS]` line moves to the step-line cadence.
5. **Alarm evaluations are decoupled from log lines (owner, 2026-10-06; implemented by `TRAINING_HEALTH_ALARMS_PLAN.md`):** live alarm evaluations run every 50 trainer steps on every path, independent of `[STATS]` / `[REPLAY]` / `[VS-UCI]` lines. Nothing in this plan is an alarm tick: the step line is logging only. The live `[LAYER-HEALTH]` *log* readout rides the step line (item 1); the alarm monitor takes its own live read on its own 50-step cadence (alarms plan). Where the two fall on the same step (every fixed line step is a multiple of 50) one read can serve both (OD-15).

Measured motivation for item 4: in arm C's resumed log, 560 `[BATCH-STATS]` lines are 43,117,507 B (41.1 MB) of the 43,457,830 B (41.4 MB) file — 99.2%. Under this design that segment (5,603 steps in 3 h 15 min, ≈ 2.09 s/step, trainer steps 514–6,116) would write about 80 step lines (≈ 65 time lines, 10 dense lines 550–1,000, 5 more at 2,000–6,000, the first line), so about 6 MB of `[BATCH-STATS]`.

---

## Design

### D1. One step-line schedule, shared by the CLI runners and the GUI

New file `Training/TrainingStepLineSchedule.swift`: a pure value type, the single source of "is a step line due".

- Constants (the only place these numbers live):
  - `denseLineIntervalSteps = 50`, `denseLinesThroughTrainerStep = 1_000`;
  - `checkpointIntervalSteps = 1_000` — also the save interval (D3), so lines and saves cannot drift apart.
- Static, pure:
  - `isFixedLineStep(trainerStep:) -> Bool` — `trainerStep > 0` and either (`trainerStep ≤ 1,000` and a multiple of 50) or a multiple of 1,000.
  - `isCheckpointStep(trainerStep:) -> Bool` — `trainerStep > 0` and a multiple of 1,000. Every checkpoint step is a fixed line step (tested).
  - `firstFixedLineStep(after trainerStep:) -> Int` — the smallest fixed line step above it.
  - `firstCheckpointStep(after trainerStep:) -> Int` — the smallest checkpoint step above it (what the start lines name as the next save; `firstFixedLineStep` is not, e.g. 50 vs 1,000 on a fresh run).
- State: `lastObservedTrainerStep: Int?`, `lastLineElapsedSec: Double?`.
- `mutating func lineDue(trainerStep:elapsedSec:carriesDiagnostics:intervalSec:) -> Reason?`, `Reason` = `.segmentStart`, `.fixedStep`, `.interval`. Rules, in order:
  1. first call → `.segmentStart` (the segment's first step, as today; it may lack diagnostics);
  2. `trainerStep < lastObservedTrainerStep` (the GUI's promotion rewind of the trainer clock, `App/SessionController+Arena.swift:480-488`) → the step baseline moves to `trainerStep`; no fixed line for the rewind itself (rule 4 still applies to the same call). A fixed step re-crossed after a rewind gets its line again;
  3. a fixed line step `s` with `lastObserved < s ≤ trainerStep` → `.fixedStep`. The CLI observes every step, so this is "the step is a fixed line step"; the GUI polls, so it is "the first poll at or past it";
  4. `carriesDiagnostics` and `elapsedSec − lastLineElapsedSec ≥ intervalSec` → `.interval`.
  - Every non-nil result records `lastLineElapsedSec`: **any line restarts the interval** (OD-7). `lastObservedTrainerStep` is updated on every call.
  - `elapsedSec` is monotonic seconds since the caller's start (`ContinuousClock`), supplied by the caller so the tests drive time directly. `intervalSec` is passed on every call (the GUI reads the live parameter; the CLI its snapshot value). `precondition(intervalSec > 0)`; the parameter's range makes it ≥ 10.
  - The time rule applies in the dense phase too (OD-8); it adds lines there only when 50 steps take longer than the interval.
  - Nothing reads `lineDue` to decide an alarm evaluation; the alarms plan's 50-step evaluation tick is its own rule on the trainer step (approved direction item 5).
- CLI wiring (both runners): after each completed step, read `trainer.completedTrainSteps` once (`observedSteps`, today read inside the block at `CorpusReplayRunner.swift:1909` / `TrainVsUciRunner.swift:767`), call `lineDue(…, carriesDiagnostics: timing.hasDiagnostics, intervalSec: p.parameters.stepLineIntervalSec)`, and write the step line block when it is non-nil. The block keeps its content and order (step line, then — new — `[BATCH-STATS]` (D6), then the live `[LAYER-HEALTH]` lines, then the `results.json` row), and stays **before** the save block. Per-step order in the loop: step-line block (when due) → the alarms plan's live evaluation (when its 50-step tick is due; added by that plan's P2) → save block. Every checkpoint step is both a fixed line step and a multiple of 50, so at every 1,000-step save the line, the live readout and the live evaluation all precede the save's checkpoint pass (alarms plan R0) — the evaluation by its position in the loop, not by riding the line.
- `[REPLAY]` / `[VS-UCI]` field format is unchanged (`trainerStep=`, `--` = not measured; alarms plan D4). No reason field is added.
- Each runner logs the cadence once at start, e.g. `[REPLAY] step lines: every 50 trainer steps through 1000, every 1000, and the first diagnostics step ≥ 180 s after the previous line; saves at trainer-step multiples of 1000 (next: 1000)`. "next" is `firstCheckpointStep(after: <start trainer step>)`.
- The `results.json` row keeps `steps: step` (segment step) and its `cum_trainer_step`; only which steps get rows changes.

### D2. The trainer computes diagnostics on every fixed line step

So that every line after the first carries diagnostics for **any** `batch_stats_interval` (OD-4):

- `ChessTrainer.diagnosticsInterval(batchStatsInterval:)` — today's rule (`interval > 0 ? interval : diagnosticsFallbackInterval`).
- `ChessTrainer.isDiagnosticsStep(trainerStep:batchStatsInterval:)` — `trainerStep % diagnosticsInterval(…) == 0 || TrainingStepLineSchedule.isFixedLineStep(trainerStep:)`.
- `ChessTrainer.isBatchStatsStep(trainerStep:batchStatsInterval:)` — `batchStatsInterval > 0 && (trainerStep % batchStatsInterval == 0 || isFixedLineStep(trainerStep:))`.
- Called from the existing site (`ChessTrainer.swift:4649-4657`) for `includeDiagnostics` and `isStatsStep`.
- **At today's interval 10 (and the fallback 10) the set of diagnostic steps and batch-stats steps is unchanged**: every fixed line step is a multiple of 10. Tested over trainer steps 1…5,000. So no numerics change for any run at the default, and `BehaviorFingerprint` (which sets `batchStatsInterval = 0`, `Training/BehaviorFingerprint.swift:263`) is unaffected.
- The rule depends on the trainer step only, so a resumed run and the uninterrupted run run the same graph on every step (hypothesis 2's objection does not apply).
- Time-based lines need no forcing: they wait for the next diagnostics step (D1 rule 4).
- The replay sampler draws the same positions whether or not a step collects batch metadata (the metadata pointers are only copied into, `Training/ReplayBuffer.swift:1873`), so forcing a batch-stats step changes no draw.

### D3. Saves at overall trainer-step multiples of 1,000

- Corpus replay: the autosave condition `step % autosaveEvery == 0` (`CorpusReplayRunner.swift:1979`) becomes `TrainingStepLineSchedule.isCheckpointStep(trainerStep: observedSteps)`. The local `autosaveEvery` (`:1110`) is removed. The final save is unchanged (every clean exit, any step).
- Train-vs-UCI: the enumerated-checkpoint condition (`TrainVsUciRunner.swift:847`) likewise; the local `autosaveEvery` (`:306`) is removed; the final enumerated copy (`:873`) is written when the final trainer step is not a checkpoint step and the segment trained at least one step (D4). The periodic session save (time-based, `:850-852`) is unchanged.
- Consecutive saves are 1,000 trainer steps apart, and the first is within 1,000 of the start (alarms plan D6 holds): e.g. a resume at 513 saves at 1,000, 2,000, ….
- A resumed run and the uninterrupted run now save at the same trainer steps, so their enumerated files at the same name describe the same point of training.

### D4. Enumerated checkpoint names carry the trainer step

Name shape (OD-2): **`<base>-<tag>-step<T>`**, where `T` is the trainer step at the save (`trainer_completed_steps`, which equals the record's `cum_trainer_step` on a trainer file). The `-seg<k>` marker is no longer written.

- Every segment of a lineage run continues one series under its stem: segment 0 of a fresh run or a branch (trainer clock starts at 0, `CorpusReplayRunner.swift:1479-1483`) gets the names it gets today, with one exception: a corpus-replay run that trains no step no longer writes `<stem>-replay-step0` (OD-3 applies to every segment; today's final save writes it, `:2009`, while train-vs-UCI already skips it, `TrainVsUciRunner.swift:873`). An exact resume writes `<stem>-replay-step37000` instead of `<stem>-replay-seg1-step1000`.
- The segment's start trainer step is known at pre-flight, before the trainer is built: `resumeSnapshot?.schedule.completedTrainSteps ?? 0` (`CorpusReplayRunner.swift:999-1009`, `TrainVsUciRunner.swift:183-223`; a branch starts its clock at 0). After `restoreExactly` / the fresh build, the runner checks `trainer.completedTrainSteps` equals it and throws (an internal error naming both) otherwise, so the pre-flight scan and the writes cannot use different bases.
- Why drop the marker rather than keep `-seg<k>-step<T>`: a kept marker would give one name shape two meanings (segment step in files already on disk, trainer step in new ones), undecidable from the name. Dropping it makes a resumed segment's files sort and glob with the rest of the run, and makes the probe loop's per-segment glob unnecessary for new runs.
- `EnumeratedCheckpointNaming` (`CLI/CorpusReplayRunner.swift:389-532`):
  - `init(rollingOutputURL:runTag:)` — the `segmentIndex` property and parameter go (`:425-438`).
  - `fileName(trainerStep:)`, `url(trainerStep:)`, `trainerStep(ofFileName:)` — labels renamed so a caller cannot pass the segment step by accident (OD-12a).
  - `step(ofEnumeratedFileNameUnderAnyStem:)` (`:478-525`) keeps recognizing the legacy `-seg<k>-step<N>` shape (a private reading, never a writer), so the rolling-output rule (c) still refuses an `--out-model` named like an old segment file. Today it confirms each reading by building an `EnumeratedCheckpointNaming(…, segmentIndex: k)` (`:513-518`); with the property gone, the legacy confirmation moves into a `private static` name builder used only by this parser (`<base>-<tag>-seg<k>-step<N>` / `<stem>-seg<k>-step<N>`), and `TrainVsUciSession.enumeratedNamingBase`'s stem check (`CLI/TrainVsUciSession.swift:257-260`) keeps using this parser unchanged.
- `EnumeratedCheckpointWriter` (`:566-595`): `write(_:trainerStep:)`; behavior unchanged (never over another file; replaces only its own earlier save, by identity).
- `TrainerOutputFileGuard`:
  - `reachableEnumeratedCheckpoints(naming:segmentStartTrainerStep:stepLimit:)` (`:726-739`) — a segment writes only at trainer steps **above** its start: `start < T ≤ start + stepLimit`, or `T > start` with no limit (`--training-step-limit` stays segment-local, `CorpusReplayRunner.swift:1854`, `TrainVsUciRunner.swift:747`). `start + stepLimit` is computed with `addingReportingOverflow`; an overflow means every step above the start is reachable (no step exceeds `Int.max`).
  - `requireNoReachableEnumeratedCheckpoints(naming:segmentStartTrainerStep:stepLimit:)` (`:743-754`): stageability is checked on the name at `start + stepLimit` (or `Int.max`); the error names the reachable range from `start + 1`.
  - New `enumeratedCopyIsWritten(trainerStep:segmentStartTrainerStep:) -> Bool` — `trainerStep > segmentStartTrainerStep`. A segment that trained no step writes no enumerated copy (OD-3): its state is the start model's, and at the start step the name is usually the start model's own file (a resume from `…-step36000` stopped before any step would otherwise collide with it). The runner logs `[REPLAY] enumerated checkpoint not written: the segment trained no step (trainerStep=<T>, the start state)`; the rolling file (corpus replay) and the session folder (train-vs-UCI) are written as today.
  - Consequence: a resume from a stem's own step file keeps the stem (files at or below the start are unreachable). A second resume of an earlier file into the same stem, after the first resume wrote above that point, is refused at pre-flight and names the files — a new `--out-model` / `--checkpoint-stem` is needed (Risks).
- Runner call sites: `CorpusReplayRunner.swift:1123-1134` (naming setup; the start line, today `naming.url(step: autosaveEvery)` at `:1131` / `TrainVsUciRunner.swift:318`, names `firstCheckpointStep(after: start)`'s file), `:1643-1660` (enumerated write uses `snapshot.schedule.completedTrainSteps`); `TrainVsUciRunner.swift:307-321`, `:646-676` (`save.snapshot.schedule.completedTrainSteps`).

### D5. `training_step` keeps its meaning; how files are read

Recommended (OD-1): **`training_step` stays what it is** — the writing segment's step on corpus-replay and train-vs-UCI trainer files (`CorpusReplayRunner.swift:1551-1556`, `TrainVsUciRunner.swift:518-521`, `:537`), equal to the record's `segment_local_step`. The trainer step already has its single source: `trainer_completed_steps` / `cum_trainer_step`.

Why not make it the trainer step: one header key would then mean two things with no marker telling which, and these readers depend on today's meaning:
- `replay.py` `lineage_checkpoints` refuses a file whose `training_step` differs from `segment_local_step` (`documentation/dashboards/replay.py:617-620`) and `lineage_cells` checks it against the row's `meta_step` (`:641-643`);
- the tracker's step axis, `cum_step = cumstep_base + training_step` (CLAUDE.md "Run tracking"), and `discover_enum_stems` (`:718`);
- `TrainerOutputFileGuard.rollingOverwriteVerdict` compares the rolling file's and the start model's `training_step` (`CorpusReplayRunner.swift:626-661`);
- the Lichess bot lineage follower ranks a line's files by `training_step` and never across segments (`LICHESS_BOT_FOLLOW_LINEAGE_PLAN.md:63,123`);
- `ExactResumeTests.swift:276-277` pins "`training_step` stays the segment-local value it was given".

So a new resumed segment's file is named by `T` while its header says `training_step = T − start`. Identification stays by header (`model_id` + `training_step`, or the lineage record), never by name. Reading old files: nothing changes — their headers are read exactly as today; their names are recognized by the any-stem parser (D4) and by the tooling's legacy branches (D9).

### D6. `[BATCH-STATS]` rides the step line

Recommended (OD-5): no separate parameter.

- `ChessTrainer` keeps computing the summary on every batch-stats step (D2) — `lastBatchStatsSummary` feeds the GUI's `bufUniq` (`App/SessionController+Training.swift:1778`) and the GUI `results.json` `batch_stats` (`:1948`); `[SAMPLER]` and `[LEGAL-COST]` keep their batch-stats cadence — but no longer logs the line (`ChessTrainer.swift:4702` removed).
- CLI runners: right after the step line, `[BATCH-STATS] <json>` when `trainer.lastBatchStatsSummary?.step == observedSteps` (the summary of this very step). With `batch_stats_interval` > 0 that holds on every fixed line and every time line (both are batch-stats steps); with 0 there is no `[BATCH-STATS]` at all, as today.
- GUI ticker: after each `[STATS]` line, the latest summary when its `step` differs from the last one logged (its JSON carries its own `step`). "Differs", not "is newer": a promotion rewinds the trainer clock (D1 rule 2), after which every new summary's step is below the last logged one until training passes the pre-rewind step, and a "newer" rule would log no `[BATCH-STATS]` for that whole stretch.
- One pure formatter, `BatchStatsLogLine.line(summary:lastLoggedStep:)`, used by all three.
- `lastBatchStatsSummary` / `lastBatchStatsUniquePct` (`ChessTrainer.swift:1986,1994`) are written on the trainer queue and read from the GUI ticker task with no lock today (the doc says so). This change makes the ticker read the summary on every line, so both move into one `SyncBox` (OD-14), per the project's lock discipline.
- `batch_stats_interval`'s description (`Training/TrainingParameters.swift:1010`) becomes: compute the per-batch statistics and the graph diagnostics every N steps (0: no batch statistics; diagnostics then every 10 steps); the `[BATCH-STATS]` line is written with the step lines. The id is unchanged (OD-11). The `SessionCheckpointState.batchStatsInterval` doc (`Persistence/SessionCheckpointFile.swift:533-537`) and the `ChessTrainer.batchStatsInterval` doc (`:1978-1981`) follow.

### D7. The new parameter: `step_line_interval_sec` (OD-10), full checklist

1. **Declare** in `Training/TrainingParameters.swift` next to `BatchStatsInterval` (`:1008-1017`): `@TrainingParameter(name: "Step Line Interval (sec)", description: …, default: 180.0, range: 10.0...86400.0, category: "Observability", liveTunable: true, absentValue: .currentSetting) public enum StepLineIntervalSec`. `.currentSetting` because it changes no training math (logging only; D2's forced steps are fixed constants, not this value). Add to `allKeys` (`:2479`).
2. **Singleton:** stored property with the `commitAssignment` `didSet` (beside `:1610`), `init` read (`:1719`), `collectValues` (`:1815`), `applyOne` (`:1998`), snapshot accessor (`:1466`).
3. **`parameters.json`:** confirm `step_line_interval_sec` in `--show-default-parameters`, and the `--create-parameters-file` → edit → reload round trip.
4. **Session save/load:** `stepLineIntervalSec: Double?` in `SessionCheckpointState` (`Persistence/SessionCheckpointFile.swift`, beside `:537`); pass it in `buildCurrentSessionState` (`App/SessionController+Checkpoint.swift:1260`) and in `TrainVsUciSession.sessionState` (`CLI/TrainVsUciSession.swift:175`); `resume.restore(StepLineIntervalSec.self, saved: rs.stepLineIntervalSec, into: \.stepLineIntervalSec)` in `SessionParameterResume.applyGuiSession` (`App/SessionParameterResume.swift:144`).
5. **`results.json`:** carried in the lineage parameter snapshot like every key; it is not a metric, so no recorder field.
6. **Runtime log:** the CLI start lines (`CorpusReplayRunner.swift:1068`, `TrainVsUciRunner.swift:388` gain `stepLineSec=`) and D1's cadence line; the GUI `[STATS]` `cfgStr` (`App/SessionController+Training.swift:1638`) gains `stepLineSec=`.
7. **UI:** Sessions tab, beside the KL-probe interval (`App/UpperContentView/TrainingSettingsPopover.swift`, `SessionsTab` `:1251`, KL row `:1342`); binding + validation in `TrainingSettingsPopoverModel.swift` beside `klProbeIntervalText` (`:199`, `:362`, `:492`, `:570`, `:1531`), with a `[PARAM] stepLineIntervalSec: a -> b` log on change.
8. **Live tunability:** the GUI ticker reads `TrainingParameters.shared.stepLineIntervalSec` on every poll (it already hops to the main actor in `logOne`). CLI paths read the snapshot once at start, like every CLI parameter.
9. **Not a rename.** `TrainingParametersTests.test_registry_size` +1 (TE-1).

### D8. GUI `[STATS]` ticker on the shared schedule

Today (`App/SessionController+Training.swift:1390-2183`): one `[STATS]` line per new step until the trainer's cumulative step count reaches 500 (`UpperContentView.bootstrapStatsStepCount`, `App/UpperContentView/UpperContentView.swift:400-406`), legalMass and live `[LAYER-HEALTH]` refreshed every 25 steps in that window (`:1426`, `:1435`), then one line every 60 s (`:1419`, `:2143-2182`). Its values are 512-step rolling means (`SessionController.rollingLossWindow`, `App/SessionController.swift:896`), so they carry diagnostics on every line after the first diagnostic step.

Recommended (OD-6): **merge** into the shared schedule.
- One loop polling every 50 ms (today's bootstrap poll): read the trainer step from the stats box, call `lineDue(trainerStep:elapsedSec:carriesDiagnostics: trainingSnap.rollingPolicyEntropy != nil, intervalSec: <live parameter>)`, and on a due line: refresh legalMass, `logOne`, `[BATCH-STATS]` (D6), live `[LAYER-HEALTH]`. The 25-step strides and both loops go; `bootstrapStatsStepCount` is removed. `lineDue` is first called once the box's step is above 0 (today's `steps > 0` guard, `App/SessionController+Training.swift:2089`), so the first line is never at step 0; a resumed session's box is seeded with the saved step (`:87-92`), so its first line comes at the first poll, as today.
- Also moved by this cadence (stated so nothing relies on the old one): the legal-mass refresh behind the chart's legal-entropy trace (`realLastLegalMassSnapshot`, `:2107`, `:2166`) refreshes on line ticks, so at steady state every 180 s instead of 60 s; the `[ALARM] policy entropy …` log line written inside `logOne` (`:2057-2069`) follows the line cadence. The GUI's streak alarms (`TrainingAlarmController.evaluate(from:)`, heartbeat, `App/SessionController+Heartbeat.swift:646`) and the alarms plan's evaluations do not ride `[STATS]` and are unaffected.
- The time rule does not require a new step, so lines keep coming while training is paused by an arena, as today's 60 s lines do.
- Why merge: per-step lines of a cumulative rolling mean add little over 50-step lines (the mean over the first k steps moves by about 1/k per step); one rule for all paths is what was asked; a GUI resume already skips the bootstrap today (the box is seeded with the saved cumulative step, `:87-92`), which the trainer-step dense rule reproduces.
- Kept: the first line at the first observed step, so `documentation/dashboards/selfplay.py`'s fresh-launch test (first line ≤ 500, `RESET_FRESH_STEP`, `:60-68`) still holds; its comments (`:60-68`, `:323-325`) are updated.
- Changed for the GUI: steady-state lines every 180 s instead of 60 s by default (Risks; the parameter can be set to 60).

### D9. Tooling

All changes read old and new files by explicit rules; no existing data file is rewritten.

- **`experiments/probe_record.py`** (`build_record`, `:67-107`): the name's step must equal the header's `training_step` (the basis every existing record has) **or** its `trainer_completed_steps` (the new basis). `step_basis` is `"segment_step"` whenever the name's step equals `training_step` — every legacy file and every segment-0 file, where the two numbers agree — and `"trainer_step"` only when the name's step equals `trainer_completed_steps` and differs from `training_step` (a resumed segment on this build). A file with no `trainer_completed_steps` can only be `"segment_step"`; nothing is inferred from a missing key. The record gains `trainer_step` (the header's `trainer_completed_steps`, `null` when absent) and `step_basis`, both placed **after** the existing leading keys (`step`, `training_step`, `model_id`, `parent_model_id`), so `test_good_probe_becomes_a_record_with_identity`'s key-order assertion (`documentation/dashboards/tests/test_tooling.py:154`) holds unmodified. A probes file must hold one basis; a record without `step_basis` counts as `"segment_step"`, so a segment-0 probes file started before P4 keeps accepting new records (a live run's probes are not broken by the tooling change); a `"trainer_step"` record into a file holding `"segment_step"` records is refused (exit 4) — in practice already excluded by the one-`model_id` rule, since each segment mints its own `model_id`. `load_probe_points` (`:109-137`) compares `step` with `trainer_step` for `"trainer_step"` records and with `training_step` otherwise; its existing check (`:130-131`) is narrowed to the `"segment_step"` records.
- **`experiments/probe_loop.sh`:** new `PROBE_ABOVE_STEP=<the segment's start trainer step>` (default 0): files at or below it are skipped, each logged once (like the step-limit skip, `:102-105`) — a resumed segment's probes then start above its start, and earlier segments' files under the same stem are never probed into this segment's file. The optional `step limit` argument (`:6`, `:10`, `:102`) is compared with the name's step, so for files on this build it is a **trainer** step (a resumed segment's limit is its start plus `--training-step-limit`); the header says so. A segment-0 probe of a stem that a later segment also wrote into needs that limit (its own last trainer step), otherwise the pass reaches the later segment's files and stops on the other-run check (exit 4). `PROBE_SEGMENT` (`:23-26`, `:49-52`, `:100-101`) stays, documented as **legacy only** (`-seg<k>-step<N>` names written before this change), and the two are refused together.
- **`documentation/dashboards/replay.py`:**
  - `freeze` / `enum_path` (`:791-810`, used by `track`): the enumerated file is looked up at the rolling file's trainer step (`trainer_completed_steps`) and accepted only when its header's `model_id` and `training_step` equal the rolling file's; then the legacy name at the segment step under the same check; else the existing `-frozen` copy. Today a resumed segment's name at `meta` could hit segment 0's file of the same number under a shared stem.
  - `probe_backfill` glob scan (`:947-955`): a file whose header `training_step` differs from the name's step is a failure ("named by trainer step; give its segment a `segment_id` with derive-registry"), never filed at `cumstep_base + name step`. Lineage-era segments are found by `segment_id` (`lineage_checkpoints`, `:603-623`), which reads headers and is unchanged.
  - `discover_enum_stems` (`:694-720`): files whose header `training_step` differs from the name's step are skipped and counted (they belong to `segment_id` segments).
  - Comments at `:526-534`, `:541-563`, `:916-922`, `:283-296` (the "every ~60 s" note) updated. `_metrics_at` / `_games_at` need no change: a checkpoint's `meta` (= `training_step`) now always has a line at exactly that segment step.
- **`documentation/dashboards/vsuci.py`** (`build`, `:138-165`): rows at the segment steps whose line has `trainerStep % 1000 == 0`; a log whose lines carry no `trainerStep=` keeps today's rows at `step % 1000 == 0`. The registered run (`sf100sl100`, three logs) has no `trainerStep=` in any line (checked 2026-10-06), so its CSV is unchanged. `STEP_RE` (`:38-41`) does not match today's `[VS-UCI]` line at all: it requires `playedP=… gNorm=` adjacent, while the current line has `pLogitMean=… vLogitMean=…` between them (`CLI/TrainVsUciRunner.swift:773-775`). So `STEP_RE` gains optional `pLogitMean=` / `vLogitMean=` fields (`[-\d.]+|--`) between `playedP=` and `gNorm=`, and a separate `trainerStep=(\d+)` search on the matched line (it trails `mom=` and the optional `lrCyc…`, so a fixed-position capture would not reach it). The new test's log uses the current line format verbatim; without this the test could only pass on a fixture shaped like logs no current build writes. The idle-removal rule (`parse_segment`, `:52-57`) bounds an interval by `Δsteps × ms` with the line's single-step `ms`; with time lines hundreds of steps apart that bound rests on one step's duration (stated in the module doc, not changed).
- **`experiments/table_common.py`:** `buffer_plies_per_game` (`:69-94`) is left as it is (the segment-step grid) because three dated experiment tables call it (`20261001-se-fc1-leaky`, `20261002-label-smoothing-C`, `20261002-noSE-noReZero`); a new `buffer_plies_per_game_by_trainer_step(log_name)` keys on `trainerStep=` multiples of 1,000, for write-ups of runs on this build (OD-13).
- **`experiments/20261005-lr-schedule-ab/bn_liveness.py`** (the live LR experiment's own script): it requires a file's name step to equal its `segment_local_step` (`:226-235`) and lists the A/B/C arms' legacy `-replay-seg1-step` prefixes (`:53-61`). Existing files keep reading. A resumed segment of any arm on the P2 build would be refused by it; if the experiment resumes an arm on that build, the script needs the same two-basis rule as `probe_record.py` — listed here so it is not found by a failure (no change made by this plan without the owner's word, OD-16).
- **`documentation/dashboards/ckpt_inventory.py`**: header-based already; its docstring (`:5-12`, "named … using the SEGMENT-LOCAL step number") is updated with D10.
- **`registry.json` / `cumstep_base`:** no change. Bases stay the segment's start on the registry axis; `cum_step = cumstep_base + training_step` stays true because `training_step` keeps its meaning (D5).

### D10. Documentation

At implementation (not now; this plan edits only itself):
- `CLAUDE.md`: `:39` (`[STATS]` cadence; `[LAYER-HEALTH]` "at the stats cadence"), `:54` (checklist item 6 still names `[BATCH-STATS]` as a value-visible tag — now only alongside step lines), `:185` (`[LAYER-HEALTH]` rides every step line; the 25-step bootstrap note goes), `:200` (names by trainer step; `-seg<k>` names are legacy; `PROBE_ABOVE_STEP`; reachable range above the start), `:202` (`training_step` stays segment-local; names carry the trainer step).
- `documentation/UCI.md`: `:209-218` (every trainer-step multiple of 1,000; names by trainer step; a resumed segment may keep its stem) and `:286` (step-line cadence).
- `experiments/probe_loop.sh` header, `experiments/probe_record.py` docstring, the dashboards' comments (D9).
- `CHANGELOG.md` entry.
- Adjacent plans' text that cites the old cadence or names is revised by whichever lands second (see Order with adjacent plans).

### Determinism and exact resume

- Logging (D1, D6, D8) never touches trainer, optimizer, buffer or RNG state.
- D2's forced steps depend on the trainer step only; at interval 10 they add none.
- Saves (D3) export only; they draw nothing (`exportResumeSnapshot`, `samplerState()`, `dropoutStreamState()` are reads; probe isolation holds). Moving them from segment to trainer multiples changes which steps are saved, not what is trained.
- A resume that starts mid-interval is already exact: arm C resumed at trainer step 513 (a final save, not a multiple of anything) and logged `[RESUME] EXACT` (`dcm_log_20261005-121841.txt`). The feed position, sampler and dropout streams are saved at whatever step a save lands on.
- The first save after a mid-interval resume is the next trainer multiple of 1,000; validated live (V-2).

### Order with adjacent plans

- **`TRAINING_HEALTH_ALARMS_PLAN.md`** (its Part P requirements):
  - R0 — at a save step the live readout and evaluation run before the save's checkpoint pass: **holds** at every 1,000-step save. The line block precedes the save and every checkpoint step is a fixed line step (the logged readout); the alarms plan's 50-step evaluation sits between the line block and the save block (D1 CLI wiring), and every checkpoint step is a multiple of 50 (the evaluation). The final save, at an arbitrary step, has no line before it, as today (OD-9); whether an evaluation precedes it is the alarms plan's rule, not this plan's.
  - D6 — saves 1,000 trainer steps apart, the first within 1,000 of the start: **holds** (D3).
  - D4 — `[REPLAY]` / `[VS-UCI]` field format: **kept** (D1).
  - **Conflicts with the owner's decoupling decision, in the alarms plan's text as of 2026-10-05 (its file, not edited here):** it still ties live evaluations to step-line ticks — the evaluation-cadence row (`:63`, CLI "every step-line tick the cadence plan defines"; GUI "every 25 session steps for the first 500, then every 60 s `[STATS]` emit"), R0 (`:202`, "Live evaluations ride whatever step-line ticks the cadence plan defines"), the sustain-span note (`:217`), the pending-window note (`:479`), the `[HEALTH] check` cadence ("at the first `[STATS]` emit past it in the GUI", `:523`), and P2 wiring into `TrainingStepLogCadence.isStepLineTick` (`:839`). Under the decision these become "every 50 trainer steps, on every path, independent of the step line". That plan's revision owns those edits; this plan defines no tick for it and exports no API it needs (`TrainingStepLineSchedule.lineDue` is logging only). Its own live `[LAYER-HEALTH]` read at 50-step ticks coincides with this plan's logged readout on every fixed line step (OD-15).
- **`HPARAM_RECORDING_PLAN.md`:** its V2/V4 commands name `$S/v2-replay-seg1-step200.safetensors` (`:1208`, `:1226`); under D4 that file is `$S/v2-replay-step400.safetensors` (segment 1 starts at trainer step 200 and ends at 400). Whichever plan lands second fixes them. Gap 10's applicability table gains `step_line_interval_sec` on every path.
- **`LICHESS_BOT_FOLLOW_LINEAGE_PLAN.md`:** unaffected under OD-1's recommendation (it relies on `training_step` being segment-local).
- **`HEAD_ACTIVATIONS_PLAN.md`:** edits other parts of the runners; whichever lands second rebases.

---

## Sites touched

| File | Lines (current `main`) | Change |
|---|---|---|
| `Training/TrainingStepLineSchedule.swift` | new | D1 |
| `Training/ChessTrainer.swift` | `:1195-1201`, `:1978-1994`, `:4645-4657`, `:4688-4703` | D2 static rules; D6 line removed; summary under `SyncBox` |
| `CLI/CorpusReplayRunner.swift` | `:389-532`, `:566-595`, `:726-754` | D4 naming, writer, guard |
| | `:1068`, `:1110`, `:1123-1134`, `:1643-1660` | D7 log; D3 constant removed; D4 setup and write |
| | `:1838-1990` | D1 line block; D6 `[BATCH-STATS]`; D3 autosave condition |
| `CLI/TrainVsUciRunner.swift` | `:306-321`, `:388`, `:646-676`, `:744-855`, `:869-875` | same as corpus replay |
| `CLI/TrainVsUciSession.swift` | `:175` | D7 step 4 |
| `Training/TrainingParameters.swift` | `:1008-1017`, `:1466`, `:1610`, `:1719`, `:1815`, `:1998`, `:2479` | D6 description; D7 |
| `Persistence/SessionCheckpointFile.swift` | `:533-537` | D7 step 4; doc |
| `App/SessionParameterResume.swift` | `:144` | D7 step 4 |
| `App/SessionController+Checkpoint.swift` | `:1260` | D7 step 4 |
| `App/SessionController+Training.swift` | `:1390-2183` (constants `:1413-1435`, loops `:2081-2182`, `cfgStr` `:1638`) | D8, D6 |
| `App/UpperContentView/UpperContentView.swift` | `:400-406` | `bootstrapStatsStepCount` removed |
| `App/UpperContentView/TrainingSettingsPopover.swift`, `…Model.swift` | `:1251`, `:1342`; `:199`, `:362`, `:492`, `:570`, `:1531` | D7 step 7 |
| `experiments/probe_record.py`, `experiments/probe_loop.sh`, `experiments/table_common.py` | see D9 | D9 |
| `documentation/dashboards/replay.py`, `vsuci.py` (incl. `STEP_RE`), `selfplay.py` and `ckpt_inventory.py` (comments) | see D9, D8 | D9 |
| `experiments/20261005-lr-schedule-ab/bn_liveness.py` | `:53-61`, `:226-235` | only under OD-16 |
| `CLAUDE.md`, `documentation/UCI.md`, `CHANGELOG.md` | see D10 | D10 |

---

## Tests

### New

Pure, `DrewsChessMachineTests/TrainingStepLineScheduleTests.swift` (new):
- `testFixedLineStepsAreEvery50ThroughStep1000ThenEvery1000` — and `0` is never one.
- `testEveryCheckpointStepIsAFixedLineStep` — trainer steps 1…100,000.
- `testEveryLineAfterTheFirstCarriesDiagnosticsAfterAResume` — the CLI simulation: resume offsets {0, 7, 513, 999, 36,000}, `batch_stats_interval` {0, 7, 10, 25, 50, 100}, segment steps 1…3,000, step durations {0.1 s, 2.07 s, 9 s}, interval {60, 180}: `carriesDiagnostics` = `ChessTrainer.isDiagnosticsStep(…)`; every line but the first has diagnostics; every fixed line step in the range has a line; for every two consecutive lines, both bounds hold: at most 1,000 trainer steps apart (50 while both are at or below trainer step 1,000), and at most `interval + diagnosticsInterval × stepDuration` seconds apart (the time line waits at most one diagnostics interval past its deadline).
- `testResumedAndUninterruptedRunsLogTheSameFixedSteps` — the fixed lines of a run resumed at k equal the uninterrupted run's fixed lines above k + 1.
- `testTheIntervalLineWaitsForADiagnosticsStep`, `testAnyLineRestartsTheInterval`, `testTheFirstObservationIsALine`.
- `testAPolledScheduleLinesUpOnTheFirstObservationPastEachFixedStep` — GUI-style observations (3, 47, 52, 980, 1,004, …).
- `testATrainerClockRewindIsNotAFixedLine` — the promotion rewind.
- `testTheTrainerComputesDiagnosticsOnEveryFixedLineStep` — intervals 7 and 100.
- `testTheDefaultIntervalComputesDiagnosticsOnTheSameStepsAsBefore` — intervals 10 and 0: `isDiagnosticsStep` ≡ `s % 10 == 0`, and `isBatchStatsStep` ≡ `s % 10 == 0` (interval 10) / never (0), over 1…5,000.

Pure, `DrewsChessMachineTests/TrainerStepCheckpointNamingTests.swift` (new):
- `testALaterSegmentsFilesContinueTheRunsSeries` — no `-seg` marker; `fileName(trainerStep: 37000)`.
- `testLegacySegmentNamesAreStillRefusedAsAnOutModel` — `checkRollingOutput` with `…-replay-seg1-step1000.safetensors`.
- `testStepFilesAtOrBelowTheSegmentStartAreNotReachable`, `testTheStepLimitCountsFromTheSegmentStart`, `testTheLongestReachableNameIsAtTheStartPlusTheLimit`, `testAStartPlusLimitOverflowReachesEveryStepAboveTheStart`.
- `testAResumeFromTheStemsOwnStepFileKeepsTheStem`.
- `testNoCopyIsWrittenForASegmentThatTrainedNoStep` (`enumeratedCopyIsWritten`).
- `testBatchStatsLineOnlyForThisStepsSummary` (`BatchStatsLogLine`).

Runner level, new methods in `DrewsChessMachineTests/ResumeEquivalenceTests.swift` (its helpers are file-private; a new private helper builds a config with `output`, `enumerateCheckpoints` and the parameter override; no existing method or helper changes):
- `testAResumedRunsStatsRowsCarryDiagnosticsOnEveryRowAfterTheFirst` — train 13 steps, save, `--resume-exact` for 120 steps, `batch_stats_interval` 10, `step_line_interval_sec` 86,400 (no time lines), temporary `results.json`. Rows at `cum_trainer_step` exactly {14, 50, 100}; every row but the first has non-null `policy_entropy`. Fails before the fix (rows at 14, 63, 113; `policy_entropy` null at 63 and 113).
- `testAResumedRunSavesAndNamesItsCheckpointsByTrainerStep` — train 990 steps with `--enumerate-checkpoints` (final `S-replay-step990`), then `--resume-exact` from that file for 30 steps into the same `--out-model`. Expect `S-replay-step1000` (header `training_step` 10, `trainer_completed_steps` 1,000) and `S-replay-step1020` (20 / 1,020), no `-seg` file, a stats row at `cum_trainer_step` 1,000 with diagnostics, `[RESUME] EXACT`. Fails before the change (no save at 1,000; the final file is `S-replay-seg1-step30`). Its run time is measured in P1 and recorded here; it is a correctness test and is not gated.

Python (`documentation/dashboards/tests/`):
- `test_tooling.py`: new `ProbeRecordTests` methods for a trainer-step-named resumed file (accepted, `step_basis` `trainer_step`), a legacy segment-step file (accepted, `segment_step`), a mixed-basis probes file (refused), `load_probe_points` on each basis; a `probe_loop.sh` test that `PROBE_ABOVE_STEP` skips earlier segments' files and that `PROBE_SEGMENT` with `PROBE_ABOVE_STEP` is refused.
- `test_lineage.py` / `test_replay_probe.py`: new methods — `freeze` takes the trainer-step file only when its header matches and never segment 0's file of the same number; the glob scan reports a trainer-step-named file as a failure; `discover_enum_stems` skips it.
- A new `vsuci.py` test: rows on `trainerStep` multiples of 1,000 for a log that carries the field; today's rows for one that does not.
- `table_common.buffer_plies_per_game_by_trainer_step` on a two-segment synthetic log.

### Existing-test edits — each needs the owner's approval

- **TE-1** `TrainingParametersTests.test_registry_size` (`:19-25`): +1 (85 → 86 if this lands first; one more than whatever the count is when it lands — the alarms plan's approved edit makes it 96 → 97 if that lands first).
- **TE-2 (mechanical, no assertion changes)** remove `segmentIndex: 0` from every `EnumeratedCheckpointNaming(...)` construction, and rename the labels `step:` → `trainerStep:` on `fileName` / `url` / `write` and `step(ofFileName:)` → `trainerStep(ofFileName:)` (OD-12a):
  - `TrainerOutputFileGuardTests.swift` constructions `:345, 348, 351, 357, 372, 389, 419, 443, 446, 452, 460, 474, 486, 497`; label sites `:346, 349, 352, 358, 359, 361-366, 373, 374, 391, 421, 430, 444, 461, 464, 466, 475, 479, 488, 490, 492, 499, 500, 502`;
  - `ReplayRunnerPreflightTests.swift:63`;
  - `TrainVsUciSessionTests.swift` constructions `:91, 97, 105, 131`; label sites `:92, 98, 106, 132` (`:120` is the `CheckpointStemError.namedLikeAnEnumeratedCheckpoint(stem:step:)` case, which this plan does not rename; not an edit).
- **TE-3 (mechanical)** add `segmentStartTrainerStep: 0` to `reachableEnumeratedCheckpoints` / `requireNoReachableEnumeratedCheckpoints` calls: `TrainerOutputFileGuardTests.swift:424-425, 435, 437, 447, 453`; `ReplayRunnerPreflightTests.swift:64-65, 70-71`. With a start of 0 every assertion means what it means today.
- **TE-4** `SegmentIndexedCheckpointNamingTests.swift` (pins the retired writing scheme):
  - its `naming(_:_:segment:)` helper (`:12-15`) loses the segment parameter; `testSegmentZeroKeepsTheExistingNames` (`:17-24`): its three `naming(…, segment: 0)` calls lose the argument and its three `.fileName(step:)` calls (`:18`, `:20`, `:22`) become `.fileName(trainerStep:)` (OD-12a); assertions and expected names unchanged;
  - **delete** `testALaterSegmentsNamesCarryItsIndex` (`:26-33`) and `testEachSegmentParsesOnlyItsOwnStepFiles` (`:35-50`) — replaced by `TrainerStepCheckpointNamingTests`;
  - `testTheSegmentIndexComesFromTheLineageRule` (`:70-97`): drop its last assertion (`:94-96`, the `-seg1-` name); the lineage-rule assertions stay (lineage records still carry the index);
  - `testSegmentStepFilesAreRecognizedUnderAnyStem` (`:52-64`) unchanged (the legacy parse stays);
  - the class doc comment (`:4-7`) describes the remaining scope.
- **TE-5** No Python test edit is expected (`ProbeRecordTests` fixtures carry no `trainer_completed_steps`; the probe-loop and lineage fixtures use segment-0 names whose two bases agree; the new record keys go after the four leading keys `test_good_probe_becomes_a_record_with_identity` pins, `test_tooling.py:154`). Any edit found necessary is listed here for approval before it is made.
- **Deletions requiring the owner's express approval — exactly two:** `SegmentIndexedCheckpointNamingTests.testALaterSegmentsNamesCarryItsIndex` (`:26-33`) and `SegmentIndexedCheckpointNamingTests.testEachSegmentParsesOnlyItsOwnStepFiles` (`:35-50`). One assertion removed: `testTheSegmentIndexComesFromTheLineageRule` `:94-96`. No other existing test is deleted or loses an assertion.

### Failing-first sequence

1. P1 adds `TrainingStepLineSchedule` and `ChessTrainer.isDiagnosticsStep` / `isBatchStatsStep` **with today's behavior** (segment start, every 50 segment steps, saves every 1,000 segment steps; diagnostics on the interval only; no time rule) and wires both runners to them — a pure refactor, no behavior change. P1 also declares `step_line_interval_sec` (D7 steps 1–2, read by nothing yet; TE-1 lands here), because both runner tests set it through the parameter snapshot and must compile and fail on the behavior, not on a missing key.
2. Add the new tests that compile against that API (the schedule tests, the two runner tests, the trainer-rule tests). Run them; record the failures here.
3. Apply the change (P2). The same tests, unmodified, pass. The naming tests are written against the new naming API and are new-feature tests, not regression tests.
4. P1 is not committed alone, so every commit passes the suite.

---

## Validation

Runs alongside live training only as the owner allows (the bot-lineage plan records an owner correction allowing builds, tests and read-only checks alongside live runs); otherwise after the live runs end. Builds are compile-only `build_project` through drews-xcode-mcp.

- **V-1 Tests.** `TrainingStepLineScheduleTests`, `TrainerStepCheckpointNamingTests`, the two new `ResumeEquivalenceTests` methods, then `ResumeEquivalenceTests`, `ExactResumeTests`, `ExactResumeCompletionTests`, `RunObservabilityResumeTests`, `CorpusReplayFeederTests`, `CorpusReplayFailLoudTests`, `ReplayRunnerPreflightTests`, `TrainerOutputFileGuardTests`, `SegmentIndexedCheckpointNamingTests`, `TrainVsUciSessionTests`, every class whose name contains `TrainVsUci`, `TrainingParametersTests`, `TrainerHyperparametersTests`, `LayerHealthTests`; the Python suites (`python3 -m unittest discover documentation/dashboards/tests`); then the full suite before merging (persistence and naming change).
- **V-2 Live corpus replay, fresh then resumed mid-interval** (scratch folder `$S`, scratch parameters file with `batch_stats_interval` 10 and `step_line_interval_sec` 60):
  - `--replay-corpus <corpus> --parameters $S/p.json --seed 1006 --training-step-limit 1013 --out-model $S/cad-replay-latest.safetensors --enumerate-checkpoints --output $S/r1.json`: lines at trainer steps 1, 50, 100, …, 1,000 and time lines between; every line after the first has `pEnt=` values; `[BATCH-STATS]` lines = step lines whose step is a batch-stats step; files `cad-replay-step1000`, `cad-replay-step1013`.
  - `--start-model $S/cad-replay-step1013.safetensors --resume-exact --training-step-limit 1100` into the same `--out-model`: `[RESUME] EXACT`; first line at trainer step 1,014; every later line carries diagnostics; save and line at 2,000 with the line before the `saved trainer model` line and before the checkpoint `[LAYER-HEALTH]` block; files `cad-replay-step2000` (header `training_step` 987, `trainer_completed_steps` 2,000) and `cad-replay-step2113`; no `-seg` file.
  - The same resume command again: refused at pre-flight by the rolling-output check first (`checkRollingOutput` runs before the enumerated scan, `CorpusReplayRunner.swift:1111` vs `:1129`; the rolling file now holds the resumed run's own `model_id`, not the start file's). Then the same command with `--overwrite-out-model` (scratch rolling file only): refused by the enumerated scan, naming `cad-replay-step2000` … `cad-replay-step2113` (2 files) and the reachable range `1014…2113`.
  - Header-only check by a short script: `model_id`, `training_step`, `trainer_completed_steps`, lineage `segment_index` / `segment_local_step` / `cum_trainer_step` of all four files.
  - Log volume: `[BATCH-STATS]` bytes against arm C's log at the same per-step rate.
- **V-3 Train-vs-UCI** (owner machine with an engine): `--train-vs-uci "cmd=/opt/homebrew/bin/stockfish;n=1;go=nodes 1" … --training-step-limit 1013 --enumerate-checkpoints --checkpoint-stem $S/uci`, then `--resume-exact` from its final session for 1,100 with the same `--checkpoint-stem` (without it a session-folder start names the stem after the new run's model ID, `CLI/TrainVsUciSession.swift:263-267`, and the shared-stem case is not exercised): the same line and name checks (`uci-vsuci-step2000`, `uci-vsuci-step2113`, and segment 0's `uci-vsuci-step1000`, `uci-vsuci-step1013` untouched).
- **V-4 Tooling on the new names:** `PROBE_ABOVE_STEP=1013 experiments/probe_loop.sh --once cad $S/probes-seg1.jsonl` records steps 2,000 and 2,113 with `step_basis` `trainer_step` and never probes `cad-replay-step1000` / `1013`; `experiments/probe_loop.sh --once cad $S/probes-seg0.jsonl 1013` (segment 0, step limit 1,013) records 1,000 and 1,013 with `step_basis` `segment_step` and logs the two later files as skipped; the same without the limit stops at `cad-replay-step2000` with exit 4 (the other-run check) — expected, and recorded as such. Appending one new record to a copy of an existing segment-0 probes file (records without `step_basis`) succeeds. `replay.py discover-stems` (no `--write`) on the scratch stem reports the trainer-step files as skipped. No command that writes `registry.json` or `data/` is run on production files.
- **V-5 Old files still read:** `--probe-model` on `20261005-lrC-cyc10-r1-replay-seg1-step1000.safetensors`; `PROBE_SEGMENT=1 probe_loop.sh --once 20261005-lrC-cyc10-r1 $S/probes-C-seg1-recheck.jsonl` reproduces `experiments/20261005-lr-schedule-ab/probes-C-seg1.jsonl`'s steps with `step_basis` `segment_step`; a `--resume-exact` from that file into a scratch `--out-model` passes pre-flight and names its files by trainer step; `vsuci.py` and `replay.py` rebuilds of existing runs produce no CSV change (dry comparison against the committed CSVs, written to `$S`).
- **V-6 GUI:** a fresh Play-and-Train to ≈ 1,100 trainer steps: `[STATS]` at the first step, every 50 through 1,000, then 1,000 and time lines; `[BATCH-STATS]` beside lines; the new Sessions-tab field; a live change of the interval takes effect on the next deadline; a promotion's rewind adds no fixed line; `--train` `results.json` rows at the same ticks.

---

## Phasing

Each phase builds and commits on its own once its tests pass (P1 and P2 together, per the failing-first sequence).

- **P1 + P2 — Core and CLI.** D1, D2, D3, D4, D5 (no code), D6 (CLI side, trainer, `SyncBox`), D7 steps 1–6 and 8–9; owner-approved TE-1 to TE-4; new Swift tests. V-1 (Swift), V-2.
- **P3 — GUI.** D8, D6's GUI side, D7 step 7. V-6.
- **P4 — Tooling.** D9 and the Python tests. V-4, V-5. **Must land before any resumed run on the P2 build is probed or tracked** (the old probe loop would try earlier segments' files and stop on the other-run check; `track` could take a wrong file).
- **P5 — Documentation.** D10, CHANGELOG. V-3 when an engine run is available.

---

## Risks

- **Fewer lines between fixed points.** Analysis that read every 50th segment step now gets time lines at wall-clock-dependent steps; A/B arms log at different steps except on the fixed lines (every 50 through 1,000, every 1,000). Comparisons should use those.
- **GUI steady cadence 60 s → 180 s by default.** `--train` `results.json` gets a third of today's steady rows (plus the dense and 1,000-step rows), the chart's legal-entropy trace and the `[ALARM] policy entropy` log line refresh at the line cadence (D8). Alarm evaluations are unaffected: they run every 50 trainer steps independent of the line (approved direction item 5). `--train` parameters files can set `step_line_interval_sec` 60.
- **A second resume of an earlier file into the same stem is refused** where `-seg<k>` names would sometimes have avoided the collision; the refusal names the files and suggests a new stem.
- **Header and name differ for a resumed segment** (`…-step37000` holds `training_step` 1,000). By design (D5); anyone reading headers by hand must know it. Documented in CLAUDE.md.
- **Forced diagnostics at a non-default `batch_stats_interval`** (not dividing 50) change which steps run the diagnostic graph compared with older builds, so such a run's numerics may differ bitwise from an older build's at those steps. On those steps the trainer's non-finite check also covers `valueMean` and entropy (`Training/ChessTrainer.swift:7089-7102`, gated on `includeDiagnostics`), so a run that would have gone on with a non-finite diagnostic and finite losses stops there instead; the GUI's rolling means also take those extra samples. No run uses such an interval today; the behavior fingerprint does not see it (it trains with `batchStatsInterval = 0`, whose fallback diagnostics already cover every fixed line step).
- **`vsuci.py` limit:** a train-vs-UCI log from a build that wrote `trainerStep=` but the old segment cadence, resumed at an offset that is not a multiple of 1,000, would get no rows. None is registered.
- **Adjacent plans cite the old names and cadence** (alarms, HPARAM); listed above for whichever lands second.

---

## Owner decisions

- **OD-1** `training_step` meaning: keep it segment-local, names carry the trainer step (recommended) — or make it the trainer step (breaks the tracker's `segment_local_step` checks, the guard's comparison basis and the bot plan's assumption; one key with two undecidable meanings).
- **OD-2** Name shape: `<stem>-<tag>-step<trainerStep>` with no `-seg<k>` (recommended) — or `-seg<k>-step<trainerStep>` (same shape as old files with a different step meaning).
- **OD-3** A segment that trained no step writes no enumerated copy, and logs that (recommended) — or write it and fail on the collision with the start file.
- **OD-4** (replaces the old OD-A) Every fixed line step is also a diagnostics and batch-stats step in the trainer (recommended; no change at interval 10) — or refuse a `batch_stats_interval` that does not divide 50 — or accept `--` on those lines.
- **OD-5** `[BATCH-STATS]` rides the step line (recommended) — or a separate interval parameter.
- **OD-6** The GUI's per-step-for-500 bootstrap (with 25-step legalMass / layer-health strides) merges into the shared schedule (recommended) — or keep it ahead of the shared schedule.
- **OD-7** Any line restarts the time interval (recommended) — or only time lines do.
- **OD-8** The time rule also applies during the dense phase (recommended).
- **OD-9** No extra line before the final save (recommended: keeps "every line after the first carries diagnostics"; R0 holds at every 1,000-step save, as today) — or add a final line that may carry `--`.
- **OD-10** Parameter `step_line_interval_sec`: Double, default 180, range 10…86,400, Observability, live-tunable, `absentValue: .currentSetting`, Sessions tab beside the KL-probe interval (recommended).
- **OD-11** Keep `batch_stats_interval`'s id and update its description (recommended) — or rename it (saved settings reset).
- **OD-12a** Rename the naming API's `step:` labels to `trainerStep:` (recommended; mechanical test edits in TE-2) — or keep `step:` with doc comments.
- **OD-12b** Probe loop: new `PROBE_ABOVE_STEP`, `PROBE_SEGMENT` kept for legacy names only, the step-limit argument read as a trainer step; probe records gain `trainer_step` and `step_basis`, with `"segment_step"` whenever the name matches `training_step` (so existing probes files keep accepting records) (recommended).
- **OD-13** `table_common`: add a trainer-step helper and leave the existing one for the dated experiments (recommended).
- **OD-14** Put `lastBatchStatsSummary` / `lastBatchStatsUniquePct` under one `SyncBox` (recommended; the GUI already reads them unlocked).
- **OD-15** (new, cross-plan with the alarms plan) On a step where both a step line and an alarm evaluation are due (every fixed line step: every 50 through 1,000, then every 1,000), take **one** live `[LAYER-HEALTH]` read and use it for both the logged readout and the monitor (recommended; the read is a `graph.run` on the trainer queue, and two reads of identical state are waste) — or two independent reads (simpler coupling, double GPU reads on those steps). The alarms plan's revision implements whichever is chosen; this plan only fixes the order (line block, then evaluation, then save).
- **OD-16** `experiments/20261005-lr-schedule-ab/bn_liveness.py`: update it to the two-basis rule if any arm of the live LR experiment is resumed on the P2 build (recommended, only then) — or leave it reading pre-change files only.
- **OD-B** Approve test edits TE-1 to TE-4, including the **two test deletions** and **one removed assertion** listed under TE-4 (TE-5: none expected). **Decided (owner, 2026-10-06): approved**, including deleting `testALaterSegmentsNamesCarryItsIndex` and `testEachSegmentParsesOnlyItsOwnStepFiles` and removing the `-seg1-` name assertion from `testTheSegmentIndexComesFromTheLineageRule`.
- **OD-C** (old) Moot: every checkpoint step is now a fixed line step and carries diagnostics.

---

## Non-goals

- `[LEGAL-COST]` and `[SAMPLER]` keep the batch-stats cadence (small lines).
- `--training-step-limit`, `--gpu-capture-step` and the train-vs-UCI eval-sync cadence stay segment-local.
- Renaming files on disk, or re-logging the fields missing from arm C's resumed log (they were never computed).
- Search of any kind; this plan changes logging, save points and names only.

---

## Review (2026-10-06, independent review of the redesign)

Scope: every design section against `main` at `5138c628`, the alarms plan (Part P, R0, D4, D6 and its cadence text), `HPARAM_RECORDING_PLAN.md` and `LICHESS_BOT_FOLLOW_LINEAGE_PLAN.md` where they touch names, `training_step` or `segment_local_step`, and the owner's 2026-10-06 decision that alarm evaluations run every 50 trainer steps independent of log lines. Over 40 file:line citations were opened; all but the ones fixed below point at what the plan says. Arm C's numbers were re-measured from the logs (`[BATCH-STATS]` 560 lines / 43,117,507 B of 43,457,830 B; 113 of 113 step lines `pEnt=--`; last step 5,603 at trainer step 6,116; first segment's final save at 513).

### Verified, no change

- D2: `includeDiagnostics` / `isStatsStep` are functions of `_completedTrainSteps + 1` and the interval only (`Training/ChessTrainer.swift:4649-4657`); the sampler's metadata pointers are only written (`Training/ReplayBuffer.swift:1873-1878`), so a forced batch-stats step draws the same indices; the KL probe's schedule is by step index. Every fixed line step is a multiple of 10, so at interval 10 and at 0 (fallback 10) the diagnostic and batch-stats step sets are unchanged; `BehaviorFingerprint` trains at interval 0 (`:263`). Exact-resume determinism holds: the rule never depends on where a process started.
- D3: the only save call sites are `CorpusReplayRunner.swift:1981, 2009` and `TrainVsUciRunner.swift:848, 852, 872, 874`; an epoch-budget, corpus-end, step-limit or abort end goes through the final save at an arbitrary step, unchanged. A step-limit run has no epoch bound (`CorpusReplayRunner.swift:1219`), so the 990-step runner test is not cut short by the synthetic corpus's short epochs, and its 75-game refeed window fits the 80-game corpus (`:1345-1360`).
- D5: the bot lineage plan ranks within a segment by `segment_local_step` and never by filename or cross-segment `training_step` (`LICHESS_BOT_FOLLOW_LINEAGE_PLAN.md:63,121-123`); keeping `training_step` segment-local leaves it unaffected. HPARAM `:1208`/`:1226` file names and the segment-1 arithmetic (200 → 400) are right.
- D7: every `TrainingParameters.swift` / popover / session citation matches; `test_registry_size` is 85 today.
- D8: the GUI stats box is the cumulative trainer step, seeded on resume and rewound on promotion (`App/SessionController+Arena.swift:480-488`), so D1 rule 2 is needed and correct.

### Must-fix (fixed in this file)

1. **Alarm decoupling.** The plan assumed the alarms plan's live evaluation rides the step line (D1 CLI wiring, R0 entry, GUI Risk "alarm evaluations come less often"). Rewritten: approved-direction item 5; D1 states the per-step order (line block → the alarms plan's 50-step evaluation → save) and that `lineDue` is logging only; R0 holds by loop position and because every checkpoint step is a multiple of 50; the Risk no longer claims slower alarms. The alarms plan's own text still ties evaluations to step-line ticks at `:63`, `:202`, `:217`, `:479`, `:523`, `:839` — listed as conflicts under Order with adjacent plans for its revision (not edited here). New OD-15 (one shared live read where a line and an evaluation coincide).
2. **V-2 expected the wrong refusal.** Re-running the resume command is refused by the rolling-output check first (it runs before the enumerated scan and the rolling file now holds the resumed run's `model_id`); the enumerated-collision refusal is now checked with `--overwrite-out-model` on the scratch file.
3. **`step_basis` would have broken live probe loops.** As written, a segment-0 record was `"trainer_step"` and a record without `step_basis` counted as legacy, so the first new record appended to any existing segment-0 probes file (every run being probed when P4 lands) would be refused as mixing (exit 4); a file without `trainer_completed_steps` was also labelled `"trainer_step"` by inference. Basis is now `"segment_step"` whenever the name matches `training_step`, `"trainer_step"` only for a resumed segment on this build; new keys go after the four keys `test_tooling.py:154` pins; `load_probe_points`' existing check is narrowed to segment-step records.
4. **`vsuci.py` `STEP_RE` cannot match today's `[VS-UCI]` line** (`pLogitMean=` / `vLogitMean=` sit between `playedP=` and `gNorm=`), so "an optional trailing `trainerStep=` capture" would leave the new test passable only on an unrealistic fixture. D9 now adds the optional fields and a separate `trainerStep=` search.
5. **V-4 segment-0 probe** without a step limit reaches the later segment's files under the same stem and stops with exit 4. V-4 now passes the limit (a trainer step) and records the no-limit stop as expected; D9 states the limit's basis.
6. **Test-edit lists.** TE-4 omitted the three `.fileName(step:)` label renames in `testSegmentZeroKeepsTheExistingNames` (`:18, :20, :22`); TE-2 listed `TrainVsUciSessionTests.swift:120`, which is an error-case label (`CheckpointStemError.namedLikeAnEnumeratedCheckpoint(stem:step:)`), not the naming API. Fixed, and the two deletions and one removed assertion are now listed explicitly for approval.
7. **Failing-first could not compile.** Both runner tests set `step_line_interval_sec`; P1 now declares the (inert) parameter so they compile and fail on behavior. TE-1 moves to P1.
8. **GUI `[BATCH-STATS]` after a promotion rewind.** "Newer than the last logged step" would log nothing until training passed the pre-rewind step; now "differs from".
9. **Start lines named the wrong next save.** `firstFixedLineStep(after:)` gives 50 on a fresh run; added `firstCheckpointStep(after:)` for the cadence line and the enumerated-checkpoints start line (today `naming.url(step: autosaveEvery)`).
10. **Facts:** the B run log has 721 step lines through step 36,000 (was 683); three dated experiment tables call `buffer_plies_per_game`, not eight.

### Should-fix (fixed in this file unless noted)

- D4: the source of the segment's start trainer step at pre-flight (before the trainer exists) was unstated — now `resumeSnapshot?.schedule.completedTrainSteps ?? 0`, checked against the trainer after restore. The any-stem parser's legacy confirmation needs its own private builder once `segmentIndex` leaves the struct — stated. A zero-step corpus-replay run no longer writes `-step0` (OD-3 applies to segment 0 too) — stated.
- D8: the GUI keeps today's `steps > 0` guard before the first `lineDue`; the legal-entropy chart trace and the `[ALARM] policy entropy` log line move with the line cadence; the heartbeat streak alarms do not ride `[STATS]`. Stated in D8 and Risks.
- D2 Risk: at a non-default interval the forced diagnostic steps also extend the non-finite `valueMean` / entropy throw (`ChessTrainer.swift:7089-7102`) and the GUI's rolling means to those steps. Stated.
- The schedule test's "max(seconds, steps)" bound mixed units; now two bounds that must both hold.
- V-3 now passes `--checkpoint-stem` to both segments (a session-folder start otherwise gets a new stem named after the run's model ID and never exercises the shared stem).
- `experiments/20261005-lr-schedule-ab/bn_liveness.py` (the live LR experiment) requires name step = `segment_local_step` and would refuse a resumed arm's new files — listed in D9 with OD-16; not changed without the owner's word.
- Not changed: `replay.py` `enum_specs` still falls back to the run-level glob for an un-derived last segment, which matches segment 0's files under a shared stem (pre-existing; new trainer-step files are now reported as failures instead of misfiled, which is an improvement); `vsuci.py`'s idle removal bounds an interval by one line's `ms` × Δsteps, coarser with sparse lines (noted in D9).

### Verdict

Sound after the fixes above; ready for the owner's decisions. The design keeps exact-resume determinism (D2 is a pure function of the trainer step and adds no diagnostic step at the default interval), keeps old files readable by name and by header, and leaves the bot lineage plan untouched. The one cross-plan dependency left open is the alarms plan's revision for the decoupled 50-step evaluation (its text listed above) and OD-15.

### Owner decisions (final list, one line each)

- OD-1 keep `training_step` segment-local; names carry the trainer step — **recommend yes**.
- OD-2 name shape `<stem>-<tag>-step<trainerStep>`, no `-seg<k>` — **recommend yes**.
- OD-3 a segment that trained no step writes no enumerated copy (also drops a fresh run's `-step0`) — **recommend yes**.
- OD-4 every fixed line step is a diagnostics and batch-stats step — **recommend yes** (no change at interval 10).
- OD-5 `[BATCH-STATS]` rides the step line, no new parameter — **recommend yes**.
- OD-6 merge the GUI bootstrap into the shared schedule — **recommend yes**.
- OD-7 any line restarts the time interval — **recommend yes**.
- OD-8 time rule also in the dense phase — **recommend yes**.
- OD-9 no extra line before the final save — **recommend yes**.
- OD-10 `step_line_interval_sec` Double, 180 s, 10…86,400, Observability, live, `.currentSetting`, Sessions tab — **recommend yes**.
- OD-11 keep `batch_stats_interval`'s id, update its description — **recommend yes**.
- OD-12a rename naming-API labels to `trainerStep:` — **recommend yes**.
- OD-12b `PROBE_ABOVE_STEP`, legacy-only `PROBE_SEGMENT`, trainer-step limit, `step_basis` with segment-step default — **recommend yes**.
- OD-13 add a trainer-step `table_common` helper, keep the old one — **recommend yes**.
- OD-14 batch-stats summary under one `SyncBox` — **recommend yes**.
- OD-15 one live `[LAYER-HEALTH]` read serves both the line and the alarm evaluation when they coincide — **recommend yes** (implemented by the alarms plan).
- OD-16 update `bn_liveness.py` only if an LR-experiment arm is resumed on the P2 build — **recommend yes, conditionally**.
- OD-B approve TE-1–TE-4, including deleting `testALaterSegmentsNamesCarryItsIndex` and `testEachSegmentParsesOnlyItsOwnStepFiles` and removing one assertion — **recommend yes**.
