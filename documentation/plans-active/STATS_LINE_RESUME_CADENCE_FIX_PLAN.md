# Step-line cadence, save points and checkpoint names on the trainer step — plan

Status:
- 2026-10-05: proposed as a bug-fix plan; revised after an independent review (M1 checkpoint-step consumers, M2 runner-level test). Owner asked for the fix ("fix that bug").
- 2026-10-06: **redesigned on the owner's approved direction** (below). The bug analysis (The bug, Cause, Hypotheses) is unchanged from the reviewed version; the fix design, tests, validation and decisions are replaced. Not implemented.
- 2026-10-06: independent review of the redesign; fixes applied in place and listed under Review at the end. Reflects the owner's decision that alarm evaluations are decoupled from log lines. Not implemented.
- 2026-10-06: **owner decisions recorded** (Owner decisions). OD-1 is reversed: from architecture format v11 every file's `training_step` is the overall trainer step and the segment step is a sidecar; older files are not rewritten, they keep their writer's meaning and are flagged when loaded (D5). OD-16: `bn_liveness.py` is updated now (D9). OD-2 to OD-15 as recommended, with OD-12b and OD-13 restated for OD-1. Changes listed under Revision at the end. Not implemented; for re-review.
- 2026-10-06: **second independent review** of the OD-1 revision (`57af1480`) against the code and every header on this Mac; must-fixes applied in place and listed under Review 2 at the end. New owner decision OD-18. Not implemented.
- 2026-10-06: **implementation started** on a branch (P1 + P2, then P3, P4, P5); OD-17 approved by the owner during it. See Implementation notes at the end.

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
6. **One overall step (owner, 2026-10-06, OD-1):** "have an overall training step be the only step we reference 99% of the time, and have a separate 'step within this segment/run' as a sidecar." Every file written from now on states the trainer step as its `training_step`; the segment step lives in the lineage record (and a flat mirror). Old files are not rewritten; a version marker tells the two apart, and loading an older file is flagged (D5).

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

Name shape (OD-2): **`<base>-<tag>-step<T>`**, where `T` is the trainer step at the save (`trainer_completed_steps`, which equals the record's `cum_trainer_step` on a trainer file, and — from format v11 — the file's own `training_step`, D5; so a file's name step and its header's `training_step` agree in both eras). The `-seg<k>` marker is no longer written.

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

### D5. `training_step` is the trainer step from format v11; older files keep their writer's meaning

**Decided (owner, 2026-10-06), reversing the earlier recommendation (OD-1):** "we already have cum_trainer_step. I don't care if old files are different. that's ok. I'd rather maybe flip things -- have an overall training step be the only step we reference 99% of the time, and have a separate 'step within this segment/run' as a sidecar. We could update old files to the best of our ability, or update a version somewhere so that we can always flag if an older file is loaded."

**What a file states, from this change on (every writer):**
- The header's `training_step` is the **trainer step** the weights were taken at: on a trainer-state file it equals `trainer_completed_steps` and the lineage record's `cum_trainer_step` (one value, three places, checked — below).
- The **segment step** (steps this segment trained) is the sidecar: the lineage record's `steps.segment_local_step` (already written today), plus a new derived flat mirror `lineage_segment_local_step` beside the existing mirrors (`LineageRecord.MirrorKey`, `Persistence/LineageRecord.swift:879-890`; never read back, like the others — it is there so a person reading a header sees both numbers without opening the JSON record). `MirrorKey.all` gains it; the tests that use `MirrorKey.all` (`LineageRecordTests.swift:206, 394, 412`) take it symbolically and need no edit.
- Writers:
  - corpus replay (`CLI/CorpusReplayRunner.swift:1566-1569`): `trainingStep: snapshot.schedule.completedTrainSteps`; notes `corpus replay <reason> @ trainer step <T> (segment step <S>)`; the lineage record's `segmentLocalStep: step` is unchanged (`:1579`);
  - train-vs-UCI trainer files (`CLI/TrainVsUciRunner.swift:535-540`, record `:548`) the same; its session `champion.safetensors` (`:608-612`, the play network synced from the trainer at the save) states the same trainer step;
  - GUI trainer files (`App/SessionController+Checkpoint.swift:541-543`; promotion, `App/SessionController+Arena.swift:713-715`): the trainer snapshot's clock (`trainerSnapshot.schedule.completedTrainSteps`; `promotionSaveTrainerStep`) instead of the stats box's count at the cut. The two are equal at every cut (`SessionSaveConsistentCutTests.swift:105` pins it), so no GUI file changes value; the writer now has one source; **[Correction, final training-side review m5, 2026-10-07: not at every cut. "New Session, keep trainer" starts a fresh stats box at 0 on a trained trainer, so there the box's count is below the trainer clock. The writer's single source (the snapshot's clock) is right in that mode too; the equality holds after a fresh start or a session resume only.]**
  - GUI champion files (`:102`, `:476`, `Arena.swift:742`): `championFileTrainingStep(origin:)` (`App/SessionController+Lineage.swift:64-70`) is unchanged in code; its input, `ParentFile.trainerCompletedSteps`, now comes from the reading below (`trainerStepOrStatedStep`), so a champion loaded from an older file states that file's trainer step, or its stated step when it never recorded one (a schedule-less pre-v7 CLI file: its segment step, as today);
  - `--new-model`, `--derive-model` and grafts write no `training_step`, as today.
- **Encode check** (`SafetensorsModelIO.encode`, beside the existing `lineageStepDisagreesWithTrainerClock` check at `Persistence/SafetensorsModelIO.swift:156-163`): a trainer-state file (`trainerSchedule` present) whose `trainingStep` is not `schedule.completedTrainSteps` (nil included) is refused with a new `IOError.trainingStepDisagreesWithTrainerClock(trainingStep:trainerClock:)`. `ModelCheckpointMetadata.trainerFile(creator:trainingStep:…)` keeps its signature (decided by implementer: deriving the step inside it would touch every test call site; refusing a mismatch touches only the five that pass a different number, TE-6 to TE-10).
- **Decode check:** a format-v11 trainer-state file whose `training_step` is absent or differs from `trainer_completed_steps` is a decode error naming the file and both values. Plain (non-trainer-state) files are not cross-checked against their lineage record (a GUI champion's record is its origin's record, whose total may be unrecorded).

**The marker: architecture format v11 (decided by implementer; recommended).** `ArchitectureFormat.currentVersion` 10 → 11 (`Network/ArchitectureFormat.swift:77`), with `static let trainingStepIsTrainerStepFromVersion = 11` and a v11 entry in its version history ("no new architecture field; `training_step` is the trainer step; before v11 it meant what its writer wrote — see `SafetensorsModelIO.trainingStepReading`"), as v7 did for the lineage requirement. Why the format version rather than a dedicated key such as `training_step_basis`:
- It is the project's one convention for "which rules does this file follow" (`ArchitectureFormat`: "the file-format version gate for every carrier"; v7 already gates a non-architecture rule). A second marker would be a second source for "how new is this file".
- **Stale readers mostly fail loudly instead of misreading.** Every build that predates the change refuses a v11 file as "newer than this build supports" (`ArchitectureFormat` `unsupportedFutureVersion`), and so does a stale `scripts/dcm_arch.py` (`checked_format_version`, `:121-133`). A stale `scripts/dcm_lineage.py` does **not**: it parses the version without an upper bound (`format_version_of` → `dcm_arch.parsed_format_version`, `scripts/dcm_lineage.py:102-110`), and `replay.py` reads `training_step` straight from headers in `meta_step_of` (`:98-101`), `_ckpt_index` (`:1189-1215`) and `discover_enum_stems` (`:718`). In a stale tracker, `track` and `probe_backfill` still stop before writing a row, because `internals_cells` reads the architecture through `dcm_arch` first (`:103-125`, `:823-825`); `discover-stems` refuses a stem of several `model_id`s (`:724-726`). So no stale in-repo path writes a misfiled row, but the protection comes from those later checks, not from the reader that misreads; P4 lands with P1 + P2 for that reason. A dedicated key would have no such backstop: a stale `internals` reads a v10-shaped architecture happily.
- The legacy flag rides the existing mechanism: decode collects legacy resolutions on its `DecodeFormat` and loaders log them once per load (`SafetensorsModelIO.Decoded.architectureFormat`, `:187-190`), while display-only readers stay quiet.
- Costs, accepted: presets and `architecture.json` are stamped v11 with nothing new in them (as at v7); files written from v11 on cannot be read by older builds, so a probe loop's `PROBE_BIN` must be a v11 build for them (D9); `scripts/dcm_arch.py`'s `CURRENT_FORMAT_VERSION` moves to 11 (`:76`), which changes one Python test's literal (TE-P1). If another plan bumps the format first, this one takes the next number; the rule is "the version that introduced it".

**Reading any file — one function, one rule set.** New `SafetensorsModelIO.trainingStepReading(fromMetadata:source:) throws -> ModelFileStepReading` (header-only, so catalogs and guards use it without decoding weights), mirrored in Python as `scripts/dcm_lineage.py` `step_reading(metadata, source)`. `ModelFileStepReading`: `basis: TrainingStepBasis`, `statedTrainingStep: Int?` (the header value as written), `trainerStep: Int?`, `segmentStep: Int?`, and `trainerStepOrStatedStep` (`trainerStep ?? statedTrainingStep`, the value catalogs order and display by). `ModelCheckpointFile` carries the reading from decode.

| Basis (`TrainingStepBasis`) | When | `trainerStep` | `segmentStep` |
|---|---|---|---|
| `.trainerStep` | format ≥ 11 | `training_step` | the record's `segment_local_step` |
| `.legacySegmentStep` | format < 11, written by corpus replay or train-vs-UCI: `creator` `replay` / `train-vs-uci` | `trainer_completed_steps`, else the record's `cum_trainer_step`, else **none** (never reconstructed) | `training_step` |
| `.legacyGUITrainerStep` | format < 11, written by the GUI: `creator` one of the GUI save tags (`manual`, `periodic`, `promote`, `sigusr2`) | `trainer_completed_steps`, else `training_step` (the GUI always wrote its cumulative count, or for a champion file the trainer clock its source stated) | the record's `segment_local_step`, else none |
| `.legacyUnknownWriter` | format < 11, any other writer (or none) | `trainer_completed_steps`, else `training_step` — exactly today's reading (`SafetensorsModelIO.trainerClock`), kept and flagged | the record's `segment_local_step`, else `training_step` — the tracker's reading today (`replay.py` `meta_step_of`), kept and flagged |

- **The writer is the file's `creator`, never the lineage record's `path_kind`.** Every writer stamps `creator` (`ModelCheckpointMetadata.creator`, written unconditionally at `Persistence/SafetensorsModelIO.swift:149`), and `path_kind` is not the writer on a GUI champion file: `championFileLineageRecord(origin:)` (`App/SessionController+Lineage.swift:433-449`) gives a champion loaded from a file that file's own record, `withoutTrainerState()` (`Persistence/LineageRecord.swift:899-914`) keeping its `invocation`, so a GUI champion loaded from a v7+ corpus-replay file states `path_kind` `replay` while its `training_step` is the source's trainer clock (`championFileTrainingStep`, `:64-70`). Reading it by `path_kind` would call that clock a segment step. No such file is on this Mac (no GUI file here carries a record at all; survey below), but GUI runs from replay checkpoints on another Mac write them.
- `.dcmmodel` files (the pre-safetensors container, 21 on this Mac, all GUI saves) get the same reading at the unversioned legacy version (`ArchitectureFormat.unversionedLegacyVersion`, 3) by their `creator`. They carry no `DecodeFormat` (`ModelCheckpointFile.architectureFormat` is nil for them, `Persistence/ModelCheckpointFile.swift:320-325`), so the legacy-log mechanism below cannot flag them: `CheckpointManager`'s loaders log the training-step line for a `.dcmmodel` themselves, through the same formatter.
- A file that states no `training_step` (`new-model`, `derive-model`, `handcraft`, grafts) has no step to read: its reading has `statedTrainingStep` nil, `trainerStep` = `trainer_completed_steps` or none, and it is never flagged, whatever its creator.
- The writer tables are named constants (the CLI creators become `ModelCheckpointMetadata` constants instead of the literals at `CorpusReplayRunner.swift:1566` / `TrainVsUciRunner.swift:536, 609`; the GUI set is pinned by a test to equal `SessionSaveTrigger.allCases.map(\.diskTag)` plus `SessionSaveTrigger.promotionDiskTag`). Survey of every header on this Mac on 2026-10-06 (4,529 files in `Models/` and `Sessions/*/`): the creators that state a `training_step` are exactly `replay`, `train-vs-uci`, `manual`, `promote`, `periodic`, `sigusr2`; `new-model`, `derive-model` and `handcraft` files state none. So no file on disk reads as `.legacyUnknownWriter`; that case exists for test fixtures (creator `test`) and anything unforeseen, and it keeps today's reading rather than refusing a readable file.
- **The flag (owner: "always flag if an older file is loaded"):** every load of a file before v11 that states a `training_step` logs one line through the decode's legacy log, e.g. `[ARCH] legacy file 20261005-lrC-cyc10-r1-replay-seg1-step1000.safetensors (format v8): training_step 1000 is the writing segment's step (corpus replay before format v11); trainer step 1513 (trainer_completed_steps)`. The GUI and unknown-writer forms say "is the GUI's trainer step" / "was stated by writer '<creator>', read as the trainer step as before v11".
- **`LineageTracker.ParentFile.trainerCompletedSteps`** (`ModelCheckpointFile.lineageParent`, `Persistence/ModelCheckpointFile.swift:370-372`; `SafetensorsModelIO.readParentFile`, `:389-417`; the derive and graft sources' parents, `Persistence/ModelDerivation.swift:386`, `Persistence/ModelGraft.swift:377`) takes the reading's `trainerStepOrStatedStep`, replacing `trainerClock(schedule:trainingStep:)` (`:414-429`; removed — one rule). This keeps the field what its design says it is — the parent's *stated* step, which may be segment-local for a source written before lineage, with the copy's total left unrecorded (`Persistence/LineageTracker.swift:362-368`, `UntrainedCopyRecordTests`' header) — and it changes the value of exactly one kind of file: a v7–v10 corpus-replay or train-vs-UCI **plain** file (the train-vs-UCI session's `champion.safetensors`), which states no trainer clock, now gives the record's `cum_trainer_step` (its trainer step) instead of its segment step. Every other file gives today's value. *(Review 2 reversed the earlier choice of the reading's `trainerStep` here, which made a schedule-less pre-v7 CLI file's parent step null: a GUI champion save or a derive copy of such a file — 3,811 v3 `replay` and 31 v3 `train-vs-uci` files on this Mac — would then state no `training_step` and a null parent step, so `ModelDerivation.requireUntrainedSource` (`Persistence/ModelDerivation.swift:505-526`) would accept trained weights as untrained and `ModelLineageTree` (`:111`, `:158`) would show them as an untrained seed.)* The comment at `Persistence/LineageTracker.swift:362-368` stays as it is. No existing test changes (`LineageRecordTests.testOlderFormatFileLoadsWithItsLineageUnrecorded` keeps 40; the other parent-step tests build `ParentFile` directly).

**Every reader of today's meaning, and its rule now:**
- `TrainerOutputFileGuard.rollingOverwriteVerdict` (`CLI/CorpusReplayRunner.swift:626-661`) via `TrainerModelFileIdentity.read(from:)` (`:290-304`) and the start model's identity (`:1112-1116`): `trainingStep` is the reading's `trainerStepOrStatedStep`. A rolling file and its start model share a `model_id`, and a `model_id` is minted by one process of one build (or, in the GUI, kept across GUI saves whose two bases agree), so both sides always read under the same rule; different `model_id`s are refused before steps are compared, as today. The struct's shape is unchanged, so `TrainerOutputFileGuardTests` (whose fixture files carry no creator and no format version: unknown writer, stated step) need no edit. Its doc (`:276-278`, "the segment-local `training_step`") is updated.
- `ModelFileCatalog` (`Persistence/ModelFileCatalog.swift:116-160`): `ModelFileEntry.trainingStep` is the reading's `trainerStepOrStatedStep` (same struct shape, so `ModelFileCatalogTests`, `ModelLineageTreeTests` and `ModelFileCatalogActivityOrderTests` are unchanged). Order within a line (one `model_id`) is the same as by the raw value; the number shown is now the trainer step wherever one is known. `ModelLineageTree`'s untrained test (`trainingStep == nil`, `Persistence/ModelLineageTree.swift:111, 158`) is unchanged. The reading decodes the lineage record only when it needs it (a pre-v11 CLI file without `trainer_completed_steps`, or a segment step from a record); the catalog does not decode records today (`:140-176`), so a file whose record is malformed now becomes a catalog error entry (`ModelFileCatalogError.notSafetensors`, as an unreadable architecture already does) where it used to list — stated, not silent.
- Lichess bot (file source, `LichessBot/Play/LichessBotModelSlots.swift:189`): `trainingStep: file.trainingStepReading.trainerStepOrStatedStep` instead of `file.metadata.trainingStep`, so the overview (`LichessBotOverviewView.swift:280`), the line picker (`LichessBotModelLinePicker.swift:184, 255`) and the chat reply (`LichessBotChatCommands.swift:86`) show the trainer step. Live sources already pass the trainer clock (`LichessBotSessionModelProvider.swift:70`). What changes for the follow-lineage plan, which is mid-implementation: see Order with adjacent plans.
- `--analyze-numerics` (`App/NumericsAuditCLI.swift:73, 154`): the step it hands `NumericsAudit.run` (which goes into the report JSON, `Network/NumericsAudit.swift:224`, and its summary's `step N`, `NumericsAudit+Summary.swift:14`) is the reading's `trainerStepOrStatedStep` — the GUI's audit of the live trainer already passes the trainer clock (`App/SessionController+RunAllAnalyses.swift:161`, `App/SessionController+NetworkWeightAnalysis.swift:61`), so both paths report one kind of number. The CLI's JSON line adds `stated_training_step` (the header value), `step_basis` and `segment_step`.
- `ModelDerivation.requireUntrainedSource` (`Persistence/ModelDerivation.swift:505-514`) and the graft's `source_training_step` argument (`Persistence/ModelGraft.swift:350-352`): both only ask "is it above 0", which is the same under every basis. Unchanged; the graft argument's doc says the value is as the source stated it.
- `TrainerScheduleState` doc (`Training/TrainerResumeState.swift:25-29`, "the CLI runners write their segment-local step there") and `LineageRecord.Steps.segmentLocalStep`'s doc (`Persistence/LineageRecord.swift:184`, "the CLI files' `training_step`") are rewritten.
- GUI session files: `session.json`'s `trainingSteps` is already the cumulative trainer step and is unchanged; the session's `trainer.safetensors` and `champion.safetensors` follow the writer rules above; a GUI resume reads the trainer clock from the schedule (`TrainerScheduleState.forSessionResume`), not from `training_step`, so resume is unaffected.
- Log lines and `results.json` are **not** renamed (decided by implementer): `[REPLAY]` / `[VS-UCI]` keep `step=` (segment) and `trainerStep=` (the alarms plan's D4 requires the format), and the `results.json` row keeps `steps` (segment) and `cum_trainer_step`.
- Python readers: D9.

Old files: nothing on disk is rewritten or renamed (owner rule; a rewrite would need its own approval). They stay readable by the new build and tools under the table above.

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
- One loop polling every 50 ms (today's bootstrap poll): read the trainer step from the stats box, call `lineDue(trainerStep:elapsedSec:carriesDiagnostics: trainingSnap.rollingPolicyEntropy != nil, intervalSec: <live parameter>)`, and on a due line: refresh legalMass, `logOne`, `[BATCH-STATS]` (D6), live `[LAYER-HEALTH]`. The 25-step strides and both loops go; `bootstrapStatsStepCount` is removed. `lineDue` is first called once the box's step is above 0 (today's `steps > 0` guard, `App/SessionController+Training.swift:2089`), so the first line is never at step 0; a resumed session's box is seeded with the saved step (`:87-92`), so its first line comes at the first poll, as today. **[Correction, final training-side review m5, 2026-10-07: reading the trainer step from the stats box was wrong after "New Session, keep trainer" (the box restarts at 0 while the trainer clock continues): the dense phase restarted on a trained trainer and the lines fell off the trainer's forced-diagnostics steps. The GUI poll now keys `lineDue` on `trainer.completedTrainSteps` (`TrainingStepLineSchedule.guiPollLineDue`); the box's step above 0 still gates the first line.]**
- Also moved by this cadence (stated so nothing relies on the old one): the legal-mass refresh behind the chart's legal-entropy trace (`realLastLegalMassSnapshot`, `:2107`, `:2166`) refreshes on line ticks, so at steady state every 180 s instead of 60 s; the `[ALARM] policy entropy …` log line written inside `logOne` (`:2057-2069`) follows the line cadence. The GUI's streak alarms (`TrainingAlarmController.evaluate(from:)`, heartbeat, `App/SessionController+Heartbeat.swift:646`) and the alarms plan's evaluations do not ride `[STATS]` and are unaffected.
- The time rule does not require a new step, so lines keep coming while training is paused by an arena, as today's 60 s lines do.
- Why merge: per-step lines of a cumulative rolling mean add little over 50-step lines (the mean over the first k steps moves by about 1/k per step); one rule for all paths is what was asked; a GUI resume already skips the bootstrap today (the box is seeded with the saved cumulative step, `:87-92`), which the trainer-step dense rule reproduces.
- Kept: the first line at the first observed step, so `documentation/dashboards/selfplay.py`'s fresh-launch test (first line ≤ 500, `RESET_FRESH_STEP`, `:60-68`) still holds; its comments (`:60-68`, `:323-325`) are updated.
- Changed for the GUI: steady-state lines every 180 s instead of 60 s by default (Risks; the parameter can be set to 60).

### D9. Tooling

All changes read old and new files by explicit rules; no existing data file is rewritten. Every Python reader of a header's step goes through one function, `scripts/dcm_lineage.py` `step_reading(metadata, source)`, the mirror of D5's `trainingStepReading` (same table, same constants; `TRAINING_STEP_IS_TRAINER_STEP_FROM_VERSION = 11`, the two creator sets; the writer from `creator` only). It reads the version through `dcm_arch.checked_format_version`, so a file newer than the tools refuses there instead of being read under the v11 rules, and it refuses a v11 trainer-state header whose `training_step` differs from `trainer_completed_steps`, as the Swift decode does. `checkpoint_facts` (`:481-499`) gains `trainer_step`, `segment_step` and `step_basis` from it. `scripts/dcm_arch.py` `CURRENT_FORMAT_VERSION` 10 → 11 (`:76`), so a v11 file is readable by the tools and a stale copy of them refuses it. A new method in `documentation/dashboards/tests/test_lineage.py`'s `SwiftMirrorTests` (`:119-133`) checks the new constants against the Swift source, like its existing checks.

- **`experiments/probe_record.py`** (`build_record`, `:67-107`): the identity rule is **unchanged** — the name's step equals the header's `training_step` in both eras (before v11 both are the segment step; from v11 both are the trainer step). The record gains `trainer_step`, `segment_step` and `step_basis` from `step_reading`, placed **after** the existing leading keys (`step`, `training_step`, `model_id`, `parent_model_id`), so `test_good_probe_becomes_a_record_with_identity`'s key-order assertion (`documentation/dashboards/tests/test_tooling.py:154`) holds unmodified. No mixing rule is needed: a probes file holds one `model_id` (`existing_model_ids`, `:52-63`, unchanged), and one `model_id` is written by one process of one build, so one basis. `load_probe_points` (`:109-137`) is unchanged. For a resumed segment on the new build the record's `step` is the trainer step, so two arms' probes line up by step without adding a segment base.
- **`experiments/probe_loop.sh`:** new `PROBE_ABOVE_STEP=<the segment's start trainer step>` (default 0): files at or below it are skipped, each logged once (like the step-limit skip, `:102-105`). It is needed because a resumed segment now writes under the same stem as the segments before it (D4): their files carry other `model_id`s, and without the bound the loop would reach them and stop on the other-run check (exit 4). The optional `step limit` argument (`:6`, `:10`, `:102`) is compared with the name's step, so for files from v11 on it is a **trainer** step (a resumed segment's limit is its start plus `--training-step-limit`); the header says so. A segment-0 probe of a stem that a later segment also wrote into needs that limit (its own last trainer step) for the same reason. `PROBE_SEGMENT` (`:23-26`, `:49-52`, `:100-101`) stays, documented as **legacy only** (`-seg<k>-step<N>` names written before this change), and the two are refused together. The header also says that `PROBE_BIN` must be a build that reads the files' format (v11 for files from this change on; the default frozen build predates it).
  - **Live loops run these files.** Four probe loops of the LR experiment were running from `experiments/probe_loop.sh` on 2026-10-06 (`20261005-lrBleakyall-cyc1`, `20261005-lrBsilu-cyc1`, `20261006-lrBsilu-clip1`, `20261006-lrBsilu-ctl15`, the last two with `PROBE_SEGMENT=1`), and each runs `probe_record.py` afresh per checkpoint, so P4's `probe_record.py` takes effect in them at their next checkpoint: it must accept their v9 files and append to their existing probes files unchanged in meaning (it does: same identity rule, keys added last). `probe_loop.sh` itself is read by a running `zsh` as it executes, so P4 replaces the file (write a new file, rename it over the old one) rather than rewriting it in place; V-4 checks that the running loops still record their next checkpoint after P4.
- **`documentation/dashboards/replay.py`** (the tracker keeps its axis: `cum_step = cumstep_base + meta_step`, where `meta_step` is the **segment step**, now taken from `step_reading` — before v11 that is `training_step`, as today; from v11 it is the record's `segment_local_step`):
  - `meta_step_of` (`:98-101`, used by `track` on the rolling file): the reading's segment step; a file with none is an error naming it.
  - `lineage_checkpoints` (`:603-623`): the contradiction check (`:617-620`, `training_step` vs `segment_local_step`) becomes per basis — from v11, `training_step` must equal the record's `cum_trainer_step` (when the record holds one); before v11, `segment_local_step` as today. It still returns the record's `segment_local_step`.
  - `lineage_cells` (`:641-643`): unchanged (its `meta` is still the segment step).
  - `discover_enum_stems` (`:694-720`): steps from the reading's segment step instead of the raw `training_step` (`:718`). A stem written by several segments of one run (possible from v11, D4) holds several `model_id`s and is refused by the existing one-`model_id`-per-stem rule (`:724-726`), with the reason: those segments are found by `segment_id` (`derive-registry`).
  - `enum_path` / `freeze` / `track` (`:791-835`): the rolling file's reading gives `meta` (segment step) and, from v11, the enumerated file's name step (the trainer step, = its `training_step`); before v11 the name step is `meta`, as today. The enumerated file is used only when its header's `model_id` and `training_step` equal the rolling file's (today it is taken by name alone); otherwise the existing `-frozen` copy.
  - `probe_backfill` glob scan (`:947-955`): a v11 file is filed at `cumstep_base + its segment step` (from the record), not at `cumstep_base + name step`; files of more than one `model_id` under one segment's glob are failures ("this stem holds several segments' files; give the segments their `segment_id` with derive-registry"), never filed. Before v11, unchanged. Lineage-identified segments (`segment_id`) are found by `lineage_checkpoints`, as today.
  - `_ckpt_index` (`:1189-1215`): keyed `(model_id, segment step)` from the reading (before v11 identical to today's `(model_id, training_step)`), so `recompute_internals`' lookup by the row's `meta_step` (`:1035-1060`) finds v11 files.
  - `import_probes` (`:1219-…`): the importer for the retired bundle monitor's `new_ckpts*.jsonl` (its docstring), whose records carry a `model` file name and no header facts. That monitor was retired before this change, so every such record names a pre-v11 file and its name step is the segment step, as today; the function is unchanged, except that when `ckpt_dirs` are given and the record's file is found there at format v11, the record is refused and reported (a v11 name step is a trainer step; filing it at `cumstep_base + name step` would double-count). Records `probe_record.py` writes are not its input (they have no `model` key).
  - Comments at `:526-534`, `:541-563`, `:916-922`, `:1190-1197`, `:283-296` (the "every ~60 s" note) updated. `_metrics_at` / `_games_at` need no change: a checkpoint's `meta` is a segment step and lands on a step line (D1, D3).
- **`documentation/dashboards/vsuci.py`** (`build`, `:138-165`): rows at the segment steps whose line has `trainerStep % 1000 == 0`; a log whose lines carry no `trainerStep=` keeps today's rows at `step % 1000 == 0`. The registered run (`sf100sl100`, three logs) has no `trainerStep=` in any line (checked 2026-10-06), so its CSV is unchanged. `STEP_RE` (`:38-41`) does not match today's `[VS-UCI]` line at all: it requires `playedP=… gNorm=` adjacent, while the current line has `pLogitMean=… vLogitMean=…` between them (`CLI/TrainVsUciRunner.swift:773-775`). So `STEP_RE` gains optional `pLogitMean=` / `vLogitMean=` fields (`[-\d.]+|--`) between `playedP=` and `gNorm=`, and a separate `trainerStep=(\d+)` search on the matched line (it trails `mom=` and the optional `lrCyc…`, so a fixed-position capture would not reach it). The new test's log uses the current line format verbatim. The idle-removal rule (`parse_segment`, `:52-57`) bounds an interval by `Δsteps × ms` with the line's single-step `ms`; with time lines hundreds of steps apart that bound rests on one step's duration (stated in the module doc, not changed). It reads no checkpoint header, so OD-1 does not touch it.
- **`experiments/table_common.py`:** `buffer_plies_per_game` (`:69-94`) is left as it is (the segment-step grid) because three dated experiment tables call it (`20261001-se-fc1-leaky`, `20261002-label-smoothing-C`, `20261002-noSE-noReZero`); a new `buffer_plies_per_game_by_trainer_step(log_name)` keys on `trainerStep=` multiples of 1,000 — the same steps new probe records carry as `step` (OD-13).
- **`experiments/20261005-lr-schedule-ab/bn_liveness.py`** (OD-16, decided: updated in this plan, P4; the alarms implementation uses it to make reference fixtures and later experiments will run it): `index_checkpoints` (`:249-268`) keeps keying by the record's `cum_trainer_step` and requiring the filename step to equal `training_step`; its record check becomes per basis through `dcm_lineage.step_reading` — from v11, `training_step` must equal `cum_trainer_step`; before v11, `segment_local_step`, as today. Its docstring (`:251-255`) and the `RUNS` comment (`:54-55`: a resumed segment on the new build continues `<stem>-replay-step<trainer step>`; earlier resumes are `-seg<k>`) are updated. `--selftest` keeps its existing cases unchanged (their metadata carry no format version, so they read before v11) and gains v11 cases (accepted when `training_step == cum_trainer_step`; refused otherwise). Its future-format refusal case (`:406`, `dcm_format_version="11"`) still passes after the bump only because that architecture also states the retired field; it is moved to `"12"` to keep testing what it was written for (TE-P2).
- **`documentation/dashboards/ckpt_inventory.py`**: header-based already; its docstring (`:5-12`, "`training_step` (segment-local)") is updated; the `KEEP` list (`:41`) gains `step_basis`, `trainer_step` and `segment_step` from `step_reading`, and the per-model step range (`:128`) uses the reading's `trainer_step` where known.
- **Dated experiment scripts** (`experiments/2026…/…/*.py` that read `training_step`, e.g. `20260929-se-style-ab/tensor_stats.py`): frozen records of their runs, which are all before v11. Not changed; run on a v11 file they would read `training_step` as written (the trainer step).
- **`registry.json` / `cumstep_base` / CSVs:** no change. Bases stay the segment's start on the registry axis and `meta_step` stays the segment step; only where the tools read the segment step from changes (D5's table). `_lineage_registry.py` (`derive-registry`) reads records only and is unchanged.

### D10. Documentation

At implementation (not now; this plan edits only itself):
- `CLAUDE.md`:
  - `:39` (`[STATS]` cadence; `[LAYER-HEALTH]` "at the stats cadence"), `:54` (checklist item 6 still names `[BATCH-STATS]` as a value-visible tag — now only alongside step lines), `:185` (`[LAYER-HEALTH]` rides every step line; the 25-step bootstrap note goes);
  - `:136` (the architecture paragraph's format list: "current v10" → v11, with the v11 rule: no new architecture field; `training_step` is the trainer step; older files are read by their writer's meaning and flagged on load);
  - `:99` (File lineage: on a trainer-state file `training_step`, `trainer_completed_steps` and `cum_trainer_step` are one value, refused by writer and reader otherwise; the new mirror `lineage_segment_local_step`);
  - `:191` (the step axis: `cum_step = segment.cumstep_base + segment step`, the segment step being the record's `segment_local_step` from v11 and `training_step` before);
  - `:200` (names by trainer step; `-seg<k>` names are legacy; `PROBE_ABOVE_STEP`; reachable range above the start);
  - `:202` (identify checkpoints by `model_id` + `training_step` still holds; `training_step` is the trainer step from v11, the writing segment's step on older CLI files).
- `documentation/UCI.md`: `:209-218` (every trainer-step multiple of 1,000; names by trainer step; a resumed segment may keep its stem) and `:286` (step-line cadence).
- `Network/ArchitectureFormat.swift` version history (v11), `TrainerResumeState.swift:25-29`, `LineageRecord.swift:184`, `CorpusReplayRunner.swift:276-278`, `LineageTracker.swift:362-368` (D5).
- `experiments/probe_loop.sh` header, `experiments/probe_record.py` docstring, `scripts/dcm_lineage.py` / `scripts/dcm_arch.py` module docs, the dashboards' comments (D9).
- `documentation/deriving-models.md`: no change (its `training_step` / `source_training_step` rules only test "above zero").
- `CHANGELOG.md` entry, naming the format bump and that builds before it cannot read files written after it.
- Adjacent plans' text that cites the old cadence, names or `training_step` meaning is revised by whichever lands second (see Order with adjacent plans).

### Determinism and exact resume

- Logging (D1, D6, D8) never touches trainer, optimizer, buffer or RNG state.
- D2's forced steps depend on the trainer step only; at interval 10 they add none.
- Saves (D3) export only; they draw nothing (`exportResumeSnapshot`, `samplerState()`, `dropoutStreamState()` are reads; probe isolation holds). Moving them from segment to trainer multiples changes which steps are saved, not what is trained.
- A resume that starts mid-interval is already exact: arm C resumed at trainer step 513 (a final save, not a multiple of anything) and logged `[RESUME] EXACT` (`dcm_log_20261005-121841.txt`). The feed position, sampler and dropout streams are saved at whatever step a save lands on.
- The first save after a mid-interval resume is the next trainer multiple of 1,000; validated live (V-2).
- OD-1 changes what headers state, never training state: an exact resume reads the trainer clock from the file's schedule (`TrainerResumeSnapshot`), not from `training_step`, so a resume across the format line (a v10 file resumed by a v11 build) restores the same state; the behavior fingerprint does not include the format version (`Training/BehaviorFingerprint.swift`), so the format bump adds no `build` gap.

### Order with adjacent plans

- **`TRAINING_HEALTH_ALARMS_PLAN.md`** (its Part P requirements):
  - R0 — at a save step the live readout and evaluation run before the save's checkpoint pass: **holds** at every 1,000-step save. The line block precedes the save and every checkpoint step is a fixed line step (the logged readout); the alarms plan's 50-step evaluation sits between the line block and the save block (D1 CLI wiring), and every checkpoint step is a multiple of 50 (the evaluation). The final save, at an arbitrary step, has no line before it, as today (OD-9); whether an evaluation precedes it is the alarms plan's rule, not this plan's.
  - D6 — saves 1,000 trainer steps apart, the first within 1,000 of the start: **holds** (D3).
  - D4 — `[REPLAY]` / `[VS-UCI]` field format: **kept** (D1).
  - **The decoupled evaluation is now in the alarms plan's own text** (its OD-21 and status, lines 3-4, revised after 2026-10-05): live evaluations every 50 trainer steps on every path, the per-step order step-line block → live evaluation → save block, one live read serving both on a shared step (OD-15, adopted there; its P2 hands the step line's `LayerHealthLog.LiveOutcome` to the evaluation, so the line block exposes it), and an explicit evaluation before the final save's checkpoint pass (its R0, `:202`). Nothing in this plan conflicts. Its GUI row (`:203`) still cites today's `[STATS]` cadence (25 steps for 500, then 60 s) as logging only; whichever plan lands second updates that citation.
- **`HPARAM_RECORDING_PLAN.md`:** its V2/V4 commands name `$S/v2-replay-seg1-step200.safetensors` (`:1208`, `:1226`); under D4 that file is `$S/v2-replay-step400.safetensors` (segment 1 starts at trainer step 200 and ends at 400), and under D5 its header's `training_step` is 400 with segment step 200 in the record and the `lineage_segment_local_step` mirror. Its V2 statement that the rolling file holds the start model's `model_id` at its `training_step` (`:1198`) stays true. Its lineage schema 2 → 3 is independent of the format bump here (different version numbers); whichever plan lands second rebases and fixes the file names. Gap 10's applicability table gains `step_line_interval_sec` on every path.
- **`LICHESS_BOT_FOLLOW_LINEAGE_PLAN.md`** (mid-implementation; whichever lands second applies these):
  - Ranking is **unchanged**: it ranks within a run by chain depth, then the record's `segment_local_step`, then `recorded_unix` (`:121`), and never by `training_step` across segments (`:123`); `segment_local_step` keeps its meaning.
  - `ModelFileEntry.trainingStep` becomes the reading's `trainerStepOrStatedStep` (D5), so "within a line, latest is the highest `training_step`" (`:63`) still holds (one `model_id`, one basis) and the step shown is the trainer step.
  - `:287` ("`trainingStep` stays the file's `training_step`") becomes "the file's trainer step, or its stated step where none is recorded (`trainingStepReading.trainerStepOrStatedStep`)", set at `LichessBotModelSlots.swift:189`.
  - `:123`'s parenthetical "(segment-local)" becomes "(the segment's step before format v11, the trainer step from v11; not used across segments for the reason given against option B, `:122`)"; `:70`'s "(the CLI files' `training_step`)" for `segment_local_step` becomes "(equal to `training_step` on CLI files before v11)".
  - Its live check (`:556-562`) reads a v11 file's `training_step` as the trainer step; the expected answer is taken from the same header, so the check is unchanged in form.
- **`TRAINING_HEALTH_ALARMS_PLAN.md` fixtures:** its reference fixtures come from `bn_liveness.py`, whose output for every existing file is unchanged by D9 (same keys, same checks before v11).
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
| `experiments/20261005-lr-schedule-ab/bn_liveness.py` | `:54-55`, `:249-268`, `:405-420` | D9 (OD-16, decided: now) |
| `Network/ArchitectureFormat.swift` | `:77`, version history | D5: format v11 |
| `Persistence/SafetensorsModelIO.swift` | `:40-48`, `:145-163`, `:389-439` | D5: encode/decode checks, `trainingStepReading`, `readParentFile`; `trainerClock` removed |
| `Persistence/ModelCheckpointFile.swift` | `:60-151`, `:370-372` | D5: creator constants, the reading on decode, `lineageParent` |
| `Persistence/LineageRecord.swift`, `Persistence/LineageTracker.swift` | `:184`, `:879-890`; `:362-368` | D5: mirror key, docs |
| `Persistence/ModelFileCatalog.swift` | `:116-160` | D5: entry step from the reading |
| `Persistence/CheckpointManager.swift` | `:1811-1870`, `:1921-1928` | D5: the training-step line for a `.dcmmodel`, which has no `DecodeFormat` |
| `App/NumericsAuditCLI.swift` (D5) | `:73` | the audit's step from the reading |
| `CLI/CorpusReplayRunner.swift` (D5) | `:276-304`, `:1112-1116`, `:1566-1569` | identity from the reading; writer step |
| `CLI/TrainVsUciRunner.swift` (D5) | `:535-540`, `:608-612` | writer steps |
| `App/SessionController+Checkpoint.swift`, `App/SessionController+Arena.swift` | `:541-543`; `:713-715` | D5: GUI trainer file states the snapshot clock |
| `App/NumericsAuditCLI.swift`, `LichessBot/Play/LichessBotModelSlots.swift`, `Training/TrainerResumeState.swift` | `:73, :154`; `:189`; `:25-29` | D5 |
| `Persistence/ModelDerivation.swift`, `Persistence/ModelGraft.swift` | `:386`; `:377` | D5: parent clock from the reading |
| `scripts/dcm_lineage.py`, `scripts/dcm_arch.py` | `:481-499`; `:73-76` | D9: `step_reading`, format 11 |
| `documentation/dashboards/ckpt_inventory.py` | `:5-12`, `:41`, `:128` | D9 |
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
- `testAResumedRunSavesAndNamesItsCheckpointsByTrainerStep` — train 990 steps with `--enumerate-checkpoints` (final `S-replay-step990`), then `--resume-exact` from that file for 30 steps into the same `--out-model`. Expect `S-replay-step1000` (header `training_step` 1,000 = `trainer_completed_steps`; the record's `segment_local_step` and the `lineage_segment_local_step` mirror 10; `dcm_format_version` 11) and `S-replay-step1020` (1,020; 20), no `-seg` file, a stats row at `cum_trainer_step` 1,000 with diagnostics, `[RESUME] EXACT`. Fails before the change (no save at 1,000; the final file is `S-replay-seg1-step30`). Its run time is measured in P1 and recorded here; it is a correctness test and is not gated.

Pure, `DrewsChessMachineTests/TrainingStepBasisTests.swift` (new; OD-1, D5):
- `testAFileFromV11StatesTheTrainerStepWithTheSegmentStepAsTheSidecar` — a trainer file written as the runners write theirs: header `training_step` = `trainer_completed_steps` = the record's `cum_trainer_step`; mirror `lineage_segment_local_step` = the record's `segment_local_step`; reading `.trainerStep`.
- `testEncodingRefusesATrainerFileWhoseStepIsNotItsClock` and `testDecodingRefusesAV11TrainerFileWhoseStepIsNotItsClock` (header rewritten after encoding).
- `testAPreV11ReplayFileReadsItsStepAsTheSegmentStep` (rewritten to v10, creator `replay`, schedule present: trainer step = the clock, segment step = `training_step`), `testAPreV11ReplayFileWithoutAScheduleHasNoTrainerStep` (v3, no record: `trainerStep` none, `trainerStepOrStatedStep` and `lineageParent.trainerCompletedSteps` = the stated step), `testAPreV11GUIFileReadsItsStepAsTheTrainerStep` (creator `manual`), `testAPreV11FileOfAnUnknownWriterKeepsItsStatedStep` (creator `test`: trainer step and segment step both the stated step).
- `testTheCreatorNamesTheWriterNotTheRecordsPathKind` — a v8 GUI champion file (creator `manual`) whose record is a corpus-replay record (`path_kind` `replay`, as `championFileLineageRecord` writes for a champion loaded from a replay file) reads as `.legacyGUITrainerStep`, its `training_step` the trainer step.
- `testACopyOfASchedulelessPreV7ReplayFileStillReadsAsTrained` — a v3 `replay` file without a schedule (stated step 41,000) as a GUI champion's origin: `championFileTrainingStep` is 41,000 and `ModelDerivation.requireUntrainedSource` refuses a derive of the saved champion file (`sourceIsTrained`). Fails under the earlier rule (parent step null).
- `testADcmmodelFileIsReadByItsCreatorAndFlaggedByItsLoader` — a `.dcmmodel` with creator `periodic` reads as `.legacyGUITrainerStep`, and `CheckpointManager.loadModelFile` logs its one training-step legacy line.
- `testTheGUICreatorsAreTheSaveTriggerTags` — the constant equals `Set(SessionSaveTrigger.allCases.map(\.diskTag))` ∪ {`SessionSaveTrigger.promotionDiskTag`}.
- `testAnOlderFileIsFlaggedOnceAndAV11FileNotAtAll` — the decode's legacy log holds exactly one training-step entry for a v10 file and none for a v11 file.
- `testTheRollingFileIdentityUsesTheReading` — `TrainerModelFileIdentity.read(from:)` on a v11 trainer file and on a v8 replay trainer file of the same model line gives the trainer step.
- `testAFileStatingNoStepIsNeverFlagged` — a v10 `new-model` file decodes with no training-step legacy entry.

Python (OD-1, D9), new methods: `test_lineage.py` — `step_reading` on the same cases as `TrainingStepBasisTests` and a `SwiftMirrorTests` method for the new constants; `test_replay_probe.py` — `meta_step_of`, `_ckpt_index` and the glob scan on a v11 resumed-segment file (filed at `cumstep_base` + segment step), `freeze` refusing a same-named file of another `model_id`, and `discover_enum_stems` refusing a stem shared by two segments' `model_id`s; `test_tooling.py` — a v11 resumed-segment file's probe record carries `step` = trainer step, `segment_step` and `step_basis` after the four leading keys. `bn_liveness.py --selftest` gains v11 cases.

Python (`documentation/dashboards/tests/`), cadence and naming (beside the OD-1 cases above):
- `test_tooling.py`: a `probe_loop.sh` test that `PROBE_ABOVE_STEP` skips an earlier segment's files under the same stem (other `model_id`) and that `PROBE_SEGMENT` with `PROBE_ABOVE_STEP` is refused; a `probe_record.py` test that a record appended to a probes file of records without `step_basis` is accepted.
- A new `vsuci.py` test: rows on `trainerStep` multiples of 1,000 for a log in today's line format that carries the field; today's rows for one that does not.
- `table_common.buffer_plies_per_game_by_trainer_step` on a two-segment synthetic log.
- `test_lineage.py`: `step_reading` refuses a version newer than `CURRENT_FORMAT_VERSION`, refuses a v11 trainer header whose `training_step` is not `trainer_completed_steps`, reads by `creator` (a v8 header with creator `manual` and a `replay` record reads as the GUI basis), and gives a creator-less pre-v11 header without a record `training_step` as both steps.
(The earlier list's mixed-basis refusal, "glob scan reports a trainer-step-named file as a failure" and "`discover_enum_stems` skips it" belonged to the design before OD-1 and are dropped: from v11 a name step always equals `training_step`, and D9 files a v11 file by its record's segment step.)

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
- **TE-5** No Python test edit is expected for the cadence and naming work (the probe-loop and lineage fixtures use segment-0 names; the new record keys go after the four leading keys `test_good_probe_becomes_a_record_with_identity` pins, `test_tooling.py:154`). OD-1's edits are TE-P1 and TE-P2 below. Any further edit found necessary is listed here for approval before it is made.
- **Deletions requiring the owner's express approval — exactly two:** `SegmentIndexedCheckpointNamingTests.testALaterSegmentsNamesCarryItsIndex` (`:26-33`) and `SegmentIndexedCheckpointNamingTests.testEachSegmentParsesOnlyItsOwnStepFiles` (`:35-50`). One assertion removed: `testTheSegmentIndexComesFromTheLineageRule` `:94-96`. No other existing test is deleted or loses an assertion.
- **OD-1 test edits (each needs the owner's approval; no test can keep passing unedited, because a trainer-state file whose `training_step` is not its clock is now refused):**
  - **TE-6** `ExactResumeTests.swift:264` `trainingStep: 17,` → `trainingStep: snapshot.schedule.completedTrainSteps,`; `:276` `// \`training_step\` stays the segment-local value it was given.` → `// \`training_step\` is the trainer step (format v11).`; `:277` `XCTAssertEqual(file.metadata.trainingStep, 17)` → `XCTAssertEqual(file.metadata.trainingStep, snapshot.schedule.completedTrainSteps)`. (Its clock is 5, `:53`, `:215`.)
  - **TE-7** `ExactResumeTests.swift:346` `creator: "manual", trainingStep: 0, parentModelID: "20260929-1-TEST", notes: "trainer",` → `trainingStep: snapshot.schedule.completedTrainSteps` (clock 5).
  - **TE-8** `DropoutRNGStateTests.swift:186` `creator: "replay", trainingStep: 0, …` → `trainingStep: snapshot.schedule.completedTrainSteps` (clock 2, `:172`).
  - **TE-9** `PolicyTailPrecisionProvenanceTests.swift:79` `trainingStep: 1,` → `trainingStep: snapshot.schedule.completedTrainSteps,` (a fresh trainer, clock 0).
  - **TE-10** `SafetensorsDecodeStrictnessTests.swift:80` `("trainer", try encoded(trainingStep: 12, schedule: schedule)),` → `encoded(trainingStep: schedule.completedTrainSteps, schedule: schedule)` (345, `:76`); the assertions at `:86-88` are unchanged.
  - **TE-P1** `documentation/dashboards/tests/test_dcm_arch_site_activations.py:196` `for version in ("11", "0", "-1", "nine"):` → `for version in (str(dcm_arch.CURRENT_FORMAT_VERSION + 1), "0", "-1", "nine"):` (11 becomes a supported version; the new form tracks the constant, as the Swift tests' `currentVersion + 1` does).
  - **TE-P2** `experiments/20261005-lr-schedule-ab/bn_liveness.py:406` (its `--selftest`) `dcm_format_version="11"` → `"12"`, so the case still tests a future version rather than passing on the retired-field rule.
  - Checked and **not** edited: every other test that writes a trainer-state file states its clock (`ChampionLineageRecordTests.swift:164`, `GuiResumeContinuationGapsTests.swift:47`, `GuiLineageLifecycleTests.swift:64`, `GuiResumeGapsTests.swift:48`, `LineageRecordTests.swift:38`, `LineageProvenanceTests.swift:44`, `TrainVsUciSessionTests.swift:199`, `ExactResumeTests.swift:405, 454`); plain-file fixtures carry no schedule; `ChampionLineageRecordTests`, `UntrainedCopyRecordTests`, `UntrackedTrainerParentTests` build `ParentFile` directly; `LineageRecordTests.testOlderFormatFileLoadsWithItsLineageUnrecorded` (`:199-212`) uses creator `test` and keeps 40; the catalog and guard tests' fixtures carry no creator (unknown writer, stated step); format-version pins are symbolic (`ArchitectureFormat.currentVersion`); `test_tooling.py:154` holds (new record keys go last).

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
  - `--start-model $S/cad-replay-step1013.safetensors --resume-exact --training-step-limit 1100` into the same `--out-model`: `[RESUME] EXACT`; first line at trainer step 1,014; every later line carries diagnostics; save and line at 2,000 with the line before the `saved trainer model` line and before the checkpoint `[LAYER-HEALTH]` block; files `cad-replay-step2000` (header `dcm_format_version` 11, `training_step` 2,000 = `trainer_completed_steps`; record `segment_local_step` and mirror 987) and `cad-replay-step2113`; no `-seg` file.
  - The same resume command again: refused at pre-flight by the rolling-output check first (`checkRollingOutput` runs before the enumerated scan, `CorpusReplayRunner.swift:1111` vs `:1129`; the rolling file now holds the resumed run's own `model_id`, not the start file's). Then the same command with `--overwrite-out-model` (scratch rolling file only): refused by the enumerated scan, naming `cad-replay-step2000` … `cad-replay-step2113` (2 files) and the reachable range `1014…2113`.
  - Header-only check by a short script: `dcm_format_version`, `model_id`, `training_step`, `trainer_completed_steps`, `lineage_segment_local_step`, lineage `segment_index` / `segment_local_step` / `cum_trainer_step` of all four files; `training_step` = `trainer_completed_steps` = `cum_trainer_step` on each.
  - Log volume: `[BATCH-STATS]` bytes against arm C's log at the same per-step rate.
- **V-3 Train-vs-UCI** (owner machine with an engine): `--train-vs-uci "cmd=/opt/homebrew/bin/stockfish;n=1;go=nodes 1" … --training-step-limit 1013 --enumerate-checkpoints --checkpoint-stem $S/uci`, then `--resume-exact` from its final session for 1,100 with the same `--checkpoint-stem` (without it a session-folder start names the stem after the new run's model ID, `CLI/TrainVsUciSession.swift:263-267`, and the shared-stem case is not exercised): the same line and name checks (`uci-vsuci-step2000`, `uci-vsuci-step2113`, and segment 0's `uci-vsuci-step1000`, `uci-vsuci-step1013` untouched).
- **V-4 Tooling on the new names:** `PROBE_ABOVE_STEP=1013 experiments/probe_loop.sh --once cad $S/probes-seg1.jsonl` records steps 2,000 and 2,113 (`segment_step` 987 and 1,100, `step_basis` `trainer_step`) and never probes `cad-replay-step1000` / `1013`; `experiments/probe_loop.sh --once cad $S/probes-seg0.jsonl 1013` (segment 0, step limit 1,013) records 1,000 and 1,013 (`step_basis` `trainer_step`; for segment 0 the segment step is the same number) and logs the two later files as skipped; `PROBE_BIN` is the v11 build for both; the same without the limit stops at `cad-replay-step2000` with exit 4 (the other-run check) — expected, and recorded as such. Appending one new record to a copy of an existing segment-0 probes file (records without `step_basis`) succeeds. `replay.py discover-stems` (no `--write`) refuses the scratch stem `cad` because it spans two `model_id`s, naming `derive-registry` as the way to place those segments. After P4 lands, each of the four live LR probe loops (D9) records its next checkpoint (same `step` basis as its file's earlier records, new keys last). No command that writes `registry.json` or `data/` is run on production files.
- **V-5 Old files still read:** `--probe-model` on `20261005-lrC-cyc10-r1-replay-seg1-step1000.safetensors`; `PROBE_SEGMENT=1 probe_loop.sh --once 20261005-lrC-cyc10-r1 $S/probes-C-seg1-recheck.jsonl` reproduces `experiments/20261005-lr-schedule-ab/probes-C-seg1.jsonl`'s steps, with `step_basis` `legacy_segment_step` and `trainer_step` = segment step + 513 (e.g. 1,513 for `seg1-step1000`); loading that file in the app logs one `[ARCH] legacy file … training_step 1000 is the writing segment's step …; trainer step 1513` line; `bn_liveness.py` over every arm reproduces its committed output; a `--resume-exact` from that file into a scratch `--out-model` passes pre-flight and names its files by trainer step; `vsuci.py` and `replay.py` rebuilds of existing runs produce no CSV change (dry comparison against the committed CSVs, written to `$S`).
- **V-6 GUI:** a fresh Play-and-Train to ≈ 1,100 trainer steps: `[STATS]` at the first step, every 50 through 1,000, then 1,000 and time lines; `[BATCH-STATS]` beside lines; the new Sessions-tab field; a live change of the interval takes effect on the next deadline; a promotion's rewind adds no fixed line; `--train` `results.json` rows at the same ticks.

- **V-7 Format line and old files (OD-1):** a read-only survey with `dcm_lineage.step_reading` over every header in `Models/` and `Sessions/*/` (4,529 on 2026-10-06): no file that states a `training_step` reads as `legacy_unknown_writer` (they are all `replay`, `train-vs-uci` or GUI creators; the 21 `.dcmmodel` files are GUI saves); no file has a `path_kind` other than its creator's path (the case D5 reads by creator; none on this Mac on 2026-10-06); every pre-v11 file that has both a trainer schedule and a lineage record has `trainer_completed_steps` = `cum_trainer_step`; the counts per basis are recorded here. A GUI load of a v10 session logs exactly one training-step legacy line per file loaded; a v11 load logs none. The frozen probe build (`PROBE_BIN` default) refuses a v11 file with "newer than this build supports", and a stale `scripts/dcm_arch.py` (format 10) refuses it too — confirming that stale readers fail loudly.

---

## Phasing

Each phase builds and commits on its own once its tests pass (P1 and P2 together, per the failing-first sequence).

- **P1 + P2 — Core and CLI.** D1, D2, D3, D4, D5 (format v11, the reading, writer steps, encode/decode checks, every Swift reader, the mirror key), D6 (CLI side, trainer, `SyncBox`), D7 steps 1–6 and 8–9; owner-approved TE-1 to TE-4 and TE-6 to TE-10; new Swift tests. V-1 (Swift), V-2, V-7 (Swift side).
- **P3 — GUI.** D8, D6's GUI side, D7 step 7. V-6.
- **P4 — Tooling.** D9 (including `step_reading`, `CURRENT_FORMAT_VERSION` 11 and `bn_liveness.py`), TE-P1, TE-P2 and the Python tests. V-4, V-5, V-7 (survey). **Lands in the same push as P1 + P2:** the current tools refuse v11 files, so tracking and probing of any run on the P2 build stop until P4 is in (loudly; nothing is misfiled). It also must precede any probing of a resumed segment (the old probe loop would reach earlier segments' files and stop on the other-run check).
- **P5 — Documentation.** D10, CHANGELOG. V-3 when an engine run is available.

---

## Risks

- **Fewer lines between fixed points.** Analysis that read every 50th segment step now gets time lines at wall-clock-dependent steps; A/B arms log at different steps except on the fixed lines (every 50 through 1,000, every 1,000). Comparisons should use those.
- **GUI steady cadence 60 s → 180 s by default.** `--train` `results.json` gets a third of today's steady rows (plus the dense and 1,000-step rows), the chart's legal-entropy trace and the `[ALARM] policy entropy` log line refresh at the line cadence (D8). Alarm evaluations are unaffected: they run every 50 trainer steps independent of the line (approved direction item 5). `--train` parameters files can set `step_line_interval_sec` 60.
- **A second resume of an earlier file into the same stem is refused** where `-seg<k>` names would sometimes have avoided the collision; the refusal names the files and suggests a new stem.
- **`training_step` means two things across the format line.** Before v11 it is what its writer wrote (the segment's step on corpus-replay and train-vs-UCI files); from v11 the trainer step. Every in-repo reader goes through the reading (D5, D9) and stale in-repo tools refuse v11 files; a reader outside the repo, or a frozen dated experiment script, run on a v11 file reads `training_step` as written. The legacy line flags every older file the app loads.
- **Older builds cannot read new files.** Every file written from v11 on — models, sessions, presets, `architecture.json` — is refused by a build before this change, including the frozen build `probe_loop.sh` uses by default; going back to an older build mid-run means it cannot read the run's newer checkpoints.
- **A pre-v7 CLI file without a trainer schedule states only its segment step** (3,811 v3 `replay` and 31 v3 `train-vs-uci` files on this Mac). Its trainer step is unknowable; it is still its parent step ("the parent's stated step", D5), so a GUI champion loaded from one states that segment step as its `training_step` — from v11, under the trainer-step meaning, as it does today under the old one. The source's legacy line at load is the only flag.
- **Files written by this build cannot be used by the frozen builds running experiments.** Presets live in one shared folder (`ArchitecturePresetStore.presetsDirURL`, `~/Library/Application Support/DrewsChessMachine/Presets/`, all ten at `format_version` 10 today) and are read by `--architecture <name>` in every build; a preset saved by this build, a seed minted by its `--new-model` or a model it derives is format 11 and is refused by `DCM-2331-…` and every other frozen build, though nothing in its architecture is new. An A/B series meant to continue on a frozen build must mint its seeds and presets with that build (OD-18).
- **Five Swift tests and two Python literals must change for OD-1** (TE-6 to TE-10, TE-P1, TE-P2); none can pass unedited.
- **Forced diagnostics at a non-default `batch_stats_interval`** (not dividing 50) change which steps run the diagnostic graph compared with older builds, so such a run's numerics may differ bitwise from an older build's at those steps. On those steps the trainer's non-finite check also covers `valueMean` and entropy (`Training/ChessTrainer.swift:7089-7102`, gated on `includeDiagnostics`), so a run that would have gone on with a non-finite diagnostic and finite losses stops there instead; the GUI's rolling means also take those extra samples. No run uses such an interval today; the behavior fingerprint does not see it (it trains with `batchStatsInterval = 0`, whose fallback diagnostics already cover every fixed line step).
- **`vsuci.py` limit:** a train-vs-UCI log from a build that wrote `trainerStep=` but the old segment cadence, resumed at an offset that is not a multiple of 1,000, would get no rows. None is registered.
- **Adjacent plans cite the old names and cadence** (alarms, HPARAM); listed above for whichever lands second.

---

## Owner decisions

All decided by the owner on 2026-10-06 unless marked otherwise.

- **OD-1** `training_step` meaning. Recommended was: keep it segment-local, names carry the trainer step. **Decided (owner, 2026-10-06): reversed** — "we already have cum_trainer_step. I don't care if old files are different. that's ok. I'd rather maybe flip things -- have an overall training step be the only step we reference 99% of the time, and have a separate 'step within this segment/run' as a sidecar. We could update old files to the best of our ability, or update a version somewhere so that we can always flag if an older file is loaded." Implemented by D5 (writers, readers, checks) and D9 (tools); no file on disk is rewritten.
  - *Decided by implementer:* the marker is **architecture format v11**, not a dedicated key — one versioning convention, and stale readers refuse new files instead of misreading them (D5).
  - *Decided by implementer:* the segment step stays in the lineage record and gains a flat mirror `lineage_segment_local_step` (never read back), so a header shows both numbers.
  - *Decided by implementer:* a trainer-state file's `training_step` must equal its clock — refused on encode and, from v11, on decode; `trainerFile(creator:trainingStep:…)` keeps its signature (five test call sites change, not thirteen).
  - *Decided by implementer, corrected in Review 2:* older files are read by their writer — the file's `creator` (not the record's `path_kind`, which on a GUI champion file names its weights' origin) — with named constants pinned by tests; an unknown writer keeps today's readings and is flagged rather than refused; `.dcmmodel` files are read the same way and flagged by their loaders.
  - *Decided by implementer, reversed in Review 2:* `ParentFile.trainerCompletedSteps` comes from the reading's `trainerStepOrStatedStep` (the parent's stated step, as designed), not its `trainerStep` — a null there let a copy of trained weights pass the derive "untrained source" check.
  - *Decided by implementer:* the GUI trainer file states the trainer snapshot's clock instead of the stats box's count (equal at every cut, pinned by `SessionSaveConsistentCutTests`).
  - *Decided by implementer:* log lines and `results.json` fields keep their names and meanings.
- **OD-2** Name shape `<stem>-<tag>-step<trainerStep>`, no `-seg<k>`. **Decided (owner, 2026-10-06): as recommended.** Under OD-1 the name step and the header's `training_step` are the same number again (D4).
- **OD-3** A segment that trained no step writes no enumerated copy and logs that (also drops a fresh corpus-replay run's `-step0`). **Decided (owner, 2026-10-06): as recommended.**
- **OD-4** Every fixed line step is also a diagnostics and batch-stats step (no change at interval 10). **Decided (owner, 2026-10-06): as recommended.**
- **OD-5** `[BATCH-STATS]` rides the step line, no new parameter. **Decided (owner, 2026-10-06): as recommended.**
- **OD-6** The GUI's per-step-for-500 bootstrap (with its 25-step strides) merges into the shared schedule. **Decided (owner, 2026-10-06): as recommended.**
- **OD-7** Any line restarts the time interval. **Decided (owner, 2026-10-06): as recommended.**
- **OD-8** The time rule also applies during the dense phase. **Decided (owner, 2026-10-06): as recommended.**
- **OD-9** No extra line before the final save. **Decided (owner, 2026-10-06): as recommended.**
- **OD-10** `step_line_interval_sec`: Double, default 180, range 10…86,400, Observability, live-tunable, `absentValue: .currentSetting`, Sessions tab beside the KL-probe interval. **Decided (owner, 2026-10-06): as recommended.**
- **OD-11** Keep `batch_stats_interval`'s id and update its description. **Decided (owner, 2026-10-06): as recommended.**
- **OD-12a** Rename the naming API's `step:` labels to `trainerStep:` (mechanical test edits in TE-2). **Decided (owner, 2026-10-06): as recommended.**
- **OD-12b** Probe loop and records. **Decided (owner, 2026-10-06): as recommended, restated for OD-1** — `PROBE_ABOVE_STEP` (new), `PROBE_SEGMENT` for legacy names only, the step-limit argument read as the name's step (a trainer step on v11 files); the identity rule is **unchanged** (name step = `training_step` in both eras), so the earlier two-basis identity and the mixed-basis refusal are dropped; records gain `trainer_step`, `segment_step` and `step_basis` from `dcm_lineage.step_reading`, after the four leading keys (D9).
- **OD-13** `table_common`: add a trainer-step helper and keep the old one for the dated experiments. **Decided (owner, 2026-10-06): as recommended.** Under OD-1 the new helper's keys are the same steps new probe records carry as `step`, so the two join directly (D9).
- **OD-14** `lastBatchStatsSummary` / `lastBatchStatsUniquePct` under one `SyncBox`. **Decided (owner, 2026-10-06): as recommended.**
- **OD-15** One live `[LAYER-HEALTH]` read serves both the step line and the alarm evaluation when they coincide (implemented by the alarms plan). **Decided (owner, 2026-10-06): as recommended.**
- **OD-16** `bn_liveness.py`. Recommended was: update it only if an arm is resumed on the P2 build. **Decided (owner, 2026-10-06): update it now** — it stays in use (the alarms implementation makes reference fixtures with it; future experiments will run it) and must understand new files (D9, P4).
- **OD-B** Approve test edits TE-1 to TE-4, including the **two test deletions** and **one removed assertion** listed under TE-4 (TE-5: none expected). **Decided (owner, 2026-10-06): approved**, including deleting `testALaterSegmentsNamesCarryItsIndex` and `testEachSegmentParsesOnlyItsOwnStepFiles` and removing the `-seg1-` name assertion from `testTheSegmentIndexComesFromTheLineageRule`.
- **OD-17** The OD-1 test edits TE-6 to TE-10, TE-P1 and TE-P2 (under Existing-test edits), with their exact before / after. (Review 2 re-searched: no other test pins the old meaning; see Review 2.) **Decided (owner, 2026-10-06): approved.**
- **OD-18 (new, needs the owner's decision)** Format v11 also stamps presets, `architecture.json` and new seeds, which frozen builds then cannot read (Risks). Recommended: accept, and mint seeds and presets for frozen-build experiments with those builds — or carry the marker on safetensors model files only (a second version number for one fact, against D5's single-marker reason). **Decided (team lead, under the owner's 2026-10-06 delegation "solve the problem yourself"): accepted as recommended** — one version number for one fact; a frozen-build A/B series mints its seeds and presets with its own build.
- **OD-C** (old) Moot: every checkpoint step is now a fixed line step and carries diagnostics.

---

## Non-goals

- `[LEGAL-COST]` and `[SAMPLER]` keep the batch-stats cadence (small lines).
- `--training-step-limit`, `--gpu-capture-step` and the train-vs-UCI eval-sync cadence stay segment-local.
- Renaming files on disk, or re-logging the fields missing from arm C's resumed log (they were never computed).
- Rewriting, re-stamping or renaming any existing file to the new `training_step` meaning (owner: never overwrite production data; a rewrite would need its own approval).
- Renaming the log lines' `step=` / `trainerStep=` or the `results.json` row's `steps` / `cum_trainer_step` (the alarms plan's D4 keeps the line format).
- Search of any kind; this plan changes logging, save points, names and what a header's `training_step` states.

---

## Review (2026-10-06, independent review of the redesign)

Kept as the record of that review. Its recommendations were then decided by the owner; where they differ (OD-1, OD-16), Owner decisions and Revision below are authoritative.

Scope: every design section against `main` at `5138c628`, the alarms plan (Part P, R0, D4, D6 and its cadence text), `HPARAM_RECORDING_PLAN.md` and `LICHESS_BOT_FOLLOW_LINEAGE_PLAN.md` where they touch names, `training_step` or `segment_local_step`, and the owner's 2026-10-06 decision that alarm evaluations run every 50 trainer steps independent of log lines. Over 40 file:line citations were opened; all but the ones fixed below point at what the plan says. Arm C's numbers were re-measured from the logs (`[BATCH-STATS]` 560 lines / 43,117,507 B of 43,457,830 B; 113 of 113 step lines `pEnt=--`; last step 5,603 at trainer step 6,116; first segment's final save at 513).

### Verified, no change

- D2: `includeDiagnostics` / `isStatsStep` are functions of `_completedTrainSteps + 1` and the interval only (`Training/ChessTrainer.swift:4649-4657`); the sampler's metadata pointers are only written (`Training/ReplayBuffer.swift:1873-1878`), so a forced batch-stats step draws the same indices; the KL probe's schedule is by step index. Every fixed line step is a multiple of 10, so at interval 10 and at 0 (fallback 10) the diagnostic and batch-stats step sets are unchanged; `BehaviorFingerprint` trains at interval 0 (`:263`). Exact-resume determinism holds: the rule never depends on where a process started.
- D3: the only save call sites are `CorpusReplayRunner.swift:1981, 2009` and `TrainVsUciRunner.swift:848, 852, 872, 874`; an epoch-budget, corpus-end, step-limit or abort end goes through the final save at an arbitrary step, unchanged. A step-limit run has no epoch bound (`CorpusReplayRunner.swift:1219`), so the 990-step runner test is not cut short by the synthetic corpus's short epochs, and its 75-game refeed window fits the 80-game corpus (`:1345-1360`).
- D5: the bot lineage plan ranks within a segment by `segment_local_step` and never by filename or cross-segment `training_step` (`LICHESS_BOT_FOLLOW_LINEAGE_PLAN.md:63,121-123`); keeping `training_step` segment-local leaves it unaffected. HPARAM `:1208`/`:1226` file names and the segment-1 arithmetic (200 → 400) are right.
- D7: every `TrainingParameters.swift` / popover / session citation matches; `test_registry_size` is 85 today.
- D8: the GUI stats box is the cumulative trainer step, seeded on resume and rewound on promotion (`App/SessionController+Arena.swift:480-488`), so D1 rule 2 is needed and correct. **[Correction, final training-side review m5, 2026-10-07: except after "New Session, keep trainer", where the fresh box counts the session from 0 on a trained trainer. D1 rule 2 is still needed (a promotion rewinds the trainer clock), but the GUI line is now keyed on the trainer clock itself, not the box.]**

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

---

## Revision (2026-10-06, owner decisions)

- OD-1 reversed by the owner. D5 rewritten: from architecture format v11 every writer states the trainer step as `training_step`; the segment step is the record's `segment_local_step` plus a new mirror `lineage_segment_local_step`; encode and decode refuse a trainer-state file whose `training_step` is not its clock; one reading (`SafetensorsModelIO.trainingStepReading`, mirrored as `dcm_lineage.step_reading`) gives every reader the trainer step and segment step of any file, with older files read by their writer and flagged once per load. Every Swift reader of the old meaning is listed with its new rule (guard identity, catalog, Lichess bot, numerics audit, derive and graft, `ParentFile`, GUI champion and trainer files, GUI sessions).
- D9 rewritten for OD-1: probe records keep their identity rule; `replay.py` takes its segment step from the reading in `meta_step_of`, `lineage_checkpoints`, `discover_enum_stems`, `freeze`, the glob scan, `_ckpt_index` and `import_probes`; `ckpt_inventory.py` and `bn_liveness.py` (OD-16, now) read through the reading; `scripts/dcm_arch.py` moves to format 11. The earlier D9 rules keyed on "the header's `training_step` differs from the name's step" are gone: from v11 the two always agree.
- D4: the name step and `training_step` agree in both eras. D10: CLAUDE.md lines `:99`, `:136`, `:191`, `:202` added, and the code docs D5 touches. Determinism: one bullet (the schedule, not `training_step`, drives resume; the format version is not in the behavior fingerprint).
- Order with adjacent plans: what changes for the Lichess bot follow-lineage plan (ranking unchanged; displayed step and three text lines); the HPARAM plan's V2/V4 file gets `training_step` 400.
- Tests: `TrainingStepBasisTests` and Python cases added; the runner test's expectations use the trainer step; new owner decision OD-17 for TE-6 to TE-10, TE-P1 and TE-P2.
- Validation: V-2, V-4 and V-5 expectations restated; V-7 added (format line, survey of every header, stale readers refuse).
- Phasing: D5 lands in P1 + P2; P4 lands in the same push (stale tools refuse v11 files).
- Risks: the header-and-name risk is gone (they agree); added the two meanings across the format line, older builds unable to read new files, and the null parent clock for schedule-less pre-v7 CLI files.
- Owner decisions: each recorded as decided, with the implementer's choices under OD-1.

---

## Review 2 (2026-10-06, the OD-1 revision at `57af1480`)

Scope: every Swift and Python reader of `training_step` (searched by key, by `metadata.trainingStep` and by `.trainingStep`), the writer and reader rules of D5 against the code, every model header on this Mac (a read-only survey of 4,530 safetensors files in `Models/` and `Sessions/*/`, plus the 21 `.dcmmodel` files), the format bump's reach (presets, `scripts/dcm_arch.py`, `scripts/dcm_lineage.py`, the running frozen builds and probe loops), the Lichess bot follow-lineage plan as committed at `a7803ac0` plus its working copy, the HPARAM and alarms plans, and every test that builds a trainer-state file or reads a step.

### Verified, no change

- Survey: creators stating a `training_step` are exactly `replay` (4,429), `train-vs-uci` (31), `manual` (13), `promote` (10), `sigusr2` (8), `periodic` (2); `new-model`, `derive-model`, `handcraft` state none. No GUI file on this Mac carries a lineage record (all are v3). Every pre-v11 file with both a schedule and a record has `trainer_completed_steps` = `cum_trainer_step`, and every v7+ CLI file has `training_step` = `segment_local_step`. The flag example (`20261005-lrC-cyc10-r1-replay-seg1-step1000`: v8, `training_step` 1000, `trainer_completed_steps` 1513) is right.
- Every production writer of a trainer-state file is listed (corpus replay `:1566`, train-vs-UCI `:536`, GUI session `:542`, promotion `Arena.swift:714`); `SessionSaveConsistentCutTests.swift:105` does pin the GUI cut's count to the trainer clock.
- Exact resume: `TrainerResumeSnapshot` takes the clock from the schedule, never from `training_step`; `BehaviorFingerprint` hashes no format version. The lineage invariants (`cum_trainer_step` = `trainer_completed_steps` on trainer files, `segment_local_step` unchanged) hold.
- Every Swift test that encodes a trainer-state file was checked: TE-6 to TE-10 are exactly the ones whose `training_step` is not the clock (`ExactResumeTests:264`, `:346`; `DropoutRNGStateTests:186`; `PolicyTailPrecisionProvenanceTests:79`; `SafetensorsDecodeStrictnessTests:80`); `ChampionLineageRecordTests:163` (1000/1000), `GuiResumeGapsTests:47`, `GuiResumeContinuationGapsTests:46`, `GuiLineageLifecycleTests:63` (4/4), `LineageProvenanceTests:43`, `LineageRecordTests:37` (steps/steps), `TrainVsUciSessionTests:198` (fresh trainer, 0/0), `ExactResumeTests:405, 454` (9/9) need nothing. No test pins format "10" (the pins are symbolic); the tests asserting no legacy log line decode current-version files or files stating no step; the legacy-line content tests use `contains`. Python: only TE-P1 and TE-P2 pin "11"; `test_dcm_arch_site_activations.md()`'s default "10" stays valid.
- The bot follow-lineage plan ranks by chain depth, `segment_local_step`, `recorded_unix` and finds files by record, not name, so D4 and D5 leave its selection unchanged; `LichessBotModelSlots.swift:189` is still the file-source line in the working copy (being edited by that plan's implementation — cite it by symbol, `LichessBotWeightsSnapshot(…trainingStep:)` in the `.file` source).
- The HPARAM plan's schema 3 is the lineage record's schema, not the architecture format; nothing in it reads `training_step`'s meaning.

### Must-fix (fixed in this file)

1. **The writer was taken from `path_kind`, which is not the writer on a GUI champion file.** `championFileLineageRecord` copies a loaded source's record (with its `invocation`) into the champion file, so a GUI champion loaded from a v7+ replay file states `path_kind` `replay` while its `training_step` is the source's trainer clock; the planned rule read that as a segment step (wrong segment step, a false flag line), and `testThePathKindDecidesBeforeTheCreator` pinned the misreading. The writer is now the `creator`, which every writer stamps; the test is replaced by `testTheCreatorNamesTheWriterNotTheRecordsPathKind`.
2. **A null parent step would have let trained weights pass as untrained.** With `ParentFile.trainerCompletedSteps` = the reading's `trainerStep`, a GUI champion save or derive copy of a schedule-less pre-v7 CLI file (3,842 on this Mac) states no `training_step` and a null parent step, so `ModelDerivation.requireUntrainedSource` accepts it for `--set-*` rewrites and `ModelLineageTree` shows it as an untrained seed. Reverted to the field's documented meaning (the parent's stated step, `LineageTracker.swift:362-368`): `trainerStepOrStatedStep`. New regression test `testACopyOfASchedulelessPreV7ReplayFileStillReadsAsTrained`; the Risk is rewritten.
3. **`.dcmmodel` files had no reading and could not be flagged.** 21 on this Mac (all GUI saves); they decode without a `DecodeFormat`, so the planned flag never fires for them. They are now read by `creator` at the unversioned version, and their loaders log the line.
4. **Stale Python test list.** The second "Python" block under Tests still specified the pre-OD-1 design (mixed-basis refusal, the glob scan failing a trainer-step-named file, `discover_enum_stems` skipping it), contradicting the revised D9. Replaced; V-4's `discover-stems` expectation corrected (the shared stem is refused for spanning two `model_id`s).
5. **"Stale readers fail loudly" overstated.** A stale `dcm_lineage.py` has no upper version bound and `replay.py` reads `training_step` from raw headers; only `dcm_arch`-based paths refuse. The claim now names what misreads and what stops it (`internals_cells` before any row is written), and `step_reading` uses `checked_format_version` and mirrors the v11 decode check.
6. **The alarms-plan conflict paragraph was stale.** The alarms plan has since adopted the decoupled 50-step evaluation (its OD-21) with this plan's per-step order and OD-15; the paragraph now says so.

### Should-fix (fixed in this file unless noted)

- An unknown pre-v11 writer's segment step falls back to `training_step` (the tracker's reading today) instead of "none", so a creator-less file without a record still tracks as it does now.
- `--analyze-numerics` hands the audit the reading's `trainerStepOrStatedStep`, matching the GUI audit, which already passes the trainer clock.
- The catalog's reading decodes a record only when needed; a malformed record now makes an error entry (stated).
- `import_probes` keeps its name-step rule (its inputs are all pre-v11) and refuses a record whose file is found at v11.
- The four live probe loops run the repo's `probe_loop.sh` / `probe_record.py`; P4 replaces the shell script rather than rewriting it in place, and V-4 checks they keep recording.
- A file stating no `training_step` is never flagged (`testAFileStatingNoStepIsNeverFlagged`); V-7's "no unknown writer" claim is limited to files that state a step.
- Not changed: the format bump's reach into presets and seeds — OD-18.

### Verdict

Sound after the fixes above. OD-1 is implemented without touching training state or exact resume; the reading is now right for every file on this Mac and for the GUI champion files other Macs hold; old files stay readable and are flagged on load. One owner decision is open (OD-18); OD-17 still needs approval.

### Owner decisions open

- **OD-17** approve TE-6 to TE-10, TE-P1, TE-P2 — recommend yes (no others needed).
- **OD-18** accept that presets, seeds and derived models written from v11 cannot be read by frozen builds — recommend yes, minting frozen-build experiment seeds and presets with those builds.

---

## Implementation notes

### Failing-first (P1 + P2)

The two runner tests were added to `ResumeEquivalenceTests` against the code before the change (with only `step_line_interval_sec` declared, so they compile) and run:
- `testAResumedRunsStatsRowsCarryDiagnosticsOnEveryRowAfterTheFirst` failed as predicted: rows at `cum_trainer_step` `[14, 63, 113]` instead of `[14, 50, 100]`; `policy_entropy` null at 63 and 113 (6.4 s).
- `testAResumedRunSavesAndNamesItsCheckpointsByTrainerStep` failed as predicted: the resumed segment wrote `S-replay-seg1-step30.safetensors`, no `S-replay-step1000` / `S-replay-step1020`, and no stats row at trainer step 1000 (59.0 s).

After the change both pass unmodified (5.9 s and 38.2 s). The 990-step test is a correctness test and is not gated.

### Decisions made during implementation

- **`[BATCH-STATS]` goes to the session log only.** The CLI step lines go through `emit` (session log and stdout); the trainer wrote `[BATCH-STATS]` to the session log only. The runners now write it with `SessionLogger.shared.log`, so stdout of a CLI run does not gain a 72 KB line per step line.
- **`--derive-model` states the source's step reading as `training_step`.** D5 said derives "write no `training_step`, as today", but `ModelDerivation` copies every source key it does not rewrite, `training_step` included. Copied verbatim into a v11 file, a pre-v11 corpus-replay or train-vs-UCI plain file's segment step would read as a trainer step. `training_step` is now one of `rewrittenMetadataKeys` and the output states the source reading's `trainerStepOrStatedStep` (the same value the output's lineage records as its parent's step; nothing when the source states none). Grafts already wrote none.
- **The derive and graft source's legacy entry** comes from their full decode of the source (it records the step reading's entry on the source's `DecodeFormat`, which `DeriveModelCLI` already logs), so no second header reading is made.
- **`ModelCheckpointFile.trainingStepReading`** is a stored property set by both decoders; a file built in memory to be encoded is read under the current format's rule (the version its encode stamps), with no segment step.
- **`CheckpointManager.legacyFileFactsLine`** is the one formatter both kinds of file go through (`ArchitectureFormat.legacyLogLine`); `logLegacyFileFacts` replaces the loaders' `architectureFormat?.logLegacyResolutions()` so a `.dcmmodel`'s training-step entry is logged too.
- **`SegmentStartTrainerStepMismatch`** is the internal error both runners throw when the pre-flight start step (from the start file's schedule) differs from the trainer's clock after restore.
- **`EnumeratedCheckpointNaming`'s legacy parse** builds names through one private static builder with an explicit segment part; a legacy reading is accepted only for `-seg<k>` with `k` a positive index without leading zeros, which is exactly what the earlier `segmentIndex` round-trip accepted (`-seg0-` and `-seg01-` stay unclaimed as segment names).
- **`table_common.buffer_plies_per_game_by_trainer_step`** reads one log, like `buffer_plies_per_game`; its test uses a resumed segment's log (lines at segment step + 513). A two-segment comparison is two calls.
- **`vsuci.py`**: a segment's label goes on its first mark (meta 1000 for logs without `trainerStep=`, as before).
- **GUI `cfgStr`** (`stepLineSec=`, D7 step 6) is done with P3, which rewrites the same ticker.

### Validation done on the branch (read-only)

- **V-7 survey** (`dcm_lineage.step_reading` over 4,537 headers in `Models/` and `Sessions/*/`, 2026-10-06): `legacy_segment_step` 4,436 `replay` + 31 `train-vs-uci`; `legacy_gui_trainer_step` 33 stating a step (`manual` 13, `promote` 10, `sigusr2` 8, `periodic` 2) + 8 `manual` stating none; `legacy_unknown_writer` only for files stating no step (`new-model` 22, `derive-model` 5, `handcraft` 2). No file stating a step reads as an unknown writer; every file with a schedule and a record has `trainer_completed_steps` = `cum_trainer_step`; no header errors.
- **V-5 `bn_liveness.py`**: the new script and the one on `main` give byte-identical output over every arm at trainer steps 1000, 6000, 18000 and 20000.
- **V-5 `vsuci.py`**: a dry rebuild (nothing written) of the registered run gives the committed CSV's 1,536 rows; the one differing cell (a segment label in `note`) differs identically with the `main` script — the registry's label was edited after the CSV was written.
- `--show-default-parameters` lists `step_line_interval_sec: 180`; `--create-parameters-file` writes it into `parameters.json` and `parameters.md`.
- Not run: V-2 / V-3 (live CLI runs; the two runner tests cover V-2's checks in-process), V-4 against real v11 files (the Python tests cover it on synthetic headers), V-6 (the GUI; launching it would offer the auto-resume of the last session), and the live LR probe loops' next checkpoint after the merge.
