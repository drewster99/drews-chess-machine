# Review fixes, 2026-10-03 (72-hour code review)

A review of every source file changed in the 72 hours before 2026-10-03 13:00 CDT
(285 commits, base `9ff2e3d8^`). Nine finder passes reported about 80 findings; they were
merged into 16 groups, each analyzed and then independently reviewed against the code.
The per-group record — the reviewer's final decision followed by the full analysis —
is in `review-2026-10-03/G<n>.md`; the raw findings are in `review-2026-10-03/find-*.md`
and the grouping in `review-2026-10-03/groups.md`. Where a group file's "Final" section
and its analysis disagree, the Final section governs.

## Scope (owner decision, 2026-10-03)

Every HIGH and MEDIUM item, plus the cheap LOW items that touch the same code. Items a
reviewer marked OPTIONAL or SKIP are out of scope unless listed below.

## Owner decisions

1. **Build New Model size: guidance, not limits.** No cap on block count, channels or
   kernels. The tower-shape check refuses only what cannot exist: a non-positive count,
   arithmetic that overflows `Int`, and a model whose training state cannot fit in this
   Mac's physical memory (`parameterCount × trainingBytesPerParameter`, fp32 working copy +
   master + velocity + gradient). Everything else is allowed and annotated with a
   parameter-count recommendation scaled to installed RAM (reference: 64 GB; batch 4096
   trains up to about 20M parameters; 5–15M recommended; batch below 512 not recommended):
   - recommended up to `15M × RAM/64 GB`;
   - fits at batch 4096 up to `20M × RAM/64 GB`;
   - above that, the largest power-of-two batch ≥ 512 that fits, from
     `maxParams(batch) = 20M × RAM/64 GB × √(4096 / batch)` (1024 → 40M, 512 → ~57M on 64 GB);
   - above the batch-512 level: "likely too large to train on this Mac" (a warning; the
     build is still allowed).
   Shown in Build New Model's readout, and logged by `--new-model`, `--derive-model` and
   GUI builds as an `[ARCH]` line. One function computes it (single source).
   The UI never expands the tower (`expandedBlocks`) before the shape validates, and the
   init-options rows read skip-projection groups computed analytically per group.
2. **Delay and arena-concurrency ranges narrowed to the enforced caps:** `SelfPlayDelayMs`
   and `TrainingStepDelayMs` 0…3000, `ArenaConcurrency` 1…1024; the app caps are derived
   from the declarations (G3 ui#5, G6 item 6).
3. **Test edits approved** where a fix needs them (mechanical call-site / helper changes;
   expectations unchanged): the `GuiResumeGapsTests` helper argument, the `championLineage:`
   argument at the existing `saveSession` call sites, new helpers in `ResumeEquivalenceTests`
   and its stale header comment. Documentation and comments are updated for every change.
4. **Behavior changes approved:** a `--resume-exact` whose epoch budget is spent is refused;
   a strictly validated corpus position (a legacy file at `next_game_index == total` is
   refused); `--seed +5` / `-0` (and the same text in parameters files and stored settings)
   are errors; a new `device` resume gap (waived when the behavior fingerprint matches);
   the CLAUDE.md wording fixes.
5. **Dashboard:** matplotlib installed for the cron job's Python 3.9 (done 2026-10-03).

## Work units (parallel worktrees, merged into `main`)

Each unit implements its items, writes each bug's regression test first and shows it
failing (where the reviewer says the only possible "red" is a compile failure, the plan
accepts that and the commit says so), builds, runs the touched test classes, and commits.
CLAUDE.md and CHANGELOG edits are made once, after the merges, to avoid conflicts.

| Unit | Groups and items | Main files |
|---|---|---|
| U1 Python tools | G13 items 1–7, 9; G4 item 6 | `documentation/dashboards/*`, `scripts/*`, `experiments/probe_loop.sh`, `experiments/table_common.py`, table/review scripts |
| U2 Lichess bot | G12 #1, #2+#5, adjacent notes loading, #4, #6 | `LichessBot/**` |
| U3 Build screen and settings UI | G1 (as revised by decision 1), G2, G3, G6 item 6 | `NetworkArchitecture`, `BuildNewModel*`, `InitSetButtonsView`, popover models/views, `AutoResumeController`, `UpperContentView`, `TrainingParameters` (ranges, reset), `SessionController.buildNetwork` |
| U4 Session lineage and saves | G4 items 1–5, G5, G6 items 1–5, G7 | `SessionController*`, `LineageTracker`, `LineageRecord`, `CheckpointManager`, `ModelDerivation`, `ModelGraft`, `SafetensorsModelIO`, `ModelCheckpointFile`, `ParallelWorkerStatsBox`, `TrainingParameters` (holds), `ReplayBuffer.restore` |
| U5 Replay, train-vs-UCI, corpus, parsing | G8 items 1–4; G10 cli#4, cli#9, cli#7; G11; G15 #1, #4, #7; G16 app#10c | `CorpusReplayRunner`, `TrainVsUci*`, `GameCorpus*`, `CorpusValidator`, `CorpusRecorder`, `SafetensorsModelIO` (legacy resume), `TrainingParameters` (inspectStored, strict decimal), `LineageRecord` (RunStreams), `SessionCheckpointFile` |
| U6 Network, trainer, misc | G14; G9 items 1(a), 2; G16 net#5, cli#8, persist#10(a,b), persist#11, app#10d, training#3 | `ChessNetwork`, `ChessTrainer`, `ChessMPSNetwork` docs, `MoveSampler`/`DCMNormalMath` docs, `ResumeExactness`, `BehaviorFingerprintTests`, `BinaryByteCount`, `DrewsChessMachineApp`, `SessionController+Training` (exit helper), `ReplayBufferAnalyzer` |

Overlaps resolved at merge: `TrainingParameters.swift` (U3 ranges/reset, U4 holds, U5
decode/parsing), `SessionController+Training.swift` (U4, U6 exit helper), `CheckpointManager`
(U4 champion lineage, U4 buffer-written callback), `LineageRecord.swift` (U4, U5).
G15 #4 and G16 app#10c are one change (one strict decimal parser), done in U5.

## Not in scope (reviewer OPTIONAL / SKIP)

G1 ui#7a and the BuildNewModelView helper-property cleanup; G9 item 1(b); G10 cli#3 and
cli#5; G12 #3, #7, #8; G13 item 8 (experiment review scripts); G15 #2, #3, #5, #6; G16
net#4, net#6, persist#10(c), app#10e, app#10f; G8 item 5 (cosmetic marker naming); G14
net#2 option B. Separately noted for later: a GUI resume never compares saved and live
batch size / buffer capacity (G6 adjacent); the early-failure returns in the Play-and-Train
start task skip teardown (G7); a resumable-session `--train` mode (owner question,
2026-10-03, not requested).

## Validation

- Every in-scope item has its regression test (named in its group file) passing, and the
  test was seen failing before the fix (or the commit states that only a compile failure
  is possible).
- Each unit's touched test classes pass in its worktree; after all merges, the full suite
  passes on `main` (slow tests included per the scheme's test plan).
- Python: `python3 -m unittest` in `documentation/dashboards/tests` passes under
  `/usr/bin/python3` 3.9.
- Manual checks: Build New Model with a negative count, `Int.max`, and a 30M-parameter
  tower (no crash; correct guidance); the launch-sheet order (invalid-settings sheet, then
  the resume prompt) with a planted invalid stored value restored afterwards; one GUI
  session save and resume with the log showing a consistent cut.
- CHANGELOG entry per merged unit; CLAUDE.md wording fixes (G9 item 3, G16 training#9,
  G6 "Run seed" sentence, the new size guidance and narrowed ranges).

## Status (2026-10-03 14:40 CDT)

All six units implemented and merged into `main`: U1 `b9eac4fd`, U2 `47cbdc50`, U6 `47f1303e`,
U5 `f273b890`, U3 `748afe20`, U4 `fbf2a4f1`. CLAUDE.md and CHANGELOG updated after the merges.

- **Full suite on `main` at `fbf2a4f1`** (scheme test plan, slow tests on): 2,318 tests, 0
  failures, 1 skipped (`LegacyDcmmodelLoadTests.testRealLegacyDcmmodelsResolveBuildAndLoad`,
  gated by `DCM_RUN_LEGACY_LOAD`).
- **Python:** `/usr/bin/python3 -m unittest` in `documentation/dashboards/tests` (3.9.6): 94
  tests OK.
- **Merge resolutions:** U4 moved the GUI resume's parameter block into
  `SessionParameterResume.applyGuiSession`; U3's unclamped `arena_concurrency` restore and U5's
  single-resolver arena promotion set restore were carried into it. U3's
  `BuildNetworkRefusalTests` observed trainer drops through `onDropTrainer`, which U4 removed; it
  now plants fed counts that dropping the trainer resets (same expectation).
- **Deviations recorded by the units:** U3 widened one of its own new regression tests from 8 to
  16 channels after it went red (at 8 the tower was narrower than the value head's conv width, a
  setup error the pre-fix crash had hidden); U3 kept Neutral init enabled (the per-group
  computation cannot trap); U5's four new corpus-replay resume refusals exit 33
  (`CorpusReplayError`), not 2.
- **Open:**
  - Manual checks (2026-10-03 14:55–15:06 CDT, Debug build of `fbf2a4f1`, real settings
    domain backed up first and restored after; only the model-ID counter moved): all passed.
    - Build New Model: a group count of −1 shows "blockGroups[0].count must be positive (got
      -1)", Total blocks "invalid", Build disabled; `Int.max` shows "the parameter count
      overflows Int"; a 30,851,626-parameter tower (5 × 248 ch, 7×7) shows "Too large for batch
      4096 on this Mac (up to 20.0M with 64 GB of memory); fits at batch 1024 (up to 40.0M) and
      below" with Build enabled; no crash. A GUI build logs `[ARCH] size guidance (Build
      Network): … batch512_max=56568542 verdict=within_recommended`.
    - Launch order: with `lr_warmup_steps` planted as 7.9 and a resume pointer present, the
      invalid-settings sheet opens first ("stored as a real number, not a whole number"),
      `[RESUME] Auto-resume prompt waits for the invalid-settings sheet to close` is logged,
      Reset rewrites only the stored entry (`… the current run's value is unchanged`), and the
      resume prompt appears after Close with its countdown starting then (29 s showing).
    - GUI save and resume: a Save Session with the buffer included paused self-play
      (`dropped 180 in-flight games`) and wrote buffer, `replayBufferTotalPositionsAdded` and
      lineage fed counts that agree (709,074 positions, 2,362 games); the resume logged
      `[RESUME] EXACT`, continued the seed, game serials (next 3234) and arena count, restored
      sampler, dropout state and buffer, and started the trainer clock at 22.
    - `--train --training-step-limit 5`: exit 0, `results.json` written, and the session log ends
      `[APP] --train: exiting process (termination_reason=step_limit_reached)`.
    - `--help` usage text: the new `--epochs`, `--resume-exact` and `--accept-inexact` text (its
      item list includes `device`).
  - Found during the manual checks (not changed):
    - `saveSessionInternal` builds the session state (`buildCurrentSessionState`) and reads the
      step count before it pauses, so `session.json`'s `trainingSteps`, `selfPlayGames` and
      `emittedGames` are from before the cut (here 20 steps and 2,340 games against the
      trainer's 22 and the record's 2,362). The G5 analysis kept this ("carries no stream
      state"); a resume takes its clocks from the trainer file and the record
      (`trainer_completed_steps: 22 (from trainer file; session step count 20)`), so only the
      resumed run's displayed counters start that far behind.
    - A bare `--help` is not a recognized argument: it prints "unrecognized argument(s):
      '--help'" before the usage and exits 2.
    - `~/Library/Preferences` holds 1,398 `<TestClass>-<UUID>.plist` domains left by test
      suites (`LichessBot*Tests` and others) that create a private defaults suite and never
      remove it.
  - `TrainVsUciRunner` still restores the replay buffer and then checks
    `totalPositionsAdded` (`verifyReplayBufferMatchesSession`); a mismatch fails the run either
    way, so there is no wrong-state outcome, but the GUI's check-before-restore
    (`restore(from:expectedTotalPositionsAdded:)`) would make it one path.
  - `CheckpointManagerSafetensorsTests` writes its session folders and models into the real
    `Sessions/` and `Models/` folders (uniquely named, removed by the test); it should take a
    temporary folder like the other save tests.

