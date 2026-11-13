# Derive a trainer-state file's precision (plan)

Status: plan only, nothing implemented. Owner decision 2026-11-12: `--derive-model` accepts a trainer-state file for operations that rewrite no tensor (`--set-compute-dtype`, `--set-policy-tail-precision`), and the output can be continued with `--replay-corpus --resume-exact`. Precision stays an architecture field in the file. A resume-time precision flag was rejected.

Paths below are relative to `DrewsChessMachine/DrewsChessMachine/` unless stated otherwise.

## Problem

- Goal: continue R-elitecyc from `Models/20261011-zlra-elitecyc-latest-step384964.safetensors` (model `20261025-1-qyUf`, run `E709F4EB…`, segment 13, bf16 compute, tail `fp32_from_pre_bn`, 41 `opt.*.velocity` tensors, all tensors F32) with the parameters of `experiments/20261008-zlra-selfplay-lr/launch/elitecyc_resume13_decay_launch.sh`, but computing in fp32.
- Today `--derive-model --set-compute-dtype float32` refuses any file that has `opt.*` tensors or `trainer_*` keys (`Persistence/ModelDerivation.swift:511-523`, `DeriveError.sourceHasOptimizerState`).
- The only fp32 override is the GUI session loader's `forceFloat32` (`App/SessionController+Checkpoint.swift:935-1070`). It changes precision at resume time, which is the kind of setting the owner rejected.

## Findings

### What exact resume reads from a trainer-state file (corpus replay)

- Load: `CheckpointManager.loadModelFile` (`CLI/CorpusReplayRunner.swift:1140`). The trainer and the feeder are built from **the file's own architecture**: `arch = file.architecture` (`:1144`), then `ChessMPSNetwork(…, arch: arch)` and `ChessTrainer(…, arch: arch)` (`:1606-1614`). An fp32 file therefore builds an fp32 trainer with no extra code.
- Trainer snapshot: `TrainerResumeSnapshot(checkpoint:)` (`Training/TrainerResumeState.swift:333-349`) needs the velocity tensors and `trainer_*` schedule keys. It takes the dropout Philox state from the lineage record (`rng.dropout_philox_state`, `:296-302`) and `trainer_grad_norm_history`.
- Corpus position: `SafetensorsModelIO.replayResumePoint` reads `record.fed.corpus` (`Persistence/SafetensorsModelIO.swift:782-805`). A record without it is refused with `noCorpusPosition`.
- Corpus content: `fed.corpus.shardSHA256` must match (`CorpusReplayRunner.swift:1414-1423`). If `feed_per_step` also matches, the feed phase continues (`:1424-1429`).
- Streams: `record.rng.streams` gives the seed, the sampler and the dropout stream (`:1430-1442`). Missing streams are the `rng_sampler` gap.
- Parameters: `record.parameters`, logged by `ParameterDifference.exactResumeLogLines` (`:1443-1449`).
- Environment: `ResumeGap.environmentGaps(writtenBy: parentRecord, …, runningFingerprint: BehaviorFingerprint.compute(for: .init(arch: arch)))` (`:1450-1456`, `Training/ResumeExactness.swift:126-179`). It compares the record's `build` and `device`. A changed build, OS or Mac is a gap only when the saved `rng.behavior_fingerprint` does not match the running one.
- **The fingerprint covers the architecture, compute dtype and tail included** (`Training/BehaviorFingerprint.swift:26-33`, `:77-85`, `:188`).
  - The saved fingerprint was computed by build 2523 under the bf16 architecture.
  - An fp32 file therefore never matches it, so every build or OS change becomes a `build` / `os` gap.
  - The R-elitecyc-hilr fork (build 2663, same file) logged `build changed … fingerprint matches` and `OS changed … fingerprint matches` (`experiments/…/train-elitecyc-hilr-seg0.stdout`). With an fp32 file both would become gaps.
  - With the **same** build, OS and Mac, the fingerprint is never consulted, so a dtype change would pass silently as EXACT. Nothing else in the resume compares the file's architecture with the architecture the trainer state was captured under.
- Schedule: `p.trainer.scheduleDifferences(from: snapshot.schedule)` gives the `schedule` gap (`:1189-1192`, `:1385`). Segment 13 logged `[RESUME] NOT EXACT: schedule` (`train-elitecyc-seg13.stdout`).
- Restore: `restoreExactly(…, .keepConfigured)` (`:1667`) calls `loadTrainerWeights` (`Training/ChessTrainer.swift:5811-5833`).
  - The base block is the fp32 masters under bf16 and the working weights under fp32 (`exportTrainerWeights`, `:5786-5805`). The layout and counts are the same.
  - Velocity is always fp32 (`ChessTrainer.swift:4271-4318`). Masters exist only when `dtype != .float32` (`:4271`, `:4337-4365`).
  - So bf16 → fp32 loads the saved fp32 masters losslessly as the fp32 working weights.
  - fp32 → bf16 rounds the working copy and keeps the full masters.
- Lineage of the resumed segment: `.resume(parent: file.lineageParent, gaps:)` (`:1591-1593`).
  - The run id, segment index + 1, totals, segments and ancestry are inherited from the parent record (`Persistence/LineageTracker.swift:282-312`, `:245-248`).
  - The derivation history is carried verbatim (`:284`).
- Model ID: every replay process mints its own `runModelID` (`App/DrewsChessMachineApp.swift:1174`). A new ID on the start file therefore costs a resume nothing; it is recorded only as the parent.
- Epoch budget: read from `fed.corpus.epoch` (`:1491-1507`). At 384,964 the file is in epoch 2, game 10,696,971.
- Train-vs-UCI accepts a model file with `--resume-exact` too and runs the same gap logic (`CLI/TrainVsUciRunner.swift:246-330`). The GUI calls `environmentGaps` as well (`App/SessionController+Lineage.swift:608`).

### What derive does today

- `requirePlainModelSource` refuses `opt.*` tensors and `trainer_*` keys (`ModelDerivation.swift:342`, `:511-523`). It is also used by graft (`Persistence/ModelGraft.swift`).
- Tensors are copied bit-exact unless an operation declares rewrites (`:360-393`). For set-compute-dtype / set-policy-tail-precision, `tensorRewrites` returns `[]` (`Persistence/ComputeDataTypeDerive.swift:77-79`, `Persistence/PolicyTailPrecisionDerive.swift:66-68`).
- Metadata:
  - Every source key is copied except `rewrittenMetadataKeys` (`:154-173`): format, model_id, created_at, creator=`derive-model`, parent_model_id=source, notes, architecture, training_step, content hash, derivation_history, `dcm_lineage` and its mirrors, test-set results, and the legacy tail key.
  - `trainer_*` keys and `trainer_grad_norm_history` are **not** in that set, so a lifted guard would copy them verbatim. That includes legacy schedule forms, which "nothing writes".
- `training_step` = the source's `trainerStepOrStatedStep` (`:435-437`). For a trainer-state file that is its clock.
- Lineage = `LineageTracker.untrainedCopyRecord` (`:447-455`, `LineageTracker.swift:684-751`). It mints a new run id at segment 0 with `start: .derive` and continues the source's totals. It also writes:
  - `fed.corpus: nil` (`:725`);
  - `rng: .withoutRunStreams(dropoutPhiloxState: nil)` (`:741`), so no dropout state, streams or fingerprint;
  - an ancestry that gains the source's run, left by `derive`, with `architecture_at_departure` = the source's architecture text when it changed (`:743`, `Persistence/LineageRecordSchema3.swift:739-776`).
- These would break an exact resume:
  - no corpus position: refused with `noCorpusPosition`;
  - no streams: `rng_sampler` gap;
  - no dropout state: `dropout_state` gap.
- Test-set results are re-evaluated on the output (`:458`, `Persistence/ModelTestSetResults.swift:311-320`). The evaluator loads only the base prefix, so trailing velocity is fine (`App/ModelTestSetEvaluator.swift:143-146`).
- The output is decoded back through the loader (`:462-472`). Decode accepts velocity and trainer keys, and it checks `training_step == trainer_completed_steps` (v11+, `Persistence/ModelFileStepReading.swift:84-89`).
- **Two trainer-state checks run only in `SafetensorsModelIO.encode`, which derive bypasses** (it calls `SafetensorsFile.encode`): the lineage `cum_trainer_step` must equal the clock, and the record's schedule keys must agree with `trainer_*` (`Persistence/SafetensorsModelIO.swift:258-284`).
- No lineage invariant forbids trainer state on a `start: derive` record (`Persistence/LineageRecord.swift:1352-1395`).

### Existing tests touching the refusal

- `SEBetaInitTests.testDeriveRefusesTrainerStateSources` (`DrewsChessMachineTests/SEBetaInitTests.swift:589-607`) uses `--set-se-beta-init`, which stays refused.
- `GraftDeriveTests.testAGraftRefusesASourceCarryingOptimizerState` (`:144-152`): graft stays refused.
- `DeriveTrainedSourceTests` and `DeriveLineageTrainedSourceTests` use plain sources only (`includesVelocity: false`).
- No test asserts the trainer-state refusal for `--set-compute-dtype` or `--set-policy-tail-precision`. **No existing test needs modification** (see Tests).

## Design

### D1. Which operations may take a trainer-state source

- Add `acceptsTrainerStateSource: Bool` to `DeriveOperationKind` (`ModelDerivation.swift:99-117`). Set it `true` only for `SetComputeDataTypeDeriveOperation.kind` and `SetPolicyTailPrecisionDeriveOperation.kind`.
- Why not every no-rewrite kind: activation and ReZero-cap setters change the function the velocity was accumulated for. Precision changes only the rounding of the same function.
- The engine still asserts that every operation's `tensorRewrites` is empty for a trainer-state source, checked before anything is built. A non-empty list is an internal error.
- Replace `requirePlainModelSource` in `derive` with a source classification: `.plain(modelID)` or `.trainerState(modelID, schedule, gradNormHistory)`.
  - A trainer-state source with any operation whose kind does not accept one keeps today's error, `sourceHasOptimizerState(source:detail:)`, with the detail naming the operation. Existing tests stay green.
  - Graft keeps calling `requirePlainModelSource` unchanged.

### D2. Tensors

- All tensors are copied bit-exact: base weights (fp32 masters as stored), BN statistics, and every `opt.<name>.velocity`. Shapes and order are unchanged (`requireSameTensorLayout` already holds for these operations).

### D3. Trainer metadata (one source: the decoders and encoders)

- Add `TrainerScheduleState.MetadataKey.all ∪ MetadataKey.legacy ∪ GradientNormHistory.metadataKey` to the keys a trainer-state derive replaces.
- Write the current forms from the decoded values: `schedule.metadataEntries()` and `history.metadataValue()`. A legacy-schedule source then comes out in the current form only, never both.
- Run the trainer-state checks that today live inline in `SafetensorsModelIO.encode` (`:255-284`) on the output. Factor them into one function (for example `SafetensorsModelIO.checkTrainerState(metadata:lineage:includesVelocity:)`) that `encode` and derive both call:
  - lineage step total == clock;
  - `training_step` == clock;
  - history ends at clock;
  - the record's schedule agrees with the trainer schedule.
- `training_step` stays the clock (already the rule at `:435-437`). `value_head_centered` and every other non-rewritten key are carried as today.

### D4. Model identity

- New ModelID, `creator = derive-model`, `parent_model_id` = the source ID, new `content_sha256` and `notes`, all as for any derive.
- `notes` adds `trainer state carried (trainer step N, velocity, schedule)`.
- Nothing in resume needs the source's ID: replay mints its own `runModelID` per process.

### D5. Lineage: a derive run that carries the trainer state

- New `LineageTracker.trainerStateCopyRecord(source:derivation:sourceArchitecture:naming:argv:at:)`, built beside `untrainedCopyRecord` and sharing its body. It follows the documented rule "run id minted on fresh / branch / derive".
  - `run`: new `lineage_run_id`, segment 0, `start: derive`, `exact_resume: false`, no gaps.
  - `parent`: the source.
  - `ancestry`: the source's run, left by `derive`, with `architecture_at_departure` = the source's architecture text. For these operations the architecture always changes: `apply` refuses a no-op.
  - Totals: continue the source's. `cum_trainer_step` = clock; `segment_start_trainer_step` nil; `segment_local_step` 0; segment games, positions and times 0.
  - `parameters`, `configuration`, `run_seeds`, `model_naming` (`ModelNaming.ofDerive`): as `untrainedCopyRecord` carries them.
  - `fed.corpus`: **the source's position carried**. Identity, shard hashes, epoch, next game, shard, fill, capacity, `feed_ahead_positions` and `feed_per_step` are kept verbatim. `segment_start` is set to the saved position itself: the derive fed nothing, so a later segment summary never claims the derive fed the source segment's games.
  - `rng`: **the source's `dropout_philox_state`, `streams` and `behavior_fingerprint` carried**. These describe the trainer snapshot the file still holds. `initialization` is nil (derive).
  - `build`, `invocation`, `device`: the derive process's, as for every derive.
  - `derivation_history`: the source's history plus this derivation.
- The flat mirrors come from the record as usual (`LineageRecord.metadataEntries`).
- Why a new run rather than continuing E709 in place:
  - It matches the documented minting rule.
  - It records the architecture the trainer state was captured under (`architecture_at_departure`), which the resume needs (D6).
  - The old run's totals and segments stay intact in `ancestry`.

### D6. Resume: one origin rule, one new gap

- New type in `Training/ResumeExactness.swift`: `TrainerStateOrigin`, built from the parent record. It is "the process and architecture that captured this trainer state".
  - Ordinary record: the record's `build`, `device` and `rng.behavior_fingerprint`; architecture = the file's; not changed.
  - Derive-start record carrying trainer state (`run.start == .derive` and the file has a trainer schedule):
    - build and device = `ancestry.runs.last.segments.last`'s (the training save's: 2523, M5 Max);
    - fingerprint = `rng.behavior_fingerprint`;
    - captured architecture = `ancestry.runs.last.architecture_at_departure`, decoded through `SafetensorsModelIO.decodeArchitecture(fromMetadata:source:)` with `{architecture, dcm_format_version}`;
    - `architectureChanged` = true, with the fields that differ (from the derivation record's `changed_architecture_fields` and the two decoded architectures).
  - A derive-start trainer-state record missing that ancestry entry, its last segment or its architecture text is a refusal (malformed file), never "no gap".
- `environmentGaps` takes the origin. The running fingerprint is computed for the **captured** architecture (`BehaviorFingerprint.compute(for: .init(arch: origin.capturedArchitecture))`, cached per architecture). `build` / `os` / `device` then keep meaning "does this build compute what the saved build computed for that trainer state". The existing `environmentGaps(writtenBy:…)` signature stays as a thin wrapper, so tests do not change.
- New `ResumeGap.architecture` (token `architecture`), declared after `schedule`.
  - Raised when `origin.architectureChanged`.
  - Logged as one line: `[RESUME] architecture: changed by derive <derivedModelID> after the trainer state was captured: compute_data_type bfloat16 → float32, policy_tail_precision fp32_from_pre_bn → does_not_apply`.
  - Why a gap: an exact resume claims to continue the saved run's trajectory. A dtype change keeps the state exact but changes every later step's arithmetic, the same reason a non-matching build is a gap. Without it, same build + OS + Mac would log EXACT. The operator accepts it with `--accept-inexact …,architecture`, recorded in `not_exact_items`.
- One call site per path: corpus replay `:1450`, train-vs-UCI `:323` and the GUI (`SessionController+Lineage.swift:608`) all go through the origin. The GUI can never meet a derive-start trainer file in a session save, but the rule stays in one place.
- `--accept-inexact` help is generated from `ResumeGap.allCases` (`App/CommandLineHelp.swift:286-290`), so it updates by itself.

### D7. Expected real-run lines

- `[RESUME] schedule: …` (as in segment 13)
- `[RESUME] architecture: changed by derive <id> …`
- `[RESUME] build changed (2523 a242b7ad → <new>), behavior fingerprint matches` (expected; 2663 matched for bf16)
- `[RESUME] OS changed (… 26B118 → …), behavior fingerprint matches` (if the OS still differs)
- `[RESUME] NOT EXACT: schedule, architecture`
- `[RESUME] rng: sampler=restored`, `dropout=restored`, `dropout stream=restored`, `grad-norm history: restored entries=…`
- `[REPLAY-ARCH] (start-model) … float32 …`, `policy_tail=does_not_apply`
- `[REPLAY] start-model trainer state restored exactly (fp32 masters, velocity, trainerStep=384964)`
- `[RUN] path=replay run=<derive run id> seg=1 (resume of <derivedID> sha=…, not exact: schedule, architecture) … policy_tail=does_not_apply cum_step=384964 cum_games=37772534 …`

## Guardrails

- Only kinds with `acceptsTrainerStateSource` take a trainer-state source. Every other kind, any mix that includes one, and graft keep `sourceHasOptimizerState`.
- Empty `tensorRewrites` is asserted for a trainer-state source (D1).
- The source must carry a lineage record (format v7+). A pre-lineage trainer-state file is refused with a new `DeriveError.trainerStateSourceWithoutLineage`: there is no corpus position, streams or fingerprint to carry. Such a file can still be branched.
- The source's record must not itself be a derive-start trainer-state record (chained trainer-state derives are refused, `DeriveError.trainerStateSourceIsDerived`). The origin rule (D6) then always points at a training save. Multiple operations in one call are allowed (`--set-compute-dtype bfloat16 --set-policy-tail-precision mixed_final_projection`).
- The trainer-state checks (D3) run on the output before it is returned. Decode-back verification stays.
- Never overwrites: unchanged (`FileSafety.publishNewFile`, `App/DeriveModelCLI.swift:247`).

## Edge cases

- **fp32 → bf16 back** (from a save trained in fp32): allowed. The tail becomes `fp32_from_pre_bn` (`ComputeDataTypeDerive.swift:66-73`). The bf16 trainer loads the fp32 working weights as masters and rounds the working copy. The resume reports the `architecture` gap.
- **Tail only on a bf16 trainer file** (`fp32_from_pre_bn` ↔ `mixed_final_projection`): allowed; `architecture` gap. A tail on fp32, or `does_not_apply` on bf16, is refused by `target.validate()` as today.
- **No-op** (`--set-compute-dtype bfloat16` on a bf16 file): refused as today (`operationNotApplicable`).
- **A session folder's `trainer.safetensors`**: allowed as a source. The output is a standalone file and cannot be put back into a session.
  - A GUI-session trainer file has no corpus position: corpus replay refuses it as today (`noCorpusPosition`). It can still be branched.
  - A train-vs-UCI trainer file resumes through `--train-vs-uci --start-model <file> --resume-exact`, with `buffer` and `architecture` gaps.
  - Document this; no extra refusal.
- **Pre-v11 trainer-state files** (`training_step` was the segment step): the output is written at the current format with `training_step` = the clock (the existing rule). Legacy schedule keys are rewritten to the current form (D3). The record is rebuilt at the current schema (the record initializer, as `withoutTrainerState` does). The parameter snapshot is carried word for word.
- **Pre-v12 files**: the tail resolves into the architecture; the flat key is already in `rewrittenMetadataKeys`.
- **Lineage unrecorded (pre-v7)**: refused (Guardrails).
- **Schema 2–4 records**: carried and converted to the current schema. Their unrecorded fields stay unrecorded.
- **Velocity NaN payloads**: copied as `[Float]` bit patterns. The test asserts `bitPattern` equality.
- **Test-set results**: re-evaluated under the target architecture (fp32). This uses the GPU, so the derive is slower than a pure file transform; that is fine next to a training job.

## Tests

Regression-first: write T1–T3 before the change and show they fail (`sourceHasOptimizerState`).

New class `DeriveTrainerStateTests`, pure (no GPU). It uses `ModelTestSetFixtures.notEvaluated` overloads and `LineageRecord.forTests(trainerCompletedSteps:corpus:)` / `forTests(adopting:…)` (`DrewsChessMachineTests/LineageTestSupport.swift:13`, `:91`). Trainer files are built as in `LineageRecordTests` (`:40`).

- T1: `--set-compute-dtype float32` on a bf16 trainer-state file (velocity, schedule, grad-norm history, a record with corpus position, streams, dropout state and fingerprint) succeeds. Then:
  - every tensor, velocity included, is bit-identical;
  - the decoded `trainerSchedule` and `trainerGradNormHistory` equal the source's;
  - `training_step` == clock;
  - architecture is float32 / `does_not_apply`.
- T2: the record of T1's output:
  - new run id, segment 0, `start: derive`, parent = the source;
  - ancestry's last run = the source's run, `left_by: derive`, `architecture_at_departure` = the source's text byte for byte;
  - totals continue;
  - `fed.corpus` equals the source's except `segment_start` = the saved position;
  - `rng` dropout state, streams and fingerprint equal the source's;
  - `derivation_history` = the source's plus one entry with `changed_architecture_fields`.
- T3: `--set-policy-tail-precision mixed_final_projection` on a bf16 `fp32_from_pre_bn` trainer file succeeds, with the same carry checks as T1.
- T4: `--set-se-beta-init`, an activation setter, `--set-rezero-alpha-cap`, and a mix of `--set-compute-dtype` with any of them on a trainer-state file are still refused with `sourceHasOptimizerState`.
- T5: a pre-lineage trainer-state file is refused (`trainerStateSourceWithoutLineage`). A source that is itself a derive-start trainer file is refused (`trainerStateSourceIsDerived`).
- T6: a source in the legacy schedule form produces only the current `trainer_lr_schedule` keys. The schedule equals the converted one.
- T7: an internally inconsistent source is refused by the shared trainer-state check, not written. Example: the lineage step total differs from the clock, hand-built.
- T8 (`ResumeExactness`, pure):
  - `TrainerStateOrigin` of an ordinary record is the record's build, device and fingerprint;
  - of a derive-start trainer record it is the departed run's last segment, the captured architecture and `architectureChanged`;
  - a malformed derive-start record throws;
  - `ResumeGap.architecture` parses from `--accept-inexact` and is ordered after `schedule` in the log line.

GPU tests (targeted classes):

- T9 `ChessTrainer` lossless switch:
  - build a bf16 trainer on `ResumeEquivalenceTests.architecture` with `computeDataType: .bfloat16`, train a few steps, `exportResumeSnapshot()`, write the file and derive it to fp32;
  - `restoreExactly` into an fp32 trainer built from the derived file's architecture;
  - `exportTrainerWeights()` must equal the snapshot's weights bit for bit (masters → fp32 working, velocity), with clock and dropout state equal.
  - Also the reverse: fp32 → bf16 keeps the masters bit-exact.
- T10 corpus-replay integration, in a new class (or a new case beside `CorpusReplayRefusalTests`). It reuses `ResumeEquivalenceTests.writeCorpus` / `writeStartModel` (static, `:110`, `:148`) with a bf16 variant of the tiny architecture.
  - Train N steps, derive the save to fp32, then `runReplay` with `resumeExact: true`.
  - With `acceptInexact: []` the refusal names `architecture`.
  - With `[.architecture]` the run trains M steps; the trainer clock ends at N+M; the segment record has `exact_resume: false`, `not_exact_items == ["architecture"]` and segment index 1 of the derive's run; the feed position continues from the saved one; the output architecture is fp32.
- T11: the T10 resume under the same build, OS and Mac still reports `architecture`. This pins the hole described in Findings.

Existing tests that would need modification: **none found**.

- These stay green unchanged: `SEBetaInitTests.testDeriveRefusesTrainerStateSources`, `GraftDeriveTests.testAGraftRefusesASourceCarryingOptimizerState`, `SEBetaInitTests.testOperationCatalogDrivesTheCLI` (flags unchanged; help still contains each flag), `PolicyTailPrecisionProvenanceTests.testPolicyTailIsNoLongerAResumeGap`, `ExactResumeCompletionTests` (gap ordering uses existing cases only), `UntrainedCopyRecordTests`, `DeriveTrainedSourceTests`, `DeriveLineageTrainedSourceTests`.
- If any turns out to depend on the exact text of `sourceHasOptimizerState` or of the `--from` help line, stop and ask before editing it.

## Docs

- `documentation/deriving-models.md`: rewrite the `--from` paragraph (`:22-37`):
  - trainer-state sources are accepted by `--set-compute-dtype` / `--set-policy-tail-precision` only;
  - what is carried (velocity, schedule, history, corpus position, streams, dropout state, fingerprint);
  - the new run with ancestry;
  - the `architecture` gap and the refusals (pre-lineage, chained);
  - a worked example (the R-elitecyc fp32 continuation).
- `App/DeriveModelCLI.swift` help (`:50`, "not a trainer-state file") and the summaries in `ComputeDataTypeDerive.swift` / `PolicyTailPrecisionDerive.swift` (`:11` there says trainer files are refused).
- `ModelDerivation.swift` file header (`:24-27`, `:37`).
- `App/CommandLineHelp.swift`: the `--derive-model` line (`:172-181`) gets one sentence about trainer-state sources. The `--accept-inexact` list is generated.
- `CLAUDE.md`:
  - the `[RESUME]` tag paragraph (add the `architecture` gap);
  - "File lineage" (a derive of a trainer-state file carries its trainer state into the new run);
  - the policy-tail bullet ("on a plain model" → "on a plain model or a trainer-state file").
- `documentation/plans-active/DETERMINISM_RESUME_LINEAGE_PLAN.md`: one line pointing here for the new gap.
- ROADMAP: no edit without owner permission.

## Validation

1. Build with drews-xcode-mcp: no errors, no new warnings.
2. Targeted tests T1–T11 and the classes listed above (owner rule: no suite runs unless asked; report them as unrun if not approved). Run the full suite before merge, because persistence and lineage change (CLAUDE.md).
3. Freeze the build as `FrozenBuilds/DCM-<build>-<git>.app`.
4. Derive (read-only on the source):

   ```sh
   BIN="$HOME/Library/Application Support/DrewsChessMachine/FrozenBuilds/DCM-<build>-<git>.app/Contents/MacOS/DrewsChessMachine"
   M="$HOME/Library/Application Support/DrewsChessMachine/Models"
   "$BIN" --derive-model --from "$M/20261011-zlra-elitecyc-latest-step384964.safetensors" \
     --set-compute-dtype float32 --out "$M/20261112-zlra-elitecyc-fp32-start.safetensors"
   ```

   - Use a name without `-step<N>`, so it is never read as a step-enumerated checkpoint.
5. Check the derived file before launching:
   - `--describe-model` shows `compute_data_type float32`, tail `does_not_apply`, `trainer_completed_steps 384964`, `training_step 384964`, 41 velocity tensors, record `start: derive` with parent `20261025-1-qyUf` sha `343a3569…`, `fed.corpus` epoch 2 / next game 10,696,971 / feed_per_step 8533, streams and dropout state present, and test sets evaluated;
   - a `python3 -I` header-and-data comparison shows every tensor's bytes identical to the source's;
   - the source file's SHA-256 is unchanged.
6. Launcher `experiments/20261008-zlra-selfplay-lr/launch/elitecyc_fp32_resume_launch.sh`, modeled on `elitecyc_resume13_decay_launch.sh`:

   ```sh
   exec "$BIN" --replay-corpus 20260704-215145-op24Gp \
     --start-model "$M/20261112-zlra-elitecyc-fp32-start.safetensors" \
     --resume-exact --accept-inexact schedule,architecture \
     --parameters $E/parameters-zlra-elitecyc-decay.json \
     --out-model "$M/20261112-zlra-elitecyc-fp32-latest.safetensors" \
     --epochs 7 --training-step-limit 615036 --enumerate-checkpoints \
     --output $E/results-elitecyc-fp32-seg1.json \
     > $E/train-elitecyc-fp32-seg1.stdout 2>&1
   ```

   - 615,036 = 1,000,000 − 384,964. Both limits apply; the first reached ends the run.
   - Checkpoints are named `20261112-zlra-elitecyc-fp32-replay-step<T>`. The first is at 385,000.
   - If the run also reports `build` or `os` (fingerprint differs for bf16 on the new build), the refusal names them. Decide then whether to accept them; do not pre-accept.
7. On launch, confirm the D7 lines. Then check:
   - the first `[REPLAY]` step line's `pLoss` / `vLoss` / `pEnt` are close to segment 13's and the hilr fork's first lines (same weights; only the rounding differs);
   - `[MEM]` footprint and plies/hour (fp32 roughly doubles activation memory and slows steps);
   - no `[ALARM]` or `[GPU-ERR]`.
8. Experiment write-up per the house rule (README, E-NNNN page, registry entry for the new lineage run, with ancestry pointing at E709 at 384,964), commit, push.

## Open questions (with recommendations)

1. New run at the derive (D5) or continue E709 in place, with the derivation appended to the old record?
   - Recommend a new run. It follows the documented "run id minted on derive" rule, records the captured architecture in `architecture_at_departure`, and keeps E709's record unchanged.
   - Continuing in place would leave the record claiming the source save's build and segment while the file was written by derive, and the trained architecture would be lost.
2. Should a precision change be a resume gap?
   - Recommend yes, a new token `architecture`. The state is exact; the trajectory is not.
   - Alternative names: `compute_precision`. `architecture` stays correct if more no-rewrite kinds are ever allowed.
3. Fingerprint comparison for a derived trainer file.
   - Recommend computing the running fingerprint on the captured (departed) architecture (D6), so `build` / `os` keep their meaning.
   - Simpler alternative: compare against the file's architecture and accept `build` / `os` gaps whenever they changed. It is honest but noisy, and the real launch would then need `build,os` too.
4. Chained trainer-state derives (derive of a derived trainer file)?
   - Recommend refusing them. One call can apply both operations, and going back means deriving from the training save.
5. Pre-lineage trainer-state sources?
   - Recommend refusing them. There is nothing to carry for an exact resume, and branching still works.
6. Allow the other no-rewrite kinds (activations, ReZero cap) on trainer state?
   - Recommend no. They change the function the velocity was built for.
7. The GUI `forceFloat32` resume override (`SessionController+Checkpoint.swift:935-1070`) is a resume-time precision setting, the kind the owner rejected.
   - Recommend a separate decision: keep it for now, or retire it once derive covers trainer files. Sessions would need a session-level derive, which is out of scope.
8. Dashboard registry: the fp32 continuation is a new lineage run whose totals continue.
   - Recommend a registry entry `elitecyc-fp32` whose segment carries `cumstep_base` 384,964 and pins from E709's segment 13 record.
