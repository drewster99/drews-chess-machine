# Derive a trainer-state file's precision (plan)

Status: implemented 2026-11-12 (design below, as built; new tests written but not yet run — see Validation 2). Owner decision 2026-11-12: `--derive-model` accepts a trainer-state file for operations that rewrite no tensor (`--set-compute-dtype`, `--set-policy-tail-precision`), and the output can be continued with `--replay-corpus --resume-exact`. Precision stays an architecture field in the file. A resume-time precision flag was rejected.

Owner decisions 2026-11-12 (second round), which replace the first draft's recommendations:

- **OD-A, gap names.** Not `architecture` (too broad here). Two `ResumeGap` tokens named after the field that changed: `compute_data_type` and `policy_tail_precision`. Each is raised whenever the file's value differs from the value the trainer state was captured under, on the same build, OS and Mac too. (The removed `policy_tail` gap was a resume-time setting; the new token is a file whose architecture a derive changed after its trainer state was captured.)
- **OD-B, no new run.** The derive does not mint a run. Like every `--accept-inexact` resume, the run ID is kept, the next segment increments, and the accepted gaps are recorded (`start: resume`, `exact_resume: false`, `not_exact_items`). The derived file carries the source's lineage record unchanged except what must change; the later `--resume-exact` is an ordinary resume of that run with the precision gaps. The first draft's `trainerStateCopyRecord` (a new run) is dropped.
- **OD-C, chained derives allowed** (bf16 → fp32 → fp16 …). The resume walks back through the derive steps to the save where training happened and compares against that.
- **OD-D, GUI `forceFloat32` stays.** It duplicates this mechanism; a ROADMAP item records moving it onto a session-level derive later.
- **OD-E, tests.** Modifying tests the change requires is approved; new tests are preferred.

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

## Design (as built)

### D1. Which operations may take a trainer-state source

- `DeriveOperationKind.acceptsTrainerStateSource: Bool`, `true` only for `SetComputeDataTypeDeriveOperation.kind` and `SetPolicyTailPrecisionDeriveOperation.kind`.
- Why not every no-rewrite kind: activation and ReZero-cap setters change the function the velocity was accumulated for. Precision changes only the rounding of the same function.
- A source carries trainer state when it has any `opt.*` tensor or any `trainer_*` schedule key (`ModelDerivation.trainerStateEvidence`). Such a source with any operation whose kind does not accept one keeps today's `sourceHasOptimizerState(source:detail:)`, the detail naming the operation.
- A trainer-state source must also carry a trainer schedule (velocity alone, written before exact resume, cannot be resumed: `sourceHasOptimizerState`) and a lineage record (`trainerStateSourceWithoutLineage`).
- An operation that declares a tensor rewrite on a trainer-state source is an internal error (`trainerStateSourceRewrite`), checked before anything is written.
- Graft keeps calling `requirePlainModelSource` unchanged.

### D2. Tensors

- All tensors are copied bit-exact: base weights (fp32 masters as stored), BN statistics, and every `opt.<name>.velocity`. The weight plan and the trainable plan (velocity names and sizes) must be unchanged by the target architecture.

### D3. Trainer metadata (one check, shared with every trainer-file writer)

- The `trainer_*` keys (current and legacy forms) and `trainer_grad_norm_history` are replaced, not copied, by the entries of the decoded values. A legacy-schedule source comes out in the current form only.
- `SafetensorsModelIO.trainerStateMetadataEntries(schedule:gradNormHistory:trainingStep:lineage:)` holds the checks that lived inline in `prepare` (lineage step total == clock, `training_step` == clock, history ends at the clock, the record's schedule keys agree with the schedule) and returns the entries. `prepare` and the derive both call it.
- `training_step` = the clock (the existing rule: the source's `trainerStepOrStatedStep`).

### D4. Model identity

- New ModelID, `creator = derive-model`, flat `parent_model_id` = the source's ModelID, new `notes`, as for any derive. `notes` adds `trainer state carried (lineage segment <id>, trainer step N)`.
- `content_sha256` covers the data region only, so the derived file has the source's content hash (identical tensors).

### D5. Lineage: the source's record, carried (OD-B)

- `LineageRecord.carryingTrainerState(through:naming:)` returns the source's record with exactly two changes:
  - `derivation_history` gains this derivation (D6 below);
  - `model_naming` follows the derive rule (`ModelNaming.ofDerive`: `--name`, and an unedited preset marked edited, since the architecture changes).
- Everything else is the source's, and why it must stay so:
  - `run` (run ID, segment index and ID, start, exactness), `steps`, `fed` (corpus position, totals), `time`, `segments`, `ancestry`: the file still holds that segment's trainer state at that step. The next `--resume-exact` continues this run as segment index + 1 (OD-B).
  - `rng` (dropout Philox state, streams, behavior fingerprint), `parameters`, `run_seeds`: they describe the trainer snapshot the file still holds.
  - `build`, `device`, `invocation`: the process that captured the trainer state. The resume's build / OS / device comparison must be against it (OD-C), and the record never claims the derive trained.
  - `parent`: the file the captured segment trained from. The derived file's own parent is the derivation record's `parent_model_id` and the flat `parent_model_id`; the resumed segment records the derived file as its parent.
  - `configuration` (including `policy_tail_precision`): how the segment that last trained these weights trained. It stays a fact about that segment; the file's architecture states the tail it now builds, and the resumed segment records its own configuration.
- The record is re-encoded at the current schema (the record initializer, as `withoutTrainerState` does); a schema 2–4 source's unrecorded facts stay unrecorded and its parameter snapshot is carried word for word.
- `[RUN]` in the derive log describes the carried record; a `[DERIVE]` line says the record is the source's and names its run, segment and step.

### D6. The derivation record and the resume's capture rule (OD-A, OD-C)

- `ModelDerivation.DerivationRecord.trainerState` (`trainer_state`, written only by a trainer-state derive; omitted otherwise, so every older history reads and writes as before):
  - `segment_id` and `trainer_completed_steps`: the record segment and clock whose trainer state was carried;
  - `source_architecture` and `source_architecture_format_version`: the source's architecture as this build resolved it (a pre-v12 source's recorded tail included), encoded at the current format.
- `TrainerStateCapture(fileArchitecture:record:)` (`Training/ResumeExactness.swift`), the one rule for "which architecture captured this trainer state":
  - the derivation-history entries whose `trainer_state.segment_id` is the record's own `run.segment_id` are the trainer-state derives since the capture (a resumed segment has a new segment ID, so earlier derives never match);
  - none: the file's own architecture;
  - otherwise the first such entry's `source_architecture` — the architecture of the last save where training happened, however many derives followed.
  - Refused (never "no gap"): an entry whose `trainer_completed_steps` is not the record's clock; a captured architecture that differs from the file's in anything but `compute_data_type` / `policy_tail_precision`; an unreadable architecture.
- `ResumeGap.precisionGaps(capture:runningArchitecture:)` raises `compute_data_type` and/or `policy_tail_precision` when the running (= file) architecture's value differs from the captured one, with one `[RESUME]` line each naming both values and the derive(s).
- The running behavior fingerprint is computed for the **captured** architecture, so `build` / `os` / `device` keep meaning "does this build compute what the saved build computed for that trainer state"; the precision change is reported by its own gaps.
- Call sites: corpus replay and train-vs-UCI `--resume-exact`. The GUI resume is unchanged: a session folder never holds a derived trainer file, and `forceFloat32` stays as it is (OD-D).
- New tokens `compute_data_type`, `policy_tail_precision`, declared after `schedule`. The `--accept-inexact` help is generated from `ResumeGap.allCases`.

### D7. Expected real-run lines

- `[RESUME] schedule: …` (as in segment 13)
- `[RESUME] compute_data_type: the trainer state was captured under bfloat16; this file computes in float32 (changed by derive <id>)`
- `[RESUME] policy_tail_precision: the trainer state was captured under fp32_from_pre_bn; this file has does_not_apply (changed by derive <id>)`
- `[RESUME] build changed (2523 a242b7ad → <new>), behavior fingerprint matches` (the fingerprint is computed for bf16; 2663 matched)
- `[RESUME] OS changed (…), behavior fingerprint matches` (if the OS still differs)
- `[RESUME] NOT EXACT: schedule, compute_data_type, policy_tail_precision`
- `[RUN] path=replay run=E709F4EB… seg=14 (resume of <derivedID> sha=343a3569…, not exact: schedule, compute_data_type, policy_tail_precision) … policy_tail=does_not_apply cum_step=384964 …`

## Guardrails

- Only kinds with `acceptsTrainerStateSource` take a trainer-state source; every other kind, any mix including one, and graft keep `sourceHasOptimizerState`.
- A trainer-state source needs a schedule and a lineage record (D1).
- No tensor rewrite on a trainer-state source (D1); same weight and trainable plans (D2).
- The shared trainer-state check (D3) runs on the output; decode-back verification stays.
- Chained derives are allowed (OD-C); each appends one `trainer_state` entry with the same segment and clock.
- Never overwrites: unchanged (`FileSafety.publishNewFile`).

## Edge cases

- **fp32 → bf16** (a save trained in fp32): allowed; tail `fp32_from_pre_bn`. The bf16 trainer loads the fp32 working weights as masters. Gaps: `compute_data_type`, `policy_tail_precision`.
- **Tail only on a bf16 trainer file**: allowed; gap `policy_tail_precision` only.
- **Round trip** (bf16 → fp32 → bf16 with the captured tail): the capture is the first derive's source, equal to the file's architecture, so no precision gap; the derivation history still records both derives.
- **No-op** (`--set-compute-dtype bfloat16` on a bf16 file): refused as today (`operationNotApplicable`).
- **A session folder's `trainer.safetensors`**: allowed as a source; the output is a standalone file. A GUI-session trainer file has no corpus position, so corpus replay refuses it (`noCorpusPosition`); a train-vs-UCI trainer file resumes through `--train-vs-uci --start-model <file> --resume-exact` with `buffer` and the precision gaps.
- **Pre-v11 / pre-v12 sources**: written at the current format, `training_step` = the clock, the tail resolved into the architecture.
- **Pre-lineage (pre-v7) trainer files**: refused; they can still be branched.
- **Schema 2–4 records**: carried at the current schema.
- **Velocity NaN payloads**: copied as raw bit patterns.
- **Test-set results**: re-evaluated under the target architecture (GPU).
- **Follow-lineage (Lichess bot)**: the derived file claims the source's segment and step with the same `content_sha256`, so it is a same-weights copy of that save, never a conflict.
- **A branch** (`--start-model` without `--resume-exact`) from the derived file: an ordinary branch; the derivation history is carried.

## Tests

- `DeriveTrainerStateTests` (pure):
  - `--set-compute-dtype float32` on a bf16 trainer file: every tensor bit-identical, schedule and history equal, `training_step` = clock, fp32 / `does_not_apply`, the record equal to the source's except history and naming, the `trainer_state` entry.
  - `--set-policy-tail-precision`, and both operations in one call.
  - Refusals: every other kind and a mix; velocity without schedule; no lineage; graft unchanged.
  - Legacy schedule keys come out in the current form only.
  - An inconsistent source (record step ≠ clock) is refused by the shared check.
  - Chained derive: two entries, the capture is the first one's source.
- `TrainerStateCaptureTests` (pure): capture of a training save, of a derived file, of a chain; refusals; gap lines; tokens parse and order after `schedule`.
- `DeriveTrainerStateResumeTests` (GPU):
  - bf16 → fp32 → bf16 trainer state loads losslessly (masters ↔ working weights, velocity).
  - Corpus replay: train N steps in bf16, derive to fp32, `--resume-exact`: refused naming both tokens without `--accept-inexact`; with them it trains M steps from the saved clock and feed position, in fp32, as segment 1 of the same run, `not_exact_items` = the two tokens — on the same build, OS and Mac.

## Docs

- `documentation/deriving-models.md`, `App/DeriveModelCLI.swift` help, the operation summaries, `ModelDerivation.swift` header, `App/CommandLineHelp.swift` `--derive-model` line.
- `CLAUDE.md`: the resume-gap and policy-tail lines.
- `ROADMAP.md`: deferred item for `forceFloat32` (OD-D).

## Validation

1. Build with drews-xcode-mcp: no errors, no new warnings.
2. The targeted test classes above plus the related existing classes (derive, lineage, resume exactness, safetensors trainer-state, policy-tail provenance). The full suite before merge (owner rule: not run unless asked).
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
   - `--describe-model` shows `compute_data_type float32`, tail `does_not_apply`, `trainer_completed_steps 384964`, `training_step 384964`, 41 velocity tensors, the record of run `E709F4EB…` segment 13 unchanged, `derivation_history` ending in a `set-compute-dtype` entry with `trainer_state`, and test sets evaluated;
   - every tensor's bytes identical to the source's (same `content_sha256`);
   - the source file's SHA-256 unchanged.
6. Launcher `experiments/20261008-zlra-selfplay-lr/launch/elitecyc_fp32_resume_launch.sh`, modeled on `elitecyc_resume13_decay_launch.sh`:

   ```sh
   exec "$BIN" --replay-corpus 20260704-215145-op24Gp \
     --start-model "$M/20261112-zlra-elitecyc-fp32-start.safetensors" \
     --resume-exact --accept-inexact schedule,compute_data_type,policy_tail_precision \
     --parameters $E/parameters-zlra-elitecyc-decay.json \
     --out-model "$M/20261112-zlra-elitecyc-fp32-latest.safetensors" \
     --epochs 7 --training-step-limit 615036 --enumerate-checkpoints \
     --output $E/results-elitecyc-fp32-seg1.json \
     > $E/train-elitecyc-fp32-seg1.stdout 2>&1
   ```

   - 615,036 = 1,000,000 − 384,964. Both limits apply; the first reached ends the run.
   - If the run also reports `build` or `os` (the bf16 fingerprint differs on the new build), the refusal names them. Decide then; do not pre-accept.
7. On launch, confirm the D7 lines; then the first `[REPLAY]` line's `pLoss` / `vLoss` / `pEnt` against segment 13's, `[MEM]` and plies/hour, no `[ALARM]` / `[GPU-ERR]`.
8. Experiment write-up per the house rule; commit, push.

## Resolved questions

1. New run or continue E709 in place: continue (OD-B).
2. A precision change is a resume gap, two tokens (OD-A).
3. Fingerprint: computed for the captured architecture (D6).
4. Chained derives: allowed (OD-C).
5. Pre-lineage trainer-state sources: refused.
6. Other no-rewrite kinds on trainer state: no.
7. GUI `forceFloat32`: kept; ROADMAP item (OD-D).
8. Dashboard registry: the fp32 continuation is the next segment of E709, branching from segment 13 at 384,964 (a fork of the run, as the hilr fork was).
