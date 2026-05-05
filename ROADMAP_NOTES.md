# ROADMAP Update Notes — 2026-05-05

This file documents the reconciliation of `ROADMAP.md` against the actual code in
`/Users/andrew/Documents/ncc_source/cursor/drews-chess-machine`.

## Summary of changes

- Reorganized the top of `ROADMAP.md` into:
  - `Future improvements (validated open)`
  - `Completed / corrected from older Future entries`
  - `Decisions not pursued / historical notes`
- Moved implemented or obsolete items out of the active future list while preserving their context.
- Corrected technical claims that had drifted since the original roadmap was written.
- Kept the existing `Findings` and `Completed` sections, with one stale v2-status paragraph corrected.
- Created `/tmp/wip_roadmap_analysis.md` with the working analysis and evidence trail.

## Corrections and code evidence

### 1. `BatchFeedsInput` remains open

**Correction:** No behavioral change; the item remains a valid future safety refactor.

**Evidence:** `DrewsChessMachine/DrewsChessMachine/Training/ChessTrainer.swift` still defines:

```swift
private func buildFeeds(
    batchSize: Int,
    boards: UnsafePointer<Float>,
    moves: UnsafePointer<Int32>,
    zs: UnsafePointer<Float>,
    vBaselines: UnsafePointer<Float>,
    legalMasks: UnsafePointer<Float>
) -> [MPSGraphTensor: MPSGraphTensorData]
```

The same-typed float pointers are still positional, so the original swap-risk remains.

### 2. Autosave retention pruning remains open, but sessions are heavier now

**Correction:** The roadmap now states that `.dcmsession` saves may include `replay_buffer.bin`; the old wording was not sufficient for current disk-footprint risk.

**Evidence:**
- `UpperContentView.periodicSaveIntervalSec` is `4 * 60 * 60`.
- `PeriodicSaveController` schedules and defers periodic saves.
- `UpperContentView.autosaveSessionsOnPromote` is `true`.
- `CheckpointManager.saveSession` optionally writes `replay_buffer.bin` when `state.hasReplayBuffer == true`.
- No implementation of `Manage Autosaves`, `Trim to last N`, or retention pruning was found.

### 3. Human-vs-model play is not implemented

**Correction:** Existing “Play Game” is model-vs-model, not human-vs-model.

**Evidence:** `UpperContentView.playSingleGame()` creates a `ChessMachine`, a `DirectMoveEvaluationSource(network: network)`, and two `MPSChessPlayer` instances:

```swift
let white = MPSChessPlayer(name: "White", source: source)
let black = MPSChessPlayer(name: "Black", source: source)
```

Searches found no `HumanPlayer`, user-move bridge, slot picker, or side picker.

### 4. Adaptive LR schedule is not implemented, but warmup/sqrt scaling are

**Correction:** The roadmap now distinguishes the implemented LR mechanics from the unimplemented schedule.

**Evidence:**
- `TrainingParameters.LearningRate` defaults to `5.0e-5`.
- `TrainingParameters.LRWarmupSteps` defaults to `100`.
- `TrainingParameters.SqrtBatchScalingLR` defaults to `true`.
- `ChessTrainer.buildFeeds` computes effective LR as base LR multiplied by optional sqrt-batch scaling and warmup.
- No code fields or UI for `lr_init`, schedule on/off, positions-per-decay `τ`, promotion multiplier, LR floor, or positions-based exponential decay were found.

### 5. Model/session save-load is implemented and moved out of Future

**Correction:** The old “Today nothing persists across app launches” claim is now historical. The roadmap now documents the as-built implementation and points to the existing detailed Completed entry for durability.

**Evidence:**
- `ModelCheckpointFile.swift` implements `.dcmmodel` encode/decode with arch/hash validation.
- `SessionCheckpointFile.swift` defines `SessionCheckpointState` and `SessionCheckpointLayout`.
- `CheckpointManager.saveSession` writes `champion.dcmmodel`, `trainer.dcmmodel`, `session.json`, and optional `replay_buffer.bin`, then verifies before final rename.
- `DrewsChessMachineApp.swift` File menu includes Save/Load Session/Model and parameter actions.
- `CheckpointPaths` uses `~/Library/Application Support/DrewsChessMachine/Sessions/` and `Models/`.

### 6. Replay buffer persistence is now v6, not the old v3/v4-only state

**Correction:** The roadmap now says replay-buffer durability is complete and current format is v6. The stale note saying TODO_NEXT #3 remained open was replaced.

**Evidence:** `ReplayBuffer.swift` has:

```swift
private static let fileVersion: UInt32 = 6
```

The v6 layout includes boards, moves, outcomes, vBaselines, v5 metadata, v6 material counts, and a SHA-256 trailer. `restore(from:)` accepts v4/v5/v6 and rejects v1-v3.

### 7. Training legal-move masking is implemented in graph

**Correction:** The old future item “Fuse legal-move masking into the policy head” was no longer accurate for training. The roadmap now says training masks in-graph while inference remains raw-logit.

**Evidence:** `ChessTrainer.buildTrainingOps` creates `legal_move_mask`, computes `masked_logits`, and passes masked logits to `softMaxCrossEntropy`:

```swift
let legalMask = graph.placeholder(... name: "legal_move_mask")
let illegalMask = graph.subtraction(oneConst, legalMask, name: "illegal_mask")
let additiveMask = graph.multiplication(illegalMask, largeNeg, name: "additive_mask")
let maskedLogits = graph.addition(network.policyOutput, additiveMask, name: "masked_logits")
let ceLossRaw = graph.softMaxCrossEntropy(maskedLogits, labels: oneHot, ...)
```

`buildFeeds` writes `legalMasks` into `cached.legalMaskND` and includes `legalMaskPlaceholder` in the feed dictionary.

### 8. Inference legal masking is not pursued by default

**Correction:** The remaining inference-side masking idea was moved to Decisions Not Pursued / historical notes.

**Evidence:** `ChessNetwork.evaluate` returns raw logits; `ChessRunner.makeInferenceResult` softmaxes those logits for UI; `MPSChessPlayer` samples from legal moves. Raw logits support illegal-mass diagnostics.

### 9. Top-k heap/quickselect is not worth doing now

**Correction:** The policy vector size was corrected from 4096 to 4864, and the item was moved out of active Future work.

**Evidence:**
- `ChessNetwork.policyChannels = 76`; `policySize = policyChannels * boardSize * boardSize = 4864`.
- `ChessRunner.extractTopMoves` intentionally full-sorts the entire vector to survive collapsed off-board top cells.
- The path is UI/demo-oriented, not self-play hot path.

### 10. `MPSGraphExecutable` remains open

**Correction:** No material change; the roadmap now cites current evidence.

**Evidence:** `ChessNetwork` inference/training paths still call `graph.run(...)`; `ChessMPSNetwork.NetworkInitMode.package` still throws `packageLoadingNotImplemented`.

## Patterns discovered

- The roadmap mixed planned work and later completed work. The codebase had advanced, but some old Future entries were never demoted.
- Persistence is now much more robust than the older plan: sessions can carry replay buffers and have atomic/fync/self-verification behavior.
- Training and inference intentionally have different policy contracts: graph-masked training loss vs raw-logit inference diagnostics.
- The parameter system is broad and macro-backed, but adaptive LR schedule controls are not present.
