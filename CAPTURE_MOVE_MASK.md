# Capture per-position legal-move mask during self-play; persist with replay records

## Context

Today, every training step regenerates each batch position's legal-move mask from scratch:

```swift
// ChessTrainer.swift:2583–2596
for pos in 0..<batchSize {
    let state = BoardEncoder.decodeSynthetic(from: boardPtr)
    let legalMoves = MoveGenerator.legalMoves(for: state)
    for move in legalMoves {
        masks[pos * policySize + PolicyEncoding.policyIndex(move, currentPlayer: .white)] = 1.0
    }
}
```

The same `decodeSynthetic` + `MoveGenerator.legalMoves` regen also happens once per `[STATS]` line inside `legalMassSnapshot` (`ChessTrainer.swift:2854`). This is the dominant per-batch CPU cost in `dataPrepMs`.

The legal-move list is already known at self-play time — `MPSChessPlayer.onChooseNextMove` receives it and uses it inside `sampleMove` to mask illegal logits before sampling. We should **build the same `Float[4864]` mask the trainer eventually needs once at self-play time, store it in the replay buffer alongside the position, and have the trainer `memcpy` it straight into the GPU staging buffer** — zero per-step CPU work, no decode, no move-gen, no loop.

## Design

### Storage: pre-built `Float[4864]` mask per position, in the replay buffer ring AND in the on-disk session checkpoint

**Scope of "the replay buffer":**

1. **In-memory ring** — a new flat `legalMaskStorage: UnsafeMutablePointer<Float>` sized `[capacity × 4864]`, parallel to the existing `boardStorage`. Allocated in `init`, freed in `deinit`. Same wraparound chunking on `append`. Same per-slot `srcIndex` indexing on `sample`.
2. **`.dcmsession` save / restore** — a new section in the on-disk binary format (v7), written/read alongside the existing 10 sections. Round-trip preserves bit-for-bit, gated by SHA-256 trailer like the rest of the file. Save Session, Resume Training, periodic 4-hour autosave, and post-promotion autosave all carry the new section automatically because they all funnel through the same `ReplayBuffer.write(to:)` / `restore(from:)` paths.
3. **Header arch check** — the header gains a 6th Int64 field `policyFloatsPerSlot` (currently 4864) so a future change to `ChessNetwork.policySize` is rejected at restore time with `PersistenceError.incompatiblePolicySize`, parallel to the existing `incompatibleBoardSize` guard on `floatsPerBoard`.

**Format choice:** store the mask in exactly the layout the trainer's GPU staging buffer expects (the `legalMaskND` `MPSNDArray` shaped `[batchSize, 4864]`, fed through the existing `legalMask` placeholder at `ChessTrainer.swift:1447`). Zero conversion in the trainer hot path — `sample(...)` does a single bulk copy per batch position straight into `replayBatchLegalMasks`.

- Per-position cost: `4864 × 4 = 19 456 B`. Existing per-position cost ~5 153 B (mostly the 1 280-float board). New per-position cost ~24 609 B (+377 %).
- On-disk size grows by the same factor per stored position. Reuse the existing `persistenceChunkBytes` chunking (32 MB) so peak `Data` allocations stay bounded for big rings.
- Trainer's per-step work to populate `replayBatchLegalMasks` collapses from "decode each board + run move generator + index-encode each move" to one bulk copy from the sampled rows.

### Files to modify

#### 1. `ReplayBuffer.swift`

- New storage:
  ```swift
  static let policyFloats = ChessNetwork.policySize  // 4864
  private let legalMaskStorage: UnsafeMutablePointer<Float>  // capacity × policyFloats
  ```
  Allocate in `init`, deinitialize in `deinit`, mirroring `boardStorage`.
- Update `static let bytesPerPosition` to add `policyFloats × MemoryLayout<Float>.size`.
- Extend `append(...)`:
  ```swift
  legalMasks: UnsafePointer<Float>,  // count × policyFloats
  ```
  Add a per-chunk `update(from:count:)` mirroring the boards section (same wraparound chunking, stride `policyFloats * MemoryLayout<Float>.size`).
- Extend `sample(...)`:
  ```swift
  legalMasks dstLegalMasks: UnsafeMutablePointer<Float>,  // mandatory; batchSize × policyFloats
  ```
  Per sample, copy `policyFloats` from `legalMaskStorage + srcIndex * policyFloats` into `dstLegalMasks + i * policyFloats`. This is the bulk copy the trainer wants.
- **Persistence schema → bump to v7, hard reject v4–v6.** (Save / Resume Training / periodic autosave / post-promotion autosave all use this path.)
  - Body adds one new section after `materialCounts`, written in oldest-first order via the existing `writeRange(...)` chunked helper:
    11. `legalMasks` (`policyFloats` × Float per slot)
  - Header gains a 6th Int64 = `policyFloatsPerSlot`. Mismatch raises a new `PersistenceError.incompatiblePolicySize(expected:got:)` (parallel to `incompatibleBoardSize`). `headerSize` increases from `8+4+4+5×8 = 56` to `8+4+4+6×8 = 64`. The `maxReasonableFloatsPerBoard`-style sanity cap pattern is mirrored as `maxReasonablePolicyFloats: Int64 = 65_536` so a corrupt header can't coax a giant allocation.
  - `fileVersion` bumped to 7. The version guard at `ReplayBuffer.swift:1237` becomes `version == fileVersion`; v4/v5/v6 throw `unsupportedVersion`. Drop the `isV4` / `isV5` branches and the v4 hash-recompute path that depended on them.
  - The SHA-256 trailer continues to cover every preceding byte (header + all 11 body sections + new section).
  - In-flight `.dcmsession` autosaves from earlier builds will fail to resume — they surface via the existing `setCheckpointStatus(.error)` path; user discards and starts fresh.

#### 2. `MPSChessPlayer.swift`

- Add per-game scratch alongside `gameBoardScratchPtr`:
  ```swift
  private var gameLegalMaskScratchPtr: UnsafeMutablePointer<Float>  // plyCapacity × ChessNetwork.policySize
  ```
  Allocate in `init` (mirror `gameBoardScratchPtr`; `initialize(repeating: 0, …)`), free in `deinit`, grow inside `growGameBoardScratch(toPlyCapacity:)` so it always tracks the boards ring's capacity.
- In `onChooseNextMove`, after `growGameBoardScratch` and before the `source.evaluate`:
  ```swift
  let maskBase = gamePliesRecorded * ChessNetwork.policySize
  // Zero the per-ply slot first — most cells stay zero.
  (gameLegalMaskScratchPtr + maskBase).update(repeating: 0, count: ChessNetwork.policySize)
  for move in legalMoves {
      let idx = PolicyEncoding.policyIndex(move, currentPlayer: gameState.currentPlayer)
      gameLegalMaskScratchPtr[maskBase + idx] = 1.0
  }
  ```
  The `PolicyEncoding.policyIndex` call here is the same one `sampleMove` already makes per legal move (`MPSChessPlayer.swift:644`). Net: one extra index call per legal move per ply at self-play time, which moves all the trainer's per-step move-gen + index-encode work to the (cheaper, parallelizable, GPU-batched) self-play side.
- In `onGameEnded` flush, pass `gameLegalMaskScratchPtr` to the new `legalMasks:` parameter on `replayBuffer.append(...)`.
- `onNewGame` does NOT need to zero the scratch — `onChooseNextMove` overwrites every ply slot before the buffer reads it (same convention as `gameBoardScratchPtr`).

#### 3. `ChessTrainer.swift`

- The existing `replayBatchLegalMasks: UnsafeMutablePointer<Float>?` staging buffer (line 1142) is already shaped `[capacity × policySize]` — exactly the destination `ReplayBuffer.sample` will write into. No new allocation needed.
- Update the production-path `sample(...)` call (~line 2455) to pass `legalMasks: replayBatchLegalMasks`.
- **Delete `ChessTrainer.swift:2583-2596`** entirely (the `BoardEncoder.decodeSynthetic` + `MoveGenerator.legalMoves` + `masks[i] = 1.0` block, including the `masks.update(repeating: 0.0, …)` zero-fill — the buffer now arrives populated).
- Step-0 `[MASK CHECK]` (lines 2598–2610) and step-200 `[MASKED-SOFTMAX]` probe (lines 2622–2657) — keep as-is; both read the just-populated `masks` and remain valid.
- **`legalMassSnapshot` (lines 2832–2924)** — same change pattern: have it ask `sample(...)` for the mask, and replace the `decodeSynthetic` + `MoveGenerator.legalMoves` + `legalIndexSet` block with a scan of the mask floats (`if mask[base + i] == 1.0 { legalExpSum += … }`). Drops the second `decodeSynthetic` site.

#### 4. Tests (`DrewsChessMachineTests/`)

- **`ReplayBufferTests.swift`** —
  - Extend `appendOnePosition` helper to populate a known `Float[4864]` mask; assert `sample` returns it bit-for-bit.
  - New round-trip persistence test: write v7, restore, verify legal masks preserved bit-for-bit, and that the new `policyFloatsPerSlot` header field rejects mismatch.
  - Update existing `unsupportedVersion` test to assert v4/v5/v6 now throw `unsupportedVersion`.
- **New equivalence test** (`ChessTrainerLegalMaskTests.swift` or in `ReplayBufferTests`): for a handful of real `GameState`s, build the mask via the new self-play path (`MPSChessPlayer`-style: zero + iterate `MoveGenerator.legalMoves` + set 1.0) and via the old trainer path (`BoardEncoder.decodeSynthetic` + `MoveGenerator.legalMoves` + `PolicyEncoding.policyIndex` with `currentPlayer: .white`); assert exact equality. Pins the round-trip semantics so a future encoder change can't silently desync them.
- **Note on `currentPlayer:`** the deleted trainer code passed `.white` as the current player at index time. `MPSChessPlayer` passes the actual `gameState.currentPlayer`. The encoded board is already pre-flipped to "current player on bottom" perspective inside `BoardEncoder.encode`, so both code paths produce the same encoder-frame indices — that invariant is exactly what the equivalence test pins.

### File path summary

| File | Change |
|---|---|
| `DrewsChessMachine/DrewsChessMachine/MPSChessPlayer.swift` | per-game `Float[4864]` mask scratch; populate in `onChooseNextMove`; flush in `onGameEnded` |
| `DrewsChessMachine/DrewsChessMachine/ReplayBuffer.swift` | new `legalMaskStorage`; `append`/`sample` gain mandatory `legalMasks`; persistence v7; hard reject v4–v6 |
| `DrewsChessMachine/DrewsChessMachine/ChessTrainer.swift` | thread `replayBatchLegalMasks` into `sample(...)`; delete the per-step `decodeSynthetic` + move-gen block (2583–2596); same in `legalMassSnapshot` (2854) |
| `DrewsChessMachine/DrewsChessMachineTests/ReplayBufferTests.swift` | extend round-trip + new persistence + equivalence tests |

## Verification

- `mcp__xcode-mcp-server__build_project` on `DrewsChessMachine.xcodeproj` — must build clean (single end-of-implementation build per project rules).
- `mcp__xcode-mcp-server__run_project_tests` — all existing tests pass; new tests pass.
- Launch the app, click **Play and Train**, watch the session log:
  - `[MASK CHECK]` at step 0 still reports `inLegalMask=true` for every of the first 8 positions (proves the mask the trainer sees from the buffer matches `moves[pos]`).
  - `[MASKED-SOFTMAX]` at step 200 still reports `legal_sum ≈ 1.0`, `illegal_sum ≈ 0.0`.
  - `[STATS]` line continues to report `legalMass`, `pEntLegal`, `top1Legal` consistent with a pre-change baseline.
- **Save / restore round-trip:** trigger File → Save Session, quit, relaunch, accept the **Resume Training** prompt. Then sample a batch from the restored buffer and confirm `[MASK CHECK]` still reports `inLegalMask=true` for every of the first 8 positions on the next training step — that proves the on-disk masks survived round-trip and feed the trainer correctly.
- **Periodic autosave round-trip:** run Play-and-Train past at least one 4-hour `periodic` autosave (or trigger one manually for the test); restart from that file; same mask sanity check.
- **Post-promotion autosave round-trip:** force a candidate promotion (or use a low `promoteThreshold` for the test); restart from the resulting `-promote.dcmsession`; same mask sanity check.
- **Old-file rejection:** confirm any pre-change `.dcmsession` file surfaces a clean `unsupportedVersion` error rather than silently corrupting a session.
- Per-step `dataPrepMs` should drop measurably — the per-batch `decodeSynthetic` + `MoveGenerator.legalMoves` regen was the dominant CPU cost in the prep window.
