# Input Plane Redesign + History Banks

A design / implementation plan for two combined architectural changes to `DrewsChessMachine`:

1. **Cleanup of the current 30-plane input encoding** — drop dead rep-related planes, replace the structurally-unused `rep≥2` plane with a useful lookahead-3fold indicator.
2. **Addition of N-bank history input** — give the network temporal context via a stack of N prior position encodings, AlphaZero-style.

Both stages force a stem-shape change (which invalidates existing `.dcmmodel` / `.dcmsession` files), so they are bundled to avoid two fresh-init training restarts.

A separate plan covers Phase 2 (early-draw adjudication with shadow-mode logging) — independent of this work, can ship before, during, or after.

---

## Background: what the analysis showed

Before deciding to make changes, we ran several diagnostics on the most-recent saved trainer model (`20260514-1-KbHZ-17`, step 494,927) and its replay buffer (1.5M positions, 4.2B totalAdded). Tools used (all in `tools/`):

- `stem_plane_norms.py` — per-input-plane L2 norm of the stem conv weight
- `rep_planes_in_buffer.py` — sampled rep-plane firing rates in the replay buffer
- `full_buffer_analysis.py` — full-buffer rep-plane co-occurrence, outcome conditioning, supplementary stats (halfmove × outcome, ply distribution, material distribution, etc.)
- `dump_dcmmodel.py` — pre-existing model-inspection script

In-network probe tests (in `DrewsChessMachineTests/RepPlaneProbeTests.swift`):
- `test_probeRepetitionPlanes_writeReport` — in-distribution rep-plane patterns (rep18 alone, rep18+mask t-4, rep18+mask t-8)
- `test_constructedKnightShuffleEndgame_probe` — real K+N+3P endgame after 4-ply knight shuffle through ChessGameEngine, naturally-populated rep state
- `test_halfmoveProbe_writeReport` — varying plane 17 (halfmove clock) across 20 positions

### Key findings

**Rep planes — 9 of 12 are dead or near-dead:**

| Plane | Label | Firings (full buffer) | % | Status |
|---|---|---|---|---|
| 18 | rep≥1 | 64,528 | 4.302% | **WORKHORSE — keep** |
| 19 | rep≥2 | 0 | 0.000% | **Structurally dead** (see below) |
| 20 | mask t-1 | 0 | 0.000% | Impossible (odd ply → STM mismatch) |
| 21 | mask t-2 | 0 | 0.000% | Possible but no 2-ply chess cycle exists |
| 22 | mask t-3 | 0 | 0.000% | Impossible (odd ply) |
| 23 | mask t-4 | 55,933 | 3.729% | **WORKHORSE — keep** |
| 24 | mask t-5 | 0 | 0.000% | Impossible (odd ply) |
| 25 | mask t-6 | 77 | 0.005% | Too rare to learn — drop |
| 26 | mask t-7 | 0 | 0.000% | Impossible (odd ply) |
| 27 | mask t-8 | 3,030 | 0.202% | Rare but in-distribution probe shows network learned it — keep |
| 28 | mask t-9 | 0 | 0.000% | Impossible (odd ply) |
| 29 | mask t-10 | 37 | 0.002% | Too rare — drop |

**Why plane 19 (`rep≥2`) is structurally dead:** Position encoding into the replay buffer happens *before* the move is applied (per the driver flow in `ActiveGame.swift:270-280`). The state where `repetitionCount` becomes 2 only exists *after* the move that creates the 3rd visit has been applied — and that state is the terminal state of the game (threefold draw). Terminal states are never written to the replay buffer because there's no policy target to learn. So plane 19's firing condition can never be present in any training sample. Confirmed by full-buffer scan: zero firings in 1.5M positions.

**Why the odd-ply masks (planes 20, 22, 24, 26, 28) are mathematically impossible:** Position repetition requires the same side-to-move. Same STM requires an even-ply offset. So planes corresponding to odd-ply offsets (`t-1`, `t-3`, `t-5`, `t-7`, `t-9`) cannot fire by the rules of chess. The encoder dutifully produces zero for them on every position, forever.

**Co-occurrence: every mask firing implies rep≥1.** Mask t-4, t-8, etc. all 100% co-fire with rep18 (if a recent position was a duplicate, the position has been visited before — trivially true). So masks add cycle-length specificity to rep18, not new presence info. Specifically:

- When `rep18` fires: 87% of the time it's accompanied by `mask t-4` (4-ply knight shuffle), 5% `mask t-8` (8-ply rotation), 8% no mask (long-ago repetition past the 10-ply window).

**Outcome correlation: rep planes ARE drawish.** Overall buffer: 14% W / 72% D / 14% L. Conditioned on plane firing:

| Combo | n | P(W) | P(D) | P(L) | Decisiveness |
|---|---|---|---|---|---|
| no rep | 1,435,472 | 0.142 | 0.716 | 0.141 | 28.4% |
| rep18 + mask t-4 | 55,933 | 0.056 | **0.894** | 0.050 | 10.6% |
| rep18 + mask t-8 | 3,030 | 0.037 | **0.930** | 0.033 | 7.0% |
| rep18 alone (no mask) | 5,451 | 0.082 | 0.849 | 0.069 | 15.1% |

**In-distribution probe results: network has learned the rep signal correctly.**

| Pattern | mean policy KL | mean ΔDraw |
|---|---|---|
| rep18 alone | 2.10 nats | -0.014 |
| rep18 + mask t-4 | 2.40 nats | **+0.069** |
| rep18 + mask t-8 | 3.18 nats | **+0.043** |

The 4-ply shuffle pattern shifts p_draw up by 6.9 percentage points — meaningful and in the right direction. The constructed K+N+3P endgame probe additionally showed the network already predicts 98% draw from board material alone in clear endgame positions, but the rep planes still reshape the *policy* by 1.4 nats KL (deciding whether to continue the shuffle or break out).

**Halfmove plane (17) is working well — leave alone.** The halfmove probe shows a clean monotonic ΔDraw with halfmove value, peaking at +20.8 pp at hm=0.99. Training data shows P(D) goes from 71% at hm=0–5 to 97% at hm≥75 — a +26 pp correlation. The network recovers ~80% of the data correlation. The continuous broadcast encoding is being learned correctly. No threshold planes needed.

**Sign consistency check passes.** Outcome by side-to-move: white-to-move scalar = +0.0113, black-to-move scalar = −0.0086. Mirrored as expected for STM-relative outcome encoding.

---

## Decisions made

1. **Drop 9 rep planes** (20, 21, 22, 24, 25, 26, 28, 29 — and replace 19 with a useful lookahead).
2. **Keep planes 18, 23, 27** — the workhorses (rep≥1, mask t-4, mask t-8).
3. **Replace plane 19 with `lookahead-3fold-available`** — a binary flag indicating "at least one of my legal moves from this position would force a 3-fold repetition (= force a draw claim)". This converts a structurally-dead plane into a real strategic signal that fires before threefold draws happen, exactly when the network can act on it.
4. **Add N=10 history banks** to give the network temporal context (AZ-style stack).
5. **Each bank uses an identical 22-plane layout** — uniform across all banks, including bank 0 (current). Slot semantics are constant: plane k in bank N means "the value of feature k at the position N plies before current."
6. **Historical banks carry their own historical values** — bank N's rep planes, halfmove, lookahead-3fold are the values that position HAD at the time it was the current position, not the current bank-0 values. This falls out naturally from the existing per-ply scratch storage which already records the full encoded board at each ply.
7. **All banks flipped into current STM's perspective at sample time.** Each historical bank whose ply's STM differs from current bank's STM gets a vertical flip + my/their piece-and-castling swap applied during the gather step. Halfmove + rep planes + lookahead are color-agnostic and don't need flipping. This preserves the existing AZ-style "network always sees position as if white-to-move" invariant.
8. **Pre-game banks (corresponding to plies before game start) are zero-padded.** The literal starting position (ply 0) is real data and uses its actual encoding; banks beyond that point are zero. This matches AlphaZero / Lc0 convention.
9. **Replay buffer stores per-ply encodings, not gathered N-bank tensors.** Storage stays at ~22 planes × 64 = 1408 floats per stored position (instead of 14,080 if we stored the full gathered tensor). At sample time, the trainer gathers N consecutive ply slots from the buffer to construct the full input. This requires positions to be stored in game-contiguous order (the "reverse-game-order" trick), with game-boundary detection (via `workerGameId` metadata) to zero-pad when the gather walks off the game's start.
10. **Halfmove plane stays as a single continuous broadcast plane.** Probe results show the network has learned the current encoding. No threshold planes added.
11. **Phase 2 (early-draw adjudication shadow logging) is independent.** Not in this plan. Separate work, can ship anytime.

### Decisions deliberately not made (open questions deferred)

- **Whether to also add a "lookahead-2fold-available" plane** (any legal move creates a 2nd visit, signaling cycle potential). Decided no — plane 18 already gives "I've been here before," and the network's own legal-move enumeration covers "can I get back to a familiar spot." Not worth a new plane.
- **Whether to enhance the halfmove encoding with binary threshold planes** (`hm≥50`, `hm≥75`). Decided no — current continuous encoding is working correctly per probe results; threshold planes would add input bandwidth for marginal benefit.
- **N=10 vs N=8 (AlphaZero used 8 history banks).** Choosing 10 for slight extra context; can revisit if profiling shows it matters.

---

## Target architecture

### Per-bank plane layout (22 planes, identical for every bank)

| Idx | Semantic | Type | STM flip behavior |
|---|---|---|---|
| 0 | my pawn | sparse | swap with 6, vertical flip |
| 1 | my knight | sparse | swap with 7, vertical flip |
| 2 | my bishop | sparse | swap with 8, vertical flip |
| 3 | my rook | sparse | swap with 9, vertical flip |
| 4 | my queen | sparse | swap with 10, vertical flip |
| 5 | my king | sparse | swap with 11, vertical flip |
| 6 | their pawn | sparse | swap with 0, vertical flip |
| 7 | their knight | sparse | swap with 1, vertical flip |
| 8 | their bishop | sparse | swap with 2, vertical flip |
| 9 | their rook | sparse | swap with 3, vertical flip |
| 10 | their queen | sparse | swap with 4, vertical flip |
| 11 | their king | sparse | swap with 5, vertical flip |
| 12 | my castling KS | broadcast | swap with 14 (no spatial flip) |
| 13 | my castling QS | broadcast | swap with 15 (no spatial flip) |
| 14 | their castling KS | broadcast | swap with 12 (no spatial flip) |
| 15 | their castling QS | broadcast | swap with 13 (no spatial flip) |
| 16 | en passant target | sparse | vertical flip in place |
| 17 | halfmove clock (normalized 0..1) | broadcast | none (color-agnostic) |
| 18 | rep≥1 | broadcast | none (color-agnostic) |
| 19 | **lookahead-3fold-available** | broadcast | none (color-agnostic) |
| 20 | mask t-4 (4-ply cycle) | broadcast | none (color-agnostic) |
| 21 | mask t-8 (8-ply rotation) | broadcast | none (color-agnostic) |

### Full input tensor

- **N = 10 history banks** (current + 9 prior plies)
- **22 planes per bank** × 10 banks = **220 planes**
- **8 × 8 spatial** → tensor shape `[batch, 220, 8, 8]`
- **14,080 floats per position** input (= 220 × 64)

### Per-bank gather rules

For a sample whose current position is at game-ply T:

```
Bank 0 = encoding of position at ply T          (no flip)
Bank 1 = encoding of position at ply T-1        (flip — opposite STM from bank 0)
Bank 2 = encoding of position at ply T-2        (no flip — same STM as bank 0)
Bank 3 = encoding of position at ply T-3        (flip)
...
Bank 9 = encoding of position at ply T-9        (flip iff 9 is odd → flip)
```

If `T - bank_index < 0`, that bank is zero-filled (1408 zero floats). The literal starting position (ply 0) is real data, NOT padding.

### Replay buffer storage layout

Per-position stored: still the **22-plane encoding (1408 floats)** plus existing metadata (move, outcome, plyIndex, gameLength, samplingTau, stateHash, workerGameId, materialCount). Storage per position grows from 7,709 bytes (current 30-plane) to **5,661 bytes (new 22-plane)** — actually shrinks because we have fewer planes. At 1.5M capacity: ~8.5 GB (down from current 10.8 GB).

**File format version bumps from v7 to v8.** Header `floatsPerBoard` field becomes 1408. Game-contiguous storage ordering is enforced (new game starts always begin at a fresh ring slot; if the new game can't fit before end-of-ring, ring write pointer advances to slot 0). This ensures sample-time gather can read N consecutive slots without crossing game boundaries (game boundary detection via `workerGameId` mismatch — if scanning back from sample offset finds a different `workerGameId`, zero-pad from that point).

---

## Implementation: Stage A (cleanup, single-bank)

Validates the new 22-plane layout independently before stacking 10 banks. No history banks yet — input is still single-position, just with the new semantics.

### Stage A code changes

#### 1. `Network/ChessNetwork.swift` (line 124)

```swift
// before:
static let inputPlanes = 30

// after:
static let inputPlanes = 22
```

Stem conv weight shape (`Network/ChessNetwork.swift:376–380`) uses `Self.inputPlanes` correctly — picks up automatically.

#### 2. `Encoding/BoardEncoder.swift` GameState — add lookahead field

Add a new stored field to the struct (after `recentRepetitionMask`):

```swift
/// True iff at least one legal move from this position would force a
/// 3-fold repetition (i.e., create a 3rd visit, allowing the player
/// to claim a draw). Drives BoardEncoder plane 19. Defaults to false
/// so legacy GameState constructions in tests/UI compile without
/// changes. Populated by ChessGameEngine in applyMoveAndAdvance after
/// it has computed currentLegalMoves and updated positionCounts.
let lookaheadThreefoldAvailable: Bool
```

Update the memberwise init signature with `lookaheadThreefoldAvailable: Bool = false` as a defaulted param. Add a `with` helper:

```swift
func withLookaheadThreefoldAvailable(_ flag: Bool) -> GameState {
    GameState(
        board: board,
        currentPlayer: currentPlayer,
        whiteKingsideCastle: whiteKingsideCastle,
        whiteQueensideCastle: whiteQueensideCastle,
        blackKingsideCastle: blackKingsideCastle,
        blackQueensideCastle: blackQueensideCastle,
        enPassantSquare: enPassantSquare,
        halfmoveClock: halfmoveClock,
        repetitionCount: repetitionCount,
        recentRepetitionMask: recentRepetitionMask,
        lookaheadThreefoldAvailable: flag
    )
}
```

Update the existing `withRepetitionCount` and `withRecentRepetitionMask` helpers to also propagate the new field.

#### 3. `Encoding/BoardEncoder.swift` encode function — new plane assignment

The current encode function has separate loops for planes 18-19 (rep counts) and 20-29 (per-bit temporal mask). Replace with explicit per-plane assignment:

```swift
// Plane 18: rep≥1 (unchanged condition)
if state.repetitionCount >= 1 {
    fillPlane(output, planeIndex: 18, value: 1.0)
}
// Plane 19: lookahead-3fold-available (NEW — was plane 19 rep≥2)
if state.lookaheadThreefoldAvailable {
    fillPlane(output, planeIndex: 19, value: 1.0)
}
// Plane 20: mask t-4 (was plane 23, bit 3 of recentRepetitionMask)
if (state.recentRepetitionMask & (1 << 3)) != 0 {
    fillPlane(output, planeIndex: 20, value: 1.0)
}
// Plane 21: mask t-8 (was plane 27, bit 7 of recentRepetitionMask)
if (state.recentRepetitionMask & (1 << 7)) != 0 {
    fillPlane(output, planeIndex: 21, value: 1.0)
}
```

Delete the old per-bit loop for planes 20-29.

#### 4. `Chess/ChessGameEngine.swift` applyMoveAndAdvance — compute lookahead

In `applyMoveAndAdvance` after the state has been updated with rep_count and mask (around line 220), and after `currentLegalMoves = nextMoves` (line 223), add:

```swift
// Compute lookahead-3fold-available: does ANY legal move from the new
// position result in a position that would be the 3rd visit (forcing
// threefold)? This is what the network would want to know — "I can
// claim a draw by playing one of these moves."
//
// Cost: N legal moves × (apply + key + dict lookup). Typically 20-40
// legal moves × ~few µs each = small. Worth measuring in profiling
// but should not dominate self-play.
var lookaheadFlag = false
for candidateMove in nextMoves {
    let candidateState = MoveGenerator.applyMove(candidateMove, to: state)
    let candidateKey = PositionKey(from: candidateState)
    if (positionCounts[candidateKey] ?? 0) >= 2 {
        // candidateState would be the 3rd visit
        lookaheadFlag = true
        break
    }
}
state = state.withLookaheadThreefoldAvailable(lookaheadFlag)
```

(Note: this must happen AFTER `positionCounts[key] = totalVisits` at line 197 so the just-applied state's visit count is reflected. And it must happen BEFORE `updateResult` at line 224 so the final state passed downstream has the lookahead flag set.)

#### 5. `Views/Board/TensorChannelNames.swift` — 30 → 22 element arrays

Replace the 30-element `names` and `shortNames` arrays with 22-element arrays:

```swift
public static let names: [String] = [
    "My Pawn", "My Knight", "My Bishop", "My Rook", "My Queen", "My King",
    "Their Pawn", "Their Knight", "Their Bishop", "Their Rook", "Their Queen", "Their King",
    "My Castling KS", "My Castling QS", "Their Castling KS", "Their Castling QS",
    "En Passant", "Halfmove Clock",
    "Rep ≥ 1", "Lookahead 3-fold", "Mask t-4", "Mask t-8",
]
public static let shortNames: [String] = [
    "mP", "mN", "mB", "mR", "mQ", "mK",
    "tP", "tN", "tB", "tR", "tQ", "tK",
    "mKS", "mQS", "tKS", "tQS",
    "EP", "HM",
    "r≥1", "L3F", "t-4", "t-8",
]
```

The assertion in `App/DrewsChessMachineApp.swift:248–250` (which checks both arrays' counts match `ChessNetwork.inputPlanes`) will catch any mismatch at launch.

#### 6. Test updates

- `BoardEncoderTests.swift:26-30` — change `1920` to `1408` (= 22 × 64), update the comment.
- `RepetitionPlaneTests.swift:277-281` — update `tensorLength` assertion to 1408. Delete tests that assert plane indices for dropped masks (planes 20, 22, 24, 26, 28, 29 — all gone; planes 21, 25 — gone). Tests for the kept masks need their indices updated:
  - Old plane 23 (mask t-4) → new plane 20
  - Old plane 27 (mask t-8) → new plane 21
  - Old plane 18 (rep≥1) → new plane 18 (unchanged)
  - Old plane 19 (rep≥2) → DELETED, replaced by new plane 19 (lookahead-3fold)
- Add a NEW test in `RepetitionPlaneTests.swift`:

```swift
func test_lookaheadThreefoldAvailable_firesWhenLegalMoveForcesRepetition() throws {
    // Build a position where a 4-ply knight shuffle has been played 2x
    // (4 plies happened, position back to original; play another 4 plies,
    // back to original — now 3rd visit just happened). Actually we want
    // the position BEFORE the final move that would create 3rd visit.
    // So: 4-ply shuffle once (positionCounts[start] = 2). Then white at
    // the start position again, about to play a move that returns to a
    // position visited before. lookahead should fire.
    // ... constructed position via PieceCotr + Move sequence as in
    //     test_constructedKnightShuffleEndgame_probe ...
    XCTAssertTrue(engine.state.lookaheadThreefoldAvailable,
                  "lookahead should fire when a legal move forces 3rd visit")
}
```

- `PolicyHeadCorrectnessTests.swift:700-706` — update `1920` to `1408`. Update plane-index references in comments.
- `RepPlaneProbeTests.swift:84` — change `30 * 64` to `22 * 64`. Update the variant patterns to use the new layout (rep18 at 18, lookahead at 19, mask t-4 at 20, mask t-8 at 21). Delete patterns referencing planes that no longer exist.

#### 7. `Persistence/ModelCheckpointFile.swift:113-135`

No code change needed — `currentArchHash` includes `mix(ChessNetwork.inputPlanes)` at line 132 and will auto-update to a new hash when inputPlanes changes from 30 → 22. This invalidates all existing `.dcmmodel` and `.dcmsession` files. Required.

#### 8. `CLAUDE.md`

Update the "Board encoding and policy space" section to describe the new 22-plane layout. Note that planes 20-29 (the old temporal-rep mask) have been consolidated into just 2 planes (mask t-4 and mask t-8), and plane 19 is now lookahead-3fold instead of rep≥2.

### Stage A verification

1. `mcp__xcode-mcp-server__build_project` — clean build, zero errors.
2. `mcp__xcode-mcp-server__run_project_tests` — all tests pass. New lookahead test passes.
3. Update `tools/stem_plane_norms.py` constants — change `INPUT_PLANES = 30` to `22` and update plane labels list. Re-run against a freshly-built network (won't load old models since archHash changed).
4. Launch DrewsChessMachine app, Build Network, observe `[APP]` log banner reports `inputPlanes=22 policySize=4864`.
5. Run Play-and-Train for ~5 minutes, verify `[STATS]` lines emit normally, no encoding-size mismatches in the log.
6. Confirm the lookahead plane is firing by adding a temporary debug log in `BoardEncoder.encode` and observing it fires during self-play.

---

## Implementation: Stage B (history banks)

Adds 10-bank history input on top of Stage A's 22-plane single-bank.

### Stage B code changes

#### 1. `Network/ChessNetwork.swift`

```swift
// Add two new constants:
static let planesPerBank = 22       // = was inputPlanes after Stage A
static let historyBankCount = 10

// Change inputPlanes to be the total input planes (banks × per-bank):
static let inputPlanes = planesPerBank * historyBankCount   // = 220
```

This means `inputPlanes` now refers to the network's TOTAL input plane count (220), not the per-bank count. All sites that previously meant "encoding for one position" need to migrate from `inputPlanes` to `planesPerBank`.

#### 2. `Encoding/BoardEncoder.swift`

Split the existing `tensorLength` into two semantic constants:

```swift
// Floats per single-position encoding (one bank): 22 × 8 × 8 = 1408
static let perBankTensorLength = ChessNetwork.planesPerBank * ChessNetwork.boardSize * ChessNetwork.boardSize

// Floats per network input tensor (all banks gathered): 220 × 8 × 8 = 14,080
static let networkInputTensorLength = ChessNetwork.inputPlanes * ChessNetwork.boardSize * ChessNetwork.boardSize

// Deprecated — exists during migration only, equals perBankTensorLength.
// Audit and remove after every site has been migrated to the right one.
@available(*, deprecated, message: "Migrate to perBankTensorLength or networkInputTensorLength")
static let tensorLength = perBankTensorLength
```

The encode function continues to produce 1408 floats (single-bank encoding). Unchanged from Stage A.

#### 3. NEW: `Encoding/HistoryBankGather.swift`

A new file containing the gather + flip logic.

```swift
import Foundation

/// Gathers N per-ply encodings from a per-game scratch buffer into a
/// network input tensor with the historical banks flipped into the
/// current bank's STM perspective.
///
/// The current position (bank 0) is never flipped. For each historical
/// bank N (1 ≤ N < historyBankCount), if N is odd the bank's
/// encoding is flipped because its source ply's STM is the opposite
/// of the current bank's STM. If N is even, no flip (same STM).
///
/// Banks corresponding to plies < 0 (game hasn't existed that long
/// yet) are zero-filled. The literal starting position (ply 0) is
/// real data and uses its actual encoding.
enum HistoryBankGather {
    /// Gather banks for a sample at currentPly into outputBuffer.
    ///
    /// - Parameters:
    ///   - perPlyEncodings: Array of per-ply 1408-float encodings,
    ///     indexed by ply (encoding for ply T is at index T). Caller
    ///     must guarantee size ≥ currentPly + 1.
    ///   - currentPly: The ply being sampled (bank 0 = this ply).
    ///   - outputBuffer: Destination, must be sized
    ///     `historyBankCount * perBankTensorLength` floats.
    static func gather(
        perPlyEncodings: UnsafeBufferPointer<Float>,
        encodingStride: Int,                // typically perBankTensorLength
        currentPly: Int,
        outputBuffer: UnsafeMutablePointer<Float>
    ) {
        let perBank = BoardEncoder.perBankTensorLength
        for bank in 0..<ChessNetwork.historyBankCount {
            let sourcePly = currentPly - bank
            let dstStart = outputBuffer.advanced(by: bank * perBank)
            if sourcePly < 0 {
                // Pre-game padding — zero fill
                dstStart.initialize(repeating: 0, count: perBank)
                continue
            }
            let srcStart = perPlyEncodings.baseAddress!
                .advanced(by: sourcePly * encodingStride)
            let needsFlip = (bank % 2) == 1
            if needsFlip {
                flipInto(source: srcStart, destination: dstStart)
            } else {
                dstStart.update(from: srcStart, count: perBank)
            }
        }
    }

    /// Apply the AZ-style perspective flip to a 22-plane encoding:
    ///   - planes 0..5 ↔ planes 6..11 (my pieces ↔ their pieces)
    ///   - within those, also vertical row flip (rows 0..7 reversed)
    ///   - planes 12,13 ↔ planes 14,15 (my castling ↔ their castling, no spatial flip — broadcast)
    ///   - plane 16 (EP) vertical row flip in place
    ///   - planes 17..21 unchanged (color-agnostic: halfmove, rep, lookahead, masks)
    private static func flipInto(
        source: UnsafePointer<Float>,
        destination: UnsafeMutablePointer<Float>
    ) {
        // Piece planes 0..11 — swap pairs + vertical flip
        for outPlane in 0..<12 {
            let srcPlane = outPlane < 6 ? outPlane + 6 : outPlane - 6
            for outRow in 0..<8 {
                let srcRow = 7 - outRow
                let srcRowBase = source.advanced(by: srcPlane * 64 + srcRow * 8)
                let dstRowBase = destination.advanced(by: outPlane * 64 + outRow * 8)
                dstRowBase.update(from: srcRowBase, count: 8)
            }
        }
        // Castling planes 12..15 — swap pairs, no spatial flip (broadcast)
        let castlingPairs: [(out: Int, src: Int)] = [
            (12, 14), (13, 15), (14, 12), (15, 13),
        ]
        for (outPlane, srcPlane) in castlingPairs {
            destination.advanced(by: outPlane * 64)
                .update(from: source.advanced(by: srcPlane * 64), count: 64)
        }
        // Plane 16 (EP) — vertical flip in place
        for outRow in 0..<8 {
            let srcRow = 7 - outRow
            destination.advanced(by: 16 * 64 + outRow * 8)
                .update(from: source.advanced(by: 16 * 64 + srcRow * 8), count: 8)
        }
        // Planes 17..21 — verbatim copy (color-agnostic)
        destination.advanced(by: 17 * 64)
            .update(from: source.advanced(by: 17 * 64), count: 5 * 64)
    }
}
```

#### 4. `Training/ActiveGame.swift`

Per-game scratch already stores per-ply encodings via `whiteBoardScratch` / `blackBoardScratch` (lines 110-111). These are already `perBankTensorLength` per ply. **No storage change here.**

Add a new method to gather the network input from the current ply's history:

```swift
/// Gather a network input tensor (14,080 floats) for the current ply,
/// pulling history from the per-side scratch. Used by the self-play
/// evaluate path.
///
/// The two per-side scratches store interleaved plies (white plies 0,2,4...
/// and black plies 1,3,5...). For the gather we need a unified per-ply
/// view ordered by absolute ply index — synthesize via a helper that
/// interleaves into a single 1408-float buffer per ply.
func gatherInputForCurrentPly(
    side: PieceColor,
    currentAbsolutePly: Int,
    into outputBuffer: UnsafeMutablePointer<Float>
) {
    // Build a temporary contiguous-ply buffer from interleaved scratches.
    // Or maintain a separate `combinedScratch` that's written to on every
    // recordPly call (cheaper but adds a copy per ply).
    // Recommended: add `combinedScratch` to ActiveGame.
    // ...
    HistoryBankGather.gather(
        perPlyEncodings: combinedScratchBufferView,
        encodingStride: BoardEncoder.perBankTensorLength,
        currentPly: currentAbsolutePly,
        outputBuffer: outputBuffer
    )
}
```

To avoid recombining on every gather, allocate a new `combinedBoardScratch` field sized for `(maxPliesPerGame + 1) * perBankTensorLength` floats. Update on every `recordPly` to mirror the encoding into the absolute-ply slot. Adds one extra memcpy per ply (1408 floats × 4 bytes = 5.6 KB) — negligible.

#### 5. `Training/ReplayBuffer.swift`

- `floatsPerBoard` becomes `BoardEncoder.perBankTensorLength` (= 1408). **Replay buffer still stores PER-PLY encodings, not the gathered tensor.**
- File format bumps: magic stays `DCMRPBUF`, version goes from 7 to 8. The header `floatsPerBoard` field becomes 1408. v7 files won't load (mismatch detected at parse time, surface as `PersistenceError.formatMismatch`).
- **Game-contiguous storage requirement**: when a game flushes, its positions must occupy a contiguous range in the ring (no wrap-around inside a single game). Implementation:

```swift
// In ReplayBuffer.append(...):
// Before writing positions for this game, check if there's room
// until end-of-ring. If not, advance writeIndex to 0 first (sacrificing
// the tail of the ring for this rotation).
private func reserveContiguousRange(count: Int) -> Int {
    let remainingUntilEnd = capacity - writeIndex
    if remainingUntilEnd < count {
        writeIndex = 0   // skip the tail
    }
    let startOffset = writeIndex
    writeIndex = (writeIndex + count) % capacity
    return startOffset
}
```

Note: this slightly reduces effective buffer capacity (some slots get skipped per rotation) but is necessary for the sample-time gather to work without crossing game boundaries.

- Add a sample-time gather method:

```swift
/// Sample N positions, gathering each as a full history-bank input tensor.
/// Replaces the existing sample() in the trainer's hot path.
func sampleWithHistoryGather(
    count: Int,
    destinationBoards: UnsafeMutablePointer<Float>,
    destinationMoves: UnsafeMutablePointer<Int32>,
    destinationOutcomes: UnsafeMutablePointer<Float>,
    // ...
) {
    for i in 0..<count {
        let sampledOffset = randomSampleOffset()
        let sampledPly = Int(plyIndexStorage[sampledOffset])
        let sampledWorkerGameId = workerGameIdStorage[sampledOffset]

        // Walk back up to historyBankCount-1 positions, stopping at
        // game boundary (workerGameId mismatch).
        let inputDst = destinationBoards.advanced(by: i * BoardEncoder.networkInputTensorLength)
        for bank in 0..<ChessNetwork.historyBankCount {
            let bankDst = inputDst.advanced(by: bank * BoardEncoder.perBankTensorLength)
            let sourceOffset = sampledOffset - bank
            if sourceOffset < 0 || workerGameIdStorage[sourceOffset] != sampledWorkerGameId {
                // Walked off game start — zero pad
                bankDst.initialize(repeating: 0, count: BoardEncoder.perBankTensorLength)
                continue
            }
            let srcEncoding = boardStorage.advanced(by: sourceOffset * BoardEncoder.perBankTensorLength)
            let needsFlip = (bank % 2) == 1
            if needsFlip {
                HistoryBankGather.flipInto(source: srcEncoding, destination: bankDst)
            } else {
                bankDst.update(from: srcEncoding, count: BoardEncoder.perBankTensorLength)
            }
        }
        // ... metadata copies (move, outcome, etc.) same as before
    }
}
```

(`flipInto` needs to be made `internal` or moved to a shared location since both `HistoryBankGather` and `ReplayBuffer` use it.)

#### 6. `Training/ChessTrainer.swift`

Staging buffer for input grows from `batchSize * perBankTensorLength * 4` bytes to `batchSize * networkInputTensorLength * 4` bytes (10x larger). At batch 256: from ~1.4 MB to ~14.4 MB per staging buffer. Acceptable. Update every site that computes `floatsPerBoard` in this file (lines 1628, 3332, 3535, 3613, 3904, 3950, 4085, 4130) to use `BoardEncoder.networkInputTensorLength`.

#### 7. `Network/MPSChessPlayer.swift`

In the self-play evaluate path (line 331 area), the `encoded` buffer needs to become the full 14,080-float gathered tensor:

```swift
// before:
var encoded = [Float](repeating: 0, count: Self.boardFloats)
encoded.withUnsafeMutableBufferPointer { BoardEncoder.encode(gameState, into: $0) }
// pass encoded to network.evaluate

// after:
var input = [Float](repeating: 0, count: BoardEncoder.networkInputTensorLength)
input.withUnsafeMutableBufferPointer { buf in
    activeGame.gatherInputForCurrentPly(
        side: side,
        currentAbsolutePly: currentPly,
        into: buf.baseAddress!
    )
}
// pass input to network.evaluate
```

Wait — `MPSChessPlayer` doesn't currently know about `ActiveGame`; the encoding flow is decoupled. Need to reroute either by:
(a) passing the `ActiveGame` reference into `MPSChessPlayer` so it can call gather, OR
(b) having the driver (`BatchedSelfPlayDriver`) construct the gathered input and pass it in.

Option (b) is cleaner — the driver already owns both the player and the ActiveGame. Update `BatchedSelfPlayDriver` to compute the gathered input via ActiveGame and hand it to the player.

#### 8. `Network/ChessMPSNetwork.swift`

`warmupBatch()` (line 115) currently produces single-position encodings concatenated. Needs to produce gathered N-bank tensors with starting-position-only history (banks 1..N-1 are all the starting position encoding, since the warmup walks a random game from `.starting`). Or simpler: have the warmup walk a few full games and use HistoryBankGather to construct each warmup sample's input. Either works.

#### 9. `Network/BatchedMoveEvaluationSource.swift`

Staging buffer for batched eval grows from `slots * perBankTensorLength * 4` to `slots * networkInputTensorLength * 4`. At 8 slots: from ~45 KB to ~450 KB. Negligible.

#### 10. Tests

- Update all `1408` / `perBankTensorLength` references to either stay (per-bank tests) or change to `networkInputTensorLength = 14,080` (full-input tests).
- Add a new test `HistoryBankGatherTests.swift`:

```swift
final class HistoryBankGatherTests: XCTestCase {

    func test_gather_zeroPadsPreGameBanks() { ... }

    func test_gather_flipsOddBanks_keepsEvenBanksAsIs() { ... }

    func test_gather_constructedKnightShuffle_banks0to4HaveExpectedPieces() { ... }

    func test_flip_isInvolutory() {
        // flip(flip(x)) == x
    }

    func test_flip_preservesColorAgnosticPlanes() {
        // Planes 17..21 unchanged by flip
    }

    func test_gather_gameBoundaryDetection() {
        // Build two synthetic games back-to-back, sample position at
        // start of game B, verify banks 1..N read into game B's plies
        // and not into game A's tail.
    }
}
```

- Update `RepPlaneProbeTests.test_constructedKnightShuffleEndgame_probe` to construct the gathered tensor (10 banks) and probe with that, not a single-position encoding. Should still show the same correctness signal (network responds to rep planes) but now with the full input shape.

#### 11. `App/UpperContentView/` — any UI surfaces that visualize planes

`TensorChannelNames` is consumed by board-visualization views. If any UI displays "all input planes" it may need updating to handle the 10-bank stack. Likely the existing UI only shows the current bank — verify no surface tries to render all 220 planes.

### Stage B verification

1. Build green, all tests pass.
2. `HistoryBankGatherTests` all pass — flip correctness, zero-padding, game-boundary handling.
3. Manual smoke: launch app, Build Network (note `[APP]` log: `inputPlanes=220`), Play-and-Train for 10 min. `[STATS]` should emit normally. Watch for any encoding-size mismatch errors.
4. Confirm flip behavior empirically: dump the input tensor for a black-to-move position during self-play. Bank 0 should have black pieces in planes 0-5 (oriented "down"), bank 1 should have white pieces in planes 0-5 (oriented "down" — flipped from its source). Visually verify with a one-shot debug log.
5. Replay-buffer round-trip: save session, reload, train another batch. Verify `floatsPerBoard=1408` in the header.
6. Throughput check: 30-min Play-and-Train, observe steps/sec. Expect ~10-20% drop vs Stage A. If >50% drop, profile the gather path (`HistoryBankGather` and `ReplayBuffer.sampleWithHistoryGather`).
7. End-to-end correctness via the probe test (rerun `test_constructedKnightShuffleEndgame_probe`): the constructed knight shuffle endgame should show the rep planes correctly set in bank 0 AND prior positions correctly stacked as banks 1-4 with appropriate flipping. The network's output should be sensible (legal-mass high, value distribution centered on draw given the endgame material).

---

## Migration / breaking changes

- **Both stages invalidate existing `.dcmmodel` and `.dcmsession` files.** `archHash` changes when `inputPlanes` changes. The user's existing 495k-step KbHZ training is unrecoverable — start fresh from random init.
- **Replay buffer file format bumps to v8** in Stage B. v7 files won't load.
- Stages A and B should be **separate commits** even though they ship together architecturally. Easier to bisect if something breaks.
- **CHANGELOG.md entries** for both stages: A) "Drop dead rep planes (20, 21, 22, 24, 25, 26, 28, 29) and replace plane 19 with lookahead-3fold-available. inputPlanes 30 → 22. Forces fresh training start." B) "Add 10-bank history input (AZ-style). inputPlanes 22 → 220, replay buffer format v7 → v8. Forces fresh training start."
- **ROADMAP.md** should mark Phase 1 and Phase 3 as complete after this lands, preserving the original design rationale.

---

## Out of scope (intentional)

- **Phase 2 (early-draw adjudication shadow logging)** — independent work, separate plan.
- **Halfmove threshold planes (hm≥50, hm≥75)** — probe results show current continuous encoding works correctly; not needed.
- **Lookahead-2fold-available plane** — redundant with rep18 + the network's existing legal-move enumeration capability.
- **AZ-style hybrid layout (global castling/halfmove planes, smaller per-bank planes)** — considered and rejected in favor of uniform 22-plane banks for memcpy simplicity.
- **Switching the value head, policy head, or trunk** — completely unrelated.

---

## File modification summary

### Stage A

| File | Action |
|---|---|
| `Network/ChessNetwork.swift:124` | inputPlanes 30 → 22 |
| `Encoding/BoardEncoder.swift` | Add lookaheadThreefoldAvailable to GameState, rewrite plane assignment for indices 18-21 |
| `Chess/ChessGameEngine.swift:155-225` | Compute lookaheadThreefoldAvailable in applyMoveAndAdvance |
| `Views/Board/TensorChannelNames.swift` | 30-element arrays → 22-element arrays with new names |
| `DrewsChessMachineTests/BoardEncoderTests.swift` | Update tensorLength assertion |
| `DrewsChessMachineTests/RepetitionPlaneTests.swift` | Update plane index references, add lookahead test |
| `DrewsChessMachineTests/PolicyHeadCorrectnessTests.swift` | Update tensorLength assertion |
| `DrewsChessMachineTests/RepPlaneProbeTests.swift` | Update perBoard assertion, update probe patterns |
| `tools/stem_plane_norms.py` | Update INPUT_PLANES constant, update plane labels list |
| `CLAUDE.md` | Update board-encoding section |

### Stage B (in addition to Stage A)

| File | Action |
|---|---|
| `Network/ChessNetwork.swift` | Add planesPerBank, historyBankCount constants; redefine inputPlanes as planesPerBank × historyBankCount = 220 |
| `Encoding/BoardEncoder.swift` | Add perBankTensorLength and networkInputTensorLength constants; deprecate tensorLength |
| `Encoding/HistoryBankGather.swift` | NEW FILE — gather + flip logic |
| `Training/ActiveGame.swift` | Add combinedBoardScratch + gatherInputForCurrentPly method |
| `Training/ReplayBuffer.swift` | Bump file format v7→v8; reserve contiguous game ranges; add sampleWithHistoryGather method |
| `Training/ChessTrainer.swift` | Migrate input-size constants throughout |
| `Network/MPSChessPlayer.swift` | Migrate from per-bank encoding to gathered input (via driver) |
| `Network/BatchedSelfPlayDriver.swift` | Wire up gather call between ActiveGame and player |
| `Network/ChessMPSNetwork.swift` | Update warmupBatch to produce gathered tensors |
| `Network/BatchedMoveEvaluationSource.swift` | Update staging buffer size |
| `DrewsChessMachineTests/HistoryBankGatherTests.swift` | NEW FILE — gather/flip/padding tests |
| All `*Tests.swift` with size assertions | Migrate to perBankTensorLength or networkInputTensorLength as appropriate |
| `CHANGELOG.md` | Two entries (Stage A and Stage B) |
| `ROADMAP.md` | Mark Phase 1 and Phase 3 complete |
