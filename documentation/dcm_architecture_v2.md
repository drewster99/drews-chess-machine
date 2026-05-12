# Implementation Plan: Bundled Architecture Refresh
*(SE blocks + AZ-style 76-channel policy head + threefold-repetition input planes)*

This is a single coordinated change requiring `currentArchHash` bump, ReplayBuffer format bump (v2 → v3), and full retrain. All existing checkpoints will fail to load with `.archMismatch` (per CLAUDE.md "no migration without request"). User has confirmed willingness to discard existing models and replay buffers.

---

## Pre-flight

**Branch:** New branch `arch-refresh-76ch-se-rep` off `main`. All work isolated until end-to-end smoke test passes.

**MPSGraph capability check (do BEFORE any other work — blocker if missing):**
- Verify `graph.sigmoid(with:name:)` compiles in the project's MPSGraph deployment target. Should be available since macOS 11.
- Verify `graph.mean(of:axes:name:)` keeps reduced dimensions (i.e., produces `[B, C, 1, 1]` from `[B, C, 8, 8]` with `axes: [2, 3]`, not `[B, C]`). The SE squeeze relies on this; existing batchNorm code does too, so it's almost certainly correct, but a 30-second standalone verification removes ambiguity.
- Verify `graph.multiplication(_:_:name:)` does NumPy-style broadcasting on `[B, C, 1, 1] × [B, C, 8, 8]` → `[B, C, 8, 8]`. The SE channel-scaling step depends on this.

If any of the three fails, the SE block design needs adjustment (specifically, sigmoid via `1/(1+exp(-x))` decomposition, or explicit broadcast via `tile` op). Don't commit to the SE bundle until all three are confirmed.

**Constants centralization (do FIRST — referenced everywhere downstream):**

In `ChessNetwork.swift`:
```swift
static let inputPlanes = 20            // was 18
static let policyChannels = 76         // 56 queen + 8 knight + 9 underpromo + 3 queen-promo
static let policySize = policyChannels * 64  // = 4864 (was 4096)
static let seReductionRatio = 4
```

In `BoardEncoder.swift`:
```swift
static let tensorLength = 20 * 64      // = 1280 (was 1152)
```

These constants drive arch hash, replay buffer size, training target depth, and policy index decoding. **Bump exactly once, in one place.**

---

## Phase 1 — Engine-side: position-history-based repetition tracking

**IMPLEMENTATION DEVIATION (2026-04-19/20):** the plan as originally written added a Zobrist hashing layer with a new `ZobristTable.swift`, an incremental `zobrist: UInt64` field on `GameState`, and a per-game `positionHistory: [UInt64]` array owned by `ChessMachine`. **This was simplified during implementation** — `ChessGameEngine` already maintained `positionCounts: [PositionKey: Int]` for its existing 3-fold detection logic, with `PositionKey` providing FIDE-correct position equality (board + side + castling + EP). We reused that mechanism instead of introducing a parallel Zobrist layer:

- **No `ZobristTable.swift` created.** No new file.
- **No `zobrist` field on `GameState`.** GameState gained a `repetitionCount: Int` field (default 0) plus a `withRepetitionCount(_:)` convenience method.
- **No new `applyMove` signature change.** `MoveGenerator.applyMove` is unchanged. The new `repetitionCount` defaults to 0 so existing callsites still compile.
- **`ChessGameEngine.applyMoveAndAdvance`** — unchanged in concept, but now after applying the move and incrementing `positionCounts[key]`, it computes `priorOccurrences = min(totalVisits - 1, 2)` and stamps that onto the new state via `state.withRepetitionCount(priorOccurrences)`. The encoder downstream reads `state.repetitionCount`.

Net effect: equivalent correctness, ~150 lines of new code avoided, no GameState memberwise-init ripple. The Zobrist hashing performance argument was real but moot — `PositionKey` is hashed via Swift's auto-synthesized `Hashable` over a 64-element `[Piece?]` array, which costs ~O(64) per move vs Zobrist's O(1) incremental. At our self-play throughput this difference is well under a microsecond per move. Negligible.

**Files actually modified: `BoardEncoder.swift`, `ChessGameEngine.swift`.**

**Important boundary:** the Zobrist hash itself is **never fed to the network**. It is purely a CPU-side bookkeeping device for fast position-equality comparison. The only thing the network ever sees from this whole mechanism is the two binary repetition planes (18 and 19) — derived from the rep count (0/1/2), which is in turn derived from counting hash matches in the per-game `positionHistory` list. The hash function could be swapped for any other position-equality scheme (full `GameState` ==, bitboard tuple compare, FEN string compare) and the network's input would be byte-identical. Zobrist is chosen because it gives O(1) incremental updates and O(1) equality tests — performance, not learning signal. Feeding the raw 64-bit hash to a CNN would be actively harmful (uniform-random bits with no spatial structure).

### 1.1 `ZobristTable.swift` (new file)

Static immutable tables of `UInt64` random values for:
- 6 piece types × 2 colors × 64 squares = 768 piece-square hashes.
- 1 side-to-move hash (XOR'd when black to move).
- 4 castling-rights hashes (one per right; XOR if right held).
- 8 en-passant file hashes (one per file; XOR if EP target exists on that file AND a pawn could legally capture — or just unconditionally; FIDE technicality, but the simpler rule is fine for our purposes).

Generate with a deterministic seeded `SplitMix64` so the table is reproducible across runs (and across machines, for cross-checking). Hard-code seed = `0xDCMZobrist2026` or similar.

### 1.2 `Zobrist.computeHash(for state: GameState) -> UInt64`

Pure function. Walks the board, XORs in piece-square hashes, side-to-move, castling, EP. Slow (O(64)) but only called once per game (from the starting position). All subsequent positions are computed *incrementally* in `applyMove`.

### 1.3 Incremental update in `MoveGenerator.applyMove`

`applyMove` becomes `(GameState, ChessMove, currentZobrist: UInt64) -> (GameState, newZobrist: UInt64)`. Or, cleaner: add `zobrist: UInt64` to `GameState` itself, computed on every `applyMove`.

I'd lean toward **storing `zobrist` on `GameState`** for the same reason `halfmoveClock` is stored there — it's a per-state property that the rest of the engine asks about. This makes `applyMove` self-contained.

For each move, XOR out the changing pieces and XOR in the new ones:
- XOR out the moving piece on its from-square.
- XOR in the moving piece (or promoted piece) on its to-square.
- If capture: XOR out the captured piece on its to-square (or EP square for en passant).
- If castling: XOR out the rook on its old square, XOR in the rook on its new square.
- XOR in/out castling rights deltas.
- XOR in/out EP file deltas.
- Always XOR side-to-move.

Cost: ~6 XORs per typical move, O(1).

### 1.4 Position history per game

`ChessMachine` already owns per-game state. Add:
```swift
private var positionHistory: [UInt64] = []  // Zobrist hashes since last halfmove-clock reset
```

Updated at the end of every move:
- If new state's `halfmoveClock == 0`: clear `positionHistory`, append `newState.zobrist`.
- Else: append `newState.zobrist`.

Repetition count for the current state:
```swift
func currentRepetitionCount() -> Int {
    let target = currentState.zobrist
    let count = positionHistory.dropLast().reduce(0) { $1 == target ? $0 + 1 : $0 }
    return min(count, 2)  // saturate at 2 — that's all the planes encode
}
```

### 1.4.5 GameState struct expansion ripple

Adding `zobrist: UInt64` to `GameState` changes its memberwise initializer. Every callsite that constructs `GameState(...)` with all fields gets a compile error. Known callsites needing fixes:
- `BoardEncoder.swift:79-89` — `GameState.starting` static init must call `Zobrist.computeHash(.starting)`.
- `MoveGenerator.swift:144-153` — inside `applyMove`; correct path because `applyMove` already computes the new zobrist incrementally.
- `ContentView.swift:2004` and `:4787` — UI editable-position pass-through.
- Any test/probe code constructing `GameState` manually.

**Pattern:** keep the memberwise initializer private. Provide a public initializer that auto-computes `zobrist` from board+player+rights+EP. This makes it impossible to construct a `GameState` with a stale or wrong hash by accident. The only exception is the path inside `applyMove` that incrementally updates from the previous hash — that uses the private memberwise init directly with the pre-computed delta.

**GameState gets ~8 bytes larger.** It's still passed by value through hot paths. The compiler optimizes most copies; absolute cost is well under a microsecond per move. No performance concern.

### 1.5 Validation

**Unit-style probes** (callable from a "Run Engine Diagnostics" UI button — see Phase 9):

- **Zobrist determinism**: compute hash for `GameState.starting` 100 times, verify identical.
- **Zobrist incremental == full**: play a 50-move game with random legal moves; after every move, verify `state.zobrist == Zobrist.computeHash(state)`. Catches XOR bugs in `applyMove`.
- **Repetition detection**: construct three known repetition scenarios (knight shuffle Nf3-Ng1-Nf3-Ng1-Nf3 etc.); verify `currentRepetitionCount()` returns 0, then 1, then 2 at the right moves.
- **Halfmove reset clears history**: shuffle pieces twice, then make a pawn move, then shuffle back to a previously-seen position; verify rep count is 0 (not 1) because the pawn move flushed history.
- **Performance**: time `applyMove` on a 5000-move sequence; should be ≤5% slower than the pre-Zobrist baseline.

---

## Phase 2 — Policy encoding bijection (76 channels)

**File: New `PolicyEncoding.swift`. The hardest-to-get-right part of the whole plan.**

### 2.1 Channel layout (locked-in spec)

**Channels 0–55: queen-style moves (8 directions × 7 distances).**
Direction order (must be stable forever): `N, NE, E, SE, S, SW, W, NW`.
- N = (row -1, col 0), NE = (row -1, col +1), etc. (encoder-frame, where row 0 = top from the current player's POV)
- `channel = direction_index * 7 + (distance - 1)` for `direction_index ∈ 0..<8`, `distance ∈ 1..7`.

**Channels 56–63: knight moves.**
Order: `(-2,+1), (-1,+2), (+1,+2), (+2,+1), (+2,-1), (+1,-2), (-1,-2), (-2,-1)` — clockwise from "up-right-knight-jump."
- `channel = 56 + jump_index`.

**Channels 64–72: underpromotions (3 pieces × 3 directions).**
- Pieces: N=0, R=1, B=2.
- Directions: forward=0, capture-left=1, capture-right=2.
- `channel = 64 + piece_index * 3 + direction_index`.

**Channels 73–75: queen-promotions (3 directions).**
- Same direction encoding as underpromotions.
- `channel = 73 + direction_index`.

**Total: 76 channels × 64 squares = 4864 logits.**

### 2.2 API

```swift
enum PolicyEncoding {
    /// Encode a legal chess move into (channel, fromRow, fromCol).
    /// Asserts on illegal/unrepresentable moves — these should never
    /// occur for moves drawn from MoveGenerator.legalMoves.
    static func encode(_ move: ChessMove, currentPlayer: PieceColor) -> (channel: Int, row: Int, col: Int)

    /// Flat policy index: channel * 64 + row * 8 + col.
    static func policyIndex(_ move: ChessMove, currentPlayer: PieceColor) -> Int

    /// Decode a (channel, fromRow, fromCol) back into the matching ChessMove.
    /// `pieceAt`: closure for piece lookup, needed to disambiguate
    /// queen-style 1-step pawn moves into the last rank (these are
    /// queen-promotions, encoded in channels 73–75 not 0–55).
    /// Returns nil if the (channel, row, col) doesn't correspond to
    /// any legal move from the given board state.
    static func decode(channel: Int, row: Int, col: Int,
                       state: GameState) -> ChessMove?
}
```

The encoder needs `currentPlayer` because move encoding happens in the **encoder frame** (vertically flipped if Black is moving), matching how board planes are flipped. Keeps everything consistent.

### 2.3 Encoding rules (the bijection in detail)

For a `ChessMove`:
1. Compute `fromRow_enc, fromCol_enc, toRow_enc, toCol_enc` in encoder frame (flip rows if `currentPlayer == .black`).
2. Compute `dr = toRow_enc - fromRow_enc`, `dc = toCol_enc - fromCol_enc`.
3. **If promotion is set:**
   - Direction = forward (dc=0), capture-left (dc=-1), or capture-right (dc=+1). The `dr` is +1 or -1 (always 1 row of movement) — but in encoder frame, "forward" for the current player is always the direction toward row 0, so `dr = -1`.
   - If `promotion == .queen`: channel = 73 + direction_index.
   - If `promotion ∈ {.knight, .rook, .bishop}`: channel = 64 + piece_index * 3 + direction_index.
4. **Else if `(dr, dc)` matches a knight pattern:** channel = 56 + jump_index.
5. **Else:** queen-style. Determine direction from sign of (dr, dc), distance = max(|dr|, |dc|). Channel = direction_index * 7 + (distance - 1).

For decoding:
- Given (channel, row, col), reverse the channel → move-type computation.
- For promotion channels, the destination row is always 0 (top rank in encoder frame) and the from-row is always 1 (one rank back).
- For queen-style and knight channels, compute (toRow, toCol) from direction × distance + (row, col) or knight offset + (row, col).
- **Off-board guard:** before any further checks, reject any (channel, row, col) whose computed destination square is outside `[0, 8) × [0, 8)`. Without this, channel 0 (N direction, distance 1) at row=0 would compute `to=(-1, col)` and either crash or return junk. The `nil` return is correct semantics; this guard ensures it happens at the bounds check, not deep in legal-move filtering.
- **Final disambiguation**: if the move appears in `MoveGenerator.legalMoves(for: state)`, return it; else return nil.

### 2.4 Validation (this is the most-likely-to-have-a-bug part)

**Round-trip every legal move from many positions:**

```swift
let positions: [GameState] = [
    .starting,
    // After 1.e4
    // Mid-game position with castling rights, en-passant available
    // Position with pawn on 7th rank for both colors
    // Position with promotion on every file
    // Position with all 4 promotion choices for the same square
    // Knight on edge/corner squares (limited jumps)
    // King in check
    // ... 20+ hand-crafted positions
]

for state in positions {
    for move in MoveGenerator.legalMoves(for: state) {
        let (chan, r, c) = PolicyEncoding.encode(move, currentPlayer: state.currentPlayer)
        guard chan >= 0, chan < 76, r >= 0, r < 8, c >= 0, c < 8 else {
            fatalError("encode produced invalid (chan, r, c)=(\(chan), \(r), \(c)) for \(move.notation)")
        }
        guard let decoded = PolicyEncoding.decode(channel: chan, row: r, col: c, state: state) else {
            fatalError("decode failed for legally-encoded move \(move.notation)")
        }
        precondition(decoded == move, "round-trip mismatch: \(move.notation) → (\(chan),\(r),\(c)) → \(decoded.notation)")
    }
}
```

**No two distinct moves share an index:**
For each test position, verify all `legalMoves` produce unique `policyIndex`. Catches the underpromotion-collapse class of bugs.

**Underpromotion specifically:** in a position with all 4 promotion choices for one square, verify the 4 moves get 4 distinct indices, and that decoding each returns the correct `promotion` piece.

**All four corner knight squares × all 8 jumps:** verify the legal subset is correctly distinguished from the off-board subset.

**Castling:** verify `O-O` and `O-O-O` round-trip (these are queen-style moves with distance=2 in E and W directions for the king).

These all run as a "Run Encoding Diagnostics" probe (Phase 9 UI button), not as XCTest — matches the project's manual-testing convention.

---

## Phase 3 — BoardEncoder: 18 → 20 input planes

**File: `BoardEncoder.swift`.**

### 3.1 Signature change

```swift
static func encode(
    _ state: GameState,
    repetitionCount: Int,        // 0, 1, or 2 (saturated)
    into buffer: UnsafeMutableBufferPointer<Float>
)

static func encode(
    _ state: GameState,
    repetitionCount: Int = 0     // default for tests / non-game usage
) -> [Float]
```

Default of 0 on the allocating variant is a deliberate test-friendliness affordance — at the starting position there are no repetitions; tests that don't care about repetition can omit the parameter without silent wrong-value failures.

### 3.2 New plane writes (always-fill pattern, per our earlier discussion)

After existing plane 17 logic:
```swift
// Plane 18: 1.0 if current position has occurred ≥1 time before in this game.
fillPlane(base, plane: 18, value: repetitionCount >= 1 ? 1.0 : 0.0)

// Plane 19: 1.0 if current position has occurred ≥2 times before
// (i.e., a third visit would force a 3-fold draw claim).
fillPlane(base, plane: 19, value: repetitionCount >= 2 ? 1.0 : 0.0)
```

### 3.3 Validation

- `BoardEncoder.encode(.starting, repetitionCount: 0)` produces 1280-float array. Planes 18 and 19 both all-zero. (Not implicitly via initial clear — explicitly via fillPlane.)
- `BoardEncoder.encode(.starting, repetitionCount: 1)` produces plane 18 = all 1.0, plane 19 = all 0.0.
- `BoardEncoder.encode(.starting, repetitionCount: 2)` produces both planes = all 1.0.
- `BoardEncoder.encode(.starting, repetitionCount: 5)` (caller passed unsaturated) produces same as count=2. Caller's responsibility to saturate, but encoder shouldn't crash.

---

## Phase 4 — Network architecture changes

**File: `ChessNetwork.swift`.**

### 4.1 Stem: 18 → 20 input channels

Single change in the conv1 weight allocation:
```swift
let convStemW = graph.variable(
    with: heInitDataConvOIHW(shape: [128, Self.inputPlanes, 3, 3]),  // was hardcoded 18
    shape: [128, Self.inputPlanes, 3, 3],
    ...
)
```

He init's `fan_in` becomes `inputPlanes * 9 = 180` (was 162). Auto-handled by `heInitDataConvOIHW` since it derives fan_in from shape.

### 4.2 SE block insertion

In `residualBlock`, insert the ~30-line SE module between `bn2` and the skip-add. Code already drafted in the previous turn — drop it in verbatim.

Per-block additions to `trainables` / `shouldDecay`:
- seFC1W (decay=true)
- seFC1Bias (decay=false)
- seFC2W (decay=true)
- seFC2Bias (decay=false)

8 blocks × 4 new variables = 32 new variables in the trainable list.

### 4.3 Policy head replacement

Delete the existing 1×1 conv → BN → ReLU → flatten → FC → bias chain. Replace with:

```swift
private static func policyHead(
    graph: MPSGraph,
    input: MPSGraphTensor,
    descriptor: MPSGraphConvolution2DOpDescriptor,
    bnMode: BNMode,           // unused but kept for signature consistency
    trainables: inout [MPSGraphTensor],
    shouldDecay: inout [Bool],
    runningStats: inout [MPSGraphTensor],
    runningStatsAssignOps: inout [MPSGraphOperation]
) -> MPSGraphTensor {
    // Single 1×1 conv: 128 → 76 channels. No BN, no activation.
    // Output is raw logits; CPU does softmax-over-legal as today.
    let convW = graph.variable(
        with: heInitDataConvOIHW(shape: [Self.policyChannels, Self.channels, 1, 1]),
        shape: [Self.policyChannels, Self.channels, 1, 1],
        dataType: Self.dataType,
        name: "policy_conv_weights"
    )
    let convBias = graph.variable(
        with: zerosData(count: Self.policyChannels),
        shape: [1, Self.policyChannels, 1, 1],
        dataType: Self.dataType,
        name: "policy_conv_bias"
    )
    trainables.append(convW);    shouldDecay.append(true)
    trainables.append(convBias); shouldDecay.append(false)
    var x = graph.convolution2D(input, weights: convW, descriptor: descriptor, name: "policy_conv")
    x = graph.addition(x, convBias, name: "policy_conv_bias_add")

    // Reshape [B, 76, 8, 8] → [B, 4864] for downstream consumption.
    return graph.reshape(x, shape: [-1, Self.policySize], name: "policy_flatten")
}
```

`bnMode` parameter retained for signature consistency with `valueHead`; document in comment that the new policy head has no BN.

### 4.4 Update doc comment at top of ChessNetwork

The architecture summary on line 59-66 must be updated to reflect:
- Input: 20×8×8 (was 18×8×8)
- Each residual block now includes SE module
- Policy head: `1×1 conv 128 → 76` (no FC, no bottleneck)
- Total parameters: ~2.47 M (down from 2.92 M)

### 4.5 Validation

- **Build succeeds**: zero errors, zero warnings.
- **Forward pass shape probe**: `network.evaluate(board: BoardEncoder.encode(.starting, repetitionCount: 0))` returns policy of length 4864 and value as a 1-element scalar in [-1, +1].
- **Channel 0 = uniform across squares at init for a constant-input batch**: as a sanity check that the conv weights are wired correctly (not flattened wrong).
- **Trainable variable count**: log it AND `precondition` against the exact expected number at graph-construction time. Computed:
  - Stem: 1 conv + 2 (BN gamma+beta) = **3**
  - Per residual block: 2 convs + 4 (2× BN gamma+beta) + 4 (2× SE FC weight+bias) = **10**
  - 8 residual blocks: **80**
  - Policy head: 1 conv + 1 bias = **2**
  - Value head: 1 conv + 2 (BN) + 4 (2× FC weight+bias) = **7**
  - **Total: 92 trainable variables.** Off-by-one in SE plumbing fails loudly at construction.
- BN running stats (mean+var per BN layer) are NOT in trainables: 1 (stem) + 16 (per-block × 2) + 1 (value head) = 18 BN layers × 2 stats = **36 running-stat variables**. Verify this count too.
- **He init magnitudes**: log mean and std of one weight from each layer category. Should match `sqrt(2/fan_in)` ± noise.

---

## Phase 5 — Trainer changes

**File: `ChessTrainer.swift`.**

### 5.1 One-hot policy target depth

The hardcoded `oneHot(policyIndex, depth: 4096)` becomes `oneHot(policyIndex, depth: ChessNetwork.policySize)`. The `policyIndex` comes from `PolicyEncoding.policyIndex(move, currentPlayer:)` instead of `ChessMove.policyIndex`.

**Important: `ChessMove.policyIndex` is now wrong everywhere.** Decision: **delete it entirely.** Force every caller to use `PolicyEncoding.policyIndex(move:currentPlayer:)` so they have to think about which encoding (and pass `currentPlayer`). A grep + delete + fix-compile-errors cycle catches every site.

### 5.2 Entropy alarm threshold

`policyEntropyAlarmThreshold` in `ContentView.swift` is currently 7.0, against `log(4096) ≈ 8.32` uniform-init entropy. New uniform entropy is `log(4864) ≈ 8.49`. The threshold should remain a "noticeably below uniform" value — bump from 7.0 to 7.2 to keep the same 1.3-nat margin below uniform.

Update the corresponding `[STATS]` log comment "log(4096)" → "log(4864)" or, better, log it as `log(ChessNetwork.policySize)` so it stays in sync.

### 5.2.5 Hidden decoder caller — `ChessRunner.extractTopMoves`

`ChessRunner.swift:99-114` decodes policy indices for the Forward Pass demo and Top Moves UI displays. Currently uses the `from*64+to` scheme + un-flips rows for display. After the change, this function must be **rewritten** to use `PolicyEncoding.decode(channel:row:col:state:)`. The row-flip un-doing logic is now embedded inside `PolicyEncoding.decode` (which decodes back into absolute board coordinates for legal-move matching), so the `extractTopMoves` row-flipping logic gets *removed*, not preserved.

This is in addition to deleting `ChessMove.policyIndex` — the deletion gives a compile error here, but the rewrite is more involved than just a function-name swap.

### 5.2.6 Hardcoded `4096`s confirmed by grep — all must be fixed

Grep results confirm these runtime-critical hardcodings (in addition to the policy-head FC weights/biases that get deleted in Phase 4):
- **`BatchedMoveEvaluationSource.swift:50`** — `private static let policySize = 4096`. **CRITICAL:** the BATCHER startup probe and per-slot slicing arithmetic both reference this. Replace with `ChessNetwork.policySize` (delete the local constant entirely).
- **`ChessTrainer.swift:734`** — `depth: 4096` in the one-hot policy target. Replace with `ChessNetwork.policySize`.
- **`ChessTrainer.swift:1100`** — `// Random move indices in [0, 4096)`. Verify whether this is a runtime path or benchmark-only; either way, update to use `ChessNetwork.policySize`.
- **`TrainingChartGridView.swift:807-860`** — Y-axis `chartYScale(domain: 0...4096)` + `"%d / 4096"` format strings. UI bug if not updated (chart axis clips to 4096 against new max ~4864). Replace literal with `ChessNetwork.policySize`.

### 5.2.7 Hidden callers swept by deleting `ChessMove.policyIndex`

Compile errors surface every site. Confirmed by grep:
- **`GameDiversityTracker.swift:81`** — `Int16(clamping: $0.policyIndex)` for compact ply representation. Mechanical fix: use `PolicyEncoding.policyIndex(move:currentPlayer:)`. The `Int16(clamping:)` was already a no-op (4096 < `Int16.max=32767`) and remains so for 4864. **Validation:** two identical games still classified as identical (encoding is deterministic per (move, player) pair).

Plus hidden non-policyIndex callers from the same code-sweep:
- **`TournamentDriver`** per-worker UI displays — anything showing top-K moves needs the new decoder.
- **`ChessRunner.extractTopMoves`** (Phase 5.2.5) — full rewrite to use `PolicyEncoding.decode`.
- Forward Pass demo (Phase 9.4) and Tensor Carousel (Phase 9.1).

### 5.2.8 Doc-string and comment cleanup

Grep also found these doc-string and comment references to "4096" that have no runtime impact but make the code self-misleading post-change. Sweep and update as part of this bundle:
- `MoveEvaluationSource.swift:28, 47`
- `ChessNetwork.swift:63, 168, 534, 536, 1032, 1034, 1036, 1192`
- `ChessMPSNetwork.swift:104`
- `ChessRunner.swift:19`
- `ContentView.swift:1412` (comment), `:2067` and `:4053` (UI text saying "policy(4096)" — must update display strings to "policy(4864)"), `:6402`
- `MPSChessPlayer.swift:359, 463`

False positives (NOT to change):
- `ContentView.swift:1408` — `trainingBatchSize = 4096` (batch size, not policy size).
- `ContentView.swift:1638` — `sweepSizes: [..., 4096, ...]` (benchmark size enumeration).

### 5.3 Validation

- Take one batch from a freshly-initialized replay buffer, run `ChessTrainer.step()` once. Verify gradients are non-zero, weights change. Verify no NaN/Inf in any reported metric.
- Verify that two consecutive training steps on the same batch produce decreasing loss (basic gradient-descent sanity).

---

## Phase 6 — ReplayBuffer format v3

**File: `ReplayBuffer.swift`.**

### 6.1 Format changes

- `floatsPerBoard` becomes `BoardEncoder.tensorLength = 1280` (was 1152). This widens the on-disk and in-memory board storage by ~11%.
- New per-slot field: `repetitionCount: Int8` (saturates at 2). One extra byte per position.
- Header: bump `fileVersion` to 3. v3 has the same 7 header fields as v2 but board stride is 1280 floats instead of 1152.
- **No backward compat.** Reader rejects v1 and v2 files cleanly with `unsupportedVersion`. Per CLAUDE.md, no migration code.

### 6.2 New storage allocation

Add a `vRepCountStorage: UnsafeMutablePointer<Int8>` parallel to the existing `vBaselineStorage`. Allocated at init, freed in deinit.

### 6.3 Append signature change

```swift
func appendGame(positions: [(board: [Float], move: ChessMove, vBaseline: Float, repCount: Int8)],
                outcome: Float)
```

Caller (`BatchedSelfPlayDriver` / `MPSChessPlayer`) provides the rep count alongside each position. Computed from the live game's `positionHistory` at the time of the move.

### 6.4 Sample signature change

`sample(count:)` returns boards (1280 floats each), moves, outcomes, vBaselines, **and repCounts**. Trainer uses repCounts only to *re-encode* boards for inputs (since the rep planes need to be regenerated from the saturated count — they're not stored as float planes in the replay buffer; the count is stored once and the planes are reconstructed on read). This avoids storing 128 redundant float bytes per slot when 1 byte suffices.

Wait — actually: simpler is to store the full 1280-float board including the rep planes. That makes append/sample symmetric: input goes in as full `[Float]`, comes out as full `[Float]`, no reconstruction at sample time. The 11% storage growth was already accepted. **Choose this path** — it's simpler. The rep count exists separately only if we ever want to inspect it; for the trainer's data flow, it's part of the board.

Update §6.3:
```swift
func appendGame(positions: [(board: [Float], move: ChessMove, vBaseline: Float)],
                outcome: Float)
```
The board is already encoded with rep planes by the caller. RepCount is implicit in the board planes 18/19. No separate field needed.

Drop the parallel storage from §6.2.

### 6.5 Format details (locked)

Header (v3): same 7 Int64 fields as v2, but `floatsPerBoard = 1280`. The reader must verify this matches the runtime `BoardEncoder.tensorLength` and reject otherwise.

### 6.6 Atomic-write redesign (deferred — see TODO #3)

The atomic-write + sidecar-hash redesign we discussed earlier is **separate work**. The v3 format change in this plan retains the existing write pattern (delete-and-write-in-place). Don't bundle that fix here — keeps the diff focused. File a follow-up; we already have one in TODO_NEXT.md.

**Explicit warning:** this architecture refresh does NOT improve replay-buffer durability. A crash during `ReplayBuffer.write(to:)` (called from the session-save path) will still produce a torn `.replay` file inside the otherwise-atomic `.dcmsession` bundle. The Phase 10 step 8 smoke test confirms a *successful* save round-trips correctly — it does not stress the partial-write failure mode. If we want to fix durability, do it as a focused follow-up after this bundle ships.

### 6.7 Side effects to flag

- **Storage growth:** 1 M positions × 11% bigger boards ≈ +512 MB resident memory. On a 16 GB Mac with concurrent workers + trainer + arena, ~3% of total memory. Fine but worth knowing.
- **`ReplayRatioController` transient:** the controller auto-tunes `stepDelay` against the `cons/prod` ratio. Both numerator and denominator scale identically, so steady-state should be unaffected, but expect 30 seconds of unstable `delay=` values at session start as the controller re-converges with the new compute profile.

### 6.7 Validation

- Save a small replay buffer (10 positions). Re-load. Verify byte-identical board floats and identical moves/outcomes/vBaselines.
- Save v3, attempt to load with a v2-expecting reader (artificially): should throw `unsupportedVersion`. Confirms version gate.
- Try to load an actual v2 file (from the previously-deleted `~/Library/Application Support/...`): should throw `unsupportedVersion`. Won't be testable since the user already deleted them, but verify the version-rejection logic exists.

---

## Phase 7 — Checkpoint serialization

**Files: `ModelCheckpointFile.swift`, `SessionCheckpointFile.swift`.**

### 7.1 Arch hash bump (automatic)

`currentArchHash` currently mixes `(channels, inputPlanes, numBlocks, policySize, dataType)`. After Phase 4 changes:
- `inputPlanes`: 18 → 20
- `policySize`: 4096 → 4864
- New variable count due to SE: arch hash should mix that too if it doesn't already.

**Verify** that `currentArchHash` actually changes by running `print(ModelCheckpointFile.currentArchHash)` before and after; if it doesn't change, the hash mixer is missing one of the inputs (likely SE-related, since SE adds variables but doesn't change any of the existing scalar constants).

If the mixer doesn't catch SE: extend it to include `seReductionRatio` and a "has SE blocks" boolean (or just bump a manual schema version constant alongside the architecture-derived hash). Either way, old `.dcmmodel` files must reject with `.archMismatch`.

### 7.2 maxTensorElementCount

Per existing TODO_NEXT.md "#5 follow-up": make it computed from live constants. Cover the new largest-tensor case:
- Stem conv: `inputPlanes × channels × 9 = 20 × 128 × 9 = 23,040`
- Residual conv: `channels × channels × 9 = 147,456`
- Policy conv: `channels × policyChannels = 9,728`
- SE FC1: `channels × (channels/r) = 4,096`
- SE FC2: `(channels/r) × channels = 4,096`

Largest is still residual conv at 147,456. Keep `max(...) + 65_536` slack. The current 600,000 is also fine but stale; bundling the computed-property fix makes sense since we're already touching this area.

### 7.3 Bit-exact round-trip verification

`CheckpointManager.saveSession` already runs a bit-exact forward-pass verification on every save (per CLAUDE.md). This will catch any serialization bug in the new SE variables automatically — if the loaded weights produce different policy/value output than the live network, the verification fails and the save is aborted. **No new test needed**; just confirm the existing verification still runs and passes after Phase 4 changes.

### 7.4 Validation

- Save a fresh-init network. Reload. Run `network.evaluate` on `.starting` before save and `loadedNet.evaluate` on `.starting` after reload. Bit-exact match required (existing verification handles this).
- Verify `currentArchHash` is different from the pre-change hash (confirm with a printed banner during build / startup).

---

## Phase 8 — Inference path

**File: `MPSChessPlayer.swift`, possibly `BatchedMoveEvaluationSource.swift`.**

### 8.1 Move scoring

`chooseMove` currently does:
```swift
for legal in legalMoves {
    let logit = policyOutput[legal.policyIndex]
    ...
}
```

Becomes:
```swift
for legal in legalMoves {
    let idx = PolicyEncoding.policyIndex(legal, currentPlayer: state.currentPlayer)
    let logit = policyOutput[idx]
    ...
}
```

Per-move encoding is O(1). The legal-move list is typically 30–40 moves; the per-ply encoding cost is negligible.

### 8.2 Repetition count plumbing

`MPSChessPlayer` (or whoever calls `BoardEncoder.encode`) must pass the current rep count. The count comes from `ChessMachine`'s `positionHistory`. Pass it down through:
- `MPSChessPlayer.evaluatePosition` adds `repetitionCount: Int` parameter.
- `BatchedMoveEvaluationSource.evaluate` adds `repetitionCount: Int` parameter (per-slot).
- The batched eval source carries one rep count per slot, encodes that into the per-slot board planes.

### 8.3 Validation

- Run a "Forward Pass" demo: evaluate `.starting`. Verify the legal moves' logits are reasonable (not all identical).
- Shuffle pieces back-and-forth twice (creating a 2-rep position). Verify the encoded board has plane 18 all 1.0 and plane 19 all 1.0. Verify the policy output for that position is *different* from the same board state with rep count 0 (confirms the network actually receives the rep-plane signal — if outputs are identical, the planes aren't being read).

---

## Phase 9 — UI surfacing

**File: `ContentView.swift`, `TensorCarouselView.swift`, plus a new `EngineDiagnosticsView.swift`.**

### 9.1 Tensor Carousel View

`TensorCarouselView` shows the input tensor planes. After Phase 3:
- Total planes: 20 (was 18).
- Need labels for the new planes: "Repetition ≥1×", "Repetition ≥2×".
- The carousel must handle the longer plane list. If it hardcodes 18, change to `BoardEncoder.tensorLength / 64`.

Also: while we're touching plane labels, double-check existing plane labels match the encoder. Mismatches here are silent debugging traps.

### 9.2 Stats line additions

`[STATS]` line currently includes `pEnt=...` (mean policy entropy). Add:
- `policySize=4864` (or include in startup banner only).
- `repFreq=N` — running fraction of training-batch positions with `repetitionCount ≥ 1`. Useful to see whether self-play is producing any repetition-bearing positions for the network to learn from. If `repFreq=0` consistently early in training, the rep planes are giving zero learning signal and we'd want to know.

### 9.3 Per-game repetition tracking

In the per-worker stats box (`ParallelWorkerStatsBox` or similar), expose:
- Current game's max rep count seen.
- Was this game decided by 3-fold? (Boolean per game, summed in the stats line.)

Optional but useful for debugging early-training behavior, where 3-fold draw loops are the pathology we're trying to fix.

### 9.4 Forward Pass demo

The demo UI shows top-K policy moves. After Phase 8:
- Decoder needs the new 76-channel encoding. `PolicyEncoding.decode` handles it.
- Display K=10 top moves with their (channel, square) interpretation labeled — useful for catching encoding bugs visually.

### 9.5 New "Engine Diagnostics" UI button

A single button in the UI menu that runs all the validation probes from Phases 1, 2, 3, 4 in sequence. Outputs to the session log:
- Zobrist diagnostic results (pass/fail per probe).
- PolicyEncoding round-trip results (count of round-trips attempted, mismatches).
- BoardEncoder shape/value probe.
- Network forward-pass shape probe + trainable-variable count check.

Total runtime should be < 5 seconds. Run on every fresh build during smoke testing.

### 9.6 Build banner

The startup `[APP]` log line includes build counter and git hash. Add `arch_hash=<hex>` so it's trivially obvious in logs that the architecture has changed. (Already implicitly there if the existing banner reads `ModelCheckpointFile.currentArchHash`; verify.)

### 9.7 Repo doc updates (must be in this bundle, not deferred)

These markdown files contain load-bearing references to the OLD architecture. Stale doc actively misleads future work, so update as part of this bundle:

- **`CLAUDE.md` (project-level):** rewrite the "Board encoding and policy space" section to reflect 20 input planes, 76 channels × 64 = 4864 logits, AlphaZero-shape encoding, SE blocks. Currently says "18 planes × 8 × 8" and "4096 logits, indexed as `fromSquare * 64 + toSquare`" — both wrong post-change.
- **`chess-engine-design.md`:** the design narrative describes the FC head + 18 planes. Add a new section "Architecture v2" that describes the new shape, references this plan, and marks the old description as historical.
- **`sampling-parameters.md`:** any mention of `log(4096)` entropy ceiling becomes `log(4864) ≈ 8.49`. Also update any references to the policy index encoding.
- **`mpsgraph-primitives.md`:** if it has FC-head examples, add SE-block + 1×1-conv-head examples since they're now the in-codebase patterns.
- **`ROADMAP.md`:** mark items related to this architecture refresh as completed (preserving detail per CLAUDE.md rule). Move from "in-flight" or "deferred" to "completed."
- **`CHANGELOG.md`:** new entry at the top describing the bundled change, commit hash, summary of param/compute deltas.

### 9.8 `[BATCHER]` startup probe verification

CLAUDE.md mentions a `[BATCHER]` correctness probe that runs at session start. It should "just work" if it queries network output dimensions dynamically rather than hardcoding 4096. **Verify before smoke test:** open `BatchedMoveEvaluationSource` (or wherever the probe lives), confirm size is read from the network, not hardcoded. If it's hardcoded, fix to use `ChessNetwork.policySize`.

### 9.9 Validation

- Visual: open the app, click "Engine Diagnostics," verify all probes pass and the log shows a clean diagnostic block.
- Visual: open Tensor Carousel, scroll through all 20 planes, verify labels match what each plane visually shows.
- Functional: trigger a Forward Pass demo on `.starting`, verify the top-10 moves include sensible opening moves (e2-e4, d2-d4, knight openings) and not absurdities (a1-h8, etc.). At init the network is random, but moves should at least be *legal*.

---

## Phase 10 — End-to-end integration smoke test

After all phases complete and Engine Diagnostics passes:

1. **Build clean** with xcode-mcp-server. Zero errors, zero warnings.
2. **Launch app**. Confirm `[APP]` banner shows new arch hash.
3. **Click "Build Network"**. Confirm parameter count log shows ~2.47 M.
4. **Click "Engine Diagnostics"**. All probes pass.
5. **Click "Forward Pass"** on `.starting`. Top-10 moves are legal opening moves.
6. **Click "Run Game"** (Play Game) once. A complete game plays without crashing or producing illegal moves. The game's PGN-ish notation in the log shows sensible moves.
7. **Click "Play and Train"**. Run for 5 minutes:
   - Stats line appears every 30s. `pLoss`, `pEnt`, `vMean`, `vAbs`, `gNorm` all in sane ranges.
   - `pEnt` starts near `log(4864) ≈ 8.49`, drifts down slowly. No `[ALARM]`.
   - At least one game completes per worker. `repFreq > 0` after 5 minutes (some self-play games will hit repetitions early in training).
   - Trainer makes ≥1000 steps. Loss decreasing (or at least not blowing up).
8. **Save session**. Manually trigger a save. Verify `.dcmsession` bundle written. Confirm bit-exact verification passes (logged).
9. **Quit and relaunch app**. **Load** the saved session. Confirm `loadedNet.evaluate(.starting)` matches a recorded probe from before quit.
10. **Run Arena**. 10 candidate games vs champion. Result reasonable (not all draws by repetition, not all losses on time).

If all 10 steps pass, the bundled change is shippable. Commit.

**Plus: run the XCTest suite via `mcp__xcode-mcp-server__run_project_tests`** before commit. All Phase 11 tests must pass without modification (per CLAUDE.md "tests MUST NOT be modified to make them pass").

---

## Phase 11 — Unit tests (new XCTest target)

This bundle introduces several pure-logic components with formal correctness invariants (bijections, hash determinism, plane-content rules) where unit tests catch bugs that diagnostic probes cannot. The project currently has no XCTest target (per CLAUDE.md "testing is manual"). **Add one** specifically for these new components — keep the diagnostic-probe pattern (Engine Diagnostics UI button) for everything that requires Metal/MPSGraph setup.

### 11.1 XCTest target setup

User creates the target in Xcode (one-time): `File → New → Target → Unit Testing Bundle`, named `DrewsChessMachineTests`, host application set to `DrewsChessMachine`. Resulting `Tests/DrewsChessMachineTests.swift` is the entry-point file. Subsequent test files added to that target's compile sources.

Build/run via `mcp__xcode-mcp-server__run_project_tests` against the project.

### 11.2 What goes into XCTest (high value, pure logic)

**`PolicyEncodingTests.swift`** — bijection guarantees:
- `testRoundTripStartingPosition`: every legal move from `.starting` round-trips exactly.
- `testRoundTripAfter1e4`: same after 1.e4.
- `testRoundTripCastlingPositions`: kingside and queenside castling for both colors round-trip.
- `testRoundTripAllPromotionsOnEveryFile`: position with pawns on the 7th rank for both colors; verify all 4 promotion variants on each file produce 4 distinct policy indices and round-trip.
- `testRoundTripEnPassant`: position with EP available; the EP capture round-trips.
- `testRoundTripCornerKnights`: knight on each corner square (a1, a8, h1, h8); all legal jumps (only 2 from each corner) round-trip; the 6 illegal jumps don't show up.
- `testNoLegalMovesShareIndex`: across 50+ hand-crafted positions, no two `legalMoves` produce the same `policyIndex`.
- `testDecodeRejectsOffBoardDestinations`: programmatically call `decode` with every (channel, row, col) combination from `.starting`; verify the off-board ones return `nil` cleanly.
- `testDecodeRejectsIllegalMovesAtCurrentPosition`: in a known position, call `decode` for a (channel, row, col) that decodes to a move that's not in `legalMoves` (e.g., decode a queenside-castle channel when castling rights are gone) — expect `nil`.

Target coverage: every channel range (queen, knight, underpromo, queen-promo) exercised by at least one round-trip test.

**`ZobristTests.swift`** — hash correctness:
- `testStartingHashDeterministic`: compute hash of `.starting` 100 times, all identical.
- `testIncrementalEqualsFull`: play a 50-move random-legal-move game; after every move, assert `state.zobrist == Zobrist.computeHash(state)`.
- `testHashDiffersForDifferentSideToMove`: same board layout, swap currentPlayer, verify hashes differ.
- `testHashDiffersForCastlingRights`: same board, toggle each castling right; each toggle changes the hash.
- `testHashDiffersForEPSquare`: identical positions with and without an EP target produce different hashes.
- `testRepetitionCountKnightShuffle`: play knight-shuffle sequence Nf3, Ng1, Nf6, Ng8, Nf3, Ng1; verify rep count goes 0, 0, 0, 0, 1, 1.
- `testHalfmoveClockResetClearsHistory`: play a sequence that creates a 1-rep, then make a pawn move, then return to the previously-seen-once position; verify rep count is 0 (history was cleared).

**`BoardEncoderTests.swift`** — plane-content invariants:
- `testStartingPositionPieceLocations`: encoded `.starting` has exactly 16 ones in each of planes 0-5 (current player's pieces) and planes 6-11 (opponent's pieces). Halfmove plane is all-zero. Castling planes (12-15) are all-1.
- `testRepetitionPlanesZeroAtCount0`: `BoardEncoder.encode(.starting, repetitionCount: 0)` has plane 18 and plane 19 all-zero.
- `testRepetitionPlanesAtCount1`: count=1 → plane 18 all 1.0, plane 19 all 0.0.
- `testRepetitionPlanesAtCount2`: count=2 → both planes all 1.0.
- `testRepetitionCountSaturates`: count=5 → same as count=2 (saturation).
- `testEnPassantPlane`: position with EP square set → plane 16 has exactly one 1.0 at the right square.
- `testHalfmoveClockNormalization`: clock=99 → plane 17 all 1.0; clock=50 → plane 17 all `50/99 ≈ 0.505`; clock=0 → plane 17 all 0.
- `testFlipFromBlackPerspective`: encode same position from white POV and from black POV; verify the row indexing of pieces is correctly flipped.
- `testTensorLength`: `encode(.starting).count == 1280`.

**`ReplayBufferTests.swift`** — write/read round-trip:
- `testEmptyBufferWriteRead`: write empty buffer to temp file, read back, verify size 0.
- `testSinglePositionWriteRead`: append one position, write, read, verify identical floats / move / outcome / vBaseline.
- `testFullBufferWraparound`: fill ring past capacity, write, read, verify oldest entries dropped and newest preserved.
- `testV3RejectsV2File`: synthesize a v2-formatted file (handcrafted bytes), attempt restore, verify throws `unsupportedVersion`.
- `testCorruptedHeaderRejected`: truncate a valid v3 file mid-header, verify restore throws.

These four test files give us coverage for all the new pure-logic components. Total ~30-50 test cases.

### 11.3 What stays as Engine Diagnostics probes (Metal/MPSGraph required, integration-level)

Keep the existing diagnostic-probe pattern for:
- Network forward-pass shape and trainable-variable count (Phase 4.5 — requires graph construction).
- Bit-exact checkpoint round-trip (Phase 7.4 — already runs on every save).
- Repetition-plane signal flows through to network output (Phase 8.3).
- BATCHER probe correctness check (it's already a runtime probe; just remove the hardcoded constant).
- Game-diversity tracker round-trip (Phase 5.2.7 — verify two identical games classified identical).

These probes are run from the "Engine Diagnostics" UI button (Phase 9.5) and from the smoke test (Phase 10).

### 11.4 CI / pre-commit hook (out of scope)

The plan does not introduce a CI pipeline or git pre-commit hook. Tests run on demand via xcode-mcp-server during development. If we want enforcement later, that's a separate piece of work.

### 11.5 Validation

Phase 11 itself is "complete" when:
- `mcp__xcode-mcp-server__run_project_tests` runs all four test files, all green.
- Total test runtime under 10 seconds (these are pure-logic tests).
- Test target builds cleanly without warnings.

---

## Risk register

| Risk | Mitigation |
|---|---|
| `PolicyEncoding` bijection has off-by-one or rotation bug | Phase 2.4 round-trip validation across many positions; Forward Pass demo visual check |
| Zobrist incremental update doesn't match full hash | Phase 1.5 incremental-vs-full check on every move of a 50-move game |
| SE block multiplication broadcasting wrong axis | Phase 4.5 channel-uniformity probe at init |
| `currentArchHash` doesn't capture SE addition (silent compat) | Phase 7.4 explicit print + diff before/after |
| ReplayBuffer v3 reader can't reject v2 cleanly | Phase 6.7 explicit version-rejection probe |
| Stale `policySize=4096` references in dependent code | Delete `ChessMove.policyIndex` entirely (§5.1) — compile errors find every caller |

---

## Estimated effort

Implementation: ~6–10 hours focused work. Validation/smoke: ~2 hours. Sequenced as one branch, one bundled commit.

---

## Decisions (resolved)

All locked-in. Recorded for traceability.

1. **Channel ordering** — locked at the values written in §2.1: queen-style directions `N, NE, E, SE, S, SW, W, NW` (clockwise from "up"); knight jumps `(-2,+1), (-1,+2), (+1,+2), (+2,+1), (+2,-1), (+1,-2), (-1,-2), (-2,-1)` (clockwise from "up-right"); underpromotion piece order `N=0, R=1, B=2`; direction-within-promotion order `forward=0, capture-left=1, capture-right=2`. These choices are arbitrary; documenting them as the locked spec is what matters. Any reordering after this point breaks every saved network and replay buffer trained under the prior order.
2. **`repetitionCount` saturates at 2.** Encoder receives 0, 1, or 2 only. Caller's responsibility to clamp; encoder accepts unsaturated input (per §3.3 validation) but treats anything ≥2 identically.
3. **`ChessMove.policyIndex` deleted entirely.** No backward-compat shim. Compile errors at every callsite are the safety net.
4. **ROADMAP.md update** — abridged version added there at the end of implementation, referencing this doc as the full plan. Do this after all phases complete and the bundle is committed; ROADMAP entry should describe outcome and link forward to this doc, not duplicate the design.

## Implementation guidance

For a change this large, suggest implementing in phases and pausing for review between:
- After Phase 4 (network architecture is done and verified — graph compiles, forward pass produces right-shaped output).
- After Phase 8 (rest of the system catches up — engine plays a complete game with the new encoding).
- After Phase 11 (XCTest suite green — all pure-logic invariants verified).

Then proceed to Phase 10 smoke test as the final integration gate.

---

## Deferred — recorded for future consideration

These came up during planning but are explicitly NOT part of this bundle. Recording them here so they aren't lost.

### WDL value head (3-class win/draw/loss output)

> **DONE — 2026-05-12 (commits `4c00983` … `29b8597`).** Implemented essentially as described below. The "why deferred / when to revisit" reasoning was overtaken by events: the build-893 fresh-from-random run showed the scalar `tanh` head goes *silent* on a draw-heavy self-play buffer (`vAbs → 0`, `vLoss` flat) and the arena candidate plateaus at ≈parity — so the WDL switch became the fix for an active bottleneck, not a future MCTS-coupled nicety. The actual implementation matched the "implementation cost" estimate closely (FC2 1→3, drop tanh, derived scalar `p_win − p_loss`, MSE→CE, arch-hash bump via a new `valueHeadClasses` constant, `vMean`/`vAbs` kept + `pW`/`pD`/`pL` added, `vBaselineDelta` *removed* rather than re-derived). Two small deviations: a defensive `clamp(slot, 0, 2)` before the one-hot, and the inference path was left scalar (the 3-wide distribution is training-graph-only — no inference call site needed changing). See `wdl-value-head.md` (design), the 2026-05-12 CHANGELOG entry, and `wdl-implementation-log.md` (build-by-build record + deferred follow-ups). The original deferral note is kept below as the historical rationale.

**What:** replace the current scalar `tanh` value head with a 3-logit softmax head outputting `(P_W, P_D, P_L)`. Train against one-hot game outcome with cross-entropy loss. Convert to scalar at inference via `v = P(W) - P(L)` for the advantage baseline and UI displays.

**Why:** distinguishes "certain draw" `(0, 1, 0)` from "uncertain anything" `(0.5, 0, 0.5)` — the scalar head conflates these. Avoids tanh saturation (better gradients near ±1). Tunable inference behavior (draw aversion / draw seeking via weighted scalar conversion). lc0 standard since ~2019; reported ~10-30 Elo gain.

**Why deferred:** marginal benefit at our current strength + we have no MCTS to amplify value-head improvements. Bundling now adds validation surface for limited near-term gain and makes Elo attribution harder. The "extra retrain" cost from sequencing is small for our networks.

**When to revisit:** **if/when we implement MCTS.** WDL benefits compound through search (Q values, virtual losses, exploration bonuses). Without MCTS, the marginal Elo doesn't justify the bundling cost; with MCTS, it does. *(Note: MCTS is currently an explicit non-goal per CLAUDE.md — this trigger is "if we ever change that stance," not "this is on the imminent roadmap." Do not get excited about MCTS.)*

**Implementation cost when triggered:** value head graph change (FC2 1→3, drop tanh), loss function swap (MSE → cross-entropy), scalar conversion at every value-reading site (advantage baseline, UI, arena), stats line redefinition (`vMean`/`vAbs` → per-class probs + scalar `v = P(W) - P(L)`). ~1-2 hours implementation + careful per-call-site validation. Forces another arch hash bump and full retrain.

### Last-N-moves history input planes

**What:** extend the input tensor to include the previous N positions in the game (each as a stack of piece planes), so the network sees recent move history rather than just the snapshot board state. AlphaZero used N=8 (8 historical positions × 14 planes per position = 112 history planes, on top of the 7 constant planes).

**Why:** gives the network access to "how the position evolved" — useful for piece-activity reasoning, reading tactical sequences, and detecting "this is a forcing move continuation" patterns. Also subsumes the dedicated repetition planes (history + comparison can compute repetition; though it's harder to learn than just being told).

**Why deferred:** *enormous* input shape expansion (currently 20 planes, would jump to ~150+). Forces:
- Stem conv `inputPlanes × channels × 9` weight count grows ~7×.
- Per-position memory in replay buffer grows ~7× (~32 KB per position × 1 M positions = 32 GB ring — infeasible without disk-backed storage).
- Per-position encoding cost grows ~7×.
- We'd need to plumb a per-game position-history *board buffer* (not just hash list) into every encoding callsite.

For us this is *much* more expensive than for AlphaZero because we don't have MCTS to amortize the bigger network forward pass over many search nodes. We do one forward per ply.

**When to revisit:** if and when training shows the network plateauing in tactical patterns that visibly require seeing history (e.g., consistently failing to follow up forcing sequences). Until then, the snapshot + repetition planes is sufficient.

**Lighter-weight alternative to consider first:** add planes for "squares of pieces moved in the last 1-2 plies" (small, cheap, gives the network most of the "what just happened" signal without the full history stack). This is a fraction of AlphaZero's full history scheme and might capture most of the benefit at 5-10% the input cost.

---

## Current state (as-built, through commit `cf1cc24`, 2026-04-20)

The plan above (Pre-flight → Phase 11) is the as-designed bundle. This section is the as-built snapshot of what actually ships in the current tree. It folds in both the immediate post-implementation follow-ups (originally logged as addenda A–H) and the four post-v2 commits that landed before this section was written: `9298273` (random-ish-move counter, full-sort top-K, sign-consistency tests), `068f805` (action-index health stats + MPSGraph reshape layout tests), `7757418` (advantage standardization + live-editable hyperparameters + diagnostics expansion), and `cf1cc24` (advantage raw-ring capped at 32K to unblock the main actor).

**This is the first v2 run.** The champion active when this section was written is the first network trained end-to-end on the v2 architecture; any discussion of "prior behavior" refers to pre-v2 code that no longer runs.

### Network architecture (fixed constants)

From `ChessNetwork.swift`:

| Constant | Value | Notes |
|---|---|---|
| `inputPlanes` | `20` | 16 piece/castling + EP + halfmove + 2 repetition planes |
| `channels` | `128` | residual-tower channel count |
| 8 residual blocks | hardcoded at `ChessNetwork.swift:323` | each block = 3×3 conv → BN → ReLU → 3×3 conv → BN → SE → add → ReLU |
| `seReductionRatio` | `4` | SE bottleneck = `channels/4 = 32` |
| `policyChannels` | `76` | 56 queen-style + 8 knight + 9 underpromotion + 3 queen-promotion |
| `policySize` | `4864` | `= policyChannels × 64` |
| Value head | 3-logit W/D/L softmax + categorical CE | scalar `tanh` removed 2026-05-12; downstream reads derived `v = p_win − p_loss`; `valueHeadClasses = 3` |
| Total trainable params | ~2.4 M | |

### Training-loop structure (as built)

Per-step, `ChessTrainer.trainStep(replayBuffer:batchSize:)` is a three-phase async sequence:

1. **Phase 1 (trainer queue):** sample from replay buffer into staging buffers; copy boards to a Sendable `[Float]`.
2. **Phase 2 (network queue, async):** `network.evaluate(batchBoards:count:)` — forward-only pass on the *current* trainer network, yielding fresh per-position `v(s)` scalars.
3. **Phase 3 (trainer queue):** overwrite the `vBaseline` staging with the fresh values, apply draw penalty, run the training graph.

Empirically verified via `MPSGraphGradientSemanticsTests`: MPSGraph has no `stop_gradient` op, and excluding tensors from the `with` array of `gradients(of:with:name:)` does NOT prune backward-pass paths. The only way to detach `vBaseline` from autograd is the placeholder boundary — which is why Phase 3 feeds the fresh values through the same `vBaseline` placeholder rather than computing them inside the graph. Cost: ~33% extra forward FLOPs per training step; `TrainStepTiming.freshBaselineMs` exposes the Phase 2 cost and `totalMs` includes it so the replay-ratio controller throttles correctly.

**Advantage standardization.** The policy-gradient weight is `A_norm = (A − mean(A)) / sqrt(var(A) + 1e-6)` computed per batch inside the graph, where `A = z − vBaseline`. This is autograd-safe because `A` depends only on the `z` and `vBaseline` placeholders — no gradient path flows through `mean`/`std` back into trainable variables. The forward `A_norm` is what multiplies `−log p(a*)`; the backward path is just the standard REINFORCE-with-baseline one. Removes the systematic bias that appeared when the value head developed a global offset (e.g. `E[v] ≈ 0.45` once draws dominated self-play, which skewed raw advantages positive for wins and negative for losses and pushed the trunk in one direction). Also stabilizes gradient magnitude batch-to-batch.

**Policy scale K = 5** (from `ChessTrainer.policyScaleKDefault`). K multiplies `A_norm · (−log p)` before the total-loss sum. Dropped from the pre-standardization value of 50 once standardization was in place — since `A_norm` already has unit stdev, the 50× multiplier was pinning `gradClipMaxNorm` almost every step (~0.6 % unclipped).

**Live-editable hyperparameters.** `weightDecayC`, `gradClipMaxNorm`, `policyScaleK`, `learningRate`, `entropyRegularizationCoeff`, and `drawPenalty` are all fed to the training graph per step through scalar `placeholder` tensors. UI edits to any of these commit immediately without rebuilding the graph; values persist in `@AppStorage` (where applicable) and are restored on session load (`SessionCheckpointState` hydrates the full set: `wd`, `clip`, `K`, `sp+ar tau`). Every commit writes a `[PARAM] name: old -> new` line to the session log. The self-play and arena `SamplingSchedule` values are similarly live-tunable: a `SamplingScheduleBox` threads through `BatchedSelfPlayDriver`, and reused `MPSChessPlayer` slots read the latest schedule at each new-game boundary.

### Current parameter defaults

Every value below is the ship-default on a fresh session. UI-editable values can be overridden at runtime; non-UI values are architectural or require source-code edits.

**Optimizer / losses (`ChessTrainer.swift`):**

| Name | Default | UI-editable | Role |
|---|---|---|---|
| `trainerLearningRateDefault` | `5e-5` | yes | Adam-like effective step (actually plain SGD with momentum in the trainer graph) |
| `entropyRegularizationCoeffDefault` | `1e-1` | yes | weight on `+H(π)` term that pushes toward uniform |
| `drawPenaltyDefault` | `0.1` | yes | adds to `z` on drawn games so policy doesn't converge to "play for a draw" |
| `weightDecayCDefault` | `1e-4` | yes | decoupled L2 on weights in the `shouldDecay` group (conv/FC weights only; BN γ/β and biases excluded) |
| `gradClipMaxNormDefault` | `5.0` | yes | global gradient L2 clip |
| `policyScaleKDefault` | `5.0` | yes | multiplier on the REINFORCE policy-loss term |

**Batching and replay (`ContentView.swift`):**

| Name | Value |
|---|---|
| `trainingBatchSize` | `4096` |
| `replayBufferCapacity` | `1_000_000` |
| `minBufferBeforeTraining` | `max(25_000, capacity/5) = 200_000` |
| `rollingLossWindow` | `512` (steps, for STATS smoothing) |
| `TrainingLiveStatsBox.advRawRingMaxCapacity` | `32_768` Float entries (see below) |
| `replayRatioTarget` | `1.0` (default in `@AppStorage`) |
| `replayRatioAutoAdjust` | `true` |

**Self-play driver (`ContentView.swift`):**

| Name | Value |
|---|---|
| `initialSelfPlayWorkerCount` | `24` |
| `absoluteMaxSelfPlayWorkers` | `64` |
| `stepDelayMaxMs` | `2000` |
| `stepDelayLadder` | `[0, 5, 10, 15, 20, 25, 50, 75, …, 2000]` (25 ms steps above 25) |

(These are higher than the figures originally quoted in the ROADMAP "N-worker concurrent self-play" entry, which cited `6 / 16`. The memory-vs-latency trade-off analysis in that entry still applies — at `64` pre-allocated workers the steady-state memory footprint is ~720 MB of idle network state plus ~300 MB of `MPSChessPlayer` scratch, which is acceptable on 64 GB Apple Silicon where this project is actually run.)

**Arena (`ContentView.swift`):**

| Name | Value |
|---|---|
| `tournamentGames` | `200` |
| `tournamentPromoteThreshold` | `0.55` |

**Sampling schedules (`MPSChessPlayer.swift`, `SamplingSchedule`):**

| Preset | `startTau` | `decayPerPly` | `floorTau` | `pliesUntilFloor` | Dirichlet noise |
|---|---|---|---|---|---|
| `.selfPlay` | `2.0` | `0.03` | `0.4` | `54` | `DirichletNoiseConfig.alphaZero` (α=0.3, ε=0.25, plyLimit=30) |
| `.arena` | `2.0` | `0.04` | `0.2` | `45` | none |
| `.uniform` | `1.0` | `0.0` | `1.0` | `Int.max` | none |

Both `.selfPlay` and `.arena` were bumped from the pre-v2 defaults (`1.0` and `0.7`) to `2.0` after the new policy head (no BN before the 1×1 conv) produced wider initial logit distributions than the FC head did. Higher starting `tau` flattens the early-game distribution, increases opening diversity, and pulls more decisive games out of the trainer-vs-champion arena.

**Alarm thresholds (`ContentView.swift`):**

| Name | Value | Rationale |
|---|---|---|
| `policyEntropyAlarmThreshold` | `5.0` | `log(4864) ≈ 8.49`; v2 init entropy ~6.5 (wider logits than the FC head produced). A `5.0` floor is ~1.5 nats below init — wide enough to avoid false alarms, narrow enough to flag genuine collapse. |

### Diagnostics surface

All figures below appear as fields on `TrainStepTiming`, are aggregated in `TrainingLiveStatsBox`, and feed both the in-app training panel and the `[STATS]` log line.

**Losses / grads / values:**
- `policyLoss`, `valueLoss`, `loss` (total)
- `policyEntropy` — Shannon entropy over the softmaxed policy, in nats; uniform = `log(4864) ≈ 8.49`
- `policyNonNegligibleCount` — count of cells with `p > 1/policySize` (displayed as "Above-uniform policy count")
- `gradGlobalNorm` — pre-clip L2 norm across all trainable vars
- `valueMean`, `valueAbsMean` — mean / mean-abs of the derived value scalar `p_win − p_loss` (post-2026-05-12; was a tanh-saturation probe). `valueProbWin` / `valueProbDraw` / `valueProbLoss` — the W/D/L softmax batch-means (`pW=/pD=/pL=` on `[STATS]`); `pD → 1` is the value-head collapse signature.

**Value-baseline divergence:**
- `vBaselineDelta` — *removed 2026-05-12* (it was the only consumer of the replay buffer's stored play-time value; the WDL switch dropped the drift diagnostic, and the trainer's `vBaseline` placeholder is fed entirely from the fresh trainer-forward).
- `freshBaselineMs` — wall-clock cost of Phase 2.

**Policy health (added by `068f805` / `7757418`):**
- `playedMoveProb` — batch mean of `p(a*)` for the actually-played move, computed as `sum(softmax · oneHot)` along the class axis. Random init sits at `1/4864 ≈ 2e-4`. A plateau near this value while `pLoss` moves would indicate an action-index mismatch.
- `policyLogitAbsMax` — batch mean of `max_i |logits[i]|`. Pairs with `policyEntropy`: entropy can look healthy while one runaway logit is pre-saturating the softmax.
- `policyHeadWeightNorm` — L2 norm of the 1×1 conv's final weight tensor (128 → 76 = 9,728 floats), read via an MPSGraph `targetTensor` so the host never pulls the weights back to CPU just to measure them. Tracks whether weight-decay is balancing the pull from `pLoss`.
- `legalMassSnapshot` — separate `ChessTrainer.legalMassSnapshot` forward-only probe over a sampled replay subset: CPU-computes `legalMass` (sum of softmax probability on legal moves) and `top1Legal` (fraction of batch where `argmax(softmax) ∈ legal`). Uses `BoardEncoder.decodeSynthetic` to reconstruct positions for legality. Refreshed every 25 steps during the bootstrap phase and on every steady-state emit.

**Advantage distribution (from `7757418`):**
- Summary scalars: `advantageMean`, `advantageStd`, `advantageMin`, `advantageMax`, `advantageFracPositive` (fraction with `A > 0`), `advantageFracSmall` (fraction with `|A| < 0.05`). Reduced graph-side.
- Percentiles: `p05 / p50 / p95` computed CPU-side from `TrainingLiveStatsBox._advRawRing`, a rolling ring of raw per-position values.

**The 32K ring cap (from `cf1cc24`).** `_advRawRing` was originally sized `rollingWindow × batchSize = 512 × 4096 ≈ 2M Float`. `snapshot()` sorts the full filled portion for percentile extraction, and the UI heartbeat's 10 Hz `Task { @MainActor }` calls `snapshot()` via `queue.sync` — once the ring filled (~step 500) each sort cost ~150 ms on main, saturating the main actor. `fireCandidateProbeIfNeeded` is `@MainActor` and awaited after every training step, so training throughput collapsed from ~2300 moves/sec to ~300 moves/sec and the UI went non-responsive. The cap drops each sort from ~150 ms to ~1 ms while keeping percentile error below 0.5 % for a log-eyeballed diagnostic.

**Self-play / game-diversity (from `068f805` / `9298273`):**
- `ParallelWorkerStatsBox` maintains a 512-entry game-length ring producing `p50 / p95 / avgLen` on top of the existing `GameDiversityTracker` unique-game histogram.
- `MPSChessPlayer.recordedRandomishMoves` — per-game counter of plies where the post-temperature legal-move softmax `max` probability fell below `1.5 / N_legal`. Signal that the sampler was effectively picking at random — a policy-collapse or degenerate-logit indicator independent of temperature. Driver-level aggregation is a follow-up.

**Top-K decoder robustness (from `9298273`).** `ChessRunner.extractTopMoves` now fully sorts the 4864-cell policy vector rather than capping at `count × 4`. The bounded-heap scheme failed when a catastrophically collapsed policy's top cells were *all* off-board: `extractTopMoves` returned an empty list and the Candidate Test panel silently showed nothing. With a full sort, collapsed policies still produce `count` legal visualizations (geometrically decoded with off-board skips).

### Candidate Test view

- Top-K displayed as RAW cells (includes illegal), labelled "Policy Head (Top 4 raw — includes illegal)" with `(illegal)` suffix where legality fails. Implemented via `PolicyEncoding.geometricDecode(channel:row:col:currentPlayer:)` (geometry only, no legality filter) + a post-hoc legal-move `Set<ChessMove>` lookup. `ChessMove` is `Hashable` (synthesized) for fast membership checks.
- Metrics: "Above uniform: X / N legal (threshold = 1/N)" and "Legal mass sum: Y %" replaced the earlier `NonNegligible: X / 4864` and the misleading `(v+1)/2 → "X % win / Y % loss"` line (scalar-tanh→WDL conversion was dishonest without a WDL head).
- `TrainingChartGridView` chart label: "Above-uniform policy count".

### UI and observability

- **Cumulative status bar** at top of window: Active training time, Training steps, Positions trained, Runs. Sums across all completed Play-and-Train segments + the in-flight one.
- **Training segments:** `SessionCheckpointState.TrainingSegment` codable struct + per-run lifecycle (start on Play-and-Train, close on Stop or Save, reopen on save-while-training-continues). Persists across save/load. `[SEGMENT]` log lines on every transition.
- **Run Arena Now** button in the status bar (visible whenever no arena is running and a network exists).
- **Engine Diagnostics** menu item (Debug → Run Engine Diagnostics): PolicyEncoding round-trip, distinct-index check, repetition-tracking 3-fold detection, BoardEncoder shape, network forward-pass shape probes. Output as `[DIAG]` log lines.
- **Build banner:** includes `arch_hash`, `inputPlanes`, `policySize`.
- **Parameter logging:** every UI commit on Learn Rate, Entropy Reg, Draw Penalty, Weight Decay, Grad Clip, Policy Scale K, Self-Play Workers, Step Delay (manual), Replay Ratio Target, Replay Ratio Auto-Adjust, `sp.startTau / floorTau / decayPerPly`, `ar.startTau / floorTau / decayPerPly` writes a `[PARAM] name: old -> new` line.
- **Alarm banner:** black title + dark-red medium-weight detail for legibility against yellow background. New `Dismiss` button alongside `Silence` (resets divergence streak counters). Every raise/clear/silence/dismiss logs `[ALARM] …`.
- **Save errors** auto-log via `setCheckpointStatus(_, kind: .error)` → `[CHECKPOINT-ERR]`. The three-kind status line (`.progress / .success / .error`) replaces the earlier two-state boolean; success messages show a green ✓ and linger 20 s for durable confirmation.
- **Periodic session autosave** every 4 hours while Play-and-Train is active, plus a post-promotion autosave that already existed. Both run through `setCheckpointStatus` with trigger-tagged wording (`(periodic)` / `(post-promotion)`). Arena-deferral and post-promotion swallow handled by the testable `PeriodicSaveController` (pure logic, no timer).
- **Launch-time auto-resume** driven by `LastSessionPointer` (UserDefaults). Sheet with 30 s countdown on first `.onAppear`; File menu "Resume Training from Autosave" covers the rest of the launch. Failed loads surface as status-bar errors — files are never deleted on failure. Stale pointers (target folder missing) are cleared on first observation.
- **Hourly rate** shown next to per-second gen/trn rates: `1m gen rate: 3500 pos/s   (12,600,000/hr)`.
- **vMean / vAbs** rows added under `Loss value:` in the Training column.
- **Title-bar contrast:** build banner and right-side ID bumped from `.caption` to `.callout`.
- **STATS emitter** runs per-step for the first 500 steps (bootstrap phase, dense signal during the most interesting period), then every 60 s in steady state. `legalMassSnapshot` refreshes on a 25-step stride during bootstrap and on each steady-state emit.
- Removed duplicate `Ent reg`, `Grad clip`, `Weight dec`, `Draw pen` rows that were already shown in the editable hyperparameter section.

### Tests (XCTest target populated)

Populated after the user added the XCTest target via Xcode. Current count: 40 + 7 + tests from `068f805` / `9298273` additions, spread across:

- `PolicyEncodingTests.swift` — round-trips, distinct indices, off-board guard, perspective flip, castling encoding.
- `BoardEncoderTests.swift` — tensor length, plane content for pieces / castling / EP / halfmove / repetition planes, perspective flip.
- `RepetitionTrackingTests.swift` — 3-fold detection, halfmove-clock-reset history clearing.
- `ReplayBufferTests.swift` — round-trip, version rejection (synthesized v2 file), bad magic, truncated header.
- `MPSGraphGradientSemanticsTests.swift` — empirical verification that excluding tensors from the `with` array of `gradients(of:with:name:)` does NOT prune backward-pass paths.
- `MPSGraphReshapeLayoutTests.swift` (from `068f805`) — empirical check that the policy head's `[B, 76, 8, 8] → [B, 4864]` reshape is NCHW row-major under `c·64 + r·8 + col`, plus end-to-end round-trip through `oneHot` + `softMaxCrossEntropy`.
- `SignConsistencyTests.swift` (from `9298273`) — encoder symmetry, policy-index symmetry for mirrored moves, outcome-sign truth table, advantage formula sign convention, geometric decode round-trip, bit-identical network output for bit-identical inputs.

### Known open items carried forward

- **TODO #3 — ReplayBuffer durability.** Resolved 2026-04-20 in the "saved means golden" durability bundle. `ReplayBuffer` moved to v4 format with SHA-256 trailer, strict file-size equality check, upper-bound caps, and `handle.synchronize()` on write; `CheckpointManager.saveSession` gained a full `F_FULLFSYNC` pipeline (every staged file, tmp directory, parent directory) plus a post-write scratch-restore verification against the snapshot atomically returned from `ReplayBuffer.write`. Launch-time orphan sweep removes crashed-mid-save `.tmp` debris. Full detail in `ROADMAP.md` → Completed; byte-level specs in `dcmmodel_file_format.md` and `replay_buffer_file_format.md`.
- **Adaptive learn-rate schedule.** Recorded in `ROADMAP.md` as a future-work entry with five candidate trigger families (step decay, plateau detection, promotion-driven, cosine annealing, replay-ratio aware). Not implemented.
- **`mps.placeholder` runtime assertion.** Observed during training after the live-hyperparameters change in `7757418`. `graph.read(…)` is defensive but speculative; the three new scalar placeholders use the same pattern as the working `lr` / `entropyCoeff` placeholders. Root cause still under investigation as of this writing.
