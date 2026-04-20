# TODO_NEXT

Deferred items from the 2026-04-18 `/stupid` audit (3 parallel Explore agents → 7 Analyzer agents → 7 Reviewer agents). Full audit context in `~/.claude/plans/golden-questing-whale.md`. Items already implemented and committed in `6c96437` (archived-hash, NaN/Inf guard, checkpoint overflow cap, window-close teardown stub) are not repeated here.

## #1 — Promotion-target collapse in the 4096 policy *(highest — capacity ceiling)*

`ChessMove.policyIndex` (`ChessMove.swift:14-18`) ignores `.promotion`. The 4 promotion variants (Q/R/B/N) collide on the same index, so:
- **Sampling** (`MPSChessPlayer.swift:382-434`): all 4 get identical logits. Softmax ties; sampler picks whichever appears first in `legalMoves`. That is always Queen.
- **Training target** (`ChessTrainer.swift:728`): one-hot over a shared index. Network cannot learn underpromotion.

**Fix** — expand policy to 4288 slots (Queen uses base 4096; R/B/N get 3×64 underpromotion planes addressed by `toSquare`):
- `ChessMove.swift:14-18` — new `policyIndex`: Queen → `from*64+to` (0..4095); Rook → `4096 + toSquare`; Bishop → `4160 + toSquare`; Knight → `4224 + toSquare`.
- `ChessNetwork.swift:90` — `policySize = 4288`.
- `ChessNetwork.swift:1048-1055` — **critical**, update hardcoded FC weight shapes `[128, 4096]` → `[128, ChessNetwork.policySize]` and bias `[1, 4096]` → `[1, ChessNetwork.policySize]`.
- `ChessTrainer.swift:728` — change hardcoded `oneHot` depth 4096 → `ChessNetwork.policySize`.
- `ReplayBuffer`, `BatchedMoveEvaluationSource`, `MPSChessPlayer`, entropy computations — already pass `policySize` through; no change needed.
- `ModelCheckpointFile` — **no migration**; the arch hash (`currentArchHash`) already mixes `policySize`, so old `.dcmmodel` files cleanly fail to load with `.archMismatch`. Do not add He-init padding (violates CLAUDE.md "no migration without request").
- Update `log(4096)` comments in ChessTrainer to reference `policySize`.
- After bumping `policySize`, confirm the existing `maxTensorElementCount = 600_000` cap in `ModelCheckpointFile.swift` still covers the new largest tensor (128 × 4288 = 548,864 — fits).

Collision check: only pawns reach row 0/7 via a promotion from/to pair, so base-index 0..4095 for queen promotions cannot collide with any non-pawn move.

Estimated: 2–3 hrs, touches policy pipeline end-to-end; forces retrain from scratch (existing `.dcmmodel` files become incompatible via arch hash).

## #3 — ReplayBuffer durability: fsync + length invariant *(partial-write protection)*

`ReplayBuffer.write(to:)` (`ReplayBuffer.swift:~316-445`) does not call `handle.synchronize()` before close; `restore(from:)` reads chunks without verifying total file size matches the header-predicted length. A crash or disk-full mid-write produces a truncated file that loads as a partial buffer silently.

**Fix** (keep format v2; do NOT bump version):
- `ReplayBuffer.swift:~375` — add `try handle.synchronize()` before the `defer { try? handle.close() }`.
- `ReplayBuffer.swift:~499-644` (restore) — compute expected file size from header fields; `guard actualSize >= expectedBytes else { throw PersistenceError.truncatedFile }`.
- `CheckpointManager.saveSession` (`CheckpointManager.swift:~318-326`) — write the replay buffer into the tmp directory *before* the JSON so the tmp→rename atomic-move covers all three files as one unit.

Skip SHA-256 / Adler-32 — too much engineering for a user-local file whose cost-of-loss is ~5 minutes of refill. Revisit if telemetry shows silent corruption.

Estimated: 30 min, touches ReplayBuffer + CheckpointManager.

## #5 follow-up — Make `maxTensorElementCount` track the live arch

The current commit hard-codes `static let maxTensorElementCount: Int = 600_000` in `ModelCheckpointFile.swift`. That's safe for today's arch (largest tensor = policy FC = 128 × 4096 = 524,288 elements, plus ~15% headroom) but it's a separate constant from the network's actual shape, so bumping `channels` to 192/256 or `policySize` to 4288 (per #1 above) will silently false-reject valid saves.

Make the cap a computed property derived from the live constants:

```swift
static var maxTensorElementCount: Int {
    // Cover the three tensor-shape classes in the current network:
    //   - policy FC weights:  channels * policySize
    //   - conv residual:      channels * channels * 9
    //   - value FC:           small, already bounded by the others
    // Add a modest slack so that a minor arch tweak (e.g. adding
    // one layer) doesn't require touching this value too.
    let policyFC = ChessNetwork.channels * ChessNetwork.policySize
    let convWeights = ChessNetwork.channels * ChessNetwork.channels * 9
    return max(policyFC, convWeights) + 65_536
}
```

Non-urgent; only matters when the arch changes. Do this before (or as part of) #1 so the test suite doesn't start throwing `implausibleTensorSize` on the first clean save.

## Not doing (already rejected in audit — don't re-open without new evidence)

These came up during the audit and were verified to not be bugs. Recording them here so the next pass doesn't waste time re-investigating:

- **Castling out of check**: `MoveGenerator.swift:311` already asserts `!isSquareAttacked(state, row: homeRow, col: 4, by: color.opposite)`. Safe.
- **ChessRunner flip inversion**: Decoding at `ChessRunner.swift:99-114` correctly un-flips rows and keeps columns. Columns don't flip under perspective flip by design.
- **`try? await Task.sleep` in polling loops**: all 9 sites reviewed; each sits in a loop whose condition re-checks `Task.isCancelled` (or returns) after the sleep. No busy-spin hazard.
- **Deprecated 1-arg `.onChange`**: no 1-arg usages in the codebase; the 2-arg form `{ _, _ in ... }` is the current non-deprecated API.
- **Softmax overflow at low tau**: `MPSChessPlayer.sampleMove` subtracts `maxLogit` before `expf`, and `floorTau > 0` is enforced by precondition on `SamplingSchedule`.

## ML / MPSGraph Review (2026-04-19) — deferred items

These came out of the deep ML/MPSGraph code review on 2026-04-19. Items #1 (He init fan_in for FC layers), #5 (weight decay on BN/biases), #7 (Dirichlet noise), #8 (vDSP_vclip comment), and #9 (sweep contaminates BN running stats) are already fixed and merged. The four items below are deferred — each requires either retraining-from-scratch, an architectural decision, or a forward-incompatible weight change, so they need ROADMAP-level discussion before implementation. Original review numbering preserved.

Note on overlap: the review's #3 (underpromotion) and the existing "Critical #1" entry above (promotion-target collapse, 4288-slot fix) are the same problem. Both descriptions are kept here so both proposed approaches stay on the table during the discussion.

### #2 — Policy head has no spatial structure

The architecture is `1×1 conv (128→2) → BN → ReLU → flatten 128 → FC 128→4096`. Squeezing 8×8×128 down to a 128-dim vector via two 1×1 channels and then expanding to 4096 with one FC throws away every spatial relationship between source and destination squares. Every "from-square × to-square" output depends globally on a 128-dim bottleneck.

This is unusual — AlphaZero-family policy heads are convolutional all the way (e.g. `1×1 conv (128→73)` → flatten to 8·8·73 = 4672 outputs, where each output cell has a meaningful spatial neighborhood). The 128-dim bottleneck is the most likely architectural ceiling on policy quality. Worth at least documenting as a known limitation alongside the "no MCTS" stance, or experimenting with a wider bottleneck (8 or 16 channels post-1×1) and a conv-style policy.

### #3 — Move encoding silently collapses underpromotion into queen-promotion

`networkPolicyIndex` at `MPSChessPlayer.swift:358`: `from*64 + to` gives 4096 slots. Pawn promotions to queen, knight, rook, bishop all share the same index. The move-generator presumably picks queen by default when sampling, so the network can never learn to prefer underpromotion (e.g. knight-promotion to give check / fork). For a self-play bootstrap engine that's a real strength ceiling — there are real positions where queen-promotion loses but knight-promotion wins.

Either (a) acknowledge this in the docs and accept the cap, or (b) extend to AlphaZero's 4672 policy space (8·8·73) which encodes promotion piece per move.

(See "Critical #1" earlier in this file for an alternative 4288-slot proposal that solves the same problem with a smaller policy expansion. Compare the two approaches when discussing.)

### #4 — `vBaseline` is a *frozen* baseline, not the current value-head estimate

Captured in `MPSChessPlayer.onChooseNextMove` from the play-time forward pass and stored in `ReplayBuffer`. Then fed back at training time as a placeholder so autodiff treats it as a constant.

This is a defensible variance-reduction choice, but it's not what "advantage baseline" usually means — the standard formulation uses the *current* value-head's prediction with `stop_gradient`. Two consequences:

- After arena promotion, the inference network's `v(s)` (the source of `vBaseline`) is one or many promotion cycles behind the trainer's `v(s)`. Replay-buffer entries from before promotion carry baselines from an even older value head. The bias is unbiased on average (any state-dependent baseline is unbiased) but the variance reduction degrades the more stale the baseline is.
- The trainer's value head improves continuously, but `vBaseline` does not — which means `(z − vBaseline)` stays roughly constant in magnitude rather than shrinking as the value head improves. Net effect: the policy gradient signal does not decay over training the way a fresh-baseline formulation would.

If you wanted the textbook formulation: drop `vBaseline` entirely and feed the trainer's `network.valueOutput` through a `stop_gradient` equivalent. MPSGraph doesn't have `stop_gradient`, but one workaround is to forward the value head twice and compute the policy-loss baseline from a `placeholder` that's fed the *just-computed* `v(s)` from the same training batch (one extra forward + one CPU readback per step). Or: accept the current design and document why the staleness is acceptable — currently the comment defends *why feeding via placeholder works*, not why a frozen baseline is the right algorithm.

### #6 — Missing repetition planes (draw-loop pathology)

The network has no input feature for "this position has occurred before." `BoardEncoder` sees only the current `GameState`, with no position history. Plane 17 (halfmove clock) partially covers the 50-move rule, but **threefold repetition is not represented at all**. Without repetition planes the network cannot in principle learn to play for or avoid 3-fold — which is one of the major sources of draw-loop pathology in early self-play (shuffle loops the value head can't even *see* are repeats).

Audit verified the existing castling encoding is correct and not "wasteful" — broadcasting a scalar bit across an 8×8 plane is the standard CNN-friendly way to feed a global feature, costs ~0.16% of network parameters, and matches how AlphaZero feeds castling, side-to-move, repetition, and total-move-count. Don't conflate that with the repetition gap; only the latter is a real bug.

**Design — 2 binary planes (AlphaZero-shape):**

```
Plane 18: 1.0 if current position has occurred ≥1 time before in this game
Plane 19: 1.0 if current position has occurred ≥2 times before  (next visit = 3-fold draw)
```

Both broadcast as 0/1 constants across all 64 squares, same idiom as castling.

Why two binary planes instead of one continuous `rep_count / 3` plane: the two cases are *qualitatively* different decisions, not points on a continuum. "Seen once" = mild signal we may be shuffling. "Seen twice" = one more visit *forces a draw by rule* — if I'm winning, avoid; if losing, seek. A single linear plane of `rep/3` requires the conv to learn its own threshold to recover those two regimes; two binary planes hand the thresholds in for free with independent weights. Cost is one extra 64 floats per position; immaterial.

Why not stack 8 historical board snapshots like real AlphaZero (112 history planes): we have no MCTS to amortize the 6× larger input over, en passant is already its own plane (16), and "how the position evolved" mostly helps a search engine. The 2-plane summary captures the rule itself, which is the actual gap.

**"Same position" semantics (FIDE 3-fold):** identical piece placement, side to move, all four castling rights, and en passant target. Standard Zobrist hashing mixes all four — one `UInt64` per ply suffices.

**Pruning trick:** any pawn move or capture makes earlier positions unreachable (irreversible). So the search window is bounded by `halfmoveClock` (≤100, usually <20). Maintain `[UInt64]` of Zobrist hashes per game; clear on every halfmove-clock reset. Repetition count = O(halfmove_clock) linear scan — trivially fast.

**Implementation work:**
- New per-game Zobrist hash list owned by `ChessGameEngine` (or the per-worker game wrapper). Updated incrementally on `applyMove`. Cleared on halfmove-clock reset.
- Extend `BoardEncoder.encode` signature to take `repetitionCount: Int` (saturates at 2). Caller computes from the hash list. Both `encode(_:into:)` and `encode(_:)` variants.
- Bump `BoardEncoder.tensorLength` 18 → 20. Use **always-fill** for the new planes: `fillPlane(base, plane: 18, value: repetitionCount >= 1 ? 1.0 : 0.0)` and the same for plane 19 with `>= 2`. Do NOT use the existing "skip if zero, rely on initial clear" pattern — it's correct today only because line 136 zeroes the whole `tensorLength` region, and that's an implicit cross-section dependency easily broken by a missed `tensorLength` bump in a future change. ~128 extra writes per encode (negligible against the 1152-float leading clear) buys self-contained per-plane correctness. Don't retrofit the existing planes — separate scope.
- Update `ChessNetwork` stem `inputChannels` 18 → 20.
- `ReplayBuffer` v3: store `repetitionCount` per slot (1 byte, saturated; or pack with another small field). Format-version bump in header.
- `MPSChessPlayer`, `BatchedMoveEvaluationSource`, `ChessTrainer` — plumb the count alongside `GameState` everywhere `BoardEncoder.encode` is called. Inference path computes from the live game's hash list; training path reads from replay-buffer slot.
- Input-shape change → `currentArchHash` bumps automatically (it mixes `inputChannels`) → existing `.dcmmodel` and `.dcmsession` files cleanly fail with `.archMismatch`. No migration code (per CLAUDE.md).
- Confirm `ModelCheckpointFile.maxTensorElementCount` still covers the largest tensor after the change (stem conv weights = `inputChannels × channels × 9 = 20 × 128 × 9 = 23,040`, much smaller than the policy FC, so no impact).
