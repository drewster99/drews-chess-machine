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

## In-flight from last session (pending verify/commit)

The window-close teardown from commit `6c96437` hooks `NSWindow.willCloseNotification` without filtering by window. That notification is global and also fires for the Log Analysis auxiliary window (`LogAnalysisWindowController`, `LogAnalysisWindow.swift:32`) and for any `NSOpenPanel` / `NSSavePanel` raised by the File menu (Save Session, Load Session…, Load Model…, Open Data Folder in Finder). As-shipped, dismissing any of those dialogs will cancel the training run mid-save.

Fix is written but not yet verified:
- Added a `WindowAccessor: NSViewRepresentable` that captures the hosting `NSWindow` into a `@State var contentWindow: NSWindow?`.
- Body now calls `.background(WindowAccessor(window: $contentWindow))`.
- The `.onReceive(...)` handler filters `note.object === contentWindow` before calling `stopAnyContinuous()` + `clearTrainingAlarm()`.

Pending: rerun `mcp__xcode-mcp-server__build_project` (last attempt hit an Apple-Event scheme-action error unrelated to the code), smoke-test (open Log Analysis window, close it, confirm training continues; raise a Save panel, cancel, confirm training continues; close the main window, confirm training stops), then commit.

Critical files still modified / uncommitted from that work:
- `DrewsChessMachine/DrewsChessMachine/ContentView.swift` — WindowAccessor + contentWindow state + filtered onReceive.

Also note: the user's AdamW `shouldDecay` refactor in `ChessNetwork.swift` was in flight at the same time. That file is modified but intentionally excluded from `6c96437` and is the user's work — don't commit it without asking.

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

### #6 — Castling-rights planes waste 4× more space than needed

`BoardEncoder.encode` fills 4 entire 8×8 planes (256 floats) to encode 4 boolean bits. This is "broadcast a scalar over a spatial plane" which works but inflates input tensor by ~22%. For a network this small the cost is in the noise, but as a model-design question it's also denying the input-stem the chance to keep castling-rights-as-scalar features distinct from spatial features. More impactful: there's **no repetition-count plane** and **no fullmove counter**. Without repetition planes the network cannot in principle learn to play for / avoid threefold repetition, which is one of the major sources of draw-loop pathology in early self-play. The halfmove-clock plane (17) covers 50-move-rule but not 3-fold.

A common compact variant: planes 12 (kingside-castle bool for me), 13 (queenside-castle bool for me), 14 (kingside opp), 15 (queenside opp), plus 2 repetition planes (1 if position has occurred 1× before, 1 if 2× before). Same plane count, much more useful.

(Note: this is an input-shape change. Existing checkpoints become incompatible — needs a `currentArchHash` bump in `ModelCheckpointFile`. Also requires plumbing position-history tracking into `ChessGameEngine` / `BoardEncoder` since the encoder currently sees only the current `GameState` with no history.)
