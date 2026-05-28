# TODO Next

Working list of pending items, ordered. Top entries are the most
immediate; lower entries are deferred items captured during prior
sessions that we agreed to revisit but did not implement at the
time.

---

## Discussed but not implemented (deferred work)

These were items raised during the 2026-05-05 momentum / decoupled-
decay / popover-refactor sessions that we agreed to capture rather
than implement in the moment.

### Cosine LR annealing post-warmup

Skipped per explicit user direction. Original recommendation: add
`cosineDecaySteps: Int` (default 0 = disabled) and
`cosineFloorRatio: Double` (default 0.1) training parameters;
in `buildFeeds` after the existing warmup multiplier, apply
`cosineMul = floorRatio + (1−floorRatio) · 0.5 · (1 + cos(π · min(t/T, 1)))`
when `cosineDecaySteps > 0`. Compose multiplicatively with warmup
and sqrt-batch multipliers (which already compose the same way).
Save both new params in `SessionCheckpointState` for resume
symmetry.

Rationale to revisit: every modern self-play system uses some form
of post-warmup decay (lc0 cosine, AlphaZero stepwise, Stockfish
NNUE plateau). The current setup goes from warmup to flat-LR-
forever, which is widely cited as a contributor to late-stage
instability — as the loss landscape's curvature shallows, the same
LR that was healthy early becomes excessive. Cosine schedule has
the right boundary derivatives (zero at `t=0` for smooth handoff
from warmup, zero at `t=T` for no LR-cliff event).

### Per-tensor gradient norm readback

A more expensive option to identify exactly which tensor is
dragging the global `gNorm` up when the policy head is suspected
but not confirmed. Would require ~92 additional `reduceSum`-of-
squares targetTensors in the training graph (one per trainable
variable), each producing a scalar readback. The indirect signals
available now (`pwNorm` on the chart + `pLogit |max|` on the
`[STATS]` line) are sufficient to suspect the policy head as the
dominant contributor. A per-tensor breakdown would be definitive
if the policy-head theory turns out wrong, or if a future
architectural change introduces a non-policy-head hot spot.

Cost: ~92 extra GPU scalar reductions per training step (small,
< 1 ms on Apple Silicon at the current architecture), plus 92
extra readback paths through `runPreparedStep`'s scratch buffer.
Implementation pattern would mirror the existing single-tensor
`policyHeadWeightNormTensor` readback. Not a difficult change;
just a deferred priority.

### Migrating away from coupled-decay-saved velocity buffers

A v2 `trainer.dcmmodel` written under the *old* coupled-decay
formula carries baked-in decay terms in the saved velocity
buffer. When loaded under the new decoupled-decay formula, those
baked-in terms wash out gradually as μ-weighted decay over
~`ln(0.01)/ln(μ)` steps (~100 steps at μ=0.95). This is a
transient, not a correctness break, and only matters if the user
had a session saved with high μ under the old code. The default
was μ=0 so most saves have zero velocity and are unaffected.

If perfect reproducibility of saved-state load were required, a
v3 trainer.dcmmodel format bump that signals the velocity's
optimizer formulation would be the rigorous answer. Not worth
doing speculatively — file for possible future implementation if
saved-session forensics ever require it.

### Weight EMA / Stochastic Weight Averaging (SWA) for inference

Discussed as a separate-scope follow-up. Currently the project's
only "stable inference network" mechanism is arena promotion;
during a Play-and-Train run the inference network swaps to the
new champion only at promotion boundaries.

lc0 maintains a Polyak EMA of recent weights and uses *that* for
inference (not the latest training weights). The effect: smoother
strength curve, smaller arena variance, fewer "unlucky" arenas
where a transient gradient spike lands on the snapshot.
Functionally an alternative to (or complement to) frequent
promotion.

Would interact with arena/promotion semantics — needs its own
design discussion before implementation. Probably belongs as a
ROADMAP entry once the design is clearer; flagging here so it's
not forgotten.

### Mixed precision (float16 / bfloat16)

`ChessNetwork.dataType` is currently `.float32` with a comment
saying "Switching this between `.float32` and `.float16` should
Just Work" but several call sites have `fatalError("only .float32
is currently supported")` guards. Concrete blocker is the trainer's
`buildFeeds` host-side path that does `writeBytes` of `[Float]`
buffers directly into NDArrays — float16 would need a reused
`[UInt16]` scratch and Accelerate downconversion.

Productivity gain on Apple Silicon is modest (~1.5–2× throughput
on smaller batches; bottleneck is more likely self-play GPU
saturation than trainer GPU saturation at the current scale). Not
asked for; flagging for possible future work.

### Horizontal-mirror data augmentation

Observation (2026-05-27, Network Weight Analyzer): the policy head's
queen-style distance-1 channels show a meaningful east-vs-west
asymmetry: chan 14 (queen E d1) has L2 = 1.341 vs chan 42 (queen W
d1) at L2 = 1.130 — a 19% gap. Other mirror-pair gaps (NE/NW,
SE/SW, the knight pairs) are within ~5%. Forward/backward asymmetry
(N/NE/NW favored over S/SE/SW) is large but expected (forward
moves matter more in chess); the lateral E-vs-W gap is harder to
justify from chess principles and likely partly training noise +
partly accumulated bias from however many self-play games have
landed in the buffer.

**Color is already normalized in the replay buffer.** `BoardEncoder`
.encode` always stores the position from the *mover's* perspective:
piece planes 0–5 hold the mover's pieces, planes 6–11 hold the
opponent's, and the board is vertically flipped when the original
side-to-move was black so the mover always "sits at rows 6–7."
The policy index encoding is also in the mover frame
(`PolicyEncoding.encode` does its own black→white row flip).
**So at sample time the trainer never has to ask "was this from
white or black?"** — every buffer position looks like a white-to-move
encoded board, and the stored move index already lives in the same
frame. The mirror augmentation below operates on this already-
color-normalized representation, so it only needs to worry about
the genuine left-vs-right symmetry of chess.

Chess is *almost* horizontally symmetric — piece moves, captures,
en passant, and promotions all are. The two things that break
left-right symmetry are:
  1. Castling: kingside castle is on the h-file, queenside on the
     a-file. Horizontal mirror swaps them (kingside ↔ queenside).
     Castling moves themselves (e1→g1 kingside, e1→c1 queenside)
     are not simple mirror images — mirroring e1→g1 gives e1→b1,
     which isn't a legal castle.
  2. Anything else tied to specific files (very little in chess).

Two implementation paths:

(A) Mirror only when both castling-right planes are zero on both
    sides — i.e. positions where castling rights are already gone.
    Endgames qualify. K+Q-vs-K, K+R-vs-K, etc. all qualify (no
    castling rights left). The buffer's bucket 0–4 and bucket 5–8
    (sparse) would mostly qualify; the K+Q-vs-K class specifically
    is the one we're starving for data on.

    Simplest path: at buffer-append time, with probability 0.5,
    horizontally flip the board planes (mirror columns within each
    of planes 0–11), the en-passant plane (16), and the repetition-
    history planes (20–29), AND mirror the stored move's column
    coordinate. Castling-rights planes (12–15) must be all zero,
    or skip the flip. No move-index re-encoding needed for queen-
    style and knight channels since the direction encoding has
    full N/NE/E/SE/S/SW/W/NW + 8 knight jumps — mirroring just
    swaps E↔W, NE↔NW, SE↔SW, knight UR↔UL, RU↔LU, RD↔LD, DR↔DL.

(B) Mirror all positions, including those with castling rights.
    Mirror swaps kingside ↔ queenside castling rights (planes
    12 ↔ 13 for mover side; 14 ↔ 15 for opponent side). Castle-
    move encodings need careful translation. More general but
    more code to write and test.

Rationale to revisit: (A) is cheap and would 2× the effective
training data on the very bucket 0–4 / bucket 5–8 positions where
the network's K+Q-vs-K conversion rate is currently the bottleneck
(1.5% conversion, ~459 K+Q-vs-K games in a 1M-position buffer).
Doubling that to ~900 — and presenting both orientations of each
position — should accelerate the trunk's learning to recognize
K+Q-vs-K material configurations regardless of which side of the
board the king is sitting on. (B) is a strictly larger benefit
but with strictly more implementation work; do (A) first.

Validation: after enabling, check the policy head's E-vs-W channel
L2 ratios (Analyze Network Weights menu item). The 19% gap should
shrink substantially within tens of arenas. Also watch the K+Q
mate tactical probe rank — should improve faster than under no-
augmentation training as the trunk receives more K+Q-vs-K experience
per buffer refresh.

---

## How this file works

Top of file = next action. Lower entries = deferred capture, in
roughly the order they were raised. When an item is taken on,
move it to ROADMAP.md (if it warrants a permanent record) or
CHANGELOG.md (when it lands), and remove it from here. When new
items emerge, append to the appropriate section.
