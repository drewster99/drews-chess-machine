# Self-play game corpus — design plan

Status: **TABLED — deferred.** Captured for later; not the current focus (that's
`FULL10PLY200_PLAN.md`). This feature is independent of full10ply200 — full10ply200 sources
its history from the live `ChessGameEngine` window, not from this corpus — so tabling B
does not block A. The emit-completeness regression test (invariants 1–5) is specced here
and travels with this workstream. Branch: `safetensors-storage`. Created 2026-06-06.

## Goal

Persist self-play output as an immutable, **encoding-agnostic** corpus of complete game
trajectories, so it can be re-streamed into the trainer under *any* input encoding and
*any* value-head target. This decouples the expensive part (GPU self-play) from the cheap
part (CPU re-encoding), and makes architecture comparisons controlled — the same games
train every candidate.

## Why raw games, not encoded batches or encoded positions

- **Encodings vary** (basic20 / basic30 / full10ply200). A stored *encoded* tensor is
  frozen to one encoding; a stored *game* re-encodes to any. Lc0 stores encoded positions
  and consequently must regenerate data when its input format changes — we avoid that by
  storing one level rawer.
- **Value targets vary** (`scalar_tanh` MSE vs `wdl_softmax` CE), but both derive from the
  same raw per-ply outcome — store the outcome once, compute either target at train time.
- **full10ply200 requires trajectories** — its 10-frame history stack can only be rebuilt
  from the ordered positions of a game, never from shuffled isolated positions.
- Reusing a frozen corpus across nets is **off-policy / fixed-dataset** training (a
  different regime from live self-play, and the right controlled setup for comparing
  architectures on identical data).

## Decision

Keep the **raw stream of emitted games**, captured at the `GameState`/move level
**upstream of encoding** — *not* the encoded tensor that currently flows into the replay
buffer (saving that would re-freeze the encoding). Games are stored as complete, ordered
units.

## Per-ply record (architecture-independent)

- **Played move** — the `ChessMove` (policy target). Prefer the move over the derived
  `policyIndex` (rawer; survives any `PolicyEncoding` change — though the 76×64 space is
  fixed-by-engine today).
- **Outcome** — raw per-ply result from side-to-move's perspective, z ∈ {−1,0,+1}; both
  value heads derive their target from it.
- **Behavior-policy probability of the played move** — the post-mask, post-temperature
  softmax probability the *generating* net assigned to the move it actually played. A
  single float; encoding-independent (it's a probability tied to the played move, not to
  the input representation). Already computed by the sampler at emit time.
- **Value-head scalar** — the generating net's perspective-adjusted value at that
  position, v ∈ [−1,+1] (for `wdl_softmax`, v = p_win − p_loss; for `scalar_tanh`, the
  tanh scalar). Head-agnostic single float; available at eval time.
- **Side to move**, **ply index**, **sampling tau**, **material count**, **game length**,
  **worker/game id** — metadata (ply + game id let you regroup/order even a flat stream).

The behavior-policy probability and value scalar are *generation-time* data (what the
champion net believed), distinct from the *targets* (played move, outcome). They are
captured now — cheap (~2 floats/ply) and impossible to backfill once the generating net is
replaced — and unlock principled off-policy correction (importance sampling / V-trace) and
TD/bootstrapped value targets later. See "Consumption deferred" below.

## Derived at stream time — NOT stored

Encoded board tensor (per target encoding), state hash (recompute, or move it to a
`PositionKey` hash so it's encoding-stable), the value-target representation, and the
full10ply200 history stack. Re-encoding regenerates all of these; the engine regenerates
repetition masks and the history window when moves are replayed.

## Corpus stream format (versioned)

The stream is **self-describing and versioned** so fields can be added later without
invalidating existing corpus files (the way Lc0 evolved chunk formats v3→v6):

- **Stream header** with a `format_version` (start at 1) plus engine-identity pins a
  reader can check for compatibility: `PolicyEncoding` version (we store moves and derive
  policy indices), board/policy dims (fixed-by-engine today, but recorded so a future
  reader detects mismatch), and creation timestamp.
- **Per-game header**: outcome, game length, worker/game id, and the **generating
  champion `ModelID`** (provenance — needed to reason about off-policy distance/staleness
  once the behavior data is consumed; reconstructable only fuzzily from timestamps
  otherwise).
- **Per-ply records**: the fields listed above, including the behavior-policy probability
  and value scalar from v1 onward.

A reader keys off `format_version` to know which fields are present. A tiny round-trip
test should assert the header version is written and read back, and that an unknown/newer
version is rejected rather than silently misparsed.

**Replay buffer is unchanged today.** The behavior-policy probability and value scalar
live only in the stream format; the current re-stream path can ignore them. No new buffer
columns now — when off-policy / TD training is actually built, the buffer and trainer get
extended to carry/use them, and the data is already in the corpus (no regeneration).

## Structural requirement

The corpus must preserve **complete, ordered games** — no shuffling, no dropped plies, no
partial-game flushes. full10ply200 history reconstruction and the controlled-comparison
property both depend on it.

## Emit-completeness regression test (build now, independent of the corpus)

Guards the emit boundary against future changes that would silently break reuse. Asserts,
for an emitted game (regrouped by `workerGameId`, ordered by `plyIndex` — independent of
the per-side internal storage layout):

1. **Starts at the start** — ply 0 is the standard starting position.
2. **Contiguous & ordered** — ply indices are exactly `0,1,…,N−1`; none missing/repeated.
3. **Correct alternation** — side-to-move flips white/black in lockstep with ply parity.
4. **Length agreement** — broadcast `gameLength` == N == actual emitted position count.
5. **Valid termination** — final position is a true terminal (mate / stalemate /
   insufficient material / 50-move / threefold) **or** the configured self-play ply cap,
   and the broadcast `outcome` is consistent and identical across the game's positions.

Approach: drive a deterministic legal game — **Fool's Mate** (`1. f3 e5 2. g4 Qh4#`,
4 plies, real checkmate, White loses) — through the actual `ActiveGame.recordPly` →
`flush` machinery into a real `ReplayBuffer`, with dummy encoded boards (the test checks
structure/metadata, not tensor content). Metal-free, in `DrewsChessMachineTests`.

Stronger follow-up (once the corpus lands): a **reconstruction round-trip** — decode each
stored move, replay from the start through `ChessGameEngine`, and assert the trajectory is
legal, complete, and reproduces the emitted sequence bit-exact (matching the suite's
existing bit-exact round-trip style). Re-assert invariants 1–5 against the *corpus* emit,
not just the buffer.

## Consumption deferred (data captured now)

The behavior-policy probability, value scalar, and generating `ModelID` are **captured in
the v1 format now** (decided), because they're cheap and impossible to backfill. What's
deferred is *using* them — the current supervised regime (hard move targets + Monte-Carlo
outcomes) ignores them. When wanted, off-policy correction (importance sampling / V-trace)
and TD/bootstrapped value targets are built by extending the buffer + trainer to carry and
consume these fields; no self-play regeneration needed since the corpus already has them.

MCTS visit-count policy targets would require richer emit still (a full distribution per
ply, not the single played move + its probability), but search is an explicit non-goal —
noted and dismissed.
