# Self-play game corpus — design plan

Status: **PLANNED, not started.** Branch: `safetensors-storage`. Created 2026-06-06.

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
- **Side to move**, **ply index**, **sampling tau**, **material count**, **game length**,
  **worker/game id** — metadata (ply + game id let you regroup/order even a flat stream).

## Derived at stream time — NOT stored

Encoded board tensor (per target encoding), state hash (recompute, or move it to a
`PositionKey` hash so it's encoding-stable), the value-target representation, and the
full10ply200 history stack. Re-encoding regenerates all of these; the engine regenerates
repetition masks and the history window when moves are replayed.

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

## Later / deferred — off-policy hedge fields

Not needed for the current supervised regime (hard move targets + Monte-Carlo outcomes),
but **cheap now and impossible to backfill** (the generating net is gone once it's
replaced). Decide whether to add before the corpus format is frozen:

- **Behavior-policy probability of the played move** per ply (the post-mask,
  post-temperature softmax prob the *generating* net assigned), optionally the net's
  **value scalar** at that ply — enables principled off-policy correction (importance
  sampling / V-trace) and TD/bootstrapped value targets if ever wanted. ~1–2 floats/ply.
- **Generating champion `ModelID`** per game — provenance for off-policy distance /
  staleness reasoning and experiment analysis (reconstructable only fuzzily from
  timestamps otherwise).

MCTS visit-count policy targets would also require richer emit, but search is an explicit
non-goal — noted and dismissed.
