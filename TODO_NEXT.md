# TODO_NEXT

Status: most of the items originally listed here have been resolved by the
v2 architecture refresh (`dcm_architecture_v2.md`). What remains is the
durability fix and the ML/MPSGraph review's deferred items that were
explicitly NOT bundled with v2.

---

## Resolved by the v2 architecture refresh (2026-04-19/20)

The following items from prior audits are no longer actionable — the
underlying code has been replaced. See `dcm_architecture_v2.md` and
`CHANGELOG.md` for details.

- ~~**#1 — Promotion-target collapse in the 4096 policy.**~~ Resolved
  by the AZ-shape 76-channel policy head. Underpromotions get dedicated
  channels (64–72 for N/R/B × 3 directions); queen-promotion gets its
  own 3-channel block (73–75). `ChessMove.policyIndex` was deleted;
  every callsite now goes through `PolicyEncoding.policyIndex(_:currentPlayer:)`.
- ~~**#5 follow-up — Make `maxTensorElementCount` track the live arch.**~~
  Resolved. `ModelCheckpointFile.maxTensorElementCount` is now a
  computed property derived from `ChessNetwork.channels`, `inputPlanes`,
  `policyChannels`, and `seReductionRatio`. Auto-updates on any
  architecture change.
- ~~**ML Review #2 — Policy head has no spatial structure.**~~
  Resolved. Old FC head replaced with a fully-convolutional 1×1 conv
  128→76. Translation equivariance preserved end-to-end. Head
  parameter count dropped ~50× (~9.8K vs ~528K).
- ~~**ML Review #3 — Move encoding silently collapses underpromotion.**~~
  Same as #1 above; resolved.
- ~~**ML Review #4 — `vBaseline` is a frozen baseline, not the current
  value-head estimate.**~~ Resolved (post-v2 follow-up). The trainer
  now runs an extra forward-only pass on its current network before
  each training step to compute fresh per-position v(s), and
  overwrites the play-time vBaseline staging with those values before
  feeding the training graph. The `vBaseline` placeholder boundary
  already provides stop-gradient semantics; only the source of values
  changed. Empirically verified (`MPSGraphGradientSemanticsTests`)
  that MPSGraph has no `stop_gradient` op and that `with`-array
  exclusion does not prune backward-pass paths, so this placeholder-
  feed approach is the only correct way. Cost ~33% extra forward
  FLOPs per training step. Diagnostic `vBaselineDelta` now in
  `[STATS]` line.
- ~~**ML Review #6 — Missing repetition planes.**~~ Resolved. Input
  tensor expanded from 18 → 20 planes; planes 18 (≥1× before) and 19
  (≥2× before) feed the network the threefold-repetition signal.
  Implementation reuses the engine's existing `positionCounts`
  table — no new Zobrist machinery was needed (see
  `dcm_architecture_v2.md` Phase 1 for the deviation rationale).

---

## Still open

### #3 — ReplayBuffer durability: fsync + length invariant *(partial-write protection)*

`ReplayBuffer.write(to:)` (`ReplayBuffer.swift:~316-445`) does not call
`handle.synchronize()` before close; `restore(from:)` reads chunks
without verifying total file size matches the header-predicted length.
A crash or disk-full mid-write produces a truncated file that loads as
a partial buffer silently. The v2 architecture refresh deliberately
did NOT bundle this fix to keep the diff focused. Now is the time.

**Fix** (do not bump version — this is a robustness fix, not a format
change):

- `ReplayBuffer.swift:~375` — add `try handle.synchronize()` before
  the `defer { try? handle.close() }`. Forces APFS to flush dirty
  pages to disk before the file handle closes.
- `ReplayBuffer.swift:~499-644` (restore) — compute expected file size
  from header fields; `guard actualSize >= expectedBytes else { throw
  PersistenceError.truncatedFile }`. Catches the torn-write case.
- `CheckpointManager.saveSession` (`CheckpointManager.swift:~318-326`)
  — write the replay buffer into the tmp directory *before* the JSON
  so the tmp→rename atomic-move covers all three files as one unit.

Skip SHA-256 / Adler-32 — too much engineering for a user-local file
whose cost-of-loss is ~5 minutes of refill. Revisit if telemetry shows
silent corruption.

Estimated: 30 min, touches ReplayBuffer + CheckpointManager.

### Adaptive learning-rate schedule

Currently `learnRate` is a static hyperparameter (default 5e-5) that
the user adjusts manually via the UI. A schedule would let the system
self-tune based on training health. Five candidate trigger families
recorded in `ROADMAP.md` (step decay, plateau detection, promotion-
driven, cosine annealing, replay-ratio aware). No design picked yet.

---

## Audit findings — verified safe (don't re-open without new evidence)

These came up during the original audit and were verified to not be
bugs. Recording them here so the next pass doesn't waste time
re-investigating:

- **Castling out of check**: `MoveGenerator.swift:311` already asserts
  `!isSquareAttacked(state, row: homeRow, col: 4, by: color.opposite)`.
  Safe.
- **`try? await Task.sleep` in polling loops**: all 9 sites reviewed;
  each sits in a loop whose condition re-checks `Task.isCancelled` (or
  returns) after the sleep. No busy-spin hazard.
- **Deprecated 1-arg `.onChange`**: no 1-arg usages in the codebase;
  the 2-arg form `{ _, _ in ... }` is the current non-deprecated API.
- **Softmax overflow at low tau**: `MPSChessPlayer.sampleMove`
  subtracts `maxLogit` before `expf`, and `floorTau > 0` is enforced
  by precondition on `SamplingSchedule`.

(Note: the audit's "ChessRunner flip inversion: correctly un-flips
rows" item is no longer accurate — `ChessRunner.extractTopMoves` was
rewritten to use `PolicyEncoding.decode`/`geometricDecode` and no
longer does manual row-flipping. Current implementation is correct;
the original observation just no longer applies to the same code.)
