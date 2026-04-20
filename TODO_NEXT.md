# TODO_NEXT

Status: the v2 architecture refresh and its post-v2 follow-ups are
recorded in `dcm_architecture_v2.md` (under "Current state
(as-built)") and `ROADMAP.md` (under "Completed"). What remains is
the durability fix and the deferred ML-review items that were
explicitly NOT bundled with v2.

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
