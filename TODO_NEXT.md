# TODO_NEXT

Status: the v2 architecture refresh, its post-v2 follow-ups, and the
session durability hardening are all landed. They're recorded in
`dcm_architecture_v2.md` (under "Current state (as-built)") and
`ROADMAP.md` (under "Completed"). Nothing in-flight right now.

---

## Still open

_(No active items.)_

See `ROADMAP.md` for future-work entries (e.g., adaptive learn-rate
schedule) that aren't yet scoped as actionable TODOs.

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
