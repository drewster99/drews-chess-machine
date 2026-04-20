# TODO_NEXT

Status: the v2 architecture refresh and its post-v2 follow-ups are
recorded in `dcm_architecture_v2.md` (under "Current state
(as-built)") and `ROADMAP.md` (under "Completed"). What remains is
the durability fix and the deferred ML-review items that were
explicitly NOT bundled with v2.

---

## Still open

### #3 — Session durability hardening: "saved means golden"

**Principle.** If any piece of a session save cannot be fully verified
end-to-end, the whole save fails, all partials are removed, and no
final-named artifact appears on disk. Restored sessions are guaranteed
to be bit-identical to the state that was saved — or they fail to
load with a precise error.

Reference design docs:
- `dcmmodel_file_format.md` — `.dcmmodel` binary spec (already has
  SHA-256 trailer, archHash, forward-pass round-trip verification).
- `replay_buffer_file_format.md` — v3 current, v4 planned (this task).

#### Part A — ReplayBuffer v4 format

Bump `ReplayBuffer.fileVersion` 3 → 4. Readers reject v1/v2/v3 cleanly
with `PersistenceError.unsupportedVersion`. No migration path (matches
the project-wide delete-and-retrain stance).

- **SHA-256 trailer.** Append 32-byte SHA-256 digest computed over
  every preceding byte (header + all four ring sections). Verified
  on load before any count or array is trusted. Matches
  `.dcmmodel` integrity-trailer convention.
- **Strict total-file-size equality check.** In `restore(from:)`,
  compute `expectedBytes = headerSize + storedCount × (floatsPerBoard
  × 4 + 12) + 32` and require `actualFileSize == expectedBytes`.
  Throws a new `PersistenceError.sizeMismatch(expected, got)`. Uses
  equality, not `>=`, because the format is fully deterministic —
  any deviation is corruption.
- **Upper-bound sanity caps** before allocation:
  - `capacity <= 10_000_000` (positions)
  - `storedCount <= 10_000_000`
  - `floatsPerBoard <= 8_192`
  - `writeIndex < capacity` (already present)
  Throw `.invalidCounts` on violation. Catches corrupted headers that
  hash-match by astronomical coincidence before they drive a massive
  allocation.
- **`synchronize()` on write.** Add `try handle.synchronize()` just
  before the `defer { try? handle.close() }` in `_writeLocked`.
  Forces APFS to flush dirty pages before the file handle closes.
  Optionally upgrade to `fcntl(fd, F_FULLFSYNC)` for platter-level
  durability; start with `synchronize()` and evaluate.
- **v4 doc** — add a v4 section to `replay_buffer_file_format.md`
  mirroring the existing v3 section with the SHA trailer and new
  size/cap invariants spelled out.

New `ReplayBuffer` tests to add to the XCTest target:
- `testV4SHAMismatchRejected` — handcraft a v4 file with a tampered
  byte in the body; verify decode throws `.hashMismatch`.
- `testV4SizeMismatchRejected` — truncate the last byte of a valid
  v4 file; verify decode throws `.sizeMismatch`.
- `testV4TrailingGarbageRejected` — append an extra byte to a valid
  v4 file; verify decode throws `.sizeMismatch`.
- `testV4RejectsV3` — synthesize a valid-looking v3 file; verify
  decode throws `.unsupportedVersion`.
- `testV4UpperBoundRejected` — handcraft a header with
  `capacity = Int64.max`; verify decode throws `.invalidCounts`
  (not an allocation crash).
- Update existing `testV3RejectsV2File` to `testV4RejectsV2File`
  (both are `.unsupportedVersion` cases).

#### Part B — `CheckpointManager.saveSession` atomicity

- **Write-all-to-tmpdir-then-verify-then-rename** is the existing
  pattern; expand it to cover the replay buffer on an equal footing
  with the two `.dcmmodel` files.
- **Per-file fsync inside tmpdir.** After each `Data.write(...,
  options: [.atomic])` and after `ReplayBuffer.write(...)`, open
  each file with a `FileHandle`, call `synchronize()`, close.
  Guarantees the file contents are on stable storage before the dir
  rename commits.
- **Tmpdir fsync.** Before `fm.moveItem(tmpDir, finalDir)`, open the
  tmp directory with `open(path, O_RDONLY)` and call `fcntl(fd,
  F_FULLFSYNC)` (closer to "platter flush" than plain `fsync` on
  Apple filesystems), then close the fd. Forces directory entries
  for all four files to stable storage before the rename commits.
- **Parent dir fsync after rename.** Same `F_FULLFSYNC` treatment on
  `CheckpointPaths.sessionsDir` after the `moveItem` succeeds — the
  rename itself is a directory-entry change in the parent, and it
  needs to hit stable storage too.
- **Expanded verification phase.** After writing everything to the
  tmpdir, before rename:
  - `.dcmmodel` bit-exact + forward-pass round-trip (already in
    place — no change).
  - `session.json` re-decode round-trip (already in place — no
    change).
  - `.replay_buffer.bin` re-load into a scratch `ReplayBuffer` via
    `restore(from:)` (which now includes SHA + size verification as
    part of Part A). Spot-check: compare the first and last stored
    board's floats, plus the same positions' `moves`/`outcomes`/
    `vBaselines`, against the live in-memory ring. Catches any
    write-path regression that hashes correctly but writes wrong
    bytes.
- **Cross-reference check on load.** In `CheckpointManager.loadSession`,
  after decoding both the replay buffer (via its own SHA verify) and
  `session.json`, cross-check that
  `session.state.replayBufferStoredCount ==
  loadedBuffer.storedCount`, same for `capacity` and
  `totalPositionsAdded`. Mismatch → throw a new
  `CheckpointManagerError.sessionReplayMismatch` that surfaces in
  the load UI. Catches buffer/session swaps and residual corruption
  not caught by SHA.
- **`fm.moveItem` overwrite safety.** `FileManager.moveItem(at:to:)`
  documents that it aborts if the destination exists (unlike POSIX
  `mv`); plus `saveSession` already has an explicit
  `fileExists(atPath: finalDirURL.path)` guard that throws
  `targetAlreadyExists`. Already safe — no change.

#### Part C — Launch-time orphan sweep

On app start, before any save/load UI becomes available, sweep for
crashed-mid-save debris:

- In `CheckpointPaths.sessionsDir`: remove any entry whose name ends
  in `.tmp` (that's the `tmpDirURL` suffix used by `saveSession`).
- In `CheckpointPaths.modelsDir`: remove any entry whose name ends
  in `.dcmmodel.tmp` (matches `saveModel`'s tmpURL pattern).
- Log each removal via `[CLEANUP] removed orphan: <name>`. Collect
  exceptions and log `[CLEANUP-ERR]` without aborting launch.

Runs once at startup from `DrewsChessMachineApp.init` (or a new
`CheckpointPaths.cleanupOrphans()` helper invoked there). Cheap —
the directories usually have zero orphans.

#### Part D — Documentation

- Add a "v4" section to `replay_buffer_file_format.md` mirroring the
  current v3 section with the trailer and new invariants.
- No changes needed to `dcmmodel_file_format.md` — `.dcmmodel` is
  already at the durability bar this plan targets.
- Add a `CHANGELOG.md` entry at the top summarizing the bundle.

#### Scope limits

- **No checksum per record.** File-level SHA-256 is sufficient.
- **No compression.** Writes stay raw-float.
- **No replay-buffer migration.** v3 files stop loading, period.
- **Session-state JSON back-compat preserved.** Adding cross-check
  reads to existing Optional fields; no schema change.

#### Implementation order (one branch, one bundle)

1. Extend `replay_buffer_file_format.md` with a v4 section.
2. Add SHA-256 trailer write path and size/cap invariants to
   `ReplayBuffer._writeLocked` + `restore(from:)`. Bump
   `fileVersion` to 4. Add new `PersistenceError` cases.
3. Add `synchronize()` to `ReplayBuffer._writeLocked`.
4. Add per-file fsync helper + tmpdir fsync + parent-dir fsync to
   `CheckpointManager.saveSession`.
5. Add replay-buffer re-load + spot-check to `saveSession` verify
   phase.
6. Add session-state cross-check to `CheckpointManager.loadSession`.
   Add `CheckpointManagerError.sessionReplayMismatch`.
7. Add `CheckpointPaths.cleanupOrphans()` + launch-time invocation.
8. Add ReplayBuffer XCTests listed in Part A.
9. `CHANGELOG.md` entry.
10. Build, run Engine Diagnostics, run the XCTest suite, manually
    save + quit + relaunch + load to validate end-to-end.

Estimated: 3-4 hours focused work.

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
