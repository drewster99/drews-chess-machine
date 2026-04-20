# Replay Buffer File Format (`.replay_buffer.bin`)

Binary serialization format for the self-play `ReplayBuffer` — the
ring of `(board, move, outcome, vBaseline)` tuples that the trainer
samples minibatches from. Produced by `ReplayBuffer.write(to:)` and
consumed by `ReplayBuffer.restore(from:)` in
`DrewsChessMachine/DrewsChessMachine/ReplayBuffer.swift`.

A replay buffer file lives inside a `.dcmsession` directory next to
`champion.dcmmodel`, `trainer.dcmmodel`, and `session.json`. It is
never used outside that session-bundle context; `session.json`'s
`hasReplayBuffer` flag controls whether the file is expected to be
present.

## Design goals

- **Byte-exact round-trip.** Save-then-load must reproduce every
  stored position bit-for-bit — identical board floats, identical
  policy-index moves, identical outcomes, identical vBaselines.
- **Stride matches the network input exactly.** Each saved board is
  `BoardEncoder.tensorLength` floats — the exact shape the live
  `ChessNetwork` consumes. No re-encoding, no separate repetition-
  count field; repetition planes are part of the stored board.
- **Fail loudly on architecture change.** A file minted against one
  board-plane shape (e.g., pre-repetition-planes 18-plane encoding)
  refuses to load into a build with a different shape. `archHash`-
  style identity — via `floatsPerBoard` check at decode time.
- **Four parallel arrays, oldest-first.** On-disk order is the
  chronological ring order, not the in-memory physical-slot order.
  Readers do not need to know the pre-save `writeIndex` to reconstruct
  the ring — they just append the file's entries in order.
- **No in-place editing.** Every save is a fresh file inside a fresh
  `.dcmsession` directory, alongside the two model files and the JSON
  state blob.

## Historical format: v3

v3 was the format produced by builds from the arch-refresh
(2026-04-19/20) through the durability-hardening bundle
(2026-04-20). It has since been superseded by v4 (see below). v3
files do not load into v4 readers — rejected cleanly with
`PersistenceError.unsupportedVersion`. No migration path; older files
do not load. Preserved here for archaeological reference.

v3 had all of v4's header structure and body layout, but **no
SHA-256 trailer**, **no strict file-size equality check**, and **no
upper-bound caps** on counter fields. The writer also did not call
`synchronize()` before close, so a crash between write and flush
could leave a torn file that the reader silently treated as valid.

### On-disk layout (v3)

All multi-byte integers are little-endian. All floats are IEEE-754
binary32 in native (little-endian) byte order. The file is
conceptually: one 56-byte header, then four parallel arrays each
`storedCount` entries long, in the declared order.

```
Offset  Size              Field                   Type       Notes
------  ----------------  ---------------------   ---------  ------------------------------------
  0        8              magic                  [UInt8]    ASCII "DCMRPBUF"
  8        4              fileVersion            UInt32     Currently 3
 12        4              pad                    UInt32     Written as 0, ignored on read
 16        8              floatsPerBoard         Int64      Must equal BoardEncoder.tensorLength (1280 today)
 24        8              capacity               Int64      Ring capacity at save time (positions)
 32        8              storedCount            Int64      Number of positions actually saved
 40        8              writeIndex             Int64      Ring write cursor at save time (< capacity)
 48        8              totalPositionsAdded    Int64      Lifetime counter (used by replay-ratio controller)

  --- boards: storedCount entries, oldest-first ---
 56        S × F × 4      boards                 Float32    S = storedCount, F = floatsPerBoard

  --- moves: storedCount entries, oldest-first ---
  ..       S × 4          moves                  Int32      Policy index per position

  --- outcomes: storedCount entries, oldest-first ---
  ..       S × 4          outcomes               Float32    Game outcome z ∈ {-1, 0, +1} from side-to-move POV

  --- vBaselines: storedCount entries, oldest-first ---
  ..       S × 4          vBaselines             Float32    Frozen v(s) at play time

  --- end of file (no trailer, no hash) ---
```

Where `S = storedCount` and `F = floatsPerBoard = 1280`.

**Total file size** (v3, no trailer):

```
totalBytes = 56 + storedCount × (floatsPerBoard × 4 + 4 + 4 + 4)
           = 56 + storedCount × (floatsPerBoard × 4 + 12)
```

For a full 1 M-position ring at the current arch: 56 + 1,000,000 ×
(1280 × 4 + 12) ≈ 5.13 GB.

### Header fields (v3)

#### `magic` — 8 bytes

The literal ASCII string `"DCMRPBUF"`. First check the decoder
performs. Catches accidental feeding of a wrong-format file (e.g., a
`.dcmmodel`, a truncated archive, a PNG) before any numeric field is
parsed.

#### `fileVersion` — UInt32

Format revision identifier. Currently **3**. If the on-disk layout
changes, the writer bumps this and the reader hard-rejects every
prior version via `PersistenceError.unsupportedVersion(v)`. Per
project convention, no backward-compatible readers. Older files do
not load.

Version history:
- **v1** — original format; no `vBaselines` array.
- **v2** — added `vBaselines`; board stride was 18 planes × 64 = 1,152
  floats per board (pre-arch-refresh).
- **v3** — board stride expanded to 20 planes × 64 = 1,280 floats per
  board (post-arch-refresh; adds repetition planes 18 and 19).

#### `pad` — UInt32

Reserved alignment slot. Writer emits `0`. Reader does not check the
value. Exists so that the Int64 fields that follow start on an
8-byte alignment boundary inside the header, which makes
`loadUnaligned` reads at known offsets more predictable to reason
about.

#### `floatsPerBoard` — Int64

Number of Float32 elements per stored board. Must equal
`BoardEncoder.tensorLength` at load time or the decoder throws
`PersistenceError.incompatibleBoardSize(expected:got:)`. This is the
replay-buffer analog of `.dcmmodel`'s `archHash`: a file minted
against a different input-plane shape cannot load.

Currently 1,280 (20 planes × 64 squares). Historical v2 value was
1,152; v1 same as v2.

#### `capacity` — Int64

Ring capacity in positions at save time. Preserved so that reloading
a capacity-mismatched buffer can make a deliberate decision:

- If `file.capacity <= this.capacity` (loading into a same-or-larger
  ring): all file entries are restored verbatim.
- If `file.capacity > this.capacity` (loading into a smaller ring):
  the oldest `(file.storedCount - this.capacity)` entries in the
  file are discarded and only the newest `this.capacity` positions
  are restored.

Validated at decode time: `capacity >= 0`.

#### `storedCount` — Int64

Number of positions actually present in the file, oldest-first.
Validated: `0 <= storedCount <= capacity`.

Files serialize to a size proportional to `storedCount`, not
`capacity` — a partially-filled ring produces a smaller file. A
completely-empty buffer produces a valid 56-byte header-only file.

#### `writeIndex` — Int64

Ring write cursor at save time — the in-memory physical slot that
would receive the next appended position. Validated:
`0 <= writeIndex < max(1, capacity)`.

**Not used by the restore path.** Since the on-disk entries are
already in chronological order, the reader fills slots 0..<target
and sets its own `writeIndex` accordingly. The field is preserved in
the file so external tools could reconstruct the pre-save physical
ring layout if they wanted.

#### `totalPositionsAdded` — Int64

Lifetime counter of positions appended to the ring, regardless of
wraparound. Preserved verbatim so the `ReplayRatioController`'s
production-rate window stays continuous across save/resume — without
it, a restored buffer would appear to have zero historical production
and the replay-ratio tuner would re-converge from scratch.

### Array sections (v3)

All four arrays are written in **logical (oldest-first) order**, not
physical-slot order. The pre-save `writeIndex` determines where the
ring starts: if `storedCount == capacity`, the oldest entry is at
slot `writeIndex`, so reading wraps from `writeIndex` to the end and
then from slot 0 back to `writeIndex - 1`. If `storedCount < capacity`,
the oldest entry is at slot 0 and the ring hasn't yet wrapped.

Chunked I/O bounded to 32 MB per `handle.write(...)` call keeps peak
`Data` allocations reasonable even on a 1 M-position ring.

#### `boards`

`storedCount × floatsPerBoard × 4` bytes. Each position is exactly
one full encoded board tensor — what `BoardEncoder.encode(...)`
produced when the position was appended. Plane layout (20 planes × 64
squares):

- Planes 0–5: current player's pieces
- Planes 6–11: opponent's pieces
- Planes 12–15: castling rights
- Plane 16: en passant
- Plane 17: halfmove clock (normalized)
- Planes 18–19: repetition signals (≥1× before, ≥2× before)

#### `moves`

`storedCount × 4` bytes. One Int32 per position — the policy index of
the move that was actually played from that position. Indexed in the
AlphaZero-shape 76-channel space (`channel × 64 + row × 8 + col`),
i.e. values are in `[0, 4864)`. Not validated numerically by the
decoder beyond size.

#### `outcomes`

`storedCount × 4` bytes. One Float32 per position — the game outcome
from that position's **side-to-move** perspective:
`+1.0` if the side to move eventually won, `-1.0` if lost, `0.0`
on a draw. Written when the game completes and the whole position
sequence is bulk-copied into the ring.

#### `vBaselines`

`storedCount × 4` bytes. One Float32 per position — the frozen
value-head estimate `v(s)` at play time. Not used directly by the
loss; the trainer recomputes a fresh `v(s)` on every training step
(see `dcm_architecture_v2.md` Addendum A for the fresh-baseline
design) and overwrites the staged vBaselines before feeding the
training graph. The stored `vBaselines` are preserved in the file so
(a) legacy-diagnostic code that reads them still works and (b) the
`vBaselineDelta` stat in the trainer can report `fresh - stored` per
batch for divergence monitoring.

## Decode protocol (v3, strict ordering)

The decoder enforces this order. Any failure at any step aborts with
a specific `PersistenceError`:

1. Open the file for reading. Close is deferred.
2. Read exactly `headerSize = 56` bytes. Fewer → `.truncatedHeader`.
3. `magic == "DCMRPBUF"` → `.badMagic`.
4. `fileVersion == 3` → `.unsupportedVersion(v)`.
5. Read `floatsPerBoard`, `capacity`, `storedCount`, `writeIndex`,
   `totalPositionsAdded` from the header.
6. `Int(floatsPerBoard) == BoardEncoder.tensorLength` →
   `.incompatibleBoardSize(expected, got)`.
7. `capacity >= 0`, `storedCount >= 0`, `storedCount <= capacity`,
   `writeIndex >= 0`, `writeIndex < max(1, capacity)` →
   `.invalidCounts(...)`.
8. If `storedCount == 0`: set lifetime counter, return.
9. Compute `target = min(storedCount, this.capacity)` and
   `skip = storedCount - target` (oldest-first entries to discard
   when loading into a smaller ring).
10. For each of the four arrays in order (boards, moves, outcomes,
    vBaselines):
    - Seek forward by `skip × slotBytes` if `skip > 0`.
    - Read `target × slotBytes` into the storage pointer for that
      array. Short read → `.truncatedBody(expected, got)` or
      `.readFailed`.
11. Set `storedCount = target`, compute new `writeIndex`, restore
    `totalPositionsAdded`.

## Write protocol (v3)

1. Caller holds the serial `queue`; appends are paused for the
   duration.
2. Compute header values from current ring state.
3. Remove any existing file at the target URL. A pre-existing file
   is treated as an overwrite (replay-buffer writes happen inside a
   fresh `.tmp` session directory that no other process touches, so
   overwrite is safe here — it's not the "never overwrite" invariant
   that `.dcmmodel` enforces).
4. `FileManager.createFile(atPath:, contents: nil)`,
   `FileHandle(forWritingTo:)`.
5. Write the 56-byte header.
6. Write the four arrays in order, chunked to 32 MB per
   `handle.write(...)`. Handle ring wraparound by emitting the
   tail-of-ring first, then the head-of-ring.
7. Close the handle (defer-wrapped).

## What's present vs. what's missing (v3)

**Present and correct:**
- Magic check, version gate, arch-fingerprint via `floatsPerBoard`.
- Count sanity: non-negative, `storedCount <= capacity`, `writeIndex`
  in range.
- Clean `PersistenceError` cases for every failure mode the decoder
  explicitly checks.
- Same "saves fresh files, never overwrites prior sessions" behavior
  at the `.dcmsession` directory level (atomic tmp-dir rename in
  `CheckpointManager.saveSession`).

**Missing (addressed in the planned v4 hardening):**
- No integrity hash. A bit-flip inside a 1 GB ring loads silently as
  valid data.
- No total-file-size check. A truncated or trailing-garbage file
  passes decode if the per-section reads happen to succeed.
- No upper bound on `capacity` / `storedCount`. A corrupted header
  with `capacity = Int64.max` would survive the `capacity >= 0`
  check and could drive a huge allocation before the per-section
  reads eventually fail.
- No cross-check against `session.json`'s
  `replayBufferStoredCount` / `replayBufferCapacity` /
  `replayBufferTotalPositionsAdded` fields.
- No `synchronize()` on write — APFS may buffer pages past `close()`.

These are the subject of the "ReplayBuffer durability" item in
`TODO_NEXT.md`. When implemented, the format moves to v4.

## Current format: v4

v4 is the durability-hardened evolution of v3. The on-disk layout
preserves v3's header and section structure verbatim, and appends a
32-byte SHA-256 integrity trailer. Readers enforce a strict file-size
invariant, hard upper-bound caps on every counter field, and full
SHA verification before any header field is trusted.

**This is the format every current build produces and consumes.** v3
and earlier files are rejected by the v4 decoder with
`PersistenceError.unsupportedVersion`. No migration path.

### On-disk layout (v4)

```
Offset  Size              Field                   Type       Notes
------  ----------------  ---------------------   ---------  ------------------------------------
  0        8              magic                  [UInt8]    ASCII "DCMRPBUF"
  8        4              fileVersion            UInt32     Currently 4
 12        4              pad                    UInt32     Written as 0, ignored on read
 16        8              floatsPerBoard         Int64      Must equal BoardEncoder.tensorLength (1280 today)
 24        8              capacity               Int64      Ring capacity at save time (positions)
 32        8              storedCount            Int64      Number of positions actually saved
 40        8              writeIndex             Int64      Ring write cursor at save time (< capacity)
 48        8              totalPositionsAdded    Int64      Lifetime counter (used by replay-ratio controller)

  --- boards: storedCount entries, oldest-first ---
 56        S × F × 4      boards                 Float32    S = storedCount, F = floatsPerBoard

  --- moves: storedCount entries, oldest-first ---
  ..       S × 4          moves                  Int32      Policy index per position

  --- outcomes: storedCount entries, oldest-first ---
  ..       S × 4          outcomes               Float32    Game outcome z ∈ {-1, 0, +1} from side-to-move POV

  --- vBaselines: storedCount entries, oldest-first ---
  ..       S × 4          vBaselines             Float32    Frozen v(s) at play time

  --- trailer ---
END-32   32              sha256                 [UInt8]    SHA-256 over all preceding bytes
```

Total file size (v4):

```
totalBytes = 56 + storedCount × (floatsPerBoard × 4 + 12) + 32
           = 88 + storedCount × (floatsPerBoard × 4 + 12)
```

For a full 1 M-position ring at the current arch: 88 + 1,000,000 ×
(1280 × 4 + 12) ≈ 5.13 GB.

### New in v4 (vs. v3)

- **SHA-256 trailer.** 32-byte digest computed over every preceding
  byte (header + all four section arrays). Same algorithm, same
  trailer-last placement, same "verify before any field is trusted"
  ordering as `.dcmmodel`.
- **Strict file-size equality check.** `actualSize == headerSize +
  storedCount × perSlotBytes + 32`, computed from header fields
  right after the header is read. Any deviation throws
  `PersistenceError.sizeMismatch`. Uses equality, not `>=`, because
  the format is fully deterministic and trailing garbage is
  corruption.
- **Upper-bound caps.** Applied to `floatsPerBoard`, `capacity`,
  `storedCount` before any allocation or seek arithmetic:
  - `floatsPerBoard ≤ 8_192` (currently 1,280; slack for future
    plane expansions)
  - `capacity ≤ 10_000_000` (far above any production ring size)
  - `storedCount ≤ 10_000_000`
  A violation throws `PersistenceError.upperBoundExceeded(field:
  value: max:)`. Paired with the SHA-256 trailer (which catches
  corruption pre-parse) this is defense-in-depth: if a corrupted
  header ever hash-matches by astronomical coincidence, we still
  refuse before allocating a multi-terabyte buffer or overflowing
  a size computation.
- **`handle.synchronize()` on write.** Before the file handle
  closes, `_writeLocked` calls `synchronize()` to commit buffered
  pages to the device. `CheckpointManager.saveSession` follows up
  with `fcntl(F_FULLFSYNC)` via `CheckpointManager.fullSyncPath` for
  drive-cache-bypass durability — the replay buffer gets the same
  treatment as the two `.dcmmodel` files and `session.json`.

### Decode protocol (v4, strict ordering)

Each step throws a specific `PersistenceError` on failure and aborts
before any later check is attempted. No field is trusted until all
preceding checks pass.

1. File opens and the 56-byte header can be fully read →
   `.truncatedHeader`.
2. `magic == "DCMRPBUF"` → `.badMagic`.
3. `fileVersion == 4` → `.unsupportedVersion(v)`.
4. `floatsPerBoard == BoardEncoder.tensorLength` at runtime →
   `.incompatibleBoardSize`. (Replay-buffer analog of the
   `.dcmmodel` arch-hash check — a file minted for a different
   board plane shape cannot load.)
5. Upper-bound caps pass (`floatsPerBoard`, `capacity`,
   `storedCount` each ≤ their `maxReasonable*` threshold) →
   `.upperBoundExceeded`.
6. Counter relationships pass (non-negative, `storedCount ≤
   capacity`, `writeIndex >= 0`, `writeIndex < max(1, capacity)`) →
   `.invalidCounts`.
7. Actual file size (via `FileManager.attributesOfItem`) equals
   `headerSize + storedCount × (floatsPerBoard × 4 + 12) +
   trailerSize` → `.sizeMismatch`.
8. SHA-256 over the first `totalBytes - 32` bytes, streamed in
   `persistenceChunkBytes` chunks through a `CryptoKit.SHA256`
   hasher, matches the 32-byte trailer →
   `.hashMismatch`.

Only after all eight checks pass does the decoder seek back to the
end of the header (offset 56) and read the four parallel arrays into
the ring storage under the serial queue lock. The "load into a
smaller ring than was saved" skip-forward semantics from v3 are
preserved verbatim — the SHA verify passes over every byte in the
file regardless of which entries the caller actually keeps.

### Write protocol (v4)

1. Caller holds the serial `queue`; appends are paused.
2. Compute header values from current ring state.
3. Create a fresh `CryptoKit.SHA256` hasher. Feed the 56-byte header
   through it, then write the header to disk.
4. For each section (boards, moves, outcomes, vBaselines), write
   the oldest-first bytes in 32 MB chunks. Every chunk is fed
   through the hasher before being handed to `FileHandle.write`.
5. Finalize the hasher and write the 32-byte digest as the trailer.
6. `try handle.synchronize()` — forces dirty pages to device before
   close.
7. Close the handle (defer-wrapped).
8. `CheckpointManager.saveSession` follows with
   `fullSyncPath(bufferTmpURL)` to issue `F_FULLFSYNC` on the file,
   bypassing the drive's write cache.

### Durability pipeline in session saves

Saving a `.dcmsession` bundle that contains a replay buffer runs the
following durability steps, in order:

1. Write all four files (two `.dcmmodel`, `session.json`,
   `.replay_buffer.bin`) into a `tmpDir` staging directory.
2. `fullSyncPath` each of the four files (`F_FULLFSYNC`).
3. Verify:
   - Both `.dcmmodel`: bit-exact + forward-pass round-trip.
   - `session.json`: decode round-trip.
   - `.replay_buffer.bin`: re-load into a scratch `ReplayBuffer`
     via `restore(from:)` (runs the full v4 verification stack),
     then compare `stateSnapshot()` counters against the live
     in-memory buffer.
4. `fullSyncPath(tmpDir)` — flush directory-entry metadata.
5. `fm.moveItem(tmpDir, finalDir)` — atomic rename commits the
   session.
6. `fullSyncPath(CheckpointPaths.sessionsDir)` — flush the parent
   directory so the rename itself is durable.

Any failure in steps 1-4 removes the tmp directory via `cleanupTmp()`
and throws; no final-named artifact appears on disk. A failure in
step 6 leaves the session visible (the rename is already committed)
but logs a warning that the parent-directory flush wasn't
guaranteed.

### Launch-time orphan sweep

At app start, before any save or load UI activates,
`CheckpointPaths.cleanupOrphans()` removes any entry ending in
`.tmp` from `Sessions/` (directories) and any entry ending in
`.dcmmodel.tmp` from `Models/` (files). Those are the suffixes
`saveSession`'s staging dir and `saveModel`'s staging file use —
their presence after launch means a previous save was interrupted
mid-flight. Each removal is logged via `[CLEANUP]`; failures log
`[CLEANUP-ERR]` and do not abort the sweep.

### Session-load cross-check

`CheckpointManager.verifyReplayBufferMatchesSession(buffer:state:)`
runs after a successful `restore(from:)` at session load time. It
compares `buffer.stateSnapshot().totalPositionsAdded` against
`state.replayBufferTotalPositionsAdded` from `session.json`. A
mismatch (throws `CheckpointManagerError.sessionReplayMismatch`)
indicates a file-pairing error — replay buffer from one save paired
with `session.json` from another — or residual corruption that
happened to SHA-match.

Only the lifetime counter is cross-checked, not `storedCount` or
`capacity`. Those two intentionally diverge when loading a larger
saved ring into a smaller live one (see `ReplayBuffer.restore`'s
`skip = fileStored - target` logic). `totalPositionsAdded` survives
that logic verbatim and is an effectively unique fingerprint across
sessions. Missing in `session.json` (Optional for back-compat) →
check is skipped rather than forced to mismatch.

### Error taxonomy (v4)

| Error | When |
|---|---|
| `.badMagic` | First 8 bytes ≠ `"DCMRPBUF"` |
| `.truncatedHeader` | File is shorter than the 56-byte header |
| `.unsupportedVersion(v)` | `fileVersion` is not 4 |
| `.incompatibleBoardSize(exp, got)` | `floatsPerBoard` does not match the running build's `BoardEncoder.tensorLength` |
| `.upperBoundExceeded(field, value, max)` | A counter field exceeds its `maxReasonable*` cap |
| `.invalidCounts(cap, stored, wi)` | Counter relationships fail (non-negative, stored ≤ capacity, writeIndex in range) |
| `.sizeMismatch(expected, got)` | Actual file size ≠ header-predicted size |
| `.hashMismatch` | SHA-256 trailer does not match recomputed hash |
| `.truncatedBody(expected, got)` | A section read returned fewer bytes than requested (should be caught by `.sizeMismatch` first, but kept as a defense) |
| `.writeFailed(err)` | Write-side I/O failure |
| `.readFailed(err)` | Read-side I/O failure that does not fit a more specific error |

## Non-goals

- **No compression.** Board float entropy is high (mostly 0.0 / 1.0
  but the per-position mix varies), and the file grows linearly with
  `storedCount`, not with any schema-level repetition the way
  columnar formats exploit.
- **No per-position hashing.** A single file-level hash is sufficient
  once v4 lands; per-record hashing would multiply write cost by 4×
  for no benefit over the file hash.
- **No variable-length records.** Every position is exactly
  `floatsPerBoard × 4 + 12` bytes. Fixed stride means per-position
  seeks are O(1), which the decode path relies on for the "load into
  smaller ring" skip-forward semantics.
- **No cross-architecture migration.** Same stance as `.dcmmodel`:
  if `floatsPerBoard` disagrees, the file will not load.
