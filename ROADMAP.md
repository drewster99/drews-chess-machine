# Roadmap

Long-term goals, deferred work, and notes on decisions.

## Future improvements

- **Adaptive learning-rate schedule.** Currently `learnRate` is a static
  hyperparameter (default 5e-5) that the user adjusts manually via the
  UI. Real-world deep-RL training benefits from LR scheduling — high LR
  early to make fast progress on raw signal, then decay as the network
  converges so late-stage updates don't oscillate.

  ### Candidate trigger families (surveyed)

  - **Step-based decay.** LR multiplied by γ every N training steps
    (e.g., γ=0.5 every 100K steps). Predictable, tunable, but blind
    to actual training health.
  - **Plateau detection.** Watch a smoothed loss; when it stops
    decreasing for N consecutive measurements, multiply LR by 0.5.
    Standard "ReduceLROnPlateau" pattern.
  - **Promotion-driven.** Drop LR by a factor on every successful
    arena promotion. The intuition: promotion proves the current
    policy has shifted meaningfully, so subsequent updates should be
    gentler to lock in that progress before the next promotion
    window. Rejected as an upward-bump mechanism: arena failure more
    likely means the candidate diverged in a worse direction (LR too
    hot, or replay overfit) than "stuck in a flat region that a
    bigger step would escape." Raising LR on failure would make those
    cases worse. Promotion-driven *downward* nudge still useful as a
    secondary overlay.
  - **Cosine annealing with restarts (SGDR).** Smoothly decays LR
    across an "epoch" then restarts to the high value. Often
    empirically strong in supervised vision but adds another tunable
    (epoch length) and has no natural "epoch" concept in self-play.
  - **Replay-ratio aware.** Tie LR to the cons/prod ratio so that
    undertraining (cons < prod) doesn't get amplified by a too-hot
    LR.

  ### What AZ-family engines actually use (survey, 2026-04)

  Researched the five canonical systems' published LR schedules,
  normalized to positions-seen so batch-size differences don't
  confuse the comparison:

  - **AlphaZero** (Silver et al., 1712.01815 v1 + Science 2018). SGD
    + momentum 0.9. Start 0.2, step decay 10× per drop at 100K /
    300K / 500K training steps → 0.0002 floor. Batch 4,096 positions.
    Normalized: first drop at ~410M positions, final at ~2.05B, total
    ~2.87B positions across 700K steps. (Go variant starts at 0.02
    rather than 0.2.)
  - **AlphaGo Zero** (Silver et al., Nature 2017). SGD + momentum
    0.9, L2 = 1e-4. Start 1e-2, step decay 10× → 1e-3 → 1e-4.
    Extended Data Table 3 milestones not confirmable from primary
    source in this pass; ELF OpenGo (1902.04522), which explicitly
    reproduces AGZ faithfully, used 500K / 1M / 1.5M minibatches.
    Batch 2,048 positions. Normalized (via ELF): ~1.0B / ~2.0B /
    ~3.0B positions.
  - **Leela Chess Zero** (`lczero-training/tf/configs/example.yaml`).
    `lr_values=[0.02, 0.002, 0.0005]`, `lr_boundaries=[100000,
    130000]` steps, 250-step warmup, 140K total. Batch 2,048.
    Normalized: drops at 205M / 266M positions, total 287M. Caveat:
    that's the documented *example* — T60/T70/T78/T80/BT production
    runs don't publish per-run YAML values anywhere I could find.
    The blog/wiki describe "LR starts high and is occasionally
    reduced" without numbers.
  - **KataGo** (Wu, 1902.10565, and the live `python/train.py`).
    Parameterized in *samples* (= positions), not gradient steps.
    Paper "g170" run: base per-sample 6e-5, batch 256 →
    effective per-batch 0.01536. Warmup: first 5M samples at ⅓ LR.
    For the b20c256 run, one 10× drop late (after ~17.5 days). Not
    really step-decay in the AZ sense — effectively constant LR with
    warmup + one late drop, combined with SWA / EMA of snapshots
    (every ~250K samples snapshot; every ~1M samples a new EMA
    candidate with decay 0.75) doing work the others get from
    discrete drops. Current `train.py` has drifted to base per-sample
    3e-5 with a 9-stage warmup over first 2M samples and a piecewise
    `lr_scale_auto2` multiplier ramping 12× down to 0.05× over ~600M
    samples — much more elaborate than the paper.
  - **MuZero** (Schrittwieser et al., Nature 2020, board-games
    config in the paper's pseudocode). SGD + momentum 0.9, WD 1e-4.
    Start 0.1 (chess/shogi) or 0.01 (Go). **Exponential** decay, not
    step decay:
    `lr = lr_init · 0.1^(training_step / 400_000)`. Batch 2,048; 1M
    total steps. Normalized: LR falls 10× every ~819M positions. At
    1M steps (~2.05B positions) LR has decayed ~316× (chess final
    ≈ 3.2e-4).

  **Synthesis.** Common pattern is step decay at 10× per drop, ~3
  drops total, first drop somewhere between ~200M and ~1B positions
  seen. Nobody in this family uses cosine, cyclic, or warm restarts.
  Batches cluster at 2,048 (AGZ, lc0, MuZero), with AZ doubling to
  4,096 and KataGo going small at 256. Outliers: KataGo in *shape*
  (mostly flat + late drop + EMA), MuZero in *mechanism* (continuous
  exponential vs. discrete drops). At our current `learnRate = 5e-5`
  we're roughly at KataGo's paper per-sample scale.

  ### Chosen design

  MuZero-style continuous exponential decay keyed to
  `trainingPositionsSeen` (an invariant under live batch-size
  changes), with a promotion-driven secondary overlay:

  - **Primary schedule**: `lr = lr_init · γ_e^(positions / τ)` with
    `γ_e = 0.1` (10× per τ, matching the family) and default
    `τ = 500M` positions per 10× — puts us in the AZ / MuZero zone.
    Both `lr_init` and `τ` live-tunable in the UI.
  - **Promotion overlay**: on every successful arena promotion,
    additionally multiply LR by ~0.9 (consolidation nudge). Monotonic
    non-increasing — no upward bumps on arena failure. Per the
    rejection above, failure-upward is more likely to hurt than help.
  - **Floor**: 1e-7 so long runs don't collapse to zero.
  - **Optional warmup**: linear ramp from 0 → `lr_init` over first
    ~2K steps, added only if early-training instability shows up.
  - **Manual UI override wins**: "Schedule: auto / off" toggle; with
    off, the slider is authoritative and the schedule is paused
    (preserves the current auto value so re-enabling doesn't snap).
    With on, the slider shows the current scheduled value read-only.
  - **Logging**: every schedule-driven change logged as
    `[PARAM] learningRate <old>→<new> reason=<decay|promotion|warmup>`
    so change history lives alongside `[STATS]` in the session log.
  - **Persistence**: `lr_init`, `τ`, `γ_promotion`, schedule on/off,
    and the last-computed LR all live in `session.json` so a reload
    resumes at the same scheduled value rather than jumping.

  Deliberately *not* doing: step-milestone decay (operates on
  training steps, which our live-tunable batch size makes
  non-invariant), plateau-on-loss (our `pLoss` is outcome-weighted
  and unbounded — unreliable plateau signal), replay-ratio aware
  (the `ReplayRatioController` already handles cons/prod; doubling
  up couples two control loops with no clear benefit), cosine/SGDR
  (no natural epoch in self-play; adds a tunable without a
  principled way to set it).

- **Model and session save/load.** Today nothing persists across app
  launches — quit mid-training and you lose the champion, the trainer,
  every accumulated counter, and the replay buffer. Two file formats,
  one underlying primitive.

  **Single model — `.dcmmodel` (flat binary file).** Wraps one network's
  weights plus identity and metadata. Fixed binary header, then the 37
  tensors that come out of `ChessMPSNetwork.exportWeights()` in declared
  order, then a trailing 32-byte SHA-256 over all preceding bytes for
  integrity. Header carries: magic `"DCMMODEL"`, format version,
  `archHash` (hash of filters / blocks / input channels / policy dim —
  hard-refuses to load on mismatch, no migration), `numTensors`
  sanity-check, creation wall-clock time, `ModelID`, parent `ModelID` at
  time of save, and a JSON metadata blob (arena stats at mint,
  training-step count, creator tag). Loadable into any training- or
  inference-mode `ChessNetwork` via the existing `loadWeights` path —
  this is the unit for "take any model at any point and use for
  inference."

  **Training session — `.dcmsession` (directory).** Holds
  `champion.dcmmodel`, `trainer.dcmmodel`, and `session.json`. Making a
  session a directory of `.dcmmodel` files rather than a custom bundle
  means (a) extraction is free — Finder-copy any model out of a session
  — and (b) only one binary format to debug. `session.json` is a
  Codable blob with the session's stable `sessionID`, format version,
  save and session-start wall-clock timestamps, accumulated training
  time, all STATS-line counters (trainingSteps, selfPlayGames,
  selfPlayMoves, trainingPositionsSeen), all hyperparameters that
  appear in the arena footer (batch, lr, promote threshold, arena
  games, sp/arena tau configs, self-play worker count), both network
  IDs duplicated from the `.dcmmodel` headers for fast index reads,
  and a light arena history (W/L/D + kept/promoted + step-at-run for
  each arena so far). Excluded from v1: the 500k-position replay
  buffer (~2.3 GB — resume warmup cost is ~5 min of self-play to
  refill, acceptable), the candidate network (only exists mid-arena —
  saving mid-arena is disallowed), and in-flight self-play games
  (workers abandon on save, same behavior as Stop).

  **Save triggers.** Menu items: Save Session, Save Champion as Model,
  Load Session, Load Model. Autosave on arena promotion defaults **on**
  — every promotion writes a full session snapshot alongside the manual
  saves. Save Session is disabled mid-arena; Load Session and Load Model
  require Play-and-Train to be stopped.

  **File locations.** All saves — manual and auto — land in a fixed
  Library path so there's one canonical place to find them:
  `~/Library/Application Support/DrewsChessMachine/Sessions/` for
  sessions, `~/Library/Application Support/DrewsChessMachine/Models/`
  for single models. **Every save keeps the old file** — nothing is
  ever overwritten. Users prune manually. Naming scheme is
  `<YYYYMMDD-HHMMSS>-<modelID>-<trigger>.<ext>` where trigger is
  `manual` or `promote`; the wall-clock prefix gives natural Finder
  sort order. A "Reveal Saves in Finder" button opens the relevant
  folder so the hidden `Application Support` location is discoverable.
  Load uses the standard `fileImporter` sheet so you can drag in a
  file from anywhere (Downloads, AirDrop, another machine) without
  having to move it into the canonical folder first.

  **Every save is self-verified before it's marked successful.** After
  writing the file(s) atomically (tmp + fsync + rename), the save code
  (1) re-reads the file from disk, (2) bit-compares the re-read tensors
  byte-for-byte against the `[[Float]]` that was exported pre-write,
  and (3) loads the saved weights into a throwaway
  `ChessMPSNetwork` and runs a forward pass on a canonical test
  position (starting position + one fixed mid-game FEN), comparing
  policy and value outputs bit-exact to the same forward pass on the
  source network. Any mismatch deletes the freshly-written `.tmp`,
  leaves any prior save in the folder untouched (since we keep
  history), and surfaces a user-visible error. This gives us production
  round-trip correctness checking for free on every save — a
  `loadWeights` regression shows up on the user's next save attempt,
  not three hours later on resume.

  **Validation — this plan doesn't complete until all of these pass.**
  (1) Build succeeds. (2) Round-trip a single model: Save Champion as
  Model → quit → relaunch → Load Model → run Forward Pass on a fixed
  FEN → policy and value outputs are bit-exact identical to pre-save.
  (3) Round-trip a session: Play-and-Train for a few minutes → Save
  Session → quit → relaunch → Load Session → `session.json` counters
  and ModelIDs match → champion and trainer Forward Pass outputs are
  bit-exact on a fixed FEN → Play-and-Train resumes, buffer refills,
  a subsequent arena runs and can promote. (4) Arch-mismatch file —
  hand-edit `archHash` or build with different filter/block counts —
  refuses to load with a clear user-facing error, no crash, no
  silent success. (5) Truncated file — cut the last 1 KB of a
  `.dcmmodel` — refuses to load with a clear error, no crash.
  (6) SHA mismatch — flip one byte in the middle of a `.dcmmodel` —
  refuses to load with a clear error. (7) Save-mid-arena is
  disallowed — menu item is disabled or errors clearly during an
  arena. (8) Save atomicity — kill the process mid-save (`SIGKILL`
  while writing `.tmp`) → prior save on disk is still intact, no
  half-written file left behind. (9) Every existing test still passes.

  **Session restore coverage.** What is saved in `.dcmsession` and
  what happens on resume:

  | Field | Save | Restore |
  |---|---|---|
  | Champion + trainer weights | `.dcmmodel` files | loaded into networks |
  | Champion + trainer model IDs | `session.json` | restored to identifiers |
  | Session ID | `session.json` | inherited for continuity |
  | Elapsed training time | `session.json` | back-dated `sessionStart` anchor |
  | Training step count | `session.json` | seeded into both stats boxes |
  | Self-play games / moves | `session.json` | seeded into `ParallelWorkerStatsBox` |
  | Game results (W/B checkmates, stalemate, 50-move, 3-fold, insuff. material) | `session.json` | seeded into `ParallelWorkerStatsBox` |
  | Learning rate | `session.json` | restored to `@AppStorage` + trainer |
  | Replay ratio target + auto-adjust toggle | `session.json` | restored to `@State` + controller |
  | Step delay + last auto-computed delay | `session.json` | restored to `@AppStorage` |
  | Self-play worker count | `session.json` | restored to `@State` |
  | Arena history (W/L/D, score, promoted flag per arena) | `session.json` | rebuilt into `tournamentHistory` |
  | Replay buffer contents | not saved (4.6 GB) | refills in ~5 min |
  | Progress rate chart samples | not saved | rebuilds from new data |
  | Rolling loss windows | not saved | rebuilds from new steps |

- **Bitboard board representation.** `GameState.board` is currently a flat
  `[Piece?]` of 64 entries. The next step for performance is twelve `UInt64`
  bitboards (one per piece kind/color), with attack tables. Move generation,
  attack detection, and tensor encoding all become bit operations and would
  be dramatically faster than the current per-square scan.

- **Engine-level legal-move validation.** `ChessGameEngine.applyMove` and
  `MoveGenerator.applyMove` no longer validate that the supplied move is
  legal — callers are required to do so. Higher-level code (the game loop in
  `ChessMachine`) currently feeds only legal moves into apply, so this is
  safe. If we ever expose a public API where moves arrive from untrusted
  sources (UI, network, file load), we should add a validating wrapper that
  calls `legalMoves(for:)` and checks membership before delegating to apply.

- **Compiled `MPSGraphExecutable`.** `ChessNetwork.evaluate` currently calls
  `graph.run(with:feeds:targetTensors:targetOperations:)`. MPSGraph caches a
  compiled executable internally keyed on feed shapes, so the steady-state
  cost is close to a hand-compiled executable, but pre-compiling via
  `graph.compile(...)` would remove per-call cache lookup and give us a
  reusable `MPSGraphExecutable` we can serialize for the
  `NetworkInitMode.package` path. Worth revisiting once training lands.

- **Fuse legal-move masking into the policy head.** Today the graph emits a
  full 4096-way softmax and the CPU masks illegal moves and renormalizes.
  An alternative is adding a `legalMask` placeholder, switching `policyHead`
  to emit logits, and computing
  `softmax(logits + (legalMask - 1) * 1e9)` inside the graph. Marginal at
  batch=1; potentially worthwhile when batching positions.

- **Partial heap or quickselect for top-k policy moves.**
  `ChessRunner.extractTopMoves` currently full-sorts the 4096-entry policy
  vector to pull the top 4 (O(n log n) ≈ 49k comparisons). A size-k min-heap
  walk would be O(n log k) ≈ 8k comparisons; quickselect would be ~O(n) on
  average. The absolute savings are microseconds and this only runs on the
  Run Forward Pass demo button — not the self-play hot path — so it's
  cosmetic. Worth doing if we ever start ranking top-k moves per ply during
  search.

## Findings

- **Batch-size sweep is reliable at 1 s per batch size.** The Batch Size
  Sweep panel runs a training-mode copy of the network through real SGD
  steps at each batch size and reports steady-state throughput. We tried
  longer per-size windows (15 s, 5 s, 3 s, 1.5 s) and found 1 s gives
  essentially the same shape and the same winner — the fast-warming MPSGraph
  caches mean each row converges within a handful of steps and the tail just
  accumulates redundant samples. Keeping it at 1 s makes the whole sweep
  cheap enough to run any time on a new machine to pick the most efficient
  batch size for *that* hardware, rather than baking a single number in.

- **Sweep memory guard is empirical, not architectural.** The sweep refuses
  to run a batch size whose predicted resident footprint exceeds 75 % of
  `min(recommendedMaxWorkingSetSize, maxBufferLength)`, or whose largest
  single buffer would exceed `maxBufferLength`. The prediction comes from
  a least-squares linear fit over the (batch, peak `phys_footprint`) pairs
  already observed during the same sweep — no per-architecture fudge
  factor. Peak `phys_footprint` is sampled by the UI heartbeat (~10 Hz)
  plus once at the start and end of each row, so we catch transient spikes
  during a step rather than relying on `MTLDevice.currentAllocatedSize`,
  which is post-step and undercounts. Skipped rows still appear in the
  table with the prediction and the reason they were skipped, so the
  sweep walks the full ladder and makes its limits visible.

## Completed

- **Session durability hardening — "saved means golden" (2026-04-20).**
  Closes TODO_NEXT.md #3. The save pipeline now guarantees that either
  a fully-verified, fsync'd `.dcmsession` bundle appears on disk under
  its final name, or nothing appears with that name. Restored
  sessions are bit-identical to what was saved, or loading fails with
  a specific error describing which check tripped.

  **Principle.** If any piece of a session save cannot be fully
  verified end-to-end, the whole save fails, all partials are removed,
  and no final-named artifact appears on disk. Built as a coordinated
  change across `ReplayBuffer.swift`, `CheckpointManager.swift`,
  `DrewsChessMachineApp.swift`, and `ContentView.swift`, plus docs in
  `replay_buffer_file_format.md` (new) and `dcmmodel_file_format.md`
  (new), and five new XCTest cases in `ReplayBufferTests.swift`.

  ### ReplayBuffer format v3 → v4

  `ReplayBuffer.fileVersion` bumped 3 → 4. Readers reject v1/v2/v3
  cleanly with `PersistenceError.unsupportedVersion`. No migration
  path — matches the project's delete-and-retrain stance (the user's
  existing v3 `.replay_buffer.bin` files will not load; the replay
  buffer is recovered by resumed self-play).

  - **SHA-256 trailer.** 32-byte digest appended over header + all
    four body sections, verified before any header field is trusted
    at load time. Mirrors the `.dcmmodel` integrity-trailer
    convention. Computed streaming during write (CryptoKit
    `SHA256.update(data:)` per chunk), so no extra hashing pass — the
    bytes fed to `handle.write(...)` are also fed to the hasher.
  - **Strict file-size equality check.** Restore computes
    `expectedBytes = headerSize(56) + storedCount × (floatsPerBoard ×
    4 + 12) + trailerSize(32)` and requires `actualFileSize ==
    expectedBytes`. Uses `==`, not `>=`, because the format is fully
    deterministic — any deviation is corruption. New
    `PersistenceError.sizeMismatch(expected, got)`.
  - **Upper-bound sanity caps.** Applied before any allocation or
    seek arithmetic so a corrupted header can't drive a
    multi-terabyte allocation or overflow the size computation.
    Caps: `floatsPerBoard ≤ 8_192`, `capacity ≤ 10_000_000`,
    `storedCount ≤ 10_000_000`. New
    `PersistenceError.upperBoundExceeded(field, value, max)`.
  - **`handle.synchronize()` before close.** In `_writeLocked`,
    forces APFS to flush dirty pages to the device before the file
    handle closes. On top of this, `CheckpointManager.saveSession`
    adds `fcntl(F_FULLFSYNC)` via `fullSyncPath` for drive-cache-bypass
    durability.
  - **Atomic write-and-snapshot.** `ReplayBuffer.write(to:)` now
    returns the `StateSnapshot` reflecting exactly the state
    serialized into the file, captured under the same `queue.sync`
    lock that serializes the write. Post-save verification compares
    against this value — a subsequent `stateSnapshot()` call would
    diverge because concurrent self-play workers resume appending
    after the save-gate releases the trainer pause. Annotated
    `@discardableResult` so existing callers (tests, etc.) compile
    unchanged. **This was a recheck-catch** — the first pass
    compared against a freshly-called `live.stateSnapshot()` and
    would have spuriously failed the counter comparison every time
    saves happened during active training.

  ### `CheckpointManager` durability pipeline

  `saveSession` now runs a full fsync pipeline on top of the existing
  tmp-dir-then-rename atomicity:

  1. Write all four files (two `.dcmmodel`, `session.json`,
     `.replay_buffer.bin`) into a `.tmp` staging directory.
  2. `fullSyncPath` each of the four files — issues `fcntl(fd,
     F_FULLFSYNC)` on Apple filesystems (falls back to `fsync` if
     unsupported). Bypasses the drive's write cache, not just the VFS
     page cache.
  3. Verify:
     - Both `.dcmmodel`: existing bit-exact + forward-pass
       round-trip (unchanged).
     - `session.json`: existing decode round-trip (unchanged).
     - **NEW** — `.replay_buffer.bin`: re-load into a scratch
       `ReplayBuffer` via `restore(from:)`. The scratch restore runs
       the full v4 verification stack (SHA, size, caps). Scratch is
       allocated at `max(1, writtenSnap.storedCount)` — sized to the
       actual data, not to the live ring's full capacity — so a
       half-full 1 M-slot ring does not double peak memory during
       verify. Then compare the restored `storedCount` and
       `totalPositionsAdded` against the `writtenSnap` captured
       atomically from the write. Drift here implies a write-path
       regression that produced a valid-SHA file with wrong bytes
       (the SHA alone cannot catch this class of bug if the write is
       internally consistent but semantically wrong).
  4. `fullSyncPath(tmpDir)` — flush directory-entry metadata.
  5. `fm.moveItem(tmpDir, finalDir)` — atomic rename.
     `FileManager.moveItem` aborts if the destination already
     exists (unlike POSIX `mv`); plus the existing
     `fileExists(finalDirURL)` guard. Two independent guards.
  6. `fullSyncPath(CheckpointPaths.sessionsDir)` — flush the parent
     so the rename itself is durable. A failure here leaves the
     session visible (rename already committed) and logs a warning
     that the parent-directory flush wasn't guaranteed — we do not
     remove the session, as it's already the "best we've got" on
     disk.

  Any failure in steps 1–4 triggers `cleanupTmp()` and throws. No
  final-named artifact appears on disk.

  `saveModel` gets the same treatment: `fullSyncPath` on the tmp file
  before verify-and-rename, and `fullSyncPath` on
  `CheckpointPaths.modelsDir` after rename.

  **New `CheckpointManagerError` cases:** `fsyncFailed(URL, Error)`,
  `replayVerificationFailed(String)`, `sessionReplayMismatch(detail:
  String)`.

  ### Load-time cross-check

  New helper `CheckpointManager.verifyReplayBufferMatchesSession(buffer:
  state:)` runs after `ReplayBuffer.restore(from:)` at session load
  time. Compares `buffer.stateSnapshot().totalPositionsAdded` against
  `state.replayBufferTotalPositionsAdded` from `session.json`.
  Mismatch throws `CheckpointManagerError.sessionReplayMismatch` and
  surfaces in the load UI.

  Only the lifetime counter is cross-checked, not `storedCount` or
  `capacity` — those two intentionally diverge when loading a larger
  saved ring into a smaller live one (existing restore
  `skip = fileStored - target` logic). `totalPositionsAdded` survives
  that logic verbatim and is effectively unique across sessions, so a
  mismatch strongly implies a file-pairing error (wrong replay paired
  with wrong session.json) or SHA-collision-scale corruption.
  `replayBufferTotalPositionsAdded` is Optional in
  `SessionCheckpointState` (back-compat) — missing → check is skipped
  rather than forced to mismatch.

  Wired into `ContentView.loadSessionFrom`'s post-restore path.

  ### Launch-time orphan sweep

  New `CheckpointPaths.cleanupOrphans()` runs from
  `DrewsChessMachineApp.init` after `SessionLogger.start`. Removes:

  - `Sessions/<name>.tmp/` directories — `saveSession`'s staging
    directory (matches the `.tmp` suffix appended to the target
    session dir name).
  - `Models/<name>.dcmmodel.tmp` files — `saveModel`'s staging file
    (matches the `.tmp` extension appended to the final `.dcmmodel`
    filename).

  Each removal is logged `[CLEANUP] Removed orphan <name>`; failures
  log `[CLEANUP-ERR]` and do not abort the sweep (a stuck orphan
  should not prevent the app from starting). Runs once at launch,
  before any save/load UI activates.

  ### Documentation

  - **NEW** `dcmmodel_file_format.md` — full byte-level spec for the
    `.dcmmodel` format, with expanded FNV-1a documentation
    (constants `0x811C9DC5` offset basis and `0x01000193` prime,
    algorithm pseudocode, Swift reference implementation, byte-order
    rationale, worked example, comparison vs. CRC32/xxHash/SHA-256),
    SHA-256 trailer spec, decode protocol, error taxonomy, explicit
    non-goals.
  - **NEW** `replay_buffer_file_format.md` — v3 (historical) + v4
    (current) sections. v4 section includes full layout, decode
    protocol ordering, write protocol, durability pipeline in
    session saves, cross-check semantics, launch-time orphan sweep,
    and full error taxonomy.
  - `CHANGELOG.md` entry at top (short form — points here for full
    detail).
  - `TODO_NEXT.md` §3 removed (was detailed there during planning;
    now done).

  ### Tests

  `DrewsChessMachineTests/ReplayBufferTests.swift` — 5 new tests, all
  green:

  - `testV3FileRejectedWithUnsupportedVersion` — synthesized v3
    header (no SHA trailer) rejects with `unsupportedVersion(3)`.
  - `testV4SHAMismatchRejected` — valid v4 file with one byte flipped
    at offset 56 rejects with `hashMismatch`.
  - `testV4SizeMismatchOnTruncation` — valid v4 file truncated by
    one byte rejects with `sizeMismatch`.
  - `testV4SizeMismatchOnTrailingGarbage` — valid v4 file with an
    extra byte appended rejects with `sizeMismatch`.
  - `testV4UpperBoundRejectedOnCapacity` — header with
    `capacity = Int64.max` rejects with `upperBoundExceeded(field:
    "capacity", ...)`, not an allocation crash.

  Existing tests (`testEmptyBufferWriteRead`,
  `testSinglePositionWriteRead`, `testV2FileRejectedWithUnsupportedVersion`,
  `testBadMagicRejected`, `testTruncatedHeaderRejected`) pass
  unchanged against v4 via the public API. Full test suite: 55/55
  green.

  ### Scope limits explicitly not taken

  - No per-record hashing (file-level SHA-256 is sufficient).
  - No compression (writes stay raw-float).
  - No cross-architecture or cross-version migration.
  - No `session.json` schema change (cross-check reads existing
    Optional fields).

  ### Parameter reference

  New constants (private statics on `ReplayBuffer`):
  - `fileVersion: UInt32 = 4` (was 3)
  - `trailerSize: Int = 32`
  - `maxReasonableCapacity: Int64 = 10_000_000`
  - `maxReasonableStoredCount: Int64 = 10_000_000`
  - `maxReasonableFloatsPerBoard: Int64 = 8_192`

  No existing hyperparameter (LR, batch size, clip, tau, etc.) was
  changed.

- **N-worker concurrent self-play in Play and Train.** Play and Train
  previously ran a single self-play worker, which at ~357 moves/sec against
  a 3,012 moves/sec training consumer meant every replay-buffer position
  was sampled ~8.4× on average before eviction — far above the 2–4×
  replay ratio common for off-policy RL, and the buffer also covered only
  ~625 games of play diversity. The fix is to spawn `N` concurrent
  self-play workers at session start, each with its own dedicated
  `ChessMPSNetwork` instance so no two concurrent `evaluate` calls share
  MPSGraph state. `ContentView.initialSelfPlayWorkerCount` (currently
  `6`) sets the default active count when a session begins;
  `ContentView.absoluteMaxSelfPlayWorkers` (currently `16`) is the hard
  ceiling — we pre-build that many inference networks and spawn that
  many worker tasks so the user can live-tune N inside
  `[1, absoluteMaxSelfPlayWorkers]` via a Stepper next to Run Arena
  without restarting the session. Topology is asymmetric: worker 0
  reuses the existing `network` (the champion, also the arena snapshot
  source), and workers `1..N-1` use new `secondarySelfPlayNetworks`
  mirrored from the champion at session start and at every arena
  promotion. Each worker owns its own `WorkerPauseGate`, so the
  arena-champion snapshot path (which only reads `network`) still
  pauses only worker 0, and only the promotion branch pauses every
  worker to `loadWeights` into every self-play network. Players
  (`MPSChessPlayer` white/black) are now allocated once per worker and
  reused across games — `ChessMachine.beginNewGame` already calls
  `onNewGame` on each, which resets per-game scratches while keeping
  backing storage alive. Under N=1 (checked live per game via
  `countBox.count == 1`, not captured at spawn), worker 0 wires
  `GameWatcher` as its `ChessMachine` delegate for the animated board;
  under N>1 no worker does, and a placeholder overlay "N = X concurrent
  games" hides the static board slot so the Candidate test picker
  remains usable. Aggregate self-play rates accumulate through the
  thread-safe `ParallelWorkerStatsBox`, which every worker calls
  identically via `recordCompletedGame(moves:durationMs:result:)` —
  no worker-0 specialness in the stats path. Setting N to 1
  reproduces the pre-change behavior (modulo the per-game player
  reuse cleanup). Memory cost is ~12 MB per additional inference
  network, trivial on unified memory.

  **Idle workers stay allocated deliberately.** When the user drops N
  from 6 to 3 via the Stepper, workers 3–5 finish their current game,
  then on their next iteration evaluate `countBox.count > workerIndex`,
  see false, and enter `WorkerPauseGate.markWaiting()` — a 50 ms
  sleep-poll loop that costs near-zero CPU. Their `ChessMPSNetwork`
  instances, `MPSChessPlayer` scratches, `WorkerPauseGate` state, and
  Swift tasks **all stay alive for the life of the session.** Only GPU
  cycles, CPU cycles for move generation / encoding / sampling, and
  replay-buffer lock contention are freed. Networks are only actually
  deallocated when Play and Train stops — and even then
  `secondarySelfPlayNetworks` persists in `@State` across sessions so
  re-entering Play and Train doesn't re-pay the MPSGraph build cost
  (~100 ms + per-network kernel JIT).

  This is a deliberate memory-vs-latency trade. The alternative design
  would cancel tasks and release networks on Stepper-down, then rebuild
  on Stepper-up — saving ~12 MB per idled worker but costing ~100–300 ms
  per + click for MPSGraph construction, first-run kernel JIT, and
  weight sync from the champion. Keeping everything pre-spawned means +
  and − clicks are effectively instant (≤50 ms, bounded by the idle
  poll interval) with no visible latency on the UI. At
  `absoluteMaxSelfPlayWorkers = 16` the steady-state memory cost is
  ~180 MB of idle network state plus ~74 MB of `MPSChessPlayer` scratch
  buffers, which is fine on Apple Silicon unified-memory systems. If
  that footprint ever becomes a problem on tighter hardware, the
  release-on-shrink design is the fallback; for now the latency win on
  live tuning is worth the static allocation.

- **Bundled architecture refresh (v2), 2026-04-19/20.** A coordinated
  redesign of the policy encoding, input planes, policy head, and
  value-baseline semantics, delivered as one bundle because the moves
  are coupled (the policy-encoding bijection and the fully-conv head
  both change the meaning of the 4864 logits; a staged rollout would
  have required throwaway migration code). Full design and phase
  breakdown live in `dcm_architecture_v2.md`. Summary of what shipped:

    - **AlphaZero-shape policy encoding (Phase 2).** The old flat 4096
      = 64×64 from-to encoding was replaced with 4864 = 76 × 64
      channel-square logits: 56 queen-style directions × distances, 8
      knight, 9 underpromotion (N/R/B × 3 directions, channels 64–72),
      3 queen-promotion (channel 73–75). Dedicated underpromotion
      channels fix the prior silent-collapse where all four promotion
      pieces mapped to the same index. The bijection lives in
      `PolicyEncoding.policyIndex(_:currentPlayer:)` — `ChessMove.policyIndex`
      was deleted so callers must think about the side-to-move frame flip.
    - **Fully-convolutional 1×1 policy head (Phase 4.3).** The old FC
      head (`128 × 64 → 4096`, ~528K params) was replaced with a 1×1
      conv `128 → 76` (~9.8K params, ~50× smaller). Translation
      equivariance is preserved end-to-end, which matches modern lc0
      practice and was the motivation cited in the ML review.
    - **20-plane input (Phase 3).** Planes 18 (≥1× before) and 19 (≥2×
      before) feed the network threefold-repetition context.
      Implementation reuses the engine's existing `positionCounts`
      table — no Zobrist machinery was needed (see `dcm_architecture_v2.md`
      Phase 1 for why we deviated from the originally-planned Zobrist
      path).
    - **Fresh-baseline value targets (Post-impl Addendum A).** The old
      `vBaseline` was a frozen self-play-time value; the trainer now
      runs an extra forward-only pass on its *current* network before
      each training step to recompute per-position `v(s)`, then
      overwrites the play-time staging before feeding the training
      graph. `MPSGraphGradientSemanticsTests` verified the
      placeholder-boundary stop-gradient semantics (MPSGraph has no
      `stop_gradient` op, and `with`-array exclusion does not prune
      backward-pass paths — the placeholder feed is the only correct
      way). Cost is ~33% extra forward FLOPs per training step;
      diagnostic `vBaselineDelta` now appears in `[STATS]`.
    - **`maxTensorElementCount` now computed from live arch
      (Phase 7.2).** `ModelCheckpointFile.maxTensorElementCount` is
      derived from `ChessNetwork.channels`, `inputPlanes`,
      `policyChannels`, and `seReductionRatio` (covers stem conv,
      residual conv, policy conv, SE FC). Acts as a defense-in-depth
      sanity cap on per-tensor element counts during `.dcmmodel`
      load — rejects implausible sizes before allocation, even when
      the SHA-256 trailer happens to match. Auto-tracks any future
      architecture change, no manual bump needed.

  The v2 bundle also introduced a `ReplayBuffer` format v3 (old
  buffers rejected cleanly on load), arch-hash bump on `.dcmmodel`
  (old checkpoints rejected with `.archMismatch`), and the first
  XCTest target. `CHANGELOG.md` has the commit-level breakdown;
  `dcm_architecture_v2.md` has the phase-by-phase design and a
  consolidated "Current state (as-built)" section that includes the
  post-impl follow-ups (sampling tau bump to 2.0, Candidate Test RAW-
  cell top-K display, entropy-alarm threshold) *and* the four post-v2
  commits listed below, plus a full parameter-defaults table.

  **Post-v2 follow-ups shipped during the first v2 run** (commits
  `9298273` → `068f805` → `7757418` → `cf1cc24`, all 2026-04-20):

    - **Advantage standardization + K dropped 50 → 5.** The policy-
      gradient weight is now `A_norm = (A − mean(A)) / sqrt(var(A) + 1e-6)`
      computed per batch inside the graph, autograd-safe because `A`
      depends only on the `z` and `vBaseline` placeholders. Removes the
      systematic bias when the value head has a global offset (e.g.
      `E[v] ≈ 0.45` once draws dominated self-play). With `A_norm`
      already at unit stdev, the pre-standardization `policyScaleK = 50`
      was pinning `gradClipMaxNorm` almost every step; dropped to `5.0`.
    - **Live-editable hyperparameters.** `weightDecayC`, `gradClipMaxNorm`,
      `policyScaleK`, `learnRate`, `entropyRegCoeff`, `drawPenalty`, and
      both sampling schedules are now fed to the training graph per step
      via scalar placeholders, so UI edits commit immediately without
      rebuilding the graph. Values persist in `@AppStorage` where
      applicable and are restored on session load. Every commit writes
      a `[PARAM] name: old -> new` line. `SessionCheckpointState` gained
      `policyScaleK: Float?` (Optional for back-compat) and the full
      `wd / clip / K / sp+ar tau` set on load.
    - **Diagnostics expansion.** New `TrainStepTiming` fields:
      `playedMoveProb`, `policyLogitAbsMax`, `policyHeadWeightNorm`,
      advantage distribution (`mean / std / min / max / fracPos / fracSmall`),
      plus `p05 / p50 / p95` from a rolling raw-advantage ring.
      Separate `legalMassSnapshot` probe (legal-mass + top1-legal) via
      `BoardEncoder.decodeSynthetic`, refreshed every 25 steps during
      bootstrap. `ParallelWorkerStatsBox` gained a 512-entry game-length
      ring (`p50 / p95 / avgLen`). `MPSChessPlayer` now counts
      "randomish" plies where post-temperature max probability is below
      `1.5 / N_legal` (policy-collapse signal independent of tau).
      `[STATS]` emitter restructured into bootstrap (per-step, first 500
      steps) + steady-state (60 s) phases.
    - **Full-sort top-K for catastrophic collapse.**
      `ChessRunner.extractTopMoves` now sorts the full 4864-cell policy
      vector rather than capping at `count × 4`, so a collapsed policy
      whose top cells are all off-board still produces `count` legal
      visualizations instead of an empty Candidate Test panel.
    - **MPSGraph reshape layout + sign-consistency tests.** New
      `MPSGraphReshapeLayoutTests` empirically verifies the policy head's
      `[B, 76, 8, 8] → [B, 4864]` reshape is NCHW row-major under
      `c·64 + r·8 + col` (plus end-to-end through `oneHot` +
      `softMaxCrossEntropy`). New `SignConsistencyTests` covers encoder
      symmetry, policy-index symmetry for mirrored moves, outcome-sign
      truth table, advantage-formula sign convention, geometric-decode
      round-trip, and bit-identical network output for bit-identical
      inputs.
    - **Advantage raw ring capped at 32K.** The `_advRawRing`
      in `TrainingLiveStatsBox` was originally sized `rollingWindow ×
      batchSize = 512 × 4096 ≈ 2 M Float`. `snapshot()` sorts the full
      filled portion for percentile extraction, and the 10 Hz UI
      heartbeat's `Task { @MainActor }` calls `snapshot()` via
      `queue.sync` — once the ring filled (~step 500) each sort cost
      ~150 ms on main, saturating the main actor. Because
      `fireCandidateProbeIfNeeded` is `@MainActor` and awaited after
      every training step, training throughput collapsed from
      ~2300 moves/sec to ~300 moves/sec and the UI went non-responsive.
      `advRawRingMaxCapacity = 32_768` drops sort cost from ~150 ms to
      ~1 ms while keeping percentile error below 0.5 % for a
      log-eyeballed diagnostic.

  **Default-parameter drift to record** (relative to this ROADMAP's
  earlier "N-worker concurrent self-play" entry):
  `initialSelfPlayWorkerCount` is now `24` (was `6`),
  `absoluteMaxSelfPlayWorkers` is now `64` (was `16`),
  `trainingBatchSize` is `4096`, `replayBufferCapacity` is `1_000_000`,
  `tournamentGames` is `200`, `tournamentPromoteThreshold` is `0.55`.
  The memory-vs-latency analysis in that older entry still applies —
  the numbers scaled up, the trade-off didn't change. The full current
  parameter-default table lives in `dcm_architecture_v2.md` under
  "Current parameter defaults".

  **Still open after v2.** `TODO_NEXT.md` #3 (ReplayBuffer durability —
  `fsync` + length invariant + reordered atomic save) remains
  unaddressed; adaptive LR schedule remains a design-only entry.
  One runtime regression observed but not yet root-caused: an
  `Unsupported MPS operation mps.placeholder` assertion during
  training after the live-hyperparameters change in `7757418`.
