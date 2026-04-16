# Roadmap

Long-term goals, deferred work, and notes on decisions.

## Future improvements

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
