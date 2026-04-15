# Roadmap

Long-term goals, deferred work, and notes on decisions.

## Future improvements

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
  self-play workers (`ContentView.selfPlayWorkerCount`, currently `3`),
  each with its own dedicated `ChessMPSNetwork` instance so no two
  concurrent `evaluate` calls share MPSGraph state. Topology is
  asymmetric: worker 0 reuses the existing `network` (the champion, also
  the arena snapshot source), and workers `1..N-1` use new
  `secondarySelfPlayNetworks` mirrored from the champion at session start
  and at every arena promotion. Each worker owns its own
  `WorkerPauseGate`, so the arena-champion snapshot path (which only
  reads `network`) still pauses only worker 0, and only the promotion
  branch pauses every worker to loadWeights into every self-play
  network. Players (`MPSChessPlayer` white/black) are now allocated
  once per worker and reused across games — `ChessMachine.beginNewGame`
  already calls `onNewGame` on each, which resets per-game scratches
  while keeping backing storage alive. Only worker 0 drives the
  `GameWatcher` live display to avoid two workers fighting over per-game
  state; aggregate self-play rates still accumulate through the
  thread-safe `ParallelWorkerStatsBox`. `ReplayBuffer` already
  serializes `append` calls under its `NSLock`, so multiple concurrent
  writers need no changes there. Setting `selfPlayWorkerCount = 1`
  reproduces the pre-change behavior exactly (modulo the per-game player
  reuse cleanup). Memory cost is ~12 MB per additional inference
  network, trivial on unified memory.
