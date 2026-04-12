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
