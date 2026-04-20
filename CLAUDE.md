# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

A from-scratch self-play chess engine written in Swift/SwiftUI for macOS. The neural network runs on MetalPerformanceShadersGraph (MPSGraph) on Apple Silicon.

**This project does not use MCTS.** There is no tree search of any kind — no MCTS, no alpha-beta, no minimax, no rollouts. Move selection is a single forward pass: network emits 4864 policy logits (76 channels × 64 squares, AlphaZero-shape encoding) + a value scalar, the CPU masks illegal moves, temperature-scales, softmaxes, and categorical-samples. That's it. Do not add search, and do not suggest "for MCTS you'd…" style edits — AlphaZero-style search is an explicit non-goal. Strength comes entirely from the network itself, bootstrapped through self-play + arena promotion. See `chess-engine-design.md` ("My Goal") and `sampling-parameters.md` ("Sampling method") for the explicit rationale.

There is also no opening book and no human training data.

The app is a single-window macOS SwiftUI app used as an interactive training console: build a fresh network, run Play-and-Train (N concurrent self-play workers feeding a replay buffer while a trainer consumes batches), trigger an Arena to measure candidate vs. champion, and promote if score ≥ threshold.

## Build / run

Always use **xcode-mcp-server** for building or running. Never invoke `xcodebuild` / `swift build` directly.

- Project: `DrewsChessMachine/DrewsChessMachine.xcodeproj` (scheme `DrewsChessMachine`, macOS only).
- A pre-Compile Run Script phase invokes `DrewsChessMachine/generate-build-info.sh`, which bumps `DrewsChessMachine/build_counter.txt` and regenerates `DrewsChessMachine/DrewsChessMachine/BuildInfo.swift` every build. Both files are expected to show up as modified after any build — never edit `BuildInfo.swift` by hand, and don't fight the counter changes.
- No XCTest target. Testing is manual: build, run, exercise the UI, read the session log.

## Where to look for runtime state

The app terminal console only shows SwiftUI chart warnings and bring-up noise. All meaningful runtime telemetry goes to the session log:

- `~/Library/Logs/DrewsChessMachine/dcm_log_YYYYMMDD-HHMMSS.txt` (one file per launch).
- `xcode-mcp-server`'s `get_runtime_output` only returns output after the app has terminated; while the session is running, read the session log file directly.
- Every log line is timestamped. Tags to look for: `[APP]` (launch banner with build+git), `[BUTTON]` (user actions), `[STATS]` (periodic training snapshot, 15-minute cadence plus every arena boundary), `[ARENA]` (arena start/end, W/L/D, kept vs promoted), `[ALARM]` (e.g. policy entropy below threshold), `[CHECKPOINT]` (autosaves), `[BATCHER]` (batched-eval startup correctness probe).

## Saved model state

`CheckpointManager` writes both single-model (`.dcmmodel`) and full-session (`.dcmsession`) checkpoints under `~/Library/Application Support/DrewsChessMachine/{Models,Sessions}/`. **Nothing is ever overwritten** — every save is a new file, naming scheme `<YYYYMMDD-HHMMSS>-<modelID>-<trigger>.<ext>`. See ROADMAP.md for the full design including the bit-exact forward-pass verification that runs on every save. Autosave on arena promotion is on by default.

## High-level architecture

### The self-play → train → arena loop

One run of Play-and-Train spins up, in parallel:

1. **N self-play workers** (driven by `BatchedSelfPlayDriver`, live-tunable via a Stepper in the UI, bounded by `ContentView.absoluteMaxSelfPlayWorkers`). Every worker owns a fresh `ChessMachine` per game; all share one `BatchedMoveEvaluationSource` that coalesces N simultaneous per-ply evaluate calls into a single batched `graph.run`. Completed games push their whole position sequence into `ReplayBuffer` in one bulk copy.
2. **One trainer** (`ChessTrainer`) that pulls minibatches from `ReplayBuffer.sample(count:)` and runs MPSGraph SGD steps on a separate training-mode copy of the network.
3. **Replay ratio controller** (`ReplayRatioController`) that auto-adjusts `stepDelay` so `cons/prod` approaches the configured target (default 1.0). The `[STATS]` line reports `ratio=(target=... cur=... prod=... cons=... auto=on/off delay=XXms)`.
4. **Arena** on demand (`TournamentDriver` via the Run Arena button). Pauses every self-play worker via `WorkerPauseGate`, snapshots the current trainer weights into a candidate inference network, and plays a fixed-game tournament candidate-vs-champion using the arena `SamplingSchedule`. If score ≥ `promoteThreshold` (default 0.55), the candidate becomes the new champion — its weights are loaded into `network` and every `secondarySelfPlayNetworks[i]`, and `CheckpointManager` writes a `-promote.dcmsession` snapshot.

### Networks are plural and pre-allocated

A session holds:
- `network` — the primary inference network (also the arena champion source). Worker 0 uses this one.
- `secondarySelfPlayNetworks[1..absoluteMax-1]` — pre-built at session start and kept alive across Stepper changes. Workers 1..N-1 use these. Mirrored from champion at session start and on every promotion.
- `trainer.network` — internal to `ChessTrainer`, training-mode BN, the single source of weights for arena candidates.
- `candidateNetwork` — ephemeral, alive only during an arena.

Idle workers stay allocated deliberately (memory vs. latency trade — see ROADMAP "N-worker concurrent self-play" for the full rationale).

### MoveEvaluationSource abstraction

`MPSChessPlayer` doesn't talk to `ChessMPSNetwork` directly. It holds a `MoveEvaluationSource`:

- `DirectMoveEvaluationSource` → single-position `network.evaluate(board:)`. Used by arena (one game at a time) and Play Game.
- `BatchedMoveEvaluationSource` → actor-based barrier batcher. N self-play slots park at `evaluate`; when the N-th submission arrives, one `network.evaluate(batchBoards:count:)` fires and every slot resumes with its slice of the policy/value output.

Shrink/grow the slot count carefully: the batcher's `expectedSlotCount` must be lowered **before** waiting on cancelled slots, or the barrier will deadlock — see the comments in `BatchedSelfPlayDriver.stopAll` / reconcile loop.

### Board encoding and policy space

- Input: **20 planes × 8 × 8** NCHW, always from the current player's perspective. Planes 0-15 are pieces + castling, plane 16 en passant, plane 17 halfmove clock normalized as `min(clock, 99) / 99` (Leela convention), planes 18-19 threefold-repetition signals (≥1× before, ≥2× before). See `BoardEncoder.swift` and `chess-engine-design.md` for the full plane table.
- Policy output: **4864 logits** = 76 channels × 64 squares. AlphaZero-shape encoding: 56 queen-style (8 directions × 7 distances) + 8 knight + 9 underpromotion (3 pieces × 3 directions) + 3 queen-promotion. Indexed as `channel * 64 + row * 8 + col` in the (vertically-flipped for black) encoder frame. The bijection between `ChessMove` and `(channel, row, col)` lives in `PolicyEncoding.swift` — every site that converts moves ↔ indices must use it (deliberately no `policyIndex` property on `ChessMove` itself, so callers must think about the side to move).
- Value output: single `tanh` scalar in `[-1, +1]`, always relative to the current player.
- Network: stem (20→128) → 8 residual blocks (each with an SE channel-attention module, reduction ratio 4) → fully-convolutional policy head (1×1 conv 128→76) + value head → ~2.4M parameters. See `ChessNetwork.swift`. SE blocks match modern lc0 practice; the fully-conv policy head replaces the prior FC bottleneck head.
- Legal-move masking is done CPU-side in `MPSChessPlayer.chooseMove` after softmax; the graph emits raw logits, not a masked softmax.

### Sampling (temperature schedules)

Move selection is temperature-softmax over legal-only logits — no top-k, no MCTS. Schedules live on `SamplingSchedule`:
- `.selfPlay` — tau 1.0 → 0.4 over 20 plies per player. Exploration-heavy for replay-buffer coverage.
- `.arena` — tau 1.0 → 0.2 over 20 plies per player. Tighter for signal-to-noise in scoring.
- `.uniform` — flat 1.0, used by Play Game / Forward Pass demo.

See `sampling-parameters.md` for rationale.

### ModelID identity

`ModelID` (`yyyymmdd-N-XXXX`) is minted at well-defined events (Build, Play-and-Train start, arena snapshot) and inherited verbatim on most weight copies. The mint/inherit rules aren't obvious — **read `sampling-parameters.md` before adjusting when IDs change**. Every `[STATS]` and `[ARENA]` line reports `trainer=...`, `champion=...`, and (during arena) `candidate=...` so logs stay traceable back to specific weight snapshots.

## Reference docs in-repo

- `chess-engine-design.md` — the original design document (input encoding, network topology, MPSGraph choices). Written as a learning narrative, but accurate and load-bearing.
- `sampling-parameters.md` — temperature schedule design, ModelID mint/inherit rules, diversity tracking.
- `mpsgraph-primitives.md` — cookbook for the MPSGraph APIs actually used. Useful when editing `ChessNetwork.swift`.
- `ROADMAP.md` — deferred work, completed-with-design-notes, and the save/load design. **Completed items stay — move to "Completed" rather than delete, and preserve detail including any deviations from the original plan.**
- `CHANGELOG.md` — commit-linked log of meaningful changes. Newest first, timestamped CDT, git-hash tagged.

## Concurrency invariants

- Most long-lived objects are `final class @unchecked Sendable` with an internal `NSLock` or serial queue — not actor-isolated. When editing any of them (`ReplayBuffer`, `ChessNetwork`, `ChessMPSNetwork`, `SessionLogger`, `ParallelWorkerStatsBox`, `GameWatcher`), preserve the lock discipline in comments rather than converting to an actor.
- Structured concurrency rule from the user's global instructions applies strongly here: **don't do long-running synchronous work inside a `Task`**. MPSGraph `.run` is the usual offender — the network wraps it in `CheckedContinuation` + `DispatchQueue` so the Swift concurrency thread keeps making progress. Follow the same pattern for any new GPU path.
- Delegate methods on `ChessMachine` fire on a private serial `.userInteractive` dispatch queue (not the main actor). Anything touching SwiftUI state must `Task { @MainActor in ... }`.

## Training observability worth knowing about

The `[STATS]` line carries a dense set of counters. A few that matter for diagnosing training health:
- `pLoss` — outcome-weighted policy cross-entropy. **Unbounded on both sides** (negative is fine when well-predicted winning plays dominate). Read alongside `pEnt`, not in isolation.
- `pEnt` — mean Shannon entropy of the policy softmax, in nats. `log(4096) ≈ 8.32` at uniform init. Below `7.0` triggers `[ALARM] policy may be collapsing` (threshold `policyEntropyAlarmThreshold` in `ContentView.swift`).
- `vMean` / `vAbs` — value-head mean and mean-abs. If `vAbs → 1.0` the tanh has saturated; gradients through it will vanish.
- `gNorm` — pre-clip global gradient L2 norm, reported every step. Compare against `ChessTrainer.gradClipMaxNorm`; values above it are clip events, not bugs.
- `diversity=unique=X/Y(%) diverge=N.N` — rolling `GameDiversityTracker` snapshot over the last 200 games; `diverge` is the avg ply at which pairs of games first differ. Steady-state healthy is `[0-5]`-heavy in the histogram tile.

## Conventions specific to this project

- Most source comments are multi-paragraph design explanations, not function-summary boilerplate. When adding a tricky mechanism, match that style — explain *why*, including failure modes that motivated the design. See `BatchedSelfPlayDriver.stopAll` or `ReplayBuffer`'s class doc for the house style.
- `ContentView.swift` is ~6k lines and is the single source of truth for UI + session lifecycle wiring. It's large on purpose; prefer adding a nearby helper over extracting new files unless the new unit has an obvious home.
- New architectural plans go into ROADMAP.md (with user's express permission, per the global rules). Don't silently invent a new markdown doc at the repo root.
