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
- XCTest target `DrewsChessMachineTests` exists and covers the pure-logic components (PolicyEncoding bijection, BoardEncoder planes, repetition tracking, ReplayBuffer, MPSGraph gradient/reshape semantics, legal-move validation, sign consistency, ArenaEloStats, ChartZoom stops). Add tests for any new pure-logic component that has correctness invariants. Higher-level behaviors that require Metal/MPSGraph setup still rely on the Engine Diagnostics UI button and session-log observation rather than XCTest.

## Where to look for runtime state

The app terminal console only shows SwiftUI chart warnings and bring-up noise. All meaningful runtime telemetry goes to the session log:

- `~/Library/Logs/DrewsChessMachine/dcm_log_YYYYMMDD-HHMMSS.txt` (one file per launch).
- `xcode-mcp-server`'s `get_runtime_output` only returns output after the app has terminated; while the session is running, read the session log file directly.
- Every log line is timestamped. Tags to look for: `[APP]` (launch banner with build+git), `[BUTTON]` (user actions), `[STATS]` (periodic training snapshot, 15-minute cadence plus every arena boundary), `[ARENA]` (arena start/end, W/L/D, kept vs promoted), `[ALARM]` (e.g. policy entropy below threshold), `[CHECKPOINT]` (autosaves), `[BATCHER]` (batched-eval startup correctness probe).

## Training parameters

All 30 tunable training parameters live in a single `@MainActor @Observable` singleton: `TrainingParameters.shared` (`DrewsChessMachine/DrewsChessMachine/TrainingParameters.swift`). Each parameter is declared via the `@TrainingParameter` macro (in the local SwiftPM package `DrewsChessMachine/Packages/TrainingParametersMacro/`) which generates the id, definition (with range + category + liveTunable flag), and typed encode/decode. Adding a parameter: write a `@TrainingParameter(...) public enum FooBar: TrainingParameterKey {}` declaration plus a stored property + collectValues / applyOne entry in `TrainingParameters`; add the type to `allKeys`.

Reading values:
- **From SwiftUI views**: `@Bindable var trainingParams = TrainingParameters.shared`, then read `trainingParams.entropyBonus` or bind `$trainingParams.entropyBonus`. Re-renders fire automatically.
- **From off-main / structured-concurrency code**: take a snapshot at session boundary — `let p = await TrainingParameters.shared.snapshot()` — then `p.entropyBonus`. The snapshot is `Sendable`, immutable, and lock-free; mid-iteration UI changes are picked up next snapshot. For the four currently-`liveTunable` params (`selfPlayWorkers`, `trainingStepDelayMs`, `replayRatioTarget`, `replayRatioAutoAdjust`), running consumers re-read from `TrainingParameters.shared` on a periodic reconcile loop instead of using a snapshot.

Persistence is automatic: every property `didSet` writes to `UserDefaults` under the parameter's id, and the next `init` reads it back (validated). There is no `@AppStorage` for training parameters anywhere — the singleton owns all persistence.

CLI flags for emitting defaults:
- `DrewsChessMachine --show-default-parameters` — flat snake_case JSON to stdout, descriptions to stderr; sub-second exit, no GUI.
- `DrewsChessMachine --create-parameters-file [path] [--force]` — writes both `parameters.json` and `parameters.md` (categorized doc).

## Saved model state

`CheckpointManager` writes both single-model (`.dcmmodel`) and full-session (`.dcmsession`) checkpoints under `~/Library/Application Support/DrewsChessMachine/{Models,Sessions}/`. **Nothing is ever overwritten** — every save is a new file, naming scheme `<YYYYMMDD-HHMMSS>-<modelID>-<trigger>.<ext>`. See ROADMAP.md for the full design including the bit-exact forward-pass verification that runs on every save.

Three triggers produce a `.dcmsession` — the trigger tag appears in the filename, the status bar, and the `[CHECKPOINT] Saved session (<trigger>): …` log line so every save is grep-distinct:

- **`manual`** — user clicked File > Save Session.
- **`post-promotion`** — fires automatically after each arena promotion (on by default; `autosaveSessionsOnPromote`). Re-uses the weight snapshots taken under the arena's self-play and training pauses.
- **`periodic`** — 4-hour autosave while Play-and-Train is active. Driven by `PeriodicSaveController`; the controller defers a deadline crossing that lands inside an arena, then either swallows it (if a post-promotion save landed during the deferred window) or fires a little late (otherwise). Any successful save of any trigger resets the 4-hour clock.

The most recent save's path is persisted to `UserDefaults` as a `LastSessionPointer`. On app launch, if the pointer's target folder still exists, a sheet offers one-click "Resume Training" with a live 30-second countdown that auto-fires if the user doesn't interact; the File menu item "Resume Training from Autosave" covers the same flow for the rest of the launch. Load failures surface via `setCheckpointStatus(.error)` and stop — the session is never auto-deleted on a failed load (the user may want to repair the folder manually). Pointers whose target was deleted externally are cleared on first observation so they don't re-prompt.

## High-level architecture

### The self-play → train → arena loop

One run of Play-and-Train spins up, in parallel:

1. **N self-play workers** (driven by `BatchedSelfPlayDriver`, live-tunable via a Stepper in the UI, bounded by `ContentView.absoluteMaxSelfPlayWorkers`). Every worker owns a fresh `ChessMachine` per game; all share one `BatchedMoveEvaluationSource` that coalesces N simultaneous per-ply evaluate calls into a single batched `graph.run`. Completed games push their whole position sequence into `ReplayBuffer` in one bulk copy.
2. **One trainer** (`ChessTrainer`) that pulls minibatches from `ReplayBuffer.sample(count:)` and runs MPSGraph SGD steps on a separate training-mode copy of the network.
3. **Replay ratio controller** (`ReplayRatioController`) that auto-adjusts `stepDelay` so `cons/prod` approaches the configured target (default 1.0). The `[STATS]` line reports `ratio=(target=... cur=... prod=... cons=... auto=on/off delay=XXms)`.
4. **Arena** on demand (`TournamentDriver` via the Run Arena button). Pauses self-play via `selfPlayGate` and training via `trainingGate`, snapshots the current trainer weights into a dedicated candidate inference network, and plays a fixed-game tournament candidate-vs-champion (candidate on one network, a dedicated `arenaChampionNetwork` holding a snapshot of champion weights on the other) using the arena `SamplingSchedule`. If score ≥ `promoteThreshold` (default 0.55), the candidate's weights are copied into **both** the live champion (`network`) and the live trainer (`trainer.network`), so both lineages converge on the arena-validated snapshot. Champion inherits the candidate's ModelID; trainer gets a freshly-minted next-generation trainer ID forked from the promoted champion. `CheckpointManager` writes a `-promote.dcmsession` snapshot when `autosaveSessionsOnPromote` is on.

### Networks are singular

A session holds exactly:
- `network` — the live champion. Also what every self-play worker evaluates against, through a shared `BatchedMoveEvaluationSource` barrier batcher. Source of the arena-champion snapshot.
- `trainer.network` — internal to `ChessTrainer`, training-mode BN. The single source of weights for arena candidates. Forked from `network` on a fresh start, or loaded from `trainer.dcmmodel` on session resume, or overwritten by a promoted candidate's weights after an arena win.
- `candidateInferenceNetwork` — inference-mode, persists for the life of the app (lazy-built on first Play-and-Train start, reused across sessions). Receives the trainer's current weights at each arena start.
- `arenaChampionNetwork` — inference-mode, also persists for the life of the app. Receives a snapshot of `network`'s weights at each arena start so the arena's "champion side" plays against a stable snapshot while the live champion remains free for continuous self-play.

There are no per-worker inference networks. The original design ran a single self-play worker against `network` directly; the current N-worker setup added the shared batched evaluator rather than fanning out to per-worker networks.

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

- Most long-lived objects are `final class @unchecked Sendable` with an internal `OSAllocatedUnfairLock` (or `SyncBox<T>`, the project's tiny wrapper over `OSAllocatedUnfairLock<T>` at `Utils/SyncBox.swift`) — not actor-isolated. When editing any of them (`ReplayBuffer`, `ChessNetwork`, `ChessMPSNetwork`, `SessionLogger`, `ParallelWorkerStatsBox`, `GameWatcher`, `ReplayRatioController`, `GameDiversityTracker`, the `TrainingLiveStatsBox` inside `ChessTrainer.swift`, all the small `*Box`/`*Flag`/`*Gate` classes in `Training/` and `Arena/`, etc.), preserve the lock discipline in comments rather than converting to an actor. Do NOT use raw `os_unfair_lock` (unsafe in Swift) and do not introduce new `NSLock`s — `OSAllocatedUnfairLock` is ~20× faster than `DispatchQueue.sync` even uncontended and is the project standard. The serial `DispatchQueue`s that remain (`ChessTrainer.executionQueue`, `ChessNetwork.executionQueue`, `ChessMachine.delegateQueue`, `SessionLogger.queue`) are *work executors* — they exist to bridge structured concurrency to long-running synchronous GPU/IO work, not to protect data — and they stay.
- Structured concurrency rule from the user's global instructions applies strongly here: **don't do long-running synchronous work inside a `Task`**. MPSGraph `.run` is the usual offender — the network wraps it in `CheckedContinuation` + `DispatchQueue` so the Swift concurrency thread keeps making progress. Follow the same pattern for any new GPU path.
- Delegate methods on `ChessMachine` fire on a private serial `.userInteractive` dispatch queue (not the main actor). Anything touching SwiftUI state must `Task { @MainActor in ... }`.

## Training observability worth knowing about

The `[STATS]` line carries a dense set of counters. A few that matter for diagnosing training health:
- `pLoss` — outcome-weighted policy cross-entropy. **Unbounded on both sides** (negative is fine when well-predicted winning plays dominate). Read alongside `pEnt`, not in isolation.
- `pEnt` — mean Shannon entropy of the policy softmax, in nats. `log(4864) ≈ 8.49` at uniform init for the current 4864-cell policy head. Below `policyEntropyAlarmThreshold` (5.0 in-repo, in `ContentView.swift`) triggers `[ALARM] policy may be collapsing`.
- `vMean` / `vAbs` — value-head mean and mean-abs. If `vAbs → 1.0` the tanh has saturated; gradients through it will vanish.
- `gNorm` — pre-clip global gradient L2 norm, reported every step. Compare against `ChessTrainer.gradClipMaxNorm`; values above it are clip events, not bugs.
- `diversity=unique=X/Y(%) diverge=N.N` — rolling `GameDiversityTracker` snapshot over the last 200 games; `diverge` is the avg ply at which pairs of games first differ. Steady-state healthy is `[0-5]`-heavy in the histogram tile.

## Conventions specific to this project

- Most source comments are multi-paragraph design explanations, not function-summary boilerplate. When adding a tricky mechanism, match that style — explain *why*, including failure modes that motivated the design. See `BatchedSelfPlayDriver.stopAll` or `ReplayBuffer`'s class doc for the house style.
- The UI layer follows **one SwiftUI `View` struct per file**. `ContentView.swift` (`App/ContentView.swift`) is just the small composer that owns the shared `ChartCoordinator` and stacks `UpperContentView` over `LowerContentView`. The bulk of the session-lifecycle wiring lives in `App/UpperContentView/UpperContentView.swift` (~9.6k lines, large on purpose). When adding new UI pieces, prefer a new file under `App/UpperContentView/` (or the matching `Views/.../` subdir for chart tiles) over wedging another `View` struct into an existing file. Helper `@ViewBuilder` properties on an existing `View` (added via an extension) are fine to keep in the same file as the View they belong to.
- New architectural plans go into ROADMAP.md (with user's express permission, per the global rules). Don't silently invent a new markdown doc at the repo root.
