# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

A from-scratch self-play chess engine written in Swift/SwiftUI for macOS. The neural network runs on MetalPerformanceShadersGraph (MPSGraph) on Apple Silicon.

**This project does not use MCTS.** There is no tree search of any kind — no MCTS, no alpha-beta, no minimax, no rollouts. Move selection is a single forward pass: network emits 4864 policy logits (76 channels × 64 squares, AlphaZero-shape encoding) + a value scalar, the CPU masks illegal moves, temperature-scales, softmaxes, and categorical-samples. That's it. Do not add search, and do not suggest "for MCTS you'd…" style edits — AlphaZero-style search is an explicit non-goal. Strength comes entirely from the network itself, bootstrapped through self-play + arena promotion. See `documentation/chess-engine-design.md` ("My Goal") and `documentation/sampling-parameters.md` ("Sampling method") for the explicit rationale.

There is also no opening book and no human training data.

The app is a single-window macOS SwiftUI app used as an interactive training console: build a fresh network, run Play-and-Train (N concurrent self-play workers feeding a replay buffer while a trainer consumes batches), trigger an Arena to measure candidate vs. champion, and promote if score ≥ threshold.

## Build / run

Always use **drews-xcode-mcp** for building or running. Never invoke `xcodebuild` / `swift build` directly.

- Project: `DrewsChessMachine/DrewsChessMachine.xcodeproj` (scheme `DrewsChessMachine`, macOS only).
- A pre-Compile Run Script phase invokes `DrewsChessMachine/generate-build-info.sh`, which bumps `DrewsChessMachine/build_counter.txt` and regenerates `DrewsChessMachine/DrewsChessMachine/BuildInfo.swift` every build. Both files are expected to show up as modified after any build — never edit `BuildInfo.swift` by hand, and don't fight the counter changes.
- XCTest target `DrewsChessMachineTests` exists and covers the pure-logic components (PolicyEncoding bijection, BoardEncoder planes, repetition tracking, ReplayBuffer, MPSGraph gradient/reshape semantics, legal-move validation, sign consistency, ArenaEloStats, ChartZoom stops). Add tests for any new pure-logic component that has correctness invariants. Higher-level behaviors that require Metal/MPSGraph setup still rely on the Engine Diagnostics UI button and session-log observation rather than XCTest.

### Running the tests

The full suite is **slow — roughly an hour cold** on this machine: a cold build-for-testing of the large test target dominates, and on top of that test *execution* alone is ~19 min (865 cases). So:

- **Most of the time, run only the specific test(s) / class you're touching.** The cold test-target build is the real cost; once it's warm, incremental rebuilds + targeted `-only-testing:DrewsChessMachineTests/<Class>[/<method>]` runs are quick. Run the **full suite only occasionally** — before merging, or after changes to the graph builders (`ChessNetwork`), training math (`ChessTrainer`), persistence/serialization, or move generation.
- **Heavy *forensic* suites are gated off by default** behind the `DCM_RUN_SLOW_TESTS` env var (`1`/`true`). They are diagnostic bug-maps, not correctness gates, and dominate execution time. Gated today (see `DrewsChessMachineTests/SlowTestGate.swift`): `MacOS27NaNIsolationTests` (**~11.6 min — ~60% of the whole exec time**; the 63-case precision×batch×step NaN-isolation matrix, several cells 1m+ each) and `ConvKernelExecutionPathNumericsTests` (~37 s — the Winograd-blowup refutation). Gating both drops default exec from ~19 min to **~7 min**. To include them, set the var in the scheme's Test action (Edit Scheme… ▸ Test ▸ Arguments ▸ Environment Variables) or the driving shell: `DCM_RUN_SLOW_TESTS=1 xcodebuild test …`.
- **Other heavy-but-core suites are NOT gated** (they assert real correctness, so they stay in the default run): `PolicyHeadCorrectnessTests` (~92 s), `CheckpointManagerSafetensorsTests` (~89 s, bit-exact round-trips), `MomentumOptimizerTests` (~86 s), `RuntimeArchReachTests` (~52 s). If one becomes a problem, prefer trimming its slowest cases over gating the whole class.
- **Gating a new forensic suite:** add `override func setUpWithError() throws { try SlowTestGate.requireEnabled("<label>") }` to the class — it skips cleanly with a reason naming the env var. Don't gate a suite that pins a correctness invariant.

## Where to look for runtime state

The app terminal console only shows SwiftUI chart warnings and bring-up noise. All meaningful runtime telemetry goes to the session log:

- `~/Library/Logs/DrewsChessMachine/dcm_log_YYYYMMDD-HHMMSS.txt` (one file per launch).
- `drews-xcode-mcp`'s `get_runtime_output` only returns output after the app has terminated; while the session is running, read the session log file directly.
- Every log line is timestamped. Tags to look for: `[APP]` (launch banner with build+git), `[BUTTON]` (user actions), `[STATS]` (periodic training snapshot — one line per training step for the first 500 steps, then one per 60 seconds), `[ARENA]` (arena start/end, W/L/D, kept vs promoted), `[ALARM]` (e.g. policy entropy below threshold), `[CHECKPOINT]` (autosaves), `[BATCHER]` (batched-eval startup correctness probe).

## Training parameters

All tunable training parameters live in a single `@MainActor @Observable` singleton: `TrainingParameters.shared` (`DrewsChessMachine/DrewsChessMachine/Training/TrainingParameters.swift`). Each parameter is declared via the `@TrainingParameter` macro (in the local SwiftPM package `DrewsChessMachine/Packages/TrainingParametersMacro/`) which generates the id, definition (with range + category + liveTunable flag), and typed encode/decode.

### Adding / removing / renaming a parameter — full checklist

The macro covers id + persistence + JSON encode/decode, but several touchpoints are still manual. **Walk all of these every time a parameter is added, removed, or renamed** — drift between them tends not to fail the build, just silently desync at runtime.

1. **Declare it.** Write a `@TrainingParameter(...) public enum FooBar: TrainingParameterKey {}` declaration and add the type to `allKeys` in `TrainingParameters.swift`.
2. **Wire the singleton.** Add a stored property + matching entries in `collectValues` / `applyOne` (and the snapshot's read accessor).
3. **`parameters.json` save/load.** Confirm the new key appears in `--show-default-parameters` output and that the round-trip of `--create-parameters-file` → manual edit → reload behaves. (Macro-generated, but verify nothing is hand-listed in the CLI path.)
4. **Session save/load (`.dcmsession`).** Add an Optional field to `SessionCheckpointState` in `Persistence/SessionCheckpointFile.swift` (Optional so older sessions still decode), pass it through `buildCurrentSessionState` in `SessionController+Checkpoint.swift`, and add a `[RESUME-PARAM]` block in `SessionController+Training.swift` that writes the saved value back onto `TrainingParameters.shared` (and, where applicable, onto the trainer) — mirroring the `batchStatsInterval` block. Both the "from session" and "saved=nil (defaulted)" branches must log.
5. **`results.json` / `CliTrainingRecorder`.** If the parameter influences sampling, training math, or anything a CLI/autotrain run cares to compare, surface it in the recorder's snapshot (e.g. `BatchStatsSnapshot.samplingConstraints`) so it shows up in `results.json` alongside metrics.
6. **Runtime log.** Make sure the parameter's *value* (or an observable derived from it) is visible in `[STATS]`, `[SAMPLER]`, `[BATCH-STATS]`, or a similar tag — otherwise a misconfiguration is invisible while the run is in progress.
7. **UI position.** Decide where the parameter lives in `TrainingSettingsPopover.swift` (or another popover/control if it's not a training-loop knob), and add a binding + validation entry to `TrainingSettingsPopoverModel.swift`. A parameter with no UI is reachable only via `parameters.json` editing — fine for autotrain-only knobs but make that choice deliberately.
8. **Live tunability.** If `liveTunable: true`, confirm the consumer re-reads from `TrainingParameters.shared` on a periodic reconcile loop instead of caching it in a snapshot at session start.
9. **Renames.** Update the parameter's `id` (snake_case key in `UserDefaults` and JSON) **and** the Swift property name. Old `UserDefaults` entries under the previous id will then be ignored on load and the user's preference is silently reset to default — note this in the commit message; do not write migration code unless the user asks.

Reading values:
- **From SwiftUI views**: `@Bindable var trainingParams = TrainingParameters.shared`, then read `trainingParams.entropyBonus` or bind `$trainingParams.entropyBonus`. Re-renders fire automatically.
- **From off-main / structured-concurrency code**: take a snapshot at session boundary — `let p = await TrainingParameters.shared.snapshot()` — then `p.entropyBonus`. The snapshot is `Sendable`, immutable, and lock-free; mid-iteration UI changes are picked up next snapshot. Most parameters are flagged `liveTunable` (≈52 of ~62), but only a handful are consumed by long-running loops that must re-read live — `selfPlayConcurrency`, `trainingStepDelayMs`, `replayRatioTarget`, `replayRatioAutoAdjust`, `periodicAutosaveIntervalSec`; for those, running consumers re-read from `TrainingParameters.shared` on a periodic reconcile loop instead of using a snapshot.

Persistence is automatic: every property `didSet` writes to `UserDefaults` under the parameter's id, and the next `init` reads it back (validated). There is no `@AppStorage` for training parameters anywhere — the singleton owns all persistence.

CLI flags for emitting defaults:
- `DrewsChessMachine --show-default-parameters` — flat snake_case JSON to stdout, descriptions to stderr; sub-second exit, no GUI.
- `DrewsChessMachine --create-parameters-file [path] [--force]` — writes both `parameters.json` and `parameters.md` (categorized doc).

## Saved model state

`CheckpointManager` writes both single-model (`.safetensors`; legacy `.dcmmodel` still readable) and full-session (`.dcmsession`) checkpoints under `~/Library/Application Support/DrewsChessMachine/{Models,Sessions}/`. **Nothing is ever overwritten** — every save is a new file, naming scheme `<YYYYMMDD-HHMMSS>-<modelID>-<trigger>.<ext>`. See ROADMAP.md for the full design including the bit-exact forward-pass verification that runs on every save.

Three triggers produce a `.dcmsession` — the trigger tag appears in the filename, the status bar, and the `[CHECKPOINT] Saved session (<trigger>): …` log line so every save is grep-distinct:

- **`manual`** — user clicked File > Save Session.
- **`post-promotion`** — fires automatically after each arena promotion (on by default; `autosaveSessionsOnPromote`). Re-uses the weight snapshots taken under the arena's self-play and training pauses.
- **`periodic`** — interval autosave while Play-and-Train is active (cadence = the `periodicAutosaveIntervalSec` parameter, default 4 hours; live-tunable from the settings popover's **Sessions** tab — the heartbeat re-anchors the running `PeriodicSaveController` via `updateInterval`). The controller defers a deadline crossing that lands inside an arena, then either swallows it (if a post-promotion save landed during the deferred window) or fires a little late (otherwise). Any successful save of any trigger resets the clock. After each successful periodic save, `CheckpointPaths.prunePeriodicAutosaves` enforces the `maxPeriodicAutosavesKept` retention cap (0 = unlimited), deleting the oldest `-periodic.dcmsession` directories beyond the cap — manual and post-promotion saves are never pruned.

The most recent save's path is persisted to `UserDefaults` as a `LastSessionPointer`. On app launch, if the pointer's target folder still exists, a sheet offers one-click "Resume Training" with a live 30-second countdown that auto-fires if the user doesn't interact; the File menu item "Resume Training from Autosave" covers the same flow for the rest of the launch. Load failures surface via `setCheckpointStatus(.error)` and stop — the session is never auto-deleted on a failed load (the user may want to repair the folder manually). Pointers whose target was deleted externally are cleared on first observation so they don't re-prompt.

## High-level architecture

### The self-play → train → arena loop

One run of Play-and-Train spins up, in parallel:

1. **N concurrent self-play games** (driven by the tick-based `BatchedSelfPlayDriver`, live-tunable via a Stepper in the UI, bounded by `UpperContentView.absoluteMaxSelfPlayWorkers`). One Swift task runs the outer loop; each *tick* advances all K games one ply in lockstep — parallel-encode the K positions, issue a single batched `network.evaluateBatched(...)`, parallel-sample + apply one move per game, then a serial game-end pass flushes completed games into `ReplayBuffer` in one bulk copy. Per-game state lives on an `ActiveGame` (no per-game `Task`, no actor barrier).
2. **One trainer** (`ChessTrainer`) that pulls minibatches from `ReplayBuffer.sample(count:)` and runs MPSGraph SGD steps on a separate training-mode copy of the network.
3. **Replay ratio controller** (`ReplayRatioController`) that auto-adjusts `stepDelay` so `cons/prod` approaches the configured target (default 1.0). The `[STATS]` line reports `ratio=(target=... cur=... prod=... cons=... auto=on/off delay=XXms)`.
4. **Arena** on demand (`TickTournamentDriver` via the Run Arena button). Pauses self-play via `selfPlayGate` and training via `trainingGate`, snapshots the current trainer weights into a dedicated candidate inference network, and plays a fixed-game tournament candidate-vs-champion (candidate on one network, a dedicated `arenaChampionNetwork` holding a snapshot of champion weights on the other) using the arena `SamplingSchedule`. If score ≥ `promoteThreshold` (default 0.53), the candidate's weights are copied into **both** the live champion (`network`) and the live trainer (`trainer.network`), so both lineages converge on the arena-validated snapshot. Champion inherits the candidate's ModelID; trainer gets a freshly-minted next-generation trainer ID forked from the promoted champion. `CheckpointManager` writes a `-promote.dcmsession` snapshot when `autosaveSessionsOnPromote` is on.

### Networks are singular

A session holds exactly:
- `network` — the live champion. Also what every self-play game is evaluated against, via the tick driver's single batched `evaluateBatched` call per tick. Source of the arena-champion snapshot.
- `trainer.network` — internal to `ChessTrainer`, training-mode BN. The single source of weights for arena candidates. Forked from `network` on a fresh start, or loaded from the session's safetensors trainer weights on session resume, or overwritten by a promoted candidate's weights after an arena win.
- `candidateInferenceNetwork` — inference-mode, persists for the life of the app (lazy-built on first Play-and-Train start, reused across sessions). Receives the trainer's current weights at each arena start.
- `arenaChampionNetwork` — inference-mode, also persists for the life of the app. Receives a snapshot of `network`'s weights at each arena start so the arena's "champion side" plays against a stable snapshot while the live champion remains free for continuous self-play.

There are no per-worker inference networks. The original design ran a single self-play worker against `network` directly; the current N-worker setup added the shared batched evaluator rather than fanning out to per-worker networks.

### MoveEvaluationSource abstraction

The interactive single-game players (`MPSChessPlayer`) don't talk to `ChessMPSNetwork` directly — they hold a `MoveEvaluationSource`:

- `DirectMoveEvaluationSource` → single-position `network.evaluate(board:)`. Used by **Play Game / Human-vs-Network and `--uci`** (the latter via `UCIEngine`; `LiveTrainerMoveEvaluationSource` / `UIGatedMoveEvaluationSource` wrap it for the live-trainer and UI-gated variants).
- **Self-play and the arena do NOT use this abstraction.** Both run the tick-based drivers (`BatchedSelfPlayDriver` / `TickTournamentDriver`) that call `network.evaluateBatched(...)` once per tick and sample inline via `MoveSampler`. The older per-slot actor-barrier batcher (`BatchedMoveEvaluationSource`, with its `expectedSlotCount` grow/shrink protocol) was removed when the tick model landed — there is no slot-count deadlock window anymore (grow = append `ActiveGame`s, shrink = `removeLast` the tail).

### Board encoding and policy space

- Input: **30 planes × 8 × 8** NCHW, always from the current player's perspective. Planes 0-15 are pieces + castling, plane 16 en passant, plane 17 halfmove clock normalized as `min(clock, 99) / 99` (Leela convention), planes 18-19 threefold-repetition *counts* (≥1× before, ≥2× before), planes 20-29 threefold-repetition *temporal pattern* — plane `20 + i` is all-1 iff the position `i + 1` plies ago is a strict chess-rules duplicate (PositionKey equality on board + STM + castling + EP) of the current position. The 10-ply history window is cleared on any irreversible move (halfmove clock = 0). See `BoardEncoder.swift` and `chess-engine-design.md` for the full plane table.
- Policy output: **4864 logits** = 76 channels × 64 squares. AlphaZero-shape encoding: 56 queen-style (8 directions × 7 distances) + 8 knight + 9 underpromotion (3 pieces × 3 directions) + 3 queen-promotion. Indexed as `channel * 64 + row * 8 + col` in the (vertically-flipped for black) encoder frame. The bijection between `ChessMove` and `(channel, row, col)` lives in `PolicyEncoding.swift` — every site that converts moves ↔ indices must use it (deliberately no `policyIndex` property on `ChessMove` itself, so callers must think about the side to move).
- Value output: a **3-logit win/draw/loss head** (slot order `[win, draw, loss]`), trained with categorical cross-entropy against a one-hot on the game result. Everything downstream that wants "am I winning?" as a scalar reads the derived `v = softmax(logits)·[+1, 0, −1] = p_win − p_loss ∈ [−1, +1]` — a difference of two probabilities, no `tanh`. (Switched from a single `tanh` scalar + MSE on 2026-05-12 because the scalar head went silent on a draw-heavy buffer; see `wdl-value-head.md` and the 2026-05-12 CHANGELOG entry.) Always relative to the current player.
- Network: **architecture is runtime-configurable per model**, not a compile-time constant — every model embeds its full `NetworkArchitecture` (the single source of truth; see `NetworkArchitecture.swift` and `documentation/plans-active/RUNTIME_ARCHITECTURE_CONFIG_PLAN.md`). Shape = stem (inputPlanes→first-group width) → an ordered list of **block groups** (`blockGroups: [BlockGroup]`, each a count + a complete per-group recipe: channels, both conv kernels, SE style/ratio, ReZero α, activation function/style, skip merge, dropout multiplier) → tower-end BN (pre-act tail only) → policy head + value head. Towers may be **heterogeneous** (WRN-style width staircase): where adjacent groups differ in width, that block gets a bias-free 1×1 skip projection. The engine consumes only the flattened `expandedBlocks`; uniform towers (one group) build byte-identically to the pre-block-groups code. The current default preset is `v4_5block_7x7` (5 blocks, 7×7, 128ch, ~8.45M params); the policy head has three styles (simple_conv / intermediate_conv / fc_bottleneck) and the value head is the 3-logit W/D/L head. Build new towers via Build-New-Model (per-group editor + live `ArchitectureDiagramView`). See `ChessNetwork.swift` (graph builders) and `documentation/plans-completed/ARCHITECTURE_EXPANSION_PLAN.md` (block-groups design). SE blocks match modern lc0 practice.
- Legal-move masking is done CPU-side in `MPSChessPlayer.chooseMove` after softmax; the graph emits raw logits, not a masked softmax.

### Sampling (temperature schedules)

Move selection is temperature-softmax over legal-only logits — no top-k, no MCTS. The self-play and arena schedules are **tunable** — built from `TrainingParameters` at session start and held on `MPSChessPlayer.SamplingSchedule` as resolved start/decay/floor. Temperature decays linearly per **game-total ply**: `tau(ply) = max(floor, start − decay·ply)`.
- **self-play** — default start 1.0 → floor 0.5, decay 0.007/ply (`selfPlayStartTau` / `selfPlayTargetTau` / `selfPlayTauDecayPerPly`). Exploration-heavy for replay-buffer coverage; self-play also adds Dirichlet root noise.
- **arena** — default start 0.6 → floor 0.2, decay 0.02/ply (`arenaStartTau` / `arenaTargetTau` / `arenaTauDecayPerPly`). Tighter for signal-to-noise in scoring.
- `.uniform` — flat 1.0, used by Play Game / Forward Pass demo.

(The hardcoded `SamplingSchedule.selfPlay`/`.arena` constants — start 2.0 — are now only fallbacks; the live schedules use the parameters above.)

See `documentation/sampling-parameters.md` for rationale.

### ModelID identity

`ModelID` (`yyyymmdd-N-XXXX`) is minted at well-defined events (Build, Play-and-Train start, arena snapshot) and inherited verbatim on most weight copies. The mint/inherit rules aren't obvious — **read `sampling-parameters.md` before adjusting when IDs change**. Every `[STATS]` and `[ARENA]` line reports `trainer=...`, `champion=...`, and (during arena) `candidate=...` so logs stay traceable back to specific weight snapshots.

## Reference docs in-repo

- `documentation/chess-engine-design.md` — the original design document (input encoding, network topology, MPSGraph choices). Written as a learning narrative, but accurate and load-bearing.
- `documentation/sampling-parameters.md` — temperature schedule design, ModelID mint/inherit rules, diversity tracking.
- `documentation/mpsgraph-primitives.md` — cookbook for the MPSGraph APIs actually used. Useful when editing `ChessNetwork.swift`.
- `documentation/disk-cleanup.md` — maintenance runbook for reclaiming disk from `Sessions/`: why `manual`/`promote` saves accumulate, the Time Machine local-snapshot reclaim step (deletion alone frees nothing until `tmutil thinlocalsnapshots`), and a guarded keeper-selection policy.
- `documentation/v5-lineage.md` — the consolidated v5 training record: the 8-segment / 2-machine chain with verified modelID links and every `cumstep_base` derivation, the three chart axes, the surviving-checkpoint inventory, run 4's shard-permission truncation, and the traps (repeated step numbers across segments, filenames not identifying checkpoints). **Read before touching any v5 number** — raw step numbers are ambiguous and cum steps shifted +100,320 from the retired bundle monitor's figures.
- `documentation/UCI.md` — DCM and the UCI protocol, both directions: **DCM as a UCI engine** (`--uci` — options `Model`/`Temperature`, the single-forward-pass `go` that ignores all limits, Temperature-0 determinism) and **DCM driving external engines** (`--train-vs-uci` — CLI spec/per-pool-vs-global, fixed-per-move `go` timing, both-sides distillation, known limitations: UCI-native only, no `go` validation/compliance, bare-`go` pitfall, hardcoded 10 s/30 s timeouts).
- `documentation/cutechess-setup.md` — concrete cutechess-cli match harness for benchmarking DCM vs Stockfish (engine registration, why TC goes on the opponent, mandatory opening book for Temperature-0 determinism, Elo-ladder rating). Linked from UCI.md.
- `ROADMAP.md` — deferred work, completed-with-design-notes, and the save/load design. **Completed items stay — move to "Completed" rather than delete, and preserve detail including any deviations from the original plan.**
- `CHANGELOG.md` — commit-linked log of meaningful changes. Newest first, timestamped CDT, git-hash tagged.

## Concurrency invariants

- Most long-lived objects are `final class @unchecked Sendable` with an internal `OSAllocatedUnfairLock` (or `SyncBox<T>`, the project's tiny wrapper over `OSAllocatedUnfairLock<T>` at `Utils/SyncBox.swift`) — not actor-isolated. When editing any of them (`ReplayBuffer`, `ChessNetwork`, `ChessMPSNetwork`, `SessionLogger`, `ParallelWorkerStatsBox`, `GameWatcher`, `ReplayRatioController`, `GameDiversityTracker`, the `TrainingLiveStatsBox` inside `ChessTrainer.swift`, all the small `*Box`/`*Flag`/`*Gate` classes in `Training/` and `Arena/`, etc.), preserve the lock discipline in comments rather than converting to an actor. Do NOT use raw `os_unfair_lock` (unsafe in Swift) and do not introduce new `NSLock`s — `OSAllocatedUnfairLock` is ~20× faster than `DispatchQueue.sync` even uncontended and is the project standard. The serial `DispatchQueue`s that remain (`ChessTrainer.executionQueue`, `ChessNetwork.executionQueue`, `ChessMachine.delegateQueue`, `SessionLogger.queue`) are *work executors* — they exist to bridge structured concurrency to long-running synchronous GPU/IO work, not to protect data — and they stay.
- Structured concurrency rule from the user's global instructions applies strongly here: **don't do long-running synchronous work inside a `Task`**. MPSGraph `.run` is the usual offender — the network wraps it in `CheckedContinuation` + `DispatchQueue` so the Swift concurrency thread keeps making progress. Follow the same pattern for any new GPU path.
- Delegate methods on `ChessMachine` fire on a private serial `.userInteractive` dispatch queue (not the main actor). Anything touching SwiftUI state must `Task { @MainActor in ... }`.

## Training observability worth knowing about

The `[STATS]` line carries a dense set of counters. A few that matter for diagnosing training health:
- `pLoss` — outcome-weighted policy cross-entropy. **Unbounded on both sides** (negative is fine when well-predicted winning plays dominate). Read alongside `pEnt`, not in isolation.
- `pEnt` — mean Shannon entropy of the policy softmax, in nats. `log(4864) ≈ 8.49` at uniform init for the current 4864-cell policy head. Below `policyEntropyAlarmThreshold` (1.0 in-repo, in `TrainingAlarmController.swift`) triggers `[ALARM] policy may be collapsing`.
- `vMean` / `vAbs` — mean / mean-abs of the derived value scalar `p_win − p_loss` (no tanh). `pW` / `pD` / `pL` — the W/D/L softmax batch-means (sum ≈ 1). The value-head collapse signature is `pD → 1.0` (equivalently `vAbs → 0` and staying there) — the post-WDL "everything is a draw"; watch `pD` falling off its `0.75` bias-init prior as the sign training is working. `vLoss` is now categorical-CE-scale (≈ `[0, ln 3]` at convergence), not the old MSE scale.
- `gNorm` — pre-clip global gradient L2 norm, reported every step. Compare against `ChessTrainer.gradClipMaxNorm`; values above it are clip events, not bugs.
- `diversity=unique=X/Y(%) diverge=N.N` — rolling `GameDiversityTracker` snapshot over the last 200 games; `diverge` is the avg ply at which pairs of games first differ. Steady-state healthy is `[0-5]`-heavy in the histogram tile.

## Run tracking: three axes (architectural decision, 2026-08-11)

The dashboards (`documentation/dashboards/`) plot every run on **three separate x-axes**, and they are not interchangeable. `registry.json` holds per-run segment config; `data/<run>.csv` (columns in `_schema.py`) is the source of truth; `master.py` renders.

- **step** — `cum_step = segment.cumstep_base + training_step`. Exact, but a *work* axis only while batch size and replay ratio hold constant.
- **time** — `elapsed_train_sec`, which is **sleep-clamped**: `_clamped_timeline` banks `min(real_gap, segment_median_s_per_step × Δsteps + 120s)` per interval, one-directionally (faster-than-median intervals pass through uncapped). `wall_sec` is the same walk uncapped, so `wall_sec − elapsed_train_sec` is exactly what the clamp discarded. Keep both; the clamp must stay auditable rather than baked in.
- **compute** — `games_fed`, cumulative corpus games from the runner's own `games=`. The device-independent axis, and the one to reach for when comparing across machines.

Rules that follow from this, and that must not be quietly reversed:

- **Never normalize time across devices.** Where a lineage spans machines the by-time curve genuinely changes slope (v5 breaks 2.59× at cum 100,320, a Pacific-TZ VM on an M5 host → native M4 Pro; that is a VM-vs-native comparison, not a silicon one). That break is hardware and stays visible. With batch and replay ratio fixed, a "normalized seconds" axis reduces to `steps × reference_ms` — a relabelled step axis that adds nothing while destroying the record of real cost.
- **`games_fed` is measured, never modeled.** Where a segment's log is missing, leave it **blank** — not zero, not interpolated from an anchor. A modeled value in a measured column is worse than a gap because it still looks plottable.
- Each segment carries `device`, `model_id`, `games_base`, and pinned `elapsed_base_sec` / `wall_base_sec` in `registry.json`. Pins exist because prior segments' logs may be absent; without them the time axis silently restarts at zero.
- **Identify checkpoints by safetensors `__metadata__` (`model_id` + `training_step`), never by filename.** Segment-local step numbering means names repeat and get overwritten in place. `import-probes` enforces a modelID match per segment and refuses mismatches; `ckpt_inventory.py` builds a manifest from headers.

## Conventions specific to this project

- Most source comments are multi-paragraph design explanations, not function-summary boilerplate. When adding a tricky mechanism, match that style — explain *why*, including failure modes that motivated the design. See `BatchedSelfPlayDriver`'s class doc or `ReplayBuffer`'s class doc for the house style.
- The UI layer follows **one SwiftUI `View` struct per file**. `ContentView.swift` (`App/ContentView.swift`) is just the small composer that owns the shared `ChartCoordinator` and stacks `UpperContentView` over `LowerContentView`. The bulk of the session-lifecycle wiring lives in `App/UpperContentView/UpperContentView.swift` (~9.6k lines, large on purpose). When adding new UI pieces, prefer a new file under `App/UpperContentView/` (or the matching `Views/.../` subdir for chart tiles) over wedging another `View` struct into an existing file. The only property with a `some View` signature allowed on a `View` is `body` itself — helper `@ViewBuilder`/`some View` properties are NOT blessed. Decompose a complex `body` into proper child `View` structs (which may live in the same file as their parent), never into helper view properties.
- New architectural plans go into ROADMAP.md (with user's express permission, per the global rules). Don't silently invent a new markdown doc at the repo root.
- Don't say "draw collapse". You tend to throw this phrase around like salt at a french fry convention and most of the time you're wrong because you never check the fucking data details. You shoot from the hip without paying any fucking attention. If you find yourself thinking about writing 'draw collapse', you've probably fucked up in some way. For example, if you start with a high number of draws -- which every single training round will do -- that's not draw collapse. That's just draws.
- Details matter. Get them right. Verify them and make sure they're right -- Every time.
