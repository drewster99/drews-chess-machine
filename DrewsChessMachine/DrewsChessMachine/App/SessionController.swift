import SwiftUI
import OSLog

private let logger = Logger(subsystem: "Foo", category: "Bar")

/// Owns long-lived session networks, lifted out of `UpperContentView` as the
/// first slice (Stage 4a) of the session-lifecycle decomposition.
///
/// **Scope of this slice.** `SessionController` holds the three life-of-app
/// inference networks the arena and the candidate-test probe run against
/// (`candidateInferenceNetwork` / `arenaChampionNetwork` / `probeInferenceNetwork`,
/// plus the probe's `ChessRunner`), and the `performBuild()` static used by the
/// build paths. These are the network pieces that have no entanglement with the
/// rest of `UpperContentView`'s `@State` — they're lazily built on the first
/// Play-and-Train start and never torn down, and the view only reads them or
/// nil-coalesces a lazy build.
///
/// **Deliberately still on the view (for now).** The champion `network` + its
/// `runner`, `networkStatus`, `isBuilding`, and the `buildNetwork()` /
/// `ensureChampionBuilt()` flow stay on `UpperContentView` this slice: the
/// champion-network property is named `network` and referenced ~150 places, and
/// moving it cleanly wants a property rename first so a mechanical `network` →
/// `session.network` rewrite can't collide with `network:` argument labels.
/// That, plus the trainer / arena / parallel-stats buckets and the two giant
/// orchestration methods (`startRealTraining`, `runArenaParallel`), land in
/// follow-up Stage 4 slices. At that point the popover / auto-resume /
/// checkpoint controllers can take `weak var session` to drop their own
/// view-capturing closures.
@MainActor
@Observable
final class SessionController {

    // MARK: - Champion network + build state

    /// The live champion network. Self-play workers evaluate against this via
    /// the shared `BatchedMoveEvaluationSource`; Play Game / Run Forward Pass
    /// use it directly; the arena snapshots it into `arenaChampionNetwork`.
    /// `nil` until Build Network (or a load) populates it.
    var network: ChessMPSNetwork?

    /// `ChessRunner` wrapping `network`. Rebuilt whenever `network` is.
    var runner: ChessRunner?

    /// Human-readable build status / error text shown under the Build button.
    var networkStatus: String = ""

    /// True while a `performBuild()` detached task is in flight.
    var isBuilding: Bool = false

    /// Whether a champion network exists. (Was `UpperContentView.networkReady`.)
    var networkReady: Bool { network != nil }

    // MARK: - Inference networks (life-of-app caches)

    /// Inference-mode network used as the arena's "candidate side" player. The
    /// trainer's current SGD weights are copied into it at each arena start so
    /// the candidate plays a coherent, stable snapshot. Built lazily on the
    /// first Play-and-Train start and cached for the life of the app.
    var candidateInferenceNetwork: ChessMPSNetwork?

    /// Inference-mode network dedicated to the candidate-test probe — kept
    /// separate from `candidateInferenceNetwork` so the probe doesn't have to
    /// pause for the whole duration of every arena (which produced a visible
    /// discontinuity in the probe trajectory across arena boundaries). Built
    /// lazily, cached for the app's life.
    var probeInferenceNetwork: ChessMPSNetwork?

    /// `ChessRunner` wrapping `probeInferenceNetwork`, used by
    /// `fireCandidateProbeIfNeeded` via the same `performInference` path as the
    /// forward-pass demo.
    var probeRunner: ChessRunner?

    /// Inference-mode network holding a snapshot of the champion's weights for
    /// the arena's "champion side" — copied once at arena start so the live
    /// champion stays free for continuous self-play during the tournament.
    /// Built lazily, cached for the app's life.
    var arenaChampionNetwork: ChessMPSNetwork?

    // MARK: - Parallel-worker stats / diversity (Stage 4b)

    /// Live-progress snapshot from the parallel self-play workers, mirrored
    /// from `parallelWorkerStatsBox` by the UI heartbeat. `nil` outside of a
    /// Play-and-Train session.
    var parallelStats: ParallelWorkerStatsBox.Snapshot?

    /// Lock-protected counter box shared across the parallel self-play and
    /// training worker tasks. Workers call `recordSelfPlayGame` /
    /// `recordTrainingStep`; the heartbeat polls `snapshot()` and mirrors into
    /// `parallelStats`. Created on Play-and-Train start, `nil` otherwise.
    var parallelWorkerStatsBox: ParallelWorkerStatsBox?

    /// Rolling-window game-diversity tracker for self-play. Fed by every
    /// self-play worker at game end; snapshot polled by the heartbeat for
    /// display and by the stats logger for `[STATS]` lines. `nil` outside a
    /// Play-and-Train session.
    var selfPlayDiversityTracker: GameDiversityTracker?

    // MARK: - Arena coordination boxes (Stage 4b)

    /// Cancellation-aware flag set while an arena tournament is in flight. The
    /// candidate-test probe checks this and skips firing so probe and arena
    /// never contend on the candidate inference network. `nil` between
    /// Play-and-Train sessions.
    var arenaActiveFlag: ArenaActiveFlag?

    /// Trigger inbox the arena coordinator polls — set by the training
    /// worker's auto-interval check and by the Run Arena button. `nil` between
    /// Play-and-Train sessions.
    var arenaTriggerBox: ArenaTriggerBox?

    /// User-override inbox for an in-flight arena. The Abort / Promote buttons
    /// (visible only while an arena is running) write to this box;
    /// `runArenaParallel` polls it to break the game loop early and to branch
    /// on promote-vs-no-promote once the driver returns. `nil` between
    /// Play-and-Train sessions.
    var arenaOverrideBox: ArenaOverrideBox?

    /// `true` while an arena is running — mirror of `arenaActiveFlag` the
    /// heartbeat maintains for UI purposes (disabling Run Arena, suppressing
    /// on-screen probe activity).
    var isArenaRunning: Bool = false

    // MARK: - Trainer + training-run state (Stage 4d)
    //
    // The trainer is built lazily on first use. It owns its own training-mode
    // `ChessNetwork` internally (not shared with `network`), so its weight
    // updates do NOT flow into inference. Only one training mode runs at a
    // time (Train Once / Continuous Training / Play-and-Train), so there's no
    // cross-mode concurrency on the trainer.

    /// The lazily-built trainer. `nil` before the first training start.
    var trainer: ChessTrainer?

    /// Demo "Train Once" in flight.
    var isTrainingOnce: Bool = false

    /// Demo "Continuous Training" (on random data) running.
    var continuousTraining: Bool = false

    /// Handle to the demo continuous-training `Task`. Cancelled on Stop.
    var trainingTask: Task<Void, Never>?

    /// Most recent single training-step timing — mirrored from `trainingBox`
    /// by the heartbeat when the step count advances.
    var lastTrainStep: TrainStepTiming?

    /// Latest training-run stats — mirrored from `trainingBox` by the heartbeat.
    var trainingStats: TrainingRunStats?

    /// Latest training error string, if a step threw. `nil` when healthy.
    var trainingError: String?

    /// Lock-protected live-stats holder shared with the background training
    /// task (continuous or self-play). The worker writes via `recordStep` with
    /// no main-actor hop; the heartbeat polls `snapshot()` and mirrors the
    /// latest values into `trainingStats` / `lastTrainStep` /
    /// `realRollingPolicyLoss` / `realRollingValueLoss` only when the step
    /// count has advanced. `nil` outside a training run.
    var trainingBox: TrainingLiveStatsBox?

    /// `true` while a Play-and-Train (self-play) session is active.
    var realTraining: Bool = false

    /// Handle to the Play-and-Train driver `Task`. Cancelled on Stop.
    var realTrainingTask: Task<Void, Never>?

    /// The replay buffer self-play workers fill and the trainer samples from.
    /// `nil` outside a Play-and-Train session.
    var replayBuffer: ReplayBuffer?

    /// Rolling-window averages of the most recent self-play training losses,
    /// split into the policy (outcome-weighted CE) and value (bounded MSE)
    /// components. Mirrored from `trainingBox` by the heartbeat.
    var realRollingPolicyLoss: Double?
    var realRollingValueLoss: Double?

    /// Live-tunable self-play worker-count box the driver's reconcile loop
    /// reads. `nil` outside a Play-and-Train session.
    var workerCountBox: WorkerCountBox?

    // MARK: - View-facing hooks (Stage 4c)
    //
    // Until the gate / clear-display / trainer-drop logic migrates here too,
    // the build flow reaches the bits that still live on `UpperContentView`
    // through these. Wired in `UpperContentView.handleBodyOnAppear`.

    /// Returns whether some operation that should block a build is in progress
    /// (training, a continuous task, a sweep, a game, a save/load). Mirrors the
    /// view's `isBusy`.
    var isBusyProvider: () -> Bool = { false }

    /// User-facing reason string for the current busy state, shown when
    /// refusing a build. Mirrors the view's `busyReasonMessage()`.
    var busyReasonProvider: () -> String = { "Another operation is in progress." }

    /// Surfaces a "can't do that right now" alert + logs a refused-action
    /// line. Wired to `UpperContentView.refuseMenuAction(_:)`.
    var onRefuseMenuAction: (String) -> Void = { _ in }

    /// Wipes the training/sweep display state. Wired to
    /// `UpperContentView.onClearTrainingDisplay()` — called on Build (and the
    /// auto-build) because rebuilding invalidates the trainer's graph state.
    var onClearTrainingDisplay: () -> Void = { }

    /// Drops the trainer (it owns graph state invalidated by a rebuild). Wired
    /// to `{ trainer = nil }` on the view until the trainer migrates here.
    var onDropTrainer: () -> Void = { }

    /// Returns the live `GameWatcher` (the on-board game state). Stays `@State`
    /// on `UpperContentView` (SwiftUI reconstruction semantics) — `startRealTraining`
    /// reads it through this to pass into the self-play driver / child tasks.
    var gameWatcherProvider: () -> GameWatcher = { GameWatcher() }

    /// The launch-time `--parameters` config (`nil` outside `--parameters` runs).
    /// Read by `startRealTraining`'s headless path. Set in `handleBodyOnAppear`.
    var cliConfig: CliTrainingConfig?

    /// The launch-time `--output` JSON URL (`nil` outside `--output` runs). When
    /// non-nil, `startRealTraining` allocates `cliRecorder`. Set in `handleBodyOnAppear`.
    var cliOutputURL: URL?

    /// The checkpoint controller, so a successful build can reset `lastSavedAt`
    /// (a freshly-built network has never been saved). Weak — the view keeps
    /// sole ownership; safe because both are `@State`-owned by the same
    /// never-deallocated `UpperContentView`.
    weak var checkpoint: CheckpointController?

    /// The chart coordinator (owned by `ContentView`, passed to `UpperContentView`
    /// by `let`). Set in `handleBodyOnAppear`. Strong is fine — `ContentView`
    /// never tears down, so there's no cycle to break; weak would just need an
    /// unwrap at every use. Used by the resume path (`seedChartCoordinatorFromLoadedSession`)
    /// and — once they migrate — by `startRealTraining` / `runArenaParallel` /
    /// `buildCurrentSessionState`.
    var chartCoordinator: ChartCoordinator?

    /// The training-alarm controller (banner + beep + divergence/value-head
    /// detectors). Weak — the view keeps sole ownership; safe because both are
    /// `@State`-owned by the same never-deallocated `UpperContentView`. Set in
    /// `handleBodyOnAppear`. Used by `startRealTraining` (`clear()` /
    /// `resetStreaks()`) and the legal-mass-collapse probe (`raise(...)`) once
    /// those migrate.
    weak var trainingAlarm: TrainingAlarmController?

    // MARK: - Periodic-autosave scheduler state (Stage 4l)

    /// The 4-hour periodic autosave scheduler. Created on Play-and-Train start,
    /// torn down on Stop, `nil` between sessions; polled by the heartbeat ~1 Hz.
    var periodicSaveController: PeriodicSaveController?

    /// Last wall-clock the heartbeat polled `periodicSaveController.decide(now:)`
    /// — throttles the poll to ~1 Hz.
    var periodicSaveLastPollAt: Date?

    /// `true` while a periodic autosave's write is in flight (guards against
    /// double-firing). Separate from `checkpoint.checkpointSaveInFlight` because
    /// a periodic save runs even while the menu items stay enabled.
    var periodicSaveInFlight: Bool = false

    /// Last auto-computed step delay (auto-controller state, persisted across
    /// sessions so the next session resumes where the auto-adjuster left off).
    /// `UserDefaults`-backed (was `@AppStorage("lastAutoComputedDelayMs")` on
    /// the view). Not a training parameter — intentionally NOT in
    /// `TrainingParameters`. Not read during `body`, so a plain computed
    /// (non-observable) UserDefaults accessor is fine.
    var lastAutoComputedDelayMs: Int {
        get { UserDefaults.standard.object(forKey: "lastAutoComputedDelayMs") as? Int ?? 50 }
        set { UserDefaults.standard.set(newValue, forKey: "lastAutoComputedDelayMs") }
    }

    // MARK: - Session-runtime boxes + replay-ratio compensator (Stage 4f)

    /// Live `SamplingScheduleBox` for the current Play-and-Train session — the
    /// `onChange` handlers on the tau fields push freshly-built `SamplingSchedule`s
    /// into it so edits take effect at each slot's next game boundary. `nil`
    /// between sessions.
    var samplingScheduleBox: SamplingScheduleBox?

    /// Worker-0 self-play pause gate for the current session. The checkpoint
    /// save path uses it to briefly pause champion exports. `nil` between sessions.
    var activeSelfPlayGate: WorkerPauseGate?

    /// Training-worker pause gate for the current session. The checkpoint save
    /// path uses it to briefly pause trainer-weight exports. `nil` between sessions.
    var activeTrainingGate: WorkerPauseGate?

    /// Replay-ratio controller (tracks the 1-minute rolling cons/prod ratio and
    /// auto-adjusts the training step delay). Created at session start, polled
    /// by the heartbeat, cleared at session end.
    var replayRatioController: ReplayRatioController?

    /// Latest `replayRatioController` snapshot, mirrored by the heartbeat for UI.
    var replayRatioSnapshot: ReplayRatioController.RatioSnapshot?

    /// Effective replay-ratio set-point (`T_eff`) — the outer integral
    /// compensator's drifted internal target. `nil` when no session is active so
    /// teardown produces a clean re-seed on the next start. See
    /// `updateReplayRatioCompensator`.
    var effectiveReplayRatioTarget: Double?

    /// Wall-clock of the previous compensator tick — drives the `dt` for the
    /// integral update so the gain is `target-units per second`.
    var lastReplayRatioCompensatorAt: Date?

    /// Outer integral compensator for the replay-ratio controller's per-tick
    /// overhead-subtraction bias. Called every heartbeat tick while a session is
    /// live. Wraps the inner controller without modifying it: each tick, observe
    /// `gap = userTarget − snap.currentRatio`, nudge the controller's INTERNAL
    /// set-point `T_eff` in the same sign (`T_eff += k·gap·dt`), bounded to
    /// `[0.5, 5.0]·userTarget`. The user-facing parameter never moves. See the
    /// long derivation comment that was on `UpperContentView` for the full why.
    @MainActor
    func updateReplayRatioCompensator(snap: ReplayRatioController.RatioSnapshot) {
        guard realTraining, let rc = replayRatioController else {
            if effectiveReplayRatioTarget != nil {
                effectiveReplayRatioTarget = nil
                lastReplayRatioCompensatorAt = nil
            }
            return
        }
        let userTarget = TrainingParameters.shared.replayRatioTarget
        guard userTarget > 0 else { return }
        if !snap.autoAdjust {
            if effectiveReplayRatioTarget != nil {
                effectiveReplayRatioTarget = nil
                lastReplayRatioCompensatorAt = nil
            }
            return
        }
        let now = Date()
        guard let prevTeff = effectiveReplayRatioTarget,
              let prevAt = lastReplayRatioCompensatorAt else {
            effectiveReplayRatioTarget = userTarget
            lastReplayRatioCompensatorAt = now
            rc.targetRatio = userTarget
            return
        }
        guard snap.currentRatio > 0 else {
            lastReplayRatioCompensatorAt = now
            return
        }
        let dt = now.timeIntervalSince(prevAt)
        let dtClamped = min(max(dt, 0.0), 1.0)
        let gainPerSecond = 0.05
        let gap = userTarget - snap.currentRatio
        var nextTeff = prevTeff + gainPerSecond * gap * dtClamped
        let lo = 0.5 * userTarget
        let hi = 5.0 * userTarget
        if nextTeff < lo { nextTeff = lo }
        if nextTeff > hi { nextTeff = hi }
        if abs(nextTeff - prevTeff) > 0.001 {
            rc.targetRatio = nextTeff
            effectiveReplayRatioTarget = nextTeff
        }
        lastReplayRatioCompensatorAt = now
    }

    /// How in-memory state from a prior Stop is handled by `startRealTraining`.
    enum TrainingStartMode {
        /// Default path. First launch, or resuming from a
        /// `pendingLoadedSession`. Existing behavior: fresh
        /// replay buffer + counters, trainer weights come from
        /// either the loaded session's `trainer.dcmmodel` or a
        /// fresh fork of champion weights.
        case freshOrFromLoadedSession
        /// User stopped training earlier this launch and wants to
        /// pick up exactly where they left off. Reuse the existing
        /// replay buffer, parallel stats box, training stats box,
        /// session ID, tournament history, chart samples, and
        /// active trainer weights. Open a new training segment.
        case continueAfterStop
        /// User stopped training and wants a fresh session on the
        /// existing trainer's weights. Fresh replay buffer +
        /// counters + session ID, but do NOT overwrite trainer
        /// weights with the champion's.
        case newSessionKeepTrainer
        /// User stopped training and wants a fresh session starting
        /// from the champion — copy champion weights into the
        /// trainer, mint a fresh trainer generation ID, fresh
        /// replay buffer + counters + session ID.
        case newSessionResetTrainerFromChampion
    }

    /// Latest legal-mass snapshot the `[STATS]` logger computed — cached so the
    /// chart-sample heartbeat (faster cadence than the `[STATS]` tick) can render
    /// the legal-masked entropy / legal-mass series. `nil` outside a session.
    var realLastLegalMassSnapshot: ChessTrainer.LegalMassSnapshot?

    /// Whether the app launched with `--train` (headless autotrain). Set in
    /// `handleBodyOnAppear`. `startRealTraining` reads it for the headless path.
    var autoTrainOnLaunch: Bool = false

    // MARK: - Batch-size sweep (Stage 4p)

    /// Batch-size-sweep running flag.
    var sweepRunning = false
    /// The sweep driver `Task` (cancel via `stopSweep`).
    var sweepTask: Task<Void, Never>?
    /// Completed sweep rows (throughput table).
    var sweepResults: [SweepRow] = []
    /// Live sweep progress mirrored from `sweepCancelBox` by the heartbeat.
    var sweepProgress: SweepProgress?
    /// Cancel box the sweep worker polls between steps.
    var sweepCancelBox: CancelBox?
    /// Device memory caps snapshot taken at sweep start (for the header).
    var sweepDeviceCaps: MetalDeviceMemoryLimits?
    nonisolated static let sweepSizes: [Int] = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
    nonisolated static let sweepSecondsPerSize: Double = 1.0

    // MARK: - Sweep Actions

    func startSweep() async {
        SessionLogger.shared.log("[BUTTON] Sweep Batch Sizes")
        guard let trainer = ensureTrainer() else { return }
        onResetBoardDisplay()
        onClearTrainingDisplay()
        sweepRunning = true
        // Snapshot device caps once at sweep start so the header has a
        // stable reference point regardless of what else is running.
        sweepDeviceCaps = await trainer.currentMetalMemoryLimits()

        let sizes = Self.sweepSizes
        let secondsPerSize = Self.sweepSecondsPerSize
        let cancelBox = CancelBox()
        sweepCancelBox = cancelBox

        sweepTask = Task { [trainer, cancelBox] in
            // Reset the trainer's internal weights so loss starts fresh
            // and small batches don't inherit overfit weights from prior
            // continuous-training runs.
            do {
                try await trainer.resetNetwork()
            } catch {
                await MainActor.run {
                    trainingError = "Reset failed: \(error.localizedDescription)"
                    sweepRunning = false
                    sweepCancelBox = nil
                }
                return
            }

            let result = await Self.runSweep(
                trainer: trainer,
                sizes: sizes,
                secondsPerSize: secondsPerSize,
                cancelBox: cancelBox
            )

            await MainActor.run {
                // Pull any final completed rows out of the box (the
                // heartbeat may have a stale cached snapshot).
                sweepResults = cancelBox.completedRows
                if case .failure(let error) = result {
                    trainingError = "Sweep failed: \(error.localizedDescription)"
                }
                sweepProgress = nil
                sweepCancelBox = nil
                sweepRunning = false
            }
        }
    }

    func stopSweep() {
        // Flip the box directly — the worker polls this between steps and
        // breaks out of the loops. Cancelling the Swift Task wouldn't help
        // because Task.isCancelled doesn't propagate to the unstructured
        // detached worker we spawned, and the worker doesn't await anything
        // it could check Task.isCancelled on.
        sweepCancelBox?.cancel()
    }

    nonisolated private static func runSweep(
        trainer: ChessTrainer,
        sizes: [Int],
        secondsPerSize: Double,
        cancelBox: CancelBox
    ) async -> Result<[SweepRow], Error> {
        do {
            return .success(try await trainer.runSweep(
                sizes: sizes,
                targetSecondsPerSize: secondsPerSize,
                cancelled: { cancelBox.isCancelled },
                progress: { batchSize, stepsSoFar, elapsed in
                    cancelBox.updateProgress(
                        SweepProgress(
                            batchSize: batchSize,
                            stepsSoFar: stepsSoFar,
                            elapsedSec: elapsed
                        )
                    )
                },
                recordPeakSampleNow: {
                    // Worker-thread sample — guarantees every row gets a
                    // fresh reading at start and end even if no UI
                    // heartbeat fired during the row's lifetime.
                    cancelBox.recordPeakSample(ChessTrainer.currentPhysFootprintBytes())
                },
                consumeRowPeak: {
                    cancelBox.takeRowPeak()
                },
                onRowCompleted: { row in
                    // Worker thread — push the completed row into the box
                    // so the heartbeat can pick it up. Lets the table grow
                    // one row at a time as the sweep progresses.
                    cancelBox.appendRow(row)
                }
            ))
        } catch {
            return .failure(error)
        }
    }

    /// Format the sweep results as a fixed-column monospaced table.
    /// Updates live as rows complete; after the run finishes, includes
    /// the throughput peak.
    func sweepStatsText() -> String {
        var lines: [String] = []
        lines.append("Batch Size Sweep (training-mode BN)")
        lines.append(String(format: "  Target: %.0f s per size", Self.sweepSecondsPerSize))
        if let caps = sweepDeviceCaps {
            lines.append(String(
                format: "  Device:  recommendedMaxWorkingSetSize=%.2f GB,  maxBufferLength=%.2f GB",
                Self.bytesToGB(caps.recommendedMaxWorkingSet),
                Self.bytesToGB(caps.maxBufferLength)
            ))
            lines.append(String(
                format: "           currentAllocatedSize=%.2f GB (at sweep start)",
                Self.bytesToGB(caps.currentAllocated)
            ))
        }
        lines.append("")

        lines.append(" Batch    Warmup    Steps    Time   Avg/step   Avg GPU    Pos/sec     Loss      Peak")
        lines.append(" -----    ------    -----    ----   --------   -------    -------     ----      ----")

        for row in sweepResults {
            switch row {
            case .completed(let r):
                let posPerSec = Int(r.positionsPerSec.rounded())
                    .formatted(.number.grouping(.automatic))
                    .padding(toLength: 9, withPad: " ", startingAt: 0)
                lines.append(String(
                    format: "%6d  %7.1f ms %6d %6.1fs  %7.2f ms %7.2f ms  %@  %+.3f  %6.2f GB",
                    r.batchSize,
                    r.warmupMs,
                    r.steps,
                    r.elapsedSec,
                    r.avgStepMs,
                    r.avgGpuMs,
                    posPerSec,
                    r.lastLoss,
                    Self.bytesToGB(r.peakResidentBytes)
                ))
            case .skipped(let s):
                let reason: String
                if s.exceededWorkingSet && s.exceededBufferLength {
                    reason = "working-set & buffer cap"
                } else if s.exceededWorkingSet {
                    reason = "working-set cap"
                } else {
                    reason = "buffer cap"
                }
                lines.append(String(
                    format: "%6d  skipped — est RAM %6.2f GB, max buf %6.2f GB  [%@]",
                    s.batchSize,
                    Self.bytesToGB(s.estimatedBytes),
                    Self.bytesToGB(s.largestBufferBytes),
                    reason
                ))
            }
        }

        if sweepRunning {
            lines.append("")
            if let p = sweepProgress {
                lines.append(String(
                    format: "  Running: batch size %d, %d steps, %.1fs",
                    p.batchSize, p.stepsSoFar, p.elapsedSec
                ))
            } else {
                lines.append("  Starting...")
            }
        } else if !sweepResults.isEmpty {
            let completed: [SweepResult] = sweepResults.compactMap {
                if case .completed(let r) = $0 { return r } else { return nil }
            }
            if let best = completed.max(by: { $0.positionsPerSec < $1.positionsPerSec }) {
                lines.append("")
                lines.append(String(
                    format: "  Best: batch size %d at %d positions/sec",
                    best.batchSize,
                    Int(best.positionsPerSec.rounded())
                ))
            }
        }

        return lines.joined(separator: "\n")
    }

    private static func bytesToGB(_ bytes: UInt64) -> Double {
        Double(bytes) / 1_073_741_824.0
    }

    // MARK: - Engine Diagnostics

    /// Run a one-shot battery of correctness probes and log results
    /// with `[DIAG]` prefix. Designed to be triggered on demand after
    /// significant code changes (architecture refactors, encoder
    /// changes) so the user can confirm the engine still passes basic
    /// invariants without waiting for a full training session to
    /// surface any regression.
    ///
    /// Probes:
    ///   1. PolicyEncoding round-trip across all legal moves at the
    ///      starting position.
    ///   2. PolicyEncoding round-trip in a position with promotions.
    ///   3. PolicyEncoding distinct-index check (no two legal moves
    ///      share an index).
    ///   4. ChessGameEngine 3-fold detection on knight shuffle.
    ///   5. BoardEncoder produces correct tensor length.
    ///   6. Network forward pass shape check (if a network exists).
    ///
    /// Designed to complete in well under a second so the user sees
    /// immediate pass/fail feedback. Results go to the session log,
    /// not to a dialog — the log is the canonical record.
    func runEngineDiagnostics() {
        SessionLogger.shared.log("[BUTTON] Engine Diagnostics")
        // Wrap in a Task so we can `await` the network's async
        // evaluate cleanly. Pure-logic probes run synchronously
        // inside; only the network probe needs the await. Failures
        // are reported via the [DIAG] log lines, not via UI alerts.
        let networkRef = network
        Task {
            await self.runEngineDiagnosticsAsync(net: networkRef)
        }
    }

    private func runEngineDiagnosticsAsync(net: ChessMPSNetwork?) async {
        SessionLogger.shared.log("[DIAG] === Engine diagnostics begin ===")
        var failed = 0
        var ran = 0

        func check(_ name: String, _ predicate: () throws -> Bool) {
            ran += 1
            do {
                if try predicate() {
                    SessionLogger.shared.log("[DIAG] PASS  \(name)")
                } else {
                    SessionLogger.shared.log("[DIAG] FAIL  \(name)")
                    failed += 1
                }
            } catch {
                SessionLogger.shared.log("[DIAG] FAIL  \(name): \(error.localizedDescription)")
                failed += 1
            }
        }

        // 1. PolicyEncoding round-trip on starting position.
        check("PolicyEncoding round-trip at starting position") {
            let state = GameState.starting
            let legals = MoveGenerator.legalMoves(for: state)
            for move in legals {
                let (chan, r, c) = PolicyEncoding.encode(move, currentPlayer: state.currentPlayer)
                guard let decoded = PolicyEncoding.decode(channel: chan, row: r, col: c, state: state),
                      decoded == move else { return false }
            }
            return !legals.isEmpty
        }

        // 2. Round-trip with promotions on the board.
        check("PolicyEncoding round-trip with promotions") {
            var board: [Piece?] = Array(repeating: nil, count: 64)
            board[7 * 8 + 0] = Piece(type: .king, color: .white)
            board[0 * 8 + 7] = Piece(type: .king, color: .black)
            for col in 1..<7 { board[1 * 8 + col] = Piece(type: .pawn, color: .white) }
            let state = GameState(
                board: board, currentPlayer: .white,
                whiteKingsideCastle: false, whiteQueensideCastle: false,
                blackKingsideCastle: false, blackQueensideCastle: false,
                enPassantSquare: nil, halfmoveClock: 0
            )
            let legals = MoveGenerator.legalMoves(for: state)
            for move in legals {
                let (chan, r, c) = PolicyEncoding.encode(move, currentPlayer: state.currentPlayer)
                guard let decoded = PolicyEncoding.decode(channel: chan, row: r, col: c, state: state),
                      decoded == move else { return false }
            }
            // Verify all 4 promotion variants are distinct
            let promos = legals.filter { $0.promotion != nil && $0.fromCol == 1 && $0.toCol == 1 }
            let promoIndices = Set(promos.map {
                PolicyEncoding.policyIndex($0, currentPlayer: state.currentPlayer)
            })
            return promos.count == 4 && promoIndices.count == 4
        }

        // 3. Distinct policy indices for all legal moves.
        check("PolicyEncoding produces distinct indices for legal moves") {
            let legals = MoveGenerator.legalMoves(for: .starting)
            let indices = legals.map { PolicyEncoding.policyIndex($0, currentPlayer: .white) }
            return Set(indices).count == indices.count
        }

        // 4. 3-fold detection via knight shuffle.
        check("ChessGameEngine detects 3-fold via knight shuffle") {
            let engine = ChessGameEngine()
            let nf3 = ChessMove(fromRow: 7, fromCol: 6, toRow: 5, toCol: 5, promotion: nil)
            let nc6 = ChessMove(fromRow: 0, fromCol: 1, toRow: 2, toCol: 2, promotion: nil)
            let ng1 = ChessMove(fromRow: 5, fromCol: 5, toRow: 7, toCol: 6, promotion: nil)
            let nb8 = ChessMove(fromRow: 2, fromCol: 2, toRow: 0, toCol: 1, promotion: nil)
            for _ in 0..<2 {
                _ = try engine.applyMoveAndAdvance(nf3)
                _ = try engine.applyMoveAndAdvance(nc6)
                _ = try engine.applyMoveAndAdvance(ng1)
                _ = try engine.applyMoveAndAdvance(nb8)
            }
            if case .drawByThreefoldRepetition = engine.result { return true }
            return false
        }

        // 5. BoardEncoder shape check.
        check("BoardEncoder produces tensorLength floats (= \(BoardEncoder.tensorLength))") {
            let tensor = BoardEncoder.encode(.starting)
            return tensor.count == BoardEncoder.tensorLength
        }

        // 6. Network forward-pass shape (only if a network is built).
        if let net = net {
            ran += 1
            do {
                let board = BoardEncoder.encode(.starting)
                nonisolated(unsafe) var policy: [Float] = []
                try await net.evaluate(board: board) { policyBuf, _ in
                    policy = Array(policyBuf)
                }
                if policy.count == ChessNetwork.policySize {
                    SessionLogger.shared.log(
                        "[DIAG] PASS  Network forward-pass produces \(ChessNetwork.policySize) logits"
                    )
                } else {
                    SessionLogger.shared.log(
                        "[DIAG] FAIL  Network forward-pass: expected \(ChessNetwork.policySize) logits, got \(policy.count)"
                    )
                    failed += 1
                }
            } catch {
                SessionLogger.shared.log(
                    "[DIAG] FAIL  Network forward-pass error: \(error.localizedDescription)"
                )
                failed += 1
            }
        } else {
            SessionLogger.shared.log("[DIAG] SKIP  Network forward-pass shape (no network built yet)")
        }

        SessionLogger.shared.log("[DIAG] === Engine diagnostics done: \(ran - failed)/\(ran) passed ===")
    }

    /// Run a one-shot "is the policy head producing position-conditional
    /// output?" probe. Two very different positions go through the live
    /// champion network in inference mode; the policy outputs are
    /// compared for L1 distance, max single-cell |Δ|, and value-head Δ.
    /// If the policy outputs are essentially identical (avg per-cell
    /// |Δ| < 1e-4) the policy head has collapsed to a position-agnostic
    /// constant — that's the symptom we've been chasing in the masked
    /// CE / entropy debugging. Healthy networks emit meaningfully
    /// different policies for unrelated boards.
    func runPolicyConditioningDiagnostic() {
        SessionLogger.shared.log("[BUTTON] Policy Conditioning Probe")
        let networkRef = network
        Task {
            await self.runPolicyConditioningDiagnosticAsync(net: networkRef)
        }
    }

    private func runPolicyConditioningDiagnosticAsync(net: ChessMPSNetwork?) async {
        SessionLogger.shared.log("[DIAG] === Policy-conditioning probe begin ===")
        guard let net else {
            SessionLogger.shared.log("[DIAG] SKIP  No network built yet")
            SessionLogger.shared.log("[DIAG] === Policy-conditioning probe done ===")
            return
        }

        // Position 1: white-to-move starting position. Plain, common,
        // policy is well-defined.
        let pos1 = GameState.starting

        // Position 2: a midgame-ish black-to-move position with very
        // different piece layout — different side to move, different
        // material, different square occupancies — so every input plane
        // looks different from pos1.
        var midboard: [Piece?] = Array(repeating: nil, count: 64)
        midboard[0 * 8 + 4] = Piece(type: .king, color: .black)
        midboard[7 * 8 + 4] = Piece(type: .king, color: .white)
        midboard[3 * 8 + 3] = Piece(type: .queen, color: .black)
        midboard[4 * 8 + 5] = Piece(type: .knight, color: .white)
        midboard[2 * 8 + 6] = Piece(type: .rook, color: .white)
        midboard[5 * 8 + 1] = Piece(type: .bishop, color: .black)
        let pos2 = GameState(
            board: midboard, currentPlayer: .black,
            whiteKingsideCastle: false, whiteQueensideCastle: false,
            blackKingsideCastle: false, blackQueensideCastle: false,
            enPassantSquare: nil, halfmoveClock: 0
        )

        do {
            let board1 = BoardEncoder.encode(pos1)
            let board2 = BoardEncoder.encode(pos2)
            nonisolated(unsafe) var policy1: [Float] = []
            nonisolated(unsafe) var value1: Float = 0
            try await net.evaluate(board: board1) { policyBuf, v in
                policy1 = Array(policyBuf)
                value1 = v
            }
            nonisolated(unsafe) var policy2: [Float] = []
            nonisolated(unsafe) var value2: Float = 0
            try await net.evaluate(board: board2) { policyBuf, v in
                policy2 = Array(policyBuf)
                value2 = v
            }

            guard policy1.count == policy2.count else {
                SessionLogger.shared.log(
                    "[DIAG] FAIL  policy length mismatch: \(policy1.count) vs \(policy2.count)"
                )
                SessionLogger.shared.log("[DIAG] === Policy-conditioning probe done ===")
                return
            }

            // Per-position summary stats.
            let mean1 = policy1.reduce(Float(0), +) / Float(policy1.count)
            let mean2 = policy2.reduce(Float(0), +) / Float(policy2.count)
            var var1: Float = 0
            var var2: Float = 0
            for v in policy1 { var1 += (v - mean1) * (v - mean1) }
            for v in policy2 { var2 += (v - mean2) * (v - mean2) }
            let std1 = (var1 / Float(policy1.count)).squareRoot()
            let std2 = (var2 / Float(policy2.count)).squareRoot()

            // L1 distance + max single-cell |Δ| + per-cell average.
            var l1: Double = 0
            var maxAbsDiff: Double = 0
            var maxAbsIdx: Int = 0
            for i in 0..<policy1.count {
                let d = Double(abs(policy1[i] - policy2[i]))
                l1 += d
                if d > maxAbsDiff {
                    maxAbsDiff = d
                    maxAbsIdx = i
                }
            }
            let avgPerCellDiff = l1 / Double(policy1.count)

            SessionLogger.shared.log(
                String(
                    format: "[DIAG]   pos1: mean=%+0.4f std=%.4f, pos2: mean=%+0.4f std=%.4f",
                    mean1, std1, mean2, std2
                )
            )
            SessionLogger.shared.log(
                String(
                    format: "[DIAG]   policy Δ: L1=%.3f, maxAbs=%.4f at idx=%d, avg per-cell |Δ|=%.6f",
                    l1, maxAbsDiff, maxAbsIdx, avgPerCellDiff
                )
            )
            SessionLogger.shared.log(
                String(
                    format: "[DIAG]   value Δ: pos1=%+0.4f, pos2=%+0.4f, Δ=%+0.4f",
                    value1, value2, value1 - value2
                )
            )

            // Pass criterion: avg per-cell |Δ| above the noise floor
            // (1e-4 is generous — a randomly-initialized network with
            // logit std ~2.5 should easily produce avg |Δ| ≈ 1.0+ on
            // unrelated inputs). Below 1e-4 means the policy head's
            // output is effectively independent of the input — exactly
            // the failure mode we've been hypothesizing.
            let policyConditional = avgPerCellDiff >= 1e-4
            if policyConditional {
                SessionLogger.shared.log(
                    String(
                        format: "[DIAG] PASS  Policy head is position-conditional (avg per-cell |Δ| = %.6f, threshold ≥ 1e-4)",
                        avgPerCellDiff
                    )
                )
            } else {
                SessionLogger.shared.log(
                    String(
                        format: "[DIAG] FAIL  Policy head appears position-AGNOSTIC (avg per-cell |Δ| = %.6f, < 1e-4 threshold) — outputs are effectively the same regardless of input",
                        avgPerCellDiff
                    )
                )
            }

            // Also probe the value head: a healthy value head should
            // give meaningfully different scalars for two unrelated
            // positions. A pinned-to-zero value head would give |Δ|≈0.
            let valueDelta = abs(Double(value1 - value2))
            if valueDelta < 1e-4 {
                SessionLogger.shared.log(
                    String(
                        format: "[DIAG] WARN  Value head also looks position-agnostic (|Δ|=%.6f)",
                        valueDelta
                    )
                )
            }
        } catch {
            SessionLogger.shared.log(
                "[DIAG] FAIL  Policy probe error: \(error.localizedDescription)"
            )
        }

        SessionLogger.shared.log("[DIAG] === Policy-conditioning probe done ===")
    }

    // MARK: - Arena-history recovery (Stage 4r)

    /// True while a one-shot log-scan recovery pass is running (disables the
    /// "Recover from logs" button against overlapping scans; drives a spinner
    /// in the arena-history sheet header).
    var arenaRecoveryInProgress: Bool = false

    func runArenaHistoryRecovery() {
        guard !arenaRecoveryInProgress else { return }
        arenaRecoveryInProgress = true
        SessionLogger.shared.log("[BUTTON] Recover Arena History from logs (start)")

        Task.detached(priority: .userInitiated) { [tournamentHistory] in
            let logsDir: URL
            do {
                logsDir = try ArenaLogRecovery.defaultLogsDirectory()
            } catch {
                await MainActor.run {
                    SessionLogger.shared.log(
                        "[RECOVER] failed: cannot resolve logs directory: \(error.localizedDescription)"
                    )
                    self.arenaRecoveryInProgress = false
                }
                return
            }
            let recovered = ArenaLogRecovery.scan(logsDirectory: logsDir)

            // Merge in-place on a copy; only adopt the new array
            // if at least one record actually changed.
            var updated = tournamentHistory
            var changedCount = 0
            for (i, record) in tournamentHistory.enumerated() {
                guard let hit = recovered[record.finishedAtStep] else { continue }
                guard hit.candidateWins == record.candidateWins,
                      hit.draws == record.draws,
                      hit.championWins == record.championWins else {
                    continue
                }
                let newFinishedAt = record.finishedAt ?? hit.finishedAt
                let newCandidateID = record.candidateID
                    ?? hit.candidateID.map { ModelID(value: $0) }
                let newChampionID = record.championID
                    ?? hit.championID.map { ModelID(value: $0) }
                if newFinishedAt != record.finishedAt
                    || newCandidateID != record.candidateID
                    || newChampionID != record.championID {
                    updated[i] = TournamentRecord(
                        finishedAtStep: record.finishedAtStep,
                        finishedAt: newFinishedAt,
                        candidateID: newCandidateID,
                        championID: newChampionID,
                        gamesPlayed: record.gamesPlayed,
                        candidateWins: record.candidateWins,
                        championWins: record.championWins,
                        draws: record.draws,
                        score: record.score,
                        promoted: record.promoted,
                        promotionKind: record.promotionKind,
                        promotedID: record.promotedID,
                        durationSec: record.durationSec,
                        candidateWinsAsWhite: record.candidateWinsAsWhite,
                        candidateWinsAsBlack: record.candidateWinsAsBlack,
                        candidateLossesAsWhite: record.candidateLossesAsWhite,
                        candidateLossesAsBlack: record.candidateLossesAsBlack,
                        candidateDrawsAsWhite: record.candidateDrawsAsWhite,
                        candidateDrawsAsBlack: record.candidateDrawsAsBlack
                    )
                    changedCount += 1
                }
            }

            await MainActor.run {
                if changedCount > 0 {
                    self.tournamentHistory = updated
                }
                SessionLogger.shared.log(
                    "[RECOVER] Arena history scan: \(recovered.count) kv lines mapped, \(changedCount) records updated (of \(tournamentHistory.count) total). Save the session to persist."
                )
                self.arenaRecoveryInProgress = false
            }
        }
    }

    // MARK: - Heartbeat display caches (Stage 4s)

    /// Mirror of the trainer's warmup-relevant state, refreshed by the heartbeat
    /// (top-bar Training Status chip + LR Warm-up status cell).
    struct TrainerWarmupSnapshot: Equatable {
        var completedSteps: Int
        var warmupSteps: Int
        var effectiveLR: Float
        var inWarmup: Bool { warmupSteps > 0 && completedSteps < warmupSteps }
    }

    /// Mirror of the trainer's warmup state, refreshed by the heartbeat. `nil` outside a session.
    var trainerWarmupSnap: TrainerWarmupSnapshot?

    /// Cached memory-stats line shown in the busy row, refreshed at most every
    /// `memoryStatsRefreshSec` by the heartbeat.
    var memoryStatsSnap: MemoryStatsSnapshot?
    /// Wall-clock of the most recent `memoryStatsSnap` refresh (`.distantPast` until first).
    var memoryStatsLastFetch: Date = .distantPast
    /// Previous `ProcessUsageSample` held so the heartbeat can compute %CPU/%GPU from the delta.
    var lastUsageSample: ProcessUsageSample?
    /// Wall-clock of the most recent usage refresh (`.distantPast` until first).
    var usageStatsLastFetch: Date = .distantPast
    /// Last-computed %CPU over the real wall-clock since the previous sample. `nil` until the 2nd sample.
    var cpuPercent: Double?
    /// Last-computed %GPU over the same interval. `nil` until the 2nd sample.
    var gpuPercent: Double?

    // MARK: - Heartbeat (Stage 4t)

    /// Re-syncs the AppKit menu command hub. Wired to `{ syncMenuCommandHubState() }`.
    var onSyncMenuCommandHubState: () -> Void = { }
    /// Mirrors the live game state into the view's `gameSnapshot @State` for the
    /// on-board display. Wired to `{ gameSnapshot = await gameWatcher.asyncSnapshot() }`.
    var onUpdateGameSnapshot: () async -> Void = { }
    nonisolated static let memoryStatsRefreshSec: Double = 10
    nonisolated static let usageStatsRefreshSec: Double = 5
    nonisolated static let progressRateRefreshSec: Double = 1.0
    nonisolated static let progressRateWindowSec: Double = 180.0

    func processSnapshotTimerTick(dispatchedAt: CFAbsoluteTime) async {
        guard !snapshotTickInFlight else { return }
        snapshotTickInFlight = true
        defer { snapshotTickInFlight = false }
        let mainActorEnqueueWaitMs = (CFAbsoluteTimeGetCurrent() - dispatchedAt) * 1000
        let start = CFAbsoluteTimeGetCurrent()
        await __processSnapshotTimerTick()
        let elapsedMs = (CFAbsoluteTimeGetCurrent() - start) * 1000

        // Throttled session-log emit so a slow tick (or growing main-
        // actor wait) is visible in the session log without spamming
        // it with one line per 100 ms heartbeat. Emits at most once
        // per `Self.snapshotTickLogIntervalSec` seconds and ALWAYS on
        // any tick whose tick body or main-actor enqueue wait exceeds
        // the alarm threshold — those are the events worth seeing
        // even if a recent throttled emit just landed.
        let now = Date()
        let shouldEmitPeriodic = snapshotTickLastLogAt == nil
            || now.timeIntervalSince(snapshotTickLastLogAt!) >= Self.snapshotTickLogIntervalSec
        let isAnomalous = elapsedMs >= Self.snapshotTickAlarmMs
            || mainActorEnqueueWaitMs >= Self.snapshotTickAlarmMs
        if shouldEmitPeriodic || isAnomalous {
            let prefix = isAnomalous && !shouldEmitPeriodic ? "[TICK-SLOW]" : "[TICK]"
            SessionLogger.shared.log(String(
                format: "%@ tickMs=%.2f mainActorEnqueueWaitMs=%.2f",
                prefix, elapsedMs, mainActorEnqueueWaitMs
            ))
            snapshotTickLastLogAt = now
        }
    }

    /// Throttling clock for the periodic `[TICK]` emit. Reset every
    /// time a periodic or anomalous emit fires.
    var snapshotTickInFlight = false
    var snapshotTickLastLogAt: Date? = nil

    /// Cadence for the periodic snapshot-tick log emit. 30 s gives a
    /// readable log file at steady state while the anomaly threshold
    /// (`snapshotTickAlarmMs`) makes sure any genuine stall surfaces
    /// promptly even between the periodic emits.
    nonisolated static let snapshotTickLogIntervalSec: TimeInterval = 30

    /// Threshold above which a snapshot-tick wall or its main-actor
    /// enqueue wait is logged immediately (out-of-band) rather than
    /// waiting for the next periodic emit. 50 ms is half the
    /// heartbeat period — anything past that is a meaningful blip.
    nonisolated static let snapshotTickAlarmMs: Double = 50
    /// Body of the 100 ms heartbeat. Pulls every cross-thread state
    /// box (game, sweep, training, arena, parallel-worker counters,
    /// replay-ratio controller, diversity tracker) into the matching
    /// `@State`, throttled internally where each consumer cares about
    /// avoiding redundant invalidations. Extracted out of the inline
    /// `.onReceive(snapshotTimer)` closure so `body`'s expression
    /// type-check stays cheap — the closure used to be ~140 lines and
    /// dragged the whole modifier chain past the
    /// `-warn-long-expression-type-checking` budget.
    private func __processSnapshotTimerTick() async {
        // Pull the latest game state into @State at most every 100ms.
        // Cheap (single locked struct copy) and bounds UI work even
        // when the game loop is doing hundreds of moves per second.
        //
        // `elap(_:)` is the per-stage trace probe used to attribute a
        // UI stall to a specific block of this tick. Cheap enough to
        // leave on at the 2 Hz heartbeat — `os_log` is hundreds of
        // nanoseconds per emit and we're producing ~13 lines/sec,
        // negligible against the work the tick body itself does.
        func elap(_ s: String) {
            logger.log(">> \(s): \(Date().timeIntervalSince(start))")
        }
        let start = Date()
        _ = start
        elap("start")
        await onUpdateGameSnapshot()
        elap("after gameWatcher")
        // Same heartbeat pulls the sweep's worker-thread progress and
        // any newly-completed rows into @State so the table grows live.
        if sweepRunning, let box = sweepCancelBox {
            sweepProgress = await box.asyncLatestProgress()
            // Sample process resident memory and feed it into the
            // sweep's per-row peak. The trainer also samples at row
            // boundaries — we just contribute extra samples while a
            // row is in flight so we don't miss mid-step spikes.
            let phys = await ChessTrainer.getAppMemoryFootprintBytes()
            await box.asyncRecordPeakSample(phys)
            let rows = await box.asyncCompletedRows()
            if rows.count != sweepResults.count {
                sweepResults = rows
            }
            elap("after sweepsom")
        }
        // Same heartbeat pulls live training stats out of the
        // lock-protected box the background training task is writing
        // into. Guarded on step count so a mid-run session that
        // hasn't advanced since the last tick doesn't trigger a
        // useless redraw — and an idle box (after Stop or before
        // first step) stays silent.
        if let box = trainingBox {
            let snap = await box.snapshot()
            if snap.stats.steps != (trainingStats?.steps ?? -1) {
                trainingStats = snap.stats
                lastTrainStep = snap.lastTiming
                realRollingPolicyLoss = snap.rollingPolicyLoss
                realRollingValueLoss = snap.rollingValueLoss
                
                // Periodic memory log to stdout to correlate with performance degradation
                if snap.stats.steps % 100 == 0 {
                    let appMB = Double(memoryStatsSnap?.appFootprintBytes ?? 0) / 1024 / 1024
                    let gpuMB = Double(memoryStatsSnap?.gpuAllocatedBytes ?? 0) / 1024 / 1024
                    print(String(format: "[MEMORY-DEBUG] step=%d app=%.1fMB gpu=%.1fMB", snap.stats.steps, appMB, gpuMB))
                }
            }
            if let err = snap.error, trainingError == nil {
                trainingError = err
            }
            elap("after 3")
        }
        // Mirror the trainer's warmup state into @State so the status
        // chip and the LR Warm-up status cell read from a snapshot
        // rather than touching the trainer's executionQueue from inside
        // `body`. Only published while a trainer exists; a missing
        // trainer yields nil so the Idle chip path doesn't accidentally
        // surface stale warmup numbers from a prior 
        if let trainer {
            let completedTrainSteps = await trainer.asyncCompletedTrainSteps()
            elap("after 3.1")
            // Pass the locally-snapshotted step count so the LR uses the
            // same observation rather than re-acquiring the SyncBox; the
            // count and LR in the published snapshot are then guaranteed
            // consistent (they were previously two independent reads with
            // a one-step disagreement window).
            let effectiveLR = await trainer.asyncEffectiveLearningRate(
                forBatchSize: TrainingParameters.shared.trainingBatchSize,
                completedSteps: completedTrainSteps
            )
            elap("after 3.2")
            let next = SessionController.TrainerWarmupSnapshot(
                completedSteps: completedTrainSteps,
                warmupSteps: trainer.lrWarmupSteps,
                effectiveLR: effectiveLR
            )
            elap("after 3.3")
            if next != trainerWarmupSnap { trainerWarmupSnap = next }
        } else if trainerWarmupSnap != nil {
            trainerWarmupSnap = nil
        }
        elap("after 4")
        // Arena progress mirror — cheap lock read, only updates the
        // @State when the game index has advanced (or transitioned
        // between non-running and running), so no redundant view
        // invalidations between tournament games.
        if let tBox = tournamentBox {
            let snap = await tBox.asyncSnapshot()
            if snap?.currentGame != tournamentProgress?.currentGame
                || (snap == nil) != (tournamentProgress == nil) {
                tournamentProgress = snap
            }
        }
        elap("after 5")
        // Parallel worker counters mirror — only updates @State when
        // totals have actually advanced so the body isn't re-evaluated
        // when nothing's changed. The sessionStart timestamp is
        // embedded in the snapshot so the Session panel and busy label
        // can compute wall-clock rates on every render. Dirty check
        // compares the fields that advance on self-play and training
        // events; if either has changed (or the rolling-window count
        // has shifted because an entry aged out), push a new snapshot.
        if let pBox = parallelWorkerStatsBox {
            let snap = await pBox.asyncSnapshot()
            let prev = parallelStats
            // `sessionStart` is included in the dirty check so the
            // one-time shift performed by `markWorkersStarted()` lands
            // in @State immediately, even if no game or training step
            // has recorded yet.
            let changed = snap.selfPlayGames != (prev?.selfPlayGames ?? -1)
            || snap.trainingSteps != (prev?.trainingSteps ?? -1)
            || snap.recentGames != (prev?.recentGames ?? -1)
            || snap.sessionStart != prev?.sessionStart
            if changed {
                parallelStats = snap
            }
        }
        elap("after 6")
        // Memory stats refresh. Throttled internally to
        // `memoryStatsRefreshSec` so this is a cheap timestamp compare
        // on most heartbeats.
        await refreshMemoryStatsIfNeeded()
        elap("after 7")
        // Process %CPU / %GPU refresh — separate (5 s) cadence from
        // memory stats (10 s) so the utilisation line updates twice as
        // often without dragging the heavier Metal property reads
        // along with it.
        await refreshUsagePercentsIfNeeded()
        elap("after 8")
        // Progress-rate chart sampler. 1 Hz during Play and Train;
        // each sample carries the moves/hr averaged over the last 3
        // minutes of work. No-op outside of realTraining.
        await refreshProgressRateIfNeeded()
        elap("after 9")
        // Replay-ratio snapshot for the UI. Persist the auto-computed
        // delay so the next session starts from where the adjuster
        // left off.
        if let rc = replayRatioController {
            let snap = await rc.asyncSnapshot()
            replayRatioSnapshot = snap
            if snap.autoAdjust {
                lastAutoComputedDelayMs = snap.computedDelayMs
            }
            // Outer integral compensator. See the doc comment on
            // `effectiveReplayRatioTarget` for full rationale; in
            // short, the inner controller's per-tick overhead
            // subtraction is mis-scaled for the batched-evaluator
            // architecture and equilibrates at a `cons/prod` ratio
            // below the user's requested target. We close the loop
            // here without modifying the controller: every heartbeat,
            // observe the gap between the user's target and the
            // reported `currentRatio`, then adjust the controller's
            // internal `targetRatio` in the direction that moves
            // observed-ratio toward user-target. Slow gain (no faster
            // than the inner controller's SMA cadence — see
            // `gainPerSecond` and `ReplayRatioController.historyWindowSec`)
            // so the two loops don't fight; bounded so a long-tail
            // noise spike can't drift the controller into a
            // degenerate set-point.
            updateReplayRatioCompensator(snap: snap)
        }
        elap("after 10")
        // Diversity-histogram mirror. Read once per heartbeat off the
        // tracker's thread-safe snapshot. Only push into @State when
        // the bucket totals actually change (or the bar array is
        // currently empty) so SwiftUI doesn't invalidate the chart
        // every tick for a stable reading.
        if let tracker = selfPlayDiversityTracker {
            let divSnap = await tracker.asyncSnapshot()
            let labels = GameDiversityTracker.histogramLabels
            var newBars: [DiversityHistogramBar] = []
            newBars.reserveCapacity(divSnap.divergenceHistogram.count)
            for (idx, count) in divSnap.divergenceHistogram.enumerated()
            where idx < labels.count {
                newBars.append(DiversityHistogramBar(
                    id: idx,
                    label: labels[idx],
                    count: count
                ))
            }
            let changed = newBars.count != (chartCoordinator?.currentDiversityHistogramBars.count ?? 0)
            || zip(newBars, chartCoordinator?.currentDiversityHistogramBars ?? [])
                .contains { $0.0.count != $0.1.count }
            if changed {
                chartCoordinator?.setDiversityHistogramBars(newBars)
            }
        }
        elap("after 11")
        refreshChartZoomTick()
        elap("after 12")
        periodicSaveTick()
        elap("after LAST")
    }

    /// Per-heartbeat tick that asks the periodic-save scheduler
    /// whether to fire. Throttled to ~1 Hz — a 4-hour deadline does
    /// not benefit from 10 Hz polling and the throttle keeps the
    /// heartbeat hot path from paying a decision cost per frame. A
    /// no-op when no session is active, and an immediate no-op
    /// when a periodic save is already in flight (the fire flag is
    /// cleared when the write task resolves).
    @MainActor
    private func periodicSaveTick() {
        guard let controller = periodicSaveController else {
            periodicSaveLastPollAt = nil
            return
        }
        let now = Date()
        if let lastPoll = periodicSaveLastPollAt,
           now.timeIntervalSince(lastPoll) < 1.0 {
            return
        }
        periodicSaveLastPollAt = now
        if periodicSaveInFlight {
            return
        }
        switch controller.decide(now: now) {
        case .idle:
            return
        case .fire:
            SessionLogger.shared.log(
                "[CHECKPOINT] Periodic save tick — firing (interval=\(Int(UpperContentView.periodicSaveIntervalSec))s)"
            )
            handleSaveSessionPeriodic()
        }
    }

    /// Drive the chart-zoom state machine once per heartbeat. The
    /// auto-snap, manual-clamp, and re-engage logic lives on the
    /// coordinator (`refreshZoomTick()`). This wrapper resyncs the
    /// menu command hub when the coordinator reports a state change
    /// so the menu's enabled / disabled flags stay in step.
    private func refreshChartZoomTick() {
        if chartCoordinator?.refreshZoomTick() == true {
            onSyncMenuCommandHubState()
        }
    }

    // ⌘= / ⌘- / Auto-button actions. Thin wrappers that route to
    // the coordinator's zoom helpers so the keyboard shortcut, the
    // menu item, and `LowerContentView`'s inline controls all share
    // the same code. Each wrapper also calls `syncMenuCommandHubState`
    // so the menu's enabled / disabled flags update immediately
    // (the menu hub doesn't observe the coordinator directly).

    @MainActor
    func chartZoomIn() {
        chartCoordinator?.zoomIn()
        onSyncMenuCommandHubState()
    }

    @MainActor
    func chartZoomOut() {
        chartCoordinator?.zoomOut()
        onSyncMenuCommandHubState()
    }

    @MainActor
    func chartZoomEnableAuto() {
        chartCoordinator?.enableAutoZoom()
        onSyncMenuCommandHubState()
    }

    /// Can the user zoom in further? Used to gate the View menu
    /// item's disabled state and the Auto-row's ⌘= hint styling.
    var canZoomChartIn: Bool { chartCoordinator?.canZoomIn ?? false }

    /// Can the user zoom out further given the current data span?
    var canZoomChartOut: Bool { chartCoordinator?.canZoomOut ?? false }

    /// Sample app and GPU memory at most every
    /// `memoryStatsRefreshSec` seconds, caching the result in
    /// `memoryStatsSnap` for the busy label to read. Cheap on a
    /// no-op tick (a single timestamp diff) so it's fine to call
    /// from the 10 Hz heartbeat. The actual sampling reads
    /// `task_info` and a couple of `MTLDevice` properties via
    /// the trainer's existing helpers.
    private func refreshMemoryStatsIfNeeded() async {
        let now = Date()
        if now.timeIntervalSince(memoryStatsLastFetch) < Self.memoryStatsRefreshSec {
            return
        }
        let app = await ChessTrainer.getAppMemoryFootprintBytes()
        let caps: MetalDeviceMemoryLimits?
        if let trainer {
            caps = await trainer.currentMetalMemoryLimits()
        } else {
            caps = nil
        }
        memoryStatsSnap = MemoryStatsSnapshot(
            appFootprintBytes: app,
            gpuAllocatedBytes: caps?.currentAllocated ?? 0,
            gpuMaxTargetBytes: caps?.recommendedMaxWorkingSet ?? 0,
            gpuTotalBytes: ProcessInfo.processInfo.physicalMemory
        )
        memoryStatsLastFetch = now
    }

    /// Append a new progress-rate sample at most once per
    /// `progressRateRefreshSec` during a Play and Train 
    /// The moves/hr fields are computed over a real trailing
    /// 3-minute window: we walk backward from the newest stored
    /// sample until we find the first one whose `timestamp` is
    /// still inside the window, then subtract its cumulative
    /// counters from the current cumulative counters and divide
    /// by the actual elapsed seconds between the two samples.
    ///
    /// Before the session has 3 minutes of history, the window
    /// shrinks gracefully to "whatever we have" — the first
    /// sample reports zero (no earlier sample to subtract from),
    /// the second reports over ~1 s, and so on until the window
    /// reaches its full 180 s width.
    ///
    /// No-op outside of `realTraining`. Sampler state is cleared
    /// by `startRealTraining()` so each session's chart starts
    /// fresh from t=0.
    /// Sample training metrics at the same 1Hz cadence as the
    /// progress rate sampler. Appends a `TrainingChartSample`
    /// with rolling loss, entropy, ratio, and non-neg count.
    /// Append a training chart sample. Called from inside
    /// `refreshProgressRateIfNeeded` at the same 1Hz cadence.
    private func refreshTrainingChartIfNeeded() async {
        let now = Date()
        // Chart sample elapsed-second axis comes off
        // `chartCoordinator?.chartElapsedAnchor`, NOT
        // `parallelStats.sessionStart` or `checkpoint?.currentSessionStart`.
        // The chart anchor is back-dated on session resume so a
        // restored chart trajectory and post-resume samples share
        // one continuous elapsed-sec axis (no visible gap, no
        // overlap). Every chart-axis call site — this one, the
        // progress-rate sampler, and both arena-event sites — must
        // use the same anchor or the elapsedSec values across
        // sources land in different coordinate spaces and the
        // shared `scrollX` binding parks some sources off-screen
        // (the same bug class an earlier mismatch between
        // `parallelStats.sessionStart` and the back-dated
        // `checkpoint?.currentSessionStart` originally introduced).
        let elapsed = max(0, now.timeIntervalSince(chartCoordinator?.chartElapsedAnchor ?? Date()))
        let trainingSnap: TrainingLiveStatsBox.Snapshot?
        if let trainingBox {
            trainingSnap = await trainingBox.snapshot()
        } else {
            trainingSnap = nil
        }
        let ratioSnap = replayRatioSnapshot

        let appMemMB = memoryStatsSnap.map { Double($0.appFootprintBytes) / (1024 * 1024) }
        let gpuMemMB = memoryStatsSnap.map { Double($0.gpuAllocatedBytes) / (1024 * 1024) }
        // GPU busy %: fraction of the last 1-second sample interval
        // that the GPU was actively running training steps. Computed
        // from the delta of cumulative GPU ms in TrainingRunStats.
        let currentGpuMs = trainingSnap?.stats.totalGpuMs ?? 0
        let gpuDeltaMs = max(0, currentGpuMs - (chartCoordinator?.prevChartTotalGpuMs ?? 0))
        let gpuBusy = gpuDeltaMs / 10.0 // delta ms / 1000ms * 100%
        // The coordinator's `appendTrainingChart(_:totalGpuMs:)` call
        // below updates `prevChartTotalGpuMs` to `currentGpuMs` for
        // the next tick — no need to write it here.
        // Power + thermal state read straight from ProcessInfo at
        // sample time. Both are cheap property reads, and both can
        // change between samples without any polling overhead on our
        // part (the OS tracks them). Captured here so the chart
        // tile can render a continuous step trace rather than
        // sampling on hover.
        let pi = ProcessInfo.processInfo
        let sample = TrainingChartSample(
            id: chartCoordinator?.trainingChartNextId ?? 0,
            elapsedSec: elapsed,
            rollingPolicyLoss: trainingSnap?.rollingPolicyLoss,
            rollingValueLoss: trainingSnap?.rollingValueLoss,
            rollingPolicyEntropy: trainingSnap?.rollingPolicyEntropy,
            rollingPolicyNonNegCount: trainingSnap?.rollingPolicyNonNegCount,
            rollingPolicyNonNegIllegalCount: trainingSnap?.rollingPolicyNonNegIllegalCount,
            rollingGradNorm: trainingSnap?.rollingGradGlobalNorm,
            rollingVelocityNorm: trainingSnap?.rollingVelocityNorm,
            rollingPolicyHeadWeightNorm: trainingSnap?.rollingPolicyHeadWeightNorm,
            replayRatio: ratioSnap?.currentRatio,
            rollingPolicyLossWin: trainingSnap?.rollingPolicyLossWin,
            rollingPolicyLossLoss: trainingSnap?.rollingPolicyLossLoss,
            rollingLegalEntropy: realLastLegalMassSnapshot.map { Double($0.legalEntropy) },
            rollingLegalMass: realLastLegalMassSnapshot.map { Double($0.legalMass) },
            rollingValueAbsMean: trainingSnap?.rollingValueAbsMean,
            cpuPercent: cpuPercent,
            gpuBusyPercent: trainingSnap != nil ? gpuBusy : nil,
            gpuMemoryMB: gpuMemMB,
            appMemoryMB: appMemMB,
            lowPowerMode: pi.isLowPowerModeEnabled,
            thermalState: pi.thermalState
        )
        // Hand the freshly-built sample to the coordinator. It
        // owns the ring append, the id-counter bump, the GPU-ms
        // baseline update, and the decimated-frame recompute — see
        // `ChartCoordinator.appendTrainingChart(_:totalGpuMs:)`.
        // The training-alarm evaluation (divergence streaks → banner / beep)
        // lives on `TrainingAlarmController` — see `trainingAlarm`.
        await chartCoordinator?.appendTrainingChart(sample, totalGpuMs: currentGpuMs)
        trainingAlarm?.evaluate(from: sample)
    }

    private func refreshProgressRateIfNeeded() async {
        guard realTraining else { return }
        let now = Date()
        if now.timeIntervalSince(chartCoordinator?.progressRateLastFetch ?? Date()) < Self.progressRateRefreshSec {
            return
        }

        guard let pStats = parallelStats else { return }
        // ElapsedSec on chart axis comes off the chart-coordinator's
        // anchor, NOT `sessionStart`. See the matching block
        // in `refreshTrainingChartIfNeeded` for full reasoning —
        // both samplers must share an anchor or `scrollX` ends up
        // straddling two coordinate spaces. `sessionStart`
        // remains the correct anchor for everything else this
        // function does (cumulative-counter deltas use 3-minute
        // window timestamps, not elapsedSec).
        let elapsed = max(0, now.timeIntervalSince(chartCoordinator?.chartElapsedAnchor ?? Date()))
        let curSp = pStats.selfPlayPositions
        let curTr = (trainingStats?.steps ?? 0) * TrainingParameters.shared.trainingBatchSize

        // Walk newest → oldest through the coordinator's ring,
        // recording the last sample we see that still falls inside
        // the 3-minute window. Breaks out as soon as we hit a
        // sample older than the cutoff — the ring is timestamp-
        // sorted, so anything older is also out of window. Bounded
        // at ~180 iterations per call in steady state regardless of
        // total session length.
        let cutoff = now.addingTimeInterval(-Self.progressRateWindowSec)
        var windowStart: ProgressRateSample?
        var i = (chartCoordinator?.progressRateRing.count ?? 0) - 1
        while i >= 0 {
            guard let sample = chartCoordinator?.progressRateRing[i] else { break }
            if sample.timestamp >= cutoff {
                windowStart = sample
            } else {
                break
            }
            i -= 1
        }

        let spRate: Double
        let trRate: Double
        if let ws = windowStart {
            let dt = now.timeIntervalSince(ws.timestamp)
            if dt > 0 {
                let spDelta = max(0, curSp - ws.selfPlayCumulativeMoves)
                let trDelta = max(0, curTr - ws.trainingCumulativeMoves)
                spRate = Double(spDelta) / dt * 3600
                trRate = Double(trDelta) / dt * 3600
            } else {
                spRate = 0
                trRate = 0
            }
        } else {
            // First sample of the session — nothing to diff
            // against yet. Rate reads as zero for this one tick
            // and the chart picks up real values from the next.
            spRate = 0
            trRate = 0
        }

        let sample = ProgressRateSample(
            id: chartCoordinator?.progressRateNextId ?? 0,
            timestamp: now,
            elapsedSec: elapsed,
            selfPlayCumulativeMoves: curSp,
            trainingCumulativeMoves: curTr,
            selfPlayMovesPerHour: spRate,
            trainingMovesPerHour: trRate
        )
        // Coordinator's `appendProgressRate(_:)` does the ring
        // append, the id-counter bump, the
        // `progressRateLastFetch` timestamp update, and the
        // auto-follow scroll-position adjustment in one shot.
        chartCoordinator?.appendProgressRate(sample)
        // Append a training chart sample at the same 1Hz cadence.
        await refreshTrainingChartIfNeeded()
    }

    /// Format a number of elapsed seconds for the Progress rate
    /// chart's X-axis. Picks a display granularity that matches
    /// the magnitude of the value so early-session axis labels
    /// read "0:15 / 0:30 / 0:45" rather than "0.0 / 0.0 / 0.0":
    ///
    /// * < 60 s: "0:SS"
    /// * < 3600 s: "M:SS"
    /// * ≥ 3600 s: "H:MM:SS"
    ///
    /// Negative values are clamped to 0 — shouldn't happen given
    /// the sampler only produces non-negative elapsed values, but
    /// the chart's axis automatic-ticks can overshoot into negative
    /// space briefly during pan gestures at the left edge.
    static func formatElapsedAxis(_ seconds: Double) -> String {
        let secs = max(0, Int(seconds.rounded()))
        let h = secs / 3600
        let m = (secs % 3600) / 60
        let s = secs % 60
        if h > 0 {
            return String(format: "%d:%02d:%02d", h, m, s)
        } else if secs >= 60 {
            return String(format: "%d:%02d", m, s)
        } else {
            return String(format: "0:%02d", s)
        }
    }

    /// Sample process CPU + GPU time at most every
    /// `usageStatsRefreshSec` seconds, compute the percentage over
    /// the real wall-clock elapsed since the previous sample, and
    /// publish the result into `cpuPercent` / `gpuPercent`. The
    /// math always uses the real `timestamp` delta (not the nominal
    /// cadence), so a paused heartbeat, a missed tick, or a session
    /// restart doesn't skew the reading. If the gap between samples
    /// is more than 3× the cadence — e.g. the app was idle for a
    /// while — the previous sample is discarded rather than used,
    /// because an interval much larger than the polling window is
    /// usually not what the user wants averaged over.
    private func refreshUsagePercentsIfNeeded() async {
        let now = Date()
        if now.timeIntervalSince(usageStatsLastFetch) < Self.usageStatsRefreshSec {
            return
        }
        usageStatsLastFetch = now
        guard let sample = await ChessTrainer.asyncSampleCurrentProcessUsage() else {
            return
        }
        if let prev = lastUsageSample {
            let wallDeltaS = sample.timestamp.timeIntervalSince(prev.timestamp)
            let maxUsefulGapS = Self.usageStatsRefreshSec * 3
            if wallDeltaS > 0 && wallDeltaS <= maxUsefulGapS {
                let wallDeltaNs = wallDeltaS * 1_000_000_000
                let cpuDeltaNs = sample.cpuNs >= prev.cpuNs
                ? Double(sample.cpuNs - prev.cpuNs)
                : 0
                let gpuDeltaNs = sample.gpuNs >= prev.gpuNs
                ? Double(sample.gpuNs - prev.gpuNs)
                : 0
                cpuPercent = cpuDeltaNs / wallDeltaNs * 100
                gpuPercent = gpuDeltaNs / wallDeltaNs * 100
            }
        }
        lastUsageSample = sample
    }

    // MARK: - Checkpoint save / load (Stage 5a)

    /// "Champion was replaced (Load Model) since the last training run" flag — set
    /// by `loadModelFrom`, cleared by `startRealTraining`, read by the view's
    /// post-Stop three-way Start dialog to annotate the "trainer still has its
    /// pre-load weights" note.
    var championLoadedSinceLastTrainingSegment: Bool = false

    /// Tells the auto-resume controller a resume-and-start completed. Wired to
    /// `{ autoResume.markResumeFinished() }` — used by `loadSessionFrom`.
    var onResumeFinished: () -> Void = { }

    /// Returns whether some operation is in progress that should block a load
    /// (training / a continuous task / a sweep / a game / a build / a save/load).
    /// Mirrors the view's `isBuildingOrBusy()`. Wired in `handleBodyOnAppear`.
    var isBuildingOrBusyProvider: () -> Bool = { false }
    /// Clears the view's `inferenceResult @State` (the forward-pass demo result).
    /// Wired to `{ inferenceResult = nil }` — used by the load paths after new
    /// weights land so a stale result doesn't linger on the board.
    var onClearInferenceResult: () -> Void = { }

    /// Manual "Save Champion as Model" — writes a standalone
    /// `.dcmmodel` containing the current champion's weights.
    /// If Play-and-Train is active, pauses self-play worker 0
    /// briefly so the export doesn't race with in-flight
    /// inference calls on the shared champion graph, then
    /// resumes. Uses `pauseAndWait(timeoutMs:)` so a
    /// mid-save session end can't deadlock the save task.
    func handleSaveChampionAsModel() {
        // Belt-and-suspenders guards — menu disable is the primary
        // gate but these cover keyboard-shortcut / URL-scheme
        // invocations under a race.
        if checkpoint?.checkpointSaveInFlight == true {
            onRefuseMenuAction("A save is already in progress. Wait for it to finish.")
            return
        }
        if isArenaRunning {
            onRefuseMenuAction("Can't save the champion while the arena is running. Wait for it to finish.")
            return
        }
        if isBusyProvider() && !realTraining {
            onRefuseMenuAction("Another operation is in progress. Wait for it to finish, then try again.")
            return
        }
        guard let champion = network else {
            onRefuseMenuAction("Build or load a model first.")
            return
        }
        let championID = champion.identifier?.description ?? "unknown"
        // Snapshot the active self-play gate up front. If there
        // is no active session, we can safely export directly —
        // nobody is racing against us.
        let gate = activeSelfPlayGate
        checkpoint?.checkpointSaveInFlight = true
        checkpoint?.setCheckpointStatus("Saving champion…", kind: .progress)
        checkpoint?.startSlowSaveWatchdog(label: "champion save")

        Task {
            // Pause worker 0 if a session is running. Bail with a
            // user-visible error on timeout (indicates the session
            // has already ended or the worker is stuck — either way
            // we shouldn't spin forever).
            if let gate {
                let acquired = await gate.pauseAndWait(timeoutMs: Self.saveGateTimeoutMs)
                if !acquired {
                    checkpoint?.cancelSlowSaveWatchdog()
                    checkpoint?.checkpointSaveInFlight = false
                    checkpoint?.setCheckpointStatus("Save aborted: could not pause self-play (timeout)", kind: .error)
                    return
                }
            }

            var championWeights: [[Float]] = []
            var exportError: Error?
            do {
                championWeights = try await Task.detached(priority: .userInitiated) {
                    try await champion.exportWeights()
                }.value
            } catch {
                exportError = error
            }
            gate?.resume()

            if let exportError {
                checkpoint?.cancelSlowSaveWatchdog()
                checkpoint?.checkpointSaveInFlight = false
                checkpoint?.setCheckpointStatus("Save failed (export): \(exportError.localizedDescription)", kind: .error)
                SessionLogger.shared.log("[CHECKPOINT] Save champion export failed: \(exportError.localizedDescription)")
                return
            }

            let metadata = ModelCheckpointMetadata(
                creator: "manual",
                trainingStep: trainingStats?.steps,
                parentModelID: "",
                notes: "Manual Save Champion export"
            )
            let createdAtUnix = Int64(Date().timeIntervalSince1970)

            let outcome: Result<URL, Error> = await Task.detached(priority: .userInitiated) {
                do {
                    let url = try await CheckpointManager.saveModel(
                        weights: championWeights,
                        modelID: championID,
                        createdAtUnix: createdAtUnix,
                        metadata: metadata,
                        trigger: "manual"
                    )
                    return .success(url)
                } catch {
                    return .failure(error)
                }
            }.value
            checkpoint?.cancelSlowSaveWatchdog()
            checkpoint?.checkpointSaveInFlight = false
            switch outcome {
            case .success(let url):
                checkpoint?.setCheckpointStatus("Saved \(url.lastPathComponent)", kind: .success)
                SessionLogger.shared.log("[CHECKPOINT] Saved champion: \(url.lastPathComponent)")
            case .failure(let error):
                checkpoint?.setCheckpointStatus("Save failed: \(error.localizedDescription)", kind: .error)
                SessionLogger.shared.log("[CHECKPOINT] Save champion failed: \(error.localizedDescription)")
            }
        }
    }

    /// Upper bound on how long a save path will wait for a
    /// worker to acknowledge a pause request. Has to cover one
    /// in-flight self-play game or training step, so 15 s is a
    /// comfortable margin above the worst-case game length at
    /// typical self-play rates. On timeout the save bails with
    /// a user-visible error rather than blocking forever.
    nonisolated static let saveGateTimeoutMs: Int = 15_000

    /// Manual "Save Session" — writes a full `.dcmsession` with
    /// champion and trainer model files plus `session.json`.
    /// Requires an active Play-and-Train session and an available
    /// trainer. Briefly pauses both self-play worker 0 and the
    /// training gate to snapshot the two networks' weights.
    func handleSaveSessionManual() {
        // Belt-and-suspenders guards — menu disable is the primary
        // gate but these cover keyboard-shortcut / URL-scheme
        // invocations under a race.
        if checkpoint?.checkpointSaveInFlight == true {
            onRefuseMenuAction("A save is already in progress. Wait for it to finish.")
            return
        }
        if isArenaRunning {
            onRefuseMenuAction("Can't save the session while the arena is running. Wait for it to finish.")
            return
        }
        guard realTraining,
              let champion = network,
              let trainer,
              let selfPlayGate = activeSelfPlayGate,
              let trainingGate = activeTrainingGate else {
            onRefuseMenuAction("No active training session to save. Start Play and Train first.")
            return
        }
        saveSessionInternal(
            champion: champion,
            trainer: trainer,
            selfPlayGate: selfPlayGate,
            trainingGate: trainingGate,
            trigger: .manual
        )
    }

    /// Fired by `PeriodicSaveController` when its 4-hour deadline
    /// elapses (after any arena-deferral has resolved). Behaves
    /// exactly like the manual save path but tagged `.periodic` so
    /// the filename, status-line, and log line distinguish the two
    /// triggers. The controller has already decided we should fire;
    /// any remaining guard failures here (no session, save already
    /// in flight) just make the periodic attempt a no-op — the next
    /// tick of the controller will re-fire since `noteSuccessfulSave`
    /// is never called.
    @MainActor
    private func handleSaveSessionPeriodic() {
        // Guard against an arena starting in the tiny race window
        // between the controller's decide() and this call.
        if isArenaRunning {
            return
        }
        if checkpoint?.checkpointSaveInFlight == true {
            SessionLogger.shared.log("[CHECKPOINT] Periodic save skipped — another save is in flight")
            return
        }
        guard realTraining,
              let champion = network,
              let trainer,
              let selfPlayGate = activeSelfPlayGate,
              let trainingGate = activeTrainingGate else {
            // Should not happen if the controller is armed correctly,
            // but disarm ourselves and bail to be safe.
            periodicSaveController?.disarm()
            return
        }
        periodicSaveInFlight = true
        saveSessionInternal(
            champion: champion,
            trainer: trainer,
            selfPlayGate: selfPlayGate,
            trainingGate: trainingGate,
            trigger: .periodic
        )
    }

    /// Shared save-session internal used by both the manual save
    /// button and the periodic autosave. Handles the gate dance,
    /// exports both networks, builds the session state on the main
    /// actor, and fires off the actual write to a detached task.
    /// The post-promotion autosave uses its own inline code path
    /// (in the arena coordinator) because it re-uses weights already
    /// snapshotted under the arena's own pause and so does not need
    /// to dance the gates again here.
    private func saveSessionInternal(
        champion: ChessMPSNetwork,
        trainer: ChessTrainer,
        selfPlayGate: WorkerPauseGate,
        trainingGate: WorkerPauseGate,
        trigger: SessionSaveTrigger
    ) {
        let championID = champion.identifier?.description ?? "unknown"
        let trainerID = trainer.identifier?.description ?? "unknown"
        let diskTag = trigger.diskTag
        let uiSuffix = trigger.uiSuffix
        checkpoint?.checkpointSaveInFlight = true
        checkpoint?.setCheckpointStatus("Saving session\(uiSuffix)…", kind: .progress)
        checkpoint?.startSlowSaveWatchdog(label: "session save\(uiSuffix)")

        // Build the state snapshot on the main actor before
        // jumping to detached work. Capture the replay buffer handle
        // here too so the detached write path can serialize it
        // alongside the two network files — `ReplayBuffer` is
        // `@unchecked Sendable` and serializes access via its own
        // lock, so the buffer can be written from a background task
        // while self-play workers (which only append) are paused.
        let sessionState = buildCurrentSessionState(
            championID: championID,
            trainerID: trainerID
        )
        let trainingStep = trainingStats?.steps ?? 0
        let bufferForSave = replayBuffer
        // Snapshot the chart-coordinator state on the main actor
        // BEFORE jumping to detached work — the rings are
        // `@MainActor`-isolated so the array copies have to happen
        // here. `buildSnapshot()` returns nil when collection is off
        // or both rings are empty, in which case the save path skips
        // writing chart-companion files entirely (matching the
        // existing `bufferForSave == nil` skip).
        let chartSnapshotForSave = chartCoordinator?.buildSnapshot()

        Task {
            // Helper to clear both in-flight flags consistently on
            // every early-return path below. The periodic flag is
            // only meaningful when `trigger == .periodic`, but it's
            // cheap to always clear so we don't have to repeat the
            // branch on every error exit. Cancels the slow-save
            // watchdog too so a fast-failure path doesn't leave a
            // stale "Saving… (still running)" amber line behind.
            @MainActor func clearInFlight() {
                checkpoint?.cancelSlowSaveWatchdog()
                checkpoint?.checkpointSaveInFlight = false
                periodicSaveInFlight = false
            }

            // Pause self-play briefly so the champion export is
            // race-free, snapshot weights, then resume. Uses the
            // bounded variant so a session end mid-save doesn't
            // spin forever waiting for workers that have exited.
            let selfPlayAcquired = await selfPlayGate.pauseAndWait(timeoutMs: Self.saveGateTimeoutMs)
            guard selfPlayAcquired else {
                clearInFlight()
                checkpoint?.setCheckpointStatus("Save aborted: could not pause self-play (timeout)", kind: .error)
                SessionLogger.shared.log("[CHECKPOINT] Save session aborted at self-play pause timeout")
                return
            }
            var championWeights: [[Float]] = []
            var championError: Error?
            do {
                championWeights = try await Task.detached(priority: .userInitiated) {
                    try await champion.exportWeights()
                }.value
            } catch {
                championError = error
            }
            selfPlayGate.resume()

            if let championError {
                clearInFlight()
                checkpoint?.setCheckpointStatus("Save failed (champion export): \(championError.localizedDescription)", kind: .error)
                SessionLogger.shared.log("[CHECKPOINT] Save session failed at champion export: \(championError.localizedDescription)")
                return
            }

            // Pause training briefly to snapshot trainer weights.
            let trainingAcquired = await trainingGate.pauseAndWait(timeoutMs: Self.saveGateTimeoutMs)
            guard trainingAcquired else {
                clearInFlight()
                checkpoint?.setCheckpointStatus("Save aborted: could not pause training (timeout)", kind: .error)
                SessionLogger.shared.log("[CHECKPOINT] Save session aborted at training pause timeout")
                return
            }
            var trainerWeights: [[Float]] = []
            var trainerError: Error?
            do {
                // exportTrainerWeights bundles trainables + bn +
                // momentum velocity. Caller is responsible for
                // pausing both gates, which we did above.
                trainerWeights = try await Task.detached(priority: .userInitiated) {
                    try await trainer.exportTrainerWeights()
                }.value
            } catch {
                trainerError = error
            }
            trainingGate.resume()

            if let trainerError {
                clearInFlight()
                checkpoint?.setCheckpointStatus("Save failed (trainer export): \(trainerError.localizedDescription)", kind: .error)
                SessionLogger.shared.log("[CHECKPOINT] Save session failed at trainer export: \(trainerError.localizedDescription)")
                return
            }

            // Final write + verify on a detached task so UI stays
            // responsive during the ~150 ms scratch-network build.
            let championMetadata = ModelCheckpointMetadata(
                creator: diskTag,
                trainingStep: trainingStep,
                parentModelID: "",
                notes: "Session checkpoint (\(diskTag))"
            )
            let trainerMetadata = ModelCheckpointMetadata(
                creator: diskTag,
                trainingStep: trainingStep,
                parentModelID: championID,
                notes: "Trainer lineage at session checkpoint (\(diskTag))"
            )
            let now = Int64(Date().timeIntervalSince1970)
            let outcome: Result<URL, Error> = await Task.detached(priority: .userInitiated) {
                [bufferForSave, chartSnapshotForSave] in
                do {
                    let url = try await CheckpointManager.saveSession(
                        championWeights: championWeights,
                        championID: championID,
                        championMetadata: championMetadata,
                        championCreatedAtUnix: now,
                        trainerWeights: trainerWeights,
                        trainerID: trainerID,
                        trainerMetadata: trainerMetadata,
                        trainerCreatedAtUnix: now,
                        state: sessionState,
                        replayBuffer: bufferForSave,
                        chartSnapshot: chartSnapshotForSave,
                        trigger: diskTag
                    )
                    return .success(url)
                } catch {
                    return .failure(error)
                }
            }.value

            clearInFlight()
            switch outcome {
            case .success(let url):
                checkpoint?.setCheckpointStatus("Saved \(url.lastPathComponent)\(uiSuffix)", kind: .success)
                let bufStr: String
                if let snap = bufferForSave?.stateSnapshot() {
                    bufStr = " replay=\(snap.storedCount)/\(snap.capacity)"
                } else {
                    bufStr = ""
                }
                SessionLogger.shared.log(
                    "[CHECKPOINT] Saved session (\(diskTag)): \(url.lastPathComponent) build=\(BuildInfo.buildNumber) git=\(BuildInfo.gitHash)\(bufStr)"
                )
                checkpoint?.recordLastSessionPointer(
                    directoryURL: url,
                    sessionID: sessionState.sessionID,
                    trigger: diskTag
                )
                periodicSaveController?.noteSuccessfulSave(at: Date())
                checkpoint?.lastSavedAt = Date()
            case .failure(let error):
                checkpoint?.setCheckpointStatus("Save failed: \(error.localizedDescription)", kind: .error)
                SessionLogger.shared.log("[CHECKPOINT] Save session (\(diskTag)) failed: \(error.localizedDescription)")
            }
        }
    }

    /// Load a standalone `.dcmmodel` into the current champion
    /// network. Triggered from the Load Model file importer. The
    /// network must exist (loading into a built network preserves
    /// the existing graph compilation; we don't rebuild).
    func handleLoadModelPickResult(_ result: Result<[URL], Error>) {
        switch result {
        case .failure(let error):
            checkpoint?.setCheckpointStatus("Load cancelled: \(error.localizedDescription)", kind: .error)
        case .success(let urls):
            guard let url = urls.first else { return }
            loadModelFrom(url: url)
        }
    }

    func loadModelFrom(url: URL) {
        // In-function guards (belt-and-suspenders with menu disable).
        if isBuildingOrBusyProvider() {
            onRefuseMenuAction(busyReasonProvider())
            return
        }

        checkpoint?.checkpointSaveInFlight = true
        checkpoint?.setCheckpointStatus("Loading \(url.lastPathComponent)…", kind: .progress)

        Task {
            // Auto-build the champion shell if it doesn't exist yet.
            // The weights are about to be overwritten, so the random
            // init is only satisfying graph compilation — no reason
            // to require the user to press Build first.
            let championResult = await self.ensureChampionBuilt()
            switch championResult {
            case .failure(let error):
                checkpoint?.checkpointSaveInFlight = false
                checkpoint?.setCheckpointStatus("Build failed: \(error.localizedDescription)", kind: .error)
                SessionLogger.shared.log("[CHECKPOINT] Load model auto-build failed: \(error.localizedDescription)")
                return
            case .success(let champion):
                // Keep the security scope open across the entire
                // detached read+load so files picked from outside the
                // sandbox (Downloads, AirDrop, external volumes) stay
                // accessible until the work finishes. Start/stop must
                // happen inside the detached closure to bracket the
                // actual I/O.
                let outcome: Result<ModelCheckpointFile, Error> = await Task.detached(priority: .userInitiated) {
                    let scopeAccessed = url.startAccessingSecurityScopedResource()
                    defer {
                        if scopeAccessed {
                            url.stopAccessingSecurityScopedResource()
                        }
                    }
                    do {
                        let file = try CheckpointManager.loadModelFile(at: url)
                        try await champion.loadWeights(file.weights)
                        return .success(file)
                    } catch {
                        return .failure(error)
                    }
                }.value
                checkpoint?.checkpointSaveInFlight = false
                switch outcome {
                case .success(let file):
                    champion.identifier = ModelID(value: file.modelID)
                    networkStatus = "Loaded model \(file.modelID)\nFrom: \(url.lastPathComponent)"
                    checkpoint?.setCheckpointStatus("Loaded \(file.modelID)", kind: .success)
                    SessionLogger.shared.log("[CHECKPOINT] Loaded model: \(url.lastPathComponent) → \(file.modelID)")
                    onClearInferenceResult()
                    // Flag champion-replaced for the post-Stop Start
                    // dialog's "Continue" annotation. Cleared as
                    // soon as a new training segment starts.
                    if replayBuffer != nil {
                        championLoadedSinceLastTrainingSegment = true
                    }
                case .failure(let error):
                    checkpoint?.setCheckpointStatus("Load failed: \(error.localizedDescription)", kind: .error)
                    SessionLogger.shared.log("[CHECKPOINT] Load model failed: \(error.localizedDescription)")
                }
            }
        }
    }

    /// Load a `.dcmsession` directory. Parses everything, loads
    /// champion weights immediately into the live champion
    /// network, and stores the session state + trainer weights
    /// in `pendingLoadedSession` so the next Play-and-Train start
    /// resumes from them.
    func handleLoadSessionPickResult(_ result: Result<[URL], Error>) {
        switch result {
        case .failure(let error):
            checkpoint?.setCheckpointStatus("Load cancelled: \(error.localizedDescription)", kind: .error)
        case .success(let urls):
            guard let url = urls.first else { return }
            loadSessionFrom(url: url)
        }
    }

    func loadSessionFrom(url: URL, startAfterLoad: Bool = false) {
        // In-function guards (belt-and-suspenders with menu disable).
        if isBuildingOrBusyProvider() {
            onRefuseMenuAction(busyReasonProvider())
            return
        }

        checkpoint?.checkpointSaveInFlight = true
        checkpoint?.setCheckpointStatus("Loading session \(url.lastPathComponent)…", kind: .progress)

        Task {
            // Auto-build the champion shell if it doesn't exist yet.
            // The weights are about to be overwritten, so the random
            // init is only satisfying graph compilation — no reason
            // to require the user to press Build first.
            let championResult = await self.ensureChampionBuilt()
            guard case .success(let champion) = championResult else {
                checkpoint?.checkpointSaveInFlight = false
                if case .failure(let error) = championResult {
                    checkpoint?.setCheckpointStatus("Build failed: \(error.localizedDescription)", kind: .error)
                    SessionLogger.shared.log("[CHECKPOINT] Load session auto-build failed: \(error.localizedDescription)")
                }
                return
            }
            let outcome: Result<LoadedSession, Error> = await Task.detached(priority: .userInitiated) {
                let scopeAccessed = url.startAccessingSecurityScopedResource()
                defer {
                    if scopeAccessed {
                        url.stopAccessingSecurityScopedResource()
                    }
                }
                do {
                    let loaded = try CheckpointManager.loadSession(at: url)
                    // Apply champion weights immediately; trainer
                    // weights are held for the next startRealTraining.
                    try await champion.loadWeights(loaded.championFile.weights)
                    return .success(loaded)
                } catch {
                    return .failure(error)
                }
            }.value
            checkpoint?.checkpointSaveInFlight = false
            switch outcome {
            case .success(let loaded):
                champion.identifier = ModelID(value: loaded.championFile.modelID)
                pendingLoadedSession = loaded
                networkStatus = """
                    Loaded session \(loaded.state.sessionID)
                    Champion: \(loaded.championFile.modelID)
                    Trainer: \(loaded.trainerFile.modelID)
                    Steps: \(loaded.state.trainingSteps) / Games: \(loaded.state.selfPlayGames)
                    Click Play and Train to resume.
                    """
                checkpoint?.lastSavedAt = nil
                checkpoint?.setCheckpointStatus("Loaded session \(loaded.state.sessionID) — click Play and Train to resume", kind: .success)
                let savedBuild = loaded.state.buildNumber.map(String.init) ?? "?"
                let savedGit = loaded.state.buildGitHash ?? "?"
                let bufStr: String
                if let stored = loaded.state.replayBufferStoredCount,
                   let cap = loaded.state.replayBufferCapacity {
                    bufStr = " replay=\(stored)/\(cap)"
                } else {
                    bufStr = " replay=none"
                }
                SessionLogger.shared.log("[CHECKPOINT] Loaded session: \(url.lastPathComponent) savedBuild=\(savedBuild) savedGit=\(savedGit)\(bufStr)")
                onClearInferenceResult()
                if startAfterLoad {
                    // Auto-resume path (from the launch-time sheet or
                    // the File menu "Resume training from autosave"
                    // command). Chain straight into Play-and-Train so
                    // the user's single click results in the session
                    // both loaded AND running.
                    SessionLogger.shared.log("[CHECKPOINT] Auto-resume: starting Play-and-Train on loaded session")
                    startRealTraining()
                }
            case .failure(let error):
                checkpoint?.setCheckpointStatus("Load failed: \(error.localizedDescription)", kind: .error)
                SessionLogger.shared.log("[CHECKPOINT] Load session failed: \(error.localizedDescription)")
            }
            if startAfterLoad {
                onResumeFinished()
            }
        }
    }

    // startRealTraining(mode:) / stopRealTraining() moved to SessionController+Training.swift.

    // MARK: - Arena tournament (Stage 4n)

    /// Max barrier-wait for the arena's batched move evaluator (ms). Arena
    /// games run a small concurrency so the batch rarely fills; the wait caps
    /// per-ply latency.
    nonisolated static let arenaBatchWaitMs: Double = 100.0

    /// Whether to write a `-promote.dcmsession` autosave after each arena
    /// promotion (re-using the weight snapshots taken under the arena's
    /// pauses). On by default.
    nonisolated static let autosaveSessionsOnPromote: Bool = true


    /// Run one arena tournament in parallel mode — 200 games between
    /// the candidate (synced from trainer at start) and the arena
    /// champion (synced from the real champion at start), while
    /// self-play and training continue running in the background.
    /// Promotes the candidate into the real champion iff the score
    /// meets the 0.55 threshold.
    ///
    /// Synchronization: this is called from the arena coordinator
    /// task, which is a peer to the self-play and training workers.
    /// Training and self-play are briefly paused at arena start so
    /// the method can take trainer and champion snapshots into
    /// dedicated arena-only networks; after that the arena runs
    /// exclusively on those two snapshots and doesn't touch the
    /// "live" trainer or champion again until promotion (which
    /// briefly re-pauses self-play to write into the champion).
    /// Candidate test probes skip while `arenaFlag.isActive` so
    /// they don't race with arena reads of the candidate inference
    /// network.
    @MainActor
    func runArenaParallel(
        trainer: ChessTrainer,
        champion: ChessMPSNetwork,
        candidateInference: ChessMPSNetwork,
        arenaChampion: ChessMPSNetwork,
        tBox: TournamentLiveBox,
        selfPlayGate: WorkerPauseGate,
        trainingGate: WorkerPauseGate,
        arenaFlag: ArenaActiveFlag,
        overrideBox: ArenaOverrideBox
    ) async {
        // Clear any stale decision from a previous tournament so this
        // run starts with a clean override slate. Normal completion
        // `consume()`s the box at the end, but early-return paths
        // (cancellation, sync errors) don't — clearing here keeps
        // all exit paths honest.
        _ = overrideBox.consume()
        let arenaStartTrainingSnapshot = await trainingBox?.snapshot()
        let steps = arenaStartTrainingSnapshot?.stats.steps ?? trainingStats?.steps ?? 0
        if let arenaStartTrainingSnapshot {
            trainingStats = arenaStartTrainingSnapshot.stats
            lastTrainStep = arenaStartTrainingSnapshot.lastTiming
            realRollingPolicyLoss = arenaStartTrainingSnapshot.rollingPolicyLoss
            realRollingValueLoss = arenaStartTrainingSnapshot.rollingValueLoss
        }

        let trainerIDStart = trainer.identifier?.description ?? "?"
        let championIDStart = champion.identifier?.description ?? "?"
        SessionLogger.shared.log(
            "[ARENA] start  step=\(steps) trainer=\(trainerIDStart) champion=\(championIDStart)"
        )
        // Snapshot losses/entropy at arena start so the log shows
        // the trainer's state entering the arena — especially
        // useful for diagnosing whether divergence was already
        // underway before the arena ran.
        if let snap = arenaStartTrainingSnapshot {
            let pStr = snap.rollingPolicyLoss.map { String(format: "%+.4f", $0) } ?? "--"
            let vStr = snap.rollingValueLoss.map { String(format: "%+.4f", $0) } ?? "--"
            let eStr = snap.rollingPolicyEntropy.map { String(format: "%.4f", $0) } ?? "--"
            let gStr = snap.rollingGradGlobalNorm.map { String(format: "%.3f", $0) } ?? "--"
            let vmStr = snap.rollingValueMean.map { String(format: "%+.4f", $0) } ?? "--"
            let vaStr = snap.rollingValueAbsMean.map { String(format: "%.4f", $0) } ?? "--"
            let bufCount = replayBuffer?.count ?? 0
            let bufCap = replayBuffer?.capacity ?? TrainingParameters.shared.replayBufferCapacity
            SessionLogger.shared.log(
                "[STATS] arena-start  steps=\(steps) buffer=\(bufCount)/\(bufCap) pLoss=\(pStr) vLoss=\(vStr) pEnt=\(eStr) gNorm=\(gStr) vMean=\(vmStr) vAbs=\(vaStr) trainer=\(trainerIDStart) champion=\(championIDStart)"
            )
        }

        // Mark arena active and seed live progress. Arena-active
        // suppresses the candidate test probe for the duration so
        // probe and arena don't race on the candidate inference
        // network. isArenaRunning is @State mirror the UI reads to
        // disable the Run Arena button and adjust the busy label.
        arenaFlag.set()
        isArenaRunning = true
        // Let the periodic-save scheduler know an arena is in
        // progress. A 4-hour deadline that crosses while an arena
        // runs will be held as a pending fire and only dispatched
        // once the arena ends — unless a post-promotion autosave
        // lands in the meantime, in which case the pending fire is
        // swallowed (see `PeriodicSaveController`).
        periodicSaveController?.noteArenaBegan()
        // Record the arena's start elapsed-second position so the
        // chart grid's arena activity tile can render a live band
        // as the arena progresses, rather than only showing the
        // arena post-hoc when the completed ArenaChartEvent lands.
        // Anchors off `chartCoordinator?.chartElapsedAnchor` so the
        // arena's x-position lands on the same axis as the training
        // + progress-rate samples. (Routing this off
        // `parallelStats.sessionStart` instead would put restored
        // chart data and post-resume arena bands in different
        // coordinate spaces — the back-dated chart axis vs. the
        // fresh per-segment parallel-stats axis.)
        chartCoordinator?.recordArenaStarted(
            elapsedSec: Date().timeIntervalSince(chartCoordinator?.chartElapsedAnchor ?? Date())
        )

        let totalGames = TrainingParameters.shared.arenaGamesPerTournament
        let startTime = Date()
        tBox.update(TournamentProgress(
            currentGame: 0,
            totalGames: totalGames,
            candidateWins: 0,
            championWins: 0,
            draws: 0,
            startTime: startTime
        ))
        tournamentProgress = tBox.snapshot()

        // --- Trainer → candidate inference snapshot ---
        //
        // Pause the training worker briefly (a few ms, at most one
        // SGD step) so we can export trainer weights without racing
        // against a concurrent `trainer.trainStep`. Release training
        // as soon as the snapshot lands — the rest of the arena
        // runs on `candidateInference`, not on `trainer`, so training
        // can continue through the 200 games.
        await trainingGate.pauseAndWait()
        if Task.isCancelled {
            trainingGate.resume()
            cleanupArenaState(arenaFlag: arenaFlag, tBox: tBox)
            return
        }
        // Capture trainer weights here and hold them through the
        // rest of the arena. At arena end we use them to autosave
        // the session without needing another training pause
        // (and therefore without any gate interaction from the
        // autosave task, which is critical to avoid deadlocking
        // a save that runs past a session cancel — unstructured
        // save tasks don't inherit realTrainingTask cancellation).
        //
        // We also snapshot the optimizer velocity. If the candidate
        // wins the arena, we'll restore THIS snapshot when we copy
        // the candidate weights back into the trainer — the velocity
        // that built the validated candidate is the right velocity
        // for the candidate's weight surface. (Earlier behavior
        // zeroed velocity on promotion, throwing away accumulated
        // gradient signal.)
        var trainerSnapshotWeights: [[Float]] = []
        var trainerSnapshotVelocity: [[Float]] = []
        do {
            let snapshot: ([[Float]], [[Float]]) = try await Task.detached(priority: .userInitiated) {
                let weights = try await trainer.network.exportWeights()
                let velocity = try await trainer.exportVelocitySnapshot()
                try await candidateInference.loadWeights(weights)
                return (weights, velocity)
            }.value
            trainerSnapshotWeights = snapshot.0
            trainerSnapshotVelocity = snapshot.1
        } catch {
            trainingBox?.recordError("Arena candidate sync failed: \(error.localizedDescription)")
            trainingGate.resume()
            cleanupArenaState(arenaFlag: arenaFlag, tBox: tBox)
            return
        }
        trainingGate.resume()

        // Arena candidate inherits the trainer's current generation
        // ID. If it gets promoted, the promoted champion should keep
        // that exact identifier and the live trainer will roll forward
        // to the next generation after being rewound to the promoted
        // weights.
        candidateInference.identifier = trainer.identifier

        // --- Champion → arena champion snapshot ---
        //
        // Same pattern for self-play: brief pause, copy weights from
        // the real champion into the arena-only champion network,
        // release. Arena games from here on only read
        // `arenaChampion`, leaving the real champion free for
        // continuous self-play through the tournament.
        await selfPlayGate.pauseAndWait()
        if Task.isCancelled {
            selfPlayGate.resume()
            cleanupArenaState(arenaFlag: arenaFlag, tBox: tBox)
            return
        }
        do {
            try await Task.detached(priority: .userInitiated) {
                let weights = try await champion.exportWeights()
                try await arenaChampion.loadWeights(weights)
            }.value
        } catch {
            trainingBox?.recordError("Arena champion sync failed: \(error.localizedDescription)")
            selfPlayGate.resume()
            cleanupArenaState(arenaFlag: arenaFlag, tBox: tBox)
            return
        }
        selfPlayGate.resume()

        // arenaChampion is a pure copy of champion's current weights,
        // so it inherits the champion's ID verbatim — no mint.
        arenaChampion.identifier = champion.identifier

        // --- Tournament games ---
        //
        // Cancellation: the detached tournament task is wrapped in
        // `withTaskCancellationHandler` so clicking Stop flips a
        // `CancelBox` that `TournamentDriver.run` checks between
        // games. The worst-case delay from Stop to actually breaking
        // out is one in-flight arena game (~400 ms).
        let cancelBox = CancelBox()
        let arenaDiversity = GameDiversityTracker(windowSize: totalGames)
        // Snapshot the current arena schedule once at tournament start
        // so every game in this arena uses the same tau settings, even
        // if the user edits the fields mid-tournament.
        let arenaScheduleSnapshot = samplingScheduleBox?.arena ?? buildArenaSchedule()

        // --- Concurrent arena: build per-network batchers ---
        //
        // K parallel arena games share two `BatchedMoveEvaluationSource`
        // batchers, one per network. Each game alternates candidate ↔
        // champion moves, so at any instant ~K/2 are pending on each
        // side; a strict count-only barrier (the self-play model) would
        // never fire either batcher in steady state. The coalescing-
        // window timer (`maxBatchWaitMs`) makes the barrier fire on
        // whichever happens first — count met OR window expired —
        // which captures the early-game synchronized phase as a full
        // K-batch and the desynchronized steady state as ~K/2 partial
        // batches.
        //
        // Live count tracking: `expectedSlotCount` starts at `liveK`
        // (clamped to game count) and stays there while replacements
        // keep the pool full. The `onSlotRetired` callback below
        // fires only when a slot leaves the pool (no replacement
        // spawned) — i.e. exactly when each of the final K games
        // finishes — so each batcher's `expectedSlotCount` decrements
        // 1:1 with live game count during the tail. Decrementing on
        // every completion (the prior approach) drove the count to
        // 0 partway through the tournament and forced the rest of
        // the run through drain mode = single-position batches.
        let liveK = max(1, min(TrainingParameters.shared.arenaConcurrency, totalGames))
        // Joint GPU-busy-wall accumulator shared across both batchers.
        // Each fire reports start/end around its `network.evaluate`
        // await; the timer accumulates wall time during which AT
        // LEAST ONE side was on the GPU. Read once at the end of
        // arena to emit a single `[ARENA] timing joint` line that's
        // directly comparable to arena wall.
        let arenaGpuTimer = ArenaGpuTimer()
        let candidateBatcher = BatchedMoveEvaluationSource(
            network: candidateInference,
            maxBatchWaitMs: Self.arenaBatchWaitMs,
            name: "candidate",
            logBatchTimings: true,
            gpuTimer: arenaGpuTimer
        )
        let championBatcher = BatchedMoveEvaluationSource(
            network: arenaChampion,
            maxBatchWaitMs: Self.arenaBatchWaitMs,
            name: "champion",
            logBatchTimings: true,
            gpuTimer: arenaGpuTimer
        )
        await candidateBatcher.setExpectedSlotCount(liveK)
        await championBatcher.setExpectedSlotCount(liveK)

        // Update the live progress box with the chosen concurrency so
        // the arena's busy-label suffix can render "(×K concurrent)"
        // for the duration of this run.
        tBox.update(TournamentProgress(
            currentGame: 0,
            totalGames: totalGames,
            candidateWins: 0,
            championWins: 0,
            draws: 0,
            startTime: startTime,
            concurrency: liveK
        ))
        tournamentProgress = tBox.snapshot()

        // Records collector. The driver fires `onGameRecorded` from
        // its parent harvest loop — already serial in that task — so
        // this Box's append happens on a single task and needs no
        // synchronization. We lift it to a `final class` only because
        // the closure must be `@Sendable`; the lock is a defensive
        // belt for that contract, not for actual contention.
        let recordsBox = TournamentRecordsBox()
        let stats: TournamentStats
        do {
            stats = try await withTaskCancellationHandler {
                try await Task.detached(priority: .userInitiated) {
                    [candidateBatcher, championBatcher, tBox, cancelBox, overrideBox, arenaDiversity, arenaScheduleSnapshot, liveK, recordsBox] in
                    let driver = TournamentDriver()
                    return try await driver.run(
                        playerA: {
                            MPSChessPlayer(
                                name: "Candidate",
                                source: candidateBatcher,
                                schedule: arenaScheduleSnapshot
                            )
                        },
                        playerB: {
                            MPSChessPlayer(
                                name: "Champion",
                                source: championBatcher,
                                schedule: arenaScheduleSnapshot
                            )
                        },
                        games: totalGames,
                        concurrency: liveK,
                        diversityTracker: arenaDiversity,
                        // The driver checks this between games. Either a
                        // task-cancel (session Stop) or a user Abort /
                        // Promote click breaks the game loop early; the
                        // caller below disambiguates the two via the
                        // override box's `consume()`.
                        isCancelled: { cancelBox.isCancelled || overrideBox.isActive },
                        onGameCompleted: { gameIndex, aWins, bWins, draws in
                            tBox.update(TournamentProgress(
                                currentGame: gameIndex,
                                totalGames: totalGames,
                                candidateWins: aWins,
                                championWins: bWins,
                                draws: draws,
                                startTime: startTime,
                                concurrency: liveK
                            ))
                        },
                        onSlotRetired: {
                            // Fires only when a slot leaves the
                            // live pool (no replacement spawned).
                            // Decrementing on every completion would
                            // peg the count at 0 mid-tournament and
                            // collapse all remaining batches to size 1.
                            await candidateBatcher.decrementExpectedSlotCount()
                            await championBatcher.decrementExpectedSlotCount()
                        },
                        onGameRecorded: { record in
                            recordsBox.append(record)
                        }
                    )
                }.value
            } onCancel: {
                cancelBox.cancel()
            }
        } catch {
            // Error path. Drain both batchers so any parked
            // continuations resume, then emit whatever telemetry we
            // can over the partial records captured before the throw
            // (so the user can still see how much concurrency we got
            // up to the failure and confirm whether the captured
            // games were internally consistent before bailing).
            // After that, surface the error and tear down arena
            // state.
            await candidateBatcher.setExpectedSlotCount(0)
            await championBatcher.setExpectedSlotCount(0)
            await ArenaTelemetryFormatter.emitPostRunTelemetry(
                candidateBatcher: candidateBatcher,
                championBatcher: championBatcher,
                gpuTimer: arenaGpuTimer,
                recordsBox: recordsBox,
                wasCancelled: cancelBox.isCancelled || overrideBox.isActive,
                context: "after error",
                arenaStartTime: startTime,
                trainingBox: trainingBox
            )
            trainingBox?.recordError("Arena tournament failed: \(error.localizedDescription)")
            cleanupArenaState(arenaFlag: arenaFlag, tBox: tBox)
            return
        }

        // Drain the batchers explicitly. Drain mode fires any
        // straggler pendings immediately so no continuation is left
        // parked, even on cancellation paths where the driver may
        // have stopped harvesting before all in-flight slots returned.
        await candidateBatcher.setExpectedSlotCount(0)
        await championBatcher.setExpectedSlotCount(0)

        await ArenaTelemetryFormatter.emitPostRunTelemetry(
            candidateBatcher: candidateBatcher,
            championBatcher: championBatcher,
            gpuTimer: arenaGpuTimer,
            recordsBox: recordsBox,
            wasCancelled: cancelBox.isCancelled || overrideBox.isActive,
            context: nil,
            arenaStartTime: startTime,
            trainingBox: trainingBox
        )

        // --- Score and promotion ---
        //
        // Branch on the user override first. `.abort` ends the
        // tournament with no promotion regardless of score; `.promote`
        // forces promotion regardless of score and games played; `nil`
        // is the normal path where the usual score-threshold check
        // decides. The consume also clears the box for the next
        // tournament.
        let overrideDecision = overrideBox.consume()
        let playedGames = stats.gamesPlayed
        let score: Double
        if playedGames > 0 {
            score = (Double(stats.playerAWins) + 0.5 * Double(stats.draws)) / Double(playedGames)
        } else {
            score = 0
        }
        var promoted = false
        var promotedID: ModelID?
        let shouldPromote: Bool
        // `promotionKind` tracks WHY the promotion is happening, so
        // the history / logs can distinguish a user-forced promotion
        // (Promote button) from a score-threshold one. Only read if
        // `promoted` ends up true.
        let promotionKind: PromotionKind?
        switch overrideDecision {
        case .abort:
            shouldPromote = false
            promotionKind = nil
        case .promote:
            shouldPromote = true
            promotionKind = .manual
        case .none:
            shouldPromote = playedGames >= totalGames && score >= TrainingParameters.shared.arenaPromoteThreshold
            promotionKind = shouldPromote ? .automatic : nil
        }
        // Holds the new champion weights if promotion succeeds,
        // so we can hand them to a detached autosave task at the
        // end without needing to re-read them from the live
        // network (which would race against self-play again).
        var promotedChampionWeights: [[Float]] = []
        if shouldPromote {
            // Pause both self-play and training, then copy the
            // promoted candidate into both the champion and the live
            // trainer. This keeps self-play and SGD aligned on the
            // exact promoted weights rather than letting training
            // continue from a later, unvalidated post-arena state.
            await selfPlayGate.pauseAndWait()
            await trainingGate.pauseAndWait()
            if !Task.isCancelled {
                do {
                    promotedChampionWeights = try await Task.detached(priority: .userInitiated) {
                        [candidateInference, champion, trainer, trainerSnapshotVelocity, steps] in
                        let weights = try await candidateInference.exportWeights()
                        try await champion.loadWeights(weights)
                        try await trainer.network.loadWeights(weights)
                        // The trainer's CURRENT velocity was built up
                        // against the post-arena weight surface (which
                        // we just discarded by overwriting with the
                        // candidate weights). Restore the velocity we
                        // snapshotted at arena-start instead — that
                        // velocity is the EMA of gradients that built
                        // the validated candidate, so it's the right
                        // accumulator for the candidate's weight
                        // surface. Both gates are paused at this point,
                        // so the trainer's velocity I/O is safe to
                        // drive directly on network.graph.
                        try await trainer.loadVelocitySnapshot(trainerSnapshotVelocity)
                        // CRITICAL: Rewind the trainer's completed step
                        // count to match the snapshotted weights. Without
                        // this, the trainer keeps its post-arena step
                        // count but uses arena-start weights, causing
                        // the LR warmup multiplier to jump ahead of
                        // the weights and drive the immature network
                        // into collapse with an oversized LR.
                        trainer.completedTrainSteps = steps
                        return weights
                    }.value
                    // Promoted: champion now holds the arena candidate's
                    // exact weights, so it inherits that snapshot ID,
                    // while the rewound live trainer rolls forward to
                    // the next mutable generation in the same lineage.
                    champion.identifier = candidateInference.identifier
                    trainer.identifier = ModelIDMinter.mintTrainerGeneration(
                        from: champion.identifier ?? candidateInference.identifier ?? ModelIDMinter.mint()
                    )
                    promoted = true
                    promotedID = candidateInference.identifier
                    if let arenaStartTrainingSnapshot {
                        trainingBox?.seed(arenaStartTrainingSnapshot.stats)
                        trainingStats = arenaStartTrainingSnapshot.stats
                    } else {
                        trainingBox?.setStepCount(steps)
                        if var stats = trainingStats {
                            stats.steps = steps
                            trainingStats = stats
                        }
                    }
                    trainingBox?.resetRollingWindows()
                    trainingAlarm?.resetStreaks()
                    trainingAlarm?.clear()
                } catch {
                    trainingBox?.recordError("Promotion copy failed: \(error.localizedDescription)")
                }
            }
            trainingGate.resume()
            selfPlayGate.resume()
        }

        // Append to history and clear arena state.
        let durationSec = Date().timeIntervalSince(startTime)
        let record = TournamentRecord(
            finishedAtStep: steps,
            finishedAt: Date(),
            candidateID: candidateInference.identifier,
            championID: arenaChampion.identifier,
            gamesPlayed: playedGames,
            candidateWins: stats.playerAWins,
            championWins: stats.playerBWins,
            draws: stats.draws,
            score: score,
            promoted: promoted,
            promotionKind: promoted ? promotionKind : nil,
            promotedID: promotedID,
            durationSec: durationSec,
            candidateWinsAsWhite: stats.playerAWinsAsWhite,
            candidateWinsAsBlack: stats.playerAWinsAsBlack,
            candidateLossesAsWhite: stats.playerALossesAsWhite,
            candidateLossesAsBlack: stats.playerALossesAsBlack,
            candidateDrawsAsWhite: stats.playerADrawsAsWhite,
            candidateDrawsAsBlack: stats.playerADrawsAsBlack
        )
        tournamentHistory.append(record)
        // Mirror into the chart-tile event stream. Compute the
        // elapsed-second start/end against the chart-coordinator's
        // anchor so the band lands on the same X axis as the
        // time-series charts. (See `recordArenaStarted` site for
        // why this MUST be the chart anchor and not
        // `parallelStats.sessionStart` — the chart anchor is
        // back-dated on resume so the band lines up with restored
        // chart data; the parallel-stats anchor is intentionally
        // fresh per segment for rate-display.)
        let endElapsed = max(0, Date().timeIntervalSince(chartCoordinator?.chartElapsedAnchor ?? Date()))
        // Prefer the live start mark captured at arena begin —
        // it avoids a ~5-second drift from backward-inferring
        // startElapsed out of (end - durationSec) after the
        // promotion work ran. Fall back to the durationSec math
        // only if the live mark is somehow nil.
        let startElapsed = chartCoordinator?.activeArenaStartElapsed
        ?? max(0, endElapsed - durationSec)
        // `recordArenaCompleted` appends the event AND clears
        // the live-band marker, so the chart drops back to just
        // the completed events on the next render.
        chartCoordinator?.recordArenaCompleted(ArenaChartEvent(
            id: (chartCoordinator?.arenaChartEvents.count ?? 0),
            startElapsedSec: startElapsed,
            endElapsedSec: endElapsed,
            score: score,
            promoted: promoted
        ))
        logArenaResult(
            record: record,
            index: tournamentHistory.count,
            trainer: trainer,
            candidate: candidateInference,
            championSide: arenaChampion,
            diversity: arenaDiversity.snapshot()
        )
        cleanupArenaState(arenaFlag: arenaFlag, tBox: tBox)

        // On promotion: reset game-play stats so the display
        // reflects only the new champion's self-play performance,
        // and emit a STATS log line so the post-promotion state
        // is visible in the session log (the fixed STATS ticker
        // may not fire for up to an hour at this point in the
        // schedule).
        if promoted {
            parallelWorkerStatsBox?.resetGameStats()
            let trainerIDStr = trainer.identifier?.description ?? "?"
            let championIDStr = champion.identifier?.description ?? "?"
            SessionLogger.shared.log(
                "[STATS] post-promote  steps=\(trainingStats?.steps ?? 0) champion=\(championIDStr) trainer=\(trainerIDStr)"
            )
        }

        // Post-promotion autosave. Fires a detached Task that
        // writes a full session snapshot using the weights we
        // already captured above under the arena-start training
        // pause and the promotion self-play pause. The detached
        // task touches no live networks and no pause gates, so
        // it is safe to run past a session cancel — unstructured
        // save tasks don't inherit `realTrainingTask`
        // cancellation, and any post-return gate interaction
        // here would potentially deadlock against workers that
        // have already exited their loops.
        //
        // Status row: we publish the "Saving… / Saved" progression
        // via the same `setCheckpointStatus` channel the manual
        // save uses, tagged "(post-promotion)" so users can tell
        // at a glance which autosave trigger produced the line.
        // `periodicSaveController` sees the success callback and
        // resets its 4-hour deadline, so a promotion that happens
        // shortly before a scheduled periodic save implicitly
        // defers the periodic one — which matches the spec: if a
        // post-promotion save already covered the window, the
        // next periodic tick runs a full 4 hours later from now.
        if promoted && Self.autosaveSessionsOnPromote && !promotedChampionWeights.isEmpty {
            let championID = champion.identifier?.description ?? "unknown"
            let trainerID = trainer.identifier?.description ?? "unknown"
            let sessionState = buildCurrentSessionState(
                championID: championID,
                trainerID: trainerID
            )
            let championMetadata = ModelCheckpointMetadata(
                creator: "promote",
                trainingStep: trainingStats?.steps ?? 0,
                parentModelID: "",
                notes: "Post-arena autosave after promotion"
            )
            let trainerMetadata = ModelCheckpointMetadata(
                creator: "promote",
                trainingStep: trainingStats?.steps ?? 0,
                parentModelID: championID,
                notes: "Trainer lineage at arena-start pause with optimizer velocity"
            )
            let createdAtUnix = Int64(Date().timeIntervalSince1970)
            // Copy captured arrays for clean Sendable semantics
            // (they're already Sendable but this makes the
            // transfer to the detached task explicit).
            let championWeightsSnapshot = promotedChampionWeights
            let trainerWeightsSnapshot = trainerSnapshotWeights + trainerSnapshotVelocity
            let bufferForAutosave = replayBuffer
            // Same main-actor snapshot rule as the manual/periodic
            // path — rings are @MainActor-isolated, so the array
            // copies have to happen here before we go detached.
            let chartSnapshotForAutosave = chartCoordinator?.buildSnapshot()

            checkpoint?.setCheckpointStatus("Saving session (post-promotion)…", kind: .progress)
            checkpoint?.checkpointSaveInFlight = true
            checkpoint?.startSlowSaveWatchdog(label: "session save (post-promotion)")
            // Fire-and-forget detached task. The closure captures
            // only Sendable value types (weight arrays, metadata
            // structs, the session state snapshot, and a few
            // strings) — explicitly NOT `self` — so we can safely
            // run past a session end without touching any View
            // @State. The completion handler hops back to the
            // MainActor to surface the outcome in the status row
            // and to reset `periodicSaveController`'s deadline;
            // that hop is safe because the handler only touches
            // the View's @State (no network or gate reads).
            Task.detached(priority: .utility) {
                [bufferForAutosave, chartSnapshotForAutosave] in
                let outcome: Result<(URL, String), Error>
                do {
                    let url = try await CheckpointManager.saveSession(
                        championWeights: championWeightsSnapshot,
                        championID: championID,
                        championMetadata: championMetadata,
                        championCreatedAtUnix: createdAtUnix,
                        trainerWeights: trainerWeightsSnapshot,
                        trainerID: trainerID,
                        trainerMetadata: trainerMetadata,
                        trainerCreatedAtUnix: createdAtUnix,
                        state: sessionState,
                        replayBuffer: bufferForAutosave,
                        chartSnapshot: chartSnapshotForAutosave,
                        trigger: "promote"
                    )
                    let bufStr: String
                    if let snap = bufferForAutosave?.stateSnapshot() {
                        bufStr = " replay=\(snap.storedCount)/\(snap.capacity)"
                    } else {
                        bufStr = ""
                    }
                    outcome = .success((url, bufStr))
                } catch {
                    outcome = .failure(error)
                }
                await MainActor.run {
                    switch outcome {
                    case .success(let (url, bufStr)):
                        self.checkpoint?.setCheckpointStatus(
                            "Saved \(url.lastPathComponent) (post-promotion)",
                            kind: .success
                        )
                        SessionLogger.shared.log(
                            "[CHECKPOINT] Saved session (post-promotion): \(url.lastPathComponent) build=\(BuildInfo.buildNumber) git=\(BuildInfo.gitHash)\(bufStr)"
                        )
                        self.checkpoint?.recordLastSessionPointer(
                            directoryURL: url,
                            sessionID: sessionState.sessionID,
                            trigger: "post-promotion"
                        )
                        self.periodicSaveController?.noteSuccessfulSave(at: Date())
                        self.checkpoint?.lastSavedAt = Date()
                    case .failure(let error):
                        self.checkpoint?.setCheckpointStatus(
                            "Autosave failed (post-promotion): \(error.localizedDescription)",
                            kind: .error
                        )
                        SessionLogger.shared.log(
                            "[CHECKPOINT] Saved session (post-promotion) failed: \(error.localizedDescription)"
                        )
                    }
                    self.checkpoint?.cancelSlowSaveWatchdog()
                    self.checkpoint?.checkpointSaveInFlight = false
                }
            }
        }
    }

    /// Emit an expanded multi-line summary of a just-finished arena
    /// tournament to stdout and the session log. Format:
    ///
    ///   [ARENA] #N Candidate vs Champion @ step S
    ///   [ARENA]     Games: G       (played/total)
    ///   [ARENA]     Result: W wins / D draws / L losses (candidate perspective)
    ///   [ARENA]     Score:  51.2% [48.1%, 54.3%]
    ///   [ARENA]     Elo diff: +28 [+13, +34]
    ///   [ARENA]     Draw rate: 17.5%
    ///   [ARENA]     By side:
    ///   [ARENA]       Candidate as white: 54.1%  (W/D/L n/n/n)
    ///   [ARENA]       Candidate as black: 48.3%  (W/D/L n/n/n)
    ///   [ARENA]     batch=… lr=… promote>=… sp.tau=… ar.tau=… workers=… build=…
    ///   [ARENA]     candidate=…  champion=…  trainer=…
    ///   [ARENA]     diversity: unique=…/… (…%) avgDiverge=…
    ///   [ARENA]     Verdict: PROMOTED(auto)=ID | kept    dur=m:ss
    ///   [ARENA] #N kv step=… games=… w=… d=… l=… score=… elo=… elo_lo=… elo_hi=… draw_rate=… cand_white_score=… cand_black_score=… promoted=… kind=… dur_sec=… build=… candidate=… champion=… trainer=…
    ///
    /// The trailing `kv` line stays a single key=value line so
    /// grep-based tooling can pull every arena outcome without
    /// reassembling the multi-line block.
    @MainActor
    private func logArenaResult(
        record: TournamentRecord,
        index: Int,
        trainer: ChessTrainer,
        candidate: ChessMPSNetwork,
        championSide: ChessMPSNetwork,
        diversity: GameDiversityTracker.Snapshot
    ) {
        let sp = samplingScheduleBox?.selfPlay ?? buildSelfPlaySchedule()
        let ar = samplingScheduleBox?.arena ?? buildArenaSchedule()
        let candidateIDStr = candidate.identifier?.description ?? "?"
        let championIDStr = championSide.identifier?.description ?? "?"
        let trainerIDStr = trainer.identifier?.description ?? "?"

        let parameters = ArenaLogFormatter.Parameters(
            batchSize: TrainingParameters.shared.trainingBatchSize,
            learningRate: trainer.learningRate,
            promoteThreshold: TrainingParameters.shared.arenaPromoteThreshold,
            tournamentGames: TrainingParameters.shared.arenaGamesPerTournament,
            spStartTau: sp.startTau,
            spFloorTau: sp.floorTau,
            spDecayPerPly: sp.decayPerPly,
            arStartTau: ar.startTau,
            arFloorTau: ar.floorTau,
            arDecayPerPly: ar.decayPerPly,
            workerCount: TrainingParameters.shared.selfPlayWorkers,
            buildNumber: BuildInfo.buildNumber
        )
        let diversityCtx = ArenaLogFormatter.Diversity(
            uniqueGames: diversity.uniqueGames,
            gamesInWindow: diversity.gamesInWindow,
            uniquePercent: diversity.uniquePercent,
            avgDivergencePly: diversity.avgDivergencePly
        )

        let lines = ArenaLogFormatter.formatHumanReadable(
            record: record,
            index: index,
            candidateID: candidateIDStr,
            championID: championIDStr,
            trainerID: trainerIDStr,
            parameters: parameters,
            diversity: diversityCtx
        )
        let kv = ArenaLogFormatter.formatKVLine(
            record: record,
            index: index,
            candidateID: candidateIDStr,
            championID: championIDStr,
            trainerID: trainerIDStr,
            buildNumber: BuildInfo.buildNumber
        )

        for line in lines {
            print(line)
            SessionLogger.shared.log(line)
        }
        print(kv)
        SessionLogger.shared.log(kv)

        // CLI-mode capture: mirror the arena block's parse-target
        // fields (KV line) plus the extras from the human-readable
        // block that don't appear in the KV but are useful for
        // downstream tooling (tau schedules, diversity snapshot,
        // per-worker count). The recorder serializes these into
        // the arena_results[] array of the `--output` JSON.
        if let recorder = cliRecorder {
            let elo = record.eloSummary
            let entry = CliTrainingRecorder.Arena(
                index: index,
                finishedAtStep: record.finishedAtStep,
                gamesPlayed: record.gamesPlayed,
                tournamentGames: TrainingParameters.shared.arenaGamesPerTournament,
                candidateWins: record.candidateWins,
                championWins: record.championWins,
                draws: record.draws,
                score: record.score,
                drawRate: ArenaLogFormatter.drawRateFraction(record: record),
                elo: elo.elo,
                eloLo: elo.eloLo,
                eloHi: elo.eloHi,
                scoreLo: elo.scoreLo,
                scoreHi: elo.scoreHi,
                candidateWinsAsWhite: record.candidateWinsAsWhite,
                candidateDrawsAsWhite: record.candidateDrawsAsWhite,
                candidateLossesAsWhite: record.candidateLossesAsWhite,
                candidateWinsAsBlack: record.candidateWinsAsBlack,
                candidateDrawsAsBlack: record.candidateDrawsAsBlack,
                candidateLossesAsBlack: record.candidateLossesAsBlack,
                candidateScoreAsWhite: record.candidateScoreAsWhite,
                candidateScoreAsBlack: record.candidateScoreAsBlack,
                promoted: record.promoted,
                promotionKind: record.promotionKind?.rawValue,
                promotedID: record.promotedID?.description,
                durationSec: record.durationSec,
                candidateID: candidateIDStr,
                championID: championIDStr,
                trainerID: trainerIDStr,
                learningRate: Double(trainer.learningRate),
                promoteThreshold: TrainingParameters.shared.arenaPromoteThreshold,
                batchSize: TrainingParameters.shared.trainingBatchSize,
                workerCount: TrainingParameters.shared.selfPlayWorkers,
                spStartTau: Double(sp.startTau),
                spFloorTau: Double(sp.floorTau),
                spDecayPerPly: Double(sp.decayPerPly),
                arStartTau: Double(ar.startTau),
                arFloorTau: Double(ar.floorTau),
                arDecayPerPly: Double(ar.decayPerPly),
                diversityUniqueGames: diversity.uniqueGames,
                diversityGamesInWindow: diversity.gamesInWindow,
                diversityUniquePercent: diversity.uniquePercent,
                diversityAvgDivergencePly: diversity.avgDivergencePly,
                buildNumber: BuildInfo.buildNumber
            )
            recorder.appendArena(entry)
        }
    }

    /// Release arena-active state on an early return from
    /// `runArenaParallel`. Clears the active flag (so the candidate
    /// probe resumes), clears the live-progress box and mirror (so
    /// the busy label reverts to normal Play and Train mode), and
    /// resets the UI's `isArenaRunning` mirror.
    @MainActor
    private func cleanupArenaState(arenaFlag: ArenaActiveFlag, tBox: TournamentLiveBox) {
        arenaFlag.clear()
        isArenaRunning = false
        tBox.clear()
        tournamentProgress = nil
        // Early-exit cleanup — make sure the live arena band on
        // the chart grid isn't left in an "arena active" state
        // after cancellation / error paths that skipped the
        // normal append-then-clear sequence. A no-op on the happy
        // path (the append site already cleared this).
        chartCoordinator?.cancelActiveArena()
        // Release any pending periodic-save fire that was held
        // back during the tournament. The controller decides on
        // the next `decide(now:)` call whether a (post-promotion)
        // successful save landed during the arena window (swallow
        // the pending fire) or not (dispatch the save a little
        // late).
        periodicSaveController?.noteArenaEnded()
    }

    // MARK: - Session-state snapshot (Stage 4m)

    nonisolated static let trainerLearningRateDefault: Float = 5e-5
    nonisolated static let entropyRegularizationCoeffDefault: Float = 1e-3

    /// Build the Codable snapshot of the current session state (counters,
    /// hyperparameters, arena history, replay-buffer footprint, build info).
    /// Called at save time with the live state read off the main actor by both
    /// the manual/periodic save path and `runArenaParallel`'s post-promotion
    /// save. Closes the active training segment at save time (and re-opens a
    /// fresh one if training is still in progress) so the on-disk cumulative
    /// wall-time totals stay correct across mid-training saves.
    @MainActor
    func buildCurrentSessionState(
        championID: String,
        trainerID: String
    ) -> SessionCheckpointState {
        let params = TrainingParameters.shared
        let wasTraining = realTraining
        checkpoint?.closeActiveTrainingSegment(reason: "save")
        if wasTraining && checkpoint?.activeSegmentStart == nil {
            checkpoint?.beginActiveTrainingSegment()
        }
        let now = Date()
        let sessionStart = checkpoint?.currentSessionStart ?? (parallelStats?.sessionStart ?? now)
        let elapsedSec = max(0, now.timeIntervalSince(sessionStart))
        let snap = parallelStats
        let trainingSnap = trainingStats
        let history = tournamentHistory.map { record in
            ArenaHistoryEntryCodable(
                finishedAtStep: record.finishedAtStep,
                candidateWins: record.candidateWins,
                championWins: record.championWins,
                draws: record.draws,
                score: record.score,
                promoted: record.promoted,
                promotedID: record.promotedID?.description,
                durationSec: record.durationSec,
                gamesPlayed: record.gamesPlayed,
                promotionKind: record.promotionKind?.rawValue,
                candidateWinsAsWhite: record.candidateWinsAsWhite,
                candidateWinsAsBlack: record.candidateWinsAsBlack,
                candidateLossesAsWhite: record.candidateLossesAsWhite,
                candidateLossesAsBlack: record.candidateLossesAsBlack,
                candidateDrawsAsWhite: record.candidateDrawsAsWhite,
                candidateDrawsAsBlack: record.candidateDrawsAsBlack,
                finishedAtUnix: record.finishedAt.map { Int64($0.timeIntervalSince1970) },
                candidateID: record.candidateID?.description,
                championID: record.championID?.description
            )
        }
        let lr = trainer?.learningRate ?? Self.trainerLearningRateDefault
        let entropyCoeff = trainer?.entropyRegularizationCoeff ?? Self.entropyRegularizationCoeffDefault
        let drawPen = trainer?.drawPenalty ?? Float(params.drawPenalty)
        let bufferSnap = replayBuffer?.stateSnapshot()
        let segments: [SessionCheckpointState.TrainingSegment]? =
            (checkpoint?.completedTrainingSegments.isEmpty ?? true)
            ? nil
            : checkpoint?.completedTrainingSegments
        return SessionCheckpointState(
            formatVersion: SessionCheckpointState.currentFormatVersion,
            sessionID: checkpoint?.currentSessionID ?? "unknown-session",
            savedAtUnix: Int64(now.timeIntervalSince1970),
            sessionStartUnix: Int64(sessionStart.timeIntervalSince1970),
            elapsedTrainingSec: elapsedSec,
            trainingSteps: trainingSnap?.steps ?? 0,
            selfPlayGames: snap?.selfPlayGames ?? 0,
            selfPlayMoves: snap?.selfPlayPositions ?? 0,
            trainingPositionsSeen: (trainingSnap?.steps ?? 0) * params.trainingBatchSize,
            batchSize: params.trainingBatchSize,
            learningRate: lr,
            entropyRegularizationCoeff: entropyCoeff,
            drawPenalty: drawPen,
            promoteThreshold: params.arenaPromoteThreshold,
            arenaGames: params.arenaGamesPerTournament,
            arenaConcurrency: params.arenaConcurrency,
            selfPlayTau: TauConfigCodable(samplingScheduleBox?.selfPlay ?? buildSelfPlaySchedule()),
            arenaTau: TauConfigCodable(samplingScheduleBox?.arena ?? buildArenaSchedule()),
            selfPlayWorkerCount: params.selfPlayWorkers,
            gradClipMaxNorm: Float(params.gradClipMaxNorm),
            weightDecayCoeff: Float(params.weightDecay),
            policyLossWeight: Float(params.policyLossWeight),
            valueLossWeight: Float(params.valueLossWeight),
            momentumCoeff: Float(params.momentumCoeff),
            illegalMassPenaltyWeight: Float(params.illegalMassWeight),
            policyLabelSmoothingEpsilon: Float(params.policyLabelSmoothingEpsilon),
            replayRatioTarget: params.replayRatioTarget,
            replayRatioAutoAdjust: params.replayRatioAutoAdjust,
            stepDelayMs: params.trainingStepDelayMs,
            selfPlayDelayMs: params.selfPlayDelayMs,
            lastAutoComputedDelayMs: lastAutoComputedDelayMs,
            // Schema-expansion fields (close the autotrain reproducibility gap
            // — these previously lived only in @AppStorage / @State and so
            // silently picked up the user's current preference on resume rather
            // than the session's saved value). All Optional for back-compat.
            lrWarmupSteps: params.lrWarmupSteps,
            sqrtBatchScalingForLR: params.sqrtBatchScalingLR,
            replayBufferMinPositionsBeforeTraining: params.replayBufferMinPositionsBeforeTraining,
            arenaAutoIntervalSec: params.arenaAutoIntervalSec,
            candidateProbeIntervalSec: params.candidateProbeIntervalSec,
            legalMassCollapseThreshold: params.legalMassCollapseThreshold,
            legalMassCollapseGraceSeconds: params.legalMassCollapseGraceSeconds,
            legalMassCollapseNoImprovementProbes: params.legalMassCollapseNoImprovementProbes,
            batchStatsInterval: params.batchStatsInterval,
            whiteCheckmates: snap?.whiteCheckmates,
            blackCheckmates: snap?.blackCheckmates,
            stalemates: snap?.stalemates,
            fiftyMoveDraws: snap?.fiftyMoveDraws,
            threefoldRepetitionDraws: snap?.threefoldRepetitionDraws,
            insufficientMaterialDraws: snap?.insufficientMaterialDraws,
            totalGameWallMs: snap?.totalGameWallMs,
            buildNumber: BuildInfo.buildNumber,
            buildGitHash: BuildInfo.gitHash,
            buildGitBranch: BuildInfo.gitBranch,
            buildDate: BuildInfo.buildDate,
            buildTimestamp: BuildInfo.buildTimestamp,
            buildGitDirty: BuildInfo.gitDirty,
            hasReplayBuffer: bufferSnap != nil,
            replayBufferStoredCount: bufferSnap?.storedCount,
            replayBufferCapacity: bufferSnap?.capacity,
            replayBufferTotalPositionsAdded: bufferSnap?.totalPositionsAdded,
            championID: championID,
            trainerID: trainerID,
            arenaHistory: history
        ).withTrainingSegments(segments)
    }

    // MARK: - Session resume (Stage 4k)

    /// Resume helper — reads `training_chart.json` / `progress_rate_chart.json`
    /// from the previously-loaded session and seeds `chartCoordinator` with the
    /// restored trajectory (plus the inline `arenaChartEvents` / `legalMassMaxAllTime`
    /// from `pendingLoadedSession.state`). Decode failures log and skip so a
    /// corrupt chart file never blocks the rest of the session-resume flow.
    /// Honors "View > Collect Chart Data" being off.
    func seedChartCoordinatorFromLoadedSession(
        chartURLs: (training: URL, progressRate: URL)
    ) {
        guard let chartCoordinator, chartCoordinator.collectionEnabled else {
            SessionLogger.shared.log(
                "[CHECKPOINT] Skipping chart-data restore — collection is disabled in View > Collect Chart Data"
            )
            return
        }
        let trainingSamples: [TrainingChartSample]
        let progressSamples: [ProgressRateSample]
        do {
            trainingSamples = try readChartFile(
                [TrainingChartSample].self, from: chartURLs.training
            )
            progressSamples = try readChartFile(
                [ProgressRateSample].self, from: chartURLs.progressRate
            )
        } catch {
            SessionLogger.shared.log(
                "[CHECKPOINT] Chart-data restore skipped — decode failed: \(error.localizedDescription)"
            )
            return
        }
        let arenaEvents = pendingLoadedSession?.state.arenaChartEvents ?? []
        let legalMassMax = pendingLoadedSession?.state.legalMassMaxAllTime ?? 0
        let lastTrainElapsed = trainingSamples.last?.elapsedSec ?? 0
        let lastProgressElapsed = progressSamples.last?.elapsedSec ?? 0
        let lastElapsed = max(lastTrainElapsed, lastProgressElapsed)
        let snapshot = ChartCoordinatorSnapshot(
            trainingSamples: trainingSamples,
            progressRateSamples: progressSamples,
            arenaChartEvents: arenaEvents,
            legalMassMaxAllTime: legalMassMax,
            lastElapsedSec: lastElapsed
        )
        chartCoordinator.seedFromRestoredSession(snapshot)
        SessionLogger.shared.log(
            "[CHECKPOINT] Restored chart data: \(trainingSamples.count) training samples, \(progressSamples.count) progress-rate samples, \(arenaEvents.count) arena events"
        )
    }

    // MARK: - Tournament / arena display state (Stage 4j)

    /// Live tournament progress mirrored from `tournamentBox` by the heartbeat.
    /// Non-nil while an arena is running, `nil` otherwise.
    var tournamentProgress: TournamentProgress?

    /// Lock-protected box the arena driver task writes into after each game
    /// completes; the heartbeat polls it into `tournamentProgress`. `nil`
    /// outside an arena.
    var tournamentBox: TournamentLiveBox?

    /// All completed tournaments this session, appended after each arena
    /// finishes. In-memory only (disk persistence deferred).
    var tournamentHistory: [TournamentRecord] = []

    /// Status-bar "Score" cell display-mode toggle: `false` = percentage
    /// (`"51.2%"`), `true` = Elo-with-CI (`"+28 [+13, +34]"`). Session-local.
    var scoreStatusShowElo: Bool = false

    // MARK: - Pending loaded session (Stage 4j)

    /// A parsed `.dcmsession` loaded from disk but not yet applied. The user
    /// loads a session while Play-and-Train is stopped; the next
    /// `startRealTraining` consumes this and seeds the trainer / counters / IDs
    /// from it, then clears it.
    var pendingLoadedSession: LoadedSession?

    // MARK: - Candidate-test probe state + CLI recorder (Stage 4h)

    /// Which board the Play-and-Train view shows: `.gameRun` (the live self-play
    /// game) or `.candidateTest` (the editable forward-pass board the user
    /// watches evolve as the weights update).
    var playAndTrainBoardMode: PlayAndTrainBoardMode = .gameRun

    /// Which network the candidate-test probe runs against: `.candidate` syncs
    /// the trainer's latest weights into the dedicated probe inference network,
    /// `.champion` probes the frozen champion directly (a stable reference for
    /// confirming whether the value head is moving or stuck at init saturation).
    var probeNetworkTarget: ProbeNetworkTarget = .candidate

    /// Set when the user edits the candidate-test board (drag, side-to-move
    /// toggle, Board-picker flip) while Play-and-Train is running. The driver
    /// task checks this at natural gap points and fires a forward-pass probe.
    var candidateProbeDirty: Bool = false

    /// Wall-clock of the last candidate-test probe — combined with
    /// `candidateProbeIntervalSec` to enforce the probe cadence.
    var lastCandidateProbeTime: Date = .distantPast

    /// Number of candidate-test probes that have actually fired this session
    /// — surfaced in the training stats text so the user can confirm probes
    /// are running even when the visible arrows barely change.
    var candidateProbeCount: Int = 0

    /// Live recorder for `--output` runs. Allocated at the start of
    /// `startRealTraining` when `cliOutputURL` is set, appended to by the
    /// stats/arena/probe paths, `nil` in normal interactive runs.
    var cliRecorder: CliTrainingRecorder?

    /// Whether the Play-and-Train view should show the candidate-test forward-
    /// pass board instead of the live game. True when training is active AND
    /// (the persisted mode is `.candidateTest` OR there are >1 self-play
    /// workers — in which case the live-game board is a hidden placeholder and
    /// the picker is unavailable, so the candidate-test board is the only
    /// useful left-side output).
    var isCandidateTestActive: Bool {
        guard realTraining else { return false }
        if TrainingParameters.shared.selfPlayWorkers > 1 { return true }
        return playAndTrainBoardMode == .candidateTest
    }

    // MARK: - Candidate-test probe execution + forward-pass inference (Stage 4i)

    /// Returns the board state the candidate-test probe should evaluate. Wired
    /// to `UpperContentView`'s `editableState` (the free-placement forward-pass
    /// board) in `handleBodyOnAppear` — that state stays on the view because
    /// the board-editing UI mutates it directly.
    var editableStateProvider: () -> GameState = { .starting }

    /// Publishes a finished forward-pass result to the on-board display. Wired
    /// to `{ inferenceResult = $0 }` on the view.
    var onInferenceResult: (EvaluationResult) -> Void = { _ in }

    /// Fire a candidate-test forward-pass probe if one is due (board edited
    /// since last probe, or the cadence interval elapsed) and the preconditions
    /// hold (candidate-test active, probe runner + network built, trainer up).
    /// Called from the Play-and-Train driver's trainer loop at natural gap
    /// points. The probe runs on a detached task so it never stalls the main
    /// actor; on the `.candidate` path it first snapshots the trainer's current
    /// weights into the dedicated probe inference network (so the probe can run
    /// concurrently with an active arena, which reads `candidateInferenceNetwork`,
    /// a different object). On the `.champion` path it reads the frozen champion
    /// directly, skipping if an arena is running (the promotion step briefly
    /// writes into the champion under a self-play pause and the probe would race
    /// that write).
    func fireCandidateProbeIfNeeded() async {
        guard
            isCandidateTestActive,
            let trainer,
            let probeRunner = probeRunner,
            let probeInference = probeInferenceNetwork,
            let championRunner = runner
        else { return }
        let now = Date()
        let dirty = candidateProbeDirty
        let intervalElapsed = now.timeIntervalSince(lastCandidateProbeTime)
            >= TrainingParameters.shared.candidateProbeIntervalSec
        guard dirty || intervalElapsed else { return }

        let state = editableStateProvider()
        let target = probeNetworkTarget
        let result: EvaluationResult
        do {
            switch target {
            case .candidate:
                // Snapshot the trainer's current state into the probe inference
                // network, then immediately run the probe. Doing the ~11.6 MB
                // trainer → probe copy here — not after every training block —
                // means it happens only when the probe is actually about to
                // fire. The probe network is dedicated; the only potentially
                // concurrent op is `trainer.network.exportWeights` during an
                // arena's own trainer-snapshot step — both reads under the
                // network's internal lock, safe. Detached so MainActor isn't
                // stalled.
                result = try await Task.detached(priority: .userInitiated) {
                    let weights = try await trainer.network.exportWeights()
                    try await probeInference.loadWeights(weights)
                    return await Self.performInference(with: probeRunner, state: state)
                }.value
                // Transient read-only snapshot — inherit the trainer's ID
                // rather than minting (arena snapshots do mint; see runArenaParallel).
                probeInference.identifier = trainer.identifier
            case .champion:
                if arenaActiveFlag?.isActive == true { return }
                result = await Task.detached(priority: .userInitiated) {
                    await Self.performInference(with: championRunner, state: state)
                }.value
            }
        } catch {
            // Leave probe state unchanged so the previous result stays on
            // screen; the next gap-point call retries.
            return
        }
        onInferenceResult(result)
        candidateProbeDirty = false
        lastCandidateProbeTime = Date()
        candidateProbeCount += 1
        // CLI-mode capture: append this probe's diagnostics if an output JSON
        // is configured. No-op in interactive runs; skipped on a failed pass.
        if let recorder = cliRecorder,
           let inf = result.rawInference,
           let sessionStart = checkpoint?.currentSessionStart {
            let elapsed = Date().timeIntervalSince(sessionStart)
            let event = buildCliCandidateTestEvent(
                elapsedSec: elapsed,
                probeIndex: candidateProbeCount,
                target: target,
                state: state,
                inference: inf
            )
            recorder.appendCandidateTest(event)
        }
    }

    /// Build a `CliTrainingRecorder.CandidateTest` from a finished forward-pass
    /// result — the same on-screen policy diagnostics (top-100 sum,
    /// above-uniform count, legal-mass sum, min/max) plus a structured top-10,
    /// mirroring `performInference` so the JSON and the UI stay in sync.
    nonisolated private func buildCliCandidateTestEvent(
        elapsedSec: Double,
        probeIndex: Int,
        target: ProbeNetworkTarget,
        state: GameState,
        inference: ChessRunner.InferenceResult
    ) -> CliTrainingRecorder.CandidateTest {
        let policy = inference.policy
        let sum = Double(policy.reduce(0, +))
        let top100Sum = Double(policy.sorted(by: >).prefix(100).reduce(0, +))
        let minP = Double(policy.min() ?? 0)
        let maxP = Double(policy.max() ?? 0)
        let legalMoves = MoveGenerator.legalMoves(for: state)
        let nLegal = max(1, legalMoves.count)
        let legalUniformThreshold = 1.0 / Double(nLegal)
        let legalIndices = legalMoves
            .map { PolicyEncoding.policyIndex($0, currentPlayer: state.currentPlayer) }
        let abovePerLegalCount = legalIndices.filter { idx in
            idx >= 0 && idx < policy.count
                && Double(policy[idx]) > legalUniformThreshold
        }.count
        let legalMassSum: Double = legalIndices.reduce(0.0) { acc, idx in
            (idx >= 0 && idx < policy.count) ? acc + Double(policy[idx]) : acc
        }
        let top10 = ChessRunner.extractTopMoves(
            from: policy,
            state: state,
            pieces: state.board,
            count: 10
        )
        let topMovesOut: [CliTrainingRecorder.CandidateTest.TopMove] = top10.enumerated().map { (rank, mv) in
            CliTrainingRecorder.CandidateTest.TopMove(
                rank: rank + 1,
                from: BoardEncoder.squareName(mv.fromRow * 8 + mv.fromCol),
                to: BoardEncoder.squareName(mv.toRow * 8 + mv.toCol),
                fromRow: mv.fromRow,
                fromCol: mv.fromCol,
                toRow: mv.toRow,
                toCol: mv.toCol,
                probability: Double(mv.probability),
                isLegal: mv.isLegal
            )
        }
        let stats = CliTrainingRecorder.CandidateTest.PolicyStats(
            sum: sum,
            top100Sum: top100Sum,
            aboveUniformCount: abovePerLegalCount,
            legalMoveCount: legalMoves.count,
            legalUniformThreshold: legalUniformThreshold,
            legalMassSum: legalMassSum,
            illegalMassSum: max(0.0, 1.0 - legalMassSum),
            min: minP,
            max: maxP
        )
        let targetStr: String
        switch target {
        case .candidate: targetStr = "candidate"
        case .champion: targetStr = "champion"
        }
        return CliTrainingRecorder.CandidateTest(
            elapsedSec: elapsedSec,
            probeIndex: probeIndex,
            probeNetworkTarget: targetStr,
            inferenceTimeMs: inference.inferenceTimeMs,
            valueHead: CliTrainingRecorder.CandidateTest.ValueHead(output: Double(inference.value)),
            policyHead: CliTrainingRecorder.CandidateTest.PolicyHead(
                policyStats: stats,
                topRaw: topMovesOut
            )
        )
    }

    /// Run a single forward pass through `runner` for `state` and assemble the
    /// `EvaluationResult` (top moves, the formatted text panel, the input
    /// tensor, the raw inference). `nonisolated` — called from detached tasks
    /// by `fireCandidateProbeIfNeeded` and by `UpperContentView`'s Run Forward
    /// Pass.
    nonisolated static func performInference(
        with runner: ChessRunner,
        state: GameState
    ) async -> EvaluationResult {
        var lines: [String] = []
        var topMoves: [MoveVisualization] = []
        var rawInference: ChessRunner.InferenceResult? = nil
        let board = BoardEncoder.encode(state)

        do {
            let inference = try await runner.evaluate(board: board, state: state, pieces: state.board)
            topMoves = inference.topMoves
            rawInference = inference

            lines.append(String(format: "Forward pass: %.2f ms", inference.inferenceTimeMs))
            lines.append("")
            lines.append("Value Head")
            lines.append(String(format: "  Output: %+.6f", inference.value))
            // Removed the (v+1)/2 → "X% win / Y% loss" line. With a single
            // tanh scalar (no WDL output) and a non-zero draw penalty in
            // training, that mapping was misleading. Just show the raw value.
            lines.append("")
            lines.append("Policy Head (Top 4 raw — includes illegal)")
            // The list deliberately includes illegal candidates so we can see
            // whether the network has learned move-validity.
            for (rank, move) in inference.topMoves.enumerated() {
                let fromName = BoardEncoder.squareName(move.fromRow * 8 + move.fromCol)
                let toName = BoardEncoder.squareName(move.toRow * 8 + move.toCol)
                let promoSuffix: String
                switch move.promotion {
                case .queen:  promoSuffix = "=Q"
                case .rook:   promoSuffix = "=R"
                case .bishop: promoSuffix = "=B"
                case .knight: promoSuffix = "=N"
                default:      promoSuffix = ""
                }
                let rankCol = String(rank + 1).padding(toLength: 4, withPad: " ", startingAt: 0)
                let moveCol = "\(fromName)-\(toName)\(promoSuffix)".padding(toLength: 10, withPad: " ", startingAt: 0)
                let legalMark = move.isLegal ? "" : "  (illegal)"
                lines.append("  \(rankCol)\(moveCol)\(String(format: "%.6f%%", move.probability * 100))\(legalMark)")
            }
            // Sum of the top-100 move probabilities — a cheap scalar that
            // changes visibly between probes even when the top-4 ordering is stable.
            let top100Sum = inference.policy.sorted(by: >).prefix(100).reduce(0, +)
            lines.append(String(format: "  Top 100 sum: %.6f%%", top100Sum * 100))
            lines.append("")
            lines.append("Policy Stats")
            lines.append(String(format: "  Sum: %.8f", inference.policy.reduce(0, +)))
            // Legality-aware "above-uniform" count for THIS position: how many
            // legal moves the network gives mass above `1 / N_legal`.
            let legalMoves = MoveGenerator.legalMoves(for: state)
            let nLegal = max(1, legalMoves.count)
            let legalUniformThreshold = 1.0 / Float(nLegal)
            let abovePerLegalCount = legalMoves
                .map { PolicyEncoding.policyIndex($0, currentPlayer: state.currentPlayer) }
                .filter { idx in
                    idx >= 0 && idx < inference.policy.count
                        && inference.policy[idx] > legalUniformThreshold
                }
                .count
            lines.append(String(
                format: "  Legal moves above uniform (%.3f%%): %d / %d  (threshold = 1/legalCount = 1/%d)",
                Double(legalUniformThreshold) * 100,
                abovePerLegalCount, nLegal, nLegal
            ))
            // Total mass on legal moves vs illegal — at convergence,
            // mass-on-illegal should approach zero.
            let legalMassSum = legalMoves
                .map { PolicyEncoding.policyIndex($0, currentPlayer: state.currentPlayer) }
                .reduce(Float(0)) { acc, idx in
                    (idx >= 0 && idx < inference.policy.count) ? acc + inference.policy[idx] : acc
                }
            lines.append(String(format: "  Legal mass sum: %.6f%%   (illegal = %.6f%%)",
                                Double(legalMassSum) * 100,
                                Double(1 - legalMassSum) * 100))
            if let maxProb = inference.policy.max(), let minProb = inference.policy.min() {
                lines.append(String(format: "  Min: %.8f", minProb))
                lines.append(String(format: "  Max: %.8f", maxProb))
            }
        } catch {
            lines.append("Error: \(error.localizedDescription)")
        }

        return EvaluationResult(
            topMoves: topMoves,
            textOutput: lines.joined(separator: "\n"),
            inputTensor: board,
            rawInference: rawInference
        )
    }

    // MARK: - Demo training (Train Once / Continuous Training) (Stage 4g)

    /// Batch size for the demo "Train Once" / "Continuous Training" buttons
    /// (training on random data, not self-play). Distinct from
    /// `TrainingParameters.shared.trainingBatchSize` which the Play-and-Train
    /// loop uses.
    nonisolated static let trainingBatchSize = 4096

    /// Rolling-window size for the live-loss averages in `TrainingLiveStatsBox`.
    nonisolated static let rollingLossWindow = 512

    /// Resets the on-board demo display (forward-pass inference result + the
    /// live game watcher). Wired to `UpperContentView` in `handleBodyOnAppear`
    /// — `gameWatcher` must stay `@State` on the view (SwiftUI reconstruction
    /// semantics), so this controller reaches it through the closure.
    var onResetBoardDisplay: () -> Void = { }

    /// Demo "Train Once": one SGD step on random data, then mirror the result
    /// into `trainingStats` / `lastTrainStep`.
    func trainOnce() {
        SessionLogger.shared.log("[BUTTON] Train Once")
        guard let trainer = ensureTrainer() else { return }
        // Switching modes — clear any stale game/inference output and start a
        // fresh stats run (single-step still uses TrainingRunStats so the
        // formatter has one path to render).
        onResetBoardDisplay()
        onClearTrainingDisplay()
        isTrainingOnce = true

        Task { [trainer] in
            let result = await Self.runOneTrainStep(trainer: trainer)
            await MainActor.run {
                switch result {
                case .success(let timing):
                    var stats = TrainingRunStats()
                    stats.record(timing)
                    trainingStats = stats
                    lastTrainStep = timing
                case .failure(let error):
                    trainingError = error.localizedDescription
                }
                isTrainingOnce = false
            }
        }
    }

    /// Demo "Continuous Training": a tight SGD loop on random data until Stop.
    func startContinuousTraining() {
        SessionLogger.shared.log("[BUTTON] Train Continuous")
        guard let trainer = ensureTrainer() else { return }
        onResetBoardDisplay()
        onClearTrainingDisplay()

        // Seed trainingStats with a fresh zero so the formatter shows "Steps
        // done: 0" immediately; the heartbeat poller replaces it with the real
        // stats as soon as the first step lands.
        let box = TrainingLiveStatsBox(rollingWindow: Self.rollingLossWindow)
        trainingBox = box
        trainingStats = TrainingRunStats()
        continuousTraining = true

        trainingTask = Task { [trainer, box] in
            var shouldStop = false
            while !Task.isCancelled && !shouldStop {
                let result = await Self.runOneTrainStep(trainer: trainer)
                switch result {
                case .success(let timing):
                    box.recordStep(timing)
                case .failure(let error):
                    box.recordError(error.localizedDescription)
                    shouldStop = true
                }
            }
            await MainActor.run { continuousTraining = false }
        }
    }

    /// Cancel the demo continuous-training task.
    func stopContinuousTraining() {
        trainingTask?.cancel()
        trainingTask = nil
    }

    nonisolated private static func runOneTrainStep(trainer: ChessTrainer) async -> Result<TrainStepTiming, Error> {
        do {
            return .success(try await trainer.trainStep(batchSize: trainingBatchSize))
        } catch {
            return .failure(error)
        }
    }

    // MARK: - Trainer build / config + sampling schedules (Stage 4e)

    /// Ensure the trainer exists, (re)applying all live `TrainingParameters`
    /// hyperparameters to it. Returns `nil` (and sets `trainingError`) if the
    /// trainer's MPSGraph build fails on first construction.
    func ensureTrainer() -> ChessTrainer? {
        let params = TrainingParameters.shared
        if let trainer {
            trainer.learningRate = Float(params.learningRate)
            trainer.entropyRegularizationCoeff = Float(params.entropyBonus)
            trainer.drawPenalty = Float(params.drawPenalty)
            trainer.weightDecayC = Float(params.weightDecay)
            trainer.gradClipMaxNorm = Float(params.gradClipMaxNorm)
            trainer.policyLossWeight = Float(params.policyLossWeight)
            trainer.valueLossWeight = Float(params.valueLossWeight)
            trainer.illegalMassPenaltyWeight = Float(params.illegalMassWeight)
            trainer.policyLabelSmoothingEpsilon = Float(params.policyLabelSmoothingEpsilon)
            trainer.momentumCoeff = Float(params.momentumCoeff)
            trainer.sqrtBatchScalingForLR = params.sqrtBatchScalingLR
            trainer.lrWarmupSteps = params.lrWarmupSteps
            trainer.batchStatsInterval = params.batchStatsInterval
            return trainer
        }
        do {
            let t = try ChessTrainer(
                learningRate: Float(params.learningRate),
                entropyRegularizationCoeff: Float(params.entropyBonus),
                drawPenalty: Float(params.drawPenalty),
                weightDecayC: Float(params.weightDecay),
                gradClipMaxNorm: Float(params.gradClipMaxNorm),
                policyLossWeight: Float(params.policyLossWeight),
                valueLossWeight: Float(params.valueLossWeight),
                illegalMassPenaltyWeight: Float(params.illegalMassWeight),
                policyLabelSmoothingEpsilon: Float(params.policyLabelSmoothingEpsilon),
                momentumCoeff: Float(params.momentumCoeff),
                sqrtBatchScalingForLR: params.sqrtBatchScalingLR,
                lrWarmupSteps: params.lrWarmupSteps
            )
            trainer = t
            return t
        } catch {
            trainingError = "Trainer init failed: \(error.localizedDescription)"
            return nil
        }
    }

    /// Build a `SamplingSchedule` for self-play from the live tau parameters.
    /// Dirichlet noise matches the `.selfPlay` preset (AlphaZero noise) — not
    /// exposed in the UI; only the temperature schedule is editable.
    func buildSelfPlaySchedule() -> SamplingSchedule {
        let params = TrainingParameters.shared
        return SamplingSchedule(
            startTau: Float(max(0.01, params.selfPlayStartTau)),
            decayPerPly: Float(max(0.0, params.selfPlayTauDecayPerPly)),
            floorTau: Float(max(0.01, params.selfPlayTargetTau)),
            dirichletNoise: SamplingSchedule.selfPlay.dirichletNoise
        )
    }

    /// Build a `SamplingSchedule` for arena play from the live tau parameters.
    /// Arena never applies Dirichlet noise (pure strength measurement).
    func buildArenaSchedule() -> SamplingSchedule {
        let params = TrainingParameters.shared
        return SamplingSchedule(
            startTau: Float(max(0.01, params.arenaStartTau)),
            decayPerPly: Float(max(0.0, params.arenaTauDecayPerPly)),
            floorTau: Float(max(0.01, params.arenaTargetTau))
        )
    }

    // MARK: - Build

    /// File > Build Network. Belt-and-suspenders guards mirror the menu's
    /// disable conditions for keyboard-shortcut / URL-scheme invocations under
    /// a race. On success, mints a fresh `ModelID`, wires the new network +
    /// runner, fills `networkStatus`, and clears the last-saved-at marker.
    func buildNetwork() {
        SessionLogger.shared.log("[BUTTON] Build Network")
        if isBusyProvider() {
            onRefuseMenuAction(busyReasonProvider())
            return
        }
        if networkReady {
            onRefuseMenuAction("A network is already built. Load Model or Load Session to replace its weights.")
            return
        }
        isBuilding = true
        networkStatus = ""
        // Drop the trainer (it owns graph state we're about to invalidate by
        // rebuilding) and wipe all training/sweep display state.
        onDropTrainer()
        onClearTrainingDisplay()

        Task {
            let result = await Task.detached(priority: .userInitiated) {
                Self.performBuild()
            }.value

            switch result {
            case .success(let net):
                net.identifier = ModelIDMinter.mint()
                network = net
                runner = ChessRunner(network: net)
                let idStr = net.identifier?.description ?? "?"
                networkStatus = """
                    Network built in \(String(format: "%.1f", net.buildTimeMs)) ms
                    ID: \(idStr)
                    Parameters: ~2,400,000 (~2.4M)
                    Architecture: 20x8x8 -> stem(128)
                      -> 8 res+SE blocks -> policy(4864) + value(1)
                    """
                checkpoint?.lastSavedAt = nil
            case .failure(let error):
                network = nil
                runner = nil
                networkStatus = "Build failed: \(error.localizedDescription)"
            }
            isBuilding = false
        }
    }

    /// Ensure `network` exists. If it's already built, returns it. Otherwise
    /// runs the same detached `performBuild()` path the menu's Build button
    /// uses and wires the result in. Used by the load paths so the user doesn't
    /// have to press Build first when the weights are about to be overwritten.
    func ensureChampionBuilt() async -> Result<ChessMPSNetwork, Error> {
        if let champion = network {
            return .success(champion)
        }
        isBuilding = true
        networkStatus = ""
        onDropTrainer()
        onClearTrainingDisplay()
        SessionLogger.shared.log("[BUILD] Auto-build before load")
        let result = await Task.detached(priority: .userInitiated) {
            Self.performBuild()
        }.value
        switch result {
        case .success(let net):
            net.identifier = ModelIDMinter.mint()
            network = net
            runner = ChessRunner(network: net)
            isBuilding = false
            return .success(net)
        case .failure(let error):
            network = nil
            runner = nil
            networkStatus = "Build failed: \(error.localizedDescription)"
            isBuilding = false
            return .failure(error)
        }
    }

    /// The actual network construction. Runs on a detached `.userInitiated`
    /// task at the call sites (MPSGraph build is long synchronous work), so
    /// this is `nonisolated`.
    nonisolated static func performBuild() -> Result<ChessMPSNetwork, Error> {
        Result { try ChessMPSNetwork(.randomWeights) }
    }
}
