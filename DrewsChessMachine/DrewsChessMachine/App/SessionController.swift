import SwiftUI

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

    /// Clears the view's "champion was replaced (Load Model) since the last
    /// training run" flag — `startRealTraining` does this after a successful
    /// start. Wired to `{ championLoadedSinceLastTrainingSegment = false }`.
    var onClearChampionLoadedFlag: () -> Void = { }

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

    // MARK: - Real (self-play) training run (Stage 4o)

    /// Kick off real-data training in parallel mode: self-play, training,
    /// and arena coordination run as three independent tasks inside a
    /// `TaskGroup`, sharing state only through the lock-protected
    /// replay buffer, stats boxes, pause gates, and arena-trigger box.
    /// Self-play plays one game at a time on the champion network and
    /// streams labeled positions into the replay buffer. Training runs
    /// a tight-loop SGD on the trainer network, sampling the buffer
    /// for each batch. The arena coordinator sleeps until triggered
    /// (either by the 30-minute auto-fire or the Run Arena button),
    /// then runs 200 games between the candidate inference network
    /// and a fourth "arena champion" network — both snapshots taken
    /// under brief per-worker pauses so game play and training never
    /// actually stop, even during a tournament.
    ///
    /// `mode` picks how in-memory state from a prior Stop is handled:
    /// see `TrainingStartMode` for the four cases.
    func startRealTraining(mode: TrainingStartMode = .freshOrFromLoadedSession) {
        SessionLogger.shared.log("[BUTTON] Play and Train")
        // Begin a new training segment for cumulative wall-time
        // tracking. Closed via `closeActiveTrainingSegment` on Stop or
        // at save time. Don't try to open one if the previous Stop
        // didn't actually close (defensive — `closeActiveTrainingSegment`
        // is idempotent on nil but we want the log line to be clean).
        if checkpoint?.activeSegmentStart != nil {
            checkpoint?.closeActiveTrainingSegment(reason: "restart-without-stop")
        }
        checkpoint?.beginActiveTrainingSegment()
        precondition(
            UpperContentView.absoluteMaxSelfPlayWorkers >= 1,
            "absoluteMaxSelfPlayWorkers must be >= 1; got \(UpperContentView.absoluteMaxSelfPlayWorkers)"
        )
        // Snap the live N into the [1, absoluteMaxSelfPlayWorkers] range
        // before doing anything else. The Stepper enforces this
        // for user input but `TrainingParameters.shared.selfPlayWorkers` is centrally managed
        // so the value could in principle be edited elsewhere.
        let initialWorkerCount = max(1, min(UpperContentView.absoluteMaxSelfPlayWorkers, TrainingParameters.shared.selfPlayWorkers))
        if initialWorkerCount != TrainingParameters.shared.selfPlayWorkers {
            TrainingParameters.shared.selfPlayWorkers = initialWorkerCount
        }
        guard let trainer = ensureTrainer(), let network else { return }
        let gameWatcher = gameWatcherProvider()
        onResetBoardDisplay()
        trainingAlarm?.clear()
        trainingAlarm?.resetStreaks()

        // `continueMode` controls whether we preserve in-memory state
        // from a prior Stop. When true: reuse replay buffer, stats
        // boxes, session ID, tournament history, chart samples, and
        // trainer weights. The only fresh objects are the transient
        // per-task gates and live schedule boxes (those get cancelled
        // with the task on Stop, so they don't survive).
        let continueMode = (mode == .continueAfterStop)
        if !continueMode {
            onClearTrainingDisplay()
        }

        let buffer: ReplayBuffer
        if continueMode, let existing = replayBuffer {
            buffer = existing
        } else {
            buffer = ReplayBuffer(capacity: TrainingParameters.shared.replayBufferCapacity)
            replayBuffer = buffer
        }
        let box: TrainingLiveStatsBox
        if continueMode, let existing = trainingBox {
            box = existing
        } else {
            let fresh = TrainingLiveStatsBox(rollingWindow: SessionController.rollingLossWindow)
            if let rs = pendingLoadedSession?.state {
                var seeded = TrainingRunStats()
                seeded.steps = rs.trainingSteps
                fresh.seed(seeded)
            }
            trainingBox = fresh
            realRollingPolicyLoss = nil
            realRollingValueLoss = nil
            box = fresh
        }
        // Seed counters from the loaded session if resuming, or
        // start fresh. This covers the stats box (game/move/result
        // counters), training step count, tournament history, and
        // worker count. In `.continueAfterStop` the trainer's
        // hyperparameters and `trainingStats` stay untouched — the
        // user wants to pick up exactly where they left off.
        let resumeState = pendingLoadedSession?.state
        if !continueMode {
            if let rs = resumeState {
                SessionLogger.shared.log(
                    "[RESUME-PARAM] learning_rate: \(TrainingParameters.shared.learningRate) -> \(rs.learningRate) (from session)"
                )
                trainer.learningRate = rs.learningRate
                TrainingParameters.shared.learningRate = Double(rs.learningRate)
                if let entropyCoeff = rs.entropyRegularizationCoeff {
                    SessionLogger.shared.log(
                        "[RESUME-PARAM] entropy_bonus: \(TrainingParameters.shared.entropyBonus) -> \(entropyCoeff) (from session)"
                    )
                    trainer.entropyRegularizationCoeff = entropyCoeff
                    TrainingParameters.shared.entropyBonus = Double(entropyCoeff)
                } else {
                    trainer.entropyRegularizationCoeff = Float(TrainingParameters.shared.entropyBonus)
                    SessionLogger.shared.log(
                        "[RESUME-PARAM] entropy_bonus: saved=nil applied=\(TrainingParameters.shared.entropyBonus) (defaulted)"
                    )
                }
                if let dp = rs.drawPenalty {
                    SessionLogger.shared.log(
                        "[RESUME-PARAM] draw_penalty: \(TrainingParameters.shared.drawPenalty) -> \(dp) (from session)"
                    )
                    trainer.drawPenalty = dp
                    TrainingParameters.shared.drawPenalty = Double(dp)
                } else {
                    trainer.drawPenalty = Float(TrainingParameters.shared.drawPenalty)
                    SessionLogger.shared.log(
                        "[RESUME-PARAM] draw_penalty: saved=nil applied=\(TrainingParameters.shared.drawPenalty) (defaulted)"
                    )
                }
                // Regularization knobs that became editable post-v1 session
                // files: hydrate when present, otherwise leave the current
                // @AppStorage-backed value alone but log the fallthrough
                // so a session saved before this field existed never
                // resumes silently under different defaults.
                if let wd = rs.weightDecayCoeff {
                    SessionLogger.shared.log(
                        "[RESUME-PARAM] weight_decay: \(TrainingParameters.shared.weightDecay) -> \(wd) (from session)"
                    )
                    trainer.weightDecayC = wd
                    TrainingParameters.shared.weightDecay = Double(wd)
                } else {
                    trainer.weightDecayC = Float(TrainingParameters.shared.weightDecay)
                    SessionLogger.shared.log(
                        "[RESUME-PARAM] weight_decay: saved=nil applied=\(TrainingParameters.shared.weightDecay) (defaulted)"
                    )
                }
                if let clip = rs.gradClipMaxNorm {
                    SessionLogger.shared.log(
                        "[RESUME-PARAM] grad_clip_max_norm: \(TrainingParameters.shared.gradClipMaxNorm) -> \(clip) (from session)"
                    )
                    trainer.gradClipMaxNorm = clip
                    TrainingParameters.shared.gradClipMaxNorm = Double(clip)
                } else {
                    trainer.gradClipMaxNorm = Float(TrainingParameters.shared.gradClipMaxNorm)
                    SessionLogger.shared.log(
                        "[RESUME-PARAM] grad_clip_max_norm: saved=nil applied=\(TrainingParameters.shared.gradClipMaxNorm) (defaulted)"
                    )
                }
                if let plw = rs.policyLossWeight {
                    SessionLogger.shared.log(
                        "[RESUME-PARAM] policy_loss_weight: \(TrainingParameters.shared.policyLossWeight) -> \(plw) (from session)"
                    )
                    trainer.policyLossWeight = plw
                    TrainingParameters.shared.policyLossWeight = Double(plw)
                } else {
                    trainer.policyLossWeight = Float(TrainingParameters.shared.policyLossWeight)
                    SessionLogger.shared.log(
                        "[RESUME-PARAM] policy_loss_weight: saved=nil applied=\(TrainingParameters.shared.policyLossWeight) (defaulted)"
                    )
                }
                if let vlw = rs.valueLossWeight {
                    SessionLogger.shared.log(
                        "[RESUME-PARAM] value_loss_weight: \(TrainingParameters.shared.valueLossWeight) -> \(vlw) (from session)"
                    )
                    trainer.valueLossWeight = vlw
                    TrainingParameters.shared.valueLossWeight = Double(vlw)
                } else {
                    trainer.valueLossWeight = Float(TrainingParameters.shared.valueLossWeight)
                    SessionLogger.shared.log(
                        "[RESUME-PARAM] value_loss_weight: saved=nil applied=\(TrainingParameters.shared.valueLossWeight) (defaulted)"
                    )
                }
                if let mu = rs.momentumCoeff {
                    SessionLogger.shared.log(
                        "[RESUME-PARAM] momentum_coeff: \(TrainingParameters.shared.momentumCoeff) -> \(mu) (from session)"
                    )
                    trainer.momentumCoeff = mu
                    TrainingParameters.shared.momentumCoeff = Double(mu)
                } else {
                    trainer.momentumCoeff = Float(TrainingParameters.shared.momentumCoeff)
                    SessionLogger.shared.log(
                        "[RESUME-PARAM] momentum_coeff: saved=nil applied=\(TrainingParameters.shared.momentumCoeff) (defaulted)"
                    )
                }
                if let imw = rs.illegalMassPenaltyWeight {
                    SessionLogger.shared.log(
                        "[RESUME-PARAM] illegal_mass_weight: \(TrainingParameters.shared.illegalMassWeight) -> \(imw) (from session)"
                    )
                    trainer.illegalMassPenaltyWeight = imw
                    TrainingParameters.shared.illegalMassWeight = Double(imw)
                } else {
                    trainer.illegalMassPenaltyWeight = Float(TrainingParameters.shared.illegalMassWeight)
                    SessionLogger.shared.log(
                        "[RESUME-PARAM] illegal_mass_weight: saved=nil applied=\(TrainingParameters.shared.illegalMassWeight) (defaulted)"
                    )
                }
                if let lse = rs.policyLabelSmoothingEpsilon {
                    SessionLogger.shared.log(
                        "[RESUME-PARAM] policy_label_smoothing_epsilon: \(TrainingParameters.shared.policyLabelSmoothingEpsilon) -> \(lse) (from session)"
                    )
                    trainer.policyLabelSmoothingEpsilon = lse
                    TrainingParameters.shared.policyLabelSmoothingEpsilon = Double(lse)
                } else {
                    trainer.policyLabelSmoothingEpsilon = Float(TrainingParameters.shared.policyLabelSmoothingEpsilon)
                    SessionLogger.shared.log(
                        "[RESUME-PARAM] policy_label_smoothing_epsilon: saved=nil applied=\(TrainingParameters.shared.policyLabelSmoothingEpsilon) (defaulted)"
                    )
                }
                if let bsi = rs.batchStatsInterval {
                    SessionLogger.shared.log(
                        "[RESUME-PARAM] batch_stats_interval: \(TrainingParameters.shared.batchStatsInterval) -> \(bsi) (from session)"
                    )
                    trainer.batchStatsInterval = bsi
                    TrainingParameters.shared.batchStatsInterval = bsi
                } else {
                    trainer.batchStatsInterval = TrainingParameters.shared.batchStatsInterval
                    SessionLogger.shared.log(
                        "[RESUME-PARAM] batch_stats_interval: saved=nil applied=\(TrainingParameters.shared.batchStatsInterval) (defaulted)"
                    )
                }
                // LR warmup length and sqrt-batch LR scaling are now
                // part of the session schema (Optional, for back-compat
                // with older `.dcmsession` files that pre-date the
                // expansion). When the saved value is present, we
                // restore it onto both the trainer and the @AppStorage
                // mirror so the UI shows what the session was running
                // with — not whatever the user's current global
                // preference happens to be. When absent (older session),
                // we fall through to the @AppStorage value as before.
                // The trainer's internal completed-step counter is
                // seeded from `trainingSteps` so warmup scaling resumes
                // mid-session instead of restarting from zero.
                //
                // Every applied parameter emits a `[RESUME-PARAM]`
                // log line — both the "from session" and the
                // "saved=nil applied=… (defaulted)" branches — so
                // resuming an older session under post-schema-
                // expansion defaults can never silently drift the
                // applied value away from what the session was
                // trained under.
                if let savedSqrt = rs.sqrtBatchScalingForLR {
                    SessionLogger.shared.log(
                        "[RESUME-PARAM] sqrt_batch_scaling_lr: \(TrainingParameters.shared.sqrtBatchScalingLR) -> \(savedSqrt) (from session)"
                    )
                    trainer.sqrtBatchScalingForLR = savedSqrt
                    TrainingParameters.shared.sqrtBatchScalingLR = savedSqrt
                } else {
                    trainer.sqrtBatchScalingForLR = TrainingParameters.shared.sqrtBatchScalingLR
                    SessionLogger.shared.log(
                        "[RESUME-PARAM] sqrt_batch_scaling_lr: saved=nil applied=\(TrainingParameters.shared.sqrtBatchScalingLR) (defaulted)"
                    )
                }
                if let savedWarmup = rs.lrWarmupSteps, savedWarmup >= 0 {
                    SessionLogger.shared.log(
                        "[RESUME-PARAM] lr_warmup_steps: \(TrainingParameters.shared.lrWarmupSteps) -> \(savedWarmup) (from session)"
                    )
                    trainer.lrWarmupSteps = savedWarmup
                    TrainingParameters.shared.lrWarmupSteps = savedWarmup
                } else {
                    trainer.lrWarmupSteps = TrainingParameters.shared.lrWarmupSteps
                    SessionLogger.shared.log(
                        "[RESUME-PARAM] lr_warmup_steps: saved=nil applied=\(TrainingParameters.shared.lrWarmupSteps) (defaulted)"
                    )
                }
                trainer.completedTrainSteps = rs.trainingSteps
                // Run-management knobs that previously lived only in
                // @AppStorage. Same Optional-with-fallback pattern but
                // logged on both branches: a saved=nil line when an
                // older session is resumed under post-`74839ee`
                // defaults makes the silent-fallback regression that
                // motivated this audit impossible.
                if let v = rs.replayBufferMinPositionsBeforeTraining, v >= 0 {
                    SessionLogger.shared.log(
                        "[RESUME-PARAM] replay_buffer_min_positions_before_training: \(TrainingParameters.shared.replayBufferMinPositionsBeforeTraining) -> \(v) (from session)"
                    )
                    TrainingParameters.shared.replayBufferMinPositionsBeforeTraining = v
                } else {
                    SessionLogger.shared.log(
                        "[RESUME-PARAM] replay_buffer_min_positions_before_training: saved=nil applied=\(TrainingParameters.shared.replayBufferMinPositionsBeforeTraining) (defaulted)"
                    )
                }
                if let v = rs.arenaAutoIntervalSec, v > 0 {
                    SessionLogger.shared.log(
                        "[RESUME-PARAM] arena_auto_interval_sec: \(TrainingParameters.shared.arenaAutoIntervalSec) -> \(v) (from session)"
                    )
                    TrainingParameters.shared.arenaAutoIntervalSec = v
                } else {
                    SessionLogger.shared.log(
                        "[RESUME-PARAM] arena_auto_interval_sec: saved=nil applied=\(TrainingParameters.shared.arenaAutoIntervalSec) (defaulted)"
                    )
                }
                if let v = rs.arenaConcurrency, v >= 1 {
                    let clamped = min(UpperContentView.absoluteMaxArenaConcurrency, v)
                    SessionLogger.shared.log(
                        "[RESUME-PARAM] arena_concurrency: \(TrainingParameters.shared.arenaConcurrency) -> \(clamped) (from session)"
                    )
                    TrainingParameters.shared.arenaConcurrency = clamped
                } else {
                    SessionLogger.shared.log(
                        "[RESUME-PARAM] arena_concurrency: saved=nil applied=\(TrainingParameters.shared.arenaConcurrency) (defaulted)"
                    )
                }
                if let v = rs.candidateProbeIntervalSec, v > 0 {
                    SessionLogger.shared.log(
                        "[RESUME-PARAM] candidate_probe_interval_sec: \(TrainingParameters.shared.candidateProbeIntervalSec) -> \(v) (from session)"
                    )
                    TrainingParameters.shared.candidateProbeIntervalSec = v
                } else {
                    SessionLogger.shared.log(
                        "[RESUME-PARAM] candidate_probe_interval_sec: saved=nil applied=\(TrainingParameters.shared.candidateProbeIntervalSec) (defaulted)"
                    )
                }
                if let v = rs.legalMassCollapseThreshold, v > 0, v < 1 {
                    SessionLogger.shared.log(
                        "[RESUME-PARAM] legal_mass_collapse_threshold: \(TrainingParameters.shared.legalMassCollapseThreshold) -> \(v) (from session)"
                    )
                    TrainingParameters.shared.legalMassCollapseThreshold = v
                } else {
                    SessionLogger.shared.log(
                        "[RESUME-PARAM] legal_mass_collapse_threshold: saved=nil applied=\(TrainingParameters.shared.legalMassCollapseThreshold) (defaulted)"
                    )
                }
                if let v = rs.legalMassCollapseGraceSeconds, v >= 0 {
                    SessionLogger.shared.log(
                        "[RESUME-PARAM] legal_mass_collapse_grace_seconds: \(TrainingParameters.shared.legalMassCollapseGraceSeconds) -> \(v) (from session)"
                    )
                    TrainingParameters.shared.legalMassCollapseGraceSeconds = v
                } else {
                    SessionLogger.shared.log(
                        "[RESUME-PARAM] legal_mass_collapse_grace_seconds: saved=nil applied=\(TrainingParameters.shared.legalMassCollapseGraceSeconds) (defaulted)"
                    )
                }
                if let v = rs.legalMassCollapseNoImprovementProbes, v >= 1 {
                    SessionLogger.shared.log(
                        "[RESUME-PARAM] legal_mass_collapse_no_improvement_probes: \(TrainingParameters.shared.legalMassCollapseNoImprovementProbes) -> \(v) (from session)"
                    )
                    TrainingParameters.shared.legalMassCollapseNoImprovementProbes = v
                } else {
                    SessionLogger.shared.log(
                        "[RESUME-PARAM] legal_mass_collapse_no_improvement_probes: saved=nil applied=\(TrainingParameters.shared.legalMassCollapseNoImprovementProbes) (defaulted)"
                    )
                }
                // Sampling schedule — TauConfigCodable is non-Optional on
                // the session schema (added in v1), so no fallback branch.
                // Writes to @AppStorage propagate through
                // `buildSelfPlaySchedule` / `buildArenaSchedule` the next
                // time the schedule box is built below.
                TrainingParameters.shared.selfPlayStartTau = Double(rs.selfPlayTau.startTau)
                TrainingParameters.shared.selfPlayTargetTau = Double(rs.selfPlayTau.floorTau)
                TrainingParameters.shared.selfPlayTauDecayPerPly = Double(rs.selfPlayTau.decayPerPly)
                TrainingParameters.shared.arenaStartTau = Double(rs.arenaTau.startTau)
                TrainingParameters.shared.arenaTargetTau = Double(rs.arenaTau.floorTau)
                TrainingParameters.shared.arenaTauDecayPerPly = Double(rs.arenaTau.decayPerPly)
            } else {
                trainer.learningRate = Float(TrainingParameters.shared.learningRate)
                trainer.entropyRegularizationCoeff = Float(TrainingParameters.shared.entropyBonus)
                trainer.drawPenalty = Float(TrainingParameters.shared.drawPenalty)
                trainer.weightDecayC = Float(TrainingParameters.shared.weightDecay)
                trainer.gradClipMaxNorm = Float(TrainingParameters.shared.gradClipMaxNorm)
                trainer.policyLossWeight = Float(TrainingParameters.shared.policyLossWeight)
                trainer.valueLossWeight = Float(TrainingParameters.shared.valueLossWeight)
                trainer.momentumCoeff = Float(TrainingParameters.shared.momentumCoeff)
                trainer.sqrtBatchScalingForLR = TrainingParameters.shared.sqrtBatchScalingLR
                trainer.lrWarmupSteps = TrainingParameters.shared.lrWarmupSteps
            }
            var initialTrainingStats = TrainingRunStats()
            if let rs = resumeState {
                initialTrainingStats.steps = rs.trainingSteps
            }
            trainingStats = initialTrainingStats
        }
        // Game-run mode is meaningless with N > 1 self-play workers
        // (the live board hides itself behind a "N concurrent games"
        // overlay) and the Game-run / Candidate-test picker is hidden
        // in that case anyway. Default the mode to Candidate test in
        // multi-worker sessions so the user gets a usable left-side
        // panel out of the gate; single-worker sessions keep the
        // historical Game-run default.
        playAndTrainBoardMode = TrainingParameters.shared.selfPlayWorkers > 1 ? .candidateTest : .gameRun
        probeNetworkTarget = .candidate
        candidateProbeDirty = false
        lastCandidateProbeTime = .distantPast
        candidateProbeCount = 0
        // The Training and Arena popovers' edit fields are owned by
        // `trainingSettingsPopover` / `arenaSettingsPopover` and re-seed
        // themselves from `trainingParams` each time the popover
        // opens (its `onAppear`), so there's nothing to re-seed here.
        if continueMode {
            // Preserve `checkpoint?.completedTrainingSegments` and
            // `tournamentHistory` as they stood at Stop. The new
            // training segment has already been opened by
            // `beginActiveTrainingSegment` above.
        } else if let rs = resumeState {
            // Hydrate prior training segments so cumulative wall-time
            // and run-count metrics carry across save/load. Missing
            // (older session files) → empty array, which means this
            // becomes the first segment in the session's history.
            checkpoint?.completedTrainingSegments = rs.trainingSegments ?? []
            tournamentHistory = rs.arenaHistory.map { entry in
                // Legacy session files don't store `gamesPlayed` —
                // reconstruct it from the W/L/D totals (same identity
                // the driver uses when building the live record).
                // Legacy files also don't store `promotionKind` — the
                // manual Promote button didn't exist then, so treat
                // any recorded promotion as automatic.
                let gp = entry.gamesPlayed
                ?? (entry.candidateWins + entry.championWins + entry.draws)
                let kind: PromotionKind?
                if entry.promoted {
                    if let raw = entry.promotionKind,
                       let parsed = PromotionKind(rawValue: raw) {
                        kind = parsed
                    } else {
                        kind = .automatic
                    }
                } else {
                    kind = nil
                }
                return TournamentRecord(
                    finishedAtStep: entry.finishedAtStep,
                    finishedAt: entry.finishedAtUnix.map {
                        Date(timeIntervalSince1970: TimeInterval($0))
                    },
                    candidateID: entry.candidateID.map { ModelID(value: $0) },
                    championID: entry.championID.map { ModelID(value: $0) },
                    gamesPlayed: gp,
                    candidateWins: entry.candidateWins,
                    championWins: entry.championWins,
                    draws: entry.draws,
                    score: entry.score,
                    promoted: entry.promoted,
                    promotionKind: kind,
                    promotedID: entry.promotedID.map { ModelID(value: $0) },
                    durationSec: entry.durationSec,
                    candidateWinsAsWhite: entry.candidateWinsAsWhite ?? 0,
                    candidateWinsAsBlack: entry.candidateWinsAsBlack ?? 0,
                    candidateLossesAsWhite: entry.candidateLossesAsWhite ?? 0,
                    candidateLossesAsBlack: entry.candidateLossesAsBlack ?? 0,
                    candidateDrawsAsWhite: entry.candidateDrawsAsWhite ?? 0,
                    candidateDrawsAsBlack: entry.candidateDrawsAsBlack ?? 0
                )
            }
        } else {
            // Fresh session (first Start of the launch, or one of
            // the new-session modes from the Start dialog). Clear
            // both the arena history and the training-segment
            // wall-time history — "new session" means fresh
            // counters. Pre-existing first-launch behavior left
            // `checkpoint?.completedTrainingSegments` alone (benign, since it
            // was already empty at first launch), but after a
            // Stop→"New session" pick the old value would bleed
            // through, so normalize to "always fresh" here.
            tournamentHistory = []
            checkpoint?.completedTrainingSegments = []
        }
        tournamentProgress = nil
        let tBox = TournamentLiveBox()
        tournamentBox = tBox
        let pStatsBox: ParallelWorkerStatsBox
        if continueMode, let existing = parallelWorkerStatsBox {
            pStatsBox = existing
            // continueMode preserves the prior box's sessionStart, so
            // the segment baseline must stay where it was — don't
            // reset it here, otherwise the very next Stop/Start cycle
            // would re-zero a still-open segment's rate.
        } else if let rs = resumeState {
            // Resumed session: trainer was seeded with `rs.trainingSteps`
            // (see `liveStatsBox.seed(seeded)` in `runRealTraining`),
            // so the rate numerator must subtract that as the
            // segment-start baseline to express this-segment steps
            // over this-segment wall time.
            checkpoint?.trainingStepsAtSegmentStart = rs.trainingSteps
            pStatsBox = ParallelWorkerStatsBox(
                sessionStart: Date(),
                totalGames: rs.selfPlayGames,
                totalMoves: rs.selfPlayMoves,
                totalGameWallMs: rs.totalGameWallMs ?? 0,
                whiteCheckmates: rs.whiteCheckmates ?? 0,
                blackCheckmates: rs.blackCheckmates ?? 0,
                stalemates: rs.stalemates ?? 0,
                fiftyMoveDraws: rs.fiftyMoveDraws ?? 0,
                threefoldRepetitionDraws: rs.threefoldRepetitionDraws ?? 0,
                insufficientMaterialDraws: rs.insufficientMaterialDraws ?? 0,
                trainingSteps: rs.trainingSteps
            )
            if let workerCount = resumeState?.selfPlayWorkerCount {
                TrainingParameters.shared.selfPlayWorkers = max(1, min(UpperContentView.absoluteMaxSelfPlayWorkers, workerCount))
            }
            if let delay = rs.stepDelayMs {
                TrainingParameters.shared.trainingStepDelayMs = delay
            }
            if let spDelay = rs.selfPlayDelayMs {
                TrainingParameters.shared.selfPlayDelayMs = spDelay
            }
            if let autoDelay = rs.lastAutoComputedDelayMs {
                lastAutoComputedDelayMs = autoDelay
            }
            TrainingParameters.shared.replayRatioTarget = rs.replayRatioTarget ?? 1.0
            TrainingParameters.shared.replayRatioAutoAdjust = rs.replayRatioAutoAdjust ?? true
        } else {
            // Fresh session — no resumed steps to subtract.
            checkpoint?.trainingStepsAtSegmentStart = 0
            pStatsBox = ParallelWorkerStatsBox(sessionStart: Date())
        }
        parallelWorkerStatsBox = pStatsBox
        parallelStats = pStatsBox.snapshot()
        let spDiversityTracker: GameDiversityTracker
        if continueMode, let existing = selfPlayDiversityTracker {
            spDiversityTracker = existing
        } else {
            spDiversityTracker = GameDiversityTracker(windowSize: 200)
            selfPlayDiversityTracker = spDiversityTracker
            chartCoordinator?.setDiversityHistogramBars([])
        }
        if !continueMode {
            // Fresh session — wipe every chart-layer field back to a
            // zero state so the new session's chart starts at t=0
            // and doesn't show a visible "step" from the previous
            // session's trailing values.
            chartCoordinator?.reset()
            // Resume path: if a session is being restored AND it was
            // saved with chart-companion files, decode them here on
            // the main actor and seed the chart coordinator. The
            // JSON decode takes ~1-2 s on a 24h session — acceptable
            // as a one-time resume cost, on the same order as the
            // replay-buffer restore that already runs at this point.
            // A decode failure logs a warning and skips the restore;
            // the rest of the session-resume flow continues so a
            // corrupt chart file never blocks the (more important)
            // network/replay-buffer load.
            if let chartURLs = pendingLoadedSession?.chartDataURLs {
                seedChartCoordinatorFromLoadedSession(chartURLs: chartURLs)
            }
        }
        // Single self-play gate. All self-play workers now share one
        // `BatchedMoveEvaluationSource` on the champion network, driven
        // by `BatchedSelfPlayDriver`, so there is exactly one consumer
        // for the arena coordinator to pause. The previous per-secondary
        // gate array is gone along with the secondary networks.
        let selfPlayGate = WorkerPauseGate()
        // Shared current-N holder. Workers poll this to decide
        // whether to play another game or sit in their idle wait.
        // The Stepper writes through it (and to `@State
        // TrainingParameters.shared.selfPlayWorkers simultaneously). Exposed via
        // so the UI can disable the buttons when the box is gone
        // (between sessions).
        let countBox = WorkerCountBox(initial: initialWorkerCount)
        workerCountBox = countBox
        // Live schedule box, seeded from the current @AppStorage tau
        // values. Edits in the tau text fields push new schedules into
        // this box; the self-play driver's slots read at the top of
        // each new game, so the next game played uses the edited
        // values.
        let spSchedule = buildSelfPlaySchedule()
        let arSchedule = buildArenaSchedule()
        let scheduleBox = SamplingScheduleBox(
            selfPlay: spSchedule,
            arena: arSchedule
        )
        samplingScheduleBox = scheduleBox
        let ratioController = ReplayRatioController(
            batchSize: TrainingParameters.shared.trainingBatchSize,
            targetRatio: TrainingParameters.shared.replayRatioTarget,
            autoAdjust: TrainingParameters.shared.replayRatioAutoAdjust,
            initialDelayMs: TrainingParameters.shared.replayRatioAutoAdjust
            ? lastAutoComputedDelayMs
            : TrainingParameters.shared.trainingStepDelayMs,
            maxTrainingStepDelayMs: UpperContentView.stepDelayMaxMs,
            maxSelfPlayDelayMs: UpperContentView.selfPlayDelayMaxMs
        )
        // Seed the controller's manual SP-delay slot from the
        // persisted training parameter so a session that starts in
        // manual mode inherits whatever the user last left in the
        // SP-delay stepper, instead of falling back to 0.
        ratioController.manualSelfPlayDelayMs = TrainingParameters.shared.selfPlayDelayMs
        replayRatioController = ratioController
        let trainingGate = WorkerPauseGate()
        let arenaFlag = ArenaActiveFlag()
        arenaActiveFlag = arenaFlag
        let triggerBox = ArenaTriggerBox()
        arenaTriggerBox = triggerBox
        let overrideBox = ArenaOverrideBox()
        arenaOverrideBox = overrideBox
        isArenaRunning = false
        realTraining = true
        // Arm the 4-hour periodic-save scheduler. Always construct
        // a fresh controller on each start — a previous stop will
        // have nil'd it out, and `.continueAfterStop` intentionally
        // resets the next-save deadline to a full interval from
        // now rather than inheriting whatever time was left on the
        // prior arm, since the whole point of the scheduler is
        // "saved within the last 4 hours" and the session was not
        // being saved while stopped.
        let controller = PeriodicSaveController(interval: UpperContentView.periodicSaveIntervalSec)
        controller.arm(now: Date())
        periodicSaveController = controller
        periodicSaveLastPollAt = Date()
        periodicSaveInFlight = false

        // Expose the two gates the checkpoint save path needs and
        // anchor the session ID + wall clock. `checkpoint?.currentSessionID`
        // is either a fresh mint or the loaded session's ID when
        // resuming. `checkpoint?.currentSessionStart` is back-dated on
        // resume by the loaded session's `elapsedTrainingSec`, so
        // successive save-resume-save cycles accumulate elapsed
        // time monotonically. This anchor is only read by the
        // save path (`buildCurrentSessionState`) — the parallel
        // worker stats box keeps its own fresh `sessionStart`
        // anchor for rate-display purposes, so games/hr doesn't
        // get polluted by the back-dated hours.
        activeSelfPlayGate = selfPlayGate
        activeTrainingGate = trainingGate
        if continueMode {
            // Preserve `checkpoint?.currentSessionID`, `checkpoint?.currentSessionStart`,
            // `replayRatioTarget`, `replayRatioAutoAdjust`, and any
            // trainer-hyperparameter edits the user made between
            // Stop and Start. Nothing to do here.
        } else if let resumed = pendingLoadedSession {
            checkpoint?.currentSessionID = resumed.state.sessionID
            checkpoint?.currentSessionStart = Date().addingTimeInterval(-resumed.state.elapsedTrainingSec)
            TrainingParameters.shared.learningRate = Double(resumed.state.learningRate)
            if let entropyCoeff = resumed.state.entropyRegularizationCoeff {
                TrainingParameters.shared.entropyBonus = Double(entropyCoeff)
            }
        } else {
            checkpoint?.currentSessionID = ModelIDMinter.mint().value
            checkpoint?.currentSessionStart = Date()
        }

        // CLI capture mode: allocate the recorder before the
        // training Task starts so every downstream capture site
        // (stats logger, arena, candidate probe) can append to it.
        // Allocation fires when `--train` is on (headless CLI run,
        // with a potential `training_time_limit` that needs a JSON
        // artifact at expiry) OR when `--output <file>` was
        // supplied (user wants artifact regardless of train mode).
        // The recorder is `final class @unchecked Sendable` so we
        // can capture the reference into each child task without
        // wrestling MainActor isolation at each append. Cleared in
        // the teardown block along with the other per-session state.
        let recorder: CliTrainingRecorder?
        if cliOutputURL != nil || autoTrainOnLaunch {
            let r = CliTrainingRecorder()
            r.setSessionID(checkpoint?.currentSessionID)
            recorder = r
            cliRecorder = r
        } else {
            recorder = nil
        }
        let outputURL = cliOutputURL
        let cliTrainingTimeLimitSec = cliConfig?.trainingTimeLimitSec
        let isAutoTrainRun = autoTrainOnLaunch
        let runStart = Date()

        // Register the early-stop flush handler so SIGUSR1 / SIGHUP /
        // applicationShouldTerminate can write `result.json` cleanly
        // before exiting. Cleared in the teardown block. The closure
        // captures the recorder, outputURL, and runStart so the
        // coordinator doesn't need to know about ContentView's state
        // shape — it just calls the closure with the termination reason.
        if let recorder {
            EarlyStopCoordinator.shared.earlyStopHandler = { reason in
                let elapsed = Date().timeIntervalSince(runStart)
                let destDescription = outputURL?.path ?? "<stdout>"
                SessionLogger.shared.log(
                    "[APP] --train: early-stop on \(reason.rawValue) at elapsed=\(String(format: "%.1f", elapsed))s; writing snapshot to \(destDescription)"
                )
                recorder.setTerminationReason(reason)
                let counts = recorder.countsSnapshot()
                do {
                    if let url = outputURL {
                        try recorder.writeJSON(to: url, totalTrainingSeconds: elapsed)
                    } else {
                        try recorder.writeJSONToStdout(totalTrainingSeconds: elapsed)
                    }
                    SessionLogger.shared.log(
                        "[APP] --train: wrote snapshot to \(destDescription) (arenas=\(counts.arenas), stats=\(counts.stats), probes=\(counts.probes))"
                    )
                } catch {
                    SessionLogger.shared.log(
                        "[APP] --train: early-stop snapshot write FAILED for \(destDescription): \(error.localizedDescription)"
                    )
                }
                SessionLogger.shared.log("[APP] --train: exiting process after early-stop snapshot")
                Darwin._exit(0)
            }
        }

        // Snapshot the CLI-overridable effective values into plain
        // `let`s so the detached Task bodies below can read them
        // without a MainActor hop. These values don't change after
        // session start in the current model, so a one-time capture
        // is safe and keeps the Task-side code MainActor-free.
        let sessionTrainingBatchSize = TrainingParameters.shared.trainingBatchSize
        let sessionMinBufferBeforeTraining = TrainingParameters.shared.replayBufferMinPositionsBeforeTraining
        let sessionTournamentGames = TrainingParameters.shared.arenaGamesPerTournament
        let sessionPromoteThreshold = TrainingParameters.shared.arenaPromoteThreshold

        realTrainingTask = Task(priority: .high) {
            [trainer, network, buffer, box, tBox, pStatsBox, spDiversityTracker,
             selfPlayGate, trainingGate, arenaFlag, triggerBox, overrideBox, countBox,
             gameWatcher, ratioController, recorder, outputURL, cliTrainingTimeLimitSec,
             isAutoTrainRun,
             sessionTrainingBatchSize, sessionMinBufferBeforeTraining,
             sessionTournamentGames, sessionPromoteThreshold] in

            // --- Setup: build any missing networks, reset the trainer ---

            let needsCandidateBuild = await MainActor.run { candidateInferenceNetwork == nil }
            if needsCandidateBuild {
                do {
                    let built = try await Task.detached(priority: .userInitiated) {
                        try ChessMPSNetwork(.randomWeights)
                    }.value
                    built.network.commandQueue.label = "startrealTraining candidate Inference"
                    await MainActor.run {
                        candidateInferenceNetwork = built
                    }
                } catch {
                    box.recordError("Candidate network init failed: \(error.localizedDescription)")
                    await MainActor.run {
                        realTraining = false
                        realTrainingTask = nil
                    }
                    return
                }
            }

            let needsProbeBuild = await MainActor.run { probeInferenceNetwork == nil }
            if needsProbeBuild {
                do {
                    let built = try await Task.detached(priority: .userInitiated) {
                        try ChessMPSNetwork(.randomWeights)
                    }.value
                    built.network.commandQueue.label = "startrealTraining probe Inference"
                    await MainActor.run {
                        probeInferenceNetwork = built
                        probeRunner = ChessRunner(network: built)
                    }
                } catch {
                    box.recordError("Probe network init failed: \(error.localizedDescription)")
                    await MainActor.run {
                        realTraining = false
                        realTrainingTask = nil
                    }
                    return
                }
            }

            let needsArenaChampionBuild = await MainActor.run { arenaChampionNetwork == nil }
            if needsArenaChampionBuild {
                do {
                    let built = try await Task.detached(priority: .userInitiated) {
                        try ChessMPSNetwork(.randomWeights)
                    }.value
                    built.network.commandQueue.label = "startrealTraining arena champion"
                    await MainActor.run {
                        arenaChampionNetwork = built
                    }
                } catch {
                    box.recordError("Arena champion init failed: \(error.localizedDescription)")
                    await MainActor.run {
                        realTraining = false
                        realTrainingTask = nil
                    }
                    return
                }
            }

            // Reset the trainer's graph AND initialize its weights,
            // branched by `mode`:
            //
            // - `.freshOrFromLoadedSession`: rebuild graph, then
            //   load trainer weights from the loaded session's
            //   `trainer.dcmmodel` if present, or fork them from
            //   the champion. This is the default first-launch and
            //   disk-resume path.
            // - `.newSessionResetTrainerFromChampion`: rebuild
            //   graph, fork trainer weights from the live champion.
            //   User explicitly asked to throw away accumulated
            //   trainer weights and start fresh from champion.
            // - `.continueAfterStop` and `.newSessionKeepTrainer`:
            //   leave the trainer's graph and weights untouched —
            //   the trainer's MPSGraph stayed alive across Stop,
            //   and preserving its weights is the whole point of
            //   both modes.
            //
            // In the fork-from-champion branch, champion weights
            // loaded from disk at file-load time are already live
            // in `network`, so the batcher (which wraps `network`)
            // picks up the restored weights automatically.
            let resumedTrainerWeights: [[Float]]? = await MainActor.run {
                pendingLoadedSession?.trainerFile.weights
            }
            let resumedBufferURL: URL? = await MainActor.run {
                pendingLoadedSession?.replayBufferURL
            }
            do {
                switch mode {
                case .continueAfterStop, .newSessionKeepTrainer:
                    break
                case .freshOrFromLoadedSession:
                    try await trainer.resetNetwork()
                    try await Task.detached(priority: .userInitiated) {
                        [resumedTrainerWeights] in
                        if let trainerWeights = resumedTrainerWeights {
                            // Session resume requires exact trainer
                            // state, including optimizer velocity.
                            try await trainer.loadTrainerWeights(trainerWeights)
                        } else {
                            // No prior trainer file (fresh session or
                            // pre-existing session without trainer.dcmmodel):
                            // fork from champion and intentionally
                            // start optimizer velocity from zero.
                            let championWeights = try await network.exportWeights()
                            try await trainer.loadBaseWeightsResetVelocity(championWeights)
                        }
                    }.value
                case .newSessionResetTrainerFromChampion:
                    try await trainer.resetNetwork()
                    try await Task.detached(priority: .userInitiated) {
                        // User explicitly asked to discard trainer state
                        // and re-fork from champion. Velocity goes back
                        // to zero, which is the correct semantics for
                        // a fresh fork.
                        let championWeights = try await network.exportWeights()
                        try await trainer.loadBaseWeightsResetVelocity(championWeights)
                    }.value
                }
            } catch {
                box.recordError("Reset failed: \(error.localizedDescription)")
                await MainActor.run {
                    realTraining = false
                    realTrainingTask = nil
                }
                return
            }

            // Dump every live MTLCommandQueue's (label, address) pair
            // so subsequent Metal-trace captures can be matched back
            // to the network they came from. The address printed here
            // is what Xcode's Metal frame-capture UI shows under
            // "Command Queues" on the trace selection sheet — pick by
            // label, then verify by address. Emitted once per session
            // start; queues persist for the app lifetime so this is
            // accurate for every later capture in the same launch.
            do {
                let candidateQ = await MainActor.run {
                    candidateInferenceNetwork?.network.commandQueue
                }
                let probeQ = await MainActor.run {
                    probeInferenceNetwork?.network.commandQueue
                }
                let arenaChampionQ = await MainActor.run {
                    arenaChampionNetwork?.network.commandQueue
                }
                let championQ = network.network.commandQueue
                let trainerQ = trainer.network.commandQueue
                func desc(_ label: String, _ q: MTLCommandQueue?) -> String {
                    guard let q else { return "\(label)=<not built>" }
                    // Pointer that matches the address Xcode's Metal
                    // frame-capture UI shows in the queue selector.
                    let opaque = Unmanaged.passUnretained(q as AnyObject).toOpaque()
                    let addr = String(format: "0x%016lx", UInt(bitPattern: opaque))
                    return "\(label)=\(q.label ?? "<no-label>") @ \(addr)"
                }
                SessionLogger.shared.log(
                    "[QUEUES] champion: \(desc("champion", championQ)); "
                    + "trainer: \(desc("trainer", trainerQ)); "
                    + "candidate: \(desc("candidate", candidateQ)); "
                    + "probe: \(desc("probe", probeQ)); "
                    + "arenaChampion: \(desc("arenaChampion", arenaChampionQ))"
                )
            }

            // Restore replay buffer before any self-play worker or
            // the training worker starts — training samples from the
            // buffer, and any worker appends after restore resets
            // would either be clobbered or (worse) race with the
            // restore's counter reset. The restore runs on a
            // detached I/O task so ~GB-scale reads don't block the
            // cooperative hop cadence.
            if let bufferURL = resumedBufferURL {
                let resumedState: SessionCheckpointState? = await MainActor.run {
                    pendingLoadedSession?.state
                }
                do {
                    try await Task.detached(priority: .userInitiated) {
                        [buffer, bufferURL] in
                        try buffer.restore(from: bufferURL)
                    }.value
                    // Cross-check lifetime counter against session.json.
                    // Mismatch here indicates file-pairing error or
                    // residual corruption that happened to SHA-match.
                    if let resumedState {
                        try CheckpointManager.verifyReplayBufferMatchesSession(
                            buffer: buffer,
                            state: resumedState
                        )
                    }
                    let snap = buffer.stateSnapshot()
                    SessionLogger.shared.log(
                        "[CHECKPOINT] Restored replay buffer: stored=\(snap.storedCount)/\(snap.capacity) totalAdded=\(snap.totalPositionsAdded) writeIndex=\(snap.writeIndex)"
                    )
                } catch {
                    box.recordError("Replay buffer restore failed: \(error.localizedDescription)")
                    SessionLogger.shared.log(
                        "[CHECKPOINT] Replay buffer restore failed: \(error.localizedDescription) — continuing with empty buffer"
                    )
                }
            }

            // Trainer ID — branched by `mode`, matching the
            // trainer-weights logic above:
            //
            // - `.freshOrFromLoadedSession`: inherit from the
            //   loaded session's trainer file if present, else
            //   mint a new generation off the champion.
            // - `.newSessionResetTrainerFromChampion`: mint a new
            //   generation off the current champion (trainer
            //   weights were just forked from champion).
            // - `.continueAfterStop` and `.newSessionKeepTrainer`:
            //   keep the trainer's existing ID — its weights
            //   weren't touched, so the lineage is continuous.
            await MainActor.run {
                switch mode {
                case .continueAfterStop, .newSessionKeepTrainer:
                    break
                case .newSessionResetTrainerFromChampion:
                    trainer.identifier = ModelIDMinter.mintTrainerGeneration(
                        from: network.identifier ?? ModelIDMinter.mint()
                    )
                case .freshOrFromLoadedSession:
                    if let resumed = pendingLoadedSession {
                        trainer.identifier = ModelID(value: resumed.trainerFile.modelID)
                    } else {
                        trainer.identifier = ModelIDMinter.mintTrainerGeneration(
                            from: network.identifier ?? ModelIDMinter.mint()
                        )
                    }
                }
                // Consume the pending load — from here on, the
                // running session owns the restored state.
                pendingLoadedSession = nil
                // A training segment has started, so clear the
                // "champion replaced since last training" flag
                // (the Start dialog's annotation is resolved).
                onClearChampionLoadedFlag()
            }

            // Grab the candidate inference network and arena champion
            // network references on the main actor once — both are
            // now guaranteed non-nil from the setup above. The
            // workers capture them as values for the duration of the
            // 
            let candidateInference = await MainActor.run { candidateInferenceNetwork }
            let arenaChampion = await MainActor.run { arenaChampionNetwork }
            guard let candidateInference, let arenaChampion else {
                box.recordError("Networks missing after setup")
                await MainActor.run {
                    realTraining = false
                    realTrainingTask = nil
                }
                return
            }

            // --- Spawn the worker tasks ---
            //
            // absoluteMaxSelfPlayWorkers self-play tasks, one training
            // worker, one arena coordinator, one session-log
            // ticker. The Stepper picks how many of the self-play
            // tasks are *active* at any moment — the rest sit in
            // their pause gate's wait state until the user raises
            // N enough to include them.

            // Anchor the session wall-clock to *now*, after all the
            // synchronous and MainActor-hop setup above has finished.
            // Rate denominators ("steps/sec", "games/hr", "avg move
            // ms", "Total session time", ...) are computed as `Date()
            // - sessionStart`, so leaving the original `Date()`-at-
            // button-press anchor in place would bake the setup
            // delay into every average for the life of the 
            pStatsBox.markWorkersStarted()

            // Build the shared self-play batcher and driver. All slots
            // play against one `ChessMPSNetwork` (the champion, via
            // `network`) through a `BatchedMoveEvaluationSource` actor
            // that coalesces N per-ply forward passes into one batched
            // `graph.run`. The driver owns the slot lifecycle: it
            // spawns up to `countBox.count` child tasks, responds to
            // Stepper-driven count changes, and pauses the whole
            // self-play subsystem through the shared `selfPlayGate`
            // when arena requests it (replacing the previous
            // per-worker gate array).
            let selfPlayBatcher = BatchedMoveEvaluationSource(network: network)
            // Wire the ratio controller into the batcher so every
            // barrier fire reports an aggregate (positions, elapsed)
            // measurement. `setReplayRatioController` is actor-
            // isolated, so it's dispatched via a `Task`.
            Task { [ratioController] in
                await selfPlayBatcher.setReplayRatioController(ratioController)
            }
            let selfPlayDriver = BatchedSelfPlayDriver(
                batcher: selfPlayBatcher,
                buffer: buffer,
                statsBox: pStatsBox,
                diversityTracker: spDiversityTracker,
                countBox: countBox,
                pauseGate: selfPlayGate,
                gameWatcher: gameWatcher,
                scheduleBox: scheduleBox,
                replayRatioController: ratioController
            )

            // Pin the probe inference network into a local the child
            // tasks can capture. It's a @State on this view, so
            // reading it requires a MainActor hop that child tasks
            // would otherwise repeat. Built lazily earlier in this
            // function (lines ~7694-7703); should be non-nil here but
            // we tolerate nil and skip probes in that case rather
            // than fall back to the pollution-prone trainer-network
            // probe path.
            let probeInferenceForProbes: ChessMPSNetwork? = await MainActor.run {
                probeInferenceNetwork
            }

            await withTaskGroup(of: Void.self) { group in
                // Self-play driver: manages N concurrent
                // `ChessMachine` game loops against the shared
                // batcher. Slot count tracks `countBox.count` live;
                // pause requests on `selfPlayGate` flow through the
                // driver and stop every slot at its next game
                // boundary. Each slot streams positions into the
                // shared replay buffer via its white/black
                // `MPSChessPlayer` pair, records stats through
                // `pStatsBox`, and feeds `spDiversityTracker`.
                //
                // Slot 0 drives the live-display `GameWatcher` when
                // there is exactly one active slot; all slots
                // contribute identically to `pStatsBox` /
                // `spDiversityTracker`.
                let selfPlayTaskCreatedAt = Date()
                group.addTask(priority: .high) {
                    [selfPlayDriver, selfPlayTaskCreatedAt] in
                    let creatingMs = Int(Date().timeIntervalSince(selfPlayTaskCreatedAt) * 1000)
                    SessionLogger.shared.log("[TASK] self-play driver: created→exec=\(creatingMs)ms")
                    await selfPlayDriver.run()
                }

                // Training worker: tight-loop SGD on the trainer,
                // sampling batches from the replay buffer. Fires the
                // candidate probe at its own 15 s cadence between
                // steps, and nudges the arena trigger box when the
                // 30 min auto cadence elapses. Pauses at `trainingGate`
                // so the arena coordinator can briefly snapshot
                // trainer weights.
                let trainingTaskCreatedAt = Date()
                group.addTask(priority: .high) {
                    [trainer, buffer, box, pStatsBox, trainingGate, triggerBox, ratioController,
                     sessionTrainingBatchSize, sessionMinBufferBeforeTraining, trainingTaskCreatedAt] in
                    let creatingMs = Int(Date().timeIntervalSince(trainingTaskCreatedAt) * 1000)
                    SessionLogger.shared.log("[TASK] training worker: created→exec=\(creatingMs)ms")
                    // Track the previous step's applied delay so the
                    // next `recordTrainingBatchAndGetDelay` can report
                    // it as the current per-batch training-side delay
                    // setting. The controller owns the inter-call
                    // wall-clock measurement directly; the caller
                    // just reports its current configured delay.
                    var lastTrainingDelaySettingMs: Int = 0
                    while !Task.isCancelled {
                        // Pause gate check (between steps).
                        if trainingGate.isRequestedToPause {
                            trainingGate.markWaiting()
                            while trainingGate.isRequestedToPause && !Task.isCancelled {
                                // `try?` is intentional: `Task.sleep` only
                                // throws `CancellationError`, and the loop
                                // condition re-checks `Task.isCancelled` on
                                // the very next iteration.
                                try? await Task.sleep(for: .milliseconds(5))
                            }
                            trainingGate.markRunning()
                        }
                        if Task.isCancelled { break }

                        // Wait for the replay buffer to warm up before
                        // starting to train — the first few games
                        // haven't produced enough decorrelated samples
                        // yet. Short sleep + retry keeps the worker
                        // responsive to Stop and to pause requests.
                        if buffer.count < sessionMinBufferBeforeTraining {
                            // `try?` is intentional: only `CancellationError`
                            // can be thrown; the enclosing
                            // `while !Task.isCancelled` exits on the next
                            // iteration.
                            try? await Task.sleep(for: .milliseconds(100))
                            continue
                        }

                        // The enclosing worker already runs at
                        // `.userInitiated` so there's nothing to escape
                        // to — run the SGD step inline and skip the
                        // per-step detached-task + continuation
                        // allocation pair. Sampling happens inside the
                        // trainer on its serial queue so replay-buffer
                        // rows are copied directly into trainer-owned
                        // staging buffers.
                        let timing: TrainStepTiming
                        do {
                            guard let sampledTiming = try await trainer.trainStep(
                                replayBuffer: buffer,
                                batchSize: sessionTrainingBatchSize
                            ) else {
                                // `try?` is intentional: only `CancellationError`
                                // is throwable; the enclosing loop's
                                // `!Task.isCancelled` check exits cleanly.
                                try? await Task.sleep(for: .milliseconds(100))
                                continue
                            }
                            timing = sampledTiming
                        } catch {
                            box.recordError(error.localizedDescription)
                            return
                        }

                        box.recordStep(timing)
                        pStatsBox.recordTrainingStep()

                        // Candidate-test probe firing check. Method
                        // guards internally on all preconditions
                        // including arena-active, so an unconditional
                        // call here is safe — it no-ops when nothing
                        // is due or when the arena is running.
                        await self.fireCandidateProbeIfNeeded()

                        // Auto-trigger the arena on the configured
                        // cadence. Fires the trigger inbox; the arena
                        // coordinator task picks it up and runs the
                        // tournament. Reads `arenaAutoIntervalSec` live
                        // (the parameter is `liveTunable: true`) so the
                        // status-bar Arena popover's Save edits take
                        // effect on the next poll without restarting
                        // the 
                        //
                        // CRITICAL: The arena clock only begins ticking
                        // once the model has reached stability (buffer
                        // prefill AND LR warmup complete). While either
                        // is in progress, we keep resetting the anchor
                        // to 'now' so the first auto-arena occurs exactly
                        // one interval after the model matures.
                        let liveParams = await TrainingParameters.shared.snapshot()
                        let liveInterval = liveParams.arenaAutoIntervalSec
                        let isWarmup = trainer.completedTrainSteps < liveParams.lrWarmupSteps
                        if isWarmup {
                            triggerBox.resetLastArenaTime(to: Date())
                        } else if triggerBox.shouldAutoTrigger(interval: liveInterval) {
                            triggerBox.trigger()
                        }

                        // Post-step pause. Symmetric API with the
                        // self-play barrier tick: caller reports its
                        // currently-applied per-batch delay setting,
                        // controller owns the wall-clock and does
                        // the subtraction. The "current setting"
                        // we report is whatever sleep we just
                        // applied on the previous loop pass — that
                        // IS the configured delay during the
                        // controller-owned inter-call period that
                        // ends right now.
                        let stepDelayMs = ratioController.recordTrainingBatchAndGetDelay(
                            currentDelaySettingMs: Double(lastTrainingDelaySettingMs)
                        )
                        lastTrainingDelaySettingMs = stepDelayMs
                        if stepDelayMs > 0 {
                            // `try?` is intentional: cancellation is the
                            // normal exit path; the enclosing
                            // `while !Task.isCancelled` re-checks
                            // cancellation on the next pass.
                            try? await Task.sleep(for: .milliseconds(stepDelayMs))
                        }
                    }
                }

                // Arena coordinator: polls the trigger inbox and runs
                // a tournament whenever one is pending. Blocks its
                // own loop (not the worker tasks) during arena
                // execution. Both the 30-minute auto-fire and the
                // Run Arena button enter here via `triggerBox.trigger()`.
                let arenaTaskCreatedAt = Date()
                group.addTask(priority: .high) {
                    [arenaTaskCreatedAt] in
                    let creatingMs = Int(Date().timeIntervalSince(arenaTaskCreatedAt) * 1000)
                    SessionLogger.shared.log("[TASK] arena coordinator: created→exec=\(creatingMs)ms")
                    // Event-driven wait — sleeps until a trigger
                    // fires, the box is cancelled, or the Task is
                    // cancelled. The trigger-box's tri-state return
                    // is the single source of truth for what to do
                    // next — no separate `Task.isCancelled` check
                    // needed after the await.
                    arenaLoop: while true {
                        switch await triggerBox.waitForTrigger() {
                        case .cancelled:
                            break arenaLoop
                        case .falseAlarm:
                            continue arenaLoop
                        case .fire:
                            await self.runArenaParallel(
                                trainer: trainer,
                                champion: network,
                                candidateInference: candidateInference,
                                arenaChampion: arenaChampion,
                                tBox: tBox,
                                selfPlayGate: selfPlayGate,
                                trainingGate: trainingGate,
                                arenaFlag: arenaFlag,
                                overrideBox: overrideBox
                            )
                            triggerBox.recordArenaCompleted()
                        }
                    }
                }

                // Periodic session-log ticker. Emits one [STATS] line
                // per training step for the first 500 steps (every step
                // matters during bootstrap — you want to see the curve
                // shape of the first few hundred updates) then drops to
                // one line per 60 seconds for the rest of the 
                //
                // Each wake-up snapshots the thread-safe stats boxes,
                // optionally refreshes `legalMass` via a sampled
                // forward pass (cadence-gated so the CPU work doesn't
                // pile up during the per-step bootstrap window), and
                // writes one `[STATS]` line. Identifiers are pulled
                // through a brief MainActor hop since they live on
                // classes whose var mutation is otherwise main-actor-
                // driven.
                group.addTask(priority: .utility) {
                    [trainer, network, box, pStatsBox, buffer, spDiversityTracker, ratioController, countBox, scheduleBox, recorder,
                     sessionTrainingBatchSize, sessionTournamentGames, sessionPromoteThreshold, probeInferenceForProbes] in
                    let sessionStart = Date()
                    // Bootstrap-phase step threshold for the per-step
                    // emit. `UpperContentView.bootstrapStatsStepCount` is tunable
                    // on the view; at default 500 steps this covers
                    // roughly the first 1-3 minutes of real-data
                    // training at typical throughput.
                    let bootstrapSteps = UpperContentView.bootstrapStatsStepCount
                    // Time between STATS emits after the bootstrap
                    // window closes. 60 s chosen so a session's
                    // steady-state log file grows at a manageable rate
                    // (~60 lines/hr) while still capturing drift
                    // inside the typical 30-minute arena cadence.
                    let steadyInterval: TimeInterval = 60
                    // Cadence for refreshing legalMass during the
                    // per-step bootstrap window — refreshing every
                    // step would double per-step CPU cost for little
                    // additional signal. Every 25 steps roughly
                    // matches the 60-second cadence used afterwards
                    // at typical throughput.
                    let legalMassBootstrapStride = 25
                    let legalMassSampleSize = 128

                    // First-observed pwNorm becomes the session baseline
                    // so each [STATS] line can report the absolute value
                    // alongside the drift since session start. Captured
                    // lazily on the first non-nil reading rather than at
                    // task launch — when training has just kicked off
                    // the trainer's rolling weight-norm window may not
                    // be populated yet. Mutated from inside the nested
                    // `logOne` closure; safe because every call site
                    // runs serially on this task.
                    var pwNormBaseline: Double? = nil

                    // Carried across [STATS] emits so each line can
                    // report the delta since the previous tick. nil
                    // until the first sample lands; the very first emit
                    // shows `drss=+0` etc., which is fine — the deltas
                    // are only meaningful from the second tick onward.
                    var prevRssBytes: UInt64 = 0
                    var prevVmTotal: UInt32 = 0
                    var prevVmIoAccel: UInt32 = 0

                    func logOne(elapsedTarget: TimeInterval, legalMassOverride: ChessTrainer.LegalMassSnapshot?) async {
                        let trainingSnap = await box.snapshot()
                        let parallelSnap = pStatsBox.snapshot()
                        let bufCount = buffer.count
                        let bufCap = buffer.capacity
                        let ratioSnap = ratioController.snapshot()
                        let workerN = countBox.count
                        let spSched = scheduleBox.selfPlay
                        let arSched = scheduleBox.arena
                        let (trainerID, championID, lr, entropyCoeff, illegalMassW, drawPen, weightDec, gradClip, policyW, valueW, momentum, sqrtLR, warmupSteps, completedSteps, arenaAutoSec, livePromoteThreshold, liveTournamentGames) = await MainActor.run {
                            (
                                trainer.identifier?.description ?? "?",
                                network.identifier?.description ?? "?",
                                trainer.learningRate,
                                trainer.entropyRegularizationCoeff,
                                trainer.illegalMassPenaltyWeight,
                                trainer.drawPenalty,
                                trainer.weightDecayC,
                                trainer.gradClipMaxNorm,
                                trainer.policyLossWeight,
                                trainer.valueLossWeight,
                                trainer.momentumCoeff,
                                trainer.sqrtBatchScalingForLR,
                                trainer.lrWarmupSteps,
                                trainer.completedTrainSteps,
                                TrainingParameters.shared.arenaAutoIntervalSec,
                                TrainingParameters.shared.arenaPromoteThreshold,
                                TrainingParameters.shared.arenaGamesPerTournament
                            )
                        }
                        let policyStr: String
                        if let p = trainingSnap.rollingPolicyLoss {
                            policyStr = String(format: "%+.4f", p)
                        } else {
                            policyStr = "--"
                        }
                        let pLossWinStr: String
                        if let p = trainingSnap.rollingPolicyLossWin {
                            pLossWinStr = String(format: "%+.4f", p)
                        } else {
                            pLossWinStr = "--"
                        }
                        let pLossLossStr: String
                        if let p = trainingSnap.rollingPolicyLossLoss {
                            pLossLossStr = String(format: "%+.4f", p)
                        } else {
                            pLossLossStr = "--"
                        }
                        let valueStr: String
                        if let v = trainingSnap.rollingValueLoss {
                            valueStr = String(format: "%+.4f", v)
                        } else {
                            valueStr = "--"
                        }
                        let entropyStr = String(format: "%.4f", trainingSnap.rollingPolicyEntropy ?? 0)
                        let illegalPenaltyStr = String(format: "%.4f", trainingSnap.rollingIllegalMassPenalty ?? 0)
                        let gradNormStr: String
                        if let g = trainingSnap.rollingGradGlobalNorm {
                            gradNormStr = String(format: "%.3f", g)
                        } else {
                            gradNormStr = "--"
                        }
                        let vNormStr: String
                        if let vn = trainingSnap.rollingVelocityNorm {
                            vNormStr = String(format: "%.3f", vn)
                        } else {
                            vNormStr = "--"
                        }
                        let muStr = String(format: "%.3f", momentum)
                        let vMeanStr: String
                        if let vm = trainingSnap.rollingValueMean {
                            vMeanStr = String(format: "%+.4f", vm)
                        } else {
                            vMeanStr = "--"
                        }
                        let vAbsStr: String
                        if let va = trainingSnap.rollingValueAbsMean {
                            vAbsStr = String(format: "%.4f", va)
                        } else {
                            vAbsStr = "--"
                        }
                        // vBaseDelta = mean abs delta between trainer's
                        // current v(s) and the play-time-frozen vBaseline
                        // from the random-init champion. Higher = trainer
                        // is genuinely diverging from the champion.
                        let vBaseDeltaStr: String
                        if let vbd = trainingSnap.rollingVBaselineDelta {
                            vBaseDeltaStr = String(format: "%.4f", vbd)
                        } else {
                            vBaseDeltaStr = "--"
                        }
                        let h = Int(elapsedTarget) / 3600
                        let m = (Int(elapsedTarget) % 3600) / 60
                        let s = Int(elapsedTarget) % 60
                        let elapsedStr = String(format: "%d:%02d:%02d", h, m, s)
                        let spTau = String(format: "%.2f/%.2f/%.3f", spSched.startTau, spSched.floorTau, spSched.decayPerPly)
                        let arTau = String(format: "%.2f/%.2f/%.3f", arSched.startTau, arSched.floorTau, arSched.decayPerPly)
                        let pwNormStr: String
                        if let pwn = trainingSnap.rollingPolicyHeadWeightNorm {
                            if pwNormBaseline == nil {
                                pwNormBaseline = pwn
                            }
                            let delta = pwn - (pwNormBaseline ?? pwn)
                            pwNormStr = String(format: "%.5f(Δ%+.5f)", pwn, delta)
                        } else {
                            pwNormStr = "--"
                        }
                        let divSnap = spDiversityTracker.snapshot()
                        let divStr = divSnap.gamesInWindow > 0
                        ? String(format: "unique=%d/%d(%.0f%%) diverge=%.1f", divSnap.uniqueGames, divSnap.gamesInWindow, divSnap.uniquePercent, divSnap.avgDivergencePly)
                        : "n/a"
                        // Append a `·√b` marker when sqrt-batch scaling is on
                        // and a `·warmup(i/N)` marker while warmup is still
                        // active, so the reader can tell at a glance what
                        // per-step multipliers the optimizer is actually
                        // seeing. The displayed LR is always the base value.
                        var lrStr = String(format: "%.1e", lr) + (sqrtLR ? "·√b" : "")
                        if warmupSteps > 0 && completedSteps < warmupSteps {
                            lrStr += "·warmup(\(completedSteps)/\(warmupSteps))"
                        }
                        let spMsStr: String
                        if let sp = ratioSnap.selfPlayMsPerMove {
                            spMsStr = String(format: "%.2f", sp)
                        } else {
                            spMsStr = "--"
                        }
                        let trMsStr: String
                        if let tr = ratioSnap.trainingMsPerMove {
                            trMsStr = String(format: "%.2f", tr)
                        } else {
                            trMsStr = "--"
                        }
                        // moves/hour companions to prod/cons (which are
                        // pos/sec). Same rolling 60-s window — these are
                        // a pure unit conversion, exposed here so the
                        // [STATS] line and result.json carry the rate in
                        // the unit a human asks for ("how many moves per
                        // hour are we training on?").
                        let spMovesPerHour = ratioSnap.productionRate * 3600.0
                        let trainMovesPerHour = ratioSnap.consumptionRate * 3600.0
                        let ratioStr = String(format: "target=%.2f cur=%.2f prod=%.1f cons=%.1f spRate=%.0f/hr trainRate=%.0f/hr auto=%@ delay=%dms spDelay=%dms spMs=%@ trMs=%@ workers=%d",
                                              ratioSnap.targetRatio, ratioSnap.currentRatio,
                                              ratioSnap.productionRate, ratioSnap.consumptionRate,
                                              spMovesPerHour, trainMovesPerHour,
                                              ratioSnap.autoAdjust ? "on" : "off",
                                              ratioSnap.computedDelayMs, ratioSnap.computedSelfPlayDelayMs,
                                              spMsStr, trMsStr, ratioSnap.workerCount)
                        let outcomeStr = String(format: "wMate=%d bMate=%d stale=%d 50mv=%d 3fold=%d insuf=%d",
                                                parallelSnap.whiteCheckmates, parallelSnap.blackCheckmates,
                                                parallelSnap.stalemates, parallelSnap.fiftyMoveDraws,
                                                parallelSnap.threefoldRepetitionDraws, parallelSnap.insufficientMaterialDraws)
                        let cfgStr = "batch=\(sessionTrainingBatchSize) lr=\(lrStr) promote>=\(String(format: "%.2f", livePromoteThreshold)) arenaGames=\(liveTournamentGames) arenaAutoSec=\(Int(arenaAutoSec)) workers=\(workerN)"
                        let regStr = String(
                            format: "clip=%.1f decay=%.0e ent=%.1e illM=%.1e drawPen=%.3f pLossW=%.2f vLossW=%.2f μ=%.2f",
                            gradClip,
                            weightDec,
                            entropyCoeff,
                            illegalMassW,
                            drawPen,
                            policyW,
                            valueW,
                            momentum
                        )
                        // Average game length: lifetime and 10-min
                        // rolling window. `selfPlayPositions` counts
                        // every ply played, so dividing by the number
                        // of completed games gives the mean plies-per-
                        // game. Rolling avg tracks recent behavior;
                        // lifetime avg catches longer-term drift.
                        let lifetimeAvgLen: Double = parallelSnap.selfPlayGames > 0
                        ? Double(parallelSnap.selfPlayPositions) / Double(parallelSnap.selfPlayGames)
                        : 0
                        let rollingAvgLen: Double = parallelSnap.recentGames > 0
                        ? Double(parallelSnap.recentMoves) / Double(parallelSnap.recentGames)
                        : 0
                        let p50Str: String
                        let p95Str: String
                        if let p50 = parallelSnap.gameLenP50 {
                            p50Str = String(p50)
                        } else {
                            p50Str = "--"
                        }
                        if let p95 = parallelSnap.gameLenP95 {
                            p95Str = String(p95)
                        } else {
                            p95Str = "--"
                        }
                        let gameLenStr = String(format: "avgLen=%.1f rollingAvgLen=%.1f p50=\(p50Str) p95=\(p95Str)", lifetimeAvgLen, rollingAvgLen)
                        // New encoding/gradient-health signals.
                        let playedProbStr: String
                        if let pm = trainingSnap.rollingPlayedMoveProb {
                            playedProbStr = String(format: "%.4f", pm)
                        } else {
                            playedProbStr = "--"
                        }
                        // Advantage-sign-conditional played-move
                        // probabilities. Under a working training loop
                        // `posAdv` rises above `1/policySize` and
                        // `negAdv` falls — divergence between the two
                        // is the real action-index/direction-of-
                        // learning signal (the unconditional
                        // `playedMoveProb` is ambiguous under adv-
                        // normalized loss). The `skip=K/N` suffix
                        // reports how many steps in the rolling window
                        // dropped out because their batch had no
                        // positions on that side of A — that's the
                        // effective sample count behind the mean.
                        let condWindowSize = trainingSnap.rollingPlayedMoveCondWindowSize
                        let playedProbPosStr: String
                        if let pm = trainingSnap.rollingPlayedMoveProbPosAdv {
                            playedProbPosStr = String(
                                format: "%.4f(skip=%d/%d)",
                                pm,
                                trainingSnap.rollingPlayedMoveProbPosAdvSkipped,
                                condWindowSize
                            )
                        } else {
                            playedProbPosStr = String(
                                format: "--(skip=%d/%d)",
                                trainingSnap.rollingPlayedMoveProbPosAdvSkipped,
                                condWindowSize
                            )
                        }
                        let playedProbNegStr: String
                        if let pm = trainingSnap.rollingPlayedMoveProbNegAdv {
                            playedProbNegStr = String(
                                format: "%.4f(skip=%d/%d)",
                                pm,
                                trainingSnap.rollingPlayedMoveProbNegAdvSkipped,
                                condWindowSize
                            )
                        } else {
                            playedProbNegStr = String(
                                format: "--(skip=%d/%d)",
                                trainingSnap.rollingPlayedMoveProbNegAdvSkipped,
                                condWindowSize
                            )
                        }
                        let pLogitMaxStr: String
                        if let pm = trainingSnap.rollingPolicyLogitAbsMax {
                            pLogitMaxStr = String(format: "%.3f", pm)
                        } else {
                            pLogitMaxStr = "--"
                        }
                        // Advantage distribution summary. Lots of
                        // fields but they go in one parenthesized
                        // block in the line so grep for
                        // "adv=(" when analyzing.
                        func advFmt(_ d: Double?) -> String {
                            guard let d else { return "--" }
                            return String(format: "%+.4f", d)
                        }
                        func advFracFmt(_ d: Double?) -> String {
                            guard let d else { return "--" }
                            return String(format: "%.2f", d)
                        }
                        let advStr = "mean=\(advFmt(trainingSnap.rollingAdvMean)) std=\(advFmt(trainingSnap.rollingAdvStd)) min=\(advFmt(trainingSnap.rollingAdvMin)) max=\(advFmt(trainingSnap.rollingAdvMax)) frac+=\(advFracFmt(trainingSnap.rollingAdvFracPositive)) fracSmall=\(advFracFmt(trainingSnap.rollingAdvFracSmall)) p05=\(advFmt(trainingSnap.advantageP05)) p50=\(advFmt(trainingSnap.advantageP50)) p95=\(advFmt(trainingSnap.advantageP95))"
                        let legalMassStr: String
                        let top1LegalStr: String
                        let pEntLegalStr: String
                        if let lm = legalMassOverride {
                            legalMassStr = String(format: "%.4f", lm.legalMass)
                            top1LegalStr = String(format: "%.2f", lm.top1LegalFraction)
                            pEntLegalStr = String(format: "%.4f", lm.legalEntropy)
                        } else {
                            legalMassStr = "--"
                            top1LegalStr = "--"
                            pEntLegalStr = "--"
                        }
                        // Surface the most-recent batch unique-position
                        // ratio so it scrolls past the eye in the same
                        // line as pEnt / gNorm. Reads NaN until the
                        // first stats-collection batch lands.
                        let bufUniqStr: String
                        let bufUniqPct = trainer.lastBatchStatsUniquePct
                        if !bufUniqPct.isNaN {
                            bufUniqStr = String(format: "%.4f", bufUniqPct)
                        } else {
                            bufUniqStr = "--"
                        }
                        // Per-step timing means over the rolling timing
                        // window. Splits `recentStepMs` into prep/gpu/
                        // read/queueWait/step so a slowdown can be
                        // attributed to one component (CPU prep, GPU
                        // compute, readback marshalling, executor
                        // queue backlog, or unaccounted overhead).
                        let timingStr: String
                        if let stepMs = trainingSnap.recentStepMs {
                            let prep = trainingSnap.recentDataPrepMs ?? 0
                            let gpu = trainingSnap.recentGpuRunMs ?? 0
                            let read = trainingSnap.recentReadbackMs ?? 0
                            let wait = trainingSnap.recentQueueWaitMs ?? 0
                            timingStr = String(
                                format: "step=%.1f gpu=%.1f prep=%.2f read=%.2f wait=%.2f n=%d",
                                stepMs, gpu, prep, read, wait, trainingSnap.recentTimingSamples
                            )
                        } else {
                            timingStr = "n=0"
                        }

                        // Process resident memory + delta since previous
                        // [STATS] tick. The delta lets a reader see
                        // sustained leak rate (e.g. "+8 MB / minute")
                        // at a glance rather than having to diff two
                        // log lines mentally.
                        let rssBytes = DiagSampler.currentResidentBytes()
                        let drss: Int64 = prevRssBytes == 0
                            ? 0
                            : Int64(rssBytes) - Int64(prevRssBytes)
                        prevRssBytes = rssBytes
                        let memStr = String(
                            format: "rss=%.2fGB drss=%+.1fMB",
                            Double(rssBytes) / 1024.0 / 1024.0 / 1024.0,
                            Double(drss) / 1024.0 / 1024.0
                        )

                        // VM region count + IOAccelerator-tagged
                        // subset. Cost ~1-3 ms per call; safe at this
                        // 60 s cadence. Each AGX-mapped GPU buffer
                        // shows up as one IOAccelerator region, so
                        // `vmAccel` growing in step with `trMs` is
                        // direct evidence that the per-`commit` cost
                        // is climbing in the kernel's residency walk.
                        let vm = DiagSampler.currentVMRegionCount()
                        let dvmTotal = prevVmTotal == 0
                            ? 0
                            : Int64(vm.total) - Int64(prevVmTotal)
                        let dvmAccel = prevVmIoAccel == 0
                            ? 0
                            : Int64(vm.ioAccelerator) - Int64(prevVmIoAccel)
                        prevVmTotal = vm.total
                        prevVmIoAccel = vm.ioAccelerator
                        let vmStr = String(
                            format: "total=%u dtotal=%+d ioAccel=%u dioAccel=%+d",
                            vm.total, dvmTotal, vm.ioAccelerator, dvmAccel
                        )

                        // Trainer feed-cache size. Stable at 1 means
                        // the trainer is calling MPSGraph with one
                        // feed shape (so any pipeline accumulation in
                        // MPSGraph is not because we're varying our
                        // inputs).
                        let shapesStr = "feedCache=\(trainer.feedCacheCount)"

                        let line = "[STATS] elapsed=\(elapsedStr) steps=\(trainingSnap.stats.steps) spGames=\(parallelSnap.selfPlayGames) spMoves=\(parallelSnap.selfPlayPositions) \(gameLenStr) buffer=\(bufCount)/\(bufCap) pLoss=\(policyStr) pLossWin=\(pLossWinStr) pLossLoss=\(pLossLossStr) vLoss=\(valueStr) pEnt=\(entropyStr) pIllM=\(illegalPenaltyStr) gNorm=\(gradNormStr) vNorm=\(vNormStr) μ=\(muStr) pwNorm=\(pwNormStr) pLogitAbsMax=\(pLogitMaxStr) playedMoveProb=\(playedProbStr) playedMoveProbPosAdv=\(playedProbPosStr) playedMoveProbNegAdv=\(playedProbNegStr) legalMass=\(legalMassStr) top1Legal=\(top1LegalStr) pEntLegal=\(pEntLegalStr) vMean=\(vMeanStr) vAbs=\(vAbsStr) vBaseDelta=\(vBaseDeltaStr) adv=(\(advStr)) sp.tau=\(spTau) ar.tau=\(arTau) diversity=\(divStr) ratio=(\(ratioStr)) outcomes=(\(outcomeStr)) bufUniq=\(bufUniqStr) \(cfgStr) reg=(\(regStr)) timing=(\(timingStr)) mem=(\(memStr)) vm=(\(vmStr)) shapes=(\(shapesStr)) build=\(BuildInfo.buildNumber) trainer=\(trainerID) champion=\(championID)"
                        SessionLogger.shared.log(line)

                        // CLI `--output` capture: one StatsLine per
                        // `[STATS]` log emit. All values come from the
                        // same thread-safe snapshots the log line just
                        // used, so the JSON and the log stay perfectly
                        // consistent. No-op when `recorder == nil`.
                        if let recorder {
                            let entry = CliTrainingRecorder.StatsLine(
                                elapsedSec: elapsedTarget,
                                steps: trainingSnap.stats.steps,
                                selfPlayGames: parallelSnap.selfPlayGames,
                                positionsTrained: parallelSnap.selfPlayPositions,
                                avgLen: lifetimeAvgLen,
                                rollingAvgLen: rollingAvgLen,
                                gameLenP50: parallelSnap.gameLenP50,
                                gameLenP95: parallelSnap.gameLenP95,
                                bufferCount: bufCount,
                                bufferCapacity: bufCap,
                                policyLoss: trainingSnap.rollingPolicyLoss,
                                valueLoss: trainingSnap.rollingValueLoss,
                                policyEntropy: trainingSnap.rollingPolicyEntropy,
                                policyIllegalMassPenalty: trainingSnap.rollingIllegalMassPenalty,
                                gradGlobalNorm: trainingSnap.rollingGradGlobalNorm,
                                policyHeadWeightNorm: trainingSnap.rollingPolicyHeadWeightNorm,
                                policyLogitAbsMax: trainingSnap.rollingPolicyLogitAbsMax,
                                playedMoveProb: trainingSnap.rollingPlayedMoveProb,
                                playedMoveProbPosAdv: trainingSnap.rollingPlayedMoveProbPosAdv,
                                playedMoveProbPosAdvSkipped: trainingSnap.rollingPlayedMoveProbPosAdvSkipped,
                                playedMoveProbNegAdv: trainingSnap.rollingPlayedMoveProbNegAdv,
                                playedMoveProbNegAdvSkipped: trainingSnap.rollingPlayedMoveProbNegAdvSkipped,
                                playedMoveCondWindowSize: trainingSnap.rollingPlayedMoveCondWindowSize,
                                legalMass: legalMassOverride.map { Double($0.legalMass) },
                                top1LegalFraction: legalMassOverride.map { Double($0.top1LegalFraction) },
                                legalEntropy: legalMassOverride.map { Double($0.legalEntropy) },
                                policyLossWin: trainingSnap.rollingPolicyLossWin,
                                policyLossLoss: trainingSnap.rollingPolicyLossLoss,
                                batchStats: trainer.lastBatchStatsSummary.map { s in
                                    CliTrainingRecorder.BatchStatsSnapshot(
                                        step: s.step,
                                        batchSize: s.batchSize,
                                        uniqueCount: s.uniqueCount,
                                        uniquePct: s.uniquePct,
                                        dupMax: s.dupMax,
                                        dupDistribution: Dictionary(uniqueKeysWithValues:
                                            s.dupDistribution.map { (String($0.key), $0.value) }
                                        ),
                                        phaseByPlyHistogram: s.phaseByPlyHistogram,
                                        phaseByMaterialHistogram: s.phaseByMaterialHistogram,
                                        gameLengthHistogram: s.gameLengthHistogram,
                                        samplingTauHistogram: s.samplingTauHistogram,
                                        workerIdHistogram: s.workerIdHistogram,
                                        outcomeHistogram: s.outcomeHistogram,
                                        phaseByPlyXOutcomeHistogram: s.phaseByPlyXOutcomeHistogram,
                                        bufferUniquePositions: s.bufferUniquePositions,
                                        bufferStoredCount: s.bufferStoredCount
                                    )
                                },
                                valueMean: trainingSnap.rollingValueMean,
                                valueAbsMean: trainingSnap.rollingValueAbsMean,
                                vBaselineDelta: trainingSnap.rollingVBaselineDelta,
                                advMean: trainingSnap.rollingAdvMean,
                                advStd: trainingSnap.rollingAdvStd,
                                advMin: trainingSnap.rollingAdvMin,
                                advMax: trainingSnap.rollingAdvMax,
                                advFracPositive: trainingSnap.rollingAdvFracPositive,
                                advFracSmall: trainingSnap.rollingAdvFracSmall,
                                advP05: trainingSnap.advantageP05,
                                advP50: trainingSnap.advantageP50,
                                advP95: trainingSnap.advantageP95,
                                spStartTau: Double(spSched.startTau),
                                spFloorTau: Double(spSched.floorTau),
                                spDecayPerPly: Double(spSched.decayPerPly),
                                arStartTau: Double(arSched.startTau),
                                arFloorTau: Double(arSched.floorTau),
                                arDecayPerPly: Double(arSched.decayPerPly),
                                diversityUniqueGames: divSnap.uniqueGames,
                                diversityGamesInWindow: divSnap.gamesInWindow,
                                diversityUniquePercent: divSnap.uniquePercent,
                                diversityAvgDivergencePly: divSnap.avgDivergencePly,
                                ratioTarget: ratioSnap.targetRatio,
                                ratioCurrent: ratioSnap.currentRatio,
                                ratioProductionRate: ratioSnap.productionRate,
                                ratioConsumptionRate: ratioSnap.consumptionRate,
                                selfPlayMovesPerHour: spMovesPerHour,
                                trainingMovesPerHour: trainMovesPerHour,
                                ratioAutoAdjust: ratioSnap.autoAdjust,
                                ratioComputedDelayMs: ratioSnap.computedDelayMs,
                                whiteCheckmates: parallelSnap.whiteCheckmates,
                                blackCheckmates: parallelSnap.blackCheckmates,
                                stalemates: parallelSnap.stalemates,
                                fiftyMoveDraws: parallelSnap.fiftyMoveDraws,
                                threefoldRepetitionDraws: parallelSnap.threefoldRepetitionDraws,
                                insufficientMaterialDraws: parallelSnap.insufficientMaterialDraws,
                                batchSize: sessionTrainingBatchSize,
                                learningRate: Double(lr),
                                promoteThreshold: sessionPromoteThreshold,
                                arenaGames: sessionTournamentGames,
                                workerCount: workerN,
                                gradClipMaxNorm: Double(gradClip),
                                weightDecayC: Double(weightDec),
                                entropyRegularizationCoeff: Double(entropyCoeff),
                                drawPenalty: Double(drawPen),
                                policyLossWeight: Double(policyW),
                                valueLossWeight: Double(valueW),
                                buildNumber: BuildInfo.buildNumber,
                                trainerID: trainerID,
                                championID: championID
                            )
                            recorder.appendStats(entry)
                        }

                        // Policy-entropy alarm: fires whenever the
                        // rolling entropy (computed over the training
                        // stats window, same as logged above) is below
                        // the threshold. Co-located with the [STATS]
                        // emit so the cadence matches — the log
                        // adjacent lines always tell a consistent
                        // story. Skipped if entropy isn't yet
                        // available (training hasn't started).
                        if let entropy = trainingSnap.rollingPolicyEntropy,
                           entropy < TrainingAlarmController.policyEntropyAlarmThreshold {
                            SessionLogger.shared.log(
                                "[ALARM] policy entropy \(String(format: "%.4f", entropy)) < \(String(format: "%.2f", TrainingAlarmController.policyEntropyAlarmThreshold)) — policy may be collapsing (steps=\(trainingSnap.stats.steps))"
                            )
                        }
                    }

                    // Cache the most recent legalMass probe result so
                    // we can include it in back-to-back per-step emits
                    // without paying the ~5-20 ms forward-pass cost on
                    // every single step. Refreshed every
                    // `legalMassBootstrapStride` steps during bootstrap
                    // and every time-based emit afterward.
                    var lastLegalMass: ChessTrainer.LegalMassSnapshot? = nil
                    var lastEmittedStep: Int = -1
                    var bootstrapDone = false

                    // Bootstrap phase: poll at short interval, emit one
                    // line per new training step until
                    // bootstrapSteps steps have been logged.
                    while !Task.isCancelled && !bootstrapDone {
                        let trainingSnap = await box.snapshot()
                        let steps = trainingSnap.stats.steps
                        if steps > lastEmittedStep && steps > 0 {
                            // Refresh legalMass on a stride so
                            // back-to-back per-step emits share the
                            // most recent probe.
                            if steps == 1 || (steps - max(0, lastEmittedStep)) >= legalMassBootstrapStride {
                                if buffer.count >= legalMassSampleSize, let probeNet = probeInferenceForProbes {
                                    do {
                                        lastLegalMass = try await trainer.legalMassSnapshot(
                                            replayBuffer: buffer,
                                            sampleSize: legalMassSampleSize,
                                            inferenceNetwork: probeNet
                                        )
                                        // Mirror to @State so the
                                        // chart-sample heartbeat can
                                        // render the legal-entropy
                                        // trace at its own cadence.
                                        let snap = lastLegalMass
                                        await MainActor.run {
                                            self.realLastLegalMassSnapshot = snap
                                        }
                                    } catch {
                                        lastLegalMass = nil
                                        SessionLogger.shared.log(
                                            "[STATS-ERR] legalMassSnapshot failed at step \(steps): \(error.localizedDescription)"
                                        )
                                    }
                                }
                            }
                            let elapsed = Date().timeIntervalSince(sessionStart)
                            await logOne(elapsedTarget: elapsed, legalMassOverride: lastLegalMass)
                            lastEmittedStep = steps
                            if steps >= bootstrapSteps {
                                bootstrapDone = true
                                break
                            }
                        }
                        // Short poll — a training step at typical
                        // throughput completes in 50-200 ms, and we
                        // want the per-step cadence to track closely.
                        do {
                            try await Task.sleep(for: .milliseconds(50))
                        } catch {
                            return
                        }
                    }
                    if Task.isCancelled { return }

                    // Steady-state: one emit every `steadyInterval`
                    // seconds. Each emit refreshes the legalMass probe
                    // too.
                    while !Task.isCancelled {
                        do {
                            try await Task.sleep(for: .seconds(steadyInterval))
                        } catch {
                            return
                        }
                        if Task.isCancelled { return }
                        if buffer.count >= legalMassSampleSize, let probeNet = probeInferenceForProbes {
                            let trainingSnap = await box.snapshot()
                            let steps = trainingSnap.stats.steps
                            do {
                                lastLegalMass = try await trainer.legalMassSnapshot(
                                    replayBuffer: buffer,
                                    sampleSize: legalMassSampleSize,
                                    inferenceNetwork: probeNet
                                )
                                // Mirror to @State so the chart-sample
                                // heartbeat sees a fresh legal-entropy
                                // value at the steady-state cadence.
                                let snap = lastLegalMass
                                await MainActor.run {
                                    self.realLastLegalMassSnapshot = snap
                                }
                            } catch {
                                lastLegalMass = nil
                                SessionLogger.shared.log(
                                    "[STATS-ERR] legalMassSnapshot failed at step \(steps): \(error.localizedDescription)"
                                )
                            }
                        }
                        let elapsed = Date().timeIntervalSince(sessionStart)
                        await logOne(elapsedTarget: elapsed, legalMassOverride: lastLegalMass)
                    }
                }

                // Headless CLI deadline. Armed whenever the run
                // started with `--train` AND `training_time_limit`
                // is set — the time limit is the primary driver of
                // output generation and exit, regardless of
                // whether `--output <file>` was supplied. When
                // `--output` is present the JSON snapshot is
                // written to that file (overwriting); when absent
                // the snapshot goes to stdout so the user can
                // redirect or pipe it. Outside of `--train` mode
                // (interactive use) the deadline is ignored — the
                // user is driving the UI and an unexpected
                // termination would be hostile.
                //
                // On wake-up: log the deadline event, emit the
                // recorder's snapshot to the configured sink, then
                // call `Darwin._exit(0)`. The crucial detail is
                // `_exit` vs `exit`: `exit` runs C++ atexit
                // handlers (including CoreAnalytics' exit barrier)
                // while the MPS self-play worker is still mid-
                // `graph.run` on its dedicated serial dispatch
                // queue — the handler tears down global state
                // that `MPSGraphOSLog` still reads, producing
                // EXC_BAD_ACCESS at 0x8 inside the MPSGraph
                // executable run path. `_exit` terminates the
                // process immediately without running atexit or
                // stdio cleanup, which sidesteps the race. It's
                // safe because:
                //   - snapshot file writes use `Data.write(atomic:)`
                //     which fsyncs before rename,
                //   - stdout writes go through `FileHandle`
                //     which is an unbuffered syscall (no libc
                //     stdio buffer to flush),
                //   - session log writes have already been
                //     flushed by SessionLogger before this point.
                if isAutoTrainRun, let recorder, let deadlineSec = cliTrainingTimeLimitSec, deadlineSec > 0 {
                    group.addTask(priority: .utility) {
                        do {
                            try await Task.sleep(for: .seconds(deadlineSec))
                        } catch {
                            // Cancelled before the deadline hit —
                            // session ended for another reason.
                            return
                        }
                        if Task.isCancelled { return }
                        let elapsed = Date().timeIntervalSince(runStart)
                        let destDescription = outputURL?.path ?? "<stdout>"
                        SessionLogger.shared.log(
                            "[APP] --train: training_time_limit=\(deadlineSec)s reached at elapsed=\(String(format: "%.1f", elapsed))s; writing snapshot to \(destDescription)"
                        )
                        recorder.setTerminationReason(.timerExpired)
                        let counts = recorder.countsSnapshot()
                        do {
                            if let outputURL {
                                try recorder.writeJSON(to: outputURL, totalTrainingSeconds: elapsed)
                            } else {
                                try recorder.writeJSONToStdout(totalTrainingSeconds: elapsed)
                            }
                            SessionLogger.shared.log(
                                "[APP] --train: wrote snapshot to \(destDescription) (arenas=\(counts.arenas), stats=\(counts.stats), probes=\(counts.probes))"
                            )
                        } catch {
                            SessionLogger.shared.log(
                                "[APP] --train: snapshot write FAILED for \(destDescription): \(error.localizedDescription)"
                            )
                        }
                        SessionLogger.shared.log("[APP] --train: exiting process after snapshot")
                        Darwin._exit(0)
                    }
                }

                // Legal-mass collapse detector. Runs in every session
                // (interactive or --train), probes at 60 s cadence
                // with a 120 s grace period measured from the first
                // observed SGD step (i.e. earliest possible detection
                // is training_start + 120 s), flags the collapse
                // banner once the network has shown
                // `illegalMass > 0.99` for `collapseProbesToAbort`
                // consecutive probes. In interactive mode we stop at
                // the banner — the user sees it and decides what to
                // do. In --train mode we ALSO write the snapshot with
                // `termination_reason = legal_mass_collapse` and exit
                // the process (status 0 — a clean early abort, not a
                // crash; scripting distinguishes via the JSON field
                // rather than exit code per the user's spec).
                // Local "session start" for the collapse detector.
                // The --train timer task uses its own `runStart`;
                // both are set at roughly the same moment in the
                // task-group setup (drift < 1 ms) and the two are
                // only used for "elapsed since session start"
                // display in logs/alarms, so a separate declaration
                // is intentional — it keeps the detector
                // self-contained and doesn't require plumbing the
                // timer task's runStart into this closure.
                let collapseRunStart = Date()
                let collapseRecorder: CliTrainingRecorder? = (isAutoTrainRun ? recorder : nil)
                let collapseOutputURL: URL? = outputURL
                // Configuration snapshot taken at task start. All three
                // are user-tunable via parameters JSON / @AppStorage; we
                // capture once so the running detector has stable behavior
                // even if the user edits the values mid-session (changes
                // take effect on the next training run).
                let collapseIllegalMassThreshold = TrainingParameters.shared.legalMassCollapseThreshold
                let collapseGracePeriodSec = TrainingParameters.shared.legalMassCollapseGraceSeconds
                let collapseNoImprovementProbeCount = max(1, TrainingParameters.shared.legalMassCollapseNoImprovementProbes)
                group.addTask(priority: .utility) {
                    [trainer, buffer, box, probeInferenceForProbes] in
                    let probeIntervalSec: UInt64 = 60
                    let sampleSize = 128
                    let gracePeriodSec: TimeInterval = collapseGracePeriodSec
                    let illegalMassThreshold: Double = collapseIllegalMassThreshold
                    let noImprovementProbeCount = collapseNoImprovementProbeCount
                    // Sliding window of recent legal-mass readings,
                    // newest at end. Trip condition: window is full,
                    // every reading's illegalMass is above threshold,
                    // AND the newest legal_mass is no better than the
                    // oldest (no upward improvement across the window).
                    // This catches *stuck* collapse — slow climbs out
                    // of near-uniform won't fire even if absolute
                    // legal mass is still below threshold for a while.
                    var legalMassWindow: [Double] = []
                    var aborted = false
                    // Anchor for the grace countdown. Lazily set the
                    // first time we observe at least one completed SGD
                    // step. The replay buffer has to fill enough for
                    // training to begin (minutes in practice), and
                    // before that first step lands, every probe on the
                    // fresh random-init network will naturally see
                    // illegalMass ≈ 0.994 — firing the alarm during
                    // that window is a false positive. Grace is 120 s
                    // measured from TRAINING start, not session start.
                    var trainingStartAt: Date? = nil
                    while !Task.isCancelled && !aborted {
                        do {
                            try await Task.sleep(for: .seconds(probeIntervalSec))
                        } catch {
                            return
                        }
                        if Task.isCancelled { return }
                        // Training-start gate. If no SGD steps have
                        // landed yet, reset the anchor and skip — the
                        // network is still at random init so no
                        // learning-progress signal exists. Once steps
                        // > 0, stamp the anchor on the first qualifying
                        // iteration and measure grace from there.
                        let trainingSteps = await box.snapshot().stats.steps
                        guard trainingSteps > 0 else {
                            trainingStartAt = nil
                            continue
                        }
                        if trainingStartAt == nil {
                            trainingStartAt = Date()
                        }
                        guard let startAt = trainingStartAt else { continue }
                        let trainingElapsed = Date().timeIntervalSince(startAt)
                        if trainingElapsed < gracePeriodSec { continue }
                        // Session-elapsed for snapshot-write / log
                        // consistency with the timer task, which also
                        // reports `totalTrainingSeconds` as
                        // session-elapsed rather than training-elapsed.
                        let elapsed = Date().timeIntervalSince(collapseRunStart)
                        // Skip the probe if the inference network isn't
                        // available — the pollution-prone trainer-network
                        // fallback would silently corrupt BN running stats
                        // and defeat the reason we switched detectors off
                        // the trainer network in the first place.
                        guard let probeNet = probeInferenceForProbes else { continue }

                        let snap: ChessTrainer.LegalMassSnapshot?
                        do {
                            snap = try await trainer.legalMassSnapshot(
                                replayBuffer: buffer,
                                sampleSize: sampleSize,
                                inferenceNetwork: probeNet
                            )
                        } catch {
                            SessionLogger.shared.log(
                                "[STATS-ERR] collapse-detector legalMassSnapshot failed: \(error.localizedDescription)"
                            )
                            continue
                        }
                        guard let snap else { continue }
                        let illegalMass = 1.0 - snap.legalMass
                        legalMassWindow.append(snap.legalMass)
                        if legalMassWindow.count > noImprovementProbeCount {
                            legalMassWindow.removeFirst()
                        }
                        // Window-full tripwire: we need
                        // `noImprovementProbeCount` samples before
                        // declaring "no improvement"; until then we
                        // log the probe and continue.
                        let windowFull = legalMassWindow.count >= noImprovementProbeCount
                        let allAboveThreshold: Bool = {
                            guard windowFull else { return false }
                            for legalMass in legalMassWindow where (1.0 - legalMass) <= illegalMassThreshold {
                                return false
                            }
                            return true
                        }()
                        let noImprovement: Bool = {
                            guard windowFull,
                                  let oldest = legalMassWindow.first,
                                  let newest = legalMassWindow.last else {
                                return false
                            }
                            return newest <= oldest
                        }()
                        let probeMatches = allAboveThreshold && noImprovement
                        if probeMatches {
                            SessionLogger.shared.log(
                                String(format: "[ALARM] legal-mass collapse window: %d/%d probes illegalMass>%.4f, newest legalMass=%.5f ≤ oldest=%.5f",
                                    legalMassWindow.count, noImprovementProbeCount,
                                    illegalMassThreshold,
                                    legalMassWindow.last ?? 0, legalMassWindow.first ?? 0)
                            )
                            await MainActor.run {
                                self.trainingAlarm?.raise(
                                    severity: .critical,
                                    title: "Policy Collapse (legal mass)",
                                    detail: String(format: "illegalMass>%.4f for %d probes with no improvement (legal_mass %.5f→%.5f)",
                                        illegalMassThreshold, legalMassWindow.count,
                                        legalMassWindow.first ?? 0, legalMassWindow.last ?? 0)
                                )
                            }
                        } else {
                            SessionLogger.shared.log(
                                String(format: "[ALARM] legal-mass probe ok: legalMass=%.5f illegalMass=%.4f window=%d/%d",
                                    snap.legalMass, illegalMass,
                                    legalMassWindow.count, noImprovementProbeCount)
                            )
                        }

                        if probeMatches {
                            aborted = true
                            SessionLogger.shared.log(
                                String(format: "[ALARM] legal-mass collapse confirmed: %d probes above %.4f with no improvement (legal_mass %.5f→%.5f) — aborting",
                                    legalMassWindow.count, illegalMassThreshold,
                                    legalMassWindow.first ?? 0, legalMassWindow.last ?? 0)
                            )
                            if let rec = collapseRecorder {
                                let destDescription = collapseOutputURL?.path ?? "<stdout>"
                                SessionLogger.shared.log(
                                    "[APP] --train: legal-mass collapse abort at elapsed=\(String(format: "%.1f", elapsed))s; writing snapshot to \(destDescription)"
                                )
                                rec.setTerminationReason(.legalMassCollapse)
                                let counts = rec.countsSnapshot()
                                do {
                                    if let url = collapseOutputURL {
                                        try rec.writeJSON(to: url, totalTrainingSeconds: elapsed)
                                    } else {
                                        try rec.writeJSONToStdout(totalTrainingSeconds: elapsed)
                                    }
                                    SessionLogger.shared.log(
                                        "[APP] --train: wrote snapshot to \(destDescription) (arenas=\(counts.arenas), stats=\(counts.stats), probes=\(counts.probes))"
                                    )
                                } catch {
                                    SessionLogger.shared.log(
                                        "[APP] --train: snapshot write FAILED for \(destDescription): \(error.localizedDescription)"
                                    )
                                }
                                SessionLogger.shared.log("[APP] --train: exiting process after collapse-abort snapshot")
                                Darwin._exit(0)
                            }
                            // Interactive session: alarm already
                            // raised, loop stays exited so we stop
                            // probing (no value continuing once a
                            // confirmed collapse has been flagged).
                        }
                    }
                }

                // Wait for all four tasks to complete (only happens
                // on cancellation since each loops forever).
                for await _ in group { }
            }

            await MainActor.run {
                trainingAlarm?.clear()
                realTraining = false
                realTrainingTask = nil
                isArenaRunning = false
                arenaActiveFlag = nil
                arenaTriggerBox = nil
                arenaOverrideBox = nil
                parallelWorkerStatsBox = nil
                parallelStats = nil
                chartCoordinator?.setDiversityHistogramBars([])
                chartCoordinator?.arenaChartEvents = []
                chartCoordinator?.cancelActiveArena()
                workerCountBox = nil
                samplingScheduleBox = nil
                activeSelfPlayGate = nil
                activeTrainingGate = nil
                checkpoint?.currentSessionID = nil
                checkpoint?.currentSessionStart = nil
                replayRatioController = nil
                replayRatioSnapshot = nil
                effectiveReplayRatioTarget = nil
                lastReplayRatioCompensatorAt = nil
                cliRecorder = nil
                EarlyStopCoordinator.shared.earlyStopHandler = nil
            }
        }
    }

    func stopRealTraining() {
        realTrainingTask?.cancel()
        realTrainingTask = nil
        trainingAlarm?.clear()
        // Close the in-progress training segment so cumulative wall-time
        // totals exclude post-Stop idle. If saving immediately after,
        // buildCurrentSessionState will see the segment already closed
        // and won't double-count.
        checkpoint?.closeActiveTrainingSegment(reason: "stop")
        // Disarm the periodic-save scheduler so a Stop-then-Start
        // doesn't fire an immediate save on Start. The next Start
        // constructs a fresh controller with a fresh 4-hour
        // deadline.
        periodicSaveController?.disarm()
        periodicSaveController = nil
        periodicSaveLastPollAt = nil
        periodicSaveInFlight = false
    }

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
