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
    /// `UpperContentView.clearTrainingDisplay()` — called on Build (and the
    /// auto-build) because rebuilding invalidates the trainer's graph state.
    var onClearTrainingDisplay: () -> Void = { }

    /// Drops the trainer (it owns graph state invalidated by a rebuild). Wired
    /// to `{ trainer = nil }` on the view until the trainer migrates here.
    var onDropTrainer: () -> Void = { }

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
