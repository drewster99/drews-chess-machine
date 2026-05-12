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
