import SwiftUI
import OSLog

/// Trace logger for the per-stage `elap(_:)` probe in `__processSnapshotTimerTick`.
/// Routed through `os_log` (not `SessionLogger`) so the ~13 lines/sec it emits
/// land in the unified-logging stream rather than the session text file — they're
/// stall-attribution breadcrumbs, not session telemetry.
private let logger = Logger(subsystem: "com.drewben.DrewsChessMachine", category: "Heartbeat")

/// `SessionController`'s heartbeat — split out of `SessionController.swift`.
/// Driven by `UpperContentView`'s `.onReceive(snapshotTimer)` once per tick;
/// `processSnapshotTimerTick` is the throttled wrapper, `__processSnapshotTimerTick`
/// the body that mirrors every cross-thread state box (game / sweep / training /
/// arena / parallel-worker counters / replay-ratio controller / diversity tracker)
/// into the matching `@State` (or session var), plus the periodic-save poll, the
/// chart-zoom state machine, and the memory/usage refreshers. The re-entrancy
/// guard `snapshotTickInFlight` + `snapshotTickLastLogAt` are stored on
/// `SessionController` (extensions can't hold stored properties).
extension SessionController {

    // MARK: - Heartbeat

    nonisolated static let memoryStatsRefreshSec: Double = 10
    nonisolated static let usageStatsRefreshSec: Double = 5
    nonisolated static let progressRateRefreshSec: Double = 1.0
    nonisolated static let progressRateWindowSec: Double = 60.0

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
        // it with one line per heartbeat. Emits at most once per
        // `Self.snapshotTickLogIntervalSec` and ALWAYS on any tick
        // whose tick body or main-actor enqueue wait exceeds the
        // alarm threshold — those are the events worth seeing even
        // if a recent throttled emit just landed.
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

    /// Cadence for the periodic snapshot-tick log emit. Long enough to
    /// keep the log file readable at steady state, short enough to
    /// confirm the heartbeat is still alive; the anomaly threshold
    /// (`snapshotTickAlarmMs`) makes sure any genuine stall surfaces
    /// promptly between the periodic emits regardless.
    nonisolated static let snapshotTickLogIntervalSec: TimeInterval = 30

    /// Threshold above which a snapshot-tick wall or its main-actor
    /// enqueue wait is logged immediately (out-of-band) rather than
    /// waiting for the next periodic emit. A tick body or main-actor
    /// wait this long is a perceptible UI hitch regardless of how
    /// often the heartbeat fires — worth surfacing on its own.
    nonisolated static let snapshotTickAlarmMs: Double = 50
    /// Body of the heartbeat. Pulls every cross-thread state
    /// box (game, sweep, training, arena, parallel-worker counters,
    /// replay-ratio controller, diversity tracker) into the matching
    /// `@State`, throttled internally where each consumer cares about
    /// avoiding redundant invalidations. Extracted out of the inline
    /// `.onReceive(snapshotTimer)` closure so `body`'s expression
    /// type-check stays cheap — the closure used to be ~140 lines and
    /// dragged the whole modifier chain past the
    /// `-warn-long-expression-type-checking` budget.
    private func __processSnapshotTimerTick() async {
        // Pull the latest game state into @State once per heartbeat.
        // Cheap (single locked struct copy) and bounds UI work even
        // when the game loop is doing hundreds of moves per second.
        //
        // `elap(_:)` is the per-stage trace probe used to attribute a
        // UI stall to a specific block of this tick. Cheap enough to
        // leave on at the heartbeat cadence — `os_log` is hundreds of
        // nanoseconds per emit, negligible against the work the tick
        // body itself does.
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
        // Replay-buffer composition mirror + per-batch sampling-constraint
        // push. The buffer owns its `SamplingConstraints` (so the off-main
        // trainer never reads `TrainingParameters.shared`); we re-push the
        // current values every tick — cheap (one lock op) and idempotent.
        // The composition snapshot is dirty-checked so @State only churns
        // when the resident set actually changed.
        if let buf = replayBuffer {
            buf.setSamplingConstraints(.fromCurrentParameters())
            let comp = buf.compositionSnapshot()
            if comp != bufferComposition { bufferComposition = comp }
            // Mirror the latest per-batch achievement report into @State
            // for the popover's "Last sampled batch" readout. The
            // `.uninitialized` sentinel (no batch yet this session) is
            // surfaced as `nil` on the controller so the popover UI can
            // collapse the column to dashes without inspecting fields.
            let sr = buf.lastSamplingResult()
            let mirrored: ReplayBuffer.SamplingResult? = sr.didSample ? sr : nil
            if mirrored != lastSamplingResult { lastSamplingResult = mirrored }
        } else {
            if bufferComposition != nil { bufferComposition = nil }
            if lastSamplingResult != nil { lastSamplingResult = nil }
        }
        elap("after 6b")
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
        // Progress-rate chart sampler — runs during Play and Train;
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
    /// whether to fire. Throttled — a multi-hour deadline doesn't
    /// benefit from being re-checked on every heartbeat, and the
    /// throttle keeps the heartbeat hot path from paying a decision
    /// cost per tick. A
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
    /// from the heartbeat. The actual sampling reads `task_info`
    /// and a couple of `MTLDevice` properties via the trainer's
    /// existing helpers.
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
    /// `progressRateRefreshSec` during a Play and Train.
    /// The moves/hr fields are computed over a real trailing
    /// `progressRateWindowSec` window: we walk backward from the
    /// newest stored sample until we find the first one whose
    /// `timestamp` is still inside the window, then subtract its
    /// cumulative counters from the current cumulative counters
    /// and divide by the actual elapsed seconds between the two
    /// samples.
    ///
    /// Before the session has a full window of history, the window
    /// shrinks gracefully to "whatever we have" — the first
    /// sample reports zero (no earlier sample to subtract from),
    /// the second reports over ~1 s, and so on until the window
    /// reaches its full width.
    ///
    /// No-op outside of `realTraining`. Sampler state is cleared
    /// by `startRealTraining()` so each session's chart starts
    /// fresh from t=0.
    /// Sample training metrics on the same cadence as the progress
    /// rate sampler. Appends a `TrainingChartSample` with rolling
    /// loss, entropy, ratio, and non-neg count.
    /// Append a training chart sample. Called from inside
    /// `refreshProgressRateIfNeeded` on the same cadence.
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
            rollingValueMean: trainingSnap?.rollingValueMean,
            rollingValueAbsMean: trainingSnap?.rollingValueAbsMean,
            rollingValueProbWin: trainingSnap?.rollingValueProbWin,
            rollingValueProbDraw: trainingSnap?.rollingValueProbDraw,
            rollingValueProbLoss: trainingSnap?.rollingValueProbLoss,
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
        // function does (cumulative-counter deltas use the 1-minute
        // window timestamps, not elapsedSec).
        let elapsed = max(0, now.timeIntervalSince(chartCoordinator?.chartElapsedAnchor ?? Date()))
        let curSp = pStats.selfPlayPositions
        let curTr = (trainingStats?.steps ?? 0) * TrainingParameters.shared.trainingBatchSize

        // Walk newest → oldest through the coordinator's ring,
        // recording the last sample we see that still falls inside
        // the rolling window. Breaks out as soon as we hit a
        // sample older than the cutoff — the ring is timestamp-
        // sorted, so anything older is also out of window. Bounded
        // at `progressRateWindowSec / progressRateRefreshSec`
        // iterations per call in steady state regardless of total
        // session length (~60 at the current 60s window / 1s refresh).
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
        // Append a training chart sample on the same cadence.
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
}
