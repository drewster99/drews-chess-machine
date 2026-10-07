import Foundation

/// One line a monitor hands its caller's log sink. `eventKind` is set for an
/// `[ALARM] health …` event line (the CLI paths also write raise, escalate
/// and stop lines to stderr) and nil for a `[HEALTH] …` line.
struct TrainingHealthLogLine: Sendable, Equatable {
    let text: String
    let eventKind: TrainingHealthEvent.Kind?
}

/// Where a monitor writes its lines. Called under the monitor's evaluation
/// lock, so lines arrive in evaluation order; it must not block (the GUI
/// sink is `SessionLogger.shared.log`, a non-blocking enqueue; the CLI sink
/// is the runner's `emit`).
typealias TrainingHealthLogSink = @Sendable (TrainingHealthLogLine) -> Void

/// The layer-health input of one live evaluation.
enum TrainingHealthLiveLayerHealth: Sendable {
    /// The live read succeeded: its digest and the trainer clock it was read
    /// at (`LayerHealthLiveState.completedTrainSteps`), which bounds the
    /// window so the window and the digest describe the same weights.
    case read(LayerHealthDigest, trainerStep: Int)
    /// The live read failed (already logged by `LayerHealthLog`): the window
    /// ends at the stamp's last recorded step, the layer-health rules have
    /// no data, and the evaluation says so (`liveReadFailed=`).
    case readFailed
    /// Offline replay only: the log has no live line for this row. Like
    /// `readFailed` for the rules, but not counted as a failed read.
    case absentFromLog
}

/// The training-health observer of one run (CLI process; GUI Play-and-Train
/// start): per-step records, serialized evaluations through the one
/// `TrainingHealthEvaluator`, trainer-clock rewind handling, and the
/// `[HEALTH]` / `[ALARM] health` lines. Design: the alarms plan, D2.
///
/// It is an observer and nothing else: no GPU, no random draws, no access to
/// the trainer, the optimizer or the replay buffer. Its inputs are values the
/// paths already hold (`TrainStepTiming`, layer-health summaries); the
/// dedicated value-FC1 read (D6) is made by the path through `ChessTrainer`,
/// and the monitor only decides when it is due and evaluates its result.
///
/// **Locks.** Two `SyncBox`es (`OSAllocatedUnfairLock`):
/// - `steps` — the hot path. `recordStep` takes only this lock, for one
///   append, once per SGD step; it never takes the evaluation lock, so the
///   trainer never waits on an evaluation. The **generation** lives here (its
///   one home): a trainer-clock rewind increments it under this lock, so a
///   stamp taken after a rewind always carries the new generation.
/// - `evaluation` — serializes every evaluation end to end (live and
///   checkpoint), and holds the evaluator, the reference history and the
///   `[HEALTH] check` counters.
///
/// Lock order is always evaluation → steps; nothing takes the evaluation
/// lock while holding the steps lock. An evaluation validates its stamp and
/// extracts its window in one steps-lock section, computes statistics and
/// runs the evaluator on a copy holding only the evaluation lock, then
/// commits in a second steps-lock section only if the generation is still the
/// one it validated — so an evaluation never judges weights a rewind
/// discarded, and never commits against a generation it did not validate.
final class TrainingHealthMonitor: @unchecked Sendable {

    let runID: UUID
    let valueFC1Applicability: TrainingHealthValueFC1Applicability

    // MARK: Step store (hot path)

    struct UnannouncedRewind: Sendable, Equatable {
        let from: Int
        let to: Int
        let generation: Int
    }

    /// The pending window: records since the previous live evaluation, in
    /// record order, with amortized O(1) dropping of the oldest.
    struct PendingRecords: Sendable {
        private(set) var storage: [TrainingHealthStepRecord] = []
        private var start = 0

        var count: Int { storage.count - start }

        mutating func append(_ record: TrainingHealthStepRecord) {
            storage.append(record)
        }

        mutating func dropOldest() {
            start += 1
            compactIfNeeded()
        }

        mutating func removeAll() {
            storage.removeAll(keepingCapacity: true)
            start = 0
        }

        /// Remove and return the records with trainer step ≤ `boundary`;
        /// newer ones stay pending.
        mutating func take(throughTrainerStep boundary: Int) -> [TrainingHealthStepRecord] {
            var end = start
            while end < storage.count, storage[end].trainerStep <= boundary {
                end += 1
            }
            let taken = Array(storage[start..<end])
            start = end
            compactIfNeeded()
            return taken
        }

        private mutating func compactIfNeeded() {
            if start == storage.count {
                storage.removeAll(keepingCapacity: true)
                start = 0
            } else if start > 1024, start * 2 > storage.count {
                storage.removeFirst(start)
                start = 0
            }
        }
    }

    struct StepStore: Sendable {
        var generation = 0
        var pending = PendingRecords()
        var stepsTrainedByThisProcess = 0
        var lastRecordedTrainerStep: Int?
        /// D6's start anchor: the clock this monitor started recording from
        /// (the first record's step − 1), or the restored clock after a
        /// rewind.
        var valueFC1AnchorBase: Int?
        var unannouncedRewinds: [UnannouncedRewind] = []
        /// Since the previous `[HEALTH] check` line.
        var truncatedSinceCheck = 0
        var recordCostSinceCheck: Duration = .zero
        var trainMsSinceCheck: Double = 0
    }

    private let steps = SyncBox(StepStore())

    // MARK: Evaluation state

    struct IntervalCounters: Sendable {
        var live = 0
        var checkpoint = 0
        var stale = 0
        var liveReadFailed = 0
        var cost: Duration = .zero
        var lossMaxRatio: Double?
        var lossMedianRatio: Double?
        var gradientMaxRatio: Double?
        var noData: [TrainingHealthRule: Int] = [:]
    }

    struct RaisedSummary: Sendable {
        var firstTrainerStep: Int
        var highestSeverity: TrainingAlarm.Severity
        var raiseCount: Int
    }

    struct EvaluationState: Sendable {
        var evaluator: TrainingHealthEvaluator
        var appliedGeneration = 0
        /// `(trainerStep, loss, gradient norm)` of committed windows, the
        /// spike rules' reference; at most `spikeReferenceHistoryCapacity`.
        var history: [TrainingHealthReferenceEntry] = []
        var counters = IntervalCounters()
        var committedEvaluations = 0
        var raised: [TrainingHealthRule: RaisedSummary] = [:]
        /// Newest trainer step of a rule-3 observation in this generation
        /// (D6's anchor).
        var newestValueFC1ObservationStep: Int?
        var finalCheckWritten = false
    }

    private let evaluation: SyncBox<EvaluationState>

    /// Test hook: runs inside an evaluation between validation and commit,
    /// holding only the evaluation lock. nil outside tests.
    private let beforeCommitForTesting: (@Sendable () -> Void)?

    // MARK: Init

    /// A monitor whose evaluator decides stops (`.byEvaluator`): the
    /// command-line paths and the offline replay.
    convenience init(valueFC1Applicability: TrainingHealthValueFC1Applicability) {
        self.init(valueFC1Applicability: valueFC1Applicability, stopDecision: .byEvaluator)
    }

    /// `stopDecision` is `.byCaller` for the GUI, which decides stops on the
    /// main actor from the actions in force when each evaluation arrives.
    init(valueFC1Applicability: TrainingHealthValueFC1Applicability, stopDecision: TrainingHealthStopDecision) {
        self.runID = UUID()
        self.valueFC1Applicability = valueFC1Applicability
        self.evaluation = SyncBox(EvaluationState(
            evaluator: TrainingHealthEvaluator(valueFC1Applicability: valueFC1Applicability, stopDecision: stopDecision)))
        self.beforeCommitForTesting = nil
    }

    /// For tests that must interleave a rewind between an evaluation's
    /// validation and its commit.
    init(
        valueFC1Applicability: TrainingHealthValueFC1Applicability,
        beforeCommitForTesting: @escaping @Sendable () -> Void
    ) {
        self.runID = UUID()
        self.valueFC1Applicability = valueFC1Applicability
        self.evaluation = SyncBox(EvaluationState(
            evaluator: TrainingHealthEvaluator(valueFC1Applicability: valueFC1Applicability)))
        self.beforeCommitForTesting = beforeCommitForTesting
    }

    // MARK: Recording (trainer hot path)

    /// Record one SGD step. Takes only the steps lock.
    func recordStep(_ timing: TrainStepTiming, trainerStep: Int) {
        recordStep(TrainingHealthStepRecord(timing: timing, trainerStep: trainerStep))
    }

    /// Record one step. A record at or below the last recorded trainer step
    /// is an unannounced trainer-clock rewind: the step-store half of R0's
    /// rewind happens right here (generation + 1, pending window discarded,
    /// the rewound span taken off the trained count); the evaluation-side
    /// half is applied by the next evaluation.
    func recordStep(_ record: TrainingHealthStepRecord) {
        let clock = ContinuousClock()
        let started = clock.now
        steps.modify { store in
            if let last = store.lastRecordedTrainerStep, record.trainerStep <= last {
                let restored = record.trainerStep - 1
                store.generation += 1
                store.pending.removeAll()
                store.stepsTrainedByThisProcess = max(0, store.stepsTrainedByThisProcess - (last - restored))
                store.valueFC1AnchorBase = restored
                store.unannouncedRewinds.append(
                    UnannouncedRewind(from: last, to: restored, generation: store.generation))
            }
            if store.valueFC1AnchorBase == nil {
                store.valueFC1AnchorBase = record.trainerStep - 1
            }
            store.pending.append(record)
            if store.pending.count > TrainingHealthThresholds.pendingWindowCapacity {
                store.pending.dropOldest()
                store.truncatedSinceCheck += 1
            }
            store.stepsTrainedByThisProcess += 1
            store.lastRecordedTrainerStep = record.trainerStep
            if let ms = record.totalMs { store.trainMsSinceCheck += ms }
            store.recordCostSinceCheck += clock.now - started
        }
    }

    /// The stamp an observation carries, taken before its data is read.
    /// Reads the three scalars inside the lock rather than copying the step
    /// store out: a copy would share the pending window's buffer, and a
    /// `recordStep` append while it is alive would reallocate that whole
    /// buffer (copy-on-write) on the trainer's hot path.
    func observationStamp() -> TrainingHealthStamp {
        let runID = self.runID
        return steps.read { store in
            TrainingHealthStamp(
                runID: runID,
                generation: store.generation,
                stepsTrainedByThisProcess: store.stepsTrainedByThisProcess,
                lastRecordedTrainerStep: store.lastRecordedTrainerStep)
        }
    }

    // MARK: Rewind

    /// An announced trainer-clock rewind (a GUI promotion restores the
    /// arena-start trainer snapshot): both halves of R0's rewind at once.
    /// `restoredClock` is the trainer's completed-step clock after the
    /// rewind; the next record will be `restoredClock + 1`.
    ///
    /// An unannounced rewind `recordStep` detected that no evaluation has
    /// logged yet is logged here first: this rewind's generation supersedes
    /// it, so no later evaluation would see it as pending and its line would
    /// be lost.
    func noteTrainerClockRewind(to restoredClock: Int, log: TrainingHealthLogSink) {
        evaluation.modify { state in
            let rewound = self.steps.mutate { store -> (from: Int?, generation: Int, pending: [UnannouncedRewind]) in
                let pending = store.unannouncedRewinds
                store.unannouncedRewinds.removeAll()
                let last = store.lastRecordedTrainerStep
                store.generation += 1
                store.pending.removeAll()
                if let last {
                    store.stepsTrainedByThisProcess = max(
                        0, store.stepsTrainedByThisProcess - max(0, last - restoredClock))
                }
                store.lastRecordedTrainerStep = restoredClock
                store.valueFC1AnchorBase = restoredClock
                return (last, store.generation, pending)
            }
            Self.applyEvaluationSideRewind(&state, generation: rewound.generation)
            Self.logUnannouncedRewinds(rewound.pending, log: log)
            log(TrainingHealthLogLine(
                text: TrainingHealthLog.rewindLine(from: rewound.from, to: restoredClock, generation: rewound.generation),
                eventKind: nil))
        }
    }

    private static func applyEvaluationSideRewind(_ state: inout EvaluationState, generation: Int) {
        state.evaluator.resetForTrainerClockRewind()
        state.history.removeAll()
        state.newestValueFC1ObservationStep = nil
        state.appliedGeneration = generation
    }

    /// Step 1 of every evaluation: if `recordStep` detected a rewind since
    /// the last one this state applied, apply the evaluation-side half; then
    /// log every detected rewind not yet logged. The reset always applies
    /// (the state must never describe discarded weights, whatever the
    /// config), but a disabled config logs nothing, so its evaluations leave
    /// the lines pending for the first enabled evaluation or announced rewind
    /// — nothing is lost and nothing is written while alarms are off.
    private func applyUnannouncedRewinds(_ state: inout EvaluationState, logging: Bool, log: TrainingHealthLogSink) {
        let detected = steps.mutate { store -> (generation: Int, rewinds: [UnannouncedRewind]) in
            guard logging else { return (store.generation, []) }
            let rewinds = store.unannouncedRewinds
            store.unannouncedRewinds.removeAll()
            return (store.generation, rewinds)
        }
        if detected.generation != state.appliedGeneration {
            Self.applyEvaluationSideRewind(&state, generation: detected.generation)
        }
        Self.logUnannouncedRewinds(detected.rewinds, log: log)
    }

    private static func logUnannouncedRewinds(_ rewinds: [UnannouncedRewind], log: TrainingHealthLogSink) {
        for rewind in rewinds {
            log(TrainingHealthLogLine(
                text: TrainingHealthLog.unannouncedRewindLine(from: rewind.from, to: rewind.to, generation: rewind.generation),
                eventKind: nil))
        }
    }

    // MARK: Live evaluation

    /// Evaluate the steps recorded since the previous live evaluation, up to
    /// the live digest's trainer step (or, when the live read failed or the
    /// log has no live line, the stamp's last recorded step). Returns nil
    /// when nothing was committed: a stale stamp, a rewind during the
    /// evaluation, or alarms disabled.
    @discardableResult
    func evaluateLive(
        stamp: TrainingHealthStamp,
        layerHealth: TrainingHealthLiveLayerHealth,
        learningRate: Double?,
        momentum: Double?,
        config: TrainingHealthConfig,
        log: TrainingHealthLogSink
    ) -> TrainingHealthEvaluation? {
        evaluation.mutate { state in
            self.performLive(
                &state, stamp: stamp, layerHealth: layerHealth, learningRate: learningRate,
                momentum: momentum, config: config, log: log)
        }
    }

    private func performLive(
        _ state: inout EvaluationState,
        stamp: TrainingHealthStamp,
        layerHealth: TrainingHealthLiveLayerHealth,
        learningRate: Double?,
        momentum: Double?,
        config: TrainingHealthConfig,
        log: TrainingHealthLogSink
    ) -> TrainingHealthEvaluation? {
        let clock = ContinuousClock()
        let started = clock.now
        applyUnannouncedRewinds(&state, logging: config.enabled, log: log)

        let digest: LayerHealthDigest?
        let boundary: Int?
        switch layerHealth {
        case .read(let read, let trainerStep):
            digest = read
            boundary = trainerStep
        case .readFailed, .absentFromLog:
            digest = nil
            boundary = stamp.lastRecordedTrainerStep
        }

        // Validate the stamp and extract the window in one steps-lock section.
        let runID = self.runID
        let extracted = steps.mutate { store -> (valid: Bool, currentGeneration: Int, records: [TrainingHealthStepRecord]) in
            guard stamp.runID == runID, stamp.generation == store.generation else {
                return (false, store.generation, [])
            }
            let records = boundary.map { store.pending.take(throughTrainerStep: $0) } ?? []
            return (true, store.generation, records)
        }
        guard extracted.valid else {
            // A disabled config judges, counts and logs nothing; the stale
            // records stay pending either way.
            if config.enabled {
                state.counters.stale += 1
                log(TrainingHealthLogLine(
                    text: TrainingHealthLog.staleObservationLine(
                        tier: .live, trainerStep: boundary, generation: stamp.generation,
                        currentGeneration: extracted.currentGeneration,
                        newestApplied: state.evaluator.newestAppliedTrainerStep,
                        reason: stamp.runID == runID ? nil : "stamp from another monitor"),
                    eventKind: nil))
            }
            state.counters.cost += clock.now - started
            return nil
        }
        if case .readFailed = layerHealth, config.enabled {
            state.counters.liveReadFailed += 1
            log(TrainingHealthLogLine(
                text: TrainingHealthLog.liveReadFailedLine(boundary: boundary),
                eventKind: nil))
        }

        let window = TrainingHealthWindowStatistics.make(records: extracted.records, history: state.history)
        let windowEntries = extracted.records.map {
            TrainingHealthReferenceEntry(trainerStep: $0.trainerStep, loss: $0.loss, gradGlobalNorm: $0.gradGlobalNorm)
        }
        guard config.enabled else {
            // Disabled: the window is drained (it must not grow to the cap)
            // and the reference kept current, but nothing is judged or
            // counted.
            appendHistory(&state, windowEntries)
            state.counters.cost += clock.now - started
            return nil
        }
        guard let observationTrainerStep = boundary ?? window.lastTrainerStep else {
            // No trainer clock at all: nothing recorded and no digest.
            state.counters.cost += clock.now - started
            return nil
        }
        let observation = TrainingHealthObservation(
            trainerStep: observationTrainerStep, window: window, layerHealth: digest, stamp: stamp,
            effectiveLearningRate: learningRate, effectiveMomentum: momentum)
        var candidate = state.evaluator
        let result = candidate.evaluate(observation, config: config)

        beforeCommitForTesting?()

        let commitGeneration = steps.read { $0.generation }
        guard commitGeneration == stamp.generation else {
            state.counters.stale += 1
            log(TrainingHealthLogLine(
                text: TrainingHealthLog.staleObservationLine(
                    tier: .live, trainerStep: observationTrainerStep, generation: stamp.generation,
                    currentGeneration: commitGeneration,
                    newestApplied: state.evaluator.newestAppliedTrainerStep,
                    reason: "trainer clock rewound during the evaluation"),
                eventKind: nil))
            state.counters.cost += clock.now - started
            return nil
        }
        state.evaluator = candidate
        appendHistory(&state, windowEntries)
        state.counters.live += 1
        if let window = observation.window {
            noteRatios(&state, window: window)
        }
        commitBookkeeping(&state, result: result, digest: digest, trainerStep: observationTrainerStep)
        emit(result, state: &state, config: config, trainerStep: observationTrainerStep, started: started,
             clock: clock, log: log)
        return result
    }

    // MARK: Checkpoint evaluation

    /// Evaluate one checkpoint-tier digest (a save's full-tensor pass, or the
    /// dedicated value-FC1 read). Never consumes the window and never
    /// detects a rewind itself (its trainer step may be old), but applies one
    /// `recordStep` detected before doing anything else.
    @discardableResult
    func evaluateCheckpoint(
        stamp: TrainingHealthStamp,
        layerHealth: LayerHealthDigest,
        digestTrainerStep: Int,
        config: TrainingHealthConfig,
        log: TrainingHealthLogSink
    ) -> TrainingHealthEvaluation? {
        evaluation.mutate { state in
            self.performCheckpoint(
                &state, stamp: stamp, digest: layerHealth, trainerStep: digestTrainerStep,
                config: config, log: log)
        }
    }

    private func performCheckpoint(
        _ state: inout EvaluationState,
        stamp: TrainingHealthStamp,
        digest: LayerHealthDigest,
        trainerStep: Int,
        config: TrainingHealthConfig,
        log: TrainingHealthLogSink
    ) -> TrainingHealthEvaluation? {
        let clock = ContinuousClock()
        let started = clock.now
        applyUnannouncedRewinds(&state, logging: config.enabled, log: log)
        guard config.enabled else {
            // Judges, counts and logs nothing — not even a stale stamp.
            state.counters.cost += clock.now - started
            return nil
        }
        let runID = self.runID
        let validation = steps.read { store -> (valid: Bool, currentGeneration: Int) in
            (stamp.runID == runID && stamp.generation == store.generation, store.generation)
        }
        guard validation.valid else {
            state.counters.stale += 1
            log(TrainingHealthLogLine(
                text: TrainingHealthLog.staleObservationLine(
                    tier: .checkpoint, trainerStep: trainerStep, generation: stamp.generation,
                    currentGeneration: validation.currentGeneration,
                    newestApplied: state.evaluator.newestAppliedTrainerStep,
                    reason: stamp.runID == runID ? nil : "stamp from another monitor"),
                eventKind: nil))
            state.counters.cost += clock.now - started
            return nil
        }
        let observation = TrainingHealthObservation(
            trainerStep: trainerStep, window: nil, layerHealth: digest, stamp: stamp,
            effectiveLearningRate: nil, effectiveMomentum: nil)
        var candidate = state.evaluator
        let result = candidate.evaluate(observation, config: config)

        beforeCommitForTesting?()

        let commitGeneration = steps.read { $0.generation }
        guard commitGeneration == stamp.generation else {
            state.counters.stale += 1
            log(TrainingHealthLogLine(
                text: TrainingHealthLog.staleObservationLine(
                    tier: .checkpoint, trainerStep: trainerStep, generation: stamp.generation,
                    currentGeneration: commitGeneration,
                    newestApplied: state.evaluator.newestAppliedTrainerStep,
                    reason: "trainer clock rewound during the evaluation"),
                eventKind: nil))
            state.counters.cost += clock.now - started
            return nil
        }
        state.evaluator = candidate
        state.counters.checkpoint += 1
        commitBookkeeping(&state, result: result, digest: digest, trainerStep: trainerStep)
        emit(result, state: &state, config: config, trainerStep: trainerStep, started: started,
             clock: clock, log: log)
        return result
    }

    // MARK: Commit helpers

    private func appendHistory(_ state: inout EvaluationState, _ entries: [TrainingHealthReferenceEntry]) {
        state.history.append(contentsOf: entries)
        let overflow = state.history.count - TrainingHealthThresholds.spikeReferenceHistoryCapacity
        if overflow > 0 {
            state.history.removeFirst(overflow)
        }
    }

    private func noteRatios(_ state: inout EvaluationState, window: TrainingHealthWindowStatistics) {
        if let reference = window.lossReference, reference.median > 0 {
            if let maximum = window.lossMax {
                state.counters.lossMaxRatio = max(state.counters.lossMaxRatio ?? -.infinity, maximum / reference.median)
            }
            if let median = window.lossMedian {
                state.counters.lossMedianRatio = max(state.counters.lossMedianRatio ?? -.infinity, median / reference.median)
            }
        }
        if let reference = window.gradientNormReference, reference.median > 0, let maximum = window.gradientNormMax {
            state.counters.gradientMaxRatio = max(state.counters.gradientMaxRatio ?? -.infinity, maximum / reference.median)
        }
    }

    private func commitBookkeeping(
        _ state: inout EvaluationState,
        result: TrainingHealthEvaluation,
        digest: LayerHealthDigest?,
        trainerStep: Int
    ) {
        state.committedEvaluations += 1
        for rule in result.noDataRules {
            state.counters.noData[rule, default: 0] += 1
        }
        state.counters.stale += result.staleRules.count
        for event in result.events where event.kind == .raise || event.kind == .escalate {
            if var entry = state.raised[event.rule] {
                if event.severity.healthRank > entry.highestSeverity.healthRank {
                    entry.highestSeverity = event.severity
                }
                if event.kind == .raise { entry.raiseCount += 1 }
                state.raised[event.rule] = entry
            } else {
                state.raised[event.rule] = RaisedSummary(
                    firstTrainerStep: event.trainerStep, highestSeverity: event.severity,
                    raiseCount: event.kind == .raise ? 1 : 0)
            }
        }
        if digest?.valueFC1 != nil {
            state.newestValueFC1ObservationStep = max(state.newestValueFC1ObservationStep ?? trainerStep, trainerStep)
        }
    }

    /// Log the evaluation's lines (rule notes, events, then a due check line)
    /// before the lock is released, so log order is evaluation order.
    private func emit(
        _ result: TrainingHealthEvaluation,
        state: inout EvaluationState,
        config: TrainingHealthConfig,
        trainerStep: Int,
        started: ContinuousClock.Instant,
        clock: ContinuousClock,
        log: TrainingHealthLogSink
    ) {
        for note in result.newlyNotApplicable {
            log(TrainingHealthLogLine(
                text: TrainingHealthLog.notApplicableLine(rule: note.rule, reason: note.reason),
                eventKind: nil))
        }
        for event in result.events {
            log(TrainingHealthLogLine(text: TrainingHealthLog.eventLine(event), eventKind: event.kind))
        }
        state.counters.cost += clock.now - started
        if result.checkDue {
            writeCheckLine(&state, trainerStep: trainerStep, marker: nil, log: log)
        } else if state.finalCheckWritten {
            writeCheckLine(&state, trainerStep: trainerStep, marker: .late, log: log)
        }
    }

    private func writeCheckLine(
        _ state: inout EvaluationState,
        trainerStep: Int?,
        marker: TrainingHealthLog.CheckMarker?,
        log: TrainingHealthLogSink
    ) {
        let moved = steps.mutate { store -> (generation: Int, truncated: Int, recordCost: Duration, trainMs: Double) in
            let values = (store.generation, store.truncatedSinceCheck, store.recordCostSinceCheck, store.trainMsSinceCheck)
            store.truncatedSinceCheck = 0
            store.recordCostSinceCheck = .zero
            store.trainMsSinceCheck = 0
            return values
        }
        let counters = state.counters
        let fields = TrainingHealthLog.CheckFields(
            trainerStep: trainerStep,
            generation: moved.generation,
            live: counters.live,
            checkpoint: counters.checkpoint,
            stale: counters.stale,
            truncated: moved.truncated,
            costMs: Self.milliseconds(counters.cost + moved.recordCost),
            trainMs: moved.trainMs,
            liveReadFailed: counters.liveReadFailed,
            lossMaxRatio: counters.lossMaxRatio,
            lossMedianRatio: counters.lossMedianRatio,
            gradientMaxRatio: counters.gradientMaxRatio,
            noData: counters.noData,
            active: state.evaluator.activeAlarms,
            marker: marker)
        log(TrainingHealthLogLine(text: TrainingHealthLog.checkLine(fields), eventKind: nil))
        state.counters = IntervalCounters()
    }

    static func milliseconds(_ duration: Duration) -> Double {
        let parts = duration.components
        return Double(parts.seconds) * 1000 + Double(parts.attoseconds) / 1e15
    }

    // MARK: End of run

    /// Write the final `[HEALTH] check … final=true` line so the last partial
    /// interval (and its cost) is never lost. A GUI checkpoint pass that
    /// finishes later still logs and applies, and writes its own
    /// `late=true` check line. Its trainer step is the newest the monitor
    /// knows — the last recorded step or the newest step an evaluation
    /// applied, whichever is later (a GUI log replayed offline has
    /// evaluations but no records) — and `none` when it knows neither.
    func writeFinalCheck(log: TrainingHealthLogSink) {
        evaluation.modify { state in
            let lastRecorded = self.steps.read { $0.lastRecordedTrainerStep }
            let newestApplied = state.evaluator.newestAppliedTrainerStep
            let trainerStep = [lastRecorded, newestApplied].compactMap { $0 }.max()
            self.writeCheckLine(&state, trainerStep: trainerStep, marker: .final, log: log)
            state.finalCheckWritten = true
        }
    }

    // MARK: Parking (GUI)

    /// Set by the GUI's main actor when a health stop suspends training
    /// (R3); the trainer worker of this monitor's run polls it at its loop
    /// top and parks. Per monitor, so a later run's worker never sees an
    /// earlier run's request.
    private let parkRequest = SyncBox(false)

    func requestPark() {
        parkRequest.value = true
    }

    var parkRequested: Bool {
        parkRequest.value
    }

    // MARK: Reads

    /// The current active set, in rule order (the GUI list mirrors it).
    func activeAlarmsSnapshot() -> [TrainingHealthActiveAlarm] {
        evaluation.value.evaluator.activeAlarms
    }

    /// Whether this monitor's evaluator has requested a stop (CLI paths).
    var stopRequested: Bool {
        evaluation.value.evaluator.stopRequested
    }

    /// The per-segment summary HPARAM_RECORDING_PLAN P4 records (OD-10).
    func segmentSummary() -> TrainingHealthSegmentSummary {
        let state = evaluation.value
        let raised = TrainingHealthRule.allCases.compactMap { rule -> TrainingHealthSegmentSummary.Raised? in
            guard let entry = state.raised[rule] else { return nil }
            return TrainingHealthSegmentSummary.Raised(
                rule: rule, firstTrainerStep: entry.firstTrainerStep,
                highestSeverity: entry.highestSeverity, raiseCount: entry.raiseCount)
        }
        return TrainingHealthSegmentSummary(evaluations: state.committedEvaluations, raised: raised)
    }

    /// D6: whether the dedicated value-FC1 velocity read is due at
    /// `trainerStep` — true when at least `valueFC1CheckIntervalSteps` have
    /// passed since the newest rule-3 observation of this generation, or,
    /// when there is none, since the clock this monitor started recording
    /// from (or the restored clock after a rewind). Never due when rule 3
    /// does not apply, before anything was recorded, or when `config` has
    /// alarms disabled: a disabled config commits no evaluation, so the
    /// anchor would never move and every step past the interval would ask
    /// for a read whose result nothing judges.
    func valueFC1ReadDue(trainerStep: Int, config: TrainingHealthConfig) -> Bool {
        guard config.enabled, case .applies = valueFC1Applicability else { return false }
        return evaluation.mutate { state in
            let store = self.steps.read { (generation: $0.generation, anchorBase: $0.valueFC1AnchorBase) }
            let observation = store.generation == state.appliedGeneration ? state.newestValueFC1ObservationStep : nil
            guard let anchor = observation ?? store.anchorBase else { return false }
            return trainerStep - anchor >= TrainingHealthThresholds.valueFC1CheckIntervalSteps
        }
    }
}
