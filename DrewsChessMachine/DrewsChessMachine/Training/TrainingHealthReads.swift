import Foundation

/// The reads a training path feeds the training-health monitor, shared by
/// the GUI and both command-line paths so the read, its timing, its log line
/// and its failure handling cannot drift between them (alarms plan D5, D6).
/// The monitor itself never touches the trainer; these do, through the
/// trainer's own read methods, which run on its `executionQueue` between SGD
/// steps.
enum TrainingHealthReads {

    /// The live layer-health input of an evaluation, from a live read's
    /// outcome: the digest (with rule 14's per-channel ratios) with the
    /// trainer clock it was read at, or a failed read (already logged by
    /// `LayerHealthLog` as one line of `outcome`). A summary without its
    /// profile cannot come out of `LayerHealthLog.live` (one pass yields
    /// both), so it is a code bug, never "no data".
    static func liveLayerHealth(from outcome: LayerHealthLog.LiveOutcome) -> TrainingHealthLiveLayerHealth {
        guard let summary = outcome.summary, let trainerStep = outcome.trainerStep else {
            return .readFailed
        }
        guard let profile = outcome.runningVarianceProfile else {
            preconditionFailure("LayerHealthLog.LiveOutcome carries a live summary without its running-variance profile")
        }
        return .read(LayerHealthDigest(liveSummary: summary, runningVarianceProfile: profile), trainerStep: trainerStep)
    }

    /// What one dedicated value-FC1 read produced.
    enum ValueFC1ReadOutcome: Sendable {
        /// Read and evaluated. The evaluation is nil when nothing was
        /// committed (a stale stamp, a rewind during it, alarms disabled).
        case evaluated(TrainingHealthEvaluation?)
        /// The read or its summary failed (logged); rule 3 has no new data.
        case failed(trainerStep: Int)
    }

    /// The dedicated value-FC1 velocity read (D6): stamp, read the one
    /// tensor's velocity on the trainer queue, summarize it with the same
    /// pure `LayerHealth.hiddenUnitVelocityHealth` a checkpoint pass uses,
    /// log the `[LAYER-HEALTH] value-fc1` line, and evaluate it as a
    /// rule-3-only checkpoint-tier observation. The stamp is taken before
    /// the read, so rule 3's gate uses the steps trained when the state was
    /// read. The summary is one pass over at most 512 KB of floats (well
    /// under a millisecond), so it runs on the calling task.
    ///
    /// A failure never stops training (the monitor is an observer): it is
    /// logged as one `[LAYER-HEALTH] value-fc1 … failed` line through `log`
    /// and reported to the caller, who waits a full interval before the next
    /// attempt (`TrainingHealthValueFC1ReadRetry`).
    static func valueFC1Read(
        trainer: ChessTrainer,
        monitor: TrainingHealthMonitor,
        config: TrainingHealthConfig,
        attemptTrainerStep: Int,
        log: TrainingHealthLogSink
    ) async -> ValueFC1ReadOutcome {
        let stamp = monitor.observationStamp()
        let layer = LayerHealth.valueFC1Layer(for: trainer.arch)
        let clock = ContinuousClock()
        let readStarted = clock.now
        do {
            let read = try await trainer.readTrainableVelocity(named: layer.weightTensorName)
            let readMs = TrainingHealthMonitor.milliseconds(clock.now - readStarted)
            let summaryStarted = clock.now
            let health = try LayerHealth.hiddenUnitVelocityHealth(layer: layer, velocity: read.velocity)
            let summaryMs = TrainingHealthMonitor.milliseconds(clock.now - summaryStarted)
            log(TrainingHealthLogLine(
                text: TrainingHealthLog.valueFC1Line(
                    trainerStep: read.completedTrainSteps,
                    stepsTrainedByThisProcess: stamp.stepsTrainedByThisProcess,
                    zeroVelocityUnitCount: health.zeroVelocityUnitCount,
                    unitCount: health.unitCount,
                    lowVelocityUnitCount: health.lowVelocityUnitCount,
                    readMs: readMs,
                    summaryMs: summaryMs),
                eventKind: nil))
            let digest = LayerHealthDigest.valueFC1Only(LayerHealthDigest.ValueFC1Velocity(
                zeroVelocityUnitCount: health.zeroVelocityUnitCount, unitCount: health.unitCount))
            return .evaluated(monitor.evaluateCheckpoint(
                stamp: stamp, layerHealth: digest, digestTrainerStep: read.completedTrainSteps,
                config: config, log: log))
        } catch {
            log(TrainingHealthLogLine(
                text: "\(LayerHealthLog.tag) value-fc1 read failed at trainerStep=\(attemptTrainerStep): "
                    + "\(error.localizedDescription); next attempt after "
                    + "\(TrainingHealthThresholds.valueFC1CheckIntervalSteps) trainer steps",
                eventKind: nil))
            return .failed(trainerStep: attemptTrainerStep)
        }
    }
}

/// Holds the dedicated value-FC1 read back for one interval after a failed
/// attempt. Without it a read that keeps failing (the monitor's deadline
/// moves only with a committed observation) would be attempted on every
/// SGD step. A pure value type; each path keeps one per monitor.
struct TrainingHealthValueFC1ReadRetry: Sendable, Equatable {
    private(set) var failedAtTrainerStep: Int?

    init() {}

    /// Whether a read may be attempted at `trainerStep`: always, unless a
    /// read failed less than `valueFC1CheckIntervalSteps` trainer steps
    /// before it. A trainer clock that went back below the failure (a GUI
    /// promotion rewind) allows the read at once: the interval it waited
    /// on no longer exists.
    func allowsRead(atTrainerStep trainerStep: Int) -> Bool {
        guard let failedAtTrainerStep else { return true }
        if trainerStep < failedAtTrainerStep { return true }
        return trainerStep - failedAtTrainerStep >= TrainingHealthThresholds.valueFC1CheckIntervalSteps
    }

    mutating func record(_ outcome: TrainingHealthReads.ValueFC1ReadOutcome) {
        switch outcome {
        case .evaluated:
            failedAtTrainerStep = nil
        case .failed(let trainerStep):
            failedAtTrainerStep = trainerStep
        }
    }
}
