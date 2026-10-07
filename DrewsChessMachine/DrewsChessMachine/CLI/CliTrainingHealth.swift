import Foundation

/// The training-health wiring both command-line training paths share
/// (`--replay-corpus` and `--train-vs-uci`; alarms plan T3, T4), so the
/// order of operations, the stop latch, the `results.json` recording and the
/// log routing are written once.
///
/// One instance per run, used only by the run's one training task, which
/// calls everything in sequence (record → step-line block → live evaluation
/// → save block, per step). That sequencing is the only synchronization it
/// needs; the monitor inside it has its own locks.
///
/// What it does at each point of the loop:
/// - after every SGD step: `record` (one uncontended lock, one append);
/// - before the step-line block: `liveEvaluationStamp(trainerStep:)` — a
///   stamp only on a live-evaluation step (every 50 trainer steps on
///   overall multiples, OD-21), taken before any read the evaluation uses;
/// - after the step-line block: `evaluateLive`, sharing the step line's
///   live `[LAYER-HEALTH]` read when the line fell on the same step (the
///   cadence plan's OD-15), otherwise taking its own;
/// - inside every save, right after the trainer export: `observationStamp`,
///   and after the save's checkpoint pass `evaluateCheckpoint`;
/// - after the save block: `valueFC1ReadIfDue` (D6);
/// - at the loop top: `requestedStop`, the stop the run honours.
///
/// The settings are read once from the run-start snapshot, as every other
/// command-line parameter is.
final class CliTrainingHealth {

    let monitor: TrainingHealthMonitor
    let config: TrainingHealthConfig
    /// The log tag of the path's own lines (`[REPLAY]` / `[VS-UCI]`).
    private let pathTag: String
    private let emit: @Sendable (String) -> Void
    private let recorder: CliTrainingRecorder?
    private var valueFC1Retry = TrainingHealthValueFC1ReadRetry()
    /// Trainer step of the newest live evaluation this run attempted, so the
    /// explicit evaluation before the final save never repeats one.
    private var lastLiveEvaluationTrainerStep: Int?
    private var recordedSinceLastLiveEvaluation = false

    /// The first stop an evaluation requested (R2: once per monitor). The
    /// runner reads it at its loop top and stops before the next step; a
    /// stop first requested by the final save's own pass is recorded here
    /// too but changes nothing, because the runner latches its own copy
    /// only inside the loop.
    private(set) var requestedStop: TrainingHealthEvent?

    /// Resolves the config from the run's parameter snapshot, records it in
    /// `results.json`, and logs the `[HEALTH] config` (or `alarms disabled`)
    /// line — call right after the run's `[RUN]` line. `path` is the
    /// config line's `path=` (`replay` / `vsuci`).
    init(
        parameters: TrainingParametersSnapshot,
        arch: NetworkArchitecture,
        path: String,
        pathTag: String,
        recorder: CliTrainingRecorder?,
        emit: @escaping @Sendable (String) -> Void
    ) throws {
        let config = try TrainingHealthConfig(parameters)
        let applicability = TrainingHealthValueFC1Applicability(
            valueFC1Activation: LayerHealth.valueFC1Layer(for: arch).activation)
        self.config = config
        self.monitor = TrainingHealthMonitor(valueFC1Applicability: applicability)
        self.pathTag = pathTag
        self.emit = emit
        self.recorder = recorder
        recorder?.setAlarmConfig(config)
        if config.enabled {
            emit(TrainingHealthLog.configLine(config: config, path: path, valueFC1Applicability: applicability))
        } else {
            emit(TrainingHealthLog.disabledLine(path: path))
        }
    }

    /// The monitor's log sink: every line to the session log and stdout
    /// (the runner's `emit`), and raise, escalate and stop lines also to
    /// stderr, as the runners' other `[ALARM]` lines are (D3).
    var logSink: TrainingHealthLogSink {
        let emit = self.emit
        return { line in
            emit(line.text)
            switch line.eventKind {
            case .raise?, .escalate?, .stop?:
                FileHandle.standardError.write(Data((line.text + "\n").utf8))
            case .worsen?, .active?, .clear?, nil:
                break
            }
        }
    }

    // MARK: Per step

    /// Record one SGD step (nothing while alarms are disabled: no
    /// evaluation would ever drain the window).
    func record(_ timing: TrainStepTiming, trainerStep: Int) {
        guard config.enabled else { return }
        monitor.recordStep(timing, trainerStep: trainerStep)
        recordedSinceLastLiveEvaluation = true
    }

    /// The stamp of this step's live evaluation, or nil when none is due.
    /// Taken before the step-line block, so it precedes the live read the
    /// evaluation uses even when that read is the step line's.
    func liveEvaluationStamp(trainerStep: Int) -> TrainingHealthStamp? {
        guard config.enabled, TrainingHealthCadence.isLiveEvaluationStep(trainerStep: trainerStep) else {
            return nil
        }
        return monitor.observationStamp()
    }

    /// The live evaluation of a due step. `sharedRead` is the step line's
    /// live read when the line fell on this step (one read serves both);
    /// otherwise the evaluation takes its own, whose line is not logged —
    /// the logged readout rides the step line — except when it failed, so
    /// the failure's reason is never lost.
    func evaluateLive(
        stamp: TrainingHealthStamp,
        sharedRead: LayerHealthLog.LiveOutcome?,
        trainer: ChessTrainer,
        trainerStep: Int,
        learningRate: Float,
        momentum: Float
    ) async {
        let outcome: LayerHealthLog.LiveOutcome
        if let sharedRead {
            outcome = sharedRead
        } else {
            outcome = await LayerHealthLog.live(trainer: trainer)
            if outcome.summary == nil {
                for line in outcome.lines { emit(line) }
            }
        }
        lastLiveEvaluationTrainerStep = trainerStep
        recordedSinceLastLiveEvaluation = false
        let evaluation = monitor.evaluateLive(
            stamp: stamp,
            layerHealth: TrainingHealthReads.liveLayerHealth(from: outcome),
            learningRate: Double(learningRate),
            momentum: Double(momentum),
            config: config,
            log: logSink)
        deliver(evaluation)
    }

    /// R0: the final save, at an arbitrary step, gets a live evaluation of
    /// its own first, so the last partial window is judged — unless this
    /// step's evaluation already ran or nothing was recorded since it.
    func evaluateBeforeFinalSave(trainer: ChessTrainer, batchSize: Int) async {
        guard config.enabled, recordedSinceLastLiveEvaluation else { return }
        let trainerStep = trainer.completedTrainSteps
        guard lastLiveEvaluationTrainerStep != trainerStep else { return }
        let stamp = monitor.observationStamp()
        await evaluateLive(
            stamp: stamp, sharedRead: nil, trainer: trainer, trainerStep: trainerStep,
            learningRate: trainer.effectiveLearningRate(forBatchSize: batchSize, completedSteps: trainerStep),
            momentum: trainer.effectiveMomentum(completedSteps: trainerStep))
    }

    // MARK: Saves

    /// The stamp a save's checkpoint evaluation carries: take it right
    /// after the trainer export, so it describes the exported state.
    func observationStamp() -> TrainingHealthStamp {
        monitor.observationStamp()
    }

    /// Evaluate a save's checkpoint pass. A failed pass (`summary` nil,
    /// already logged) is no observation, so rule 3's deadline stays where
    /// it was and the dedicated read covers the interval (D6).
    func evaluateCheckpoint(stamp: TrainingHealthStamp, summary: LayerHealthSummary?, trainerStep: Int) {
        guard config.enabled, let summary else { return }
        let evaluation = monitor.evaluateCheckpoint(
            stamp: stamp, layerHealth: LayerHealthDigest(summary: summary), digestTrainerStep: trainerStep,
            config: config, log: logSink)
        deliver(evaluation)
    }

    /// D6: the dedicated value-FC1 read when rule 3's deadline has passed
    /// with no observation — ask after the step's save block, so a save
    /// whose pass just fed rule 3 has already moved the deadline.
    func valueFC1ReadIfDue(trainer: ChessTrainer, trainerStep: Int) async {
        guard monitor.valueFC1ReadDue(trainerStep: trainerStep, config: config),
              valueFC1Retry.allowsRead(atTrainerStep: trainerStep) else { return }
        let outcome = await TrainingHealthReads.valueFC1Read(
            trainer: trainer, monitor: monitor, config: config, attemptTrainerStep: trainerStep, log: logSink)
        valueFC1Retry.record(outcome)
        if case .evaluated(let evaluation) = outcome {
            deliver(evaluation)
        }
    }

    // MARK: End of run

    /// The `[HEALTH] check … final=true` line, after the final save's
    /// checkpoint evaluation, so the last partial interval is never lost.
    func finish() {
        guard config.enabled else { return }
        monitor.writeFinalCheck(log: logSink)
    }

    /// The line the runner logs when it honours a stop at its loop top.
    func stopLine(segmentStep: Int) -> String? {
        guard let requestedStop else { return nil }
        return "\(pathTag) training health alarm \(requestedStop.rule.rawValue) requested a stop — stopping at step \(segmentStep)"
    }

    // MARK: Private

    /// Record a committed evaluation's events in `results.json` (same order
    /// as the log lines) and latch its stop request.
    private func deliver(_ evaluation: TrainingHealthEvaluation?) {
        guard let evaluation else { return }
        recorder?.appendAlarmEvents(evaluation.events)
        if requestedStop == nil, let stop = evaluation.stopRequest {
            requestedStop = stop
        }
    }
}
