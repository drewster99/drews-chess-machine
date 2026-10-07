import Foundation

/// A GUI `--train` run's termination claim, for a training-health stop: the
/// same single-winner claim the deadline, step-limit, early-stop and
/// legal-mass-collapse paths take (`AutoTrainTermination`), and the run's
/// start for the results' elapsed time.
struct TrainingHealthAutoTrainStop: Sendable {
    let termination: AutoTrainTermination
    let runStart: Date
}

/// What a GUI session save's detached checkpoint pass needs to evaluate its
/// summary for the training-health monitor (alarms plan D2, T8): the monitor
/// of the run the save belongs to, the stamp taken with the trainer export
/// (for the promotion save, after the rewind, in the same pause), and the
/// config in force when that state was exported — the checkpoint's data is
/// judged under it, while the stop decision is made on the main actor from
/// the actions in force when the result arrives (R2).
struct GuiTrainingHealthCheckpoint: Sendable {
    let monitor: TrainingHealthMonitor
    let stamp: TrainingHealthStamp
    let config: TrainingHealthConfig
    /// Hops to the main actor and hands the monitor to
    /// `SessionController.deliverTrainingHealth(from:)`.
    let deliver: @Sendable (TrainingHealthMonitor) -> Void
}

/// The training-health work the GUI trainer worker does, used only by that
/// one task (which calls it in sequence between SGD steps, so it needs no
/// lock of its own; the monitor has its own). Mirrors the command-line
/// `CliTrainingHealth` loop points, with two GUI differences:
/// - the config is resolved from `TrainingParameters.shared` on the main
///   actor at every live evaluation, so a Health-tab edit applies at the
///   next one (Part K item 8);
/// - stops are decided on the main actor after delivery
///   (`TrainingHealthStopDecision.byCaller`), and the worker parks when the
///   monitor asks it to.
///
/// Live evaluations run here, on the worker, every 50 trainer steps on
/// overall multiples (OD-21: "from the trainer worker's step count"), not on
/// the `[STATS]` ticker, which polls on its own time-based cadence and so
/// cannot land on exact trainer-step multiples. The cost the worker pays is
/// one live read on the trainer queue (≈ 1 ms), its summary on a GCD queue,
/// and the evaluation itself: well under 1% at the GUI's ≈ 0.65 s/step.
final class GuiTrainingHealthWorker {
    let monitor: TrainingHealthMonitor
    private let trainer: ChessTrainer
    private let batchSize: Int
    private let recorder: CliTrainingRecorder?
    private let resolveConfig: @Sendable () async -> TrainingHealthConfig?
    private let deliver: @Sendable (TrainingHealthMonitor) -> Void
    /// The config of the newest live evaluation (nil before the first, or
    /// when resolution failed): what the per-step value-FC1 deadline check
    /// uses, so the worker does not hop to the main actor every step.
    private var latestConfig: TrainingHealthConfig?
    private var valueFC1Retry = TrainingHealthValueFC1ReadRetry()

    init(
        monitor: TrainingHealthMonitor,
        trainer: ChessTrainer,
        batchSize: Int,
        recorder: CliTrainingRecorder?,
        resolveConfig: @escaping @Sendable () async -> TrainingHealthConfig?,
        deliver: @escaping @Sendable (TrainingHealthMonitor) -> Void
    ) {
        self.monitor = monitor
        self.trainer = trainer
        self.batchSize = batchSize
        self.recorder = recorder
        self.resolveConfig = resolveConfig
        self.deliver = deliver
    }

    /// The GUI's log sink: the session log (a non-blocking enqueue).
    static let logSink: TrainingHealthLogSink = { line in
        SessionLogger.shared.log(line.text)
    }

    /// Whether the main actor asked this run's worker to park (a health
    /// stop in an interactive session).
    var parkRequested: Bool { monitor.parkRequested }

    /// After every SGD step: record it; on a live-evaluation step, stamp,
    /// read, evaluate and deliver; then the dedicated value-FC1 read when
    /// its deadline has passed (D6).
    func afterStep(_ timing: TrainStepTiming, trainerStep: Int) async {
        monitor.recordStep(timing, trainerStep: trainerStep)
        if TrainingHealthCadence.isLiveEvaluationStep(trainerStep: trainerStep) {
            // The stamp precedes the read (D2).
            let stamp = monitor.observationStamp()
            latestConfig = await resolveConfig()
            if let config = latestConfig {
                // results.json records the last config in force (live).
                recorder?.setAlarmConfig(config)
                let outcome = await LayerHealthLog.live(trainer: trainer)
                if outcome.summary == nil {
                    // The logged readout rides the [STATS] ticker; only a
                    // failure of this read is logged here, with its reason.
                    for line in outcome.lines { SessionLogger.shared.log(line) }
                }
                let evaluation = monitor.evaluateLive(
                    stamp: stamp,
                    layerHealth: TrainingHealthReads.liveLayerHealth(from: outcome),
                    learningRate: Double(trainer.effectiveLearningRate(forBatchSize: batchSize, completedSteps: trainerStep)),
                    momentum: Double(trainer.effectiveMomentum(completedSteps: trainerStep)),
                    config: config,
                    log: Self.logSink)
                if let evaluation {
                    recorder?.appendAlarmEvents(evaluation.events)
                }
                deliver(monitor)
            }
        }
        guard let config = latestConfig,
              monitor.valueFC1ReadDue(trainerStep: trainerStep, config: config),
              valueFC1Retry.allowsRead(atTrainerStep: trainerStep) else { return }
        let outcome = await TrainingHealthReads.valueFC1Read(
            trainer: trainer, monitor: monitor, config: config, attemptTrainerStep: trainerStep, log: Self.logSink)
        valueFC1Retry.record(outcome)
        if case .evaluated(let evaluation) = outcome {
            if let evaluation {
                recorder?.appendAlarmEvents(evaluation.events)
            }
            deliver(monitor)
        }
    }

    /// The parked trainer worker (R3): no training step, but every pause
    /// request is acknowledged exactly as the loop top does, so a session
    /// save (which waits for the worker to acknowledge the training pause)
    /// and an arena already running still complete. Returns only when the
    /// task is cancelled (Stop).
    static func park(at trainingGate: WorkerPauseGate) async {
        while !Task.isCancelled {
            if trainingGate.isRequestedToPause {
                trainingGate.markWaiting()
                while trainingGate.isRequestedToPause && !Task.isCancelled {
                    do {
                        try await Task.sleep(for: .milliseconds(5))
                    } catch {
                        return
                    }
                }
                trainingGate.markRunning()
            }
            do {
                try await Task.sleep(for: .milliseconds(100))
            } catch {
                return
            }
        }
    }
}

extension SessionController {

    /// Start a run's training-health monitoring: a new monitor (every start,
    /// a continue after Stop included, T8), the health list emptied, and the
    /// `[HEALTH] config` line. Called by `startRealTraining`.
    func beginTrainingHealthRun(
        arch: NetworkArchitecture,
        recorder: CliTrainingRecorder?,
        autoTrainStop: TrainingHealthAutoTrainStop?
    ) -> TrainingHealthMonitor {
        let applicability = TrainingHealthValueFC1Applicability(
            valueFC1Activation: LayerHealth.valueFC1Layer(for: arch).activation)
        let monitor = TrainingHealthMonitor(valueFC1Applicability: applicability, stopDecision: .byCaller)
        // The monitor this start replaces has ended: fold its summary into
        // the segment it reported to (the lineage segment still held here —
        // `beginRunLineage` runs later in the start, and a new segment's
        // tracker starts from an empty summary). This is the one place a
        // monitor is replaced, so each is folded in exactly once.
        if let ended = trainingHealthMonitor, let tracker = lineageTracker {
            tracker.mergeEndedHealthMonitor(ended.segmentSummary())
        }
        trainingHealthMonitor = monitor
        trainingHealthAutoTrainStop = autoTrainStop
        trainingAlarm?.setHealthSuspension(nil)
        trainingAlarm?.refreshHealth(from: monitor)
        if let config = resolveTrainingHealthConfig() {
            recorder?.setAlarmConfig(config)
            SessionLogger.shared.log(config.enabled
                ? TrainingHealthLog.configLine(config: config, path: "gui", valueFC1Applicability: applicability)
                : TrainingHealthLog.disabledLine(path: "gui"))
        }
        return monitor
    }

    /// The config in force now, from the live settings. A config the
    /// declared ranges should have made impossible is logged, and nothing
    /// is evaluated under it (never a default in its place).
    func resolveTrainingHealthConfig() -> TrainingHealthConfig? {
        do {
            return try TrainingHealthConfig(TrainingParameters.shared.snapshot())
        } catch {
            SessionLogger.shared.log("[HEALTH] settings refused, nothing evaluated: \(error.localizedDescription)")
            return nil
        }
    }

    /// What a session save's checkpoint pass needs for its evaluation:
    /// call right after the save's trainer export (for the promotion save,
    /// after the rewind), so the stamp describes the exported state. Nil
    /// outside a run or when the settings were refused.
    func makeTrainingHealthCheckpoint() -> GuiTrainingHealthCheckpoint? {
        guard let monitor = trainingHealthMonitor, let config = resolveTrainingHealthConfig() else { return nil }
        return GuiTrainingHealthCheckpoint(
            monitor: monitor, stamp: monitor.observationStamp(), config: config,
            deliver: makeTrainingHealthDelivery())
    }

    /// The hop every evaluation takes to the main actor. Fire and forget:
    /// the evaluating task never waits for the main actor, and two hops
    /// arriving out of order cannot show an older state, because the
    /// delivery re-reads the monitor's current active set.
    func makeTrainingHealthDelivery() -> @Sendable (TrainingHealthMonitor) -> Void {
        { [weak self] monitor in
            Task { @MainActor in
                self?.deliverTrainingHealth(from: monitor)
            }
        }
    }

    /// One evaluation arrived (live, checkpoint or value-FC1): mirror the
    /// monitor's active set into the alarm list, then decide a stop from the
    /// actions in force now (R2). A hop from an earlier start's monitor is
    /// ignored and logged.
    func deliverTrainingHealth(from monitor: TrainingHealthMonitor) {
        guard monitor === trainingHealthMonitor else {
            SessionLogger.shared.log("[HEALTH] evaluation from an earlier run's monitor ignored (run \(monitor.runID))")
            return
        }
        trainingAlarm?.refreshHealth(from: monitor)
        // `trainer` exists for the whole of a Play-and-Train run.
        guard realTraining, trainingSuspension == nil, let trainer else { return }
        let parameters = TrainingParameters.shared
        guard parameters.trainingHealthAlarmsEnabled else { return }
        let actions = TrainingHealthActions { parameters[keyPath: TrainingParameters.trainingHealthActionKeyPath(for: $0)] }
        guard let alarm = TrainingHealthStopPolicy.firstQualifying(
            active: monitor.activeAlarmsSnapshot(), actions: actions) else { return }
        stopTrainingForHealthAlarm(
            alarm, action: actions[alarm.rule], trainerStep: trainer.completedTrainSteps, monitor: monitor)
    }

    /// Carry out a health stop: the `stop` event line (the GUI evaluator
    /// writes none, `TrainingHealthStopDecision.byCaller`), recorded in the
    /// results; then `--train` ends through `AutoTrainTermination` (status
    /// 0, no session save first: OD-4, OD-13, as the legal-mass-collapse
    /// path), and an interactive session suspends training with the worker
    /// parked.
    private func stopTrainingForHealthAlarm(
        _ alarm: TrainingHealthActiveAlarm,
        action: TrainingHealthAction,
        trainerStep: Int,
        monitor: TrainingHealthMonitor
    ) {
        let stop = TrainingHealthEvent(
            kind: .stop, rule: alarm.rule, severity: alarm.severity, trainerStep: trainerStep,
            since: alarm.since, value: alarm.value, threshold: "", detail: "", action: action,
            learningRate: nil, momentum: nil)
        SessionLogger.shared.log(TrainingHealthLog.eventLine(stop))
        cliRecorder?.appendAlarmEvents([stop])

        if let autoTrainStop = trainingHealthAutoTrainStop {
            guard autoTrainStop.termination.claim() else {
                SessionLogger.shared.log(
                    "[APP] --train: training health stop (\(alarm.rule.rawValue)) left to the termination already writing the results")
                return
            }
            monitor.writeFinalCheck(log: GuiTrainingHealthWorker.logSink)
            autoTrainStop.termination.writeResultsAndExit(
                reason: .trainingHealthAlarm,
                trigger: "training health alarm \(alarm.rule.rawValue) at trainerStep=\(trainerStep)",
                elapsed: Date().timeIntervalSince(autoTrainStop.runStart))
        }
        suspendTrainingForHealthAlarm(alarm, trainerStep: trainerStep, monitor: monitor)
    }

    /// Suspend (not tear down) the run for a health stop (R3): the
    /// suspension gates arenas and Promote Trainee Now but not the periodic
    /// autosave; the worker parks at its next loop top, still acknowledging
    /// pause requests, so saves complete. Stop, then Start, clears it.
    func suspendTrainingForHealthAlarm(
        _ alarm: TrainingHealthActiveAlarm,
        trainerStep: Int,
        monitor: TrainingHealthMonitor
    ) {
        guard trainingSuspension == nil else { return }
        let detail = "\(alarm.severity.rawValue) \(alarm.value) since trainer step \(alarm.since)"
        trainingSuspension = .healthAlarm(rule: alarm.rule, detail: detail)
        trainingAlarm?.setHealthSuspension(alarm.rule)
        monitor.requestPark()
        // Close the in-progress training segment so cumulative wall-time
        // totals exclude the parked idle.
        checkpoint?.closeActiveTrainingSegment(reason: "health-suspend")
        SessionLogger.shared.log(
            "[HEALTH] training suspended: rule=\(alarm.rule.rawValue) severity=\(alarm.severity.rawValue) "
                + "trainerStep=\(trainerStep) value=\(alarm.value); trainer parked, arenas and Promote Trainee Now "
                + "refused, self-play and the periodic autosave continue; Stop to clear")
    }

    /// At Stop: the final `[HEALTH] check … final=true` line of this start's
    /// monitor. A checkpoint pass still running then writes its own
    /// `late=true` line when it finishes.
    func finishTrainingHealthRun() {
        trainingHealthMonitor?.writeFinalCheck(log: GuiTrainingHealthWorker.logSink)
        trainingHealthAutoTrainStop = nil
    }
}
