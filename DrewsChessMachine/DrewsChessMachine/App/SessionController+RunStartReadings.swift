import Foundation

/// The readers of what a Play-and-Train run captured at its start — its batch
/// size, pre-train fill and buffer capacity (`RunStartParameterCapture`) —
/// and the positions it trained, for everything outside the run's records
/// that states them: the status displays, the learning-rate readouts, the
/// Run All Analyses export and the Lichess probe history.
///
/// **Why one place.** Those three keys are not live-tunable, but the
/// settings popover writes them during a run for the next start. Every
/// reader that took them from `TrainingParameters.shared` therefore showed
/// an edit the run was not using: "Batch size: 1024" beside a trainer
/// stepping at 4096, moves/sec and positions trained 4× low, a "Self-play
/// prefill" chip over a trainer already stepping, and two exported files
/// stating the edited batch. While a run is active these readers take the
/// capture; with none active they show the setting and say that it is one.
extension SessionController {

    // MARK: - Captured parameters

    /// What a display or export shows for a run-start-captured parameter.
    enum CapturedParameterDisplay: Equatable, Sendable {
        /// The active run's value, from its start-time capture.
        case activeRun(Int)
        /// No run is active: the setting, which applies at the next start.
        /// Shown labelled as the setting, never as a run's value.
        case setting(Int)

        /// The number shown, whichever source it comes from.
        var value: Int {
            switch self {
            case .activeRun(let value), .setting(let value): return value
            }
        }

        /// `name` as a display labels this value: a setting says it is one.
        func label(_ name: String) -> String {
            switch self {
            case .activeRun: return name
            case .setting: return "\(name) (setting)"
            }
        }
    }

    /// The capture of the run that is active now, nil when none is.
    /// `realTraining` and the capture are set in the same main-actor turn of
    /// a start (and a failed start clears the one as it restores the
    /// other), so an active run always has its capture.
    var activeRunStartCapture: RunStartParameterCapture? {
        guard realTraining else { return nil }
        return runStartCapture
    }

    /// The training batch size to display: the active run's, else the setting.
    func trainingBatchSizeDisplay() -> CapturedParameterDisplay {
        if let capture = activeRunStartCapture {
            return .activeRun(capture.trainingBatchSize)
        }
        return .setting(TrainingParameters.shared.trainingBatchSize)
    }

    /// The pre-train fill to display: the active run's, else the setting.
    func replayBufferMinPositionsDisplay() -> CapturedParameterDisplay {
        if let capture = activeRunStartCapture {
            return .activeRun(capture.replayBufferMinPositionsBeforeTraining)
        }
        return .setting(TrainingParameters.shared.replayBufferMinPositionsBeforeTraining)
    }

    /// The batch size the effective learning-rate readouts (the status bar's
    /// LR and momentum, the training chart's LR / momentum / effective step)
    /// are computed at: the active run's, nil when no run is active. The
    /// √batch scale inside `effectiveLearningRate` is only the rate a trainer
    /// is fed while a run steps it at that batch; a stopped run's capture is
    /// kept for its saves, but a sweep or the demo training may step the same
    /// trainer at another batch, so nothing is published then.
    func optimizerReadoutBatchSize() -> Int? {
        activeRunStartCapture?.trainingBatchSize
    }

    // MARK: - Trained positions

    /// The positions a Play-and-Train run has trained, on the session's step
    /// axis (the run's stats box, which session.json's `trainingSteps` and the
    /// status bar count): begun at every start with its capture, kept through
    /// Stop, carried across a Continue, put back with the capture by a failed
    /// start.
    struct RunTrainedPositions: Sendable {
        /// The stats box whose step count is this count's axis; nil when the
        /// controller had none at the start (no step had been counted).
        /// Readers compare it with `trainingBox`: once demo training replaces
        /// the box, the displayed steps are no longer this run's.
        let statsBox: TrainingLiveStatsBox?
        /// The recorded count: what session.json, the status bar and the
        /// Lichess probe report.
        let recorded: TrainedPositionsCount
        /// Positions this process saw trained on `statsBox` before the start,
        /// recorded or not — the progress-rate chart's counter, which needs
        /// only differences, so it never depends on history before the
        /// process.
        let observedPositionsBeforeStart: Int
        /// The trainer clock when the trainer held this start's starting state
        /// (`beginRunLineage`); nil until then. With `recorded.stepsAtStart`,
        /// whether the session's steps are the trainer's clock.
        var trainerStepAtStart: Int?

        /// Positions this process saw trained by the time the stats read
        /// `steps`.
        func observedPositions(atSteps steps: Int) throws -> Int {
            guard steps >= recorded.stepsAtStart else {
                throw TrainedPositionsCount.CountError.stepsBeforeStart(steps: steps, stepsAtStart: recorded.stepsAtStart)
            }
            return observedPositionsBeforeStart + (steps - recorded.stepsAtStart) * recorded.batchSize
        }
    }

    /// What the latest start replaced: put back together by a start that
    /// fails before its lineage segment begins, so the segment left in place
    /// keeps the capture, positions count and replay buffer it was trained
    /// with (`restoreRunStartCaptureAfterFailedStart`).
    struct ReplacedRunStartState {
        let capture: RunStartParameterCapture?
        let trainedPositions: RunTrainedPositions?
        let replayBuffer: ReplayBuffer?
    }

    /// The positions count for a start whose capture trains at `batchSize`,
    /// on the session's step axis as it stands now:
    /// - a Continue (the same stats box the previous count was on) carries
    ///   that count forward to here at the previous start's batch;
    /// - a new stats box at step 0 (a fresh run or a new session) starts at 0;
    /// - a box seeded from the session being resumed carries the count that
    ///   session recorded (nil when it recorded none);
    /// - anything else counts steps no record describes: unrecorded, logged.
    func beginRunTrainedPositions(batchSize: Int) -> RunTrainedPositions {
        let box = trainingBox
        let steps: Int
        if let box {
            steps = box.snapshot().stats.steps
        } else {
            // No stats box: this controller has counted no training step.
            steps = 0
        }
        let recordedBefore: Int?
        let observedBefore: Int
        if let previous = runTrainedPositions, let box, previous.statsBox === box {
            do {
                let recorded = try previous.recorded.positions(atSteps: steps)
                let observed = try previous.observedPositions(atSteps: steps)
                recordedBefore = recorded
                observedBefore = observed
            } catch {
                // The stats only count forward within a run (a promotion
                // rewinds them, but never below the start of the run it ran
                // in), so this is a bug: the steps since the previous start
                // cannot be accounted, and the count says so.
                SessionLogger.shared.log("[PARAM] error: positions trained before step \(steps) cannot be carried "
                    + "(\(error.localizedDescription)); they are recorded as unrecorded")
                recordedBefore = nil
                observedBefore = 0
            }
        } else if steps == 0 {
            recordedBefore = 0
            observedBefore = 0
        } else if let resumed = pendingLoadedSession?.state, resumed.trainingSteps == steps {
            recordedBefore = resumed.trainingPositionsSeen
            observedBefore = 0
        } else {
            SessionLogger.shared.log("[PARAM] positions trained over the \(steps) steps before this start are unrecorded")
            recordedBefore = nil
            observedBefore = 0
        }
        return RunTrainedPositions(
            statsBox: box,
            recorded: TrainedPositionsCount(stepsAtStart: steps, positionsBeforeStart: recordedBefore, batchSize: batchSize),
            observedPositionsBeforeStart: observedBefore,
            trainerStepAtStart: nil)
    }

    /// Note the trainer clock at which this start's trainer holds its
    /// starting state. Called by `beginRunLineage` once the trainer has been
    /// reset, loaded or kept — the moment the lineage segment reads the same
    /// clock.
    func anchorRunTrainedPositions(atTrainerStep trainerStep: Int) {
        guard var ledger = runTrainedPositions else {
            SessionLogger.shared.log("[PARAM] error: a Play-and-Train start reached its lineage segment without a positions count")
            return
        }
        ledger.trainerStepAtStart = trainerStep
        runTrainedPositions = ledger
    }

    /// session.json's `trainingPositionsSeen` at the session's `steps`: nil
    /// when steps before the run are unrecorded, or when the stats are no
    /// longer the run's (demo training replaced the box since). A step count
    /// below the start's is a bug and throws. A save happens only with a
    /// run's state, so a missing count is an error, as the missing capture is.
    func trainedPositionsForSessionState(atSessionSteps steps: Int) throws -> Int? {
        guard let ledger = runTrainedPositions else {
            throw LineageSegmentError.noSegment("session.json's positions trained (no run-start positions count)")
        }
        guard ledger.statsBox === trainingBox else { return nil }
        return try ledger.recorded.positions(atSteps: steps)
    }

    /// The status bar's "Positions trained" at the displayed `steps`, nil
    /// ("—") when no run's count describes those steps: no run has started,
    /// the stats are demo training's, or they read below the run's start.
    func trainedPositionsForDisplay(atSessionSteps steps: Int) -> Int? {
        guard let ledger = runTrainedPositions, ledger.statsBox === trainingBox else { return nil }
        do {
            return try ledger.recorded.positions(atSteps: steps)
        } catch {
            // The displayed stats are not the run's (Train Once replaces
            // them without a box): there is no run count for them.
            return nil
        }
    }

    /// The batch size the displayed stats' current segment trained at, for
    /// the Training panel's moves/sec: the run's capture while the stats are
    /// the run's; the demo batch while demo training runs; nil (no rate)
    /// otherwise — stats left behind by a finished demo run, or no run yet.
    func trainedBatchSizeForDisplayedStats() -> Int? {
        if let ledger = runTrainedPositions, let box = trainingBox, ledger.statsBox === box {
            return ledger.recorded.batchSize
        }
        if continuousTraining || isTrainingOnce {
            return Self.trainingBatchSize
        }
        return nil
    }

    /// Positions this process has seen the run train so far, for the
    /// progress-rate chart (only differences matter, so history before the
    /// process is irrelevant). Read from the run's stats box itself, not the
    /// heartbeat's published copy, so a copy one tick behind a Continue can
    /// never read as steps before the start. Throws when no count describes
    /// the stats — inside a run, a bug.
    func observedTrainedPositionsForRate() throws -> Int {
        guard let ledger = runTrainedPositions else {
            throw LineageSegmentError.noSegment("the progress rate (no run-start positions count)")
        }
        guard let box = ledger.statsBox, box === trainingBox else {
            throw LineageSegmentError.noSegment("the progress rate (the stats box is not the run's)")
        }
        return try ledger.observedPositions(atSteps: box.snapshot().stats.steps)
    }

    /// Positions trained behind the trainer's weights at trainer clock
    /// `trainerStep`, for the Lichess probe history and its export (which
    /// pair it with that clock): the run's recorded count, when the session's
    /// steps are the trainer's clock and nothing outside the run has moved
    /// the trainer since. Nil — unrecorded — otherwise:
    /// - no run's count, or a start that has not reached its trainer state;
    /// - a session whose steps are not the trainer's clock ("New Session,
    ///   keep trainer" counts the session from 0 on a trained trainer);
    /// - stats demo training replaced, or a trainer clock that moved while no
    ///   run was stepping (demo training or a sweep on the same trainer);
    /// - steps before the session that no record describes.
    func trainedPositions(atTrainerStep trainerStep: Int) -> Int? {
        guard let ledger = runTrainedPositions,
              let anchor = ledger.trainerStepAtStart,
              ledger.recorded.stepsAtStart == anchor,
              let box = trainingBox, ledger.statsBox === box,
              trainerStep >= anchor else { return nil }
        if !realTraining {
            // Nothing is stepping, so the clock and the stats hold still and
            // must have advanced together since the start.
            let sessionSteps = box.snapshot().stats.steps
            guard sessionSteps - ledger.recorded.stepsAtStart == trainerStep - anchor else { return nil }
        }
        return ledger.recorded.positionsBeforeStart.map { $0 + (trainerStep - anchor) * ledger.recorded.batchSize }
    }
}
