import XCTest
@testable import DrewsChessMachine

/// The shared step-line schedule (`TrainingStepLineSchedule`) and the
/// trainer's diagnostics rule it relies on (`ChessTrainer.isDiagnosticsStep`
/// / `isBatchStatsStep`): lines keyed on the trainer step, every line after a
/// segment's first carrying diagnostics for any `batch_stats_interval` and
/// any resume point, and the default interval computing diagnostics on
/// exactly the steps it did before.
final class TrainingStepLineScheduleTests: XCTestCase {

    // MARK: - Fixed steps

    func testFixedLineStepsAreEvery50ThroughStep1000ThenEvery1000() {
        XCTAssertFalse(TrainingStepLineSchedule.isFixedLineStep(trainerStep: 0), "0 is never a line step")
        XCTAssertFalse(TrainingStepLineSchedule.isFixedLineStep(trainerStep: -50))
        for step in 1...20_000 {
            let expected = (step <= 1_000 && step % 50 == 0) || step % 1_000 == 0
            XCTAssertEqual(TrainingStepLineSchedule.isFixedLineStep(trainerStep: step), expected, "step \(step)")
        }
        XCTAssertEqual(TrainingStepLineSchedule.firstFixedLineStep(after: 0), 50)
        XCTAssertEqual(TrainingStepLineSchedule.firstFixedLineStep(after: 50), 100)
        XCTAssertEqual(TrainingStepLineSchedule.firstFixedLineStep(after: 999), 1_000)
        XCTAssertEqual(TrainingStepLineSchedule.firstFixedLineStep(after: 1_000), 2_000)
        XCTAssertEqual(TrainingStepLineSchedule.firstFixedLineStep(after: 1_513), 2_000)
        XCTAssertEqual(TrainingStepLineSchedule.firstCheckpointStep(after: 0), 1_000)
        XCTAssertEqual(TrainingStepLineSchedule.firstCheckpointStep(after: 513), 1_000)
        XCTAssertEqual(TrainingStepLineSchedule.firstCheckpointStep(after: 1_000), 2_000)
        XCTAssertEqual(TrainingStepLineSchedule.firstCheckpointStep(after: 36_000), 37_000)
    }

    func testEveryCheckpointStepIsAFixedLineStep() {
        for step in 1...100_000 where TrainingStepLineSchedule.isCheckpointStep(trainerStep: step) {
            XCTAssertTrue(TrainingStepLineSchedule.isFixedLineStep(trainerStep: step), "step \(step)")
            XCTAssertEqual(step % 1_000, 0, "step \(step)")
        }
        XCTAssertFalse(TrainingStepLineSchedule.isCheckpointStep(trainerStep: 0))
    }

    // MARK: - The CLI simulation

    /// One simulated segment as the CLI runners observe it: every step, with
    /// `carriesDiagnostics` from the trainer's own rule.
    private struct SimulatedLine {
        let trainerStep: Int
        let elapsedSec: Double
        let carriesDiagnostics: Bool
        let reason: TrainingStepLineSchedule.Reason
    }

    private func simulate(resumeOffset: Int, batchStatsInterval: Int, segmentSteps: Int,
                          stepDuration: Double, intervalSec: Double) -> [SimulatedLine] {
        var schedule = TrainingStepLineSchedule()
        var lines: [SimulatedLine] = []
        for segmentStep in 1...segmentSteps {
            let trainerStep = resumeOffset + segmentStep
            let elapsed = Double(segmentStep) * stepDuration
            let diagnostics = ChessTrainer.isDiagnosticsStep(trainerStep: trainerStep,
                                                             batchStatsInterval: batchStatsInterval)
            if let reason = schedule.lineDue(trainerStep: trainerStep, elapsedSec: elapsed,
                                             carriesDiagnostics: diagnostics, intervalSec: intervalSec) {
                lines.append(SimulatedLine(trainerStep: trainerStep, elapsedSec: elapsed,
                                           carriesDiagnostics: diagnostics, reason: reason))
            }
        }
        return lines
    }

    func testEveryLineAfterTheFirstCarriesDiagnosticsAfterAResume() {
        let segmentSteps = 3_000
        for offset in [0, 7, 513, 999, 36_000] {
            for batchStatsInterval in [0, 7, 10, 25, 50, 100] {
                for duration in [0.1, 2.07, 9.0] {
                    for interval in [60.0, 180.0] {
                        let what = "offset \(offset) bsi \(batchStatsInterval) dur \(duration) interval \(interval)"
                        let lines = simulate(resumeOffset: offset, batchStatsInterval: batchStatsInterval,
                                             segmentSteps: segmentSteps, stepDuration: duration, intervalSec: interval)
                        XCTAssertEqual(lines.first?.trainerStep, offset + 1, "\(what): the segment's first step")
                        XCTAssertEqual(lines.first?.reason, .segmentStart, what)
                        for line in lines.dropFirst() where !line.carriesDiagnostics {
                            XCTFail("\(what): the line at trainer step \(line.trainerStep) lacks diagnostics")
                        }
                        let logged = Set(lines.map(\.trainerStep))
                        for step in (offset + 1)...(offset + segmentSteps)
                        where TrainingStepLineSchedule.isFixedLineStep(trainerStep: step) && !logged.contains(step) {
                            XCTFail("\(what): fixed line step \(step) has no line")
                        }
                        let diagnosticsInterval = ChessTrainer.diagnosticsInterval(batchStatsInterval: batchStatsInterval)
                        let timeBound = interval + Double(diagnosticsInterval) * duration
                        for (earlier, later) in zip(lines, lines.dropFirst()) {
                            let stepBound = later.trainerStep <= 1_000 ? 50 : 1_000
                            XCTAssertLessThanOrEqual(later.trainerStep - earlier.trainerStep, stepBound,
                                                     "\(what): lines at \(earlier.trainerStep) and \(later.trainerStep)")
                            XCTAssertLessThanOrEqual(later.elapsedSec - earlier.elapsedSec, timeBound + 1e-9,
                                                     "\(what): lines at \(earlier.trainerStep) and \(later.trainerStep)")
                        }
                    }
                }
            }
        }
    }

    func testResumedAndUninterruptedRunsLogTheSameFixedSteps() {
        let total = 5_000
        let uninterrupted = simulate(resumeOffset: 0, batchStatsInterval: 10, segmentSteps: total,
                                     stepDuration: 0.5, intervalSec: 86_400)
        for k in [7, 513, 999, 1_000, 3_333] {
            let resumed = simulate(resumeOffset: k, batchStatsInterval: 10, segmentSteps: total - k,
                                   stepDuration: 0.5, intervalSec: 86_400)
            let resumedFixed = resumed.filter { $0.reason == .fixedStep }.map(\.trainerStep)
            let expected = uninterrupted.filter { $0.reason == .fixedStep && $0.trainerStep > k + 1 }.map(\.trainerStep)
            XCTAssertEqual(resumedFixed, expected, "resume at \(k)")
        }
    }

    // MARK: - Rules

    func testTheFirstObservationIsALine() {
        var schedule = TrainingStepLineSchedule()
        XCTAssertEqual(schedule.lineDue(trainerStep: 514, elapsedSec: 0, carriesDiagnostics: false, intervalSec: 180),
                       .segmentStart)
        XCTAssertNil(schedule.lineDue(trainerStep: 515, elapsedSec: 1, carriesDiagnostics: true, intervalSec: 180))
    }

    func testTheIntervalLineWaitsForADiagnosticsStep() {
        var schedule = TrainingStepLineSchedule()
        _ = schedule.lineDue(trainerStep: 2_001, elapsedSec: 0, carriesDiagnostics: false, intervalSec: 180)
        XCTAssertNil(schedule.lineDue(trainerStep: 2_002, elapsedSec: 179.9, carriesDiagnostics: true, intervalSec: 180))
        XCTAssertNil(schedule.lineDue(trainerStep: 2_003, elapsedSec: 200, carriesDiagnostics: false, intervalSec: 180),
                     "past the deadline, but no diagnostics on this step")
        XCTAssertEqual(schedule.lineDue(trainerStep: 2_010, elapsedSec: 210, carriesDiagnostics: true, intervalSec: 180),
                       .interval)
        XCTAssertNil(schedule.lineDue(trainerStep: 2_020, elapsedSec: 220, carriesDiagnostics: true, intervalSec: 180))
    }

    func testAnyLineRestartsTheInterval() {
        var schedule = TrainingStepLineSchedule()
        XCTAssertEqual(schedule.lineDue(trainerStep: 1, elapsedSec: 0, carriesDiagnostics: false, intervalSec: 180),
                       .segmentStart)
        XCTAssertEqual(schedule.lineDue(trainerStep: 50, elapsedSec: 100, carriesDiagnostics: true, intervalSec: 180),
                       .fixedStep)
        XCTAssertNil(schedule.lineDue(trainerStep: 60, elapsedSec: 181, carriesDiagnostics: true, intervalSec: 180),
                     "181 s after the start but only 81 s after the fixed line")
        XCTAssertEqual(schedule.lineDue(trainerStep: 70, elapsedSec: 280, carriesDiagnostics: true, intervalSec: 180),
                       .interval)
        XCTAssertNil(schedule.lineDue(trainerStep: 80, elapsedSec: 400, carriesDiagnostics: true, intervalSec: 180),
                     "the interval line restarted the interval too")
        // The interval is read on every call (the GUI reads it live).
        XCTAssertEqual(schedule.lineDue(trainerStep: 90, elapsedSec: 400, carriesDiagnostics: true, intervalSec: 60),
                       .interval)
    }

    func testAPolledScheduleLinesUpOnTheFirstObservationPastEachFixedStep() {
        var schedule = TrainingStepLineSchedule()
        let observations: [(Int, TrainingStepLineSchedule.Reason?)] = [
            (3, .segmentStart), (47, nil), (52, .fixedStep), (980, .fixedStep), (980, nil), (1_004, .fixedStep),
            (1_500, nil), (1_999, nil), (2_001, .fixedStep), (2_001, nil), (5_500, .fixedStep),
        ]
        for (index, (step, expected)) in observations.enumerated() {
            XCTAssertEqual(schedule.lineDue(trainerStep: step, elapsedSec: Double(index), carriesDiagnostics: false,
                                            intervalSec: 86_400),
                           expected, "observation \(index) at step \(step)")
        }
    }

    func testATrainerClockRewindIsNotAFixedLine() {
        var schedule = TrainingStepLineSchedule()
        XCTAssertEqual(schedule.lineDue(trainerStep: 1_200, elapsedSec: 0, carriesDiagnostics: false, intervalSec: 86_400),
                       .segmentStart)
        XCTAssertNil(schedule.lineDue(trainerStep: 1_990, elapsedSec: 1, carriesDiagnostics: false, intervalSec: 86_400))
        // A promotion rewinds the trainer clock to the arena's start.
        XCTAssertNil(schedule.lineDue(trainerStep: 900, elapsedSec: 2, carriesDiagnostics: false, intervalSec: 86_400),
                     "the rewind itself is not a fixed line")
        XCTAssertNil(schedule.lineDue(trainerStep: 940, elapsedSec: 3, carriesDiagnostics: false, intervalSec: 86_400))
        XCTAssertEqual(schedule.lineDue(trainerStep: 950, elapsedSec: 4, carriesDiagnostics: false, intervalSec: 86_400),
                       .fixedStep, "a fixed step re-crossed after the rewind gets its line again")
        XCTAssertEqual(schedule.lineDue(trainerStep: 1_000, elapsedSec: 5, carriesDiagnostics: false, intervalSec: 86_400),
                       .fixedStep)
    }

    // MARK: - The trainer's rule

    func testTheTrainerComputesDiagnosticsOnEveryFixedLineStep() {
        for interval in [7, 100] {
            for step in 1...5_000 where TrainingStepLineSchedule.isFixedLineStep(trainerStep: step) {
                XCTAssertTrue(ChessTrainer.isDiagnosticsStep(trainerStep: step, batchStatsInterval: interval),
                              "interval \(interval) step \(step)")
                XCTAssertTrue(ChessTrainer.isBatchStatsStep(trainerStep: step, batchStatsInterval: interval),
                              "interval \(interval) step \(step)")
            }
            for step in 1...5_000 where step % interval == 0 {
                XCTAssertTrue(ChessTrainer.isDiagnosticsStep(trainerStep: step, batchStatsInterval: interval))
                XCTAssertTrue(ChessTrainer.isBatchStatsStep(trainerStep: step, batchStatsInterval: interval))
            }
        }
    }

    func testTheDefaultIntervalComputesDiagnosticsOnTheSameStepsAsBefore() {
        for step in 1...5_000 {
            XCTAssertEqual(ChessTrainer.isDiagnosticsStep(trainerStep: step, batchStatsInterval: 10), step % 10 == 0,
                           "interval 10 step \(step)")
            XCTAssertEqual(ChessTrainer.isBatchStatsStep(trainerStep: step, batchStatsInterval: 10), step % 10 == 0,
                           "interval 10 step \(step)")
            XCTAssertEqual(ChessTrainer.isDiagnosticsStep(trainerStep: step, batchStatsInterval: 0), step % 10 == 0,
                           "interval 0 (fallback 10) step \(step)")
            XCTAssertFalse(ChessTrainer.isBatchStatsStep(trainerStep: step, batchStatsInterval: 0),
                           "interval 0 step \(step)")
        }
        XCTAssertEqual(ChessTrainer.diagnosticsFallbackInterval, 10)
    }
}
