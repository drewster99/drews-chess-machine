//
//  AdvantagePowerStepLineTests.swift
//  DrewsChessMachineTests
//
//  The step lines' `advRMS=` / `advFloored=` window (`AdvantagePowerStepLine`):
//  its per-window math, the floor comparison at the boundary, the empty
//  window, the reset at each line, and the rewind that drops a discarded
//  trajectory's steps.
//

import XCTest
@testable import DrewsChessMachine

final class AdvantagePowerStepLineTests: XCTestCase {

    private let floor = Float(ChessTrainer.advantagePowerFloor)

    /// Protects: a line reports mean / min / max of the per-step RMS
    /// (sqrt of the mean square, not the mean square) and the floored count,
    /// in the logged field format dashboards parse by key.
    /// Fails if: the RMS is taken of the mean of mean squares, min/max swap,
    /// the count uses the wrong comparison, or the field text changes.
    func testReadingSummarizesPerStepRMSAndFlooredCount() throws {
        var window = AdvantagePowerStepLineWindow(floor: floor)
        try window.record(trainerStep: 11, meanSquare: 0.25)    // RMS 0.5
        try window.record(trainerStep: 12, meanSquare: 0.0625)  // RMS 0.25
        try window.record(trainerStep: 13, meanSquare: 1.0)     // RMS 1.0
        try window.record(trainerStep: 14, meanSquare: 0.01)    // RMS 0.1, floored

        let reading = window.take(throughTrainerStep: 14)

        XCTAssertEqual(reading.steps, 4)
        XCTAssertEqual(reading.flooredSteps, 1)
        XCTAssertEqual(try XCTUnwrap(reading.rmsMean), (0.5 + 0.25 + 1.0 + 0.1) / 4, accuracy: 1e-6)
        XCTAssertEqual(try XCTUnwrap(reading.rmsMin), 0.1, accuracy: 1e-6)
        XCTAssertEqual(try XCTUnwrap(reading.rmsMax), 1.0, accuracy: 1e-12)
        XCTAssertEqual(reading.logFields, " advRMS=0.4625/0.1000/1.0000 advFloored=1/4")
    }

    /// Protects: `advFloored=` counts exactly the steps whose mean square is
    /// below the floor the graph applies (`max(E[A²], floor)` changes nothing
    /// at equality).
    /// Fails if: the comparison becomes `<=`, or the window compares against
    /// a floor other than the trainer's constant.
    func testFloorCountAtJustBelowAndJustAboveTheFloor() throws {
        let cases: [(name: String, meanSquare: Float, floored: Int)] = [
            ("just below", floor.nextDown, 1),
            ("at", floor, 0),
            ("just above", floor.nextUp, 0),
        ]
        for testCase in cases {
            var window = AdvantagePowerStepLineWindow(floor: Float(ChessTrainer.advantagePowerFloor))
            try window.record(trainerStep: 1, meanSquare: testCase.meanSquare)
            let reading = window.take(throughTrainerStep: 1)
            XCTAssertEqual(reading.flooredSteps, testCase.floored, testCase.name)
            XCTAssertEqual(reading.steps, 1, testCase.name)
        }
    }

    /// Protects: a line with no step since the previous one reports
    /// `advRMS=none advFloored=0/0` rather than zeros or NaN.
    /// Fails if: an empty window divides by zero or prints a numeric RMS.
    func testEmptyWindowReportsNone() {
        var window = AdvantagePowerStepLineWindow(floor: floor)
        let reading = window.take(throughTrainerStep: 100)
        XCTAssertEqual(reading.steps, 0)
        XCTAssertNil(reading.rmsMean)
        XCTAssertNil(reading.rmsMin)
        XCTAssertNil(reading.rmsMax)
        XCTAssertEqual(reading.logFields, " advRMS=none advFloored=0/0")
    }

    /// Protects: each line covers only the steps since the previous line, and
    /// a step above the line's clock (a GUI step that landed after the line
    /// read it) waits for the next line instead of being lost or doubled.
    /// Fails if: `take` keeps reported steps, or drops/reports steps above
    /// its bound.
    func testTakeResetsAtTheLineAndKeepsLaterSteps() throws {
        var window = AdvantagePowerStepLineWindow(floor: floor)
        try window.record(trainerStep: 1, meanSquare: 0.25)
        try window.record(trainerStep: 2, meanSquare: 0.01)
        try window.record(trainerStep: 3, meanSquare: 1.0)

        let first = window.take(throughTrainerStep: 2)
        XCTAssertEqual(first.steps, 2)
        XCTAssertEqual(first.flooredSteps, 1)

        let second = window.take(throughTrainerStep: 3)
        XCTAssertEqual(second.steps, 1)
        XCTAssertEqual(second.flooredSteps, 0)
        XCTAssertEqual(try XCTUnwrap(second.rmsMean), 1.0, accuracy: 1e-12)

        XCTAssertEqual(window.take(throughTrainerStep: 3).steps, 0)
    }

    /// Protects: a rewind (promotion to the arena start, a restored clock)
    /// drops the pending steps of the discarded trajectory and keeps the
    /// steps at or below the new clock; the re-trained step numbers then
    /// record and report once.
    /// Fails if: a rewind keeps discarded steps (they'd be reported, and the
    /// re-trained steps would be refused), or drops steps the kept weights
    /// trained.
    func testRewindDropsStepsAboveTheClockAndAllowsRetrainingThem() throws {
        var window = AdvantagePowerStepLineWindow(floor: floor)
        try window.record(trainerStep: 5, meanSquare: 0.25)
        try window.record(trainerStep: 6, meanSquare: 0.01)
        try window.record(trainerStep: 7, meanSquare: 0.01)

        window.rewind(toTrainerStep: 5)
        try window.record(trainerStep: 6, meanSquare: 1.0)

        let reading = window.take(throughTrainerStep: 6)
        XCTAssertEqual(reading.steps, 2)
        XCTAssertEqual(reading.flooredSteps, 0)
        XCTAssertEqual(try XCTUnwrap(reading.rmsMin), 0.5, accuracy: 1e-12)
        XCTAssertEqual(try XCTUnwrap(reading.rmsMax), 1.0, accuracy: 1e-12)
    }

    /// Protects: a rewind below every pending step (a fresh network's clock 0)
    /// empties the window.
    /// Fails if: the "no step at or below the clock" case keeps entries.
    func testRewindBelowEveryPendingStepEmptiesTheWindow() throws {
        var window = AdvantagePowerStepLineWindow(floor: floor)
        try window.record(trainerStep: 40, meanSquare: 0.25)
        try window.record(trainerStep: 41, meanSquare: 0.25)
        window.rewind(toTrainerStep: 0)
        XCTAssertEqual(window.take(throughTrainerStep: 41).steps, 0)
    }

    /// Protects: a step recorded at or below a pending one without a rewind
    /// is refused, so two trajectories never mix in one line.
    /// Fails if: `record` accepts a repeated or earlier trainer step.
    func testRecordingANonIncreasingStepWithoutRewindThrows() throws {
        var window = AdvantagePowerStepLineWindow(floor: floor)
        try window.record(trainerStep: 9, meanSquare: 0.25)
        XCTAssertThrowsError(try window.record(trainerStep: 9, meanSquare: 0.25))
        XCTAssertThrowsError(try window.record(trainerStep: 8, meanSquare: 0.25))
    }
}
