//
//  UntrainedCopyRecordTests.swift
//  DrewsChessMachineTests
//
//  The record of a model made from a file without training (a derive, a
//  graft, a GUI save of a loaded model) continues the source's totals only
//  where the source's own lineage recorded them. A file written before
//  lineage still states a trainer clock, but a resumed segment of that era
//  restarted its clock, so that number is segment-local, not the line's
//  total: it stays the parent's stated step and the copy's total stays
//  unrecorded.
//

import XCTest
@testable import DrewsChessMachine

final class UntrainedCopyRecordTests: XCTestCase {

    func testCopyOfAPreLineageFileLeavesTheStepTotalUnrecorded() throws {
        let source = LineageTracker.ParentFile(
            modelID: "20260801-1-PREL", contentSHA256: nil, trainerCompletedSteps: 300_000,
            lineage: .unrecorded(formatVersion: 6), derivationHistory: [])
        let record = try LineageTracker.untrainedCopyRecord(
            source: source, derivation: nil, sourceArchitecture: nil, naming: .unrecorded, pathKind: .derive, argv: ["test"],
            at: Date(timeIntervalSince1970: 1_790_000_000))
        XCTAssertNil(record.steps.cumTrainerStep)
        XCTAssertEqual(record.parent?.trainerCompletedSteps, 300_000)
        XCTAssertTrue(record.run.continuesUnrecordedHistory)
    }

    func testCopyOfARecordedFileContinuesItsStepTotal() throws {
        let source = LineageTracker.ParentFile(
            modelID: "20261003-1-RECD", contentSHA256: nil, trainerCompletedSteps: 1200,
            lineage: .recorded(try LineageRecord.forTests(trainerCompletedSteps: 1200, corpus: nil)),
            derivationHistory: [])
        let record = try LineageTracker.untrainedCopyRecord(
            source: source, derivation: nil, sourceArchitecture: nil, naming: .unrecorded, pathKind: .derive, argv: ["test"],
            at: Date(timeIntervalSince1970: 1_790_000_000))
        XCTAssertEqual(record.steps.cumTrainerStep, 1200)
    }
}
