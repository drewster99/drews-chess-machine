//
//  ResumeDiffAbsentValueTests.swift
//  DrewsChessMachineTests
//
//  A CLI exact resume's `[RESUME-DIFF]` comparison reads a key the parent's
//  snapshot predates through the key's declared `absentValue`, as the GUI
//  resume does: a parent that predates a `.preFeature` (or
//  `.declaredRangeMaximum`) key trained at that value, so a run still at it
//  changed nothing, and one that differs is reported against it. A key with
//  no known pre-feature value, and an operational setting, are reported as
//  what they are.
//

import XCTest
@testable import DrewsChessMachine

final class ResumeDiffAbsentValueTests: XCTestCase {

    /// A parent snapshot of every declared default, without `removed`.
    private func parentWithout(_ removed: [String]) throws -> LineageRecord.Parameters {
        var values = try TrainingParametersSnapshot.declaredDefaults(overriding: [:]).rawValueMap()
        for id in removed { values[id] = nil }
        return try LineageRecord.Parameters(values: values)
    }

    // MARK: - Regression

    func testAKeyAtItsPreFeatureValueIsNotADifference() throws {
        let parent = try parentWithout([DropoutRate.id, MaxDrawPercentPerBatch.id, MaxPliesFromAnyOneGame.id])
        let thisRun = try TrainingParametersSnapshot.declaredDefaults(overriding: [
            DropoutRate.id: .double(0.0),
            MaxDrawPercentPerBatch.id: .int(100),
            MaxPliesFromAnyOneGame.id: .int(400),
        ])
        XCTAssertEqual(try thisRun.differences(fromLineage: parent).map(\.logLine), [],
                       "the parent trained at each pre-feature value, which this run keeps")
    }

    // MARK: - Line forms

    func testAKeyAwayFromItsPreFeatureValueNamesIt() throws {
        let parent = try parentWithout([DropoutRate.id, MaxPliesFromAnyOneGame.id])
        let thisRun = try TrainingParametersSnapshot.declaredDefaults(overriding: [
            DropoutRate.id: .double(0.1),
            MaxPliesFromAnyOneGame.id: .int(10),
        ])
        let differences = try thisRun.differences(fromLineage: parent)
        XCTAssertEqual(differences.map(\.id), [DropoutRate.id, MaxPliesFromAnyOneGame.id].sorted())
        let dropout = try XCTUnwrap(differences.first { $0.id == DropoutRate.id })
        XCTAssertEqual(dropout.presence, .thisRunOnly)
        XCTAssertEqual(dropout.parentAbsence, .preFeatureValue(ParameterValue.double(0.0).displayText))
        XCTAssertTrue(dropout.logLine.hasPrefix(
            "[RESUME-DIFF] dropout_rate: parent=absent(pre-feature \(ParameterValue.double(0.0).displayText)) this_run=0.1"),
            dropout.logLine)
        let maxPlies = try XCTUnwrap(differences.first { $0.id == MaxPliesFromAnyOneGame.id })
        XCTAssertEqual(maxPlies.parentAbsence, .preFeatureValue("400"), "the declared range maximum stands in for unbounded")
    }

    func testAnOperationalSettingAndAnUnknownTrainingValueAreReportedAsSuch() throws {
        let parent = try parentWithout([ReplayBufferMinPositionsBeforeTraining.id, WeightDecay.id])
        let differences = try TrainingParametersSnapshot.declaredDefaults(overriding: [:]).differences(fromLineage: parent)
        XCTAssertEqual(differences.map(\.id), [ReplayBufferMinPositionsBeforeTraining.id, WeightDecay.id].sorted())
        let fill = try XCTUnwrap(differences.first { $0.id == ReplayBufferMinPositionsBeforeTraining.id })
        XCTAssertEqual(fill.parentAbsence, .operationalSetting)
        XCTAssertTrue(fill.logLine.contains("operational setting"), fill.logLine)
        let weightDecay = try XCTUnwrap(differences.first { $0.id == WeightDecay.id })
        XCTAssertEqual(weightDecay.parentAbsence, .unknownTrainingValue)
        XCTAssertTrue(weightDecay.logLine.contains("not recorded"), weightDecay.logLine)
    }

    func testOnlyAKeyThisRunAloneHoldsHasAnAbsence() throws {
        let parent = try LineageRecord.Parameters(values: TrainingParametersSnapshot.declaredDefaults(overriding: [
            WeightDecay.id: .double(0.0005),
        ]).rawValueMap())
        let differences = try TrainingParametersSnapshot.declaredDefaults(overriding: [:]).differences(fromLineage: parent)
        XCTAssertEqual(differences.map(\.id), [WeightDecay.id])
        XCTAssertNil(differences.first?.parentAbsence, "both snapshots hold the key")
    }
}
