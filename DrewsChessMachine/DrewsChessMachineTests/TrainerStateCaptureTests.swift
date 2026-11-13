//
//  TrainerStateCaptureTests.swift
//  DrewsChessMachineTests
//
//  The resume's rule for "which architecture captured this trainer state"
//  (`TrainerStateCapture`) and the precision gaps it feeds
//  (`ResumeGap.precisionGaps`; owner decisions 2026-11-12: tokens
//  `compute_data_type` / `policy_tail_precision`, raised whenever the value
//  differs, chained derives walked back to the training save). Pure.
//
//  What each test protects:
//  - a training save compares against its own architecture (no gap), so
//    every existing resume is unchanged;
//  - a derivation that names another segment (a derive made before the run
//    resumed and trained on) is ignored — otherwise every later save of the
//    run would report the old precision change forever;
//  - a malformed file (clock mismatch, unreadable architecture, a captured
//    architecture that differs beyond precision) is refused, never read as
//    "no gap";
//  - the tokens parse from `--accept-inexact`, order after `schedule`, and
//    the log lines name both values and the derive.
//

import XCTest
@testable import DrewsChessMachine

final class TrainerStateCaptureTests: XCTestCase {

    private static let segmentID = "11111111-2222-3333-4444-555555555555"
    private static let clock = 40

    private func bf16() throws -> NetworkArchitecture {
        try ResumeEquivalenceTests.architecture.withComputeDataType(.bFloat16, tail: .float32FromPreBatchNorm)
    }

    private func fp32() throws -> NetworkArchitecture {
        try bf16().withComputeDataType(.float32, tail: nil)
    }

    /// A training save's record (fixture at `clock`, `segmentID`).
    private func trainingRecord() throws -> LineageRecord {
        let fixture = LineageRecord.sessionTestFixture
        return try LineageRecord(
            run: LineageRecord.Run(
                lineageRunID: fixture.run.lineageRunID, segmentIndex: 3, segmentID: Self.segmentID,
                segmentStartedUnix: fixture.run.segmentStartedUnix, start: .resume, exactResume: true, notExactItems: [],
                continuesUnrecordedHistory: false, recordedUnix: fixture.run.recordedUnix),
            parent: fixture.parent,
            steps: LineageRecord.Steps(cumTrainerStep: Self.clock, segmentStartTrainerStep: 30, segmentLocalStep: 10),
            fed: fixture.fed, time: fixture.time, parameters: fixture.parameters, configuration: fixture.configuration,
            runSeeds: fixture.runSeeds, build: fixture.build, invocation: fixture.invocation, device: fixture.device,
            rng: fixture.rng, segments: fixture.segments, ancestry: fixture.ancestry,
            derivationHistory: fixture.derivationHistory, modelNaming: fixture.modelNaming)
    }

    private func derivation(modelID: String, segmentID: String = TrainerStateCaptureTests.segmentID,
                            clock: Int = TrainerStateCaptureTests.clock,
                            sourceArchitectureJSON: String) -> ModelDerivation.DerivationRecord {
        ModelDerivation.DerivationRecord(
            modelID: modelID, parentModelID: "20261112-1-SRCE", sourceFile: "source.safetensors",
            sourceSHA256: String(repeating: "a", count: 64), sourceFormatVersion: ArchitectureFormat.currentVersion,
            createdAtUnix: 1_790_000_000, build: "test",
            operations: [ModelDerivation.OperationRecord(operation: "set-compute-dtype", arguments: ["value": "float32"],
                                                         changedArchitectureFields: [], rewrittenTensors: [])],
            trainerState: ModelDerivation.CarriedTrainerState(
                segmentID: segmentID, trainerCompletedSteps: clock, sourceArchitecture: sourceArchitectureJSON,
                sourceArchitectureFormatVersion: ArchitectureFormat.currentVersion))
    }

    private func json(_ arch: NetworkArchitecture) throws -> String {
        String(decoding: try JSONEncoder().encode(arch), as: UTF8.self)
    }

    private func derived(_ record: LineageRecord, _ entries: [ModelDerivation.DerivationRecord]) throws -> LineageRecord {
        var result = record
        for entry in entries {
            result = try result.carryingTrainerState(through: entry, naming: result.modelNaming)
        }
        return result
    }

    // MARK: - The capture

    func testATrainingSaveIsCapturedUnderItsOwnArchitecture() throws {
        let capture = try TrainerStateCapture(fileArchitecture: try bf16(), record: try trainingRecord())
        XCTAssertEqual(capture.architecture, try bf16())
        XCTAssertEqual(capture.derivations, [])
        XCTAssertEqual(ResumeGap.precisionGaps(capture: capture, runningArchitecture: try bf16()),
                       EnvironmentComparison(gaps: [], logLines: []))
    }

    func testADerivedFileIsCapturedUnderItsSourcesArchitecture() throws {
        let entry = derivation(modelID: "20261112-2-FP32", sourceArchitectureJSON: try json(try bf16()))
        let record = try derived(try trainingRecord(), [entry])
        let capture = try TrainerStateCapture(fileArchitecture: try fp32(), record: record)
        XCTAssertEqual(capture.architecture, try bf16())
        XCTAssertEqual(capture.derivations, [entry])

        let comparison = ResumeGap.precisionGaps(capture: capture, runningArchitecture: try fp32())
        XCTAssertEqual(comparison.gaps, [.computeDataType, .policyTailPrecision])
        XCTAssertEqual(comparison.logLines, [
            "[RESUME] compute_data_type: the trainer state was captured under bfloat16; this file computes in float32 "
                + "(changed by derive 20261112-2-FP32)",
            "[RESUME] policy_tail_precision: the trainer state was captured under fp32_from_pre_bn; this file has "
                + "does_not_apply (changed by derive 20261112-2-FP32)",
        ])
    }

    /// A derive whose entry names an earlier segment: the run resumed from
    /// it and saved again under a new segment ID, so this save captured its
    /// own state under its own architecture.
    func testADeriveOfAnEarlierSegmentIsIgnored() throws {
        let old = derivation(modelID: "20261101-2-OLDD", segmentID: "an earlier segment", clock: 7,
                             sourceArchitectureJSON: try json(try bf16()))
        let record = try derived(try trainingRecord(), [old])
        let capture = try TrainerStateCapture(fileArchitecture: try fp32(), record: record)
        XCTAssertEqual(capture.architecture, try fp32())
        XCTAssertEqual(capture.derivations, [])
    }

    // MARK: - Refusals

    func testAnEntryAtAnotherClockIsRefused() throws {
        let entry = derivation(modelID: "20261112-2-SKEW", clock: Self.clock - 1, sourceArchitectureJSON: try json(try bf16()))
        let record = try derived(try trainingRecord(), [entry])
        XCTAssertThrowsError(try TrainerStateCapture(fileArchitecture: try fp32(), record: record)) { error in
            XCTAssertEqual(error as? TrainerStateCapture.CaptureError,
                           .clockMismatch(derive: "20261112-2-SKEW", recorded: Self.clock - 1, recordClock: Self.clock))
        }
    }

    func testACapturedArchitectureThatDiffersBeyondPrecisionIsRefused() throws {
        var otherCap = try bf16()
        otherCap.blockGroups[0].rezeroAlphaCap = 9
        let entry = derivation(modelID: "20261112-2-WIDE", sourceArchitectureJSON: try json(otherCap))
        let record = try derived(try trainingRecord(), [entry])
        XCTAssertThrowsError(try TrainerStateCapture(fileArchitecture: try fp32(), record: record)) { error in
            XCTAssertEqual(error as? TrainerStateCapture.CaptureError, .changedBeyondPrecision(derive: "20261112-2-WIDE"))
        }
    }

    func testAnUnreadableCapturedArchitectureIsRefused() throws {
        let entry = derivation(modelID: "20261112-2-JUNK", sourceArchitectureJSON: "{not json")
        let record = try derived(try trainingRecord(), [entry])
        XCTAssertThrowsError(try TrainerStateCapture(fileArchitecture: try fp32(), record: record)) { error in
            guard case .unreadableArchitecture(let derive, _)? = error as? TrainerStateCapture.CaptureError else {
                return XCTFail("expected unreadableArchitecture, got \(error)")
            }
            XCTAssertEqual(derive, "20261112-2-JUNK")
        }
    }

    // MARK: - Tokens

    func testThePrecisionTokensParseAndOrderAfterSchedule() throws {
        XCTAssertEqual(try ResumeGap.parseAcceptList("compute_data_type,policy_tail_precision"),
                       [.computeDataType, .policyTailPrecision])
        XCTAssertEqual(ResumeExactness(gaps: [.policyTailPrecision, .computeDataType, .schedule, .build]).tokens,
                       ["build", "schedule", "compute_data_type", "policy_tail_precision"])
        XCTAssertTrue(CommandLineHelp.usageText.contains("compute_data_type"))
        XCTAssertTrue(CommandLineHelp.usageText.contains("policy_tail_precision"))
    }

    /// The derivation record's `trainer_state` is omitted when absent, so a
    /// history written before it reads and writes byte for byte as before.
    func testAHistoryWithoutTrainerStateEncodesWithoutTheKey() throws {
        let plain = ModelDerivation.DerivationRecord(
            modelID: "20261112-2-PLAN", parentModelID: "20261112-1-SRCE", sourceFile: "s.safetensors",
            sourceSHA256: "b", sourceFormatVersion: 12, createdAtUnix: 1, build: "test", operations: [])
        let text = try LineageRecord.derivationHistoryJSON([plain])
        XCTAssertFalse(text.contains("trainer_state"), text)
        XCTAssertEqual(try ModelDerivation.decodeHistory(text), [plain])
        let carried = derivation(modelID: "20261112-2-FP32", sourceArchitectureJSON: try json(try bf16()))
        let carriedText = try LineageRecord.derivationHistoryJSON([carried])
        XCTAssertTrue(carriedText.contains("\"trainer_state\""), carriedText)
        XCTAssertEqual(try ModelDerivation.decodeHistory(carriedText), [carried])
    }
}
