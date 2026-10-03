//
//  BehaviorFingerprintTests.swift
//  DrewsChessMachineTests
//
//  Determinism plan C1 #33 as refined by the owner: a changed build or OS is
//  a resume gap only when it changes what the run computes, which the
//  behavior fingerprint detects. Same build ⇒ same fingerprint; perturbing a
//  covered behavior ⇒ a different one; a different recipe never compares
//  equal.
//

import XCTest
@testable import DrewsChessMachine

final class BehaviorFingerprintTests: XCTestCase {

    private let fp32 = BehaviorFingerprint.Settings(
        inputEncoding: .basic30, computeDataType: .float32, policyTailPrecision: .float32FromPreBatchNorm)

    /// Run streams named under another derivation: every stream's master
    /// seed is shifted, as a changed derivation would shift them.
    private enum ShiftedStreams: FingerprintStreamSource {
        static func streams(masterSeed: UInt64) -> DCMRandomStreams {
            DCMRandomStreams(masterSeed: masterSeed ^ 0x1)
        }
    }

    func testTheSameBuildComputesTheSameFingerprint() async throws {
        let first = try await BehaviorFingerprint.computeUncached(for: fp32, streamDerivation: DCMRandomStreams.self)
        let second = try await BehaviorFingerprint.computeUncached(for: fp32, streamDerivation: DCMRandomStreams.self)
        XCTAssertEqual(first, second)
        XCTAssertEqual(first.recipe, BehaviorFingerprint.recipe)
        XCTAssertEqual(first.sha256.count, 64)
        let cached = try await BehaviorFingerprint.compute(for: fp32)
        XCTAssertEqual(cached, first)
    }

    func testAnotherStreamDerivationChangesTheFingerprint() async throws {
        let base = try await BehaviorFingerprint.computeUncached(for: fp32, streamDerivation: DCMRandomStreams.self)
        let shifted = try await BehaviorFingerprint.computeUncached(for: fp32, streamDerivation: ShiftedStreams.self)
        XCTAssertNotEqual(base.sha256, shifted.sha256)
    }

    func testNumericsSettingsChangeTheFingerprint() async throws {
        let base = try await BehaviorFingerprint.computeUncached(for: fp32, streamDerivation: DCMRandomStreams.self)
        let bf16 = BehaviorFingerprint.Settings(inputEncoding: .basic30, computeDataType: .bFloat16,
                                                policyTailPrecision: .float32FromPreBatchNorm)
        let bf16Fingerprint = try await BehaviorFingerprint.computeUncached(for: bf16, streamDerivation: DCMRandomStreams.self)
        XCTAssertNotEqual(base.sha256, bf16Fingerprint.sha256)
        let otherTail = try await BehaviorFingerprint.computeUncached(
            for: BehaviorFingerprint.Settings(inputEncoding: .basic30, computeDataType: .bFloat16,
                                              policyTailPrecision: .mixedFinalProjection),
            streamDerivation: DCMRandomStreams.self)
        XCTAssertNotEqual(bf16Fingerprint.sha256, otherTail.sha256)
        let otherEncoding = try await BehaviorFingerprint.computeUncached(
            for: BehaviorFingerprint.Settings(inputEncoding: .basic20, computeDataType: .float32,
                                              policyTailPrecision: .float32FromPreBatchNorm),
            streamDerivation: DCMRandomStreams.self)
        XCTAssertNotEqual(base.sha256, otherEncoding.sha256)
    }

    // MARK: - The resume decision

    private func record(fingerprint: BehaviorFingerprint.Record?) throws -> LineageRecord {
        let start = Date(timeIntervalSince1970: 1_790_000_000)
        let tracker = try LineageTracker(start: .fresh(initialization: .forTests), pathKind: .replay, argv: ["dcm"],
                                         startedAt: start, segmentStartTrainerStep: 0)
        return try tracker.record(at: start, trainerCompletedSteps: 1, segmentLocalStep: 1, segmentGames: 0,
                                  segmentPositions: 0, corpus: nil, parameters: nil,
                                  rng: LineageRecord.RNG(dropoutPhiloxState: nil, streams: nil,
                                                         behaviorFingerprint: fingerprint))
    }

    private func laterBuild(than build: LineageRecord.Build) -> LineageRecord.Build {
        LineageRecord.Build(buildNumber: build.buildNumber + 1, gitHash: build.gitHash + "x",
                            gitBranch: build.gitBranch, gitDirty: build.gitDirty)
    }

    func testARebuildWithAMatchingFingerprintIsNotAGap() throws {
        let fingerprint = BehaviorFingerprint.Record(recipe: BehaviorFingerprint.recipe, sha256: "ab12")
        let saved = try record(fingerprint: fingerprint)
        let comparison = ResumeGap.environmentGaps(writtenBy: saved, runningBuild: laterBuild(than: saved.build),
                                                   runningDevice: saved.device, runningFingerprint: fingerprint)
        XCTAssertEqual(comparison.gaps, [])
        XCTAssertEqual(comparison.logLines.count, 1)
        XCTAssertTrue(comparison.logLines[0].contains("behavior fingerprint matches"), comparison.logLines[0])

        let otherOS = LineageRecord.Device(hardwareModel: saved.device.hardwareModel, cpu: saved.device.cpu,
                                           isVirtualMachine: saved.device.isVirtualMachine,
                                           osVersion: saved.device.osVersion + " (later)", gpu: saved.device.gpu)
        XCTAssertEqual(ResumeGap.environmentGaps(writtenBy: saved, runningBuild: saved.build, runningDevice: otherOS,
                                                 runningFingerprint: fingerprint).gaps, [])
    }

    func testADifferentMissingOrOtherRecipeFingerprintIsAGap() throws {
        let running = BehaviorFingerprint.Record(recipe: BehaviorFingerprint.recipe, sha256: "ab12")
        let cases: [BehaviorFingerprint.Record?] = [
            BehaviorFingerprint.Record(recipe: BehaviorFingerprint.recipe, sha256: "cd34"),
            nil,
            BehaviorFingerprint.Record(recipe: BehaviorFingerprint.recipe + 1, sha256: "ab12"),
        ]
        for savedFingerprint in cases {
            let saved = try record(fingerprint: savedFingerprint)
            let comparison = ResumeGap.environmentGaps(writtenBy: saved, runningBuild: laterBuild(than: saved.build),
                                                       runningDevice: saved.device, runningFingerprint: running)
            XCTAssertEqual(comparison.gaps, [.build], "\(String(describing: savedFingerprint))")
        }
    }

    func testTheFingerprintRoundTripsInTheRecord() throws {
        let fingerprint = BehaviorFingerprint.Record(recipe: BehaviorFingerprint.recipe, sha256: "ef56")
        let saved = try record(fingerprint: fingerprint)
        let text = try saved.jsonText()
        XCTAssertTrue(text.contains("\"behavior_fingerprint\""), text)
        XCTAssertEqual(try LineageRecord.decode(jsonText: text).rng.behaviorFingerprint, fingerprint)
    }
}
