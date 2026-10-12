//
//  LRKeyframeParametersTests.swift
//  DrewsChessMachineTests
//
//  `lr_cycle_keyframes` as a training parameter (LR_KEYFRAMES_PLAN.md):
//  JSON and lineage-snapshot round trips with exact doubles (V-4), and the
//  previous schedule's nine keys in a parameters file converted exactly,
//  partial files filled from the previous defaults, a mixed file refused (V-5).
//

import XCTest
import TrainingParametersMacroSupport
@testable import DrewsChessMachine

final class LRKeyframeParametersTests: XCTestCase {

    private let sample = LRKeyframeSchedule(keyframes: [
        LRKeyframe(step: 0, peak: 1.0, trough: 0.001, momentumAtPeak: 0.85, momentumAtTrough: 0.95),
        LRKeyframe(step: 50_000, peak: 0.1, trough: 0.0003, momentumAtPeak: 0.85, momentumAtTrough: 0.95),
        LRKeyframe(step: 200_000, peak: 0.02, trough: 0.0001, momentumAtPeak: 0.9, momentumAtTrough: 0.95),
    ])

    /// Kept keys a conversion needs, at their declared defaults (so a test
    /// never depends on the machine's saved settings).
    private func declaredDefault(_ id: String) -> ParameterValue? {
        TrainingParameters.allKeys.first { $0.id == id }?.definition.defaultValue
    }

    // MARK: V-4

    func testAKeyframeValueRoundTripsThroughParametersJSONExactly() throws {
        let object: [String: Any] = [LRCycleKeyframes.id: ParameterValue.lrKeyframes(sample).jsonValue]
        let data = try JSONSerialization.data(withJSONObject: object, options: [.sortedKeys])
        let values = try ParameterValue.parametersObject(fromJSON: data, currentValue: declaredDefault)
        XCTAssertEqual(values[LRCycleKeyframes.id], .lrKeyframes(sample))
        try LRCycleKeyframes.definition.validate(.lrKeyframes(sample))
    }

    func testTheDefaultsFileReadsBackToTheDeclaredDefaultsBitForBit() throws {
        let values = try ParameterValue.parametersObject(fromJSON: try TrainingParameters.defaultsJSON(),
                                                         currentValue: declaredDefault)
        XCTAssertEqual(values[LRCycleKeyframes.id], LRCycleKeyframes.definition.defaultValue)
        for id in LegacyLRScheduleKeys.removedIDs {
            XCTAssertNil(values[id], "the defaults file no longer writes \(id)")
        }
    }

    func testTheLineageSnapshotHashIsStableAcrossEncodeAndDecode() throws {
        let parameters = try LineageRecord.Parameters(values: [LRCycleKeyframes.id: .lrKeyframes(sample), "lr_warmup_steps": .int(1000)])
        let encoded = try JSONEncoder().encode(parameters)
        let decoded = try JSONDecoder().decode(LineageRecord.Parameters.self, from: encoded)
        XCTAssertEqual(decoded.sha256, parameters.sha256)
        let values = try ParameterValue.parametersObject(fromJSON: Data(decoded.snapshotJSON.utf8), currentValue: declaredDefault)
        XCTAssertEqual(values[LRCycleKeyframes.id], .lrKeyframes(sample))
    }

    func testAnInvalidScheduleIsRefusedWithTheValidatorsReason() {
        var bad = sample
        bad.keyframes[1].trough = 0.5   // above its peak
        XCTAssertThrowsError(try LRCycleKeyframes.definition.validate(.lrKeyframes(bad))) { error in
            guard case TrainingConfigError.invalidValue(let id, let reason) = error else {
                return XCTFail("expected invalidValue, got \(error)")
            }
            XCTAssertEqual(id, LRCycleKeyframes.id)
            XCTAssertTrue(reason.contains("keyframe 1"), reason)
        }
    }

    // MARK: V-5

    private var repoRoot: URL {
        URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
    }

    /// Every repo parameters file with the previous schedule's keys reads as
    /// `lr_cycle_keyframes`, equal to the exact conversion of its own values;
    /// the nine keys are gone from the result.
    func testEveryRepoLegacyParametersFileConvertsExactly() throws {
        let enumerator = try XCTUnwrap(FileManager.default.enumerator(at: repoRoot, includingPropertiesForKeys: nil))
        var complete = 0
        var partial = 0
        for case let url as URL in enumerator {
            let path = url.path
            if path.contains("/.git/") || path.contains("/DerivedData/") || url.pathExtension != "json"
                || url.lastPathComponent.hasPrefix("results") { continue }
            let data = try Data(contentsOf: url)
            let object: Any
            do {
                object = try JSONSerialization.jsonObject(with: data)
            } catch {
                continue   // not JSON text this test is about
            }
            guard let dictionary = object as? [String: Any], dictionary[LegacyLRScheduleKeys.lrCycleMin] != nil else { continue }
            let values = try ParameterValue.parametersObject(fromJSON: data, currentValue: declaredDefault)
            for id in LegacyLRScheduleKeys.removedIDs {
                XCTAssertNil(values[id], "\(path): \(id) converted away")
            }
            guard case .lrKeyframes(let converted) = values[LRCycleKeyframes.id] else {
                return XCTFail("\(path): no lr_cycle_keyframes after conversion")
            }
            XCTAssertNoThrow(try converted.validated(), path)
            let hasAll = LegacyLRScheduleKeys.removedIDs.allSatisfy { dictionary[$0] != nil }
            if hasAll {
                complete += 1
                // Recompute the conversion from the file's own numbers.
                func number(_ key: String) throws -> Double { try XCTUnwrap((dictionary[key] as? NSNumber)?.doubleValue, "\(path) \(key)") }
                func flag(_ key: String) throws -> Bool { try XCTUnwrap((dictionary[key] as? NSNumber)?.boolValue, "\(path) \(key)") }
                let legacy = LegacyLRSchedule(
                    lrEnabled: try flag("lr_cycle_enabled"), lrPeriodSteps: Int(try number("lr_cycle_period_steps")),
                    lrCount: Int(try number("lr_cycle_count")), lrMin: try number("lr_cycle_min"), lrMax: try number("lr_cycle_max"),
                    lrInvert: try flag("lr_cycle_invert"), momentumEnabled: try flag("momentum_cycle_enabled"),
                    momentumPeriodSteps: Int(try number("momentum_cycle_period_steps")),
                    momentumCount: Int(try number("momentum_cycle_count")), momentumMin: try number("momentum_cycle_min"),
                    momentumMax: try number("momentum_cycle_max"), momentumInvert: try flag("momentum_cycle_invert"),
                    lrPeakEnd: try number("lr_cycle_peak_end"), lrTroughEnd: try number("lr_cycle_trough_end"),
                    decayHorizonSteps: Int(try number("lr_cycle_decay_horizon_steps")),
                    momentumFollowsLRCycle: try flag("momentum_follows_lr_cycle"),
                    momentumFollowStartLow: try number("momentum_follow_start_low"),
                    momentumFollowStartHigh: try number("momentum_follow_start_high"),
                    momentumFollowEndLow: try number("momentum_follow_end_low"),
                    momentumFollowEndHigh: try number("momentum_follow_end_high"),
                    lrWarmupSteps: Int(try number("lr_warmup_steps")))
                XCTAssertEqual(converted, try LRMomentumCycle.converting(legacy).cycle.lrKeyframes, path)
            } else {
                partial += 1
            }
        }
        XCTAssertGreaterThanOrEqual(complete, 30, "complete legacy files found")
        XCTAssertEqual(partial, 2, "the two v5 dashboard records hold only lr_cycle_min / lr_cycle_max")
    }

    func testAPartialLegacyFileTakesTheMissingKeysFromThePreviousDefaults() throws {
        let data = Data(#"{"lr_cycle_min": 0.002, "lr_cycle_max": 0.02, "lr_warmup_steps": 0}"#.utf8)
        let values = try ParameterValue.parametersObject(fromJSON: data, currentValue: declaredDefault)
        guard case .lrKeyframes(let schedule) = values[LRCycleKeyframes.id] else {
            return XCTFail("expected lr_cycle_keyframes")
        }
        // Previous defaults: horizon 1,000,000, peak end 1e-4, trough end 1e-6,
        // follow 0.85/0.95 → 0.90/0.95; warmup 0 → two keyframes.
        XCTAssertEqual(schedule.keyframes, [
            LRKeyframe(step: 0, peak: 0.02, trough: 0.002, momentumAtPeak: 0.85, momentumAtTrough: 0.95),
            LRKeyframe(step: 1_000_000, peak: 1.0e-4, trough: 1.0e-6, momentumAtPeak: 0.90, momentumAtTrough: 0.95),
        ])
    }

    func testAFileWithBothFormsIsRefused() {
        let object: [String: Any] = [
            LegacyLRScheduleKeys.lrCycleMin: 0.001,
            LRCycleKeyframes.id: ParameterValue.lrKeyframes(sample).jsonValue,
        ]
        XCTAssertThrowsError(try ParameterValue.parametersObject(
            fromJSON: try JSONSerialization.data(withJSONObject: object), currentValue: declaredDefault)) { error in
            guard case TrainingConfigError.invalidValue(let id, _) = error else {
                return XCTFail("expected invalidValue, got \(error)")
            }
            XCTAssertEqual(id, LRCycleKeyframes.id)
        }
    }

    func testALegacyFileWhoseCycleWasInactiveTurnsTheCycleOff() throws {
        // lr_cycle_max below lr_cycle_min: the previous code ran the static LR.
        let data = Data(#"{"lr_cycle_enabled": true, "lr_cycle_min": 0.5, "lr_cycle_max": 0.1, "lr_warmup_steps": 1000}"#.utf8)
        let values = try ParameterValue.parametersObject(fromJSON: data, currentValue: declaredDefault)
        XCTAssertEqual(values[LRCycleEnabled.id], .bool(false))
        guard case .lrKeyframes(let schedule) = values[LRCycleKeyframes.id] else {
            return XCTFail("expected lr_cycle_keyframes")
        }
        XCTAssertNoThrow(try schedule.validated())
    }

    // MARK: V-8 --show-lr-schedule

    func testShowLRScheduleReportsTheFedLRAtTheGentleCyclesTroughsAndPeaks() throws {
        let path = repoRoot.appendingPathComponent("experiments/20261008-zlra-selfplay-lr/parameters-zlra-elite-gentlecycle.json").path
        let snapshot = try ShowLRScheduleCLI.snapshot(parametersPath: path)
        var options = ShowLRScheduleCLI.Options()
        options.fromStep = 0
        options.toStep = 20_000
        options.bucketCount = 20_001   // one step per bucket
        let report = try ShowLRScheduleCLI.render(snapshot: snapshot, options: options)
        func lr(at step: Int) throws -> Double {
            try XCTUnwrap(report.sample.buckets.first { $0.firstStep == step }).lrLast
        }
        for trough in [1000, 11_000] {
            XCTAssertEqual(try lr(at: trough), 0.002, accuracy: 1e-9, "trough at \(trough)")
        }
        for peak in [6000, 16_000] {
            XCTAssertEqual(try lr(at: peak), 0.02, accuracy: 1e-8, "peak at \(peak)")
        }
        XCTAssertTrue(report.csv.hasPrefix("step_from,step_to,peak,trough,lr_min,lr_max,lr_last,mom_min,mom_max\n"))
    }

    func testShowLRScheduleBucketsContainEveryStep() throws {
        let snapshot = try ShowLRScheduleCLI.snapshot(parametersPath: nil)
        var options = ShowLRScheduleCLI.Options()
        options.toStep = 30_000
        options.bucketCount = 37
        let coarse = try ShowLRScheduleCLI.render(snapshot: snapshot, options: options)
        options.bucketCount = 30_001
        let fine = try ShowLRScheduleCLI.render(snapshot: snapshot, options: options)
        for bucket in coarse.sample.buckets {
            for point in fine.sample.buckets where point.firstStep >= bucket.firstStep && point.firstStep <= bucket.lastStep {
                XCTAssertTrue((bucket.lrMin...bucket.lrMax).contains(point.lrLast), "step \(point.firstStep)")
            }
        }
    }

    /// A `--from` past the default end, or a step past the bound, is a
    /// usage error, never a trapped range.
    func testShowLRScheduleRefusesARangeItCannotBuild() throws {
        let snapshot = try ShowLRScheduleCLI.snapshot(parametersPath: nil)
        let options = try ShowLRScheduleCLI.parse(["--show-lr-schedule", "--from", "1900000000"])
        XCTAssertThrowsError(try ShowLRScheduleCLI.render(snapshot: snapshot, options: options))
        XCTAssertThrowsError(try ShowLRScheduleCLI.parse(["--show-lr-schedule", "--to", "9223372036854775807"]))
        XCTAssertThrowsError(try ShowLRScheduleCLI.parse(["--show-lr-schedule", "--from", String(ShowLRScheduleCLI.maximumStep + 1)]))
    }

    /// Add Keyframe after a step that isn't usable copies it rather than
    /// guessing (or overflowing).
    @MainActor
    func testAddingAfterAnUnusableStepCopiesIt() {
        func row(_ step: String) -> LRKeyframeRowDraft {
            LRKeyframeRowDraft(step: step, peak: "0.1", trough: "0.001", momentumAtPeak: "0.85", momentumAtTrough: "0.95")
        }
        XCTAssertEqual(LRKeyframeRowDraft.appended(after: [row("9223372036854775807")], periodSteps: 10_000).step, "9223372036854775807")
        XCTAssertEqual(LRKeyframeRowDraft.appended(after: [row("12a")], periodSteps: 10_000).step, "12a")
        XCTAssertEqual(LRKeyframeRowDraft.appended(after: [row(" 5000 ")], periodSteps: 10_000).step, "15000")
    }

    func testShowLRScheduleRefusesBadArguments() {
        XCTAssertThrowsError(try ShowLRScheduleCLI.parse(["--show-lr-schedule", "--bogus"]))
        XCTAssertThrowsError(try ShowLRScheduleCLI.parse(["--show-lr-schedule", "--from", "10", "--to", "5"]))
        XCTAssertThrowsError(try ShowLRScheduleCLI.parse(["--show-lr-schedule", "--buckets", "0"]))
        XCTAssertEqual(try ShowLRScheduleCLI.parse(["--show-lr-schedule", "--to", "100"]).toStep, 100)
    }
}
