//
//  LRKeyframeScheduleTests.swift
//  DrewsChessMachineTests
//
//  The LR keyframe schedule (documentation/plans-active/LR_KEYFRAMES_PLAN.md):
//  segment arithmetic and the validator (V-1), and the conversion of the
//  previous single-decay schedule, which must feed bit-identical LR and
//  momentum at every step (owner decision Q-4, V-2). The equivalence tests
//  compare against the previous cycle code itself (`LRMomentumCycle`), for
//  every repo parameters file that carries a cycle and for the edge cases
//  (warmup 0, no decay, the step the decay ends on, drifting follow bounds,
//  schedules the previous code treated as "no cycle").
//

import XCTest
import TrainingParametersMacroSupport
@testable import DrewsChessMachine

final class LRKeyframeScheduleTests: XCTestCase {

    private func keyframe(_ step: Int, _ peak: Double, _ trough: Double,
                          _ atPeak: Double = 0.85, _ atTrough: Double = 0.95) -> LRKeyframe {
        LRKeyframe(step: step, peak: peak, trough: trough, momentumAtPeak: atPeak, momentumAtTrough: atTrough)
    }

    // MARK: - V-1 segment arithmetic

    func testValuesAtKeyframesAndGeometricMidpoints() throws {
        let schedule = try LRKeyframeSchedule(keyframes: [
            keyframe(0, 1.0, 0.01, 0.80, 0.90),
            keyframe(100, 0.1, 0.001, 0.85, 0.95),
            keyframe(300, 0.01, 0.0001, 0.90, 0.95),
        ]).validated()
        XCTAssertEqual(schedule.bounds(atTrainerStep: 0).peak, 1.0)
        XCTAssertEqual(schedule.bounds(atTrainerStep: 100).peak, 0.1, "an interior keyframe is exact")
        XCTAssertEqual(schedule.bounds(atTrainerStep: 100).momentumAtPeak, 0.85)
        // Geometric: halfway from 1.0 to 0.1 is √0.1, not 0.55.
        XCTAssertEqual(schedule.bounds(atTrainerStep: 50).peak, sqrt(0.1), accuracy: 1e-15)
        XCTAssertEqual(schedule.bounds(atTrainerStep: 50).trough, sqrt(0.01) * 0.1, accuracy: 1e-15)
        // Momentum linear.
        XCTAssertEqual(schedule.bounds(atTrainerStep: 50).momentumAtPeak, 0.825, accuracy: 1e-15)
        XCTAssertEqual(schedule.bounds(atTrainerStep: 200).peak, sqrt(0.001), accuracy: 1e-15)
    }

    func testAStepOnAKeyframeStartsItsSegment() throws {
        // Segment k covers [s_k, s_{k+1}); step 100 is segment 1 at fraction 0.
        let schedule = try LRKeyframeSchedule(keyframes: [keyframe(0, 1.0, 0.5), keyframe(100, 0.1, 0.05),
                                                         keyframe(200, 0.01, 0.005)]).validated()
        let at100 = schedule.bounds(atTrainerStep: 100)
        XCTAssertEqual(at100.peak.bitPattern, LRKeyframeSchedule.interpolate(schedule.keyframes[1], schedule.keyframes[2],
                                                                             fraction: 0.0).peak.bitPattern)
        XCTAssertEqual(schedule.bounds(atTrainerStep: 99).peak.bitPattern,
                       LRKeyframeSchedule.interpolate(schedule.keyframes[0], schedule.keyframes[1],
                                                      fraction: 99.0 / 100.0).peak.bitPattern)
    }

    func testAfterTheLastKeyframeTheLastSegmentAtFractionOne() throws {
        let a = keyframe(0, 1.0, 0.001)
        let b = keyframe(1000, 0.0001, 0.000001, 0.90, 0.95)
        let schedule = try LRKeyframeSchedule(keyframes: [a, b]).validated()
        let held = LRKeyframeSchedule.interpolate(a, b, fraction: 1.0)
        for step in [1000, 1001, 5_000_000] {
            let bounds = schedule.bounds(atTrainerStep: step)
            XCTAssertEqual(bounds.peak.bitPattern, held.peak.bitPattern)
            XCTAssertEqual(bounds.trough.bitPattern, held.trough.bitPattern)
            XCTAssertEqual(bounds.momentumAtPeak.bitPattern, held.momentumAtPeak.bitPattern)
        }
        // The held value is the old envelope's post-horizon arithmetic, not
        // necessarily the stored value.
        XCTAssertEqual(held.peak.bitPattern, (1.0 * pow(0.0001 / 1.0, 1.0)).bitPattern)
    }

    func testASingleKeyframeIsConstant() throws {
        let schedule = try LRKeyframeSchedule(keyframes: [keyframe(0, 0.02, 0.002, 0.88, 0.93)]).validated()
        for step in [0, 1, 10_000, Int(Int32.max)] {
            XCTAssertEqual(schedule.bounds(atTrainerStep: step),
                           LRKeyframeSchedule.Bounds(peak: 0.02, trough: 0.002, momentumAtPeak: 0.88, momentumAtTrough: 0.93))
        }
    }

    func testAStepChangeIsTwoKeyframesOneStepApart() throws {
        let schedule = try LRKeyframeSchedule(keyframes: [keyframe(0, 0.1, 0.01), keyframe(500, 0.1, 0.01),
                                                         keyframe(501, 0.02, 0.002)]).validated()
        XCTAssertEqual(schedule.bounds(atTrainerStep: 500).peak, 0.1)
        XCTAssertEqual(schedule.bounds(atTrainerStep: 501).peak, 0.1 * pow(0.02 / 0.1, 1.0))
        XCTAssertEqual(schedule.bounds(atTrainerStep: 499).peak, 0.1)
    }

    // MARK: - V-1 validator

    func testValidatorRefusesEachInvalidSchedule() {
        let good = keyframe(0, 0.1, 0.01)
        let cases: [(String, [LRKeyframe], LRKeyframeScheduleError)] = [
            ("empty", [], .empty),
            ("too many", (0...LRKeyframeSchedule.maximumKeyframeCount).map { keyframe($0, 0.1, 0.01) },
             .tooManyKeyframes(count: 65, maximum: 64)),
            ("first step", [keyframe(5, 0.1, 0.01)], .firstStepNotZero(step: 5)),
            ("unsorted", [good, keyframe(10, 0.1, 0.01), keyframe(5, 0.1, 0.01)],
             .stepsNotIncreasing(index: 2, step: 5, previous: 10)),
            ("duplicate", [good, keyframe(0, 0.1, 0.01)], .stepsNotIncreasing(index: 1, step: 0, previous: 0)),
            ("non-finite", [keyframe(0, .nan, 0.01)], .notFinite(index: 0, field: "peak")),
            ("trough 0", [keyframe(0, 0.1, 0)], .troughNotPositive(index: 0, trough: 0)),
            ("peak < trough", [keyframe(0, 0.001, 0.01)], .peakBelowTrough(index: 0, peak: 0.001, trough: 0.01)),
            ("peak > 10", [keyframe(0, 11, 0.01)], .peakTooLarge(index: 0, peak: 11, maximum: 10)),
            ("momentum range", [keyframe(0, 0.1, 0.01, 0.85, 1.0)],
             .momentumOutOfRange(index: 0, field: "momentum_at_trough", value: 1.0, range: 0.0...0.99)),
            ("momentum order", [keyframe(0, 0.1, 0.01, 0.95, 0.85)],
             .momentumAtPeakAboveTrough(index: 0, atPeak: 0.95, atTrough: 0.85)),
        ]
        for (name, frames, expected) in cases {
            XCTAssertThrowsError(try LRKeyframeSchedule(keyframes: frames).validated(), name) { error in
                XCTAssertEqual(error as? LRKeyframeScheduleError, expected, name)
                XCTAssertFalse(String(describing: error).isEmpty)
            }
        }
        XCTAssertNoThrow(try LRKeyframeSchedule(keyframes: [good]).validated())
    }

    func testCodableIsABareArrayWithSnakeCaseMomentumKeys() throws {
        let schedule = LRKeyframeSchedule(keyframes: [keyframe(0, 0.1, 0.001), keyframe(1000, 0.02, 0.002, 0.9, 0.95)])
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let text = String(decoding: try encoder.encode(schedule), as: UTF8.self)
        XCTAssertTrue(text.hasPrefix("[{"), text)
        XCTAssertTrue(text.contains("\"momentum_at_peak\":0.85"), text)
        XCTAssertEqual(try JSONDecoder().decode(LRKeyframeSchedule.self, from: Data(text.utf8)), schedule)
    }

    // MARK: - V-2 conversion of the previous schedule is bit-identical

    /// The previous cycle (the code under replacement) built from the same
    /// legacy settings.
    private func previousCycle(_ legacy: LegacyLRSchedule) -> LRMomentumCycle {
        LRMomentumCycle(
            lrEnabled: legacy.lrEnabled, lrPeriodSteps: legacy.lrPeriodSteps, lrCount: legacy.lrCount,
            lrMin: legacy.lrMin, lrMax: legacy.lrMax, lrInvert: legacy.lrInvert,
            momentumEnabled: legacy.momentumEnabled, momentumPeriodSteps: legacy.momentumPeriodSteps,
            momentumCount: legacy.momentumCount, momentumMin: legacy.momentumMin, momentumMax: legacy.momentumMax,
            momentumInvert: legacy.momentumInvert,
            envelope: LRMomentumCycleEnvelope(
                lrPeakEnd: legacy.lrPeakEnd, lrTroughEnd: legacy.lrTroughEnd, decayHorizonSteps: legacy.decayHorizonSteps,
                momentumFollowsLRCycle: legacy.momentumFollowsLRCycle,
                momentumFollowStartLow: legacy.momentumFollowStartLow, momentumFollowStartHigh: legacy.momentumFollowStartHigh,
                momentumFollowEndLow: legacy.momentumFollowEndLow, momentumFollowEndHigh: legacy.momentumFollowEndHigh)
        )
    }

    private static func bits(_ value: Double?) -> UInt64? { value.map(\.bitPattern) }

    /// Every step in `0...lastStep`, split across cores; returns the first
    /// mismatching step, if any.
    private func firstMismatch(_ legacy: LegacyLRSchedule, through lastStep: Int) throws -> (step: Int, detail: String)? {
        let old = previousCycle(legacy)
        let new = try LRKeyframeCycle.converting(legacy).cycle
        let w = legacy.lrWarmupSteps
        let chunks = 64
        let chunk = (lastStep + chunks) / chunks
        let found = SyncBox<[(Int, String)]>([])
        DispatchQueue.concurrentPerform(iterations: chunks) { c in
            let from = c * chunk
            let to = min(lastStep, from + chunk - 1)
            guard from <= to else { return }
            for step in from...to {
                let a = old.values(completedTrainSteps: step, lrWarmupSteps: w)
                let b = new.values(completedTrainSteps: step, lrWarmupSteps: w)
                if Self.bits(a.learningRate) != Self.bits(b.learningRate) || Self.bits(a.momentum) != Self.bits(b.momentum)
                    || Self.bits(a.lrPeak) != Self.bits(b.lrPeak) || Self.bits(a.lrTrough) != Self.bits(b.lrTrough) {
                    let detail = "old lr=\(String(describing: a.learningRate)) mom=\(String(describing: a.momentum)) "
                        + "pk=\(String(describing: a.lrPeak)) tr=\(String(describing: a.lrTrough)); new lr="
                        + "\(String(describing: b.learningRate)) mom=\(String(describing: b.momentum)) "
                        + "pk=\(String(describing: b.lrPeak)) tr=\(String(describing: b.lrTrough))"
                    found.mutate { $0.append((step, detail)) }
                    return
                }
            }
        }
        return found.value.min { $0.0 < $1.0 }.map { (step: $0.0, detail: $0.1) }
    }

    private func assertBitIdentical(_ legacy: LegacyLRSchedule, through lastStep: Int? = nil, _ name: String,
                                    file: StaticString = #filePath, line: UInt = #line) throws {
        let w = max(0, legacy.lrWarmupSteps)
        let last = lastStep ?? (w + max(0, legacy.decayHorizonSteps) + 1_000_000)
        if let mismatch = try firstMismatch(legacy, through: last) {
            XCTFail("\(name): step \(mismatch.step) differs: \(mismatch.detail)", file: file, line: line)
        }
    }

    /// R-replay's settings (`experiments/20261008-zlra-selfplay-lr/parameters-zlra.json`).
    private let rReplay = LegacyLRSchedule(
        lrEnabled: true, lrPeriodSteps: 10_000, lrCount: 0, lrMin: 0.001, lrMax: 1, lrInvert: true,
        momentumEnabled: true, momentumPeriodSteps: 1000, momentumCount: 0, momentumMin: 0.75, momentumMax: 0.9,
        momentumInvert: false, lrPeakEnd: 0.001, lrTroughEnd: 1e-6, decayHorizonSteps: 1_000_000,
        momentumFollowsLRCycle: true, momentumFollowStartLow: 0.85, momentumFollowStartHigh: 0.95,
        momentumFollowEndLow: 0.9, momentumFollowEndHigh: 0.95, lrWarmupSteps: 1000)

    func testRReplayScheduleIsBitIdenticalAtEveryStep() throws {
        try assertBitIdentical(rReplay, "R-replay")
    }

    func testWarmupZeroConvertsWithoutADuplicateKeyframe() throws {
        var legacy = rReplay
        legacy.lrWarmupSteps = 0
        let converted = try LRKeyframeCycle.converting(legacy).cycle
        XCTAssertEqual(converted.lrKeyframes.keyframes.map(\.step), [0, 1_000_000])
        try assertBitIdentical(legacy, through: 1_100_000, "warmup 0")
    }

    func testNoDecayConvertsToOneKeyframe() throws {
        var legacy = rReplay
        legacy.decayHorizonSteps = 0
        let converted = try LRKeyframeCycle.converting(legacy).cycle
        XCTAssertEqual(converted.lrKeyframes.keyframes.count, 1)
        try assertBitIdentical(legacy, through: 300_000, "no decay")
    }

    func testTheStepsAroundTheEndOfTheDecayAreBitIdentical() throws {
        var legacy = rReplay
        legacy.decayHorizonSteps = 12_345
        legacy.lrWarmupSteps = 777
        try assertBitIdentical(legacy, through: 777 + 12_345 + 50_000, "odd horizon and warmup")
    }

    func testFollowOnWithMomentumCyclingOffIsBitIdentical() throws {
        var legacy = rReplay
        legacy.momentumEnabled = false
        try assertBitIdentical(legacy, through: 200_000, "follow on, momentum off")
        legacy.momentumEnabled = true
        legacy.momentumFollowsLRCycle = false
        try assertBitIdentical(legacy, through: 200_000, "independent momentum cycle")
        legacy.lrEnabled = false
        try assertBitIdentical(legacy, through: 200_000, "LR off, independent momentum")
    }

    func testAnUnusableLRPairConvertsAsDisabled() throws {
        var legacy = rReplay
        legacy.lrMin = 0.5
        legacy.lrMax = 0.1   // peak below trough: the previous code ran the static LR
        let (cycle, notes) = try LRKeyframeCycle.converting(legacy)
        XCTAssertFalse(cycle.lrEnabled)
        XCTAssertFalse(notes.isEmpty)
        try assertBitIdentical(legacy, through: 100_000, "unusable start pair")

        legacy = rReplay
        legacy.lrPeakEnd = 1e-7
        legacy.lrTroughEnd = 1e-6   // end pair inverted: the previous code ran the static LR throughout
        XCTAssertFalse(try LRKeyframeCycle.converting(legacy).cycle.lrEnabled)
        try assertBitIdentical(legacy, through: 100_000, "unusable end pair")
    }

    func testFollowBoundsInvertedThroughoutConvertAsMomentumOff() throws {
        var legacy = rReplay
        legacy.momentumFollowStartLow = 0.95
        legacy.momentumFollowStartHigh = 0.85
        legacy.momentumFollowEndLow = 0.95
        legacy.momentumFollowEndHigh = 0.90
        let (cycle, notes) = try LRKeyframeCycle.converting(legacy)
        XCTAssertFalse(cycle.momentumEnabled)
        XCTAssertFalse(notes.isEmpty)
        try assertBitIdentical(legacy, through: 100_000, "follow inverted throughout")
    }

    func testFollowBoundsInvertedOnPartOfTheRunAreRefused() {
        var legacy = rReplay
        legacy.momentumFollowStartLow = 0.95
        legacy.momentumFollowStartHigh = 0.90
        legacy.momentumFollowEndLow = 0.85
        legacy.momentumFollowEndHigh = 0.95
        XCTAssertThrowsError(try LRKeyframeCycle.converting(legacy))
    }

    func testInertInvalidFollowBoundsAreReplacedAndStayInert() throws {
        var legacy = rReplay
        legacy.momentumFollowsLRCycle = false   // independent momentum cycle drives momentum
        legacy.momentumFollowStartLow = 0.95
        legacy.momentumFollowStartHigh = 0.85
        let (cycle, _) = try LRKeyframeCycle.converting(legacy)
        XCTAssertNoThrow(try cycle.lrKeyframes.validated())
        try assertBitIdentical(legacy, through: 100_000, "inert inverted follow bounds")
    }

    /// Every repo parameters file with the previous schedule's keys: the
    /// conversion feeds what the previous code fed, at every step through the
    /// end of the decay plus a million.
    func testEveryRepoParametersFileIsBitIdentical() throws {
        let repoRoot = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent()
        let enumerator = try XCTUnwrap(FileManager.default.enumerator(at: repoRoot, includingPropertiesForKeys: nil))
        var checked: [String] = []
        var distinct: [LegacyLRSchedule] = []
        for case let url as URL in enumerator {
            let path = url.path
            if path.contains("/.git/") || path.contains("/DerivedData/") || url.pathExtension != "json"
                || url.lastPathComponent.hasPrefix("results") { continue }
            guard let data = FileManager.default.contents(atPath: path),
                  let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                  object["lr_cycle_decay_horizon_steps"] != nil, object["momentum_follow_end_high"] != nil,
                  object["lr_cycle_min"] != nil else { continue }
            func number(_ key: String) throws -> Double {
                try XCTUnwrap((object[key] as? NSNumber)?.doubleValue, "\(path): \(key)")
            }
            func flag(_ key: String) throws -> Bool { try XCTUnwrap((object[key] as? NSNumber)?.boolValue, "\(path): \(key)") }
            let legacy = LegacyLRSchedule(
                lrEnabled: try flag("lr_cycle_enabled"), lrPeriodSteps: Int(try number("lr_cycle_period_steps")),
                lrCount: Int(try number("lr_cycle_count")), lrMin: try number("lr_cycle_min"),
                lrMax: try number("lr_cycle_max"), lrInvert: try flag("lr_cycle_invert"),
                momentumEnabled: try flag("momentum_cycle_enabled"),
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
            // Many files share a schedule; each distinct one is checked once.
            if !distinct.contains(legacy) {
                distinct.append(legacy)
                try assertBitIdentical(legacy, path)
            }
            checked.append(path)
        }
        XCTAssertTrue(checked.contains { $0.hasSuffix("20261008-zlra-selfplay-lr/parameters-zlra.json") }, "R-replay's file")
        XCTAssertTrue(checked.contains { $0.hasSuffix("20261005-lr-schedule-ab/parameters-B-relcap-v3.json") }, "B-siluall's file")
        XCTAssertGreaterThanOrEqual(checked.count, 30, "complete legacy parameters files found: \(checked.count)")
    }

    // MARK: - Sampler

    func testSamplerBucketsContainEveryStepsValue() throws {
        let cycle = try LRKeyframeCycle.converting(rReplay).cycle
        let point = { (step: Int) -> LRScheduleSample.Point in
            let v = cycle.values(completedTrainSteps: step, lrWarmupSteps: 1000)
            return .init(learningRate: v.learningRate ?? 0, momentum: v.momentum ?? 0, lrPeak: v.lrPeak, lrTrough: v.lrTrough)
        }
        let sample = try XCTUnwrap(LRScheduleSample.sample(steps: 0...99_999, bucketCount: 37, valueAt: point))
        XCTAssertEqual(sample.buckets.first?.firstStep, 0)
        XCTAssertEqual(sample.buckets.last?.lastStep, 99_999)
        for (a, b) in zip(sample.buckets, sample.buckets.dropFirst()) {
            XCTAssertEqual(b.firstStep, a.lastStep + 1, "buckets tile the range")
        }
        for bucket in sample.buckets {
            for step in bucket.firstStep...bucket.lastStep {
                let p = point(step)
                XCTAssertTrue((bucket.lrMin...bucket.lrMax).contains(p.learningRate))
                XCTAssertTrue((bucket.momentumMin...bucket.momentumMax).contains(p.momentum))
            }
        }
        // A cancelled sampling returns nothing.
        XCTAssertNil(LRScheduleSample.sample(steps: 0...1000, bucketCount: 10, isCancelled: { true }, valueAt: point))
    }
}
