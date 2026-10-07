//
//  GradientNormHistoryTests.swift
//  DrewsChessMachineTests
//
//  The relative cap's per-step history (plan X2): contiguity is enforced (a
//  gap or repeat means the clock moved without the history), capacity drops
//  the oldest, the step-line summary, a bit-exact metadata round trip, the
//  decode refusals, and that the cap and the `gradient_spike` rule share one
//  trailing-median definition.
//

import XCTest
@testable import DrewsChessMachine

final class GradientNormHistoryTests: XCTestCase {

    // MARK: - Append

    func test_append_refusesAGapARepeatAndNonFiniteValues() throws {
        var history = GradientNormHistory()
        try history.append(trainerStep: 7, preClipNorm: 1, fedCap: 15)
        XCTAssertThrowsError(try history.append(trainerStep: 9, preClipNorm: 1, fedCap: 15)) {
            XCTAssertEqual($0 as? GradientNormHistoryError, .discontinuity(expected: 8, got: 9))
        }
        XCTAssertThrowsError(try history.append(trainerStep: 7, preClipNorm: 1, fedCap: 15)) {
            XCTAssertEqual($0 as? GradientNormHistoryError, .discontinuity(expected: 8, got: 7))
        }
        XCTAssertThrowsError(try history.append(trainerStep: 8, preClipNorm: .nan, fedCap: 15))
        XCTAssertThrowsError(try history.append(trainerStep: 8, preClipNorm: .infinity, fedCap: 15))
        XCTAssertThrowsError(try history.append(trainerStep: 8, preClipNorm: 1, fedCap: .nan))
        XCTAssertEqual(history.count, 1, "a refused append changes nothing")
        try history.append(trainerStep: 8, preClipNorm: 2, fedCap: 15)
        XCTAssertEqual(history.lastTrainerStep, 8)
        XCTAssertEqual(history.firstTrainerStep, 7)
        XCTAssertThrowsError(try GradientNormHistory().checkContinues(toTrainerStep: 0))
    }

    func test_capacityDropsTheOldest() throws {
        var history = GradientNormHistory()
        let total = GradientNormHistory.capacity + 5
        for step in 1...total {
            try history.append(trainerStep: step, preClipNorm: Float(step), fedCap: 15)
        }
        XCTAssertEqual(history.count, GradientNormHistory.capacity)
        XCTAssertEqual(history.firstTrainerStep, 6)
        XCTAssertEqual(history.preClipNorms.first, 6)
        XCTAssertEqual(history.lastTrainerStep, total)
    }

    func test_summary_maxAndClipCountOverARange() throws {
        var history = GradientNormHistory()
        let norms: [Float] = [1, 4, 2, 9, 3]
        let caps: [Float] = [15, 3, 15, 5, 15]
        for (index, (norm, cap)) in zip(norms, caps).enumerated() {
            try history.append(trainerStep: index + 11, preClipNorm: norm, fedCap: cap)
        }
        let all = history.summary(trainerSteps: 11...15)
        XCTAssertEqual(all.maxPreClipNorm, 9)
        XCTAssertEqual(all.clipped, 2)
        XCTAssertEqual(all.steps, 5)
        let tail = history.summary(trainerSteps: 15...100)
        XCTAssertEqual(tail.maxPreClipNorm, 3)
        XCTAssertEqual(tail.clipped, 0)
        XCTAssertEqual(tail.steps, 1)
        let none = history.summary(trainerSteps: 1...10)
        XCTAssertNil(none.maxPreClipNorm)
        XCTAssertEqual(none.steps, 0)
    }

    func test_checkEnds() throws {
        var history = GradientNormHistory()
        XCTAssertNoThrow(try history.checkEnds(atTrainerClock: 500), "an empty history fits any clock")
        try history.append(trainerStep: 5, preClipNorm: 1, fedCap: 15)
        XCTAssertNoThrow(try history.checkEnds(atTrainerClock: 5))
        XCTAssertThrowsError(try history.checkEnds(atTrainerClock: 6)) {
            XCTAssertEqual($0 as? GradientNormHistoryError, .clockMismatch(historyLastStep: 5, trainerClock: 6))
        }
    }

    // MARK: - Metadata round trip

    func test_metadataRoundTrip_isBitExact() throws {
        var rng = SystemRandomNumberGenerator()
        var history = GradientNormHistory()
        let specials: [Float] = [
            .leastNonzeroMagnitude, .leastNormalMagnitude, Float.leastNormalMagnitude.nextDown,
            .greatestFiniteMagnitude, Float.greatestFiniteMagnitude.nextDown, 0, -0.0, 1, 0.1, 1.0 / 3.0,
        ]
        for step in 1...GradientNormHistory.capacity {
            let norm: Float
            if step <= specials.count {
                norm = specials[step - 1]
            } else {
                // Any finite bit pattern, subnormals included.
                var bits: UInt32
                repeat { bits = UInt32.random(in: 0...UInt32.max, using: &rng) } while !Float(bitPattern: bits).isFinite
                norm = Float(bitPattern: bits)
            }
            try history.append(trainerStep: step + 40, preClipNorm: norm, fedCap: Float(step) / 7)
        }
        let decoded = try GradientNormHistory(metadataValue: try history.metadataValue())
        XCTAssertEqual(decoded.lastTrainerStep, history.lastTrainerStep)
        XCTAssertEqual(decoded.preClipNorms.map(\.bitPattern), history.preClipNorms.map(\.bitPattern))
        XCTAssertEqual(decoded.fedCaps.map(\.bitPattern), history.fedCaps.map(\.bitPattern))

        let empty = try GradientNormHistory(metadataValue: try GradientNormHistory().metadataValue())
        XCTAssertEqual(empty, GradientNormHistory())
        XCTAssertNil(try GradientNormHistory.decode(fromMetadata: [:]), "a file without the key has no history")
    }

    func test_decodeRefusals() {
        let refused = [
            #"{"version":1,"last_trainer_step":3,"pre_clip_norms":[1,2],"fed_caps":[1]}"#,
            #"{"version":1,"last_trainer_step":3,"pre_clip_norms":[1,"NaN"],"fed_caps":[1,2]}"#,
            #"{"version":1,"last_trainer_step":1,"pre_clip_norms":[1,2],"fed_caps":[1,2]}"#,
            #"{"version":1,"last_trainer_step":null,"pre_clip_norms":[1],"fed_caps":[1]}"#,
            #"{"version":1,"last_trainer_step":4,"pre_clip_norms":[],"fed_caps":[]}"#,
            #"{"version":1,"last_trainer_step":-2,"pre_clip_norms":[],"fed_caps":[]}"#,
            #"{"version":2,"last_trainer_step":null,"pre_clip_norms":[],"fed_caps":[]}"#,
            #"not json"#,
        ]
        for text in refused {
            XCTAssertThrowsError(try GradientNormHistory(metadataValue: text), text)
        }
        let overCapacity = Array(repeating: "1", count: GradientNormHistory.capacity + 1).joined(separator: ",")
        XCTAssertThrowsError(try GradientNormHistory(metadataValue:
            #"{"version":1,"last_trainer_step":20000,"pre_clip_norms":[\#(overCapacity)],"fed_caps":[\#(overCapacity)]}"#))
    }

    // MARK: - One trailing-median definition

    func test_spikeRulesEntryPoint_isThePolicyFunctionWithSpikeRules() {
        let cases: [[(trainerStep: Int, value: Float?)]] = [
            [100, 200, 300, 400].map { (trainerStep: $0, value: Float(1)) },
            [300, 350, 400, 450, 499].map { (trainerStep: $0, value: Float(1)) },
            [299, 350, 400, 450, 499].map { (trainerStep: $0, value: Float($0)) },
            [(trainerStep: 299, value: nil), (trainerStep: 350, value: .nan), (trainerStep: 400, value: 2),
             (trainerStep: 450, value: 3), (trainerStep: 499, value: 4)],
        ]
        for values in cases {
            XCTAssertEqual(TrainingHealthReference.make(values, windowStart: 500),
                           TrainingHealthReference.make(values, windowStart: 500, policy: .spikeRules))
        }
        XCTAssertEqual(TrailingReferencePolicy.spikeRules, TrailingReferencePolicy(
            lookbackSteps: TrainingHealthThresholds.spikeReferenceLookbackSteps,
            minimumRecords: TrainingHealthThresholds.spikeReferenceMinimumRecords,
            minimumSpanSteps: TrainingHealthThresholds.spikeReferenceMinimumSpanSteps))
    }
}
