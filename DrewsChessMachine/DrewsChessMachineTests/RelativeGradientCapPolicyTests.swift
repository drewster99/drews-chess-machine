//
//  RelativeGradientCapPolicyTests.swift
//  DrewsChessMachineTests
//
//  The relative gradient cap's rule (plan X1): cap = min(hardMax, max(floor,
//  k × median of the last N pre-clip norms)) once the window holds W entries,
//  which term binds, the warm-up and window bounds, that the median is of
//  pre-clip norms (never the fed caps, which would ratchet), the
//  configuration's cross-parameter refusals, and the `[GRAD-CLIP]` line.
//

import XCTest
@testable import DrewsChessMachine

final class RelativeGradientCapPolicyTests: XCTestCase {

    private func configuration(
        _ mode: RelativeGradientCapMode = .clip, k: Double = 3, n: Int = 4, w: Int = 2, floor: Double = 0.5
    ) throws -> RelativeGradientCapConfiguration {
        try RelativeGradientCapConfiguration(mode: mode, multiple: k, windowSteps: n, minimumHistorySteps: w, floor: floor)
    }

    /// A history of `norms` at trainer steps 1…count, each fed `fedCap`.
    private func history(_ norms: [Float], fedCap: Float = 15) throws -> GradientNormHistory {
        var history = GradientNormHistory()
        for (index, norm) in norms.enumerated() {
            try history.append(trainerStep: index + 1, preClipNorm: norm, fedCap: fedCap)
        }
        return history
    }

    private func decide(_ configuration: RelativeGradientCapConfiguration, hardMax: Float = 15,
                        _ history: GradientNormHistory) -> GradientCapDecision {
        GradientCapPolicy.decide(configuration: configuration, hardMax: hardMax, history: history,
                                 nextTrainerStep: (history.lastTrainerStep ?? 0) + 1)
    }

    // MARK: - Formula and binding

    func test_relativeTermBinds() throws {
        let decision = decide(try configuration(), try history([1, 1, 1, 1]))
        XCTAssertEqual(decision.decidedCap, 3)
        XCTAssertEqual(decision.fedCap, 3)
        XCTAssertEqual(decision.binding, .relative)
        XCTAssertEqual(decision.referenceMedian, 1)
        XCTAssertEqual(decision.referenceCount, 4)
    }

    func test_hardMaxBinds_whenTheRelativeTermIsAboveIt() throws {
        let decision = decide(try configuration(), hardMax: 2, try history([1, 1, 1, 1]))
        XCTAssertEqual(decision.decidedCap, 2)
        XCTAssertEqual(decision.binding, .hard)
    }

    func test_floorBinds_whenKTimesTheMedianIsBelowIt() throws {
        let decision = decide(try configuration(floor: 0.5), try history([0.1, 0.1, 0.1, 0.1]))
        XCTAssertEqual(decision.decidedCap, 0.5)
        XCTAssertEqual(decision.binding, .floor)
    }

    func test_floorAtOrAboveTheHardMax_isHard() throws {
        let decision = decide(try configuration(floor: 20), hardMax: 15, try history([0.1, 0.1, 0.1, 0.1]))
        XCTAssertEqual(decision.decidedCap, 15)
        XCTAssertEqual(decision.binding, .hard)
        XCTAssertTrue(RelativeGradientCapLogFormat.configLine(try configuration(floor: 20), hardMax: 15)
            .hasSuffix(" relative_cap_inert=floor_at_or_above_hard_max"))
        XCTAssertFalse(RelativeGradientCapLogFormat.configLine(try configuration(floor: 0.5), hardMax: 15)
            .contains("relative_cap_inert"))
    }

    func test_modeOff_feedsTheHardMaxWithNoMedian() throws {
        let decision = decide(try configuration(.off), try history([1, 1, 1, 1]))
        XCTAssertEqual(decision.fedCap, 15)
        XCTAssertEqual(decision.decidedCap, 15)
        XCTAssertEqual(decision.binding, .hard)
        XCTAssertNil(decision.referenceMedian)
    }

    func test_logOnly_feedsTheHardMaxButDecidesTheRule() throws {
        let decision = decide(try configuration(.logOnly), try history([1, 1, 1, 1]))
        XCTAssertEqual(decision.fedCap, 15)
        XCTAssertEqual(decision.decidedCap, 3)
        XCTAssertEqual(decision.binding, .relative)
        XCTAssertFalse(decision.clipped(preClipNorm: 7))
        XCTAssertTrue(decision.wouldClip(preClipNorm: 7))
    }

    // MARK: - Warm-up and window

    func test_warmUp_noRelativeTermBelowW_thenTheMedianOfW() throws {
        let config = try configuration(n: 10, w: 3)
        let short = decide(config, try history([1, 2]))
        XCTAssertEqual(short.binding, .hard)
        XCTAssertNil(short.referenceMedian)
        let atW = decide(config, try history([1, 2, 4]))
        XCTAssertEqual(atW.referenceMedian, 2)
        XCTAssertEqual(atW.referenceCount, 3)
        XCTAssertEqual(atW.decidedCap, 6)
    }

    func test_onlyTheLastNCount_andAnOlderOutlierIsIgnored() throws {
        // Step 1's 1000 is outside s−N … s−1 for s = 6, N = 4.
        let decision = decide(try configuration(n: 4, w: 2), hardMax: 1e9, try history([1000, 1, 1, 1, 1]))
        XCTAssertEqual(decision.referenceMedian, 1)
        XCTAssertEqual(decision.referenceCount, 4)
        // Window bounds exactly: steps 2…5.
        let window = try history([1000, 1, 1, 1, 1]).window(endingBefore: 6, count: 4)
        XCTAssertEqual(window.map(\.trainerStep), [2, 3, 4, 5])
    }

    func test_evenCount_isTheMeanOfTheTwoMiddleValues() throws {
        let decision = decide(try configuration(n: 4, w: 2), try history([1, 2, 3, 10]))
        XCTAssertEqual(decision.referenceMedian, 2.5)
    }

    func test_theMedianIsOfPreClipNorms_notOfFedCaps() throws {
        // Clipped steps recorded large pre-clip norms and small fed caps; a
        // median of the caps would be 1, of the pre-clip norms 8.
        let clipped = try history([8, 8, 8, 8], fedCap: 1)
        let decision = decide(try configuration(), hardMax: 1e9, clipped)
        XCTAssertEqual(decision.referenceMedian, 8)
        XCTAssertEqual(decision.decidedCap, 24)
    }

    func test_decisionIsDeterministic() throws {
        let config = try configuration()
        let h = try history([0.3, 0.7, 0.2, 0.9, 0.4])
        XCTAssertEqual(decide(config, h), decide(config, h))
    }

    // MARK: - Configuration

    func test_configurationRefusals_nameTheParameters() throws {
        XCTAssertThrowsError(try configuration(n: 100, w: 101)) { error in
            let text = "\(error)"
            XCTAssertTrue(text.contains(RelativeGradClipMinHistorySteps.id), text)
            XCTAssertTrue(text.contains(RelativeGradClipWindowSteps.id), text)
        }
        XCTAssertThrowsError(try configuration(k: 0)) { XCTAssertTrue("\($0)".contains(RelativeGradClipMultiple.id)) }
        XCTAssertThrowsError(try configuration(k: -1)) { XCTAssertTrue("\($0)".contains(RelativeGradClipMultiple.id)) }
        XCTAssertThrowsError(try configuration(floor: .nan)) { XCTAssertTrue("\($0)".contains(RelativeGradClipFloor.id)) }
        XCTAssertThrowsError(try configuration(n: GradientNormHistory.capacity + 1, w: 2)) {
            XCTAssertTrue("\($0)".contains(RelativeGradClipWindowSteps.id))
        }
        XCTAssertThrowsError(try RelativeGradientCapConfiguration(
            modeRawValue: 3, multiple: 3, windowSteps: 10, minimumHistorySteps: 2, floor: 0.5)) {
            XCTAssertTrue("\($0)".contains(RelativeGradClipMode.id))
        }
    }

    func test_declaredDefaults_validate() throws {
        let defaults = try RelativeGradientCapConfiguration.declaredDefaults()
        XCTAssertEqual(defaults.mode, .logOnly)
        XCTAssertEqual(defaults.multiple, 3)
        XCTAssertEqual(defaults.windowSteps, 1000)
        XCTAssertEqual(defaults.minimumHistorySteps, 100)
        XCTAssertEqual(defaults.floor, 0.5)
        XCTAssertEqual(GradientNormHistory.capacity, RelativeGradClipWindowSteps.declaredClosedRange.upperBound)
    }

    // MARK: - `[GRAD-CLIP]` line

    func test_eventLine_appliedClip() throws {
        let decision = decide(try configuration(), try history([1, 1, 1, 1]))
        XCTAssertEqual(
            RelativeGradientCapLogFormat.eventLine(trainerStep: 5, preClipNorm: 7.5, decision: decision, learningRate: 0.01),
            "[GRAD-CLIP] trainerStep=5 preNorm=7.5000 cap=3.0000 decided=3.0000 binding=relative applied=true mode=clip "
                + "median=1.0000 n=4 k=3 floor=0.5 hardMax=15 ratio=7.50 lr=0.01"
        )
        XCTAssertNil(RelativeGradientCapLogFormat.eventLine(trainerStep: 5, preClipNorm: 2.9, decision: decision, learningRate: 0.01))
    }

    func test_eventLine_logOnly() throws {
        let decision = decide(try configuration(.logOnly), try history([1, 1, 1, 1]))
        XCTAssertEqual(
            RelativeGradientCapLogFormat.eventLine(trainerStep: 5, preClipNorm: 7.5, decision: decision, learningRate: 0.5),
            "[GRAD-CLIP] trainerStep=5 preNorm=7.5000 cap=15.0000 decided=3.0000 binding=relative applied=false mode=log_only "
                + "median=1.0000 n=4 k=3 floor=0.5 hardMax=15 ratio=7.50 lr=0.5"
        )
    }

    func test_eventLine_hardMaxClipWithNoMedian() throws {
        let decision = decide(try configuration(), GradientNormHistory())
        XCTAssertEqual(
            RelativeGradientCapLogFormat.eventLine(trainerStep: 1, preClipNorm: 27, decision: decision, learningRate: 0),
            "[GRAD-CLIP] trainerStep=1 preNorm=27.0000 cap=15.0000 decided=15.0000 binding=hard applied=true mode=clip "
                + "median=none n=0 k=3 floor=0.5 hardMax=15 ratio=none lr=0"
        )
    }

    func test_stepLineFields() throws {
        let h = try history([1, 5, 2], fedCap: 3)
        XCTAssertEqual(
            RelativeGradientCapLogFormat.stepLineFields(summary: h.summary(trainerSteps: 1...3), fedCap: 3),
            " gNormMax=5.0000 clips=1 gCap=3.0000"
        )
        XCTAssertEqual(
            RelativeGradientCapLogFormat.stepLineFields(summary: GradientNormHistory().summary(trainerSteps: 1...3), fedCap: 15),
            " gNormMax=none clips=0 gCap=15.0000"
        )
    }
}
