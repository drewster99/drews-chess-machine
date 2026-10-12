//
//  TrainingSettingsCycleValidationTests.swift
//  DrewsChessMachineTests
//
//  Guards the cross-field validation for the LR / momentum cycle editors in
//  `TrainingSettingsPopoverModel.save()`.
//
//  Momentum cycle: each endpoint is range-checked individually, but an
//  inverted pair (max < min) is a distinct misconfiguration — the momentum
//  cycle silently runs reversed (`momentum(forStep:)` has no guard) with no
//  signal to the user. The cross-field check flags the max field so the
//  misconfig surfaces as the red invalid-input overlay.
//
//  LR cycle: since LR keyframes (LR_KEYFRAMES_PLAN.md) the LR bounds and the
//  momentum-follow bounds are rows of the keyframe table, validated on Save
//  as one schedule by the parameter's own validator
//  (`LRKeyframeSchedule.validated()`). The previous "LR max >= min" and
//  "follow high >= low" checks are its per-keyframe "peak >= trough" and
//  "momentum at the peak <= momentum at the trough", reported through
//  `lrKeyframesError` with the validator's message, and a refused schedule is
//  never written.
//
//  Pure model/validation logic (no Metal). `save()` writes through to
//  `TrainingParameters.shared`, so each test snapshots and restores the
//  cycle settings it can touch.
//

import XCTest
import TrainingParametersMacroSupport
@testable import DrewsChessMachine

@MainActor
final class TrainingSettingsCycleValidationTests: XCTestCase {

    private func makeModel() -> TrainingSettingsPopoverModel {
        // trainerProvider defaults to { nil }, so save() commits to
        // TrainingParameters.shared without needing a live trainer.
        let model = TrainingSettingsPopoverModel(
            selfPlayDelayMaxMs: 1000,
            stepDelayMaxMs: 1000,
            maxSelfPlayWorkers: 8
        )
        // Seeded as the popover is on open, so Save leaves every field a test
        // does not edit untouched instead of writing the model's blank
        // defaults into the user's settings.
        model.seedFromParams()
        return model
    }

    /// One keyframe row at step 0.
    private func row(peak: String, trough: String,
                     momentumAtPeak: String = "0.85", momentumAtTrough: String = "0.95") -> LRKeyframeRowDraft {
        LRKeyframeRowDraft(step: "0", peak: peak, trough: trough,
                           momentumAtPeak: momentumAtPeak, momentumAtTrough: momentumAtTrough)
    }

    /// Was `test_lrCycleMaxBelowMin_flagsMaxError` (LR min/max text fields,
    /// `lrCycleMaxError`). Those fields became the keyframe table, so the
    /// same inverted pair is now a keyframe whose peak is below its trough,
    /// refused with the validator's message and not written.
    func test_lrKeyframePeakBelowTrough_flagsKeyframesError() {
        let p = TrainingParameters.shared
        let savedKeyframes = p.lrCycleKeyframes
        defer { p.lrCycleKeyframes = savedKeyframes }

        let model = makeModel()
        model.lrKeyframeRows = [row(peak: "0.01", trough: "0.1")]   // peak < trough
        model.save()

        XCTAssertEqual(
            model.lrKeyframesError,
            "keyframe 0: peak 0.01 is below trough 0.1",
            "LR peak < trough must be refused (would otherwise fall back to static LR silently)")
        XCTAssertEqual(p.lrCycleKeyframes, savedKeyframes, "a refused schedule must not be written")
    }

    /// The keyframe form of the previous "momentum follow high >= low" check:
    /// momentum at the LR peak above momentum at the LR trough is refused.
    func test_lrKeyframeMomentumAtPeakAboveTrough_flagsKeyframesError() {
        let p = TrainingParameters.shared
        let savedKeyframes = p.lrCycleKeyframes
        defer { p.lrCycleKeyframes = savedKeyframes }

        let model = makeModel()
        model.lrKeyframeRows = [row(peak: "0.03", trough: "0.001", momentumAtPeak: "0.95", momentumAtTrough: "0.85")]
        model.save()

        XCTAssertEqual(
            model.lrKeyframesError,
            "keyframe 0: momentum at the LR peak 0.95 is above momentum at the LR trough 0.85")
        XCTAssertEqual(p.lrCycleKeyframes, savedKeyframes, "a refused schedule must not be written")
    }

    /// The check runs on every keyframe, not only the first, and names the
    /// keyframe.
    func test_lrKeyframeLaterRowPeakBelowTrough_namesThatKeyframe() {
        let p = TrainingParameters.shared
        let savedKeyframes = p.lrCycleKeyframes
        defer { p.lrCycleKeyframes = savedKeyframes }

        let model = makeModel()
        model.lrKeyframeRows = [
            row(peak: "0.03", trough: "0.001"),
            LRKeyframeRowDraft(step: "5000", peak: "0.0001", trough: "0.001", momentumAtPeak: "0.85", momentumAtTrough: "0.95"),
        ]
        model.save()

        XCTAssertEqual(model.lrKeyframesError, "keyframe 1: peak 0.0001 is below trough 0.001")
        XCTAssertEqual(p.lrCycleKeyframes, savedKeyframes, "a refused schedule must not be written")
    }

    func test_momentumCycleMaxBelowMin_flagsMaxError() {
        let p = TrainingParameters.shared
        let savedMin = p.momentumCycleMin
        let savedMax = p.momentumCycleMax
        defer { p.momentumCycleMin = savedMin; p.momentumCycleMax = savedMax }

        let model = makeModel()
        model.momentumCycleMinText = "0.95"
        model.momentumCycleMaxText = "0.85"   // max < min
        model.save()

        XCTAssertTrue(
            model.momentumCycleMaxError,
            "Momentum cycle max < min must flag the max field (would otherwise run the schedule reversed silently)")
    }

    /// LR keyframes: the LR half now edits a keyframe row (peak > trough)
    /// and checks `lrKeyframesError` and the written schedule instead of
    /// `lrCycleMaxError`; the momentum half is unchanged.
    func test_validCycleEndpoints_noCrossFieldError() {
        let p = TrainingParameters.shared
        let savedKeyframes = p.lrCycleKeyframes
        let savedMMin = p.momentumCycleMin, savedMMax = p.momentumCycleMax
        defer {
            p.lrCycleKeyframes = savedKeyframes
            p.momentumCycleMin = savedMMin; p.momentumCycleMax = savedMMax
        }

        let model = makeModel()
        model.lrKeyframeRows = [row(peak: "0.03", trough: "0.001")]   // peak > trough
        model.momentumCycleMinText = "0.85"
        model.momentumCycleMaxText = "0.95"    // max > min
        model.save()

        XCTAssertNil(model.lrKeyframesError, "a valid LR keyframe must not flag the cross-field error")
        XCTAssertEqual(p.lrCycleKeyframes, LRKeyframeSchedule(keyframes: [
            LRKeyframe(step: 0, peak: 0.03, trough: 0.001, momentumAtPeak: 0.85, momentumAtTrough: 0.95),
        ]), "a valid edited schedule is written")
        XCTAssertFalse(model.momentumCycleMaxError, "valid momentum endpoints must not flag the cross-field error")
    }

    /// LR keyframes: peak == trough (was LR max == min) in a keyframe row,
    /// and equal momentum at the peak and at the trough (was follow
    /// high == low), are both legal.
    func test_equalCycleEndpoints_allowed() {
        let p = TrainingParameters.shared
        let savedKeyframes = p.lrCycleKeyframes
        defer { p.lrCycleKeyframes = savedKeyframes }

        let model = makeModel()
        model.lrKeyframeRows = [row(peak: "0.01", trough: "0.01",            // peak == trough is legal (flat cycle)
                                    momentumAtPeak: "0.9", momentumAtTrough: "0.9")]
        model.save()

        XCTAssertNil(model.lrKeyframesError, "peak == trough is a valid (degenerate flat) cycle")
        XCTAssertEqual(p.lrCycleKeyframes, LRKeyframeSchedule(keyframes: [
            LRKeyframe(step: 0, peak: 0.01, trough: 0.01, momentumAtPeak: 0.9, momentumAtTrough: 0.9),
        ]))
    }
}
