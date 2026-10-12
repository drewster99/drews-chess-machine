import XCTest
import TrainingParametersMacroSupport
@testable import DrewsChessMachine

/// Locks the resume-time fallback policy for session-persisted training
/// parameters whose checkpoint field was introduced together with the
/// feature it controls ("feature-introducing" fields).
///
/// For such fields, an absent value in `session.json` proves the file
/// predates the feature — the saved run factually trained in the
/// pre-feature regime. The nil-fallback must therefore reproduce that
/// pre-feature behavior, NOT the user's current `TrainingParameters`
/// default. Falling through to the live default silently changes the
/// training regime of an old run on resume; that is exactly what
/// happened when a 2026-05 KbHZ-era session resumed with the (then
/// nonexistent) signed-advantage complement CE switched on, producing
/// an unexplained policy-loss jump against its own chart history.
///
/// These tests are deliberately about the POLICY, not Codable
/// mechanics: if someone "simplifies" the fallback back to the current
/// default, they fail.
final class SessionResumeParameterFallbackTests: XCTestCase {

    /// Absent field ⇒ the session predates complement CE ⇒ the resumed
    /// run must use the legacy clamp-on regime (off) — even though the
    /// live `TrainingParameters` default for fresh sessions is on.
    func testComplementCEAbsentFieldResolvesToOff() {
        XCTAssertFalse(
            SessionCheckpointState.resolvedSignedAdvantageComplementCE(savedFlag: nil),
            "a session file without the complement-CE field predates the feature; resume must preserve the legacy clamp-on regime (off), not inherit the live default"
        )
    }

    /// A present field is authoritative in both directions.
    func testComplementCEPresentFieldIsAuthoritative() {
        XCTAssertTrue(
            SessionCheckpointState.resolvedSignedAdvantageComplementCE(savedFlag: true),
            "a session saved with complement CE on must resume with it on"
        )
        XCTAssertFalse(
            SessionCheckpointState.resolvedSignedAdvantageComplementCE(savedFlag: false),
            "a session saved with complement CE off must resume with it off"
        )
    }

    /// Absent field ⇒ the session predates the momentum feature ⇒ the
    /// saved run was plain SGD.
    func testMomentumAbsentFieldResolvesToZero() {
        XCTAssertEqual(SessionCheckpointState.resolvedMomentumCoeff(saved: nil), 0.0)
        XCTAssertEqual(SessionCheckpointState.resolvedMomentumCoeff(saved: 0.65), 0.65)
    }

    /// Absent field ⇒ the session predates channel dropout ⇒ the saved run
    /// trained at rate 0; resume must NOT inherit the live dropout rate.
    func testDropoutRateAbsentFieldResolvesToZero() {
        XCTAssertEqual(SessionCheckpointState.resolvedDropoutRate(saved: nil), 0.0)
        XCTAssertEqual(SessionCheckpointState.resolvedDropoutRate(saved: 0.30), 0.30)
    }

    /// Absent field ⇒ no illegal-mass penalty term existed in the loss.
    func testIllegalMassWeightAbsentFieldResolvesToZero() {
        XCTAssertEqual(SessionCheckpointState.resolvedIllegalMassPenaltyWeight(saved: nil), 0.0)
        XCTAssertEqual(SessionCheckpointState.resolvedIllegalMassPenaltyWeight(saved: 1.0), 1.0)
    }

    /// Absent field ⇒ the saved run trained with one-hot policy CE.
    func testPolicyLabelSmoothingAbsentFieldResolvesToZero() {
        XCTAssertEqual(SessionCheckpointState.resolvedPolicyLabelSmoothingEpsilon(saved: nil), 0.0)
        XCTAssertEqual(SessionCheckpointState.resolvedPolicyLabelSmoothingEpsilon(saved: 0.1), 0.1)
    }

    /// Absent field ⇒ the saved run trained with a hard one-hot W/D/L
    /// target (no value-head label smoothing). Must NOT inherit the live
    /// `TrainingParameters` value on resume — same regime-preservation
    /// contract as the sibling loss-shaping resolvers.
    func testValueLabelSmoothingAbsentFieldResolvesToZero() {
        XCTAssertEqual(SessionCheckpointState.resolvedValueLabelSmoothingEpsilon(saved: nil), 0.0)
        XCTAssertEqual(SessionCheckpointState.resolvedValueLabelSmoothingEpsilon(saved: 0.05), 0.05)
    }

    /// Absent field ⇒ the saved run sampled batches with no per-game
    /// cap; the range maximum is the closest representable equivalent.
    func testMaxPliesFromAnyOneGameAbsentFieldResolvesToRangeMax() {
        XCTAssertEqual(SessionCheckpointState.resolvedMaxPliesFromAnyOneGame(saved: nil), 400)
        XCTAssertEqual(SessionCheckpointState.resolvedMaxPliesFromAnyOneGame(saved: 10), 10)
    }

    /// Absent field ⇒ the saved run predates LR/momentum cycling: the
    /// user's current cycle NUMBERS (and keyframes) are preserved (so the
    /// popover keeps their values) but both enabled flags and momentum
    /// following must come back off — a live cycle must never be applied to
    /// a pre-feature session.
    ///
    /// Rewritten for LR keyframes: `SessionCheckpointState.resolvedLRMomentumCycle`
    /// was removed; the GUI resume now resolves each cycle key from
    /// `rs.lrSchedule` through `TrainingParameterResolution` (the keys'
    /// declared `absentValue`s), so the same policy is pinned there.
    func testLRMomentumCycleAbsentFieldResolvesToDisabledFlags() {
        let liveCycle = LRMomentumCycle(
            lrEnabled: true,
            lrPeriodSteps: 1234,
            lrCount: 2,
            lrInvert: true,
            lrKeyframes: LRKeyframeSchedule(keyframes: [
                LRKeyframe(step: 0, peak: 0.02, trough: 0.002, momentumAtPeak: 0.85, momentumAtTrough: 0.95),
                LRKeyframe(step: 5000, peak: 0.002, trough: 0.0002, momentumAtPeak: 0.9, momentumAtTrough: 0.95),
            ]),
            momentumEnabled: true,
            momentumPeriodSteps: 4321,
            momentumCount: 3,
            momentumMin: 0.5,
            momentumMax: 0.95,
            momentumInvert: false,
            momentumFollowsLRCycle: true
        )
        let resolved = Self.resolvedCycle(saved: nil, current: liveCycle)
        XCTAssertFalse(resolved.cycle.lrEnabled, "a live LR cycle must not apply to a pre-feature session")
        XCTAssertFalse(resolved.cycle.momentumEnabled, "a live momentum cycle must not apply to a pre-feature session")
        XCTAssertFalse(resolved.cycle.momentumFollowsLRCycle, "live momentum following must not apply to a pre-feature session")
        XCTAssertEqual(resolved.cycle.lrPeriodSteps, 1234, "the user's cycle numbers should be preserved, only disabled")
        XCTAssertEqual(resolved.cycle.momentumPeriodSteps, 4321)
        XCTAssertEqual(resolved.cycle.lrKeyframes, liveCycle.lrKeyframes, "the user's keyframes should be preserved")
        XCTAssertEqual(resolved.cycle.momentumMax, 0.95)
        XCTAssertEqual(resolved.preFeatureKeyIDs, [LRCycleEnabled.id, MomentumCycleEnabled.id, MomentumFollowsLRCycle.id],
                       "exactly the flags take their pre-feature value; every number is the live setting")

        let saved = liveCycle
        let resumed = Self.resolvedCycle(saved: saved, current: .disabled)
        XCTAssertEqual(resumed.cycle, saved, "a session saved with a cycle must resume with exactly that cycle")
        XCTAssertEqual(resumed.preFeatureKeyIDs, [])
    }

    /// The cycle the GUI resume (`SessionParameterResume.applyGuiSession`)
    /// leaves in force for a session schedule `saved` over the live `current`,
    /// key by key as that resume restores it, and the ids that resolved to a
    /// pre-feature value.
    private static func resolvedCycle(
        saved: LRMomentumCycle?,
        current: LRMomentumCycle
    ) -> (cycle: LRMomentumCycle, preFeatureKeyIDs: [String]) {
        var preFeature: [String] = []
        func resolve<K: TrainingParameterKey>(_ key: K.Type, _ savedValue: K.Value?, _ currentValue: K.Value) -> K.Value {
            let resolved = TrainingParameterResolution.resolve(key, saved: savedValue, current: currentValue)
            if resolved.source == .preFeature { preFeature.append(K.id) }
            return resolved.applied
        }
        let cycle = LRMomentumCycle(
            lrEnabled: resolve(LRCycleEnabled.self, saved?.lrEnabled, current.lrEnabled),
            lrPeriodSteps: resolve(LRCyclePeriodSteps.self, saved?.lrPeriodSteps, current.lrPeriodSteps),
            lrCount: resolve(LRCycleCount.self, saved?.lrCount, current.lrCount),
            lrInvert: resolve(LRCycleInvert.self, saved?.lrInvert, current.lrInvert),
            lrKeyframes: resolve(LRCycleKeyframes.self, saved?.lrKeyframes, current.lrKeyframes),
            momentumEnabled: resolve(MomentumCycleEnabled.self, saved?.momentumEnabled, current.momentumEnabled),
            momentumPeriodSteps: resolve(MomentumCyclePeriodSteps.self, saved?.momentumPeriodSteps, current.momentumPeriodSteps),
            momentumCount: resolve(MomentumCycleCount.self, saved?.momentumCount, current.momentumCount),
            momentumMin: resolve(MomentumCycleMin.self, saved?.momentumMin, current.momentumMin),
            momentumMax: resolve(MomentumCycleMax.self, saved?.momentumMax, current.momentumMax),
            momentumInvert: resolve(MomentumCycleInvert.self, saved?.momentumInvert, current.momentumInvert),
            momentumFollowsLRCycle: resolve(MomentumFollowsLRCycle.self, saved?.momentumFollowsLRCycle, current.momentumFollowsLRCycle)
        )
        return (cycle, preFeature)
    }
}
