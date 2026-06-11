import XCTest
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

    /// Absent field ⇒ the saved run sampled batches with no per-game
    /// cap; the range maximum is the closest representable equivalent.
    func testMaxPliesFromAnyOneGameAbsentFieldResolvesToRangeMax() {
        XCTAssertEqual(SessionCheckpointState.resolvedMaxPliesFromAnyOneGame(saved: nil), 400)
        XCTAssertEqual(SessionCheckpointState.resolvedMaxPliesFromAnyOneGame(saved: 10), 10)
    }

    /// Absent field ⇒ the saved run predates LR/momentum cycling: the
    /// user's current cycle NUMBERS are preserved (so the popover keeps
    /// their values) but both enabled flags must come back off — a live
    /// cycle must never be applied to a pre-feature session.
    func testLRMomentumCycleAbsentFieldResolvesToDisabledFlags() {
        let liveCycle = LRMomentumCycle(
            lrEnabled: true,
            lrPeriodSteps: 1234,
            lrCount: 2,
            lrMin: 0.002,
            lrMax: 0.02,
            lrInvert: true,
            momentumEnabled: true,
            momentumPeriodSteps: 4321,
            momentumCount: 3,
            momentumMin: 0.5,
            momentumMax: 0.95,
            momentumInvert: false
        )
        let resolved = SessionCheckpointState.resolvedLRMomentumCycle(saved: nil, current: liveCycle)
        XCTAssertFalse(resolved.lrEnabled, "a live LR cycle must not apply to a pre-feature session")
        XCTAssertFalse(resolved.momentumEnabled, "a live momentum cycle must not apply to a pre-feature session")
        XCTAssertEqual(resolved.lrPeriodSteps, 1234, "the user's cycle numbers should be preserved, only disabled")
        XCTAssertEqual(resolved.momentumPeriodSteps, 4321)
        XCTAssertEqual(resolved.lrMin, 0.002)
        XCTAssertEqual(resolved.momentumMax, 0.95)

        let saved = liveCycle
        XCTAssertEqual(
            SessionCheckpointState.resolvedLRMomentumCycle(saved: saved, current: .disabled),
            saved,
            "a session saved with a cycle must resume with exactly that cycle"
        )
    }
}
