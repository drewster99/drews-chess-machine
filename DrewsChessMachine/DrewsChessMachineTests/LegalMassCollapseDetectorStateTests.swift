import XCTest
@testable import DrewsChessMachine

/// The legal-mass-collapse detector's window and grace anchor survive a
/// session save and resume (determinism plan C1 #20): a resume neither
/// restarts the grace period nor forgets the probes building toward (or
/// away from) a collapse.
final class LegalMassCollapseDetectorStateTests: XCTestCase {

    private let t0 = Date(timeIntervalSince1970: 1_790_000_000)

    func testTheWindowKeepsTheNewestReadings() {
        let detector = LegalMassCollapseDetectorBox()
        XCTAssertEqual(detector.append(legalMass: 0.1, capacity: 3), [0.1])
        XCTAssertEqual(detector.append(legalMass: 0.2, capacity: 3), [0.1, 0.2])
        XCTAssertEqual(detector.append(legalMass: 0.3, capacity: 3), [0.1, 0.2, 0.3])
        XCTAssertEqual(detector.append(legalMass: 0.4, capacity: 3), [0.2, 0.3, 0.4])
        // A restored window longer than the current capacity is cut to the
        // newest readings at the next probe.
        detector.restore(LegalMassCollapseDetectorState(legalMassWindow: [0.1, 0.2, 0.3, 0.4, 0.5],
                                                       graceElapsedSec: nil))
        XCTAssertEqual(detector.append(legalMass: 0.6, capacity: 2), [0.5, 0.6])
    }

    func testGraceStartsAtTheFirstObservedTrainingStep() {
        let detector = LegalMassCollapseDetectorBox()
        XCTAssertNil(detector.snapshot(now: t0).graceElapsedSec)
        XCTAssertEqual(detector.graceElapsed(observingTrainingAt: t0), 0)
        XCTAssertEqual(detector.graceElapsed(observingTrainingAt: t0.addingTimeInterval(45)), 45)
        XCTAssertEqual(detector.snapshot(now: t0.addingTimeInterval(50)).graceElapsedSec, 50)
    }

    func testAResumeCarriesTheGraceItHadUsed() {
        let saved = LegalMassCollapseDetectorState(legalMassWindow: [0.004, 0.005], graceElapsedSec: 90)
        let detector = LegalMassCollapseDetectorBox()
        detector.restore(saved)
        // While the resumed run refills its buffer no grace is spent, and a
        // save in that window records the carried amount unchanged.
        detector.noteNoTrainingStepsYet()
        XCTAssertEqual(detector.snapshot(now: t0.addingTimeInterval(600)), saved)
        // The first observed step re-anchors the countdown at the carried amount.
        let resumedAt = t0.addingTimeInterval(1_000)
        XCTAssertEqual(detector.graceElapsed(observingTrainingAt: resumedAt), 90, accuracy: 1e-9)
        XCTAssertEqual(detector.graceElapsed(observingTrainingAt: resumedAt.addingTimeInterval(30)), 120, accuracy: 1e-9)
        XCTAssertEqual(detector.append(legalMass: 0.006, capacity: 3), [0.004, 0.005, 0.006])
    }

    func testTheStateRoundTripsThroughSessionJSON() throws {
        let detector = LegalMassCollapseDetectorState(legalMassWindow: [0.01, 0.02, 0.015], graceElapsedSec: 240.5)
        let base = SessionCheckpointState(
            formatVersion: SessionCheckpointState.currentFormatVersion,
            sessionID: "legal-mass", savedAtUnix: 1_790_000_000, sessionStartUnix: 1_789_999_000,
            elapsedTrainingSec: 10, trainingSteps: 1, selfPlayGames: 1, selfPlayMoves: 1,
            trainingPositionsSeen: 1, batchSize: 1, learningRate: 1e-3,
            promoteThreshold: 0.55, arenaGames: 10,
            selfPlayTau: TauConfigCodable(SamplingSchedule.selfPlay),
            arenaTau: TauConfigCodable(SamplingSchedule.arena),
            selfPlayWorkerCount: 1, championID: "c", trainerID: "t", arenaHistory: []
        ).withLineage(LineageRecord.sessionTestFixture)
        let decoded = try SessionCheckpointState.decode(try base.withLegalMassCollapseDetector(detector).encode())
        XCTAssertEqual(decoded.legalMassCollapseDetector, detector)
        // A session saved before the detector state was recorded decodes
        // without it, and the resume logs that the detector starts fresh.
        XCTAssertNil(try SessionCheckpointState.decode(try base.encode()).legalMassCollapseDetector)
    }
}
