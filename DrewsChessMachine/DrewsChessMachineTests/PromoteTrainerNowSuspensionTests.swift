import XCTest
@testable import DrewsChessMachine

/// Train ▸ Promote Trainee Now is refused while training is suspended, for a
/// divergence and for a training-health stop alike (owner decision OD-5):
/// a suspended trainer must not become the champion. Uses the save harness
/// the other promote tests use; the champion's weights and ID must be
/// unchanged and nothing saved.
@MainActor
final class PromoteTrainerNowSuspensionTests: XCTestCase {

    private func assertRefused(
        _ suspension: TrainingSuspension,
        mentioning expected: String,
        file: StaticString = #filePath,
        line: UInt = #line
    ) async throws {
        let harness = try GuiSaveHarness()
        defer {
            do { try harness.removeSessionsDirectory() } catch { XCTFail("\(error)") }
        }
        var refusals: [String] = []
        harness.controller.onRefuseMenuAction = { refusals.append($0) }
        harness.controller.trainingSuspension = suspension
        let championID = harness.champion.identifier
        let weightsBefore = try await harness.champion.network.exportWeights()

        harness.controller.promoteTrainerNow(sessionsDirectory: harness.sessionsDirectory)

        XCTAssertEqual(refusals.count, 1, file: file, line: line)
        XCTAssertTrue(refusals.first?.contains(expected) == true, "\(refusals)", file: file, line: line)
        XCTAssertEqual(harness.champion.identifier, championID, file: file, line: line)
        let weightsAfter = try await harness.champion.network.exportWeights()
        XCTAssertEqual(weightsAfter, weightsBefore, "the champion's weights must not change", file: file, line: line)
        let saved = try FileManager.default.contentsOfDirectory(atPath: harness.sessionsDirectory.path)
        XCTAssertEqual(saved, [], "no promotion save", file: file, line: line)
    }

    func testRefusedDuringADivergenceSuspension() async throws {
        try await assertRefused(.divergence(reason: "non-finite loss"), mentioning: "divergence")
    }

    func testRefusedDuringAHealthAlarmSuspension() async throws {
        try await assertRefused(
            .healthAlarm(rule: .deadChannels, detail: "critical dead=339/1040 since trainer step 514"),
            mentioning: "dead_channels")
    }

    func testSuspensionGatesFollowTheCause() {
        let divergence = TrainingSuspension.divergence(reason: "x")
        let health = TrainingSuspension.healthAlarm(rule: .illegalMass, detail: "y")
        XCTAssertTrue(divergence.skipsPeriodicAutosave)
        XCTAssertFalse(health.skipsPeriodicAutosave, "finite weights: the periodic autosave runs")
        XCTAssertTrue(divergence.skipsHeartbeatAlarmEvaluation)
        XCTAssertFalse(health.skipsHeartbeatAlarmEvaluation)
        XCTAssertEqual(divergence.arenaSkipLabel, "divergence")
        XCTAssertEqual(health.arenaSkipLabel, "health alarm illegal_mass")
    }
}
