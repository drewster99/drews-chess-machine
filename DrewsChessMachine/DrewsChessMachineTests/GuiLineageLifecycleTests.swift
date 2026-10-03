//
//  GuiLineageLifecycleTests.swift
//  DrewsChessMachineTests
//
//  The GUI lineage segment lives exactly as long as the trainer whose
//  training it records:
//  - rebuilding the champion for another architecture drops the trainer,
//    and its lineage segment must end with it — both rebuild paths, Build
//    Network and the auto-build before a load, the same way — so no later
//    save attributes another trainer's weights to the old segment;
//  - a Play-and-Train start whose lineage segment cannot begin leaves
//    things as a next start can use them: a loaded session stays pending
//    (it is redone in full next time), a continuing segment stays as it
//    was, and a segment whose trainer was just reset is not left behind
//    describing weights it no longer matches.
//

import XCTest
@testable import DrewsChessMachine

@MainActor
final class GuiLineageLifecycleTests: XCTestCase {

    private static let architecture = ResumeEquivalenceTests.architecture

    private func resumedTracker(segmentStart: Int) throws -> LineageTracker {
        let parent = LineageTracker.ParentFile(
            modelID: "20261003-2-OLDT", contentSHA256: nil, trainerCompletedSteps: segmentStart,
            lineage: .recorded(try LineageRecord.forTests(trainerCompletedSteps: segmentStart, corpus: nil)),
            derivationHistory: [])
        return try LineageTracker(start: .resume(parent: parent, gaps: [], legacyTotals: nil), pathKind: .gui,
                                  argv: ["DrewsChessMachine"], startedAt: Date(), segmentStartTrainerStep: segmentStart)
    }

    private func trainer() throws -> ChessTrainer {
        try ChessTrainer(dropoutStream: DCMRandom(seed: 31),
                         hyperparameters: TrainerHyperparameters(TrainingParameters.shared.snapshot()),
                         arch: Self.architecture, initialization: .seeded(initSeed: 31))
    }

    func testRebuildingTheChampionForAnotherArchitectureEndsTheTrainersLineageSegment() async throws {
        let controller = SessionController()
        controller.trainer = try trainer()
        controller.lineageTracker = try resumedTracker(segmentStart: 50)
        controller.lineageFedCarry = SessionController.LineageFedCarry(games: 7, positions: 300,
                                                                       baselineGames: 2, baselinePositions: 80)

        let rebuilt = await controller.ensureChampionBuilt(arch: Self.architecture)
        guard case .success = rebuilt else { return XCTFail("the rebuild failed: \(rebuilt)") }

        XCTAssertNil(controller.trainer, "the rebuild drops the trainer")
        XCTAssertNil(controller.lineageTracker, "the dropped trainer's lineage segment ends with it")
        XCTAssertEqual(controller.lineageFedCarry, SessionController.LineageFedCarry())
    }
}
