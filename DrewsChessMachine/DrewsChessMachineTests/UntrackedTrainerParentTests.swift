//
//  UntrackedTrainerParentTests.swift
//  DrewsChessMachineTests
//
//  A Play-and-Train start that keeps a trainer whose history this process
//  never tracked begins a new run with that trainer as its parent. The
//  parent's model ID is written into every later record, so a trainer with
//  no identity must stop the start rather than be recorded under a
//  placeholder.
//

import XCTest
@testable import DrewsChessMachine

final class UntrackedTrainerParentTests: XCTestCase {

    func testATrainerWithoutAnIdentityIsRefused() {
        XCTAssertThrowsError(try LineageTracker.ParentFile.untrackedTrainer(identifier: nil, completedSteps: 40)) { error in
            XCTAssertTrue(String(describing: error).contains("model ID"), "\(error)")
        }
    }

    func testAnIdentifiedTrainerIsTheParentWithUnrecordedHistory() throws {
        let parent = try LineageTracker.ParentFile.untrackedTrainer(
            identifier: ModelID(value: "20261003-2-KEPT"), completedSteps: 40)
        XCTAssertEqual(parent.modelID, "20261003-2-KEPT")
        XCTAssertEqual(parent.trainerCompletedSteps, 40)
        XCTAssertNil(parent.contentSHA256)
        XCTAssertNil(parent.lineage.record)
        XCTAssertTrue(parent.derivationHistory.isEmpty)
    }
}
