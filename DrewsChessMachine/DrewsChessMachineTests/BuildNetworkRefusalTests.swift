//
//  BuildNetworkRefusalTests.swift
//  DrewsChessMachineTests
//
//  A refused Build Network has no side effects. The Build sheet used to
//  store its architecture and seed on the session controller, which then
//  drew a seed and logged a `[BUTTON] Build Network … init_seed=` line before
//  checking whether it could build at all — so a build refused because the
//  app had become busy (or a network had been built) while the sheet was
//  open logged a build that never ran and left the refused design behind as
//  the next build's default. The architecture and seed are now parameters,
//  and every refusal comes first.
//

import XCTest
@testable import DrewsChessMachine

@MainActor
final class BuildNetworkRefusalTests: XCTestCase {

    /// A controller whose refusals and display clears are recorded.
    private final class Recorder {
        var refusals: [String] = []
        var displayClears = 0
    }

    /// Fed counts planted on the controller before a build. Dropping the
    /// trainer (`dropTrainerEndingLineageSegment`) resets them, so finding
    /// them unchanged afterwards shows the trainer was not dropped.
    private let plantedFedCarry = SessionController.LineageFedCarry(games: 7, positions: 11)

    private func controller(recordingInto recorder: Recorder) -> SessionController {
        let controller = SessionController()
        controller.onRefuseMenuAction = { recorder.refusals.append($0) }
        controller.lineageFedCarry = plantedFedCarry
        controller.onClearTrainingDisplay = { recorder.displayClears += 1 }
        return controller
    }

    private func assertNothingHappened(_ controller: SessionController, _ recorder: Recorder,
                                       file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertEqual(recorder.refusals.count, 1, file: file, line: line)
        XCTAssertFalse(controller.isBuilding, file: file, line: line)
        XCTAssertEqual(controller.lineageFedCarry, plantedFedCarry, file: file, line: line)
        XCTAssertEqual(recorder.displayClears, 0, file: file, line: line)
    }

    func testRefusedBuildWhileBusyHasNoSideEffects() {
        let recorder = Recorder()
        let controller = controller(recordingInto: recorder)
        controller.isBusyProvider = { true }
        controller.busyReasonProvider = { "busy" }
        controller.buildNetwork(architecture: .preset(.v3_8block_3x3), enteredInitSeed: 7)
        assertNothingHappened(controller, recorder)
        XCTAssertEqual(recorder.refusals, ["busy"])
        XCTAssertNil(controller.network)
    }

    func testRefusedBuildWithANetworkAlreadyBuiltHasNoSideEffects() throws {
        var arch = NetworkArchitecture.current
        arch.blockGroups[0].count = 1
        arch.blockGroups[0].channels = 16
        let existing = try ChessMPSNetwork(.randomWeights(initSeed: 1), arch: arch)
        let recorder = Recorder()
        let controller = controller(recordingInto: recorder)
        controller.network = existing
        controller.buildNetwork(architecture: .preset(.v3_8block_3x3), enteredInitSeed: 7)
        assertNothingHappened(controller, recorder)
        XCTAssertTrue(controller.network === existing)
    }

    func testATowerTooLargeForThisMacIsRefusedBeforeBuilding() throws {
        var arch = NetworkArchitecture.current
        arch.blockGroups[0].count = 1 << 30
        let recorder = Recorder()
        let controller = controller(recordingInto: recorder)
        controller.buildNetwork(architecture: arch, enteredInitSeed: nil)
        assertNothingHappened(controller, recorder)
        let refusal = try XCTUnwrap(recorder.refusals.first)
        XCTAssertTrue(refusal.contains("physical memory"), refusal)
        XCTAssertNil(controller.network)
    }
}
