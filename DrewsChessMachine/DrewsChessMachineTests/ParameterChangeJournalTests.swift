//
//  ParameterChangeJournalTests.swift
//  DrewsChessMachineTests
//
//  The settings change journal (hyperparameter recording plan gap 4): every
//  committed change of a setting a running GUI segment reads is journalled
//  with the trainer step it takes effect at, so a record's
//  `configuration.parameter_changes` undoes its in-force snapshot back to
//  the segment's start. A change that is no change, a seed setting (the
//  run's actual seed is `run_seeds`), a key the run captured at its start
//  (journalled at the next start's recapture) and a rejected assignment are
//  not journalled.
//
//  Every test that assigns `TrainingParameters.shared` snapshots it in
//  setUp, suppresses persistence, and restores it in tearDown; the observer
//  is removed before the restore so the restore is not journalled.
//

import XCTest
@testable import DrewsChessMachine

@MainActor
final class ParameterChangeJournalTests: XCTestCase {

    private var savedParameterValues: [String: ParameterValue] = [:]
    private let seen = SyncBox<[(String, ParameterValue, ParameterValue)]>([])

    override func setUp() async throws {
        try await super.setUp()
        savedParameterValues = TrainingParameters.shared.snapshot().rawValueMap()
        TrainingParameters.suppressPersistence = true
    }

    override func tearDown() async throws {
        TrainingParameters.runChangeObserver.value = nil
        try TrainingParameters.shared.apply(savedParameterValues)
        TrainingParameters.suppressPersistence = false
        try await super.tearDown()
    }

    private func observe() {
        let seen = seen
        TrainingParameters.runChangeObserver.value = { id, old, new in
            seen.modify { $0.append((id, old, new)) }
        }
    }

    func testACommittedChangeIsReportedWithItsOldAndNewValues() {
        let params = TrainingParameters.shared
        let start = params.entropyBonus
        let next = start == 0.002 ? 0.003 : 0.002
        observe()
        params.entropyBonus = next
        let events = seen.value
        XCTAssertEqual(events.count, 1)
        XCTAssertEqual(events.first?.0, EntropyBonus.id)
        XCTAssertEqual(events.first?.1, EntropyBonus.encode(start))
        XCTAssertEqual(events.first?.2, EntropyBonus.encode(next))

        params.entropyBonus = next
        XCTAssertEqual(seen.value.count, 1, "assigning the same value is no change")
    }

    func testSeedSettingsCapturedKeysAndRejectedAssignmentsAreNotReported() {
        let params = TrainingParameters.shared
        observe()
        params.randomSeed = params.randomSeed == 11 ? 12 : 11
        params.trainingBatchSize = params.trainingBatchSize == 512 ? 1024 : 512
        params.entropyBonus = -1
        XCTAssertTrue(seen.value.isEmpty, "\(seen.value.map(\.0))")
        XCTAssertNotEqual(params.entropyBonus, -1, "the rejected assignment was reverted")
    }

    /// A GUI segment's journal: the change lands in the next save's record
    /// at the trainer step it takes effect from.
    func testAJournalledChangeIsInTheNextRecordAtTheTrainersClock() throws {
        let harness = try GuiSaveHarness()
        defer {
            do { try harness.removeSessionsDirectory() } catch { XCTFail("\(error)") }
        }
        let controller = harness.controller
        let tracker = try XCTUnwrap(controller.lineageTracker)
        controller.installParameterChangeJournal(tracker: tracker, trainer: harness.trainer)
        let params = TrainingParameters.shared
        let start = params.entropyBonus
        let next = start == 0.002 ? 0.003 : 0.002
        params.entropyBonus = next

        let record = try controller.lineageRecordForSave(
            at: Date(), cut: try controller.takeConfigurationCut(trainer: harness.trainer),
            trainerCompletedSteps: harness.trainer.completedTrainSteps,
            dropoutPhiloxState: nil, dropoutStreamState: nil)
        let changes = try XCTUnwrap(record.configuration.value?.parameterChanges)
        XCTAssertEqual(changes.count, 1)
        XCTAssertEqual(changes.first?.id, EntropyBonus.id)
        XCTAssertEqual(changes.first?.old, EntropyBonus.encode(start))
        XCTAssertEqual(changes.first?.new, EntropyBonus.encode(next))
        XCTAssertEqual(changes.first?.committedAtTrainerStep, harness.trainer.completedTrainSteps)
        XCTAssertNil(changes.first?.restampedFrom)
    }
}
