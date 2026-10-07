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

    /// An edit away from a resume's held out-of-range value is journalled
    /// (final review M2): the old value being outside today's range is no
    /// reason to drop a committed change.
    func testAnEditAwayFromAHeldOutOfRangeValueIsReported() {
        let params = TrainingParameters.shared
        let outOfRange = -1.0
        XCTAssertFalse(EntropyBonus.isWithinDeclaration(outOfRange))
        params.restoreFromSession(EntropyBonus.self, outOfRange, into: \.entropyBonus)
        XCTAssertEqual(params.entropyBonus, outOfRange, "the session's value is held")
        observe()
        params.entropyBonus = 0.002
        let events = seen.value
        XCTAssertEqual(events.count, 1, "\(events.map(\.0))")
        XCTAssertEqual(events.first?.0, EntropyBonus.id)
        XCTAssertEqual(events.first?.1, EntropyBonus.encode(outOfRange))
        XCTAssertEqual(events.first?.2, EntropyBonus.encode(0.002))
    }

    /// A rejected assignment while a resume's held out-of-range value is in
    /// force puts that value back without validating it, journals nothing
    /// and keeps the hold (the revert is recognised, not re-validated; it
    /// used to be rejected in turn, reverting back and forth).
    func testARejectedAssignmentOverAHeldOutOfRangeValueRestoresIt() {
        let params = TrainingParameters.shared
        params.restoreFromSession(EntropyBonus.self, -1.0, into: \.entropyBonus)
        XCTAssertNotNil(TrainingParameters.runHeldPriorValues[EntropyBonus.id])
        observe()
        params.entropyBonus = -2.0
        XCTAssertEqual(params.entropyBonus, -1.0, "the held value is back")
        XCTAssertTrue(seen.value.isEmpty, "\(seen.value.map(\.0))")
        XCTAssertNotNil(TrainingParameters.runHeldPriorValues[EntropyBonus.id], "the hold is kept")
        params.entropyBonus = 0.002
        XCTAssertEqual(seen.value.count, 1, "the next real edit is journalled")
    }

    /// The results row's lineage (final review m3) is built at the trainer
    /// clock of its own configuration cut, so a change journalled after an
    /// earlier read of the clock cannot sit above the record's clock.
    func testTheResultsLineageIsAtTheCutsClockAfterALaterChange() throws {
        let harness = try GuiSaveHarness()
        defer {
            do { try harness.removeSessionsDirectory() } catch { XCTFail("\(error)") }
        }
        let controller = harness.controller
        let tracker = try XCTUnwrap(controller.lineageTracker)
        controller.installParameterChangeJournal(tracker: tracker, trainer: harness.trainer)
        let earlierRead = harness.trainer.completedTrainSteps
        harness.trainer.completedTrainSteps = earlierRead + 3
        let params = TrainingParameters.shared
        params.entropyBonus = params.entropyBonus == 0.002 ? 0.003 : 0.002

        let lineage = try controller.lineageForResults()

        XCTAssertEqual(lineage.record.steps.cumTrainerStep, earlierRead + 3)
        XCTAssertEqual(lineage.record.configuration.value?.parameterChanges.count, 1)
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
