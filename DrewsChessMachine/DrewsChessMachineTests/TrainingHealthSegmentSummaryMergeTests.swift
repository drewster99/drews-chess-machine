import XCTest
@testable import DrewsChessMachine

/// `TrainingHealthSegmentSummary.merging(_:)`: the typed value
/// HPARAM_RECORDING_PLAN P4 records as `configuration.health_alarms`, merged
/// across a GUI segment's monitors (OD-10).
final class TrainingHealthSegmentSummaryMergeTests: XCTestCase {

    typealias Summary = TrainingHealthSegmentSummary

    func testMergeSumsCountsKeepsEarliestStepAndHighestSeverity() {
        let first = Summary(evaluations: 20, raised: [
            .init(rule: .deadChannels, firstTrainerStep: 300, highestSeverity: .warning, raiseCount: 1),
            .init(rule: .lossSpike, firstTrainerStep: 900, highestSeverity: .warning, raiseCount: 2),
        ])
        let second = Summary(evaluations: 5, raised: [
            .init(rule: .nonFinite, firstTrainerStep: 1200, highestSeverity: .critical, raiseCount: 1),
            .init(rule: .deadChannels, firstTrainerStep: 1100, highestSeverity: .critical, raiseCount: 1),
        ])
        let merged = first.merging(second)
        XCTAssertEqual(merged.evaluations, 25)
        XCTAssertEqual(merged.raised.map(\.rule), [.nonFinite, .deadChannels, .lossSpike], "rule order")
        XCTAssertEqual(merged.raised[1], .init(rule: .deadChannels, firstTrainerStep: 300, highestSeverity: .critical, raiseCount: 2))
        XCTAssertEqual(second.merging(first), merged, "order-independent")
    }

    func testEmptyIsTheIdentity() {
        let summary = Summary(evaluations: 3, raised: [
            .init(rule: .gradientSpike, firstTrainerStep: 50, highestSeverity: .warning, raiseCount: 1),
        ])
        XCTAssertEqual(summary.merging(.empty), summary)
        XCTAssertEqual(Summary.empty.merging(summary), summary)
    }

    func testEncodesInTheRecordedShape() throws {
        let summary = Summary(evaluations: 12, raised: [
            .init(rule: .illegalMass, firstTrainerStep: 350, highestSeverity: .critical, raiseCount: 1),
        ])
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        XCTAssertEqual(
            String(decoding: try encoder.encode(summary), as: UTF8.self),
            #"{"evaluations":12,"raised":[{"first_trainer_step":350,"highest_severity":"critical","raise_count":1,"rule":"illegal_mass"}]}"#)
    }
}
