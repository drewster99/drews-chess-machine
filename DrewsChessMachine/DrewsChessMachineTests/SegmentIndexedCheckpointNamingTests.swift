import XCTest
@testable import DrewsChessMachine

/// Step-enumerated checkpoint names carry the writing segment's lineage
/// index (determinism plan C1 #26): a resumed segment's step files never
/// share a name with an earlier segment's, and segment 0 keeps the names
/// runs have always written.
final class SegmentIndexedCheckpointNamingTests: XCTestCase {

    private let root = URL(fileURLWithPath: "/tmp/dcm-segment-naming")

    private func naming(_ rolling: String, _ runTag: String, segment: Int) -> EnumeratedCheckpointNaming {
        EnumeratedCheckpointNaming(
            rollingOutputURL: root.appendingPathComponent(rolling), runTag: runTag, segmentIndex: segment)
    }

    func testSegmentZeroKeepsTheExistingNames() {
        XCTAssertEqual(naming("run-replay-latest.safetensors", "replay", segment: 0).fileName(step: 41000),
                       "run-replay-step41000.safetensors")
        XCTAssertEqual(naming("run-vsuci-latest.safetensors", "vsuci", segment: 0).fileName(step: 7),
                       "run-vsuci-step7.safetensors")
        XCTAssertEqual(naming("plain.safetensors", "replay", segment: 0).fileName(step: 3),
                       "plain-step3.safetensors")
    }

    func testALaterSegmentsNamesCarryItsIndex() {
        XCTAssertEqual(naming("run-replay-latest.safetensors", "replay", segment: 3).fileName(step: 41000),
                       "run-replay-seg3-step41000.safetensors")
        XCTAssertEqual(naming("run-vsuci-latest.safetensors", "vsuci", segment: 1).fileName(step: 7),
                       "run-vsuci-seg1-step7.safetensors")
        XCTAssertEqual(naming("plain.safetensors", "replay", segment: 2).fileName(step: 3),
                       "plain-seg2-step3.safetensors")
    }

    func testEachSegmentParsesOnlyItsOwnStepFiles() {
        let first = naming("run-replay-latest.safetensors", "replay", segment: 0)
        let second = naming("run-replay-latest.safetensors", "replay", segment: 1)
        let eleventh = naming("run-replay-latest.safetensors", "replay", segment: 10)
        XCTAssertEqual(first.step(ofFileName: "run-replay-step1000.safetensors"), 1000)
        XCTAssertNil(first.step(ofFileName: "run-replay-seg1-step1000.safetensors"))
        XCTAssertEqual(second.step(ofFileName: "run-replay-seg1-step1000.safetensors"), 1000)
        XCTAssertNil(second.step(ofFileName: "run-replay-step1000.safetensors"))
        XCTAssertNil(second.step(ofFileName: "run-replay-seg10-step1000.safetensors"))
        XCTAssertEqual(eleventh.step(ofFileName: "run-replay-seg10-step1000.safetensors"), 1000)
        XCTAssertNil(eleventh.step(ofFileName: "run-replay-seg1-step1000.safetensors"))
        // Segment 0 never writes a marker, so no segment owns a `-seg0-` name
        // built from this stem.
        XCTAssertNil(first.step(ofFileName: "run-replay-seg0-step1000.safetensors"))
        XCTAssertNil(second.step(ofFileName: "run-replay-seg01-step1000.safetensors"))
    }

    func testSegmentStepFilesAreRecognizedUnderAnyStem() {
        for name in ["run-replay-seg3-step41000.safetensors", "run-vsuci-seg1-step7.safetensors",
                     "plain-seg2-step3.safetensors", "run-replay-step29000.safetensors"] {
            XCTAssertNotNil(EnumeratedCheckpointNaming.step(ofEnumeratedFileNameUnderAnyStem: name), name)
        }
        XCTAssertEqual(EnumeratedCheckpointNaming.step(ofEnumeratedFileNameUnderAnyStem:
            "run-replay-seg3-step41000.safetensors"), 41000)
        // A stem that itself ends in `-seg<k>` is still a segment-0 stem.
        XCTAssertEqual(EnumeratedCheckpointNaming.step(ofEnumeratedFileNameUnderAnyStem:
            "x-replay-seg0-step5.safetensors"), 5)
        XCTAssertNil(EnumeratedCheckpointNaming.step(ofEnumeratedFileNameUnderAnyStem:
            "run-replay-seg3-latest.safetensors"))
    }

    /// The names and the lineage records take the segment index from one
    /// rule: a fresh run and a branch are segment 0, each exact resume of a
    /// recorded run the next index, and a resume of a file without lineage
    /// starts again at 0.
    func testTheSegmentIndexComesFromTheLineageRule() throws {
        XCTAssertEqual(LineageTracker.segmentIndex(exactResumeOf: nil), 0)
        let unrecorded = LineageTracker.ParentFile(
            modelID: "20261003-1-Abcd", contentSHA256: nil, trainerCompletedSteps: 10,
            lineage: .unrecorded(formatVersion: 6), derivationHistory: [])
        XCTAssertEqual(LineageTracker.segmentIndex(exactResumeOf: unrecorded), 0)

        let firstRecord = try LineageRecord.forTests(trainerCompletedSteps: 10, corpus: nil)
        XCTAssertEqual(firstRecord.run.segmentIndex, 0)
        let firstParent = LineageTracker.ParentFile(
            modelID: "20261003-2-Efgh", contentSHA256: nil, trainerCompletedSteps: 10,
            lineage: .recorded(firstRecord), derivationHistory: [])
        XCTAssertEqual(LineageTracker.segmentIndex(exactResumeOf: firstParent), 1)

        let start = Date(timeIntervalSince1970: 1_790_000_000)
        let resumed = try LineageTracker(
            start: .resume(parent: firstParent, gaps: [], legacyTotals: nil), pathKind: .replay,
            argv: ["DrewsChessMachine", "--test"], startedAt: start, segmentStartTrainerStep: 10)
        let secondRecord = try resumed.record(
            at: start.addingTimeInterval(60), trainerCompletedSteps: 20, segmentLocalStep: 10,
            segmentGames: 0, segmentPositions: 0, corpus: nil, parameters: nil,
            rng: .withoutRunStreams(dropoutPhiloxState: nil))
        XCTAssertEqual(secondRecord.run.segmentIndex, LineageTracker.segmentIndex(exactResumeOf: firstParent))
        XCTAssertEqual(
            naming("run-replay-latest.safetensors", "replay",
                   segment: secondRecord.run.segmentIndex).fileName(step: 1000),
            "run-replay-seg1-step1000.safetensors")
    }
}
