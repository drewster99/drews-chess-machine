import XCTest
@testable import DrewsChessMachine

/// Step-enumerated checkpoint names written now carry the trainer step and
/// no segment marker (a run's first segment keeps the names runs have always
/// written); the `-seg<k>` names resumed segments wrote before that are
/// still recognized under any stem, so a rolling `--out-model` is never named
/// like one; and the lineage records still take the segment index from one
/// rule.
final class SegmentIndexedCheckpointNamingTests: XCTestCase {

    private let root = URL(fileURLWithPath: "/tmp/dcm-segment-naming")

    private func naming(_ rolling: String, _ runTag: String) -> EnumeratedCheckpointNaming {
        EnumeratedCheckpointNaming(
            rollingOutputURL: root.appendingPathComponent(rolling), runTag: runTag)
    }

    func testSegmentZeroKeepsTheExistingNames() {
        XCTAssertEqual(naming("run-replay-latest.safetensors", "replay").fileName(trainerStep: 41000),
                       "run-replay-step41000.safetensors")
        XCTAssertEqual(naming("run-vsuci-latest.safetensors", "vsuci").fileName(trainerStep: 7),
                       "run-vsuci-step7.safetensors")
        XCTAssertEqual(naming("plain.safetensors", "replay").fileName(trainerStep: 3),
                       "plain-step3.safetensors")
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
    }
}
