import XCTest
@testable import DrewsChessMachine

/// Step-enumerated checkpoint names carry the trainer step, so every segment
/// of a lineage run continues one series under its stem; a segment writes
/// and reaches only trainer steps above the one it starts from; the legacy
/// `-seg<k>` names files on disk still carry are still refused as a rolling
/// `--out-model`; a segment that trained no step writes no enumerated copy;
/// and the `[BATCH-STATS]` line logs only the summary it is meant to.
final class TrainerStepCheckpointNamingTests: XCTestCase {

    private var root: URL!

    override func setUpWithError() throws {
        root = FileManager.default.temporaryDirectory
            .appendingPathComponent("TrainerStepCheckpointNamingTests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        if let root, FileManager.default.fileExists(atPath: root.path) {
            try FileManager.default.removeItem(at: root)
        }
    }

    private func naming(_ rolling: String = "run-replay-latest.safetensors") -> EnumeratedCheckpointNaming {
        EnumeratedCheckpointNaming(rollingOutputURL: root.appendingPathComponent(rolling),
                                   runTag: EnumeratedCheckpointNaming.corpusReplayRunTag)
    }

    private func writeStepFiles(_ steps: [Int], naming: EnumeratedCheckpointNaming) throws {
        for step in steps {
            try Data("checkpoint".utf8).write(to: naming.url(trainerStep: step))
        }
    }

    // MARK: - Names

    func testALaterSegmentsFilesContinueTheRunsSeries() {
        let names = naming()
        XCTAssertEqual(names.fileName(trainerStep: 37_000), "run-replay-step37000.safetensors")
        XCTAssertFalse(names.fileName(trainerStep: 37_000).contains("-seg"))
        XCTAssertEqual(names.trainerStep(ofFileName: "run-replay-step37000.safetensors"), 37_000)
        XCTAssertNil(names.trainerStep(ofFileName: "run-replay-seg1-step1000.safetensors"),
                     "a legacy segment file is not one of this stem's trainer-step files")
        let vsuci = EnumeratedCheckpointNaming(rollingOutputURL: root.appendingPathComponent("uci"),
                                               runTag: EnumeratedCheckpointNaming.trainVsUciRunTag)
        XCTAssertEqual(vsuci.fileName(trainerStep: 2_113), "uci-step2113.safetensors")
    }

    func testLegacySegmentNamesAreStillRefusedAsAnOutModel() {
        for (name, step) in [("x-replay-seg1-step1000.safetensors", 1_000), ("x-vsuci-seg2-step7.safetensors", 7),
                             ("plain-seg3-step30.safetensors", 30)] {
            XCTAssertThrowsError(try TrainerOutputFileGuard.checkRollingOutput(
                outModelURL: root.appendingPathComponent(name), startModelURL: nil, startModel: nil,
                overwriteAuthorized: false), name) { error in
                XCTAssertEqual(error as? TrainerOutputFileError,
                               .outModelNamedLikeAnEnumeratedCheckpoint(path: self.root.appendingPathComponent(name).path,
                                                                        step: step), name)
            }
        }
    }

    // MARK: - Reach

    func testStepFilesAtOrBelowTheSegmentStartAreNotReachable() throws {
        let names = naming()
        try writeStepFiles([500, 1_000, 1_513, 2_000, 3_000], naming: names)
        XCTAssertEqual(try TrainerOutputFileGuard.reachableEnumeratedCheckpoints(
            naming: names, segmentStartTrainerStep: 1_513, stepLimit: nil).map(\.step), [2_000, 3_000])
        XCTAssertThrowsError(try TrainerOutputFileGuard.requireNoReachableEnumeratedCheckpoints(
            naming: names, segmentStartTrainerStep: 1_513, stepLimit: nil)) { error in
            guard case .enumeratedStepsAlreadyPresent(_, let count, let steps, let reachable, _)? =
                    error as? TrainerOutputFileError else {
                return XCTFail("expected enumeratedStepsAlreadyPresent, got \(error)")
            }
            XCTAssertEqual(count, 2)
            XCTAssertEqual(steps, "2000…3000")
            XCTAssertTrue(reachable.contains("1514"), reachable)
        }
        XCTAssertNoThrow(try TrainerOutputFileGuard.requireNoReachableEnumeratedCheckpoints(
            naming: names, segmentStartTrainerStep: 3_000, stepLimit: nil))
    }

    func testTheStepLimitCountsFromTheSegmentStart() throws {
        let names = naming()
        try writeStepFiles([1_000, 2_000, 2_113, 2_114, 3_000], naming: names)
        XCTAssertEqual(try TrainerOutputFileGuard.reachableEnumeratedCheckpoints(
            naming: names, segmentStartTrainerStep: 1_513, stepLimit: 600).map(\.step), [2_000, 2_113])
        XCTAssertEqual(TrainerOutputFileGuard.lastReachableTrainerStep(segmentStartTrainerStep: 1_513, stepLimit: 600),
                       2_113)
        XCTAssertEqual(TrainerOutputFileGuard.lastReachableTrainerStep(segmentStartTrainerStep: 1_513, stepLimit: nil),
                       Int.max)
    }

    func testTheLongestReachableNameIsAtTheStartPlusTheLimit() {
        // `<r…>-replay-latest.safetensors` of N bytes enumerates to
        // `<r…>-replay-step<T>.safetensors` of N − 2 + digits(T) bytes, so a
        // rolling name two bytes under the longest stageable one leaves room
        // for a four-digit step and not a five-digit one.
        let longest = Int(NAME_MAX) - FileSafety.temporarySiblingNameOverhead
        let suffix = "-replay-latest.safetensors"
        let rolling = String(repeating: "r", count: longest - 2 - suffix.utf8.count) + suffix
        let names = naming(rolling)
        XCTAssertNoThrow(try TrainerOutputFileGuard.requireNoReachableEnumeratedCheckpoints(
            naming: names, segmentStartTrainerStep: 9_000, stepLimit: 999), "trainer step 9999 still fits")
        XCTAssertThrowsError(try TrainerOutputFileGuard.requireNoReachableEnumeratedCheckpoints(
            naming: names, segmentStartTrainerStep: 9_000, stepLimit: 1_000), "trainer step 10000 does not")
        XCTAssertThrowsError(try TrainerOutputFileGuard.requireNoReachableEnumeratedCheckpoints(
            naming: names, segmentStartTrainerStep: 0, stepLimit: nil), "no limit reaches Int.max")
    }

    func testAStartPlusLimitOverflowReachesEveryStepAboveTheStart() throws {
        XCTAssertEqual(TrainerOutputFileGuard.lastReachableTrainerStep(segmentStartTrainerStep: Int.max - 5,
                                                                       stepLimit: 10), Int.max)
        XCTAssertEqual(TrainerOutputFileGuard.lastReachableTrainerStep(segmentStartTrainerStep: 10,
                                                                       stepLimit: Int.max), Int.max)
        let names = naming()
        try writeStepFiles([10, 5_000_000], naming: names)
        XCTAssertEqual(try TrainerOutputFileGuard.reachableEnumeratedCheckpoints(
            naming: names, segmentStartTrainerStep: 10, stepLimit: Int.max).map(\.step), [5_000_000])
    }

    func testAResumeFromTheStemsOwnStepFileKeepsTheStem() throws {
        // Segment 0 wrote its saves and its final file; a resume from that
        // final file starts at its trainer step and reaches nothing below.
        let names = naming()
        try writeStepFiles([1_000, 2_000, 2_513], naming: names)
        XCTAssertNoThrow(try TrainerOutputFileGuard.requireNoReachableEnumeratedCheckpoints(
            naming: names, segmentStartTrainerStep: 2_513, stepLimit: nil))
        XCTAssertNoThrow(try TrainerOutputFileGuard.requireNoReachableEnumeratedCheckpoints(
            naming: names, segmentStartTrainerStep: 2_513, stepLimit: 5_000))
        // A resume of an earlier file of the stem reaches the later files.
        XCTAssertThrowsError(try TrainerOutputFileGuard.requireNoReachableEnumeratedCheckpoints(
            naming: names, segmentStartTrainerStep: 1_000, stepLimit: nil))
    }

    func testNoCopyIsWrittenForASegmentThatTrainedNoStep() {
        XCTAssertFalse(TrainerOutputFileGuard.enumeratedCopyIsWritten(trainerStep: 36_000, segmentStartTrainerStep: 36_000))
        XCTAssertFalse(TrainerOutputFileGuard.enumeratedCopyIsWritten(trainerStep: 0, segmentStartTrainerStep: 0))
        XCTAssertTrue(TrainerOutputFileGuard.enumeratedCopyIsWritten(trainerStep: 36_001, segmentStartTrainerStep: 36_000))
        XCTAssertTrue(TrainerOutputFileGuard.enumeratedCopyIsWritten(trainerStep: 1, segmentStartTrainerStep: 0))
    }

    // MARK: - [BATCH-STATS]

    func testBatchStatsLineOnlyForThisStepsSummary() {
        // The CLI rule: only the summary of the very step it logs.
        XCTAssertTrue(BatchStatsLogLine.isDue(summaryStep: 2_000, ofTrainerStep: 2_000))
        XCTAssertFalse(BatchStatsLogLine.isDue(summaryStep: 1_990, ofTrainerStep: 2_000))
        XCTAssertFalse(BatchStatsLogLine.isDue(summaryStep: nil, ofTrainerStep: 2_000))
        // The GUI rule: the latest summary when its step differs from the
        // last one logged — including after a promotion rewinds the clock.
        XCTAssertTrue(BatchStatsLogLine.isDue(summaryStep: 50, lastLoggedStep: nil))
        XCTAssertTrue(BatchStatsLogLine.isDue(summaryStep: 100, lastLoggedStep: 50))
        XCTAssertFalse(BatchStatsLogLine.isDue(summaryStep: 100, lastLoggedStep: 100))
        XCTAssertTrue(BatchStatsLogLine.isDue(summaryStep: 900, lastLoggedStep: 1_200), "after a rewind")
        XCTAssertFalse(BatchStatsLogLine.isDue(summaryStep: nil, lastLoggedStep: 100))
        XCTAssertNil(BatchStatsLogLine.line(summary: nil, ofTrainerStep: 10))
        XCTAssertNil(BatchStatsLogLine.line(summary: nil, lastLoggedStep: nil))
    }
}
