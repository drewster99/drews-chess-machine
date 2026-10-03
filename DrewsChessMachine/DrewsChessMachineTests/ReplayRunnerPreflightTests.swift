import XCTest
@testable import DrewsChessMachine

/// Pre-flight checks a corpus-replay or train-vs-UCI run makes before it
/// trains, so a problem that would otherwise surface hours in — or never —
/// stops the run at launch:
///
/// - every name the run will write through a staged save leaves room for the
///   staging name, or each save fails with ENAMETOOLONG and the run trains to
///   the end without ever writing a model;
/// - a training step in a model header is never negative (the step arithmetic
///   on it could overflow and trap).
final class ReplayRunnerPreflightTests: XCTestCase {

    private var root: URL!

    override func setUpWithError() throws {
        root = FileManager.default.temporaryDirectory
            .appendingPathComponent("ReplayRunnerPreflightTests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        if let root, FileManager.default.fileExists(atPath: root.path) {
            try FileManager.default.removeItem(at: root)
        }
    }

    /// The longest file name a staged write can publish: the staging sibling
    /// adds `temporarySiblingNameOverhead` bytes and must itself fit `NAME_MAX`.
    private var longestStageableName: Int { Int(NAME_MAX) - FileSafety.temporarySiblingNameOverhead }

    /// A `…-replay-latest.safetensors` name of exactly `bytes` UTF-8 bytes.
    private func rollingName(bytes: Int) -> String {
        let suffix = "-replay-latest.safetensors"
        return String(repeating: "r", count: bytes - suffix.utf8.count) + suffix
    }

    // MARK: Staging-name length

    func testRollingOutputNameTooLongToStageIsRefusedBeforeTraining() throws {
        let tooLong = root.appendingPathComponent(rollingName(bytes: longestStageableName + 1))
        for overwrite in [false, true] {
            XCTAssertThrowsError(try TrainerOutputFileGuard.checkRollingOutput(
                outModelURL: tooLong, startModelURL: nil, startModel: nil, overwriteAuthorized: overwrite)) { error in
                XCTAssertTrue(error.localizedDescription.contains(String(self.longestStageableName)),
                              error.localizedDescription)
            }
        }
        XCTAssertFalse(FileManager.default.fileExists(atPath: tooLong.path), "nothing is created by a refusal")
    }

    func testRollingOutputNameAtTheStagingLimitIsAccepted() throws {
        let atLimit = root.appendingPathComponent(rollingName(bytes: longestStageableName))
        XCTAssertNoThrow(try TrainerOutputFileGuard.checkRollingOutput(
            outModelURL: atLimit, startModelURL: nil, startModel: nil, overwriteAuthorized: false))
    }

    func testEnumeratedNamesThatCannotBeStagedRefuseTheRun() throws {
        // The rolling name fits, but `-latest` becomes `-step<N>`, and with no
        // step limit N can need every digit an Int has.
        let rolling = root.appendingPathComponent(rollingName(bytes: longestStageableName - 8))
        let naming = EnumeratedCheckpointNaming(rollingOutputURL: rolling, runTag: "replay")
        XCTAssertThrowsError(try TrainerOutputFileGuard.requireNoReachableEnumeratedCheckpoints(
            naming: naming, stepLimit: nil)) { error in
            XCTAssertTrue(error.localizedDescription.contains(String(self.longestStageableName)),
                          error.localizedDescription)
        }
        // With a small step limit every step name still fits.
        XCTAssertNoThrow(try TrainerOutputFileGuard.requireNoReachableEnumeratedCheckpoints(
            naming: naming, stepLimit: 1000))
    }

    // MARK: Negative training steps

    func testNegativeTrainingStepsAreRefused() {
        let verdict = TrainerOutputFileGuard.rollingOverwriteVerdict(
            existing: TrainerModelFileIdentity(modelID: "A", trainingStep: -1),
            startModel: TrainerModelFileIdentity(modelID: "A", trainingStep: -1))
        guard case .refused(let reason) = verdict else {
            return XCTFail("a negative training step is never written by a run; got \(verdict)")
        }
        XCTAssertTrue(reason.contains("negative"), reason)
    }

    /// Steps at the extremes of `Int` are refused instead of overflowing the
    /// step difference (before the negative-step guard, these trapped).
    func testExtremeTrainingStepsAreRefusedWithoutOverflow() {
        for (existingStep, startStep) in [(Int.min, 0), (Int.max, -1), (0, Int.min)] {
            let verdict = TrainerOutputFileGuard.rollingOverwriteVerdict(
                existing: TrainerModelFileIdentity(modelID: "A", trainingStep: existingStep),
                startModel: TrainerModelFileIdentity(modelID: "A", trainingStep: startStep))
            guard case .refused = verdict else {
                return XCTFail("steps \(existingStep) / \(startStep) must be refused; got \(verdict)")
            }
        }
    }

    // MARK: FileSafety.requireStageableDestination

    func testStageableDestinationLimits() throws {
        XCTAssertEqual(FileSafety.longestStageableFileNameUTF8Bytes, longestStageableName)
        XCTAssertNoThrow(try FileSafety.requireStageableDestination(
            root.appendingPathComponent(String(repeating: "a", count: longestStageableName))))
        XCTAssertThrowsError(try FileSafety.requireStageableDestination(
            root.appendingPathComponent(String(repeating: "a", count: longestStageableName + 1)))) { error in
            guard case .nameTooLongToStage(_, let nameBytes, let limit)? = error as? FileSafetyError else {
                return XCTFail("expected nameTooLongToStage, got \(error)")
            }
            XCTAssertEqual(nameBytes, self.longestStageableName + 1)
            XCTAssertEqual(limit, self.longestStageableName)
        }
        // A path whose staging copy would not fit PATH_MAX, built from short
        // components so only the path length is at issue.
        var deep = root!
        while deep.path.utf8.count + FileSafety.temporarySiblingNameOverhead < Int(PATH_MAX) {
            deep = deep.appendingPathComponent("dddddddddd")
        }
        XCTAssertThrowsError(try FileSafety.requireStageableDestination(deep)) { error in
            guard case .pathTooLongToStage? = error as? FileSafetyError else {
                return XCTFail("expected pathTooLongToStage, got \(error)")
            }
        }
    }

    func testMultiByteNamesAreMeasuredInUTF8Bytes() {
        // "日" is three UTF-8 bytes and has no decomposed form, so a third as
        // many fit.
        let third = longestStageableName / 3
        XCTAssertNoThrow(try FileSafety.requireStageableDestination(
            root.appendingPathComponent(String(repeating: "日", count: third))))
        XCTAssertThrowsError(try FileSafety.requireStageableDestination(
            root.appendingPathComponent(String(repeating: "日", count: third + 1))))
    }
}
