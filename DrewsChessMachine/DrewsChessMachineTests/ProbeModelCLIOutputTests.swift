//
//  ProbeModelCLIOutputTests.swift
//  DrewsChessMachineTests
//
//  `--probe-model` writes two optional files: `--probe-out` (one summary line
//  per checkpoint and battery) and `--probe-positions-out` (one line per
//  position). Both are opened before any checkpoint is read, so a bad pair of
//  paths must be refused before anything is created or emptied:
//
//  - An output path that names a probed checkpoint — directly, through `..`,
//    through a hard link, or differing only in case on a case-insensitive
//    volume — would empty that checkpoint before it is loaded, even with
//    `--probe-out-overwrite`. It is always refused.
//  - The two outputs naming one file (a case variant, a hard link, or one
//    path through a symbolic link to the other's folder) would interleave two
//    writers in one file. Always refused.
//  - A refusal discovered after the first output was opened must not leave
//    that output behind (a freshly created empty file blocks the identical
//    re-run) or emptied (an existing summary file truncated by a run that
//    then refused).
//
//  The tests assert only observable outcomes — the call throws, the files'
//  bytes are unchanged, the folder holds nothing new — so the same assertions
//  hold whatever error type the refusal uses.
//

import XCTest
@testable import DrewsChessMachine

final class ProbeModelCLIOutputTests: XCTestCase {

    private var root: URL!
    private let checkpointBytes = Data("checkpoint-bytes".utf8)
    private let existingBytes = Data("an earlier run's results\n".utf8)

    override func setUpWithError() throws {
        root = FileManager.default.temporaryDirectory
            .appendingPathComponent("ProbeModelCLIOutputTests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try FileManager.default.removeItem(at: root)
    }

    private func makeCheckpoint(named name: String = "model.safetensors") throws -> URL {
        let url = root.appendingPathComponent(name)
        try checkpointBytes.write(to: url)
        return url
    }

    /// Every entry directly in `directory`, sorted.
    private func entries(_ directory: URL) throws -> [String] {
        try FileManager.default.contentsOfDirectory(atPath: directory.path).sorted()
    }

    /// Whether the folder holding the tests' files is case-insensitive.
    private func volumeIsCaseInsensitive() throws -> Bool {
        let probe = root.appendingPathComponent("case-probe")
        try Data().write(to: probe)
        defer {
            do { try FileManager.default.removeItem(at: probe) } catch { XCTFail("removing \(probe.path): \(error)") }
        }
        return FileManager.default.fileExists(atPath: root.appendingPathComponent("CASE-PROBE").path)
    }

    private func open(summary: URL?, positions: URL?, targets: [URL], overwrite: Bool) throws {
        let files = try ProbeModelCLI.openOutputs(summaryPath: summary?.path,
                                                  positionsPath: positions?.path,
                                                  probeTargets: targets,
                                                  replaceExisting: overwrite)
        try files.summaryHandle?.close()
        try files.positionsHandle?.close()
    }

    // MARK: - An output is never a probed checkpoint

    func testSummaryNamingTheProbedCheckpointIsRefusedEvenWithOverwrite() throws {
        let checkpoint = try makeCheckpoint()
        XCTAssertThrowsError(try open(summary: checkpoint, positions: nil, targets: [checkpoint], overwrite: true))
        XCTAssertEqual(try Data(contentsOf: checkpoint), checkpointBytes, "the checkpoint must not be emptied")
    }

    func testPositionsNamingTheProbedCheckpointIsRefusedEvenWithOverwrite() throws {
        let checkpoint = try makeCheckpoint()
        XCTAssertThrowsError(try open(summary: nil, positions: checkpoint, targets: [checkpoint], overwrite: true))
        XCTAssertEqual(try Data(contentsOf: checkpoint), checkpointBytes, "the checkpoint must not be emptied")
    }

    func testOutputReachingTheCheckpointThroughDotDotIsRefused() throws {
        let checkpoint = try makeCheckpoint()
        try FileManager.default.createDirectory(at: root.appendingPathComponent("sub"), withIntermediateDirectories: false)
        let viaDotDot = URL(fileURLWithPath: root.path + "/sub/../model.safetensors")
        XCTAssertThrowsError(try open(summary: viaDotDot, positions: nil, targets: [checkpoint], overwrite: true))
        XCTAssertEqual(try Data(contentsOf: checkpoint), checkpointBytes)
    }

    func testOutputThatIsAHardLinkToTheCheckpointIsRefused() throws {
        let checkpoint = try makeCheckpoint()
        let hardLink = root.appendingPathComponent("results.jsonl")
        try FileManager.default.linkItem(at: checkpoint, to: hardLink)
        XCTAssertThrowsError(try open(summary: hardLink, positions: nil, targets: [checkpoint], overwrite: true))
        XCTAssertEqual(try Data(contentsOf: checkpoint), checkpointBytes)
    }

    func testOutputDifferingFromTheCheckpointOnlyInCaseIsRefused() throws {
        guard try volumeIsCaseInsensitive() else {
            throw XCTSkip("the temporary folder's volume is case-sensitive")
        }
        let checkpoint = try makeCheckpoint()
        let caseVariant = root.appendingPathComponent("MODEL.safetensors")
        XCTAssertThrowsError(try open(summary: caseVariant, positions: nil, targets: [checkpoint], overwrite: true))
        XCTAssertEqual(try Data(contentsOf: checkpoint), checkpointBytes)
    }

    // MARK: - The two outputs are never one file

    func testOutputsDifferingOnlyInCaseAreRefusedAndNothingIsLeftBehind() throws {
        guard try volumeIsCaseInsensitive() else {
            throw XCTSkip("the temporary folder's volume is case-sensitive")
        }
        let checkpoint = try makeCheckpoint()
        let summary = root.appendingPathComponent("R.jsonl")
        let positions = root.appendingPathComponent("r.jsonl")
        for overwrite in [false, true] {
            XCTAssertThrowsError(try open(summary: summary, positions: positions, targets: [checkpoint], overwrite: overwrite),
                                 "overwrite \(overwrite)")
            XCTAssertEqual(try entries(root), ["model.safetensors"], "overwrite \(overwrite): nothing may be created")
        }
    }

    func testHardLinkedExistingOutputsAreRefusedWithOverwriteAndKeepTheirContents() throws {
        let checkpoint = try makeCheckpoint()
        let summary = root.appendingPathComponent("summary.jsonl")
        try existingBytes.write(to: summary)
        let positions = root.appendingPathComponent("positions.jsonl")
        try FileManager.default.linkItem(at: summary, to: positions)
        XCTAssertThrowsError(try open(summary: summary, positions: positions, targets: [checkpoint], overwrite: true))
        XCTAssertEqual(try Data(contentsOf: summary), existingBytes)
    }

    func testOutputsReachingOneNewFileThroughASymlinkedFolderAreRefusedAndNothingIsLeftBehind() throws {
        let checkpoint = try makeCheckpoint()
        let real = root.appendingPathComponent("real", isDirectory: true)
        try FileManager.default.createDirectory(at: real, withIntermediateDirectories: false)
        let link = root.appendingPathComponent("link", isDirectory: true)
        try FileManager.default.createSymbolicLink(at: link, withDestinationURL: real)
        let summary = real.appendingPathComponent("out.jsonl")
        let positions = URL(fileURLWithPath: link.path + "/out.jsonl")
        for overwrite in [false, true] {
            XCTAssertThrowsError(try open(summary: summary, positions: positions, targets: [checkpoint], overwrite: overwrite),
                                 "overwrite \(overwrite)")
            XCTAssertEqual(try entries(real), [], "overwrite \(overwrite): nothing may be created")
        }
    }

    // MARK: - A refusal leaves nothing behind and empties nothing

    func testExistingPositionsFileWithoutOverwriteLeavesNoSummaryFileBehind() throws {
        let checkpoint = try makeCheckpoint()
        let summary = root.appendingPathComponent("summary.jsonl")
        let positions = root.appendingPathComponent("positions.jsonl")
        try existingBytes.write(to: positions)
        XCTAssertThrowsError(try open(summary: summary, positions: positions, targets: [checkpoint], overwrite: false))
        XCTAssertFalse(FileManager.default.fileExists(atPath: summary.path),
                       "the summary file must not be created when the run is refused")
        XCTAssertEqual(try Data(contentsOf: positions), existingBytes)
    }

    func testPositionsPathThatIsAFolderLeavesAnExistingSummaryIntactEvenWithOverwrite() throws {
        let checkpoint = try makeCheckpoint()
        let summary = root.appendingPathComponent("summary.jsonl")
        try existingBytes.write(to: summary)
        let positions = root.appendingPathComponent("positions.jsonl", isDirectory: true)
        try FileManager.default.createDirectory(at: positions, withIntermediateDirectories: false)
        XCTAssertThrowsError(try open(summary: summary, positions: positions, targets: [checkpoint], overwrite: true))
        XCTAssertEqual(try Data(contentsOf: summary), existingBytes, "a refused run must not empty the summary file")
    }

    // MARK: - Allowed cases still work

    func testDistinctNewOutputsAreCreated() throws {
        let checkpoint = try makeCheckpoint()
        let summary = root.appendingPathComponent("summary.jsonl")
        let positions = root.appendingPathComponent("positions.jsonl")
        try open(summary: summary, positions: positions, targets: [checkpoint], overwrite: false)
        XCTAssertTrue(FileManager.default.fileExists(atPath: summary.path))
        XCTAssertTrue(FileManager.default.fileExists(atPath: positions.path))
    }

    func testExistingDistinctOutputsAreEmptiedWithOverwrite() throws {
        let checkpoint = try makeCheckpoint()
        let summary = root.appendingPathComponent("summary.jsonl")
        let positions = root.appendingPathComponent("positions.jsonl")
        try existingBytes.write(to: summary)
        try existingBytes.write(to: positions)
        try open(summary: summary, positions: positions, targets: [checkpoint], overwrite: true)
        XCTAssertEqual(try Data(contentsOf: summary), Data())
        XCTAssertEqual(try Data(contentsOf: positions), Data())
        XCTAssertEqual(try Data(contentsOf: checkpoint), checkpointBytes)
    }

    // MARK: - Writing

    func testAppendLineToAReadOnlyHandleThrows() throws {
        let url = root.appendingPathComponent("read-only.jsonl")
        try existingBytes.write(to: url)
        let handle = try FileHandle(forReadingFrom: url)
        defer {
            do { try handle.close() } catch { XCTFail("closing: \(error)") }
        }
        XCTAssertThrowsError(try ProbeModelCLI.appendLine(Data("{}".utf8), to: handle))
    }
}
