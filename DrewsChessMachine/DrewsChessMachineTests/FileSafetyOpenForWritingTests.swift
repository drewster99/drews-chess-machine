//
//  FileSafetyOpenForWritingTests.swift
//  DrewsChessMachineTests
//
//  `--probe-out` and `--arch-sweep-out` opened their destination with
//  `FileManager.createFile` (which empties an existing file and follows a
//  symbolic link to empty its target), ignored that call's result, and let a
//  nil `FileHandle` drop every line of output. The session logger used the
//  same pair, so two launches in one second shared `dcm_log_<stamp>.txt` and
//  the first process went on writing into an unlinked file. These pin the
//  replacement: exclusive creation by default, truncation only of a regular
//  file and only on request, numeric suffixes for the log name, and a thrown
//  error — never a silent nil — for everything else. All of it is
//  `FileSafety.openForWriting` / `createNewFileWithNumericSuffix`. They also
//  pin what `--new-model`, `--derive-model` and the preset store rely on:
//  `FileSafety.publishNewFile` refuses an existing file, naming it, and leaves
//  it alone.
//

import Darwin
import XCTest
@testable import DrewsChessMachine

final class FileSafetyOpenForWritingTests: XCTestCase {

    private var root: URL!

    override func setUpWithError() throws {
        root = FileManager.default.temporaryDirectory
            .appendingPathComponent("FileSafetyOpenForWritingTests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try FileManager.default.removeItem(at: root)
    }

    private func write(_ text: String, to url: URL) throws {
        try Data(text.utf8).write(to: url)
    }

    private func contents(of url: URL) throws -> String {
        String(decoding: try Data(contentsOf: url), as: UTF8.self)
    }

    private func writeAndClose(_ text: String, to handle: FileHandle) throws {
        try handle.write(contentsOf: Data(text.utf8))
        try handle.close()
    }

    private func isDirectory(_ url: URL) -> Bool {
        var directory: ObjCBool = false
        return FileManager.default.fileExists(atPath: url.path, isDirectory: &directory) && directory.boolValue
    }

    /// A folder with something in it, to show it survives.
    private func makeFolder(_ name: String) throws -> URL {
        let folder = root.appendingPathComponent(name, isDirectory: true)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        try write("keep", to: folder.appendingPathComponent("keep.txt"))
        return folder
    }

    private func assertNotARegularFile(_ error: Error, path: URL, file: StaticString = #filePath, line: UInt = #line) {
        guard case FileSafetyError.notARegularFile(let reportedPath, _) = error else {
            XCTFail("expected notARegularFile, got \(error)", file: file, line: line)
            return
        }
        XCTAssertEqual(reportedPath, path.path, file: file, line: line)
    }

    /// Under `.refuse` the call is an exclusive create, so anything in the
    /// way — regular or not — is reported as already existing, with its kind.
    private func assertRefusedAsExisting(_ error: Error, path: URL, kind: FileSafety.ItemKind,
                                         file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertEqual(error as? FileSafetyError, .alreadyExists(path: path.path, kind: kind), file: file, line: line)
    }

    // MARK: - Refuse (the default for --probe-out / --arch-sweep-out)

    func testRefuseCreatesANewFile() throws {
        let url = root.appendingPathComponent("out.jsonl")
        let handle = try FileSafety.openForWriting(at: url, existingRegularFile: .refuse)
        try writeAndClose("line\n", to: handle)
        XCTAssertEqual(try contents(of: url), "line\n")
    }

    func testRefuseLeavesAnExistingRegularFileUntouched() throws {
        let url = root.appendingPathComponent("out.jsonl")
        try write("previous results\n", to: url)
        XCTAssertThrowsError(try FileSafety.openForWriting(at: url, existingRegularFile: .refuse)) { error in
            XCTAssertEqual(error as? FileSafetyError, .alreadyExists(path: url.path, kind: .regularFile))
        }
        XCTAssertEqual(try contents(of: url), "previous results\n", "an existing output must never be emptied without the overwrite flag")
    }

    func testRefuseRejectsAFolder() throws {
        let folder = try makeFolder("out.jsonl")
        XCTAssertThrowsError(try FileSafety.openForWriting(at: folder, existingRegularFile: .refuse)) { error in
            assertRefusedAsExisting(error, path: folder, kind: .directory)
        }
        XCTAssertTrue(isDirectory(folder))
        XCTAssertEqual(try contents(of: folder.appendingPathComponent("keep.txt")), "keep")
    }

    func testRefuseRejectsASymbolicLinkAndLeavesItsTargetAlone() throws {
        let target = root.appendingPathComponent("target.txt")
        try write("target contents", to: target)
        let link = root.appendingPathComponent("out.jsonl")
        try FileManager.default.createSymbolicLink(at: link, withDestinationURL: target)
        XCTAssertThrowsError(try FileSafety.openForWriting(at: link, existingRegularFile: .refuse)) { error in
            assertRefusedAsExisting(error, path: link, kind: .symbolicLink)
        }
        XCTAssertEqual(try contents(of: target), "target contents")
    }

    func testRefuseRejectsADanglingSymbolicLinkWithoutCreatingItsTarget() throws {
        let target = root.appendingPathComponent("never-created.txt")
        let link = root.appendingPathComponent("out.jsonl")
        try FileManager.default.createSymbolicLink(at: link, withDestinationURL: target)
        XCTAssertThrowsError(try FileSafety.openForWriting(at: link, existingRegularFile: .refuse)) { error in
            assertRefusedAsExisting(error, path: link, kind: .symbolicLink)
        }
        XCTAssertFalse(FileManager.default.fileExists(atPath: target.path), "a link must never be followed to create its target")
    }

    func testMissingParentFolderIsAThrownSystemErrorNotANilHandle() throws {
        let url = root.appendingPathComponent("missing", isDirectory: true).appendingPathComponent("out.jsonl")
        for policy in [FileSafety.ExistingRegularFilePolicy.refuse, .truncate] {
            XCTAssertThrowsError(try FileSafety.openForWriting(at: url, existingRegularFile: policy)) { error in
                XCTAssertEqual(error as? FileSafetyError,
                               .systemCallFailed(path: url.path, call: "open", errnoValue: ENOENT), "policy \(policy)")
            }
        }
    }

    // MARK: - Truncate (--probe-out-overwrite / --arch-sweep-out-overwrite)

    func testTruncateEmptiesAnExistingRegularFileAndWritesFromTheStart() throws {
        let url = root.appendingPathComponent("out.jsonl")
        try write("a much longer previous run's output\n", to: url)
        let handle = try FileSafety.openForWriting(at: url, existingRegularFile: .truncate)
        try writeAndClose("new\n", to: handle)
        XCTAssertEqual(try contents(of: url), "new\n", "no tail of the previous contents may survive")
    }

    func testTruncateCreatesAMissingFile() throws {
        let url = root.appendingPathComponent("out.jsonl")
        let handle = try FileSafety.openForWriting(at: url, existingRegularFile: .truncate)
        try writeAndClose("new\n", to: handle)
        XCTAssertEqual(try contents(of: url), "new\n")
    }

    func testTruncateStillRejectsAFolder() throws {
        let folder = try makeFolder("out.jsonl")
        XCTAssertThrowsError(try FileSafety.openForWriting(at: folder, existingRegularFile: .truncate)) { error in
            assertNotARegularFile(error, path: folder)
        }
        XCTAssertTrue(isDirectory(folder))
        XCTAssertEqual(try contents(of: folder.appendingPathComponent("keep.txt")), "keep")
    }

    func testTruncateRejectsASymbolicLinkAndLeavesItsTargetAlone() throws {
        let target = root.appendingPathComponent("target.txt")
        try write("target contents", to: target)
        let link = root.appendingPathComponent("out.jsonl")
        try FileManager.default.createSymbolicLink(at: link, withDestinationURL: target)
        XCTAssertThrowsError(try FileSafety.openForWriting(at: link, existingRegularFile: .truncate)) { error in
            assertNotARegularFile(error, path: link)
        }
        XCTAssertEqual(try contents(of: target), "target contents", "truncation must never reach through a link")
    }

    /// A FIFO with no reader would block a plain `open` for writing forever;
    /// it must be refused promptly instead.
    func testTruncateRejectsAFIFOWithoutBlocking() throws {
        let fifo = root.appendingPathComponent("out.jsonl")
        XCTAssertEqual(mkfifo(fifo.path, 0o644), 0, "mkfifo failed: errno \(errno)")
        XCTAssertThrowsError(try FileSafety.openForWriting(at: fifo, existingRegularFile: .truncate)) { error in
            assertNotARegularFile(error, path: fifo)
        }
    }

    // MARK: - Numeric suffix (session log names)

    func testNumericSuffixGivesEachSameSecondLaunchItsOwnFile() throws {
        let stem = "dcm_log_20261001-120000"
        let first = try FileSafety.createNewFileWithNumericSuffix(in: root, stem: stem, pathExtension: "txt", maxAttempts: 10)
        try first.handle.write(contentsOf: Data("first process\n".utf8))
        let second = try FileSafety.createNewFileWithNumericSuffix(in: root, stem: stem, pathExtension: "txt", maxAttempts: 10)
        let third = try FileSafety.createNewFileWithNumericSuffix(in: root, stem: stem, pathExtension: "txt", maxAttempts: 10)
        try writeAndClose("second process\n", to: second.handle)
        try writeAndClose("third process\n", to: third.handle)
        try writeAndClose("first process, later line\n", to: first.handle)

        XCTAssertEqual(first.url.lastPathComponent, "dcm_log_20261001-120000.txt")
        XCTAssertEqual(second.url.lastPathComponent, "dcm_log_20261001-120000-2.txt")
        XCTAssertEqual(third.url.lastPathComponent, "dcm_log_20261001-120000-3.txt")
        XCTAssertEqual(try contents(of: first.url), "first process\nfirst process, later line\n",
                       "a later launch must never empty or replace the first launch's log")
        XCTAssertEqual(try contents(of: second.url), "second process\n")
        XCTAssertEqual(try contents(of: third.url), "third process\n")
    }

    /// Log tooling globs `dcm_log_*.txt` and reads the launch stamp from the
    /// characters right after `dcm_log_`; a suffixed name must keep both.
    func testSuffixedLogNameKeepsTheGlobShapeAndStampPosition() throws {
        let stem = "dcm_log_20261001-120000"
        _ = try FileSafety.createNewFileWithNumericSuffix(in: root, stem: stem, pathExtension: "txt", maxAttempts: 10)
        let suffixed = try FileSafety.createNewFileWithNumericSuffix(in: root, stem: stem, pathExtension: "txt", maxAttempts: 10)
        let name = suffixed.url.lastPathComponent
        XCTAssertTrue(name.hasPrefix("dcm_log_"))
        XCTAssertTrue(name.hasSuffix(".txt"))
        XCTAssertEqual(String(name.dropFirst("dcm_log_".count).prefix("20261001-120000".count)), "20261001-120000")
    }

    func testNumericSuffixSkipsANameTakenByAFolder() throws {
        let folder = try makeFolder("dcm_log_20261001-120000.txt")
        let created = try FileSafety.createNewFileWithNumericSuffix(
            in: root, stem: "dcm_log_20261001-120000", pathExtension: "txt", maxAttempts: 10)
        XCTAssertEqual(created.url.lastPathComponent, "dcm_log_20261001-120000-2.txt")
        XCTAssertTrue(isDirectory(folder))
        XCTAssertEqual(try contents(of: folder.appendingPathComponent("keep.txt")), "keep")
    }

    func testNumericSuffixThrowsWhenEveryCandidateIsTaken() throws {
        let stem = "dcm_log_20261001-120000"
        for _ in 0..<3 {
            _ = try FileSafety.createNewFileWithNumericSuffix(in: root, stem: stem, pathExtension: "txt", maxAttempts: 3)
        }
        XCTAssertThrowsError(try FileSafety.createNewFileWithNumericSuffix(in: root, stem: stem, pathExtension: "txt", maxAttempts: 3)) { error in
            XCTAssertEqual(error as? FileSafetyError,
                           .noFreeNumericSuffix(directory: root.path, stem: stem, pathExtension: "txt", maxAttempts: 3))
        }
    }

    // MARK: - Exclusive publish (--new-model, --derive-model, preset save)

    func testPublishNewFileFailsWithAlreadyExistsAndKeepsTheFile() throws {
        let url = root.appendingPathComponent("model.safetensors")
        try write("precious starting net", to: url)
        XCTAssertThrowsError(try FileSafety.publishNewFile(Data("replacement".utf8), to: url)) { error in
            XCTAssertEqual(error as? FileSafetyError, .alreadyExists(path: url.path, kind: .regularFile), "got \(error)")
        }
        XCTAssertEqual(try contents(of: url), "precious starting net")
    }

    func testPublishNewFileRefusesASymbolicLink() throws {
        let target = root.appendingPathComponent("target.txt")
        try write("target contents", to: target)
        let link = root.appendingPathComponent("model.safetensors")
        try FileManager.default.createSymbolicLink(at: link, withDestinationURL: target)
        XCTAssertThrowsError(try FileSafety.publishNewFile(Data("replacement".utf8), to: link))
        XCTAssertEqual(try contents(of: target), "target contents")
    }
}
