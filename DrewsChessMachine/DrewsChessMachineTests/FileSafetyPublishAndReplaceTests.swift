//
//  FileSafetyPublishAndReplaceTests.swift
//  DrewsChessMachineTests
//
//  `FileSafety.publishNewFile` / `replaceRegularFile` are the write paths
//  that only ever touch the file they create or a regular file the caller
//  owns. Among their users are the training-output writers and
//  `GameCorpus.persistMetadata`, whose old form called `replaceItemAt` over
//  `corpus.json` without checking what was there (a directory of that name
//  would be swapped away) and, on failure, removed `corpus.json.tmp`
//  recursively even when that name was a pre-existing folder it never
//  created. These pin: a new file never lands on anything that exists; a
//  replacement goes only over a regular file; a folder at either name
//  survives untouched; and no staging file is left behind. They also pin the
//  staging-name shape the launch sweep recognizes, and that removing an owned
//  folder never empties a folder that took its place: the folder is moved
//  to a private name and checked there before anything is deleted.
//

import XCTest
import Darwin
@testable import DrewsChessMachine

final class FileSafetyPublishAndReplaceTests: XCTestCase {

    private var root: URL!

    override func setUpWithError() throws {
        root = FileManager.default.temporaryDirectory
            .appendingPathComponent("FileSafetyPublishAndReplaceTests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try FileManager.default.removeItem(at: root)
    }

    private func isDirectory(_ url: URL) -> Bool {
        var directory: ObjCBool = false
        return FileManager.default.fileExists(atPath: url.path, isDirectory: &directory) && directory.boolValue
    }

    /// A folder with something in it, to show it survives.
    private func makeFolder(at url: URL) throws {
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        try Data("keep".utf8).write(to: url.appendingPathComponent("keep.txt"))
    }

    private func assertFolderSurvived(_ url: URL, file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertTrue(isDirectory(url), "the folder must survive", file: file, line: line)
        XCTAssertEqual(try Data(contentsOf: url.appendingPathComponent("keep.txt")), Data("keep".utf8),
                       "the folder's contents must survive", file: file, line: line)
    }

    private func allEntryNames() throws -> [String] {
        try FileManager.default.contentsOfDirectory(atPath: root.path).sorted()
    }

    /// Put a different file at `url` the way another process would: written
    /// beside it, then renamed over it. Both files exist at once, so the
    /// newcomer cannot reuse the original's inode number.
    private func swapInAnotherFile(at url: URL, contents: String) throws {
        let sibling = url.deletingLastPathComponent().appendingPathComponent("incoming-\(UUID().uuidString)")
        try Data(contents.utf8).write(to: sibling)
        XCTAssertEqual(Darwin.rename(sibling.path, url.path), 0, String(cString: strerror(errno)))
    }

    // MARK: publishNewFile

    func testPublishCreatesANewFile() throws {
        let url = root.appendingPathComponent("new.bin")
        let identity = try FileSafety.publishNewFile(Data("hello".utf8), to: url)
        XCTAssertEqual(try Data(contentsOf: url), Data("hello".utf8))
        XCTAssertEqual(try FileSafety.existingItem(at: url), FileSafety.ExistingItem(kind: .regularFile, identity: identity))
        XCTAssertEqual(try allEntryNames(), ["new.bin"])
    }

    func testPublishNeverOverwritesAFile() throws {
        let url = root.appendingPathComponent("existing.bin")
        try Data("original".utf8).write(to: url)
        XCTAssertThrowsError(try FileSafety.publishNewFile(Data("new".utf8), to: url)) { error in
            XCTAssertEqual(error as? FileSafetyError, .alreadyExists(path: url.path, kind: .regularFile))
        }
        XCTAssertEqual(try Data(contentsOf: url), Data("original".utf8))
        XCTAssertEqual(try allEntryNames(), ["existing.bin"], "the staging file must be removed")
    }

    func testPublishNeverReplacesAFolderOrASymlink() throws {
        let folder = root.appendingPathComponent("folder.bin", isDirectory: true)
        try makeFolder(at: folder)
        XCTAssertThrowsError(try FileSafety.publishNewFile(Data("new".utf8), to: folder))
        assertFolderSurvived(folder)

        let target = root.appendingPathComponent("target.bin")
        try Data("target".utf8).write(to: target)
        let link = root.appendingPathComponent("link.bin")
        try FileManager.default.createSymbolicLink(at: link, withDestinationURL: target)
        XCTAssertThrowsError(try FileSafety.publishNewFile(Data("new".utf8), to: link))
        XCTAssertEqual(try FileSafety.existingItem(at: link)?.kind, .symbolicLink)
        XCTAssertEqual(try Data(contentsOf: target), Data("target".utf8))
        XCTAssertEqual(try allEntryNames(), ["folder.bin", "link.bin", "target.bin"])
    }

    // MARK: publishNewFileWithNumericSuffix

    func testNumericSuffixPublishUsesThePlainNameWhenFree() throws {
        let url = try FileSafety.publishNewFileWithNumericSuffix(
            Data("a".utf8), in: root, stem: "report", pathExtension: "json", maxAttempts: 3)
        XCTAssertEqual(url.lastPathComponent, "report.json")
        XCTAssertEqual(try Data(contentsOf: url), Data("a".utf8))
        XCTAssertEqual(try allEntryNames(), ["report.json"], "the staging file must be gone")
    }

    func testNumericSuffixPublishNeverOverwritesAndTakesTheNextFreeName() throws {
        let plain = root.appendingPathComponent("report.json")
        try Data("original".utf8).write(to: plain)
        let folder = root.appendingPathComponent("report-2.json", isDirectory: true)
        try makeFolder(at: folder)
        let url = try FileSafety.publishNewFileWithNumericSuffix(
            Data("new".utf8), in: root, stem: "report", pathExtension: "json", maxAttempts: 5)
        XCTAssertEqual(url.lastPathComponent, "report-3.json")
        XCTAssertEqual(try Data(contentsOf: url), Data("new".utf8))
        XCTAssertEqual(try Data(contentsOf: plain), Data("original".utf8))
        assertFolderSurvived(folder)
        XCTAssertEqual(try allEntryNames(), ["report-2.json", "report-3.json", "report.json"])
    }

    func testNumericSuffixPublishThrowsWhenEveryNameIsTakenAndLeavesNoStaging() throws {
        try Data("1".utf8).write(to: root.appendingPathComponent("report.json"))
        try Data("2".utf8).write(to: root.appendingPathComponent("report-2.json"))
        XCTAssertThrowsError(try FileSafety.publishNewFileWithNumericSuffix(
            Data("new".utf8), in: root, stem: "report", pathExtension: "json", maxAttempts: 2)) { error in
            XCTAssertEqual(error as? FileSafetyError,
                           .noFreeNumericSuffix(directory: self.root.path, stem: "report", pathExtension: "json", maxAttempts: 2))
        }
        XCTAssertEqual(try allEntryNames(), ["report-2.json", "report.json"], "the staging file must be removed")
        XCTAssertEqual(try Data(contentsOf: root.appendingPathComponent("report.json")), Data("1".utf8))
    }

    // MARK: replaceRegularFile

    func testReplaceGoesOverARegularFile() throws {
        let url = root.appendingPathComponent("file.bin")
        try Data("old".utf8).write(to: url)
        try FileSafety.replaceRegularFile(Data("new".utf8), at: url, expectedIdentity: nil)
        XCTAssertEqual(try Data(contentsOf: url), Data("new".utf8))
        XCTAssertEqual(try allEntryNames(), ["file.bin"])
    }

    func testReplaceNeverGoesOverAFolder() throws {
        let folder = root.appendingPathComponent("folder.bin", isDirectory: true)
        try makeFolder(at: folder)
        XCTAssertThrowsError(try FileSafety.replaceRegularFile(Data("new".utf8), at: folder, expectedIdentity: nil)) { error in
            XCTAssertEqual(error as? FileSafetyError, .notARegularFile(path: folder.path, kind: .directory))
        }
        assertFolderSurvived(folder)
        XCTAssertEqual(try allEntryNames(), ["folder.bin"])
    }

    func testReplaceWithAnExpectedIdentityRefusesADifferentFile() throws {
        let url = root.appendingPathComponent("file.bin")
        let mine = try FileSafety.publishNewFile(Data("mine".utf8), to: url)
        try swapInAnotherFile(at: url, contents: "theirs")
        XCTAssertThrowsError(try FileSafety.replaceRegularFile(Data("new".utf8), at: url, expectedIdentity: mine)) { error in
            XCTAssertEqual(error as? FileSafetyError, .fileChangedSinceWritten(path: url.path))
        }
        XCTAssertEqual(try Data(contentsOf: url), Data("theirs".utf8))
    }

    func testReplaceRecreatesAnOwnedFileThatWasDeleted() throws {
        let url = root.appendingPathComponent("file.bin")
        let mine = try FileSafety.publishNewFile(Data("mine".utf8), to: url)
        try FileManager.default.removeItem(at: url)
        try FileSafety.replaceRegularFile(Data("again".utf8), at: url, expectedIdentity: mine)
        XCTAssertEqual(try Data(contentsOf: url), Data("again".utf8))
        XCTAssertEqual(try allEntryNames(), ["file.bin"])
    }

    // MARK: corpus.json (GameCorpus.persistMetadata)

    private func sampleMetadata() -> CorpusMetadata {
        CorpusMetadata(formatVersion: CorpusMetadata.currentFormatVersion,
                       corpusID: "20260101-000000-TESTAA",
                       name: "t",
                       comment: nil,
                       state: "recording",
                       createdAtUnix: 1,
                       sources: [])
    }

    func testPersistMetadataRefusesAFolderNamedCorpusJSON() throws {
        let folder = root.appendingPathComponent(GameCorpus.metadataFilename, isDirectory: true)
        try makeFolder(at: folder)
        XCTAssertThrowsError(try GameCorpus.persistMetadata(sampleMetadata(), to: root))
        assertFolderSurvived(folder)
        XCTAssertEqual(try allEntryNames(), [GameCorpus.metadataFilename], "no staging file may be left behind")
    }

    func testPersistMetadataLeavesAPreexistingTmpFolderAlone() throws {
        let tmpFolder = root.appendingPathComponent("\(GameCorpus.metadataFilename).tmp", isDirectory: true)
        try makeFolder(at: tmpFolder)

        // Success path: the write stages under its own unique name.
        try GameCorpus.persistMetadata(sampleMetadata(), to: root)
        assertFolderSurvived(tmpFolder)
        XCTAssertEqual(try GameCorpus.loadMetadata(directory: root), sampleMetadata())

        // Failure path (corpus.json replaced by a folder): the cleanup must not
        // touch a temp-named folder it did not create.
        let metadataURL = root.appendingPathComponent(GameCorpus.metadataFilename)
        try FileManager.default.removeItem(at: metadataURL)
        try makeFolder(at: metadataURL)
        XCTAssertThrowsError(try GameCorpus.persistMetadata(sampleMetadata(), to: root))
        assertFolderSurvived(tmpFolder)
        assertFolderSurvived(metadataURL)
        XCTAssertEqual(try allEntryNames(), [GameCorpus.metadataFilename, "\(GameCorpus.metadataFilename).tmp"])
    }

    func testPersistMetadataReplacesARegularCorpusJSON() throws {
        try GameCorpus.persistMetadata(sampleMetadata(), to: root)
        var updated = sampleMetadata()
        updated.state = "sealed"
        try GameCorpus.persistMetadata(updated, to: root)
        XCTAssertEqual(try GameCorpus.loadMetadata(directory: root), updated)
        XCTAssertEqual(try allEntryNames(), [GameCorpus.metadataFilename])
    }

    // MARK: Staging names

    func testTemporarySiblingNamesAreRecognizedWithTheirDestination() {
        for name in ["a", "corpus.json", "x.y.z", "café.json", ".hidden", "run-replay-latest.safetensors"] {
            let destination = root.appendingPathComponent(name)
            let staging = FileSafety.temporarySibling(of: destination)
            XCTAssertEqual(staging.deletingLastPathComponent().standardizedFileURL.path, root.standardizedFileURL.path)
            XCTAssertEqual(FileSafety.destinationName(ofTemporarySiblingName: staging.lastPathComponent),
                           destination.lastPathComponent)
            XCTAssertEqual(staging.lastPathComponent.utf8.count - destination.lastPathComponent.utf8.count,
                           FileSafety.temporarySiblingNameOverhead)
        }
    }

    func testOnlyExactTemporarySiblingNamesAreRecognized() {
        let uuid = UUID().uuidString
        for name in [
            "corpus.json.tmp",
            ".corpus.json.tmp",
            ".x.not-a-uuid.tmp",
            ".x.\(uuid.lowercased()).tmp",
            "..\(uuid).tmp",
            "x.\(uuid).tmp",
            ".x.\(uuid).temp",
            ".x.\(uuid)",
            ".x\(uuid).tmp",
            ".tmp",
            "",
        ] {
            XCTAssertNil(FileSafety.destinationName(ofTemporarySiblingName: name), name.debugDescription)
        }
    }

    // MARK: removeOwnedItem — directories

    func testRemovingAnOwnedFolderLeavesNothingBehind() throws {
        let folder = root.appendingPathComponent("owned", isDirectory: true)
        try makeFolder(at: folder)
        let identity = try XCTUnwrap(try FileSafety.existingItem(at: folder)).identity
        XCTAssertEqual(try FileSafety.removeOwnedItem(at: folder, identity: identity), .removed)
        XCTAssertEqual(try allEntryNames(), [], "not even the hidden name the folder is moved to before deletion")
        XCTAssertEqual(try FileSafety.removeOwnedItem(at: folder, identity: identity), .alreadyGone)
    }

    func testRemovingAFolderThatWasReplacedIsRefusedAndTheReplacementSurvivesWhole() throws {
        let folder = root.appendingPathComponent("owned", isDirectory: true)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: false)
        let owned = try XCTUnwrap(try FileSafety.existingItem(at: folder)).identity
        // Another folder takes the path; the owned one is parked, not
        // deleted, so the newcomer cannot reuse its inode number.
        let replacement = root.appendingPathComponent("incoming", isDirectory: true)
        try makeFolder(at: replacement)
        XCTAssertEqual(Darwin.rename(folder.path, root.appendingPathComponent("parked").path), 0)
        XCTAssertEqual(Darwin.rename(replacement.path, folder.path), 0)

        XCTAssertThrowsError(try FileSafety.removeOwnedItem(at: folder, identity: owned)) { error in
            XCTAssertEqual(error as? FileSafetyError, .fileChangedSinceWritten(path: folder.path))
        }
        assertFolderSurvived(folder)
        XCTAssertEqual(try allEntryNames(), ["owned", "parked"])
    }

    /// The case `removeOwnedItem` reaches only through a race: the folder
    /// moved aside turns out not to be the owned one. It must come back to
    /// its path whole.
    func testAMovedAsideFolderThatIsNotTheOwnedOneIsMovedBackUntouched() throws {
        let original = root.appendingPathComponent("owned", isDirectory: true)
        let movedAside = FileSafety.temporarySibling(of: original)
        try makeFolder(at: movedAside)
        let otherFile = root.appendingPathComponent("other")
        let otherIdentity = try FileSafety.publishNewFile(Data("other".utf8), to: otherFile)

        XCTAssertThrowsError(try FileSafety.deleteMovedAsideDirectory(
            at: movedAside, movedFrom: original, expectedIdentity: otherIdentity)) { error in
            XCTAssertEqual(error as? FileSafetyError, .fileChangedSinceWritten(path: original.path))
        }
        assertFolderSurvived(original)
        XCTAssertEqual(try allEntryNames(), ["other", "owned"])
    }

    func testAMovedAsideFolderWhoseOldPathWasRetakenIsLeftWholeWhereItIs() throws {
        let original = root.appendingPathComponent("owned", isDirectory: true)
        try Data("newcomer".utf8).write(to: original)
        let movedAside = FileSafety.temporarySibling(of: original)
        try makeFolder(at: movedAside)
        let otherIdentity = try FileSafety.publishNewFile(Data("other".utf8), to: root.appendingPathComponent("other"))

        XCTAssertThrowsError(try FileSafety.deleteMovedAsideDirectory(
            at: movedAside, movedFrom: original, expectedIdentity: otherIdentity))
        assertFolderSurvived(movedAside)
        XCTAssertEqual(try Data(contentsOf: original), Data("newcomer".utf8))
    }

    func testAMovedAsideFolderThatIsTheOwnedOneIsDeleted() throws {
        let original = root.appendingPathComponent("owned", isDirectory: true)
        let movedAside = FileSafety.temporarySibling(of: original)
        try makeFolder(at: movedAside)
        let identity = try XCTUnwrap(try FileSafety.existingItem(at: movedAside)).identity

        try FileSafety.deleteMovedAsideDirectory(at: movedAside, movedFrom: original, expectedIdentity: identity)
        XCTAssertEqual(try allEntryNames(), [])
    }
}
