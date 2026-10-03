//
//  FileSafetyExclusiveLockTests.swift
//  DrewsChessMachineTests
//
//  The two lock primitives corpus shards rely on to tell a live writer from a
//  crash leftover. `createNewFileHoldingExclusiveLock` holds the file's lock
//  for as long as its handle is open; `openExistingRegularFileWithExclusiveLock`
//  takes it only when it is free, never waits, and refuses anything that is
//  not a regular file before touching it.
//

import XCTest
import Darwin
@testable import DrewsChessMachine

final class FileSafetyExclusiveLockTests: XCTestCase {

    private var root: URL!

    override func setUpWithError() throws {
        root = FileManager.default.temporaryDirectory
            .appendingPathComponent("FileSafetyExclusiveLockTests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try FileManager.default.removeItem(at: root)
    }

    private func isHeld(_ attempt: FileSafety.ExistingFileLockAttempt) -> Bool {
        if case .heldByAnotherOpenFile = attempt { return true }
        return false
    }

    func testANewLockedFileIsHeldUntilItsHandleCloses() throws {
        let url = root.appendingPathComponent("shard.open")
        let created = try FileSafety.createNewFileHoldingExclusiveLock(at: url)
        XCTAssertTrue(isHeld(try FileSafety.openExistingRegularFileWithExclusiveLock(at: url)))
        try created.handle.close()
        guard case let .locked(handle, identity) = try FileSafety.openExistingRegularFileWithExclusiveLock(at: url) else {
            return XCTFail("the lock must be free once the creating handle is closed")
        }
        XCTAssertEqual(identity, created.identity)
        XCTAssertTrue(isHeld(try FileSafety.openExistingRegularFileWithExclusiveLock(at: url)),
                      "the recovering handle now holds the lock in turn")
        try handle.close()
    }

    func testAMissingFileIsReportedGone() throws {
        guard case .gone = try FileSafety.openExistingRegularFileWithExclusiveLock(at: root.appendingPathComponent("none")) else {
            return XCTFail("nothing at the path must be .gone")
        }
    }

    func testNonRegularItemsAreRefusedUntouched() throws {
        let target = root.appendingPathComponent("target")
        try Data("target".utf8).write(to: target)
        let link = root.appendingPathComponent("link.open")
        try FileManager.default.createSymbolicLink(at: link, withDestinationURL: target)
        let folder = root.appendingPathComponent("folder.open", isDirectory: true)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: false)
        let fifo = root.appendingPathComponent("fifo.open")
        XCTAssertEqual(mkfifo(fifo.path, 0o644), 0)

        for (url, kind) in [(link, FileSafety.ItemKind.symbolicLink), (folder, .directory), (fifo, .fifo)] {
            XCTAssertThrowsError(try FileSafety.openExistingRegularFileWithExclusiveLock(at: url)) { error in
                XCTAssertEqual(error as? FileSafetyError, .notARegularFile(path: url.path, kind: kind))
            }
        }
        XCTAssertEqual(try Data(contentsOf: target), Data("target".utf8), "a link's target is never opened for writing")
    }
}
