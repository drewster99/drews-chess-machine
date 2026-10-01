//
//  ReplayBufferWriteDestinationTests.swift
//  DrewsChessMachineTests
//
//  `ReplayBuffer.write(to:)` used to remove whatever sat at the destination
//  before writing — a folder included, recursively. It then truncated an
//  existing regular file in place, which a crash mid-write would leave torn.
//  It now only ever creates a new file: anything already at the destination
//  — a previous buffer file included — is refused and left untouched. (The
//  one production caller writes into a save's freshly created staging
//  folder, where nothing can be at the name.)
//

import XCTest
@testable import DrewsChessMachine

final class ReplayBufferWriteDestinationTests: XCTestCase {

    private var root: URL!

    override func setUpWithError() throws {
        root = FileManager.default.temporaryDirectory
            .appendingPathComponent("ReplayBufferWriteDestinationTests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try FileManager.default.removeItem(at: root)
    }

    func testAFolderAtTheDestinationIsRefusedAndKept() throws {
        let folder = root.appendingPathComponent("replay_buffer.bin", isDirectory: true)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        let keep = folder.appendingPathComponent("keep.txt")
        try Data("keep".utf8).write(to: keep)

        XCTAssertThrowsError(try ReplayBuffer(capacity: 10).write(to: folder)) { error in
            guard case .destinationNotAFile? = error as? ReplayBuffer.PersistenceError else {
                return XCTFail("expected destinationNotAFile, got \(error)")
            }
        }
        XCTAssertEqual(try Data(contentsOf: keep), Data("keep".utf8), "the folder and its contents must survive")
    }

    func testASymbolicLinkAtTheDestinationIsRefusedAndKept() throws {
        let target = root.appendingPathComponent("target.txt")
        try Data("target".utf8).write(to: target)
        let link = root.appendingPathComponent("replay_buffer.bin")
        try FileManager.default.createSymbolicLink(at: link, withDestinationURL: target)

        XCTAssertThrowsError(try ReplayBuffer(capacity: 10).write(to: link))
        XCTAssertEqual(try FileManager.default.destinationOfSymbolicLink(atPath: link.path), target.path)
        XCTAssertEqual(try Data(contentsOf: target), Data("target".utf8))
    }

    func testAnExistingRegularFileIsRefusedAndLeftByteIdentical() throws {
        let url = root.appendingPathComponent("replay_buffer.bin")
        try Data("stale".utf8).write(to: url)
        let identityBefore = try FileSafety.existingItem(at: url)?.identity
        XCTAssertThrowsError(try ReplayBuffer(capacity: 10).write(to: url)) { error in
            guard case .destinationExists(let path)? = error as? ReplayBuffer.PersistenceError else {
                return XCTFail("expected destinationExists, got \(error)")
            }
            XCTAssertEqual(path, url.path)
        }
        XCTAssertEqual(try Data(contentsOf: url), Data("stale".utf8), "the existing file must be left byte-identical")
        XCTAssertEqual(try FileSafety.existingItem(at: url)?.identity, identityBefore, "and must be the same file")
    }

    func testAnExistingValidBufferFileIsRefusedToo() throws {
        let url = root.appendingPathComponent("replay_buffer.bin")
        _ = try ReplayBuffer(capacity: 10).write(to: url)
        let before = try Data(contentsOf: url)
        XCTAssertThrowsError(try ReplayBuffer(capacity: 10).write(to: url)) { error in
            guard case .destinationExists? = error as? ReplayBuffer.PersistenceError else {
                return XCTFail("expected destinationExists, got \(error)")
            }
        }
        XCTAssertEqual(try Data(contentsOf: url), before)
    }

    func testANewDestinationIsCreated() throws {
        let url = root.appendingPathComponent("replay_buffer.bin")
        _ = try ReplayBuffer(capacity: 10).write(to: url)
        XCTAssertNoThrow(try ReplayBuffer(capacity: 10).restore(from: url))
    }
}
