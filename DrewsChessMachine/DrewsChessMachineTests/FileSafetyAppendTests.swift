//
//  FileSafetyAppendTests.swift
//  DrewsChessMachineTests
//
//  The append path the Lichess bot's journals and logs share
//  (challenge-log plan §3.3, OD-5): `FileSafety.openForAppending` refuses
//  anything but a regular file without touching it, reports exactly whether
//  it created the file, and lands every write at the end;
//  `LichessBotJSONLines.append` holds the file's `flock` across its tail
//  check, cut, write and sync, and cuts on its own descriptor.
//
//  `flock` locks belong to an open file, not to a process, so two
//  descriptors opened separately in this process contend for the lock
//  exactly as two DCM instances do; that is how these tests stand in for a
//  second instance.
//

import XCTest
import Darwin
@testable import DrewsChessMachine

final class FileSafetyAppendTests: XCTestCase {

    private var root: URL!

    override func setUpWithError() throws {
        root = FileManager.default.temporaryDirectory
            .appendingPathComponent("FileSafetyAppendTests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try FileManager.default.removeItem(at: root)
    }

    private func file(_ name: String) -> URL {
        root.appendingPathComponent(name, isDirectory: false)
    }

    private static func line(_ text: String) -> Data {
        Data("{\"m\":\"\(text)\"}\n".utf8)
    }

    /// The real calls, with `write` replaced.
    private static func systemCalls(write: @escaping @Sendable (Data, FileHandle) throws -> Void) -> LichessBotJSONLines.AppendSystemCalls {
        let system = LichessBotJSONLines.AppendSystemCalls.system
        return LichessBotJSONLines.AppendSystemCalls(
            write: write, fsync: system.fsync, fullSync: system.fullSync, fullSyncDirectory: system.fullSyncDirectory)
    }

    /// `flock(LOCK_EX | LOCK_NB)` on a fresh descriptor of `url`: 0 when the
    /// lock was free (it is released again at once), otherwise the `errno`.
    private static func tryLockFromAnotherOpenFile(_ url: URL) -> Int32 {
        let probe = open(url.path, O_RDONLY | O_CLOEXEC)
        guard probe >= 0 else { return -1 }
        defer { close(probe) }
        return flock(probe, LOCK_EX | LOCK_NB) == 0 ? 0 : errno
    }

    // MARK: - openForAppending

    func testCreatesTheFileOnlyOnceAndAppends() throws {
        let url = file("log.jsonl")
        let first = try FileSafety.openForAppending(at: url)
        XCTAssertTrue(first.createdByThisCall)
        try first.handle.write(contentsOf: Self.line("a"))
        try first.handle.close()

        let second = try FileSafety.openForAppending(at: url)
        XCTAssertFalse(second.createdByThisCall, "the file already existed")
        XCTAssertEqual(second.identity, first.identity)
        try second.handle.write(contentsOf: Self.line("b"))
        try second.handle.close()

        XCTAssertEqual(try Data(contentsOf: url), Self.line("a") + Self.line("b"))
    }

    func testTwoHandlesBothLandAtTheEnd() throws {
        let url = file("shared.jsonl")
        let first = try FileSafety.openForAppending(at: url)
        let second = try FileSafety.openForAppending(at: url)
        try first.handle.write(contentsOf: Self.line("1"))
        try second.handle.write(contentsOf: Self.line("2"))
        try first.handle.write(contentsOf: Self.line("3"))
        try first.handle.close()
        try second.handle.close()
        XCTAssertEqual(try Data(contentsOf: url), Self.line("1") + Self.line("2") + Self.line("3"))
    }

    func testRefusesALinkAFolderAndAFIFOUntouched() throws {
        let target = file("target.jsonl")
        let targetBytes = Self.line("kept") + Data("{\"unterminated".utf8)
        try targetBytes.write(to: target)
        let link = file("link.jsonl")
        try FileManager.default.createSymbolicLink(at: link, withDestinationURL: target)
        let folder = root.appendingPathComponent("folder.jsonl", isDirectory: true)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: false)
        let fifo = file("fifo.jsonl")
        XCTAssertEqual(mkfifo(fifo.path, 0o644), 0, "mkfifo failed: errno \(errno)")

        for (url, kind) in [(link, FileSafety.ItemKind.symbolicLink), (folder, .directory), (fifo, .fifo)] {
            XCTAssertThrowsError(try FileSafety.openForAppending(at: url)) { error in
                XCTAssertEqual(error as? FileSafetyError, .notARegularFile(path: url.path, kind: kind))
            }
            XCTAssertThrowsError(try FileSafety.openExistingRegularFileForAppending(at: url)) { error in
                XCTAssertEqual(error as? FileSafetyError, .notARegularFile(path: url.path, kind: kind))
            }
            XCTAssertThrowsError(try LichessBotJSONLines.append(to: url, synchronization: .none, systemCalls: .system) { _ in
                Self.line("never written")
            }) { error in
                XCTAssertEqual(error as? FileSafetyError, .notARegularFile(path: url.path, kind: kind))
            }
            XCTAssertThrowsError(try LichessBotJSONLines.cutUnterminatedFinalLine(of: url)) { error in
                XCTAssertEqual(error as? FileSafetyError, .notARegularFile(path: url.path, kind: kind))
            }
        }
        XCTAssertEqual(try Data(contentsOf: target), targetBytes,
                       "the link's target, unterminated end included, is neither written nor cut")
        XCTAssertEqual(try FileSafety.existingItem(at: link)?.kind, .symbolicLink)
        XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath: folder.path), [])
    }

    func testRefusesADanglingLinkWithoutCreatingItsTarget() throws {
        let missingTarget = file("missing-target.jsonl")
        let link = file("dangling.jsonl")
        try FileManager.default.createSymbolicLink(at: link, withDestinationURL: missingTarget)
        XCTAssertThrowsError(try LichessBotJSONLines.append(to: link, synchronization: .none, systemCalls: .system) { _ in
            Self.line("never written")
        }) { error in
            XCTAssertEqual(error as? FileSafetyError, .notARegularFile(path: link.path, kind: .symbolicLink))
        }
        XCTAssertNil(try FileSafety.existingItem(at: missingTarget), "following the link would have created its target")
    }

    func testAFIFOIsRefusedPromptlyRatherThanHanging() throws {
        let fifo = file("no-reader.jsonl")
        XCTAssertEqual(mkfifo(fifo.path, 0o644), 0, "mkfifo failed: errno \(errno)")
        let outcome = SyncBox<String?>(nil)
        let done = DispatchSemaphore(value: 0)
        DispatchQueue.global().async {
            do {
                _ = try FileSafety.openForAppending(at: fifo)
                outcome.value = "opened"
            } catch {
                outcome.value = String(describing: error)
            }
            done.signal()
        }
        XCTAssertEqual(done.wait(timeout: .now() + 5), .success, "open blocked on the FIFO")
        XCTAssertEqual(outcome.value, String(describing: FileSafetyError.notARegularFile(path: fifo.path, kind: .fifo)))
    }

    func testOpeningAnExistingFileNeverCreatesOne() throws {
        let url = file("absent.jsonl")
        XCTAssertThrowsError(try FileSafety.openExistingRegularFileForAppending(at: url)) { error in
            XCTAssertEqual(error as? FileSafetyError, .systemCallFailed(path: url.path, call: "open", errnoValue: ENOENT))
        }
        XCTAssertThrowsError(try LichessBotJSONLines.cutUnterminatedFinalLine(of: url))
        XCTAssertNil(try FileSafety.existingItem(at: url))
    }

    // MARK: - The append's lock

    func testTheAppendHoldsTheFileLockWhileItWrites() throws {
        let url = file("locked.jsonl")
        try Self.line("before").write(to: url)
        let probes = SyncBox<[Int32]>([])
        let calls = Self.systemCalls { data, handle in
            probes.modify { $0.append(Self.tryLockFromAnotherOpenFile(url)) }
            try handle.write(contentsOf: data)
        }
        try LichessBotJSONLines.append(to: url, synchronization: .none, systemCalls: calls) { _ in Self.line("after") }
        XCTAssertEqual(probes.value, [EWOULDBLOCK], "another open file must find the lock held during the write")
        XCTAssertEqual(Self.tryLockFromAnotherOpenFile(url), 0, "closing the append's descriptor releases the lock")
        XCTAssertEqual(try Data(contentsOf: url), Self.line("before") + Self.line("after"))
    }

    /// A second writer (another instance) holding the lock makes the append
    /// wait — the descriptor's `O_NONBLOCK`, which guards only the open, does
    /// not turn the wait into a failure — and the append goes ahead once the
    /// lock is free.
    func testTheAppendWaitsForALockAnotherWriterHolds() throws {
        let url = file("contended.jsonl")
        try Self.line("first").write(to: url)
        let otherWriter = try FileSafety.openForAppending(at: url)
        try FileSafety.waitForExclusiveLock(onOpenFile: otherWriter.handle.fileDescriptor, path: url.path)

        let writing = DispatchSemaphore(value: 0)
        let finished = DispatchSemaphore(value: 0)
        let failure = SyncBox<String?>(nil)
        let calls = Self.systemCalls { data, handle in
            writing.signal()
            try handle.write(contentsOf: data)
        }
        DispatchQueue.global().async {
            do {
                try LichessBotJSONLines.append(to: url, synchronization: .none, systemCalls: calls) { _ in Self.line("third") }
            } catch {
                failure.value = String(describing: error)
            }
            finished.signal()
        }
        XCTAssertEqual(writing.wait(timeout: .now() + 0.5), .timedOut, "the append must wait while another open file holds the lock")
        try otherWriter.handle.write(contentsOf: Self.line("second"))
        try otherWriter.handle.close()
        XCTAssertEqual(writing.wait(timeout: .now() + 10), .success)
        XCTAssertEqual(finished.wait(timeout: .now() + 10), .success)
        XCTAssertNil(failure.value)
        XCTAssertEqual(try Data(contentsOf: url), Self.line("first") + Self.line("second") + Self.line("third"))
    }

    // MARK: - The tail cut

    func testAnUnterminatedTailIsCutUnderTheLockAndHandedToCompose() throws {
        let url = file("torn.jsonl")
        let fragment = Data("{\"m\":\"interr".utf8)
        try (Self.line("kept") + fragment).write(to: url)
        let lockProbes = SyncBox<[Int32]>([])
        let sizesSeenByCompose = SyncBox<[Int64]>([])
        let outcome = try LichessBotJSONLines.append(to: url, synchronization: .none, systemCalls: .system) { cut in
            XCTAssertEqual(cut, fragment)
            lockProbes.modify { $0.append(Self.tryLockFromAnotherOpenFile(url)) }
            var info = stat()
            XCTAssertEqual(lstat(url.path, &info), 0)
            let size = Int64(info.st_size)
            sizesSeenByCompose.modify { $0.append(size) }
            return Self.line("note") + Self.line("entry")
        }
        XCTAssertEqual(outcome, LichessBotJSONLines.AppendOutcome(cutTail: fragment, createdFile: false))
        XCTAssertEqual(lockProbes.value, [EWOULDBLOCK], "the cut happens under the append's lock")
        XCTAssertEqual(sizesSeenByCompose.value, [Int64(Self.line("kept").count)], "the file is already cut when compose runs")
        XCTAssertEqual(try Data(contentsOf: url), Self.line("kept") + Self.line("note") + Self.line("entry"))
    }

    func testATerminatedFileIsLeftAlone() throws {
        let url = file("clean.jsonl")
        try Self.line("kept").write(to: url)
        let outcome = try LichessBotJSONLines.append(to: url, synchronization: .none, systemCalls: .system) { cut in
            XCTAssertEqual(cut, Data())
            return Self.line("entry")
        }
        XCTAssertEqual(outcome, LichessBotJSONLines.AppendOutcome(cutTail: Data(), createdFile: false))
        XCTAssertEqual(try Data(contentsOf: url), Self.line("kept") + Self.line("entry"))
    }

    func testAFileWithNoNewlineAtAllIsOneFragment() throws {
        let url = file("fragment-only.jsonl")
        let fragment = Data(repeating: UInt8(ascii: "x"), count: 70_000)
        try fragment.write(to: url)
        let outcome = try LichessBotJSONLines.append(to: url, synchronization: .none, systemCalls: .system) { _ in Self.line("entry") }
        XCTAssertEqual(outcome.cutTail, fragment)
        XCTAssertEqual(try Data(contentsOf: url), Self.line("entry"))
    }

    func testANewFileIsReportedCreatedWithNothingCut() throws {
        let url = root.appendingPathComponent("NewFolder/new.jsonl", isDirectory: false)
        let outcome = try LichessBotJSONLines.append(to: url, synchronization: .none, systemCalls: .system) { _ in Self.line("entry") }
        XCTAssertEqual(outcome, LichessBotJSONLines.AppendOutcome(cutTail: Data(), createdFile: true))
        XCTAssertEqual(try Data(contentsOf: url), Self.line("entry"))
    }

    /// The cut is checked on every append, not only on a launch's first one
    /// to a file: another instance that crashed mid-append leaves a fragment
    /// after this one's last good write.
    func testAFragmentLeftAfterThisWritersOwnAppendIsStillCut() throws {
        let url = file("two-writers.jsonl")
        try LichessBotJSONLines.append(to: url, synchronization: .none, systemCalls: .system) { _ in Self.line("mine") }
        let othersFragment = Data("{\"m\":\"theirs, cut sh".utf8)
        let other = try FileSafety.openForAppending(at: url)
        try other.handle.write(contentsOf: othersFragment)
        try other.handle.close()
        let outcome = try LichessBotJSONLines.append(to: url, synchronization: .none, systemCalls: .system) { _ in Self.line("mine again") }
        XCTAssertEqual(outcome.cutTail, othersFragment)
        XCTAssertEqual(try Data(contentsOf: url), Self.line("mine") + Self.line("mine again"))
    }

    /// A write that fails after the cut leaves the file cut (the bytes were
    /// handed to `compose` first), the descriptor closed and the lock free.
    func testAFailedWriteReleasesTheLock() throws {
        struct InjectedWriteFailure: Error {}
        let url = file("failing.jsonl")
        try (Self.line("kept") + Data("{\"frag".utf8)).write(to: url)
        let calls = Self.systemCalls { _, _ in throw InjectedWriteFailure() }
        XCTAssertThrowsError(try LichessBotJSONLines.append(to: url, synchronization: .none, systemCalls: calls) { _ in Self.line("lost") }) { error in
            XCTAssertTrue(error is InjectedWriteFailure, "\(error)")
        }
        XCTAssertEqual(try Data(contentsOf: url), Self.line("kept"))
        XCTAssertEqual(Self.tryLockFromAnotherOpenFile(url), 0)
    }
}
