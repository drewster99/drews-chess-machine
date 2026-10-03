//
//  SessionLoggerLazyFileTests.swift
//  DrewsChessMachineTests
//
//  Every CLI tool calls `SessionLogger.start()`, and a `--probe-model` run of
//  a current-format checkpoint never logs a line — so each such run left an
//  empty `dcm_log_*.txt` behind (dozens a day from a probe loop). The log
//  file is created on the first line written instead, under the name the
//  launch time gives it, so a run that never logs leaves no file, and a run
//  that does log gets exactly one.
//

import XCTest
@testable import DrewsChessMachine

final class SessionLoggerLazyFileTests: XCTestCase {

    private var folder: URL!

    override func setUpWithError() throws {
        folder = FileManager.default.temporaryDirectory
            .appendingPathComponent("SessionLoggerLazyFileTests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try FileManager.default.removeItem(at: folder)
    }

    private func logFiles() throws -> [String] {
        try FileManager.default.contentsOfDirectory(atPath: folder.path).filter { $0.hasPrefix("dcm_log_") }.sorted()
    }

    func testStartAloneCreatesNoFile() throws {
        let logger = SessionLogger(location: .directory(folder))
        logger.start()
        XCTAssertNil(logger.activeLogPath)
        logger.shutdown()
        XCTAssertEqual(try logFiles(), [], "a session that never logs must leave no file")
    }

    func testFirstLineCreatesTheFileAndHoldsTheLine() throws {
        let logger = SessionLogger(location: .directory(folder))
        logger.start()
        logger.log("[TEST] first line")
        logger.log("[TEST] second line")
        // `activeLogPath` waits behind the queued writes.
        let path = try XCTUnwrap(logger.activeLogPath)
        logger.shutdown()
        XCTAssertEqual(try logFiles(), [URL(fileURLWithPath: path).lastPathComponent])
        let text = try String(contentsOfFile: path, encoding: .utf8)
        XCTAssertTrue(text.contains("[TEST] first line"))
        XCTAssertTrue(text.contains("[TEST] second line"))
    }

    func testALineAfterShutdownCreatesNoFile() throws {
        let logger = SessionLogger(location: .directory(folder))
        logger.start()
        logger.shutdown()
        logger.log("[TEST] after shutdown")
        XCTAssertNil(logger.activeLogPath)
        XCTAssertEqual(try logFiles(), [])
    }
}
