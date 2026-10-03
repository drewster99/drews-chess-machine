//
//  AutoTrainTerminationTests.swift
//  DrewsChessMachineTests
//
//  The non-exiting part of how a GUI `--train` run ends: one claim winner
//  among the paths that can end the run, and the results write with its log
//  lines. (`writeResultsAndExit` adds the logger shutdown and `_exit`, which
//  a test process cannot run.)
//

import XCTest
@testable import DrewsChessMachine

final class AutoTrainTerminationTests: XCTestCase {

    /// A fresh temporary folder for one test's results, removed when the
    /// test ends.
    private func makeFolder() throws -> URL {
        let folder = FileManager.default.temporaryDirectory
            .appendingPathComponent("AutoTrainTerminationTests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: false)
        addTeardownBlock {
            if FileManager.default.fileExists(atPath: folder.path) {
                try FileManager.default.removeItem(at: folder)
            }
        }
        return folder
    }

    func testOnlyTheFirstClaimWins() async {
        let termination = AutoTrainTermination(recorder: CliTrainingRecorder(), resultsOutput: nil)
        let wins = await withTaskGroup(of: Bool.self, returning: Int.self) { group in
            for _ in 0..<32 {
                group.addTask { termination.claim() }
            }
            var count = 0
            for await won in group where won { count += 1 }
            return count
        }
        XCTAssertEqual(wins, 1)
        XCTAssertFalse(termination.claim())
    }

    func testWritesTheResultsWithTheReasonAndLogsTheOutcome() throws {
        let url = try makeFolder().appendingPathComponent("results.json")
        let output = try CliResultsOutput.preflight(url: url, overwriteAuthorized: false)
        let termination = AutoTrainTermination(recorder: CliTrainingRecorder(), resultsOutput: output)
        var lines: [String] = []
        let outcome = termination.writeResults(
            reason: .stepLimitReached, trigger: "training_step_limit=5 reached at steps=5", elapsed: 12.5,
            log: { lines.append($0) })
        XCTAssertEqual(outcome, .file(url))

        let json = try XCTUnwrap(
            try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any])
        XCTAssertEqual(json["termination_reason"] as? String, "step_limit_reached")
        XCTAssertEqual(json["total_training_seconds"] as? Double, 12.5)

        XCTAssertEqual(lines.count, 2, "\(lines)")
        XCTAssertEqual(lines.first,
                       "[APP] --train: training_step_limit=5 reached at steps=5 (elapsed=12.5s); writing snapshot to \(url.path)")
        XCTAssertTrue(lines.last?.hasPrefix("[APP] --train: wrote snapshot to \(url.path) (arenas=0, stats=0, probes=0)") ?? false,
                      lines.last ?? "no line")
    }

    func testAFailedWriteIsLoggedAndReportedNotThrown() throws {
        let folder = try makeFolder()
        let url = folder.appendingPathComponent("results.json")
        let output = try CliResultsOutput.preflight(url: url, overwriteAuthorized: false)
        // The destination's folder disappears after the pre-flight.
        try FileManager.default.removeItem(at: folder)
        let termination = AutoTrainTermination(recorder: CliTrainingRecorder(), resultsOutput: output)
        var lines: [String] = []
        let outcome = termination.writeResults(
            reason: .timerExpired, trigger: "training_time_limit=60s reached", elapsed: 60, log: { lines.append($0) })
        guard case .failed = outcome else {
            return XCTFail("expected a failed write, got \(outcome)")
        }
        XCTAssertEqual(lines.count, 2, "\(lines)")
        XCTAssertTrue(lines.last?.hasPrefix("[APP] --train: snapshot write FAILED for \(url.path): ") ?? false,
                      lines.last ?? "no line")
        XCTAssertFalse(FileManager.default.fileExists(atPath: url.path))
    }
}
