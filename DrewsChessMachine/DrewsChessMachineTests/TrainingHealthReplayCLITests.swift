import XCTest
@testable import DrewsChessMachine

/// `--replay-health-log` (`TrainingHealthReplayCLI`): argument parsing, the
/// config it runs under, reading logs from disk, and the summary table. The
/// replay itself is `TrainingHealthLogReplay`, covered by its own tests and
/// the incident tests.
final class TrainingHealthReplayCLITests: XCTestCase {

    typealias CLI = TrainingHealthReplayCLI

    func testParsesLogsAndOverrides() throws {
        let arguments = try CLI.parse([
            "--replay-health-log", "a.txt", "b.txt", "--learning-grace-steps", "250",
            "--lr-warmup-steps", "40", "--segment-step-as-trainer-step",
        ])
        XCTAssertEqual(arguments, CLI.Arguments(
            logPaths: ["a.txt", "b.txt"], learningGraceSteps: 250, lrWarmupSteps: 40,
            segmentStepAsTrainerStep: true))
    }

    func testRefusesBadArguments() {
        XCTAssertThrowsError(try CLI.parse(["--replay-health-log"])) {
            XCTAssertEqual($0 as? CLI.UsageError, .noLogs)
        }
        XCTAssertThrowsError(try CLI.parse(["--replay-health-log", "a", "--bogus"])) {
            XCTAssertEqual($0 as? CLI.UsageError, .unknownFlag("--bogus"))
        }
        XCTAssertThrowsError(try CLI.parse(["--replay-health-log", "a", "--lr-warmup-steps"])) {
            XCTAssertEqual($0 as? CLI.UsageError, .missingValue("--lr-warmup-steps"))
        }
        XCTAssertThrowsError(try CLI.parse(["--replay-health-log", "a", "--lr-warmup-steps", "x"])) {
            XCTAssertEqual($0 as? CLI.UsageError, .notAnInteger(flag: "--lr-warmup-steps", value: "x"))
        }
        XCTAssertThrowsError(try CLI.parse(["--replay-health-log", "a", "--lr-warmup-steps", "-5"])) {
            XCTAssertEqual($0 as? CLI.UsageError, .notAnInteger(flag: "--lr-warmup-steps", value: "-5"))
        }
        XCTAssertThrowsError(try CLI.parse(["--replay-health-log", "a", "--segment-step-as-trainer-step",
                                            "--segment-step-as-trainer-step"])) {
            XCTAssertEqual($0 as? CLI.UsageError, .repeatedFlag("--segment-step-as-trainer-step"))
        }
    }

    /// The declared defaults, never the user's saved settings: alarms on,
    /// every action log, the two overrides applied.
    func testConfigIsTheDeclaredDefaultsWithTheOverrides() throws {
        let config = try CLI.config(for: CLI.Arguments(
            logPaths: ["x"], learningGraceSteps: 300, lrWarmupSteps: 20, segmentStepAsTrainerStep: false))
        XCTAssertTrue(config.enabled)
        XCTAssertEqual(config.learningGraceSteps, 300)
        XCTAssertEqual(config.lrWarmupSteps, 20)
        XCTAssertEqual(config.checkIntervalSteps, 1000)
        for rule in TrainingHealthRule.allCases {
            XCTAssertEqual(config.actions[rule], .log)
        }
    }

    func testUnreadableLogThrows() {
        let missing = FileManager.default.temporaryDirectory
            .appendingPathComponent("no-such-log-\(UUID().uuidString).txt").path
        XCTAssertThrowsError(try CLI.replay(CLI.Arguments(
            logPaths: [missing], learningGraceSteps: nil, lrWarmupSteps: nil, segmentStepAsTrainerStep: false)))
    }

    /// Arm C's excerpt from the test bundle, read from disk as the CLI does:
    /// the summary names the critical rules at the plan's steps.
    func testReplaysALogFromDiskAndSummarizes() throws {
        let bundle = Bundle(for: TrainingHealthTestSupport.BundleMarker.self)
        let url = try XCTUnwrap(bundle.url(forResource: "TrainingHealthIncident-C-seg0", withExtension: "log"))
        let output = try CLI.replay(CLI.Arguments(
            logPaths: [url.path], learningGraceSteps: 1000, lrWarmupSteps: 1000, segmentStepAsTrainerStep: false))
        XCTAssertTrue(output.lines.contains { $0.text.hasPrefix("[ALARM] health raise rule=dead_channels severity=critical trainerStep=50 ") })
        let summary = CLI.summaryLines(output)
        XCTAssertEqual(summary.first, "[HEALTH] replay summary")
        XCTAssertEqual(summary.count, TrainingHealthRule.allCases.count + 2)
        let dead = try XCTUnwrap(summary.first { $0.contains("dead_channels ") })
        XCTAssertTrue(dead.contains(" 50 ") && dead.contains("critical"), dead)
        let nonFinite = try XCTUnwrap(summary.first { $0.contains("non_finite ") })
        XCTAssertTrue(nonFinite.hasSuffix("-"), nonFinite)
    }
}
