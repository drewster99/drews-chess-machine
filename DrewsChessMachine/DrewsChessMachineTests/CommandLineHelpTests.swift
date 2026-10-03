import XCTest
@testable import DrewsChessMachine

/// Pins how the launch path resolves a help request. `--help` / `-h` used to
/// fall through to the top-level unknown-argument scan, which printed
/// "unrecognized argument(s): '--help'" to stderr and exited with the usage
/// error status. A help request must instead resolve to a help mode before any
/// other mode runs; `--derive-model` keeps its own, operation-catalog help.
final class CommandLineHelpTests: XCTestCase {

    func testLongHelpFlagAloneRequestsTopLevelUsage() {
        XCTAssertEqual(CommandLineHelp.request(in: ["--help"]), .topLevelUsage)
    }

    func testShortHelpFlagAloneRequestsTopLevelUsage() {
        XCTAssertEqual(CommandLineHelp.request(in: ["-h"]), .topLevelUsage)
    }

    func testHelpBesideAnotherModeStillRequestsTopLevelUsage() {
        XCTAssertEqual(CommandLineHelp.request(in: ["--train", "--help"]), .topLevelUsage)
        XCTAssertEqual(CommandLineHelp.request(in: ["-h", "--uci"]), .topLevelUsage)
        XCTAssertEqual(CommandLineHelp.request(in: ["--replay-corpus", "corpus", "--help"]), .topLevelUsage)
    }

    func testDeriveModelHelpRequestsDeriveModelUsage() {
        XCTAssertEqual(CommandLineHelp.request(in: [DeriveModelCLI.flag, "--help"]), .deriveModelUsage)
        XCTAssertEqual(CommandLineHelp.request(in: ["--help", DeriveModelCLI.flag]), .deriveModelUsage)
        XCTAssertEqual(CommandLineHelp.request(in: [DeriveModelCLI.flag, "-h"]), .deriveModelUsage)
    }

    func testNoHelpFlagRequestsNothing() {
        XCTAssertNil(CommandLineHelp.request(in: []))
        XCTAssertNil(CommandLineHelp.request(in: ["--train"]))
        XCTAssertNil(CommandLineHelp.request(in: ["--bogus"]))
        XCTAssertNil(CommandLineHelp.request(in: [DeriveModelCLI.flag, "--from", "a.safetensors"]))
        XCTAssertNil(CommandLineHelp.request(in: ["--helpme", "-help", "--h"]))
    }

    func testUsageTextDocumentsBothHelpSpellings() {
        let usageLines = CommandLineHelp.usageText.split(separator: "\n")
        XCTAssertTrue(usageLines.contains { $0.contains("--help, -h") },
                      "the usage text must list the help flag itself")
    }
}
