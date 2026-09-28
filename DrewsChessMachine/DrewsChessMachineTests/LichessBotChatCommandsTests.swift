import XCTest
@testable import DrewsChessMachine

/// Chat commands (plan §12.5a): parsing, reply limits, and the per-game
/// budget. Lichess counts the 140 limit in UTF-16 code units.
final class LichessBotChatCommandsTests: XCTestCase {

    private func line(_ text: String, from user: String = "someone", room: LichessBotChatRoom = .player) -> LichessBotChatLine {
        LichessBotChatLine(room: LichessBotOpenValue(room), username: user, text: text)
    }

    private let hardware = HardwareInfo(
        cpuBrand: "Apple M5 Max",
        cpuPhysicalCores: 18,
        cpuPerformanceLevels: [
            HardwareInfo.CorePerformanceLevel(name: "Super", physicalCores: 6),
            HardwareInfo.CorePerformanceLevel(name: "Performance", physicalCores: 12),
        ],
        memoryBytes: 64 << 30,
        gpuModel: "Apple M5 Max",
        gpuCoreCount: 40,
        readFailures: []
    )

    private func context(username: String = "DrewsChessMachine", modelID: String = "20260727-1-Ejp0", step: Int? = 681_000, hardware: HardwareInfo? = nil) -> LichessBotChatCommandContext {
        LichessBotChatCommandContext(ourUsername: username, modelID: modelID, trainingStep: step, build: 2178, hardware: hardware ?? self.hardware)
    }

    // MARK: - Parsing

    func testCommandsMatchAsCaseInsensitivePrefixes() {
        XCTAssertEqual(LichessBotChatCommands.parse(line("!help"), ourAccountID: "drewschessmachine"), .help)
        XCTAssertEqual(LichessBotChatCommands.parse(line("  !NAME2 "), ourAccountID: "drewschessmachine"), .name)
        XCTAssertEqual(LichessBotChatCommands.parse(line("!about please"), ourAccountID: "drewschessmachine"), .about)
        XCTAssertEqual(LichessBotChatCommands.parse(line("!gpu", room: .spectator), ourAccountID: "drewschessmachine"), .gpu)
    }

    func testNonCommandsAreIgnored() {
        XCTAssertNil(LichessBotChatCommands.parse(line("hello !help"), ourAccountID: "drewschessmachine"))
        XCTAssertNil(LichessBotChatCommands.parse(line("!eval"), ourAccountID: "drewschessmachine"))
        XCTAssertNil(LichessBotChatCommands.parse(line("!"), ourAccountID: "drewschessmachine"))
        XCTAssertNil(LichessBotChatCommands.parse(line("gg"), ourAccountID: "drewschessmachine"))
    }

    /// Our greeting mentions !help; our own lines and the system user's must
    /// never trigger a reply.
    func testOurOwnAndSystemLinesAreNeverCommands() {
        XCTAssertNil(LichessBotChatCommands.parse(line("Type !help for commands", from: "DrewsChessMachine"), ourAccountID: "drewschessmachine"))
        XCTAssertNil(LichessBotChatCommands.parse(line("!help", from: "lichess"), ourAccountID: "drewschessmachine"))
    }

    // MARK: - Replies

    func testEveryReplyFitsTheLimit() throws {
        for command in LichessBotChatCommand.allCases {
            for text in try LichessBotChatCommands.replies(to: command, context: context()) {
                XCTAssertLessThanOrEqual(text.utf16.count, LichessBotChat.maximumLength, "!\(command.rawValue): \(text)")
                XCTAssertFalse(text.lowercased().contains(".com"), "link-like text is dropped by Lichess: \(text)")
            }
        }
    }

    func testAboutIsThreeMessagesEndingWithThisMachine() throws {
        let about = try LichessBotChatCommands.replies(to: .about, context: context())
        XCTAssertEqual(about.count, 3)
        XCTAssertEqual(about[2], "This machine: Apple M5 Max, 18-core CPU (6 super + 12 performance), 40-core GPU, 64 GB unified memory.")
    }

    func testNameIncludesStepAndTailWhenItFits() throws {
        XCTAssertEqual(
            try LichessBotChatCommands.replies(to: .name, context: context()),
            ["DrewsChessMachine running DCM 20260727-1-Ejp0 step 681000 (build 2178) · no search, one forward pass per move"]
        )
    }

    func testNameOmitsAMissingStep() throws {
        XCTAssertEqual(try LichessBotChatCommands.replies(to: .motor, context: context(step: nil)), ["DCM 20260727-1-Ejp0"])
    }

    /// With the longest plausible values the optional tail is dropped
    /// rather than the message failing.
    func testNameDropsTheTailWhenItWouldNotFit() throws {
        let long = context(username: String(repeating: "U", count: 30), modelID: String(repeating: "M", count: 40), step: 99_999_999)
        let reply = try XCTUnwrap(LichessBotChatCommands.replies(to: .name, context: long).first)
        XCTAssertLessThanOrEqual(reply.utf16.count, LichessBotChat.maximumLength)
        XCTAssertFalse(reply.contains("no search"))
    }

    func testAReplyWhoseCoreCannotFitThrows() {
        let absurd = context(modelID: String(repeating: "M", count: 200))
        XCTAssertThrowsError(try LichessBotChatCommands.replies(to: .motor, context: absurd))
    }

    func testMissingHardwareFactsSayUnknown() throws {
        let blank = HardwareInfo(cpuBrand: nil, cpuPhysicalCores: nil, cpuPerformanceLevels: [], memoryBytes: nil, gpuModel: nil, gpuCoreCount: nil, readFailures: ["test"])
        XCTAssertEqual(try LichessBotChatCommands.replies(to: .cpu, context: context(hardware: blank)), ["unknown CPU · unknown core count"])
        XCTAssertEqual(try LichessBotChatCommands.replies(to: .ram, context: context(hardware: blank)), ["unknown memory"])
    }

    // MARK: - Budget

    func testBudgetCooldownCapAndLowClock() {
        var budget = LichessBotChatCommandBudget()
        XCTAssertEqual(budget.decide(now: .seconds(10), ourClockMilliseconds: 60_000), .reply)
        guard case .skip = budget.decide(now: .seconds(11), ourClockMilliseconds: 60_000) else {
            return XCTFail("a reply within the cooldown must be skipped")
        }
        XCTAssertEqual(budget.decide(now: .seconds(13), ourClockMilliseconds: 60_000), .reply)
        guard case .skip = budget.decide(now: .seconds(100), ourClockMilliseconds: 29_999) else {
            return XCTFail("no replies with our clock under 30 s")
        }
        var now = Duration.seconds(200)
        for _ in 2..<LichessBotChatCommandBudget.maximumRepliesPerGame {
            XCTAssertEqual(budget.decide(now: now, ourClockMilliseconds: nil), .reply)
            now += .seconds(5)
        }
        guard case .skip = budget.decide(now: now + .seconds(60), ourClockMilliseconds: nil) else {
            return XCTFail("the per-game cap must hold")
        }
    }

    // MARK: - UTF-16 counting

    /// Lichess counts UTF-16 units: a character outside the Basic
    /// Multilingual Plane is one `Character` but two units.
    func testTemplatesAreMeasuredInUTF16Units() {
        let astral = String(repeating: "𝕏", count: 71)
        XCTAssertEqual(astral.count, 71)
        XCTAssertEqual(astral.utf16.count, 142)
        XCTAssertNil(LichessBotChat.message(from: astral, values: [:]))
        XCTAssertNotNil(LichessBotChat.templateProblem(astral))
    }
}
