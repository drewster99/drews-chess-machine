import XCTest
@testable import DrewsChessMachine

/// Lichess spells a decline reason two ways: the decline POST takes the API
/// documentation's camelCase (`reason=noBot`), while the event stream's
/// `challengeDeclined` reports `declineReasonKey` lowercased (`"nobot"`).
/// DCM once matched the event key against the camelCase raw values, so
/// every reason with a capital letter (`noBot`, `timeControl`, `tooFast`,
/// `tooSlow`, `onlyBot`) was recorded as unrecognized, and the Challenge
/// Outcomes card counted those declines under "other" rows while their own
/// rows read zero. Pins: every lowercased event key maps to its reason; an
/// outcome log saved with such a key as unrecognized reads it as the
/// reason, while the record's stored form is unchanged; the POST keeps its
/// camelCase; and a real `nobot` decline, end to end through the
/// controller, is counted as `noBot` and still starts matchmaking's
/// cool-down.
@MainActor
final class LichessBotDeclineReasonKeyTests: XCTestCase {

    /// Every key the event stream sends, spelled as in the protocol logs
    /// (those logs show `nobot`, `timecontrol`, `toofast`, `tooslow`,
    /// `later`, `generic`, `casual` and `variant`; the rest follow Lichess's
    /// rule of lowercasing the reason's name).
    private static let eventStreamKeys: [(key: String, reason: LichessBotDeclineReason)] = [
        ("generic", .generic), ("later", .later), ("toofast", .tooFast), ("tooslow", .tooSlow),
        ("timecontrol", .timeControl), ("rated", .rated), ("casual", .casual), ("standard", .standard),
        ("variant", .variant), ("nobot", .noBot), ("onlybot", .onlyBot),
    ]

    // MARK: - The event key mapping

    func testEveryEventStreamKeyIsItsKnownReason() {
        for (key, reason) in Self.eventStreamKeys {
            XCTAssertEqual(LichessBotDeclineReasonRecord(reasonKey: key), .known(reason), key)
        }
        XCTAssertEqual(Set(Self.eventStreamKeys.map(\.reason)), Set(LichessBotDeclineReason.allCases), "the list covers every reason")
    }

    func testARealDeclineEventDecodesToItsKnownReason() throws {
        // A line from the protocol log, trimmed to the fields DCM reads.
        let line = #"{"type":"challengeDeclined","challenge":{"id":"FXbivR0V","status":"declined","declineReason":"I'm not accepting challenges from bots.","declineReasonKey":"nobot"}}"#
        guard case .challengeDeclined(let reference) = try LichessBotEvent.decode(Data(line.utf8)) else {
            return XCTFail("not decoded as challengeDeclined")
        }
        XCTAssertEqual(LichessBotDeclineReasonRecord(reasonKey: reference.declineReasonKey), .known(.noBot))
    }

    func testAKeyOutsideTheReasonsStaysUnrecognizedAsSent() {
        XCTAssertEqual(LichessBotDeclineReasonRecord(reasonKey: "somethingnew"), .unrecognized("somethingnew"))
        XCTAssertEqual(LichessBotDeclineReasonRecord(reasonKey: ""), .unrecognized(""))
        XCTAssertEqual(LichessBotDeclineReasonRecord(reasonKey: nil), .unstated)
    }

    // MARK: - Saved records

    /// `challenge-outcomes.json` as builds before the fix wrote it: the
    /// lowercased keys stored as unrecognized, next to a known reason and a
    /// key that is really unknown.
    private static let outcomesFileFromBeforeTheFix = #"""
    {
      "records" : [
        {"challengeID" : "a1", "creditCost" : 1, "id" : "00000000-0000-0000-0000-000000000001", "opponentID" : "bot1",
         "opponentKind" : {"bot" : {}}, "outcome" : {"declined" : {"_0" : {"unrecognized" : {"_0" : "nobot"}}}},
         "resolvedAt" : "2026-10-06T16:34:07Z", "sentAt" : "2026-10-06T16:34:07Z"},
        {"challengeID" : "a2", "creditCost" : 1, "id" : "00000000-0000-0000-0000-000000000002", "opponentID" : "bot2",
         "opponentKind" : {"bot" : {}}, "outcome" : {"declined" : {"_0" : {"unrecognized" : {"_0" : "timecontrol"}}}},
         "resolvedAt" : "2026-10-06T16:34:07Z", "sentAt" : "2026-10-06T16:34:07Z"},
        {"challengeID" : "a3", "creditCost" : 1, "id" : "00000000-0000-0000-0000-000000000003", "opponentID" : "bot3",
         "opponentKind" : {"bot" : {}}, "outcome" : {"declined" : {"_0" : {"unrecognized" : {"_0" : "toofast"}}}},
         "resolvedAt" : "2026-10-06T16:34:07Z", "sentAt" : "2026-10-06T16:34:07Z"},
        {"challengeID" : "a4", "creditCost" : 1, "id" : "00000000-0000-0000-0000-000000000004", "opponentID" : "bot4",
         "opponentKind" : {"bot" : {}}, "outcome" : {"declined" : {"_0" : {"known" : {"_0" : "later"}}}},
         "resolvedAt" : "2026-10-06T16:34:07Z", "sentAt" : "2026-10-06T16:34:07Z"},
        {"challengeID" : "a5", "creditCost" : 1, "id" : "00000000-0000-0000-0000-000000000005", "opponentID" : "bot5",
         "opponentKind" : {"bot" : {}}, "outcome" : {"declined" : {"_0" : {"unrecognized" : {"_0" : "somethingnew"}}}},
         "resolvedAt" : "2026-10-06T16:34:07Z", "sentAt" : "2026-10-06T16:34:07Z"}
      ]
    }
    """#

    func testAnOutcomeLogSavedBeforeTheFixReadsLowercasedKeysAsKnownReasons() throws {
        let folder = FileManager.default.temporaryDirectory.appendingPathComponent("LichessBotDeclineReasonKeyTests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        addTeardownBlock {
            do {
                try FileManager.default.removeItem(at: folder)
            } catch {
                XCTFail("cleanup failed: \(error)")
            }
        }
        let url = folder.appendingPathComponent("challenge-outcomes.json")
        try Data(Self.outcomesFileFromBeforeTheFix.utf8).write(to: url)

        let log = try LichessBotChallengeOutcomeLog.load(from: url)
        let sentAt = try XCTUnwrap(ISO8601DateFormatter().date(from: "2026-10-06T16:34:07Z"))
        let summary = log.summary(now: sentAt.addingTimeInterval(1))
        XCTAssertEqual(summary.declined, 5)
        XCTAssertEqual(summary.declinedByReason, [
            .known(.noBot): 1, .known(.timeControl): 1, .known(.tooFast): 1, .known(.later): 1,
            .unrecognized("somethingnew"): 1,
        ])
    }

    /// The stored form is the one Swift synthesizes for the enum, as before:
    /// the reason by its camelCase raw value. The challenge log's golden
    /// lines pin the same shape inside its events.
    func testTheStoredFormIsUnchanged() throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        func stored(_ record: LichessBotDeclineReasonRecord) throws -> String {
            String(decoding: try encoder.encode(record), as: UTF8.self)
        }
        XCTAssertEqual(try stored(.known(.noBot)), #"{"known":{"_0":"noBot"}}"#)
        XCTAssertEqual(try stored(.unrecognized("somethingnew")), #"{"unrecognized":{"_0":"somethingnew"}}"#)
        XCTAssertEqual(try stored(.unstated), #"{"unstated":{}}"#)
        for record in [LichessBotDeclineReasonRecord.known(.timeControl), .unrecognized("somethingnew"), .unstated] {
            XCTAssertEqual(try JSONDecoder().decode(LichessBotDeclineReasonRecord.self, from: encoder.encode(record)), record)
        }
    }

    // MARK: - The decline POST

    /// Lichess's decline endpoint documents the camelCase values; the fix
    /// must not change what DCM sends.
    func testTheDeclinePOSTKeepsTheAPIsCamelCaseSpelling() async throws {
        let expected: [LichessBotDeclineReason: String] = [
            .generic: "generic", .later: "later", .tooFast: "tooFast", .tooSlow: "tooSlow",
            .timeControl: "timeControl", .rated: "rated", .casual: "casual", .standard: "standard",
            .variant: "variant", .noBot: "noBot", .onlyBot: "onlyBot",
        ]
        XCTAssertEqual(Set(expected.keys), Set(LichessBotDeclineReason.allCases))
        let transport = LichessBotScriptedTransport { _ in (200, Data(#"{"ok":true}"#.utf8), [:]) }
        let gate = LichessBotRequestGate(time: LichessBotVirtualTime(), breakerWindow: .seconds(3600)) { _ in }
        let client = LichessBotAPIClient(baseURL: try LichessBotAPIClient.lichessBaseURL(), token: "lip_TESTTOKEN123", transport: transport, gate: gate)
        for reason in LichessBotDeclineReason.allCases {
            try await client.declineChallenge(id: "c1", reason: reason)
        }
        let bodies = transport.requests.value.map { String(decoding: $0.httpBody ?? Data(), as: UTF8.self) }
        XCTAssertEqual(bodies, LichessBotDeclineReason.allCases.map { "reason=\(expected[$0, default: "missing"])" })
    }

    // MARK: - End to end through the controller

    private func waitUntil(_ description: String, _ condition: () -> Bool) async throws {
        for _ in 0..<3000 {
            if condition() { return }
            try await Task.sleep(for: .milliseconds(10))
        }
        XCTFail("timed out waiting until \(description)")
        throw CancellationError()
    }

    /// An online controller whose only matchmaking candidate is FitBot,
    /// with automatic matchmaking off so only the test's Fill Open Slots
    /// sends (the setup of `LichessBotCasualFallbackTests`).
    private func makeOnlineController(lichess: LichessBotFakeLichess) async throws -> LichessBotController {
        let defaults = try makeTemporaryDefaults()
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("LichessBotDeclineReasonKeyTests-\(UUID().uuidString)", isDirectory: true)
        var settings = LichessBotSettings.testBaseline()
        settings.chat.greetingEnabled = false
        settings.connection.preventSleepWhileOnline = false
        settings.challenge.outgoingChallengeTimeoutSeconds = 0
        settings.challenge.maxConcurrentGames = 2
        settings.challenge.gamesReservedForHumans = 1
        settings.matchmaking.enabled = false
        settings.matchmaking.timeControls = [.blitz5plus3]
        settings.matchmaking.rated = true
        settings.matchmaking.declineCooldownHours = 6
        settings.matchmaking.fallBackToCasual = true
        try LichessBotSettingsStore.save(settings, to: defaults)
        let token = LichessBotFakeLichess.token
        let controller = LichessBotController(
            modelProvider: try await LichessBotFakeModelProvider.randomChampion(),
            defaults: defaults,
            dataDirectory: LichessBotDataDirectory(root: root),
            services: LichessBotControllerServices(
                makeTransport: { lichess },
                readToken: { _ in token }
            )
        )
        addTeardownBlock { @MainActor in
            controller.abandonAndStop()
            await controller.shutdown(reason: "test teardown")
            if FileManager.default.fileExists(atPath: root.path) {
                do {
                    try FileManager.default.removeItem(at: root)
                } catch {
                    XCTFail("cleanup failed: \(error)")
                }
            }
        }
        await controller.loadPlayerNotes()
        await controller.goOnline()
        XCTAssertEqual(controller.connection, .online)
        XCTAssertNotNil(controller.challengeOutcomeLog)
        try await waitUntil("the event stream is open") { lichess.eventStreamIsOpen }
        return controller
    }

    /// A bot refusing bots answers matchmaking's challenge with `nobot`: the
    /// outcome is counted as `noBot` (what the Challenge Outcomes card's
    /// `noBot` row reads), the cool-down starts, and nothing is resent.
    func testANobotDeclineOfAMatchmakingChallengeIsCountedAsNoBotAndStartsTheCooldown() async throws {
        let lichess = LichessBotFakeLichess(accountPerfsJSON: #"{"blitz":{"games":100,"rating":1500,"rd":50,"prog":0}}"#)
        lichess.onlineBotsNDJSON.value = #"{"id":"fitbot","username":"FitBot","title":"BOT","perfs":{"blitz":{"games":50,"rating":1600,"rd":60,"prog":0}}}"# + "\n"
        let controller = try await makeOnlineController(lichess: lichess)
        await controller.fillOpenSlots()
        XCTAssertEqual(lichess.challengedNames.value, ["FitBot"])

        lichess.sendEvent(#"{"type":"challengeDeclined","challenge":{"id":"\#(LichessBotFakeLichess.challengeID(for: "FitBot"))","declineReason":"I'm not accepting challenges from bots.","declineReasonKey":"nobot"}}"#)
        try await waitUntil("the decline starts the cool-down") {
            controller.playerNotes?.declineCooldownEnds("fitbot", now: Date()) != nil
        }

        let summary = try XCTUnwrap(controller.challengeOutcomeLog).summary(now: Date())
        XCTAssertEqual(summary.declinedByReason, [.known(.noBot): 1])
        XCTAssertEqual(lichess.challengedNames.value, ["FitBot"], "not a casual decline: nothing is resent")
        XCTAssertNil(controller.casualResendOffer)
        XCTAssertTrue(controller.pendingChallenges.isEmpty)
    }
}
