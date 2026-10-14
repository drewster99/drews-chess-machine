import XCTest
@testable import DrewsChessMachine

/// The 2026-10-14 incident, replayed through the real API client and request
/// gate: Lichess answered two matchmaking challenges within an hour with HTTP
/// 429 because DCM itself had played 100 games against other bots that day.
/// Treated as request-rate limits, the second tripped the gate's breaker,
/// which closed the gate and took the whole bot offline (Error) for nine
/// hours, human games included.
///
/// Written against API that existed before the fix, so it compiles on the
/// code that had the bug and fails there: the second 429 trips the breaker
/// and the later request is refused by the closed gate.
final class LichessBotOwnBotGameLimitRegressionTests: XCTestCase {

    /// lila's exact answer (`BotLimit.challengeLimitError` → `Limited`,
    /// written by `CtrlErrors`' `Writes[Limited]` as `JsonLimited`).
    private static let ownLimitBody = Data(#"{"error":"You played 100 games against other bots today, please wait before challenging another bot.","ratelimit":{"key":"bot.vsBot.day","seconds":6311}}"#.utf8)
    private static let okBody = Data(#"{"ok":true}"#.utf8)

    func testTwoOwnBotGameLimitRefusalsWithinAnHourLeaveTheGateOpen() async throws {
        let events = SyncBox<[LichessBotGateEvent]>([])
        let transport = LichessBotScriptedTransport { request in
            if request.url?.path.hasPrefix("/api/challenge/") == true {
                return (429, Self.ownLimitBody, ["Content-Type": "application/json"])
            }
            return (200, Self.okBody, [:])
        }
        let gate = LichessBotRequestGate(time: LichessBotVirtualTime(), breakerWindow: .seconds(3600)) { event in
            events.modify { $0.append(event) }
        }
        let client = LichessBotAPIClient(baseURL: try LichessBotAPIClient.lichessBaseURL(), token: "lip_TESTTOKEN123", transport: transport, gate: gate)
        let request = LichessBotOutgoingChallenge(rated: true, clockLimitSeconds: 1800, clockIncrementSeconds: 20, color: .random)

        for opponent in ["DarkOnWeakBot", "RatKingBot"] {
            do {
                _ = try await client.challenge(username: opponent, request: request)
                XCTFail("Lichess refused the challenge to \(opponent); it must throw")
            } catch {
                XCTAssertFalse(error is LichessBotGateError, "DCM's own bot-game limit is Lichess's answer, not the gate's request-rate stop: \(error)")
            }
        }

        let breakerTripped = events.value.contains { if case .breakerTripped = $0 { return true } else { return false } }
        let cooledDown = events.value.contains { if case .rateLimited = $0 { return true } else { return false } }
        XCTAssertFalse(breakerTripped, "the breaker must not trip on DCM's own bot-game limit")
        XCTAssertFalse(cooledDown, "DCM's own bot-game limit must not start a request-rate cooldown")
        let phase = await gate.snapshot().phase
        XCTAssertEqual(phase, .open)

        // The bot stays online: a game's move still goes straight out.
        try await client.makeMove(gameID: "pG3WSP96", uci: "e2e4", offeringDraw: false)
    }
}
