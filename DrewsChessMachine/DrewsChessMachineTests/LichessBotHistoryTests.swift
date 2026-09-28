import XCTest
@testable import DrewsChessMachine

/// The Challenge sheet's History tab: past opponents aggregated from DCM's
/// records, one row each, most recent first.
final class LichessBotHistoryTests: XCTestCase {

    private func row(_ id: String, opponent: String?, name: String?, kind: LichessBotOpponentKind, at seconds: TimeInterval, score: Double?) throws -> LichessBotGameSummary {
        var fields = #""gameID":"\#(id)","createdAt":"\#(Date(timeIntervalSince1970: seconds).formatted(Date.ISO8601FormatStyle(includingFractionalSeconds: true)))","speed":"blitz","rated":false,"ourColor":"white","opponentKind":"\#(kind.rawValue)","status":"mate","plies":40,"modelIDs":[],"sourceKinds":[],"builds":[],"reconciliation":"matched","anomalyCount":0"#
        if let opponent { fields += #","opponentID":"\#(opponent)""# }
        if let name { fields += #","opponentName":"\#(name)""# }
        if let score { fields += #","ourScore":\#(score)"# }
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601withFractionalSeconds
        return try decoder.decode(LichessBotGameSummary.self, from: Data("{\(fields)}".utf8))
    }

    func testAggregatesPerOpponentMostRecentFirst() throws {
        let rows = [
            try row("a", opponent: "alice", name: "Alice", kind: .human, at: 100, score: 1),
            try row("b", opponent: "somebot", name: "SomeBot", kind: .bot, at: 300, score: 0),
            try row("c", opponent: "alice", name: "Alice2", kind: .human, at: 200, score: 0.5),
            try row("d", opponent: nil, name: "Stockfish level 3", kind: .lichessAI, at: 400, score: 1),
        ]
        let opponents = LichessBotRecordSummary.pastOpponents(rows: rows)
        XCTAssertEqual(opponents.map(\.id), ["somebot", "alice"], "no-account opponents are left out")
        let alice = try XCTUnwrap(opponents.last)
        XCTAssertEqual(alice.name, "Alice2", "the name from the most recent game")
        XCTAssertEqual(alice.record, LichessBotResultTally(wins: 1, draws: 1, losses: 0, unscored: 0))
        XCTAssertEqual(alice.lastPlayedAt, Date(timeIntervalSince1970: 200))
    }
}
