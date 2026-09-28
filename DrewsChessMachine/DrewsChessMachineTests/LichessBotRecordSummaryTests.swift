import XCTest
@testable import DrewsChessMachine

/// The Overview's Record card: today / this week / all time, by opponent
/// kind and by color.
final class LichessBotRecordSummaryTests: XCTestCase {

    private func summary(id: String, at date: Date, score: Double?, kind: LichessBotOpponentKind, color: LichessBotColorName) throws -> LichessBotGameSummary {
        let json = """
        {"gameID":"\(id)","createdAt":"\(date.formatted(Date.ISO8601FormatStyle(includingFractionalSeconds: true)))","speed":"blitz","rated":false,"ourColor":"\(color.rawValue)","opponentKind":"\(kind.rawValue)","status":"mate","plies":40,"modelIDs":[],"sourceKinds":[],"builds":[],"reconciliation":"matched","anomalyCount":0\(score.map { ",\"ourScore\":\($0)" } ?? "")}
        """
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601withFractionalSeconds
        return try decoder.decode(LichessBotGameSummary.self, from: Data(json.utf8))
    }

    func testPeriodsAndSplits() throws {
        var calendar = Calendar(identifier: .gregorian)
        calendar.timeZone = .gmt
        calendar.firstWeekday = 2
        // Wednesday 2026-09-30 12:00 UTC.
        let now = Date(timeIntervalSince1970: 1_790_769_600)
        let hour: TimeInterval = 3600
        let rows = [
            try summary(id: "a", at: now - 1 * hour, score: 1, kind: .bot, color: .white),
            try summary(id: "b", at: now - 2 * hour, score: 0, kind: .human, color: .black),
            try summary(id: "c", at: now - 30 * hour, score: 0.5, kind: .bot, color: .black),
            try summary(id: "d", at: now - 30 * 24 * hour, score: 1, kind: .lichessAI, color: .white),
            try summary(id: "e", at: now - 3 * hour, score: nil, kind: .human, color: .white),
        ]
        let records = try LichessBotRecordSummary.compute(rows: rows, now: now, calendar: calendar)
        XCTAssertEqual(records.today.all, LichessBotResultTally(wins: 1, draws: 0, losses: 1, unscored: 1))
        XCTAssertEqual(records.thisWeek.all, LichessBotResultTally(wins: 1, draws: 1, losses: 1, unscored: 1))
        XCTAssertEqual(records.allTime.all.games, 5)
        XCTAssertEqual(records.allTime.versusBots, LichessBotResultTally(wins: 1, draws: 1, losses: 0, unscored: 0))
        XCTAssertEqual(records.allTime.versusHumans, LichessBotResultTally(wins: 0, draws: 0, losses: 1, unscored: 1))
        XCTAssertEqual(records.allTime.versusLichessAI.wins, 1)
        XCTAssertEqual(records.allTime.asWhite, LichessBotResultTally(wins: 2, draws: 0, losses: 0, unscored: 1))
        XCTAssertEqual(records.allTime.asBlack, LichessBotResultTally(wins: 0, draws: 1, losses: 1, unscored: 0))
        XCTAssertEqual(try XCTUnwrap(records.allTime.all.score), 2.5 / 4, accuracy: 1e-9)
    }
}
