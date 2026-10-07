import SwiftUI
import XCTest
@testable import DrewsChessMachine

/// A move Lichess refused because the game had already ended (game
/// 9jXSDaFa: a threefold drawn while DCM's reply was in flight) is a
/// timeline event, not a rejected move, in the filed record; and the Record
/// card's dragged height is a minimum, never a clip.
final class LichessBotRefusedMoveAfterGameEndTests: XCTestCase {
    private static let botID = "drewschessmachine"

    private static func stateJSON(_ tokens: [String], status: String) -> String {
        #"{"type":"gameState","moves":"\#(tokens.joined(separator: " "))","wtime":170000,"btime":160000,"winc":2000,"binc":2000,"status":"\#(status)"}"#
    }

    private static func gameFullJSON(tokens: [String], status: String) -> String {
        let us = #"{"id":"\#(botID)","name":"DrewsChessMachine","title":"BOT","rating":1500}"#
        let them = #"{"id":"alice","name":"Alice","rating":1600}"#
        return #"{"type":"gameFull","id":"g1","variant":{"key":"standard"},"clock":{"initial":180000,"increment":2000},"speed":"blitz","perf":{"name":"Blitz"},"rated":true,"createdAt":1759500000000,"white":\#(us),"black":\#(them),"initialFen":"startpos","state":\#(stateJSON(tokens, status: status))}"#
    }

    /// DCM (white) plays e2e4, Alice e7e5; DCM's d2d4 at ply 2 is refused,
    /// and the reopened stream's `gameFull` has the game drawn at ply 2.
    /// `explained` adds the session's explanation of the refusal.
    private func record(explained: Bool) throws -> LichessBotGameRecord {
        var at = Date(timeIntervalSince1970: 1_759_500_000)
        func next() -> Date {
            at = at.addingTimeInterval(1)
            return at
        }
        var entries: [LichessBotJournalEntry] = [
            .init(at: next(), event: .header(schemaVersion: LichessBotJournal.schemaVersion, gameID: "g1", build: 100, gitHash: "test", resumed: false)),
            .init(at: next(), event: .streamOpened(attempt: 0)),
            .init(at: next(), event: .streamLine(raw: Self.gameFullJSON(tokens: [], status: "started"))),
            .init(at: next(), event: .movePosted(ply: 0, uci: "e2e4", offeringDraw: false, milliseconds: 40)),
            .init(at: next(), event: .streamLine(raw: Self.stateJSON(["e2e4"], status: "started"))),
            .init(at: next(), event: .streamLine(raw: Self.stateJSON(["e2e4", "e7e5"], status: "started"))),
            .init(at: next(), event: .moveRejected(ply: 2, uci: "d2d4", error: "Lichess returned HTTP 400: Not your turn, or game already over")),
            .init(at: next(), event: .streamEnded(reason: "resync: move rejected")),
            .init(at: next(), event: .streamOpened(attempt: 0)),
            .init(at: next(), event: .streamLine(raw: Self.gameFullJSON(tokens: ["e2e4", "e7e5"], status: "draw"))),
        ]
        if explained {
            entries.append(.init(at: next(), event: .moveRefusedAfterGameEnded(ply: 2, uci: "d2d4", status: "draw")))
        }
        entries.append(.init(at: next(), event: .finished(status: "draw", winner: nil, localDrawCondition: nil)))
        return try LichessBotRecordBuilder.build(
            gameID: "g1",
            journal: .init(elements: entries, droppedTrailingByteCount: 0),
            export: nil,
            exportUnavailableReason: "test: no export",
            ourAccountID: Self.botID,
            checkedAt: Date(timeIntervalSince1970: 1_759_510_000)
        )
    }

    func testAnExplainedRefusalIsATimelineEventNotARejectedMove() throws {
        let record = try record(explained: true)
        XCTAssertEqual(record.rejectedMoves, [])
        XCTAssertEqual(LichessBotGameFacts(record: record).rejectedMoves, 0)
        XCTAssertTrue(record.events.contains { $0.text == "the game ended (draw) before Lichess took d2d4 at ply 2" }, "\(record.events)")
        XCTAssertEqual(record.anomalies, [])
    }

    func testAnUnexplainedRefusalStaysARejectedMove() throws {
        let record = try record(explained: false)
        XCTAssertEqual(record.rejectedMoves.map(\.uci), ["d2d4"])
        XCTAssertEqual(LichessBotGameFacts(record: record).rejectedMoves, 1)
    }

    /// The journal case round-trips through its persisted JSON.
    func testTheJournalEventRoundTrips() throws {
        let entry = LichessBotJournalEntry(at: Date(timeIntervalSince1970: 1_759_500_000), event: .moveRefusedAfterGameEnded(ply: 66, uci: "g4f4", status: "draw"))
        let decoded = try JSONDecoder().decode(LichessBotJournalEntry.self, from: JSONEncoder().encode(entry))
        XCTAssertEqual(decoded, entry)
    }

    // MARK: - Record card height

    /// Content taller than the minimum is laid out at its own height, as a
    /// scroll view sizes it (no proposed height). The frame the card used
    /// before reported only the minimum there, so the overflow (the stacked
    /// recent games) was drawn under the next card; that is asserted too, as
    /// the reproduction of the bug.
    @MainActor
    func testTheCardContentIsNeverClippedToTheDraggedHeight() {
        let tall = NSHostingView(rootView: LichessBotAtLeastHeightLayout(minimumHeight: 160) {
            Color.clear.frame(height: 400)
        }.frame(width: 300))
        XCTAssertEqual(tall.fittingSize.height, 400, accuracy: 0.5)

        let flexible = NSHostingView(rootView: LichessBotAtLeastHeightLayout(minimumHeight: 160) {
            Color.clear
        }.frame(width: 300))
        XCTAssertEqual(flexible.fittingSize.height, 160, accuracy: 0.5)

        let previousFrame = NSHostingView(rootView: Color.clear.frame(height: 400)
            .frame(minHeight: 160, idealHeight: 160, alignment: .top)
            .frame(width: 300))
        XCTAssertEqual(previousFrame.fittingSize.height, 160, accuracy: 0.5, "the frame the card used reports the minimum, clipping taller content")
    }
}
