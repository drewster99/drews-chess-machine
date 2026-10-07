import Foundation
import XCTest
@testable import DrewsChessMachine

/// Fixtures for the Record card's statistics tests
/// (`LICHESS_BOT_RECORD_STATS_PLAN.md` §7): index rows built by hand (with
/// or without per-game facts), and game records built through
/// `LichessBotRecordBuilder.build` from journals, as the data-layer tests do.
enum LichessBotStatsFixtures {

    // MARK: - Index rows

    /// An index row. `LichessBotGameSummary` has no memberwise initializer
    /// (its one initializer reduces a record), so the row goes through its
    /// own `Codable` form — the same path a stored `index.json` takes.
    static func row(
        id: String,
        at createdAt: Date,
        score: Double?,
        speed: String = "blitz",
        rated: Bool = true,
        color: LichessBotColorName = .white,
        kind: LichessBotOpponentKind = .bot,
        opponentID: String? = "opponent",
        opponentName: String? = "Opponent",
        opponentRating: Int? = 1500,
        ourRatingBefore: Int? = 1500,
        ourRatingDiff: Int? = nil,
        status: String = "mate",
        winner: String? = nil,
        plies: Int = 40,
        facts: LichessBotGameFacts? = nil
    ) throws -> LichessBotGameSummary {
        let resolvedWinner: String?
        if let winner {
            resolvedWinner = winner
        } else if let score, score != 0.5 {
            let ourSide = color.rawValue
            let theirSide = color == .white ? "black" : "white"
            resolvedWinner = score == 1 ? ourSide : theirSide
        } else {
            resolvedWinner = nil
        }
        var object: [String: Any] = [
            "gameID": id,
            "createdAt": createdAt.formatted(Date.ISO8601FormatStyle(includingFractionalSeconds: true)),
            "speed": speed,
            "rated": rated,
            "ourColor": color.rawValue,
            "opponentKind": kind.rawValue,
            "status": status,
            "plies": plies,
            "modelIDs": [String](),
            "sourceKinds": [String](),
            "builds": [Int](),
            "reconciliation": "matched",
            "anomalyCount": 0,
        ]
        object["opponentID"] = opponentID
        object["opponentName"] = opponentName
        object["opponentRating"] = opponentRating
        object["ourRatingBefore"] = ourRatingBefore
        object["ourRatingDiff"] = ourRatingDiff
        object["winner"] = resolvedWinner
        object["ourScore"] = score
        if let facts {
            let encoder = JSONEncoder()
            object["facts"] = try JSONSerialization.jsonObject(with: encoder.encode(facts))
        }
        let data = try JSONSerialization.data(withJSONObject: object)
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601withFractionalSeconds
        return try decoder.decode(LichessBotGameSummary.self, from: data)
    }

    /// Facts with move data: the given checkpoints, buckets, held runs,
    /// decisive ply and generations.
    static func facts(
        localDrawCondition: ChessDrawCondition? = nil,
        ourMoveCount: Int = 20,
        ourMovesWithDecision: Int = 20,
        checkpoints: [LichessBotGameMoveFacts.Checkpoint] = [],
        checkpointsWithoutDecision: [Int] = [],
        buckets: [LichessBotGameMoveFacts.Bucket] = [],
        heldWinStartPly: Int? = nil,
        heldLossStartPly: Int? = nil,
        decisive: LichessBotDecisivePly = .notApplicable,
        generations: [LichessBotGenerationFacts] = [],
        decisionsWithoutGeneration: Int = 0
    ) -> LichessBotGameFacts {
        LichessBotGameFacts(
            localDrawCondition: localDrawCondition,
            moves: LichessBotGameMoveFacts(
                ourMoveCount: ourMoveCount,
                ourMovesWithDecision: ourMovesWithDecision,
                checkpoints: checkpoints,
                checkpointsWithoutDecision: checkpointsWithoutDecision,
                expectedScoreBuckets: buckets,
                heldWinStartPly: heldWinStartPly,
                heldLossStartPly: heldLossStartPly,
                decisive: decisive,
                generations: generations,
                decisionsWithoutGeneration: decisionsWithoutGeneration
            )
        )
    }

    static func generation(
        _ modelID: String,
        source: LichessBotModelSourceKind = .trainerSnapshot,
        step: Int? = nil,
        sha: String? = nil,
        lineageRunID: String? = nil,
        cumTrainerStep: Int? = nil,
        moves: Int
    ) -> LichessBotGenerationFacts {
        LichessBotGenerationFacts(
            sourceKind: source,
            modelID: modelID,
            trainingStep: step,
            fileSHA256: sha,
            lineageRunID: lineageRunID,
            segmentIndex: nil,
            cumTrainerStep: cumTrainerStep,
            ourMoves: moves
        )
    }

    /// A UTC Gregorian calendar with Monday weeks.
    static var utcCalendar: Calendar {
        var calendar = Calendar(identifier: .gregorian)
        calendar.timeZone = .gmt
        calendar.firstWeekday = 2
        return calendar
    }

    // MARK: - Records from journals

    static let botID = "drewschessmachine"
    static let createdAtMilliseconds: Int64 = 1_759_500_000_000

    /// A legal 42-ply game from the start (no castling), long enough for
    /// either color's 20th move.
    static let longGame = [
        "e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6", "b5a4", "g8f6", "d2d3", "f8e7", "c2c3",
        "b7b5", "a4b3", "d7d6", "h2h3", "c8b7", "b1d2", "d8d7", "d2f1", "a8d8", "f1g3", "h7h6",
        "c1e3", "g7g6", "d1d2", "h6h5", "a2a3", "h5h4", "g3f1", "f6h7", "f1h2", "h7g5", "f3g5",
        "e7g5", "e3g5", "f7f6", "g5e3", "g6g5", "b3d5", "d7e7", "d5c6", "b7c6",
    ]

    static func generationInfo(id: Int, modelID: String, snapshotAt: Date = Date(timeIntervalSince1970: 1_759_500_000), sha: String? = nil, step: Int? = nil) -> LichessBotGenerationInfo {
        LichessBotGenerationInfo(
            generationID: id,
            sourceKind: sha == nil ? .champion : .file,
            modelID: modelID,
            trainingStep: step,
            snapshotAt: snapshotAt,
            architectureSummary: "test",
            filePath: sha == nil ? nil : "/tmp/model.safetensors",
            fileSHA256: sha
        )
    }

    static func decision(_ uci: String, win: Float, draw: Float, loss: Float) -> LichessBotMoveDecision {
        LichessBotMoveDecision(
            uci: uci, san: uci, chosenProbability: 0.5, topMoves: [],
            win: win, draw: draw, loss: loss, temperature: 0.5, legalMoveCount: 20, randomish: false,
            encodeMilliseconds: 0.1, inferenceMilliseconds: 2, sampleMilliseconds: 0.1
        )
    }

    private static func stateJSON(_ tokens: [String], status: String = "started", winner: String? = nil) -> String {
        let winnerField = winner.map { #","winner":"\#($0)""# } ?? ""
        return #"{"type":"gameState","moves":"\#(tokens.joined(separator: " "))","wtime":170000,"btime":160000,"winc":2000,"binc":2000,"status":"\#(status)"\#(winnerField)}"#
    }

    private static func gameFullJSON(ourColor: LichessBotColorName, tokens: [String]) -> String {
        let us = #"{"id":"\#(botID)","name":"DrewsChessMachine","title":"BOT","rating":1500}"#
        let them = #"{"id":"alice","name":"Alice","rating":1600}"#
        let (white, black) = ourColor == .white ? (us, them) : (them, us)
        return #"{"type":"gameFull","id":"g1","variant":{"key":"standard"},"clock":{"initial":180000,"increment":2000},"speed":"blitz","perf":{"name":"Blitz"},"rated":true,"createdAt":\#(createdAtMilliseconds),"white":\#(white),"black":\#(black),"initialFen":"startpos","state":\#(stateJSON(tokens))}"#
    }

    /// What DCM did at one of its plies.
    enum OurMove {
        /// Decided (by `generation`) and posted.
        case decided(win: Float, draw: Float, loss: Float, generation: LichessBotGenerationInfo)
        /// Posted with no decision journaled (as if the decision line was
        /// lost): the move has no decision in the record.
        case postedWithoutDecision
    }

    /// A record of the first `plies` plies of `longGame`, DCM playing
    /// `ourColor`, finished with `status` / `winner`. `ourMove(ply)` says
    /// what DCM did at each of its plies.
    static func record(
        ourColor: LichessBotColorName,
        plies: Int,
        status: String,
        winner: String?,
        localDrawCondition: ChessDrawCondition? = nil,
        ourMove: (Int) -> OurMove
    ) throws -> LichessBotGameRecord {
        precondition(plies <= longGame.count)
        let tokens = Array(longGame.prefix(plies))
        var at = Date(timeIntervalSince1970: Double(createdAtMilliseconds) / 1000)
        func next() -> Date {
            at = at.addingTimeInterval(1)
            return at
        }
        var entries: [LichessBotJournalEntry] = [
            .init(at: next(), event: .header(schemaVersion: LichessBotJournal.schemaVersion, gameID: "g1", build: 100, gitHash: "test", resumed: false)),
            .init(at: next(), event: .streamOpened(attempt: 0)),
            .init(at: next(), event: .streamLine(raw: gameFullJSON(ourColor: ourColor, tokens: []))),
        ]
        let ourParity = ourColor == .white ? 0 : 1
        for ply in 0..<plies {
            let token = tokens[ply]
            if ply % 2 == ourParity {
                switch ourMove(ply) {
                case .decided(let win, let draw, let loss, let generation):
                    entries.append(.init(at: next(), event: .moveDecided(ply: ply, decision: decision(token, win: win, draw: draw, loss: loss), generation: generation)))
                case .postedWithoutDecision:
                    break
                }
                entries.append(.init(at: next(), event: .movePosted(ply: ply, uci: token, offeringDraw: false, milliseconds: 40)))
            }
            entries.append(.init(at: next(), event: .streamLine(raw: stateJSON(Array(tokens.prefix(ply + 1))))))
        }
        entries.append(.init(at: next(), event: .streamLine(raw: stateJSON(tokens, status: status, winner: winner))))
        entries.append(.init(at: next(), event: .finished(status: status, winner: winner, localDrawCondition: localDrawCondition)))
        return try LichessBotRecordBuilder.build(
            gameID: "g1",
            journal: .init(elements: entries, droppedTrailingByteCount: 0),
            export: nil,
            exportUnavailableReason: "test: no export",
            ourAccountID: botID,
            checkedAt: Date(timeIntervalSince1970: 1_759_510_000)
        )
    }
}
