import XCTest
@testable import DrewsChessMachine

/// The per-game facts reduced from a record (`LICHESS_BOT_RECORD_STATS_PLAN.md`
/// §3.6, §4.2): checkpoints, buckets, held runs, the decisive ply and the
/// per-generation move counts. Records are built from journals through
/// `LichessBotRecordBuilder.build`.
final class LichessBotGameFactsTests: XCTestCase {
    private typealias Fixtures = LichessBotStatsFixtures
    private typealias Definition = LichessBotSelfAssessmentDefinition

    private let modelA = Fixtures.generationInfo(id: 1, modelID: "20261001-1-AAAA")
    private let modelB = Fixtures.generationInfo(id: 1, modelID: "20261005-3-BBBB", snapshotAt: Date(timeIntervalSince1970: 1_759_503_600))

    /// A decision whose W/D/L encode its ply (win = ply / 100), so a
    /// checkpoint shows which ply it was read from.
    private func plyTagged(_ ply: Int) -> Fixtures.OurMove {
        .decided(win: Float(ply) / 100, draw: 0.1, loss: 0.9 - Float(ply) / 100, generation: modelA)
    }

    private func facts(_ record: LichessBotGameRecord) throws -> LichessBotGameMoveFacts {
        try XCTUnwrap(LichessBotGameFacts(record: record).moves)
    }

    func testTheFixtureReplaysCleanly() throws {
        let record = try Fixtures.record(ourColor: .white, plies: 42, status: "resign", winner: "white") { self.plyTagged($0) }
        XCTAssertEqual(record.anomalies, [])
        XCTAssertEqual(record.moves.count, 42)
        XCTAssertTrue(record.moves.allSatisfy { $0.san != nil })
        XCTAssertEqual(record.outcome.ourScore, 1)
    }

    func testCheckpointsAreReadAtTheRightPlyForEachColor() throws {
        XCTAssertEqual(Definition.ply(ofOurMove: 10, ourColor: .white), 18)
        XCTAssertEqual(Definition.ply(ofOurMove: 10, ourColor: .black), 19)
        let white = try facts(Fixtures.record(ourColor: .white, plies: 42, status: "resign", winner: "white") { self.plyTagged($0) })
        XCTAssertEqual(white.checkpoints.map(\.moveNumber), [10, 20])
        XCTAssertEqual(white.checkpoints.map(\.win), [Float(18) / 100, Float(38) / 100])
        // The game ends before DCM's 40th move: no checkpoint 40, not even
        // a missing one (survivors only).
        XCTAssertEqual(white.checkpointsWithoutDecision, [])
        let black = try facts(Fixtures.record(ourColor: .black, plies: 42, status: "resign", winner: "white") { self.plyTagged($0) })
        XCTAssertEqual(black.checkpoints.map(\.moveNumber), [10, 20])
        XCTAssertEqual(black.checkpoints.map(\.win), [Float(19) / 100, Float(39) / 100])
        XCTAssertEqual(black.ourMoveCount, 21)
        XCTAssertEqual(black.ourMovesWithDecision, 21)
    }

    func testAGameEndingBeforeMove20HasOnlyCheckpoint10() throws {
        let short = try facts(Fixtures.record(ourColor: .white, plies: 30, status: "resign", winner: "black") { self.plyTagged($0) })
        XCTAssertEqual(short.checkpoints.map(\.moveNumber), [10])
    }

    func testAMissingDecisionAtACheckpointIsCountedAsMissing() throws {
        let record = try Fixtures.record(ourColor: .white, plies: 42, status: "resign", winner: "white") { ply in
            ply == 18 ? .postedWithoutDecision : self.plyTagged(ply)
        }
        let moves = try facts(record)
        XCTAssertEqual(moves.checkpoints.map(\.moveNumber), [20])
        XCTAssertEqual(moves.checkpointsWithoutDecision, [10])
        XCTAssertEqual(moves.ourMoveCount, 21)
        XCTAssertEqual(moves.ourMovesWithDecision, 20)
    }

    func testBucketEdgesThroughTheOneIndexFunction() {
        // `Float` expected scores are not decimal: 0.1 and 0.3 round up and
        // land in their own tenth, 0.7 and 0.9 round down and land in the
        // tenth below. 1.0 and a step above clamp into the top bucket.
        XCTAssertEqual(Definition.bucketIndex(expectedScore: 0), 0)
        XCTAssertEqual(Definition.bucketIndex(expectedScore: 0.1), 1)
        XCTAssertEqual(Definition.bucketIndex(expectedScore: 0.3), 3)
        XCTAssertEqual(Definition.bucketIndex(expectedScore: 0.7), 6)
        XCTAssertEqual(Definition.bucketIndex(expectedScore: 0.9), 8)
        XCTAssertEqual(Definition.bucketIndex(expectedScore: 1.0), 9)
        XCTAssertEqual(Definition.bucketIndex(expectedScore: Float(1.0).nextUp), 9)
        XCTAssertEqual(Definition.bucketIndex(expectedScore: -Float.ulpOfOne), 0)
    }

    func testBucketsCountEveryDecision() throws {
        // E = win + ½·draw: 0.1, 0.3, 0.9, 1.0 and a step above 1.0, then
        // the rest at 0.55.
        let wins: [Int: (Float, Float)] = [0: (0.1, 0), 2: (0.3, 0), 4: (0.9, 0), 6: (1.0, 0), 8: (Float(1.0).nextUp, 0)]
        let record = try Fixtures.record(ourColor: .white, plies: 20, status: "draw", winner: nil) { ply in
            let (win, draw) = wins[ply] ?? (0.3, 0.5)
            return .decided(win: win, draw: draw, loss: max(0, 1 - win - draw), generation: self.modelA)
        }
        let moves = try facts(record)
        XCTAssertEqual(moves.expectedScoreBuckets.map(\.index), [1, 3, 5, 8, 9])
        XCTAssertEqual(moves.expectedScoreBuckets.map(\.positions), [1, 1, 5, 1, 2])
        XCTAssertEqual(moves.expectedScoreBuckets.map(\.positions).reduce(0, +), moves.ourMovesWithDecision)
        let top = try XCTUnwrap(moves.expectedScoreBuckets.last)
        XCTAssertEqual(top.sumExpected, 1.0 + Double(Float(1.0).nextUp), accuracy: 1e-12)
    }

    func testHeldRuns() throws {
        // Exactly two consecutive DCM moves at ≥ 0.80: held from the first.
        let exactlyTwo = try facts(Fixtures.record(ourColor: .white, plies: 20, status: "resign", winner: "white") { ply in
            let win: Float = (ply == 4 || ply == 6) ? 0.85 : 0.5
            return .decided(win: win, draw: 0.1, loss: 0.9 - win, generation: self.modelA)
        })
        XCTAssertEqual(exactlyTwo.heldWinStartPly, 4)
        XCTAssertNil(exactlyTwo.heldLossStartPly)
        // A move without a decision between two high moves breaks the run.
        let broken = try facts(Fixtures.record(ourColor: .white, plies: 20, status: "resign", winner: "white") { ply in
            if ply == 6 { return .postedWithoutDecision }
            let win: Float = (ply == 4 || ply == 8) ? 0.9 : 0.5
            return .decided(win: win, draw: 0.05, loss: 0.95 - win, generation: self.modelA)
        })
        XCTAssertNil(broken.heldWinStartPly)
        // A single-move spike is not a hold.
        let spike = try facts(Fixtures.record(ourColor: .black, plies: 20, status: "resign", winner: "black") { ply in
            let loss: Float = ply == 9 ? 0.95 : 0.2
            return .decided(win: (1 - loss) / 2, draw: (1 - loss) / 2, loss: loss, generation: self.modelA)
        })
        XCTAssertNil(spike.heldLossStartPly)
        // Two runs: the first one's start is kept.
        let twoRuns = try facts(Fixtures.record(ourColor: .black, plies: 30, status: "resign", winner: "white") { ply in
            let loss: Float = [5, 7, 15, 17, 19].contains(ply) ? 0.85 : 0.1
            return .decided(win: 0.9 - loss, draw: 0.1, loss: loss, generation: self.modelA)
        })
        XCTAssertEqual(twoRuns.heldLossStartPly, 5)
    }

    func testDecisivePly() throws {
        // A win: ≥ 0.80 at 20, a dip at 22, then ≥ 0.80 from 24 to the
        // end, with a decision-less move at 30 inside the stretch (skipped).
        let win = try facts(Fixtures.record(ourColor: .white, plies: 42, status: "mate", winner: "white") { ply in
            if ply == 30 { return .postedWithoutDecision }
            let p: Float = ply == 20 || ply >= 24 ? 0.85 : 0.4
            return .decided(win: p, draw: 0.1, loss: 0.9 - p, generation: self.modelA)
        })
        XCTAssertEqual(win.decisive, .atPly(24))
        // A loss uses the loss probability.
        let loss = try facts(Fixtures.record(ourColor: .black, plies: 41, status: "mate", winner: "white") { ply in
            let p: Float = ply >= 35 ? 0.9 : 0.3
            return .decided(win: 0.95 - p, draw: 0.05, loss: p, generation: self.modelA)
        })
        XCTAssertEqual(loss.decisive, .atPly(35))
        // The last decision below the threshold: never.
        let never = try facts(Fixtures.record(ourColor: .white, plies: 20, status: "resign", winner: "white") { ply in
            let p: Float = ply == 18 ? 0.6 : 0.95
            return .decided(win: p, draw: 0.02, loss: 0.98 - p, generation: self.modelA)
        })
        XCTAssertEqual(never.decisive, .never)
        // A won game with no decision at all: no data, not never.
        let noData = try facts(Fixtures.record(ourColor: .white, plies: 20, status: "resign", winner: "white") { _ in .postedWithoutDecision })
        XCTAssertEqual(noData.decisive, .noDecisions)
        XCTAssertEqual(noData.ourMovesWithDecision, 0)
        XCTAssertNil(noData.heldWinStartPly)
        XCTAssertEqual(noData.expectedScoreBuckets, [])
        // A draw: not applicable.
        let draw = try facts(Fixtures.record(ourColor: .white, plies: 20, status: "draw", winner: nil, localDrawCondition: .threefoldRepetition) { _ in
            .decided(win: 0.9, draw: 0.05, loss: 0.05, generation: self.modelA)
        })
        XCTAssertEqual(draw.decisive, .notApplicable)
        XCTAssertEqual(LichessBotGameFacts(record: try Fixtures.record(ourColor: .white, plies: 20, status: "draw", winner: nil, localDrawCondition: .threefoldRepetition) { _ in .postedWithoutDecision }).localDrawCondition, .threefoldRepetition)
    }

    func testPerGenerationMoveCountsUseTheGenerationIndex() throws {
        // Both generations are numbered 1 (two going-online sessions).
        let record = try Fixtures.record(ourColor: .white, plies: 42, status: "resign", winner: "white") { ply in
            .decided(win: 0.5, draw: 0.2, loss: 0.3, generation: ply <= 20 ? self.modelA : self.modelB)
        }
        let moves = try facts(record)
        XCTAssertEqual(moves.generations.map(\.modelID), [modelA.modelID, modelB.modelID])
        XCTAssertEqual(moves.generations.map(\.ourMoves), [11, 10])
        XCTAssertEqual(moves.decisionsWithoutGeneration, 0)
        XCTAssertEqual(moves.generations.map(\.modelKey), [
            .snapshot(sourceKind: .champion, modelID: modelA.modelID, trainingStep: nil),
            .snapshot(sourceKind: .champion, modelID: modelB.modelID, trainingStep: nil),
        ])
    }

    /// The record JSON with every move's `generationIndex` removed, as a
    /// record written before the field existed.
    private func withoutGenerationIndex(_ record: LichessBotGameRecord) throws -> LichessBotGameRecord {
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601withFractionalSeconds
        var object = try XCTUnwrap(JSONSerialization.jsonObject(with: encoder.encode(record)) as? [String: Any])
        let moves = try XCTUnwrap(object["moves"] as? [[String: Any]])
        object["moves"] = moves.map { move in
            var copy = move
            copy.removeValue(forKey: "generationIndex")
            return copy
        }
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601withFractionalSeconds
        return try decoder.decode(LichessBotGameRecord.self, from: JSONSerialization.data(withJSONObject: object))
    }

    func testAnOldRecordIsAttributedByGenerationID() throws {
        let fileModel = Fixtures.generationInfo(id: 2, modelID: "20261002-1-FILE", sha: String(repeating: "ab", count: 32), step: 5000)
        let record = try Fixtures.record(ourColor: .black, plies: 30, status: "resign", winner: "black") { ply in
            .decided(win: 0.5, draw: 0.2, loss: 0.3, generation: ply < 15 ? self.modelA : fileModel)
        }
        let old = try withoutGenerationIndex(record)
        XCTAssertTrue(old.moves.allSatisfy { $0.generationIndex == nil })
        let moves = try facts(old)
        XCTAssertEqual(moves.generations.map(\.ourMoves), [7, 8])
        XCTAssertEqual(moves.generations[1].modelKey, .file(sha256: String(repeating: "ab", count: 32)))
        XCTAssertEqual(moves.generations[1].trainingStep, 5000)
        XCTAssertEqual(moves.decisionsWithoutGeneration, 0)
        XCTAssertEqual(moves, try facts(record), "the ID resolves exactly what the index resolves")
    }

    func testAGameInWhichDCMNeverMovedHasNoGeneration() throws {
        // DCM is Black; the game ends after White's first move.
        let record = try Fixtures.record(ourColor: .black, plies: 1, status: "resign", winner: "black") { _ in .postedWithoutDecision }
        let moves = try facts(record)
        XCTAssertEqual(moves.ourMoveCount, 0)
        XCTAssertEqual(moves.generations, [])
        XCTAssertEqual(moves.decisive, .noDecisions)
    }

    func testANonStandardStartHasNoMoveData() throws {
        let record = try Fixtures.record(ourColor: .white, plies: 10, status: "resign", winner: "white") { self.plyTagged($0) }
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601withFractionalSeconds
        var object = try XCTUnwrap(JSONSerialization.jsonObject(with: encoder.encode(record)) as? [String: Any])
        var setup = try XCTUnwrap(object["setup"] as? [String: Any])
        setup["initialFen"] = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBN1 w Qkq - 0 1"
        object["setup"] = setup
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601withFractionalSeconds
        let odds = try decoder.decode(LichessBotGameRecord.self, from: JSONSerialization.data(withJSONObject: object))
        let facts = LichessBotGameFacts(record: odds)
        XCTAssertNil(facts.moves)
        XCTAssertNil(facts.localDrawCondition)
    }

    func testAnUnknownGenerationReferenceIsCountedNotGivenToANeighbor() throws {
        let record = try Fixtures.record(ourColor: .white, plies: 10, status: "resign", winner: "white") { self.plyTagged($0) }
        let move = try XCTUnwrap(record.moves.first)
        XCTAssertEqual(LichessBotGameMoveFacts.generationIndex(of: move, in: record.generations), 0)
        XCTAssertNil(LichessBotGameMoveFacts.generationIndex(of: move, in: []))
    }
}
