import XCTest
@testable import DrewsChessMachine

/// Tests for `CorpusReplayFeeder`: a recorded game is replayed to its recorded
/// end — draws the players did not claim never end the replay — while games
/// that genuinely cannot be replayed (a move after mate, an illegal move) are
/// still rejected.
///
/// The fixture shuffles both knights out and back twice, so the start
/// position recurs for the third time after the eighth ply — an unclaimed
/// threefold repetition — and the game then continues with `e2e4 e7e5`.
/// Humans on Lichess legally play on past an unclaimed threefold, so corpus
/// games of exactly this shape exist.
final class CorpusReplayFeederTests: XCTestCase {

    private static var sharedNetwork: ChessMPSNetwork = {
        do {
            return try ChessMPSNetwork(.randomWeights(initSeed: 1))
        } catch {
            fatalError("CorpusReplayFeederTests: ChessMPSNetwork(.randomWeights) failed: \(error)")
        }
    }()

    private static let knightShuffle = ["g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6", "f3g1", "f6g8"]

    /// Parse UCI tokens into moves by playing them through an engine that
    /// never adjudicates draws, so the parse itself can't stop at a
    /// repetition.
    private func moves(_ tokens: [String]) throws -> [ChessMove] {
        let engine = ChessGameEngine(adjudication: .serverAuthoritative)
        var parsed: [ChessMove] = []
        for token in tokens {
            let legal = MoveGenerator.legalMoves(for: engine.state)
            let move = try XCTUnwrap(ChessMove.parseUCI(token, legal: legal), "\(token) is not legal here")
            try engine.applyMoveAndAdvance(move)
            parsed.append(move)
        }
        return parsed
    }

    private func makeBuffer() -> ReplayBuffer {
        ReplayBuffer(capacity: 64, inputEncoding: Self.sharedNetwork.inputEncoding, sampler: DCMRandom(seed: 1))
    }

    private struct StoredSlots: Sendable {
        let boards: [Float]
        let outcomes: [Float]
        let floatsPerBoard: Int
    }

    private func storedSlots(_ buffer: ReplayBuffer) -> StoredSlots {
        let floatsPerBoard = buffer.floatsPerBoard
        return buffer.withSlotData { view in
            StoredSlots(
                boards: Array(UnsafeBufferPointer(start: view.boards, count: view.count * floatsPerBoard)),
                outcomes: Array(UnsafeBufferPointer(start: view.outcomes, count: view.count)),
                floatsPerBoard: floatsPerBoard
            )
        }
    }

    // MARK: - Regression: games continued past an unclaimed threefold

    func testAGameContinuedPastAnUnclaimedThreefoldIsFedWhole() throws {
        let game = GameRecord(moves: try moves(Self.knightShuffle + ["e2e4", "e7e5"]), outcome: .draw)
        let buffer = makeBuffer()
        _ = CorpusReplayFeeder(network: Self.sharedNetwork, buffer: buffer).feed(game)
        XCTAssertEqual(buffer.count, 10, "every recorded position of the game is fed, including those after the unclaimed threefold")
    }

    func testExactlyThePositionAtTheThirdOccurrenceCarriesTheTwiceBeforePlane() throws {
        let game = GameRecord(moves: try moves(Self.knightShuffle + ["e2e4", "e7e5"]), outcome: .draw)
        let buffer = makeBuffer()
        _ = CorpusReplayFeeder(network: Self.sharedNetwork, buffer: buffer).feed(game)
        let slots = storedSlots(buffer)
        XCTAssertEqual(slots.outcomes.count, 10)
        let planeOffset = 19 * 64
        var slotsWithPlane = 0
        for slot in 0..<slots.outcomes.count where slots.boards[slot * slots.floatsPerBoard + planeOffset] == 1 {
            slotsWithPlane += 1
        }
        XCTAssertEqual(slotsWithPlane, 1, "only the start position's third occurrence has occurred twice before")
    }

    func testValueTargetsFollowTheRecordedResultPastTheThreefold() throws {
        let game = GameRecord(moves: try moves(Self.knightShuffle + ["e2e4", "e7e5"]), outcome: .whiteWin)
        let buffer = makeBuffer()
        _ = CorpusReplayFeeder(network: Self.sharedNetwork, buffer: buffer).feed(game)
        let outcomes = storedSlots(buffer).outcomes
        XCTAssertEqual(outcomes.count, 10)
        XCTAssertEqual(outcomes.filter { $0 == 1 }.count, 5, "white-to-move positions target a win")
        XCTAssertEqual(outcomes.filter { $0 == -1 }.count, 5, "black-to-move positions target a loss")
    }

    // MARK: - Guards: behavior that must not change

    func testAGameEndingExactlyOnTheThreefoldIsFedWhole() throws {
        let game = GameRecord(moves: try moves(Self.knightShuffle), outcome: .draw)
        let buffer = makeBuffer()
        _ = CorpusReplayFeeder(network: Self.sharedNetwork, buffer: buffer).feed(game)
        XCTAssertEqual(buffer.count, 8)
    }

    func testAMoveAfterCheckmateRejectsTheWholeGame() throws {
        var played = try moves(["f2f3", "e7e5", "g2g4", "d8h4"])
        // White is mated; any further move is corrupt. a2a3 is a move White
        // could otherwise make from the mated position.
        played.append(ChessMove(fromRow: 6, fromCol: 0, toRow: 5, toCol: 0, promotion: nil))
        let game = GameRecord(moves: played, outcome: .blackWin)
        let buffer = makeBuffer()
        _ = CorpusReplayFeeder(network: Self.sharedNetwork, buffer: buffer).feed(game)
        XCTAssertEqual(buffer.count, 0)
    }

    func testAnIllegalMoveRejectsTheWholeGame() throws {
        var played = try moves(["e2e4", "e7e5"])
        // e4e6: a pawn cannot move two squares from e4.
        played.append(ChessMove(fromRow: 4, fromCol: 4, toRow: 2, toCol: 4, promotion: nil))
        let game = GameRecord(moves: played, outcome: .draw)
        let buffer = makeBuffer()
        _ = CorpusReplayFeeder(network: Self.sharedNetwork, buffer: buffer).feed(game)
        XCTAssertEqual(buffer.count, 0)
    }

    // MARK: - Outcome and tally

    func testAMoveAfterCheckmateIsReportedAsRejectedAtThatPly() throws {
        var played = try moves(["f2f3", "e7e5", "g2g4", "d8h4"])
        played.append(ChessMove(fromRow: 6, fromCol: 0, toRow: 5, toCol: 0, promotion: nil))
        let outcome = CorpusReplayFeeder(network: Self.sharedNetwork, buffer: makeBuffer())
            .feed(GameRecord(moves: played, outcome: .blackWin))
        guard case .rejected(let ply, let error) = outcome else {
            return XCTFail("expected a rejection, got \(outcome)")
        }
        XCTAssertEqual(ply, 4)
        guard case ChessGameError.gameAlreadyOver = error else {
            return XCTFail("expected gameAlreadyOver, got \(error)")
        }
    }

    func testFedAndSkippedOutcomes() throws {
        let feeder = CorpusReplayFeeder(network: Self.sharedNetwork, buffer: makeBuffer())
        guard case .fed(let positions) = feeder.feed(GameRecord(moves: try moves(["e2e4", "e7e5"]), outcome: .draw)) else {
            return XCTFail("a legal two-move game must be fed")
        }
        XCTAssertEqual(positions, 2)
        guard case .skippedEmpty = feeder.feed(GameRecord(moves: [], outcome: .draw)) else {
            return XCTFail("a game with no moves is skipped as empty")
        }
        guard case .skippedStartFEN = feeder.feed(GameRecord(startFEN: "8/8/8/8/8/8/8/K6k w - - 0 1", moves: try moves(["e2e4"]), outcome: .draw)) else {
            return XCTFail("a FEN-setup game is skipped")
        }
    }

    func testTallyCountsEveryConsumedGameAndDescribesOnlyRejections() {
        var tally = CorpusReplayFeedTally()
        XCTAssertNil(tally.record(.fed(positions: 7)))
        XCTAssertNil(tally.record(.skippedEmpty))
        XCTAssertNil(tally.record(.skippedStartFEN))
        let report = tally.record(.rejected(ply: 12, error: ChessGameError.gameAlreadyOver))
        XCTAssertEqual(tally.games, 4)
        XCTAssertEqual(tally.positions, 7)
        XCTAssertEqual(tally.skipped, 2)
        XCTAssertEqual(tally.rejected, 1)
        XCTAssertEqual(report?.hasPrefix("rejected at ply 12: "), true)
        XCTAssertEqual(tally.countsSuffix, " rejected=1 skipped=2")
    }

    /// A game with no draw condition anywhere must feed exactly the positions
    /// an always-adjudicating engine would encode: switching who decides
    /// draws only changes games that reach one.
    func testAGameWithNoDrawConditionFeedsTheSamePositionsAnAdjudicatingReplayEncodes() throws {
        let tokens = ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6", "b5a4", "g8f6", "e1g1", "f8e7"]
        let played = try moves(tokens)
        let game = GameRecord(moves: played, outcome: .whiteWin)
        let buffer = makeBuffer()
        _ = CorpusReplayFeeder(network: Self.sharedNetwork, buffer: buffer).feed(game)
        let slots = storedSlots(buffer)
        XCTAssertEqual(slots.outcomes.count, played.count)

        // Reference: encode every pre-move position with an automatic engine.
        let encoding = Self.sharedNetwork.inputEncoding
        let reference = ChessGameEngine(adjudication: .automatic)
        var expectedBoards: [[Float]] = []
        for move in played {
            let full = BoardEncoder.encode(reference.state, history: reference.recentStates, encoding: encoding)
            expectedBoards.append(Array(full.prefix(slots.floatsPerBoard)))
            try reference.applyMoveAndAdvance(move)
        }
        // The buffer stores the game's positions in some slot order; compare
        // as multisets of exact board bytes.
        var stored: [[Float]] = []
        for slot in 0..<slots.outcomes.count {
            let start = slot * slots.floatsPerBoard
            stored.append(Array(slots.boards[start..<(start + slots.floatsPerBoard)]))
        }
        func key(_ board: [Float]) -> [UInt32] { board.map(\.bitPattern) }
        XCTAssertEqual(stored.map(key).sorted { $0.lexicographicallyPrecedes($1) },
                       expectedBoards.map(key).sorted { $0.lexicographicallyPrecedes($1) })
    }
}
