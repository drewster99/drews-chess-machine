//
//  TournamentDriverSideTallyTests.swift
//  DrewsChessMachineTests
//
//  End-to-end test for TournamentDriver.run's per-side tally logic.
//  The data-model layer (`TournamentRecordSideTests`) covers the
//  computed-property correctness assuming correct raw counts; this
//  file covers the bookkeeping that produces the raw counts —
//  specifically, the branches of the result switch-statement inside
//  the driver's tally loop.
//
//  Test strategy: use a scripted player that plays Fool's Mate —
//  the fastest possible checkmate from the starting position (4
//  plies, always delivered by black). Two instances of the same
//  scripted class facing each other produce a deterministic outcome
//  where whoever plays black wins. Since `TournamentDriver.run`
//  alternates colors every game (game 0 → A=white, game 1 → A=black,
//  ...), we can predict every per-side counter exactly for a given
//  `games: N` input:
//   - A always loses when white; always wins when black.
//   - Game counts split evenly by color for even N.
//   - With 1000 games simulated in ~4 plies each, the test runs in
//     well under a second.
//

import XCTest
@testable import DrewsChessMachine

/// Scripted player that plays both sides of Fool's Mate. As white,
/// plays the inducing moves f2-f3 and g2-g4 that expose the king.
/// As black, plays the delivering moves e7-e5 and Qd8-h4#. Two
/// instances facing each other reach checkmate at ply 4 with black
/// the winner every time.
///
/// Coordinate convention matches the rest of the project:
/// `row 0` = black back rank (rank 8), `row 7` = white back rank
/// (rank 1); `col 0` = file a, `col 7` = file h.
final class FoolsMateScriptedPlayer: ChessPlayer {
    let identifier = UUID().uuidString
    let name: String
    private var isWhite = false
    private var ply = 0

    // Fool's mate transcript, one list per color, indexed by the
    // player's own move counter (0-based). Both scripts run out at
    // index 2, which is past the mate — the guard below falls back
    // to any legal move if somehow the game continues past there,
    // but in practice black's Qh4# ends the game and this player's
    // `onChooseNextMove` won't be called again.
    private static let whiteScript: [(Int, Int, Int, Int)] = [
        (6, 5, 5, 5),   // f2-f3
        (6, 6, 4, 6)    // g2-g4
    ]
    private static let blackScript: [(Int, Int, Int, Int)] = [
        (1, 4, 3, 4),   // e7-e5
        (0, 3, 4, 7)    // Qd8-h4#
    ]

    init(name: String) { self.name = name }

    func onNewGame(_ isWhite: Bool) {
        self.isWhite = isWhite
        self.ply = 0
    }

    func onChooseNextMove(
        opponentMove: ChessMove?,
        newGameState gameState: GameState,
        legalMoves: [ChessMove]
    ) async throws -> ChessMove {
        let script = isWhite ? Self.whiteScript : Self.blackScript
        guard ply < script.count else {
            guard let any = legalMoves.first else {
                throw ChessPlayerError.noLegalMoves
            }
            return any
        }
        let (fr, fc, tr, tc) = script[ply]
        ply += 1
        guard let move = legalMoves.first(where: {
            $0.fromRow == fr && $0.fromCol == fc
                && $0.toRow == tr && $0.toCol == tc
        }) else {
            throw ChessPlayerError.noLegalMoves
        }
        return move
    }

    func onGameEnded(_ result: GameResult, finalState: GameState) {}
}

private enum ScriptedPlayerFailure: Error {
    case boom
}

private final class ThrowingPlayer: ChessPlayer {
    let identifier = UUID().uuidString
    let name: String

    init(name: String) {
        self.name = name
    }

    func onNewGame(_ isWhite: Bool) {}

    func onChooseNextMove(
        opponentMove: ChessMove?,
        newGameState gameState: GameState,
        legalMoves: [ChessMove]
    ) async throws -> ChessMove {
        throw ScriptedPlayerFailure.boom
    }

    func onGameEnded(_ result: GameResult, finalState: GameState) {}
}

/// Lock-protected sink for capturing `@Sendable` callback
/// invocations from the driver's `onGameCompleted` hook. Matches
/// the project's `@unchecked Sendable` + NSLock pattern.
private final class CallbackSink: @unchecked Sendable {
    private let lock = NSLock()
    private var _invocations: [(gameIndex: Int, aWins: Int, bWins: Int, draws: Int)] = []

    func append(_ tuple: (Int, Int, Int, Int)) {
        lock.lock()
        _invocations.append((tuple.0, tuple.1, tuple.2, tuple.3))
        lock.unlock()
    }

    var invocations: [(gameIndex: Int, aWins: Int, bWins: Int, draws: Int)] {
        lock.lock()
        defer { lock.unlock() }
        return _invocations
    }
}

final class TournamentDriverSideTallyTests: XCTestCase {

    // MARK: - Deterministic Fool's Mate outcomes

    func testFoolsMateEvenGameCount() async throws {
        let driver = TournamentDriver()
        let stats = try await driver.run(
            playerA: { FoolsMateScriptedPlayer(name: "A") },
            playerB: { FoolsMateScriptedPlayer(name: "B") },
            games: 4
        )
        XCTAssertEqual(stats.gamesPlayed, 4)
        // Color alternates starting with A=white: games 0, 2 → A is
        // white → A loses (black delivers mate); games 1, 3 → A is
        // black → A wins (A delivers mate).
        XCTAssertEqual(stats.playerAWins, 2)
        XCTAssertEqual(stats.playerBWins, 2)
        XCTAssertEqual(stats.draws, 0)
        // Per-side: A always loses when white, always wins when black.
        XCTAssertEqual(stats.playerAWinsAsWhite, 0,
            "A never wins as white — black delivers the mate every time")
        XCTAssertEqual(stats.playerAWinsAsBlack, 2)
        XCTAssertEqual(stats.playerALossesAsWhite, 2)
        XCTAssertEqual(stats.playerALossesAsBlack, 0)
        XCTAssertEqual(stats.playerADrawsAsWhite, 0)
        XCTAssertEqual(stats.playerADrawsAsBlack, 0)
    }

    func testFoolsMateOddGameCount() async throws {
        // Odd N means A plays white one more time than black.
        // 3 games: A=white in games 0, 2; A=black in game 1.
        let driver = TournamentDriver()
        let stats = try await driver.run(
            playerA: { FoolsMateScriptedPlayer(name: "A") },
            playerB: { FoolsMateScriptedPlayer(name: "B") },
            games: 3
        )
        XCTAssertEqual(stats.gamesPlayed, 3)
        XCTAssertEqual(stats.playerAWinsAsWhite, 0)
        XCTAssertEqual(stats.playerAWinsAsBlack, 1)
        XCTAssertEqual(stats.playerALossesAsWhite, 2)
        XCTAssertEqual(stats.playerALossesAsBlack, 0)
        XCTAssertEqual(stats.playerAWhiteGames, 2)
        XCTAssertEqual(stats.playerABlackGames, 1)
    }

    func testFoolsMateLargerEvenGameCount() async throws {
        // Scales up to 10 games. Confirms the invariants hold over
        // a longer run and that the scripted player correctly
        // resets `ply` between games (the `onNewGame` hook).
        let driver = TournamentDriver()
        let stats = try await driver.run(
            playerA: { FoolsMateScriptedPlayer(name: "A") },
            playerB: { FoolsMateScriptedPlayer(name: "B") },
            games: 10
        )
        XCTAssertEqual(stats.gamesPlayed, 10)
        XCTAssertEqual(stats.playerAWinsAsWhite, 0)
        XCTAssertEqual(stats.playerAWinsAsBlack, 5)
        XCTAssertEqual(stats.playerALossesAsWhite, 5)
        XCTAssertEqual(stats.playerALossesAsBlack, 0)
        XCTAssertEqual(stats.playerAWhiteGames, 5)
        XCTAssertEqual(stats.playerABlackGames, 5)
    }

    // MARK: - Identity invariants over a live driver run

    func testPerSideCountersSumToTotals() async throws {
        // Identity invariants: per-side tallies must reconcile with
        // side-agnostic totals. Catches any future regression that
        // forgets to bump one of the branch counters in the
        // tally switch-statement.
        let driver = TournamentDriver()
        let stats = try await driver.run(
            playerA: { FoolsMateScriptedPlayer(name: "A") },
            playerB: { FoolsMateScriptedPlayer(name: "B") },
            games: 6
        )
        XCTAssertEqual(
            stats.playerAWinsAsWhite + stats.playerAWinsAsBlack,
            stats.playerAWins,
            "per-side A-wins must sum to total A-wins"
        )
        XCTAssertEqual(
            stats.playerALossesAsWhite + stats.playerALossesAsBlack,
            stats.playerBWins,
            "per-side A-losses must sum to B-wins (A's losses)"
        )
        XCTAssertEqual(
            stats.playerADrawsAsWhite + stats.playerADrawsAsBlack,
            stats.draws,
            "per-side A-draws must sum to total draws"
        )
        XCTAssertEqual(
            stats.playerAWhiteGames + stats.playerABlackGames,
            stats.gamesPlayed,
            "per-side game counts must sum to total games"
        )
    }

    // MARK: - Edge cases

    func testZeroGamesReturnsEmptyStats() async throws {
        let driver = TournamentDriver()
        let stats = try await driver.run(
            playerA: { FoolsMateScriptedPlayer(name: "A") },
            playerB: { FoolsMateScriptedPlayer(name: "B") },
            games: 0
        )
        XCTAssertEqual(stats.gamesPlayed, 0)
        XCTAssertEqual(stats.playerAWins, 0)
        XCTAssertEqual(stats.playerBWins, 0)
        XCTAssertEqual(stats.draws, 0)
        XCTAssertEqual(stats.playerAWhiteGames, 0)
        XCTAssertEqual(stats.playerABlackGames, 0)
        // Derived scores should return 0 (not NaN) on empty samples.
        XCTAssertEqual(stats.playerAScoreAsWhite, 0)
        XCTAssertEqual(stats.playerAScoreAsBlack, 0)
        XCTAssertFalse(stats.playerAScoreAsWhite.isNaN)
        XCTAssertFalse(stats.playerAScoreAsBlack.isNaN)
    }

    func testSingleGameAIsWhite() async throws {
        // Game 0 → A=white always. Single-game run should produce
        // exactly one A-loss-as-white and nothing else.
        let driver = TournamentDriver()
        let stats = try await driver.run(
            playerA: { FoolsMateScriptedPlayer(name: "A") },
            playerB: { FoolsMateScriptedPlayer(name: "B") },
            games: 1
        )
        XCTAssertEqual(stats.gamesPlayed, 1)
        XCTAssertEqual(stats.playerALossesAsWhite, 1)
        XCTAssertEqual(stats.playerAWhiteGames, 1)
        XCTAssertEqual(stats.playerABlackGames, 0)
    }

    // MARK: - Cancellation + callback semantics

    func testIsCancelledFlagStopsTournamentEarly() async throws {
        // The driver checks an external `isCancelled` closure on
        // every iteration. When it returns true, the tournament
        // exits after the currently-in-flight game (not mid-game),
        // and the stats reflect games actually completed.
        //
        // We set the flag after 2 games have completed (via the
        // onGameCompleted callback) and verify the run stops at 2.
        let sink = CallbackSink()
        let cancelledFlag = CancelFlag()
        let driver = TournamentDriver()
        let stats = try await driver.run(
            playerA: { FoolsMateScriptedPlayer(name: "A") },
            playerB: { FoolsMateScriptedPlayer(name: "B") },
            games: 100,
            isCancelled: { cancelledFlag.value },
            onGameCompleted: { idx, aW, bW, d in
                sink.append((idx, aW, bW, d))
                if idx >= 2 { cancelledFlag.set() }
            }
        )
        // Should have stopped well before 100.
        XCTAssertLessThan(stats.gamesPlayed, 100)
        XCTAssertGreaterThanOrEqual(stats.gamesPlayed, 2)
        // Per-side identity still holds on the partial run.
        XCTAssertEqual(
            stats.playerAWhiteGames + stats.playerABlackGames,
            stats.gamesPlayed
        )
    }

    func testOnGameCompletedFiresOncePerGame() async throws {
        // The arena progress UI depends on this callback — verify
        // it fires once per completed game with cumulative totals.
        let sink = CallbackSink()
        let driver = TournamentDriver()
        let stats = try await driver.run(
            playerA: { FoolsMateScriptedPlayer(name: "A") },
            playerB: { FoolsMateScriptedPlayer(name: "B") },
            games: 4,
            onGameCompleted: { idx, aW, bW, d in
                sink.append((idx, aW, bW, d))
            }
        )
        let invocations = sink.invocations
        XCTAssertEqual(invocations.count, 4, "exactly one callback per completed game")
        // gameIndex is 1-based per the driver's comment ("games
        // completed so far").
        XCTAssertEqual(invocations.map(\.gameIndex), [1, 2, 3, 4])
        // Cumulative tallies are monotonic and consistent.
        for inv in invocations {
            XCTAssertEqual(inv.aWins + inv.bWins + inv.draws, inv.gameIndex,
                "cumulative W+L+D must equal games completed")
        }
        // The final callback matches the returned stats.
        XCTAssertEqual(invocations.last!.aWins, stats.playerAWins)
        XCTAssertEqual(invocations.last!.bWins, stats.playerBWins)
        XCTAssertEqual(invocations.last!.draws, stats.draws)
    }

    func testNonCancellationPlayerErrorPropagates() async {
        let sink = CallbackSink()
        let driver = TournamentDriver()
        var didThrow = false

        do {
            _ = try await driver.run(
                playerA: { ThrowingPlayer(name: "A") },
                playerB: { FoolsMateScriptedPlayer(name: "B") },
                games: 4,
                onGameCompleted: { idx, aW, bW, d in
                    sink.append((idx, aW, bW, d))
                }
            )
        } catch {
            didThrow = true
        }

        XCTAssertTrue(didThrow, "TournamentDriver.run should throw on player error")
        XCTAssertTrue(
            sink.invocations.isEmpty,
            "Errored games should not be reported as completed tournament progress"
        )
    }

    // MARK: - Identity holds when A/B are swapped

    func testSwappingPlayersFlipsWinsAsBlackAndLosses() async throws {
        // If we swap which factory lands in playerA vs playerB, the
        // outcome flips: Fool's Mate always goes to whoever plays
        // black, so swapping the roles swaps the side-aware win
        // distribution. This pin catches a future reversal of the
        // `aIsWhite` branch that might look identity-correct but
        // produce wrong per-side attribution.
        let driver = TournamentDriver()
        let stats = try await driver.run(
            playerA: { FoolsMateScriptedPlayer(name: "A") },
            playerB: { FoolsMateScriptedPlayer(name: "B") },
            games: 4
        )
        // A is white in games 0, 2 → loses; A is black in games
        // 1, 3 → wins. So A's wins only happen as black.
        XCTAssertEqual(stats.playerAWinsAsWhite, 0)
        XCTAssertGreaterThan(stats.playerAWinsAsBlack, 0)
    }
}

/// Tiny @unchecked-Sendable flag used by the cancellation test to
/// flip a bool from inside one @Sendable callback and read it from
/// another. Same pattern as the project's other cancel boxes.
private final class CancelFlag: @unchecked Sendable {
    private let lock = NSLock()
    private var _value: Bool = false
    var value: Bool {
        lock.lock(); defer { lock.unlock() }
        return _value
    }
    func set() {
        lock.lock(); defer { lock.unlock() }
        _value = true
    }
}
