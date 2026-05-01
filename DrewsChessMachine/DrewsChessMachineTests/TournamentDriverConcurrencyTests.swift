//
//  TournamentDriverConcurrencyTests.swift
//  DrewsChessMachineTests
//
//  Validates that running `TournamentDriver.run` with concurrency > 1
//  produces the same per-side W/L/D distribution as concurrency == 1
//  for a deterministic player pair. The driver's color alternation
//  is keyed off `gameIndex` (assigned at slot-spawn time in the parent
//  task), so out-of-order completion under concurrency must NOT
//  perturb the side-attribution invariants tested elsewhere in
//  TournamentDriverSideTallyTests.
//
//  This is the headline correctness check for the concurrent-arena
//  feature: if it passes, the parent-task accumulation pattern and
//  the gameIndex-claim discipline are working correctly even when K
//  slot tasks finish out of order.
//

import XCTest
@testable import DrewsChessMachine

final class TournamentDriverConcurrencyTests: XCTestCase {

    func testConcurrencyKMatchesK1ForDeterministicPlayers() async throws {
        // Fool's Mate is fully deterministic: white always loses on
        // ply 4. Running N games at concurrency=1 vs concurrency=4
        // must produce identical per-side tallies, because color
        // alternation is `gameIndex % 2` and `gameIndex` is claimed
        // serially in the parent task at slot-spawn time, regardless
        // of which slot finishes first.
        let games = 12

        let serialDriver = TournamentDriver()
        let serialStats = try await serialDriver.run(
            playerA: { FoolsMateScriptedPlayer(name: "A") },
            playerB: { FoolsMateScriptedPlayer(name: "B") },
            games: games,
            concurrency: 1
        )

        let parallelDriver = TournamentDriver()
        let parallelStats = try await parallelDriver.run(
            playerA: { FoolsMateScriptedPlayer(name: "A") },
            playerB: { FoolsMateScriptedPlayer(name: "B") },
            games: games,
            concurrency: 4
        )

        XCTAssertEqual(serialStats.gamesPlayed, parallelStats.gamesPlayed)
        XCTAssertEqual(serialStats.playerAWins, parallelStats.playerAWins)
        XCTAssertEqual(serialStats.playerBWins, parallelStats.playerBWins)
        XCTAssertEqual(serialStats.draws, parallelStats.draws)
        XCTAssertEqual(serialStats.playerAWinsAsWhite, parallelStats.playerAWinsAsWhite)
        XCTAssertEqual(serialStats.playerAWinsAsBlack, parallelStats.playerAWinsAsBlack)
        XCTAssertEqual(serialStats.playerALossesAsWhite, parallelStats.playerALossesAsWhite)
        XCTAssertEqual(serialStats.playerALossesAsBlack, parallelStats.playerALossesAsBlack)
        XCTAssertEqual(serialStats.playerADrawsAsWhite, parallelStats.playerADrawsAsWhite)
        XCTAssertEqual(serialStats.playerADrawsAsBlack, parallelStats.playerADrawsAsBlack)
    }

    func testOnGameRecordedFiresOncePerCompletedGame() async throws {
        // The `onGameRecorded` callback is the channel the arena
        // callsite uses to collect per-game records for the
        // post-arena validity sweep. Verify it fires exactly once
        // per completed game even under concurrency, and that every
        // record carries a non-empty move history for a Fool's-Mate
        // pair (every game must reach mate at ply 4).
        let games = 8
        let collector = RecordsCollector()
        let driver = TournamentDriver()
        let stats = try await driver.run(
            playerA: { FoolsMateScriptedPlayer(name: "A") },
            playerB: { FoolsMateScriptedPlayer(name: "B") },
            games: games,
            concurrency: 4,
            onGameRecorded: { record in
                collector.append(record)
            }
        )
        XCTAssertEqual(stats.gamesPlayed, games)
        let records = collector.snapshot()
        XCTAssertEqual(records.count, games)
        // Fool's Mate is exactly 4 plies; every record's move
        // history should reflect that.
        for record in records {
            XCTAssertEqual(record.moveHistory.count, 4,
                "every Fool's Mate game ends at ply 4")
        }
        // gameIndex values must be the full 0..<games set, regardless
        // of completion order — confirms the parent task's serial
        // claim of `nextGameIndex` is intact under concurrency.
        let claimedIndices = Set(records.map(\.gameIndex))
        XCTAssertEqual(claimedIndices, Set(0..<games))
    }

    func testOnSlotRetiredFiresOncePerLiveSlot() async throws {
        // Arena uses this callback to decrement each batcher's
        // `expectedSlotCount` as the live pool shrinks. Retirement
        // fires only when a slot leaves the pool (no replacement
        // spawned), so the count must equal the initial fan-out —
        // once per slot as each of the final K games drains. Over-
        // firing (e.g. once per completed game) would pin the
        // barrier count to 0 partway through the run and collapse
        // the remaining batches to size 1; under-firing would leave
        // it above achievable values during the tail.
        let games = 8
        let concurrency = 4
        let counter = AtomicCounter()
        let driver = TournamentDriver()
        _ = try await driver.run(
            playerA: { FoolsMateScriptedPlayer(name: "A") },
            playerB: { FoolsMateScriptedPlayer(name: "B") },
            games: games,
            concurrency: concurrency,
            onSlotRetired: {
                counter.increment()
            }
        )
        XCTAssertEqual(counter.value, min(games, concurrency))
    }

    func testOnSlotRetiredFiresPerGameWhenGamesEqualsConcurrency() async throws {
        // Boundary: when games == concurrency every slot in the
        // initial fan-out runs exactly one game and no replacements
        // are ever spawned, so retirement count equals game count.
        let games = 4
        let counter = AtomicCounter()
        let driver = TournamentDriver()
        _ = try await driver.run(
            playerA: { FoolsMateScriptedPlayer(name: "A") },
            playerB: { FoolsMateScriptedPlayer(name: "B") },
            games: games,
            concurrency: 4,
            onSlotRetired: {
                counter.increment()
            }
        )
        XCTAssertEqual(counter.value, games)
    }

    func testValidateTournamentRecordsAcceptsLegalGames() async throws {
        // The validity sweep helper used by the post-arena log line.
        // Every captured Fool's-Mate game must replay cleanly through
        // a fresh ChessGameEngine.
        let games = 6
        let collector = RecordsCollector()
        let driver = TournamentDriver()
        _ = try await driver.run(
            playerA: { FoolsMateScriptedPlayer(name: "A") },
            playerB: { FoolsMateScriptedPlayer(name: "B") },
            games: games,
            concurrency: 3,
            onGameRecorded: { collector.append($0) }
        )
        let report = validateTournamentRecords(collector.snapshot())
        XCTAssertTrue(report.passed,
            "every captured game must replay cleanly: \(report.failureDescription ?? "")")
        XCTAssertEqual(report.gamesChecked, games)
        XCTAssertEqual(report.totalMovesChecked, 4 * games)
    }
}

private final class RecordsCollector: @unchecked Sendable {
    private let queue = DispatchQueue(label: "drewschess.testcollector.serial")
    private var _records: [TournamentGameRecord] = []
    func append(_ r: TournamentGameRecord) {
        queue.async { [weak self] in self?._records.append(r) }
    }
    func snapshot() -> [TournamentGameRecord] {
        queue.sync { _records }
    }
}

private final class AtomicCounter: @unchecked Sendable {
    private let queue = DispatchQueue(label: "drewschess.testcounter.serial")
    private var _value: Int = 0
    func increment() {
        queue.async { [weak self] in self?._value += 1 }
    }
    var value: Int {
        queue.sync { _value }
    }
}
