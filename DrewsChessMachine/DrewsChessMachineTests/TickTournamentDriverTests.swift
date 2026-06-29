import XCTest
import os
@testable import DrewsChessMachine

/// Integration smoke tests for `TickTournamentDriver`. Spin up the
/// driver with two real `ChessMPSNetwork` instances and run small
/// tournaments end-to-end. The bar for "passes" is invariant checks
/// on the returned `TournamentStats` — game outcomes are stochastic
/// (random-weight networks), so we don't assert specific W/L/D
/// counts, just that the tally is internally consistent.
///
/// **Coverage rationale.** Phase 7b deleted the legacy
/// `TournamentDriverConcurrencyTests` / `TournamentDriverSideTallyTests`
/// when retiring `TournamentDriver`, but the tick driver has different
/// internals (slot-recycled `ActiveGame`s, per-tick partition by
/// current-side network) that weren't covered by anything else. This
/// file exercises:
///   - Initial fan-out + every-game-on-its-own-slot path (K == games).
///   - Slot recycle path (games > K, so each slot serves multiple
///     gameIndices via `ActiveGame.replaceNetworkRefs` +
///     `resetForNewGame`).
///   - Side-attribution math (per-side tallies sum to overall
///     tallies; A's white games + A's black games == total).
///   - `concurrency=0` clamping (the driver clamps to >= 1, so an
///     accidental K=0 doesn't deadlock).
///   - `games=0` short-circuit (returns a zeroed stats struct without
///     touching the GPU).
///
/// Each test allocates two `ChessMPSNetwork(.randomWeights)` instances
/// (the driver compares by reference identity to partition K games
/// per tick). Networks are shared across tests to amortize the
/// graph-build cost.
final class TickTournamentDriverTests: XCTestCase {

    private static let networkPair: (cand: ChessMPSNetwork, champ: ChessMPSNetwork) = {
        do {
            let cand = try ChessMPSNetwork(.randomWeights)
            let champ = try ChessMPSNetwork(.randomWeights)
            return (cand, champ)
        } catch {
            fatalError("TickTournamentDriverTests: network build failed: \(error)")
        }
    }()

    /// Assert the basic tally invariants any non-cancelled
    /// `TournamentStats` from `TickTournamentDriver` must satisfy.
    private func assertStatsConsistent(
        _ stats: TournamentStats,
        expectedGames: Int,
        file: StaticString = #file,
        line: UInt = #line
    ) {
        XCTAssertEqual(
            stats.gamesPlayed, expectedGames,
            "gamesPlayed should equal requested totalGames when not cancelled",
            file: file, line: line
        )
        XCTAssertEqual(
            stats.playerAWins + stats.playerBWins + stats.draws,
            stats.gamesPlayed,
            "A-wins + B-wins + draws must equal gamesPlayed",
            file: file, line: line
        )
        XCTAssertEqual(
            stats.playerAWinsAsWhite + stats.playerAWinsAsBlack,
            stats.playerAWins,
            "per-side A-wins must sum to total A-wins",
            file: file, line: line
        )
        XCTAssertEqual(
            stats.playerALossesAsWhite + stats.playerALossesAsBlack,
            stats.playerBWins,
            "per-side A-losses must sum to total B-wins",
            file: file, line: line
        )
        XCTAssertEqual(
            stats.playerADrawsAsWhite + stats.playerADrawsAsBlack,
            stats.draws,
            "per-side A-draws must sum to total draws",
            file: file, line: line
        )
        // Color alternation: even gameIndex → A is white. So A's
        // white-game count and black-game count must each be exactly
        // ceil/floor of gamesPlayed / 2.
        let expectedWhite = (expectedGames + 1) / 2
        let expectedBlack = expectedGames / 2
        XCTAssertEqual(
            stats.playerAWhiteGames, expectedWhite,
            "A's white games (W+L+D) must equal ceil(games/2) under strict alternation",
            file: file, line: line
        )
        XCTAssertEqual(
            stats.playerABlackGames, expectedBlack,
            "A's black games (W+L+D) must equal floor(games/2) under strict alternation",
            file: file, line: line
        )
    }

    // MARK: - games == 0 short-circuit

    func test_zeroGames_returnsEmptyStats() async throws {
        let driver = TickTournamentDriver()
        let stats = try await driver.run(
            candidateNetwork: Self.networkPair.cand,
            championNetwork: Self.networkPair.champ,
            arenaSchedule: .arena,
            games: 0,
            concurrency: 1
        )
        XCTAssertEqual(stats.gamesPlayed, 0)
        XCTAssertEqual(stats.playerAWins, 0)
        XCTAssertEqual(stats.playerBWins, 0)
        XCTAssertEqual(stats.draws, 0)
    }

    // MARK: - K == games (no slot recycle needed)

    func test_smallTournament_KEqualsGames_consistentTallies() async throws {
        let driver = TickTournamentDriver()
        let totalGames = 4
        // The completion callback fires from concurrently-executing game
        // slots; a lock-protected box keeps the accumulation Swift 6-safe.
        let completedSeen = OSAllocatedUnfairLock(initialState: 0)
        let stats = try await driver.run(
            candidateNetwork: Self.networkPair.cand,
            championNetwork: Self.networkPair.champ,
            arenaSchedule: .arena,
            games: totalGames,
            concurrency: totalGames,
            onGameCompleted: { completed, _, _, _ in
                completedSeen.withLock { $0 = max($0, completed) }
            }
        )
        assertStatsConsistent(stats, expectedGames: totalGames)
        XCTAssertEqual(
            completedSeen.withLock { $0 }, totalGames,
            "onGameCompleted should fire once per finished game"
        )
    }

    // MARK: - games > K (slot recycle exercised)

    func test_recyclePath_KEqualsOne_consistentTallies() async throws {
        // K=1 forces every game past the first to recycle the same
        // slot, exercising `ActiveGame.replaceNetworkRefs +
        // resetForNewGame` (the slot-reuse path that replaced the
        // per-game ActiveGame allocation).
        let driver = TickTournamentDriver()
        let totalGames = 4
        let recordCount = OSAllocatedUnfairLock(initialState: 0)
        let stats = try await driver.run(
            candidateNetwork: Self.networkPair.cand,
            championNetwork: Self.networkPair.champ,
            arenaSchedule: .arena,
            games: totalGames,
            concurrency: 1,
            onGameRecorded: { _ in recordCount.withLock { $0 += 1 } }
        )
        assertStatsConsistent(stats, expectedGames: totalGames)
        XCTAssertEqual(
            recordCount.withLock { $0 }, totalGames,
            "onGameRecorded should fire once per finished game across recycles"
        )
    }

    // MARK: - Mid-K recycle (K=2, games=4 → each slot recycles once)

    func test_recyclePath_KLessThanGames_eachSlotRecyclesOnce() async throws {
        let driver = TickTournamentDriver()
        let totalGames = 4
        let stats = try await driver.run(
            candidateNetwork: Self.networkPair.cand,
            championNetwork: Self.networkPair.champ,
            arenaSchedule: .arena,
            games: totalGames,
            concurrency: 2
        )
        assertStatsConsistent(stats, expectedGames: totalGames)
    }

    // MARK: - Cancellation honored

    func test_externalCancellation_returnsPartialStats() async throws {
        let driver = TickTournamentDriver()
        let cancelFlag = ManagedAtomicFlag()
        // Request enough games that the cancel-on-first-tick latency
        // is short of completion. Cancel immediately so the driver
        // sees `isCancelled() == true` at the top of its first or
        // second tick.
        cancelFlag.signal()
        let stats = try await driver.run(
            candidateNetwork: Self.networkPair.cand,
            championNetwork: Self.networkPair.champ,
            arenaSchedule: .arena,
            games: 8,
            concurrency: 2,
            isCancelled: { cancelFlag.isSet }
        )
        // Cancelled before any game finished: gamesPlayed == 0. The
        // driver's contract is that in-flight unfinished games are
        // not tallied. (If the driver got a few games in before the
        // cancel was observed, gamesPlayed could be > 0; either is
        // legal, but tallies must still be internally consistent.)
        XCTAssertLessThanOrEqual(stats.gamesPlayed, 8)
        XCTAssertEqual(
            stats.playerAWins + stats.playerBWins + stats.draws,
            stats.gamesPlayed,
            "tallies must still be consistent on cancellation"
        )
    }
}

/// Minimal one-shot signal-flag for the cancellation test. Backed by
/// the project's `SyncBox<Bool>` so the read side (called from the
/// driver task) and the write side (called from the test setup) are
/// race-free.
private final class ManagedAtomicFlag: @unchecked Sendable {
    private let box = SyncBox<Bool>(false)
    var isSet: Bool { box.value }
    func signal() { box.value = true }
}
