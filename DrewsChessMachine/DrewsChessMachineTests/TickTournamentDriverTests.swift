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
            let cand = try ChessMPSNetwork(.randomWeights(initSeed: 1))
            let champ = try ChessMPSNetwork(.randomWeights(initSeed: 2))
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
            randomStreams: DCMRandomStreams(masterSeed: 1),
            arenaIndex: 0,
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
            randomStreams: DCMRandomStreams(masterSeed: 1),
            arenaIndex: 0,
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
            randomStreams: DCMRandomStreams(masterSeed: 1),
            arenaIndex: 0,
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
            randomStreams: DCMRandomStreams(masterSeed: 1),
            arenaIndex: 0,
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
            randomStreams: DCMRandomStreams(masterSeed: 1),
            arenaIndex: 0,
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

    // MARK: - SPRT mode

    /// The score-threshold path must be bit-for-bit the behaviour it was.
    /// Passing no `sprt:` config leaves `sprtVerdict` nil and the fixed
    /// schedule in charge, which is what every existing call site relies on.
    func test_noSPRTConfig_behavesAsFixedScheduleAndReportsNoVerdict() async throws {
        let driver = TickTournamentDriver()
        let totalGames = 4
        let stats = try await driver.run(
            randomStreams: DCMRandomStreams(masterSeed: 1),
            arenaIndex: 0,
            candidateNetwork: Self.networkPair.cand,
            championNetwork: Self.networkPair.champ,
            arenaSchedule: .arena,
            games: totalGames,
            concurrency: 2
        )
        assertStatsConsistent(stats, expectedGames: totalGames)
        XCTAssertNil(stats.sprtVerdict, "a threshold-mode tournament has no sequential verdict")
    }

    /// In SPRT mode `games` is ignored — the test owns the sample size — so a
    /// `games: 0` call must still play. This is the one behaviour that would
    /// silently do nothing if the old `guard totalGames > 0` short-circuit
    /// were left in place, and nothing else in the suite would catch it.
    ///
    /// The outcome is a coin flip (random-weight networks), so this asserts
    /// termination and internal consistency, not which way it decided. A tiny
    /// `maxGames` guarantees the run ends quickly whichever way the evidence
    /// goes: it will either cross a bound or hit the guard.
    func test_sprtMode_ignoresGameCountAndAlwaysTerminates() async throws {
        let driver = TickTournamentDriver()
        let config = try ArenaSPRT.SPRTConfig(
            elo0: 0, elo1: 10, alpha: 0.05, beta: 0.05,
            minGames: 2, maxGames: 4
        )
        let stats = try await driver.run(
            randomStreams: DCMRandomStreams(masterSeed: 1),
            arenaIndex: 0,
            candidateNetwork: Self.networkPair.cand,
            championNetwork: Self.networkPair.champ,
            arenaSchedule: .arena,
            games: 0,
            sprt: config,
            concurrency: 2
        )

        XCTAssertGreaterThan(stats.gamesPlayed, 0, "SPRT mode must not honour games: 0")
        XCTAssertEqual(
            stats.playerAWins + stats.playerBWins + stats.draws,
            stats.gamesPlayed,
            "tallies must be internally consistent"
        )

        let verdict = try XCTUnwrap(stats.sprtVerdict, "an uncancelled SPRT run must reach a verdict")
        XCTAssertTrue(verdict.decision.isFinal)
        XCTAssertEqual(verdict.config, config)

        // The verdict's tally is the one at the crossing; the tournament's is
        // that plus whatever drained. Never the other way round.
        XCTAssertLessThanOrEqual(verdict.gamesAtDecision, stats.gamesPlayed)
        XCTAssertGreaterThanOrEqual(verdict.gamesAtDecision, config.minGames)
    }

    /// Cancelling before the test decides leaves no verdict — and a `nil`
    /// verdict must never be read as "did not promote for statistical
    /// reasons". It means the run was interrupted.
    func test_sprtMode_cancelledBeforeDecisionLeavesNoVerdict() async throws {
        let driver = TickTournamentDriver()
        let cancelFlag = ManagedAtomicFlag()
        cancelFlag.signal()
        let stats = try await driver.run(
            randomStreams: DCMRandomStreams(masterSeed: 1),
            arenaIndex: 0,
            candidateNetwork: Self.networkPair.cand,
            championNetwork: Self.networkPair.champ,
            arenaSchedule: .arena,
            games: 0,
            sprt: try ArenaSPRT.SPRTConfig(
                elo0: 0, elo1: 10, alpha: 0.05, beta: 0.05,
                minGames: 64, maxGames: 0
            ),
            concurrency: 2,
            isCancelled: { cancelFlag.isSet }
        )
        XCTAssertNil(stats.sprtVerdict)
        XCTAssertEqual(
            stats.playerAWins + stats.playerBWins + stats.draws,
            stats.gamesPlayed
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

/// Candidate-perspective W/D/L of a set of finished games, computed straight
/// from each record's result — independent of the driver's own tally and of
/// `ArenaSPRTStartOrderFeed`.
private struct CandidateTally: Equatable {
    var wins = 0
    var draws = 0
    var losses = 0

    init<Records: Sequence>(_ records: Records) where Records.Element == TournamentGameRecord {
        for record in records {
            switch record.result {
            case .checkmate(let winner):
                if (winner == .white) == record.aIsWhite { wins += 1 } else { losses += 1 }
            case .stalemate, .drawByFiftyMoveRule, .drawByInsufficientMaterial, .drawByThreefoldRepetition:
                draws += 1
            }
        }
    }
}

extension TickTournamentDriverTests {
    /// The sequential test must decide on the games in the order they
    /// *started*: the verdict's tally is games `0..<n` by game index, never
    /// the first `n` games to finish (`ArenaSPRTStartOrderFeed` documents the
    /// bias finishing order caused). This runs the driver itself, so it pins
    /// the feed's wiring, not only the feed.
    ///
    /// `elo1 = 0.1` puts the hypotheses so close together that no tally of 24
    /// or fewer games reaches a Wald bound — the largest |LLR| over every
    /// W/D/L with n ≤ 24 is about 0.166, against bounds of ±2.944 — so the
    /// runaway guard decides at exactly 24 games in start order whatever the
    /// results. With 96 games started at once the first 24 to finish are the
    /// shortest; the fixture check fails, asking for another seed, if their
    /// tally ever matches games 0..<24's, since the test could then not tell
    /// the two orders apart.
    func test_sprtMode_verdictTalliesTheStartOrderPrefix() async throws {
        let sampleSize = 24
        let concurrency = 96
        let config = try ArenaSPRT.SPRTConfig(
            elo0: 0, elo1: 0.1, alpha: 0.05, beta: 0.05,
            minGames: 2, maxGames: sampleSize
        )
        // Finishing order: the driver reports each game from its serial
        // game-end pass, in the order the games finish.
        let finishOrder = OSAllocatedUnfairLock(initialState: [TournamentGameRecord]())
        let driver = TickTournamentDriver()
        let stats = try await driver.run(
            randomStreams: DCMRandomStreams(masterSeed: 1),
            arenaIndex: 0,
            candidateNetwork: Self.networkPair.cand,
            championNetwork: Self.networkPair.champ,
            arenaSchedule: .arena,
            games: 0,
            sprt: config,
            concurrency: concurrency,
            onGameRecorded: { record in finishOrder.withLock { $0.append(record) } }
        )
        let finished = finishOrder.withLock { $0 }
        XCTAssertEqual(finished.count, stats.gamesPlayed, "one record per tallied game")

        let verdict = try XCTUnwrap(stats.sprtVerdict, "an uncancelled SPRT run must reach a verdict")
        XCTAssertEqual(verdict.decision, .inconclusive, "no tally of \(sampleSize) games can cross a bound at elo1 = 0.1")
        XCTAssertEqual(verdict.gamesAtDecision, sampleSize)

        let startOrderSample = finished.filter { $0.gameIndex < sampleSize }
        XCTAssertEqual(startOrderSample.count, sampleSize, "every game of the sample finished")
        let startOrder = CandidateTally(startOrderSample)
        let firstFinishers = CandidateTally(finished.prefix(sampleSize))
        XCTAssertNotEqual(
            firstFinishers, startOrder,
            "fixture: the first \(sampleSize) finishers tally the same as games 0..<\(sampleSize), "
                + "so this run cannot tell start order from finishing order — pick another masterSeed"
        )

        XCTAssertEqual(verdict.wins, startOrder.wins, "verdict wins must be games 0..<\(sampleSize)'s")
        XCTAssertEqual(verdict.draws, startOrder.draws, "verdict draws must be games 0..<\(sampleSize)'s")
        XCTAssertEqual(verdict.losses, startOrder.losses, "verdict losses must be games 0..<\(sampleSize)'s")

        // The verdict latched as the last game of the sample finished: the
        // games counted by then are every record up to and including it, and
        // the K − 1 games still in flight are the ones drained after it.
        let lastSampleGame = try XCTUnwrap(finished.lastIndex { $0.gameIndex < sampleSize })
        XCTAssertEqual(stats.sprtGamesFinishedAtDecision, lastSampleGame + 1)
        XCTAssertEqual(stats.gamesPlayed - (lastSampleGame + 1), concurrency - 1,
                       "the games drained after the verdict are the K − 1 in flight at it")
    }
}
