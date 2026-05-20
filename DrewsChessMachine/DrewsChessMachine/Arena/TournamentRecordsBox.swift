import Foundation
import os

/// Sink the arena callsite uses to gather per-game `TournamentGameRecord`
/// values fired by `TickTournamentDriver.run`'s `onGameRecorded` callback.
/// The driver fires that callback serially from its game-end pass,
/// so in practice there is no concurrent access — the lock is a
/// defensive belt for the `@Sendable` callback contract, not for
/// actual contention. Read once via `snapshot()` after the
/// tournament has returned for the post-arena validity sweep.
///
/// Backed by `OSAllocatedUnfairLock<[TournamentGameRecord]>`; each
/// method holds the lock briefly and never across an `await`.
final class TournamentRecordsBox: @unchecked Sendable {
    private let lock = OSAllocatedUnfairLock<[TournamentGameRecord]>(initialState: [])

    func append(_ record: TournamentGameRecord) {
        lock.withLock { $0.append(record) }
    }

    func snapshot() -> [TournamentGameRecord] {
        lock.withLock { $0 }
    }
}
