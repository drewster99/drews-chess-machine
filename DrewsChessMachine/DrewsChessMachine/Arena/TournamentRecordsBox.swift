import Foundation

/// Sink the arena callsite uses to gather per-game `TournamentGameRecord`
/// values fired by `TournamentDriver.run`'s `onGameRecorded` callback.
/// The driver fires that callback serially from its parent harvest
/// loop, so in practice there is no concurrent access — the serial
/// queue is a defensive belt for the `@Sendable` callback contract,
/// not for actual contention. Read once via `snapshot()` after the
/// tournament has returned for the post-arena validity sweep.
final class TournamentRecordsBox: @unchecked Sendable {
    private let queue = DispatchQueue(label: "drewschess.tournamentrecords.serial")
    private var _records: [TournamentGameRecord] = []

    func append(_ record: TournamentGameRecord) {
        queue.async { [weak self] in self?._records.append(record) }
    }

    func snapshot() -> [TournamentGameRecord] {
        queue.sync { _records }
    }
}
