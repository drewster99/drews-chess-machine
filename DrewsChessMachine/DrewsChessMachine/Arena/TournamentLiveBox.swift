import Foundation

/// Serial-queue-protected holder for live tournament progress, shared
/// between the driver task (writer, one update per finished game) and
/// the UI heartbeat (reader, polling at 10 Hz). Same pattern as
/// `TrainingLiveStatsBox` and `CancelBox` — a private serial
/// `DispatchQueue` serializes all state access, so the class is
/// safely `@unchecked Sendable`.
final class TournamentLiveBox: @unchecked Sendable {
    private let queue = DispatchQueue(label: "drewschess.tournamentlivebox.serial")
    private var _progress: TournamentProgress?

    func update(_ progress: TournamentProgress) {
        queue.async { [weak self] in self?._progress = progress }
    }

    func snapshot() -> TournamentProgress? {
        queue.sync { _progress }
    }

    func clear() {
        queue.async { [weak self] in self?._progress = nil }
    }
}
