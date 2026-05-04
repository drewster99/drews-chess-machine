import Foundation

/// Lock-protected holder for live tournament progress, shared between
/// the driver task (writer, one update per finished game) and the UI
/// heartbeat (reader, polling at 10 Hz). Backed by `SyncBox` (an
/// `OSAllocatedUnfairLock`); reads and writes are sub-microsecond
/// and never queue behind any other work.
final class TournamentLiveBox: @unchecked Sendable {
    private let _progress = SyncBox<TournamentProgress?>(nil)

    func update(_ progress: TournamentProgress) {
        _progress.value = progress
    }

    func snapshot() -> TournamentProgress? {
        _progress.value
    }

    /// Off-main async variant of `snapshot()`. Lock acquisition runs
    /// on a global executor so the awaiter is never synchronously
    /// blocked on `_progress.value`.
    func asyncSnapshot() async -> TournamentProgress? {
        await withCheckedContinuation { (cont: CheckedContinuation<TournamentProgress?, Never>) in
            DispatchQueue.global(qos: .userInitiated).async {
                cont.resume(returning: self.snapshot())
            }
        }
    }

    func clear() {
        _progress.value = nil
    }
}
