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
        let start = Date()
        return await withCheckedContinuation { (cont: CheckedContinuation<TournamentProgress?, Never>) in
            let inContinuation = Date()
            DispatchQueue.global(qos: .userInitiated).async {
                let dispatched = Date()
                let result = self.snapshot()
                let now = Date()
                let total = now.timeIntervalSince(start)
                if total > 0.05 {
                    let pre = inContinuation.timeIntervalSince(start)
                    let queue = dispatched.timeIntervalSince(inContinuation)
                    let work = now.timeIntervalSince(dispatched)
                    print(String(format: "[DISPATCH-LATENCY] TournamentLiveBox.asyncSnapshot: total=%.2fms (pre=%.2fms queue=%.2fms work=%.2fms)", total*1000, pre*1000, queue*1000, work*1000))
                }
                cont.resume(returning: result)
            }
        }
    }

    func clear() {
        _progress.value = nil
    }
}
