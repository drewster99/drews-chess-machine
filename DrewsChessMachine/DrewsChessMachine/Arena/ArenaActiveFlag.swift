import Foundation

/// Lock-protected flag indicating an arena tournament is currently
/// in progress. Used to mutually exclude the Candidate test probe
/// from the arena, since both touch the candidate inference network
/// and the probe can't write while the arena is reading.
///
/// Backed by `SyncBox<Bool>` (an `OSAllocatedUnfairLock`); reads
/// and writes are sub-microsecond.
final class ArenaActiveFlag: @unchecked Sendable {
    private let _active = SyncBox<Bool>(false)

    var isActive: Bool {
        _active.value
    }

    func set() {
        _active.value = true
    }

    func clear() {
        _active.value = false
    }
}
