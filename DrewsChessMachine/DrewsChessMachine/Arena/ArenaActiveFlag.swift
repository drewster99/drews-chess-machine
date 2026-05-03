import Foundation

/// Lock-protected flag indicating an arena tournament is currently
/// in progress. Used to mutually exclude the Candidate test probe
/// from the arena, since both touch the candidate inference network
/// and the probe can't write while the arena is reading.
final class ArenaActiveFlag: @unchecked Sendable {
    private let queue = DispatchQueue(label: "drewschess.arenaactiveflag.serial")
    private var _active = false

    var isActive: Bool {
        queue.sync { _active }
    }

    func set() {
        queue.async { [weak self] in self?._active = true }
    }

    func clear() {
        queue.async { [weak self] in self?._active = false }
    }
}
