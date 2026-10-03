import Foundation

/// Hands out the serial numbers that name per-game random streams
/// (`selfplay.game.<serial>`, `vsuci.game.<serial>`), in the order games
/// start.
///
/// A game's serial picks its stream, so the serial must be assigned in a
/// fixed order: drivers call `next()` only from their serial (non-parallel)
/// passes, in slot order — never inside the parallel per-tick task group.
/// The counter outlives one driver: a Play-and-Train session that is stopped
/// and continued keeps its counter, so the continued run never reuses a
/// serial (and with it a game stream) it already played.
///
/// Locked with `SyncBox` (the project's `OSAllocatedUnfairLock` wrapper)
/// because the owner reads `nextSerial` from the main actor while a driver
/// task draws serials.
final class GameSerialCounter: @unchecked Sendable {
    private let box: SyncBox<Int>

    /// A counter whose first `next()` returns `firstSerial`.
    init(firstSerial: Int) {
        precondition(firstSerial >= 0, "GameSerialCounter: first serial must be non-negative; got \(firstSerial)")
        box = SyncBox(firstSerial)
    }

    /// The serial for the next game to start; each call returns the previous
    /// value plus one.
    func next() -> Int {
        box.mutate { serial -> Int in
            let current = serial
            serial += 1
            return current
        }
    }

    /// The serial the next game will get, without taking it.
    var nextSerial: Int { box.value }
}
