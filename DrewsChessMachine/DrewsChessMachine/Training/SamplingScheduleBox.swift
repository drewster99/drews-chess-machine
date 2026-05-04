import Foundation
import os

/// Lock-protected holder for the current self-play and arena
/// `SamplingSchedule` objects. Written by the SwiftUI edit fields
/// (on the main actor) and read by the `BatchedSelfPlayDriver`'s
/// slot tasks when constructing each new `MPSChessPlayer` pair, and
/// by the arena setup code when building arena players. Because
/// `MPSChessPlayer` captures its `SamplingSchedule` at init and the
/// player is reused across games within a slot, edits take effect
/// on newly-constructed players — i.e. at the next game-boundary
/// within the driver's slotLoop, not mid-game.
///
/// State protected by `OSAllocatedUnfairLock<State>`; each public
/// access is a single short critical section.
final class SamplingScheduleBox: @unchecked Sendable {
    private struct State {
        var selfPlay: SamplingSchedule
        var arena: SamplingSchedule
    }
    private let lock: OSAllocatedUnfairLock<State>

    init(selfPlay: SamplingSchedule, arena: SamplingSchedule) {
        self.lock = OSAllocatedUnfairLock(initialState: State(selfPlay: selfPlay, arena: arena))
    }

    var selfPlay: SamplingSchedule {
        lock.withLock { $0.selfPlay }
    }

    var arena: SamplingSchedule {
        lock.withLock { $0.arena }
    }

    func setSelfPlay(_ s: SamplingSchedule) {
        lock.withLock { $0.selfPlay = s }
    }

    func setArena(_ s: SamplingSchedule) {
        lock.withLock { $0.arena = s }
    }
}
