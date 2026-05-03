import Foundation

/// Lock-protected holder for the current self-play and arena
/// `SamplingSchedule` objects. Written by the SwiftUI edit fields
/// (on the main actor) and read by the `BatchedSelfPlayDriver`'s
/// slot tasks when constructing each new `MPSChessPlayer` pair, and
/// by the arena setup code when building arena players. Because
/// `MPSChessPlayer` captures its `SamplingSchedule` at init and the
/// player is reused across games within a slot, edits take effect
/// on newly-constructed players — i.e. at the next game-boundary
/// within the driver's slotLoop, not mid-game.
final class SamplingScheduleBox: @unchecked Sendable {
    private let queue = DispatchQueue(label: "drewschess.samplingschedulebox.serial")
    private var _selfPlay: SamplingSchedule
    private var _arena: SamplingSchedule

    init(selfPlay: SamplingSchedule, arena: SamplingSchedule) {
        self._selfPlay = selfPlay
        self._arena = arena
    }

    var selfPlay: SamplingSchedule {
        queue.sync { _selfPlay }
    }

    var arena: SamplingSchedule {
        queue.sync { _arena }
    }

    func setSelfPlay(_ s: SamplingSchedule) {
        queue.sync { _selfPlay = s }
    }

    func setArena(_ s: SamplingSchedule) {
        queue.sync { _arena = s }
    }
}
