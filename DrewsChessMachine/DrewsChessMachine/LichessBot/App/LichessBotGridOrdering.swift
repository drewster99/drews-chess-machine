import Foundation

/// The live grid's ordering and each tile's post-game phase, as pure
/// functions of the games' start and finish times and a clock value.
///
/// A game that just ended keeps its place among the live games for
/// `positionHold`, so the tile the operator was watching doesn't jump away
/// the instant the result arrives; after that the grid reshuffles it into
/// the finished section. Separately, its tile shows the result's color for
/// `resultHighlight` before settling into the ordinary game-over gray.
/// Both follow from `finishedAt` alone, so the controller only has to
/// advance its clock at the moments these windows close.
enum LichessBotGridOrdering {
    /// How long a finished game keeps its live-section grid position.
    static let positionHold: TimeInterval = 5
    /// How long a finished game's tile shows its result color.
    static let resultHighlight: TimeInterval = 10

    /// The ordering inputs for one game.
    struct Entry: Equatable {
        let id: String
        let startedAt: Date
        let finishedAt: Date?
    }

    /// A tile's styling phase.
    enum Phase: Equatable {
        case live
        /// Finished within the result-highlight window.
        case justFinished
        case finished
    }

    static func phase(finishedAt: Date?, now: Date) -> Phase {
        guard let finishedAt else { return .live }
        return now.timeIntervalSince(finishedAt) < resultHighlight ? .justFinished : .finished
    }

    /// Whether the game still sorts among the live games: unfinished, or
    /// finished within the position hold. A finish time later than `now`
    /// (the clock hasn't advanced since the game ended) counts as within.
    static func sortsAsLive(finishedAt: Date?, now: Date) -> Bool {
        guard let finishedAt else { return true }
        return now.timeIntervalSince(finishedAt) < positionHold
    }

    /// Positions into `entries` in grid order: live (and held) games
    /// first, oldest start first — the order they began; then finished
    /// games, most recently finished first. Ties fall back to the id so the
    /// order is total and never flickers.
    static func orderedIndices(_ entries: [Entry], now: Date) -> [Int] {
        var live: [Int] = []
        var finished: [(index: Int, finishedAt: Date)] = []
        for (index, entry) in entries.enumerated() {
            if let finishedAt = entry.finishedAt, !sortsAsLive(finishedAt: finishedAt, now: now) {
                finished.append((index, finishedAt))
            } else {
                live.append(index)
            }
        }
        live.sort { (entries[$0].startedAt, entries[$0].id) < (entries[$1].startedAt, entries[$1].id) }
        finished.sort { lhs, rhs in
            lhs.finishedAt != rhs.finishedAt ? lhs.finishedAt > rhs.finishedAt : entries[lhs.index].id < entries[rhs.index].id
        }
        return live + finished.map(\.index)
    }

    /// The earliest moment after `now` at which some game's position hold
    /// or result highlight ends, or nil when none is pending.
    static func nextTransition(finishTimes: [Date], after now: Date) -> Date? {
        finishTimes
            .flatMap { [$0.addingTimeInterval(positionHold), $0.addingTimeInterval(resultHighlight)] }
            .filter { $0 > now }
            .min()
    }

    /// Whether some game's window closed in `(since, now]`, so a clock last
    /// set at `since` is stale.
    static func hasTransition(finishTimes: [Date], since: Date, through now: Date) -> Bool {
        finishTimes
            .flatMap { [$0.addingTimeInterval(positionHold), $0.addingTimeInterval(resultHighlight)] }
            .contains { $0 > since && $0 <= now }
    }
}
