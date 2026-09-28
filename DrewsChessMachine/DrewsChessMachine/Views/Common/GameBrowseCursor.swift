import Foundation

/// Which position a game view shows, independent of the game itself (plan
/// §14.3a). Browsing never changes the game: it only chooses how many of
/// the game's plies the view applies. `nil` means "follow the live
/// position", which is also where stepping past the last ply returns to.
///
/// A position is named by its ply count: 0 is the start, `n` is the
/// position after the first `n` plies.
struct GameBrowseCursor: Equatable, Sendable {
    /// The ply count being viewed, or nil when following live.
    private(set) var viewedPlyCount: Int?

    init() {}

    var isLive: Bool {
        viewedPlyCount == nil
    }

    /// The ply count to display for a game currently `totalPlies` long. A
    /// takeback can shrink the game under a browsed position; the view then
    /// shows the last position that still exists.
    func displayedPlyCount(totalPlies: Int) -> Int {
        min(viewedPlyCount ?? totalPlies, totalPlies)
    }

    mutating func stepBack(totalPlies: Int) {
        let current = displayedPlyCount(totalPlies: totalPlies)
        guard current > 0 else { return }
        viewedPlyCount = current - 1
    }

    mutating func stepForward(totalPlies: Int) {
        guard let viewed = viewedPlyCount else { return }
        select(plyCount: viewed + 1, totalPlies: totalPlies)
    }

    mutating func goToStart(totalPlies: Int) {
        select(plyCount: 0, totalPlies: totalPlies)
    }

    mutating func goLive() {
        viewedPlyCount = nil
    }

    /// View the position after `plyCount` plies. Reaching the current end of
    /// the game means following live again.
    mutating func select(plyCount: Int, totalPlies: Int) {
        viewedPlyCount = plyCount >= totalPlies ? nil : max(0, plyCount)
    }
}
