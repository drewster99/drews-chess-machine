import Foundation

/// Which game the Live tab's single view follows while the operator hasn't
/// picked one (owner request 2026-10-13: "prioritize showing [games vs
/// humans] in the tab where we just show 1 game"). Pure, so the rule is
/// tested without a runtime; `LichessBotController.updateAutoFollow` applies
/// it and keeps the hold on a finished game, which this type knows nothing
/// about.
///
/// The rule:
/// - With nothing followed, follow the first-started live game against a
///   human, else the first-started live game.
/// - A followed live game against a bot (or an opponent not known yet)
///   gives way to a live game against a human: the operator wants to watch
///   people play DCM, and a human game is rarer than a bot game.
/// - A followed live game against a human is kept, whatever starts; two
///   human games never trade places, so the view doesn't jump between them.
/// - A followed live bot game is otherwise kept: a new bot game never
///   takes the view.
///
/// Every switch is toward a human game and a human game is never left while
/// it is live, so the view can't flip back and forth from tick to tick.
enum LichessBotAutoFollowChoice {

    /// One live game, in start order.
    struct Candidate: Equatable, Sendable {
        let id: String
        /// False for a bot, Lichess's AI, or an opponent not known yet
        /// (`LichessBotLiveGame.opponentIsHuman`).
        let isHuman: Bool
    }

    /// The game to follow when none is: the first-started human game, else
    /// the first-started game. `live` is in start order.
    static func initial(among live: [Candidate]) -> String? {
        (live.first { $0.isHuman } ?? live.first)?.id
    }

    /// The game to follow while `followed` is still live: a human game that
    /// takes the view from a non-human one, else `followed` itself.
    static func next(following followed: Candidate, among live: [Candidate]) -> String {
        guard !followed.isHuman, let human = live.first(where: { $0.isHuman }) else {
            return followed.id
        }
        return human.id
    }
}
