import Foundation

/// Which board the Play and Train UI is showing. `.gameRun` is the
/// live self-play game (the only option before this feature existed);
/// `.candidateTest` swaps in the free-placement forward-pass editor so
/// the user can probe a fixed test position and watch the network's
/// evaluation of it drift as training progresses in the background;
/// `.progressRate` replaces the board with a line chart of rolling
/// moves/hr for self-play, training, and combined, sampled once per
/// second across the life of the session.
enum PlayAndTrainBoardMode: Sendable, Hashable {
    case gameRun
    case candidateTest
    case progressRate
}
