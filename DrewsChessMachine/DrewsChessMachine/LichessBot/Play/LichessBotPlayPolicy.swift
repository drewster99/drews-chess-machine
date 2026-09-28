import Foundation

/// The value head's view at one of our moves, as the play policy uses it.
struct LichessBotValueReading: Sendable, Equatable, Codable {
    let ply: Int
    let win: Float
    let draw: Float
    let loss: Float

    var expectedScore: Float {
        win + 0.5 * draw
    }
}

/// In-game decisions driven by the value head (plan §12.4). Pure functions
/// of settings and the readings from our own recent moves, so every
/// threshold, streak and minimum-ply edge is testable.
///
/// These decisions trust the value head's calibration, which is why each is
/// off by default; the Stats calibration chart on real games is the evidence
/// for turning them on.
enum LichessBotPlayPolicy {

    /// Resign when `p_loss` has been at or above the threshold for the last
    /// `resignConsecutiveMoves` of our moves, and the game is past
    /// `resignMinimumPly`.
    static func shouldResign(readings: [LichessBotValueReading], settings: LichessBotPlaySettings) -> Bool {
        guard settings.resignEnabled,
              let latest = readings.last,
              latest.ply >= settings.resignMinimumPly else {
            return false
        }
        return lastReadings(readings, count: settings.resignConsecutiveMoves)?
            .allSatisfy { $0.loss >= settings.resignLossProbability } ?? false
    }

    /// Offer a draw (riding on our move) when `p_draw` has been at or above
    /// the threshold for the last `offerDrawConsecutiveMoves` of our moves,
    /// past `offerDrawMinimumPly`.
    static func shouldOfferDraw(readings: [LichessBotValueReading], settings: LichessBotPlaySettings) -> Bool {
        guard settings.offerDrawEnabled,
              let latest = readings.last,
              latest.ply >= settings.offerDrawMinimumPly else {
            return false
        }
        return lastReadings(readings, count: settings.offerDrawConsecutiveMoves)?
            .allSatisfy { $0.draw >= settings.offerDrawProbability } ?? false
    }

    /// Accept the opponent's draw offer when our expected score in the
    /// current position is at or below the threshold. Declining costs
    /// nothing: making a move declines a pending offer on Lichess.
    static func shouldAcceptDraw(current: LichessBotValueReading, settings: LichessBotPlaySettings) -> Bool {
        settings.acceptDrawEnabled && current.expectedScore <= settings.acceptDrawExpectedScore
    }

    /// Accept a takeback while this game's accepted count is under the limit.
    static func shouldAcceptTakeback(acceptedSoFar: Int, settings: LichessBotPlaySettings) -> Bool {
        acceptedSoFar < settings.maxTakebacksAcceptedPerGame
    }

    /// The last `count` readings, or nil if there aren't that many yet.
    private static func lastReadings(_ readings: [LichessBotValueReading], count: Int) -> ArraySlice<LichessBotValueReading>? {
        guard count >= 1, readings.count >= count else { return nil }
        return readings.suffix(count)
    }
}
