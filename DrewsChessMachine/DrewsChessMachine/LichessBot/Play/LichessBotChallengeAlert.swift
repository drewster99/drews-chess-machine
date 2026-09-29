import Foundation

/// Decides which tone, if any, announces an incoming challenge. Kept pure so
/// the decision is testable without the event stream or AppKit.
///
/// Every incoming challenge sounds, whatever the policy then does with it:
/// the tone says "someone wants a game", not "a game is starting". Our own
/// outgoing challenges are echoed back on the event stream and must stay
/// silent, or every matchmaking send would sound like an arrival.
enum LichessBotChallengeAlert {
    /// The Lichess title that marks a BOT account.
    static let botTitle = "BOT"

    /// Whether this challenge is our own outgoing one echoed back. Lichess
    /// omits `direction` on the echo, so the challenger being us is the test.
    static func isOwnOutgoingEcho(challengerID: String, ourAccountID: String) -> Bool {
        challengerID.lowercased() == ourAccountID.lowercased()
    }

    /// The system sound to play for a challenge arrival, or nil for none.
    static func soundName(
        challengerID: String,
        challengerTitle: String?,
        ourAccountID: String,
        alerts: LichessBotAlertSettings
    ) -> String? {
        if isOwnOutgoingEcho(challengerID: challengerID, ourAccountID: ourAccountID) {
            return nil
        }
        if challengerTitle == botTitle {
            return alerts.botChallengeSoundName
        }
        return alerts.humanChallengeSoundName
    }
}
