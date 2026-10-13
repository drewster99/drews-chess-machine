import Foundation

/// The alert that announces a new game against a human that began from an
/// accepted challenge (owner request 2026-10-13). Kept pure, like
/// `LichessBotChallengeAlert`, so which alert a game gets is testable
/// without the event stream or AppKit.
///
/// Which way the challenge went comes only from the game's origin
/// (`LichessBotGameOrigin`), the one record of how a game began:
/// - `acceptedIncomingChallenge` — someone challenged DCM and the
///   challenge was accepted (by the policy, or by the operator on
///   lichess.org): `weAcceptedHumanChallenge`.
/// - `outgoingChallengeAccepted`, whatever its sender (the challenge
///   sheet, the challenge queue, a casual resend, matchmaking — which today
///   challenges only bots, but would count if it ever challenged a human):
///   `humanAcceptedOurChallenge`.
/// - `outgoingChallengeSenderNotRecorded` — DCM's challenge, proven by its
///   echo, sent by an earlier run or another client: the human still
///   accepted DCM's challenge, so `humanAcceptedOurChallenge`.
/// - `tournament` and `undetermined`: no challenge was accepted (or none is
///   known to have been), so no alert.
///
/// Whether the opponent is human is `LichessBotOpponentKind.of`'s rule, the
/// one the live game and the filed record use; Lichess's AI and BOT
/// accounts never alert.
enum LichessBotGameStartAlert: Sendable, Equatable, CaseIterable {
    /// A human accepted a challenge DCM sent.
    case humanAcceptedOurChallenge
    /// DCM accepted a challenge a human sent.
    case weAcceptedHumanChallenge

    /// The alert for a game with this origin and opponent, or nil for none.
    static func alert(origin: LichessBotGameOrigin, opponent: LichessBotOpponentKind) -> LichessBotGameStartAlert? {
        guard opponent == .human else { return nil }
        switch origin {
        case .acceptedIncomingChallenge:
            return .weAcceptedHumanChallenge
        case .outgoingChallengeAccepted, .outgoingChallengeSenderNotRecorded:
            return .humanAcceptedOurChallenge
        case .tournament, .undetermined:
            return nil
        }
    }

    /// This alert's sound setting.
    func sound(in alerts: LichessBotAlertSettings) -> LichessBotRepeatingAlertSound {
        switch self {
        case .humanAcceptedOurChallenge: return alerts.humanAcceptedOurChallengeSound
        case .weAcceptedHumanChallenge: return alerts.weAcceptedHumanChallengeSound
        }
    }

    /// How the session log names the alert.
    var logDescription: String {
        switch self {
        case .humanAcceptedOurChallenge: return "a human accepted our challenge"
        case .weAcceptedHumanChallenge: return "we accepted a human's challenge"
        }
    }
}

/// Decides, once per game, whether a game-start alert plays. Pure: the
/// controller feeds it session starts and ends, the game's origin and its
/// opponent's kind, and plays what it returns.
///
/// Why a tracker and not one check at session start: the two facts the
/// alert needs arrive separately and in either order. The origin may be
/// decided at session start or later (the challenge log still loading, the
/// POST racing the game start); the opponent is known only once `gameFull`
/// names the players. The alert is decided when the second fact arrives —
/// so it may sound a moment after the game starts, never before it is
/// certain — and the game is then settled, so later repeats of either fact
/// (a re-sent `gameFull` on a stream resync, a later session for the same
/// game) never sound it again.
///
/// Only a game this runtime is the first to see (`LichessBotSessionOrigin.new`)
/// can alert: a game resumed after a relaunch or reconnect started earlier,
/// and was announced then if at all. A session that ends before both facts
/// are known settles its game silently.
///
/// Settled game ids are kept for the life of the controller (one short
/// string per game played; a few thousand a day at most), so a game whose
/// session restarts within the same launch is never announced twice.
struct LichessBotGameStartAlertTracker: Sendable {
    private struct Facts: Sendable {
        var origin: LichessBotGameOrigin?
        var opponent: LichessBotOpponentKind?
    }

    /// Games that may still alert, with what is known so far.
    private var waiting: [String: Facts] = [:]
    /// Games decided (alerted or not), never to alert again.
    private var settled: Set<String> = []

    init() {}

    /// Whether `gameID` may still alert.
    func isWaiting(_ gameID: String) -> Bool {
        waiting[gameID] != nil
    }

    /// A game session started. `isNewGame` is false for a resumed game,
    /// which never alerts.
    /// A resumed session settles the game even while it is still waiting:
    /// the facts a previous session left (an opponent known before the bot
    /// went offline) must not complete into an alert for a game that was
    /// already under way.
    mutating func sessionStarted(gameID: String, isNewGame: Bool) {
        guard !settled.contains(gameID) else { return }
        if !isNewGame {
            waiting[gameID] = nil
            settled.insert(gameID)
            return
        }
        guard waiting[gameID] == nil else { return }
        waiting[gameID] = Facts()
    }

    /// The game's origin is known: the alert to play now, if this completes
    /// the facts and the game earns one.
    mutating func originKnown(_ origin: LichessBotGameOrigin, gameID: String) -> LichessBotGameStartAlert? {
        guard var facts = waiting[gameID] else { return nil }
        facts.origin = origin
        waiting[gameID] = facts
        return decideIfComplete(gameID: gameID)
    }

    /// The game's opponent is known: the alert to play now, if this
    /// completes the facts and the game earns one.
    mutating func opponentKnown(_ opponent: LichessBotOpponentKind, gameID: String) -> LichessBotGameStartAlert? {
        guard var facts = waiting[gameID] else { return nil }
        facts.opponent = opponent
        waiting[gameID] = facts
        return decideIfComplete(gameID: gameID)
    }

    /// The game's session ended: a game still waiting is settled without an
    /// alert, since it is no longer starting.
    mutating func sessionEnded(gameID: String) {
        if waiting.removeValue(forKey: gameID) != nil {
            settled.insert(gameID)
        }
    }

    private mutating func decideIfComplete(gameID: String) -> LichessBotGameStartAlert? {
        guard let facts = waiting[gameID], let origin = facts.origin, let opponent = facts.opponent else { return nil }
        waiting[gameID] = nil
        settled.insert(gameID)
        return LichessBotGameStartAlert.alert(origin: origin, opponent: opponent)
    }
}
