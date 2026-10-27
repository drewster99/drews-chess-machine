import Foundation

/// Why the bot is taking no new games.
enum LichessBotNewGameHoldReason: String, Sendable, CaseIterable {
    /// One 429: no new games until a while after its cooldown ends.
    case rateLimit
    /// The rate-limit breaker: repeated 429s.
    case rateLimitBreaker
    case lostOnTime
    case movePostFailures
    case reconnectStorm
    /// A game session stopped moving: DCM and Lichess disagree about a game.
    case gameStoppedMoving

    var label: String {
        switch self {
        case .rateLimit:
            return "rate limited (HTTP 429)"
        case .rateLimitBreaker:
            return "rate-limit breaker"
        case .lostOnTime:
            return "lost-on-time breaker"
        case .movePostFailures:
            return "move-post failure breaker"
        case .reconnectStorm:
            return "reconnect-storm breaker"
        case .gameStoppedMoving:
            return "a game stopped moving"
        }
    }
}

/// When a hold on new games ends.
enum LichessBotNewGameHoldEnd: Sendable, Equatable {
    case at(Date)
    /// Only the operator going online again ends it; the bot finishes its
    /// games and then goes offline.
    case whenOperatorGoesOnline
}

/// The holds on new games, one per reason (owner decisions 2026-10-26,
/// `LICHESS_BOT_CIRCUIT_BREAKERS_PLAN.md`).
///
/// Every circuit breaker, and a single 429, stops *new* games — incoming
/// challenges are declined `later`, nothing outgoing is sent, matchmaking
/// pauses — and never touches the games in progress: no automatic action
/// abandons a game. This is the one record of what is holding and until
/// when; the controller owns it and derives the manager's flag, the
/// connection state and the UI from it.
///
/// Holding a reason again keeps the later end, so a short hold (a lone 429)
/// never cuts a longer one (a breaker) short, and an operator hold outlasts
/// any date.
struct LichessBotNewGameHolds: Sendable, Equatable {
    private(set) var ends: [LichessBotNewGameHoldReason: LichessBotNewGameHoldEnd] = [:]

    var isHolding: Bool {
        !ends.isEmpty
    }

    /// Whether only the operator can end the hold: the bot goes offline once
    /// its games finish.
    var requiresOperator: Bool {
        ends.values.contains(.whenOperatorGoesOnline)
    }

    /// Hold new games for `reason` until `end`, or longer if it already
    /// holds for longer.
    mutating func hold(_ reason: LichessBotNewGameHoldReason, until end: LichessBotNewGameHoldEnd) {
        switch (ends[reason], end) {
        case (.whenOperatorGoesOnline?, _):
            return
        case (.at(let existing)?, .at(let new)) where existing >= new:
            return
        default:
            ends[reason] = end
        }
    }

    /// The reasons holding now, in declaration order.
    var reasons: [LichessBotNewGameHoldReason] {
        LichessBotNewGameHoldReason.allCases.filter { ends[$0] != nil }
    }

    /// Ends the dated holds that are over at `now`; returns their reasons.
    mutating func expire(now: Date) -> [LichessBotNewGameHoldReason] {
        let ended = reasons.filter { reason in
            guard case .at(let date) = ends[reason] else { return false }
            return date <= now
        }
        for reason in ended {
            ends[reason] = nil
        }
        return ended
    }

    /// The operator went online: holds only the operator could end are over;
    /// dated holds still running stay (a 429's hold outlives going offline
    /// and back). Returns the reasons ended.
    mutating func operatorWentOnline() -> [LichessBotNewGameHoldReason] {
        let ended = reasons.filter { ends[$0] == .whenOperatorGoesOnline }
        for reason in ended {
            ends[reason] = nil
        }
        return ended
    }

    /// One line naming every hold and its end, or nil when nothing holds.
    func summary(formatTime: (Date) -> String) -> String? {
        guard isHolding else { return nil }
        return reasons.map { reason in
            switch ends[reason] {
            case .at(let date)?:
                return "\(reason.label) until \(formatTime(date))"
            case .whenOperatorGoesOnline?, nil:
                return "\(reason.label) until you go online"
            }
        }.joined(separator: "; ")
    }
}
