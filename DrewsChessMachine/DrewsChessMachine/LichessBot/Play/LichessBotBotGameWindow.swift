import Foundation

/// Lichess' daily limit on a BOT account's games against other bots, the
/// way lila enforces it (`modules/bot/src/main/BotLimit.scala`: a
/// `RateLimit` of `LichessBotLimits.botGamesPerDay` per day, hit on every
/// game whose players are all bots). It is a fixed window, not a rolling
/// one: the first bot-vs-bot game that starts while no window is open opens
/// one lasting exactly 24 hours from that game; every bot-vs-bot game
/// started inside it counts; once the limit is reached, challenges between
/// DCM and another bot are refused until the window ends, when the whole
/// count clears at once. Games against humans never count.
///
/// Reconstructed from DCM's own game records, so it is an estimate: a game
/// the account played outside DCM, or before its records begin, is not
/// seen, and a record's `createdAt` is Lichess' creation time for the game,
/// a moment before lila counts its start.
struct LichessBotBotGameWindow: Sendable, Equatable {
    static let length: TimeInterval = 24 * 3600

    /// The moment this describes.
    let asOf: Date
    /// The open window's first game; nil when no window is open at `asOf`.
    let openedAt: Date?
    /// Bot games started in the open window; 0 when none is open. Can
    /// exceed the limit: a challenge accepted before the limit was reached
    /// still starts its game.
    let gamesCounted: Int

    /// When the open window ends and the count clears; nil when none is
    /// open.
    var closesAt: Date? { openedAt?.addingTimeInterval(Self.length) }

    /// The window open at `now`, from the start times of every bot-vs-bot
    /// game in DCM's records. lila resets the count only for a game started
    /// after the window's end (`nowMillis > clearAt`), so a game exactly at
    /// the end still counts in the old window.
    init(botGameStarts: [Date], now: Date) {
        asOf = now
        var openedAt: Date?
        var count = 0
        for start in botGameStarts.sorted() {
            if let opened = openedAt, start <= opened.addingTimeInterval(Self.length) {
                count += 1
            } else {
                openedAt = start
                count = 1
            }
        }
        if let opened = openedAt, now <= opened.addingTimeInterval(Self.length) {
            self.openedAt = opened
            self.gamesCounted = count
        } else {
            self.openedAt = nil
            self.gamesCounted = 0
        }
    }
}

/// How much of Lichess' daily bot-game limit each kind of outgoing challenge
/// may use. Games reserved for incoming challenges are left for bots that
/// challenge DCM; games reserved for the challenge queue are left for the
/// operator's queued challenges. Matchmaking stops before both reserves, the
/// queue before the incoming reserve. A challenge the operator sends by hand
/// may use the whole limit, which Lichess itself enforces.
enum LichessBotBotGameBudget {
    enum Sender: Sendable, Equatable {
        case matchmaking
        case challengeQueue
        case operatorChallenge
    }

    /// The bot games in one window `sender` may start challenges for.
    static func allowance(for sender: Sender, settings: LichessBotChallengeSettings) -> Int {
        let limit = LichessBotLimits.botGamesPerDay
        switch sender {
        case .matchmaking:
            return limit - settings.botGamesReservedForIncoming - settings.botGamesReservedForChallengeQueue
        case .challengeQueue:
            return limit - settings.botGamesReservedForIncoming
        case .operatorChallenge:
            return limit
        }
    }

    /// Why `sender` may not challenge another bot at `window.asOf`, or nil
    /// when it may.
    static func blockedReason(window: LichessBotBotGameWindow, sender: Sender, settings: LichessBotChallengeSettings, calendar: Calendar = .current) -> String? {
        let allowance = allowance(for: sender, settings: settings)
        guard window.gamesCounted >= allowance else { return nil }
        let limit = LichessBotLimits.botGamesPerDay
        var reserved: [String] = []
        if sender != .operatorChallenge && settings.botGamesReservedForIncoming > 0 {
            reserved.append("\(settings.botGamesReservedForIncoming) held for incoming challenges")
        }
        if sender == .matchmaking && settings.botGamesReservedForChallengeQueue > 0 {
            reserved.append("\(settings.botGamesReservedForChallengeQueue) for the challenge queue")
        }
        let reservedText = reserved.isEmpty ? "" : " (\(reserved.joined(separator: ", ")))"
        return "DCM has played \(window.gamesCounted) of Lichess' \(limit) daily bot games\(reservedText)\(resumeText(window, calendar: calendar))"
    }

    /// "; resumes about 11:34 AM" ("… 11:34 AM tomorrow" when the window
    /// ends on the next day of `calendar` after `window.asOf`); empty when
    /// no window is open (an allowance of zero blocks without one).
    static func resumeText(_ window: LichessBotBotGameWindow, calendar: Calendar = .current) -> String {
        guard let closesAt = window.closesAt else { return "" }
        let time = closesAt.formatted(Date.FormatStyle(date: .omitted, time: .shortened, timeZone: calendar.timeZone))
        return calendar.isDate(closesAt, inSameDayAs: window.asOf) ? "; resumes about \(time)" : "; resumes about \(time) tomorrow"
    }
}
