import Foundation

/// Lichess' daily limit on a BOT account's games against other bots,
/// replayed from DCM's records the way lila keeps it.
///
/// lila (`modules/bot/src/main/BotLimit.scala`) keeps one `RateLimit` entry
/// per account (`modules/memo/src/main/RateLimit.scala`: credits
/// `LichessBotLimits.botGamesPerDay`, duration one day) in a Caffeine cache
/// that drops an entry a day after it was last written
/// (`expireAfterWrite`). Every game whose players are all bots hits it once
/// when it starts:
/// - no entry (none yet, or expired): a new one, count 1, clearing a day later;
/// - count below the limit: count + 1, the clear time unchanged;
/// - count at the limit and the clear time passed: a new one, as above;
/// - otherwise (at the limit before the clear time): nothing is written.
/// The account is limited while its entry is at the limit and the clear time
/// has not passed (`isLimited`); lila then refuses bot-vs-bot challenges from
/// or to it, at creation and at acceptance.
///
/// Unlike "24 hours from the first game", an entry below the limit outlives
/// its clear time while games keep coming less than a day apart, and once
/// its clear time has passed it can never limit the account again: the game
/// after it reaches the limit replaces it. So only the count up to the clear
/// time can limit DCM; that is what `count(asOf:)` reports.
///
/// Reconstructed from DCM's own game records, so it is an estimate: a game
/// played outside DCM, or before its records begin, is not seen; a record's
/// `createdAt` is Lichess' creation time, a moment before lila counts the
/// start; and lila compares millisecond clocks (the cache's expiry a
/// different, monotonic one), so instants at a boundary may fall either side.
///
/// Above all, the cache lives in lila's memory: a lila restart (a deploy,
/// seen from DCM as the event stream dropping and HTTP 502s for a minute)
/// empties it, and the next bot game starts a fresh count. DCM cannot see
/// that, so after a restart this replay can be wrong either way: it counted
/// DCM limited on 2026-10-07 while Lichess let bot games start, and on
/// 2026-10-14 it counted 67 when Lichess refused DCM at 100 — replaying with
/// the cache emptied at the 2026-10-12 outages reproduces Lichess exactly
/// (plan `LICHESS_BOT_OWN_BOT_LIMIT_PLAN.md`). So when Lichess says DCM is
/// at its limit (`LichessBotOwnBotGameLimitRefusal`), that answer replaces
/// the replayed entry: `lichessReportedLimitUntil`.
struct LichessBotBotGameWindow: Sendable, Equatable {
    static let length: TimeInterval = 24 * 3600

    /// lila's entry for the account after the last game DCM knows of.
    struct Entry: Sendable, Equatable {
        /// Never above the limit: a game started while limited is not counted.
        let count: Int
        /// A day after the game that created the entry.
        let clearAt: Date
        /// The last game that created or counted in it.
        let lastWrittenAt: Date

        /// Caffeine tests `now - written >= duration`, so the entry is gone
        /// at this instant itself.
        var expiresAt: Date { lastWrittenAt.addingTimeInterval(LichessBotBotGameWindow.length) }

        func isPresent(at instant: Date) -> Bool { instant < expiresAt }

        /// lila resets only when `nowMillis > clearAt`, so the clear time
        /// itself still counts.
        func canLimit(at instant: Date) -> Bool { isPresent(at: instant) && instant <= clearAt }
    }

    /// The games that can still limit DCM, and when they stop counting.
    struct Count: Sendable, Equatable {
        let games: Int
        let clearsAt: Date
    }

    /// Nil when DCM's records hold no bot game (and Lichess reported no
    /// limit).
    let entry: Entry?

    /// When DCM's own limit ends by Lichess's latest refusal, if one was
    /// recorded.
    let lichessReportedLimitUntil: Date?

    /// Replays every bot-vs-bot game start, in any order, with no report
    /// from Lichess. The result does not depend on the clock; `count(asOf:)`
    /// reads it at an instant no earlier than the last start.
    init(botGameStarts: [Date]) {
        self.init(botGameStarts: botGameStarts, lichessReportedLimitUntil: nil)
    }

    /// Replays every bot-vs-bot game start, with Lichess's word that DCM is
    /// at its limit until `lichessReportedLimitUntil` taking the place of
    /// whatever the replay holds then: lila's entry was at the limit,
    /// clearing at that time. Games at or before it change nothing (lila
    /// counts none while limited, and the reported count already holds the
    /// earlier ones); the first game after it starts a new count, as lila's
    /// does.
    init(botGameStarts: [Date], lichessReportedLimitUntil: Date?) {
        var entry: Entry?
        var reportApplied = lichessReportedLimitUntil == nil
        for start in botGameStarts.sorted() {
            if !reportApplied, let until = lichessReportedLimitUntil, start > until {
                entry = Self.reportedEntry(until: until, replayed: entry)
                reportApplied = true
            }
            entry = Self.entry(afterGameAt: start, from: entry)
        }
        if !reportApplied, let until = lichessReportedLimitUntil {
            entry = Self.reportedEntry(until: until, replayed: entry)
        }
        self.entry = entry
        self.lichessReportedLimitUntil = lichessReportedLimitUntil
    }

    /// lila's entry as Lichess reported it. The game that brought it to the
    /// limit was written after the entry was created a day before `until`,
    /// so the entry is present until `until` at least; the replay's own
    /// last write stands when it is later.
    private static func reportedEntry(until: Date, replayed: Entry?) -> Entry {
        let earliestWrite = until.addingTimeInterval(-length)
        let lastWrite = replayed.map { max($0.lastWrittenAt, earliestWrite) } ?? earliestWrite
        return Entry(count: LichessBotLimits.botGamesPerDay, clearAt: until, lastWrittenAt: lastWrite)
    }

    /// When DCM's own limit ends, if Lichess reported one still in force at
    /// `instant`. Lichess's answer, so it wins over every estimate.
    func lichessReportedLimit(asOf instant: Date) -> Date? {
        guard let until = lichessReportedLimitUntil, instant < until else { return nil }
        return until
    }

    /// lila's `RateLimit.apply` for one game.
    private static func entry(afterGameAt start: Date, from entry: Entry?) -> Entry {
        let opened = Entry(count: 1, clearAt: start.addingTimeInterval(length), lastWrittenAt: start)
        guard let entry, entry.isPresent(at: start) else { return opened }
        if entry.count < LichessBotLimits.botGamesPerDay {
            return Entry(count: entry.count + 1, clearAt: entry.clearAt, lastWrittenAt: start)
        }
        if start > entry.clearAt {
            return opened
        }
        return entry
    }

    /// The count that can limit DCM at `instant`; nil when none can.
    func count(asOf instant: Date) -> Count? {
        guard let entry, entry.canLimit(at: instant) else { return nil }
        return Count(games: entry.count, clearsAt: entry.clearAt)
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

    /// Why a sender may not challenge another bot now.
    enum Block: Sendable, Equatable {
        /// DCM's game records have not loaded, so nothing is known yet.
        case recordsNotLoaded
        /// Games already counted use the allowance (or the reserves take all
        /// of it): it lasts until the count clears.
        case allowanceUsed(reason: String)
        /// Only with games that may still start counted: it lifts as they
        /// are answered or start.
        case awaitingGamesThatMayStart(reason: String)
        /// Lichess refused DCM for being at its limit: no bot game may start
        /// until `until`, whoever sends, and whatever DCM's estimate says.
        case lichessReportedLimit(until: Date, reason: String)

        var reason: String {
            switch self {
            case .recordsNotLoaded: return LichessBotBotGameBudget.recordsNotLoadedReason
            case .allowanceUsed(let reason), .awaitingGamesThatMayStart(let reason), .lichessReportedLimit(_, let reason): return reason
            }
        }
    }

    /// Lichess's own word that DCM is at its limit at `now`, for every
    /// sender (the operator's own challenges included: Lichess would refuse
    /// them). Nil when Lichess reported no limit in force.
    static func lichessReportedBlock(window: LichessBotBotGameWindow?, now: Date, calendar: Calendar = .current) -> Block? {
        guard let until = window?.lichessReportedLimit(asOf: now) else { return nil }
        return .lichessReportedLimit(until: until, reason: LichessBotOwnBotGameLimit.pausedText(until: until, now: now, calendar: calendar))
    }

    static let recordsNotLoadedReason = "DCM's game records have not loaded"

    /// The bot games that count toward Lichess' limit which `sender` may use.
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

    /// Why `sender` may not challenge another bot at `now`, or nil when it
    /// may. `prospectiveBotGames` are games Lichess has not counted yet but
    /// may (DCM's challenges to bots unanswered or being sent, games
    /// starting): the reserves must hold if they all start.
    static func block(window: LichessBotBotGameWindow?, prospectiveBotGames: Int, sender: Sender, settings: LichessBotChallengeSettings, now: Date, calendar: Calendar = .current) -> Block? {
        guard let window else { return .recordsNotLoaded }
        if let reported = lichessReportedBlock(window: window, now: now, calendar: calendar) {
            return reported
        }
        let allowance = allowance(for: sender, settings: settings)
        let count = window.count(asOf: now)
        let counted = count?.games ?? 0 // nil means no game can limit DCM: none count
        guard counted + prospectiveBotGames >= allowance else { return nil }
        var reserved: [String] = []
        if sender != .operatorChallenge && settings.botGamesReservedForIncoming > 0 {
            reserved.append("\(settings.botGamesReservedForIncoming) held for incoming challenges")
        }
        if sender == .matchmaking && settings.botGamesReservedForChallengeQueue > 0 {
            reserved.append("\(settings.botGamesReservedForChallengeQueue) for the challenge queue")
        }
        let reservedText = reserved.isEmpty ? "" : " (\(reserved.joined(separator: ", ")))"
        let mayStart = prospectiveBotGames > 0 ? " and \(prospectiveBotGames) more may start" : ""
        let base = "DCM has used \(counted) of Lichess' \(LichessBotLimits.botGamesPerDay) daily bot games\(mayStart)\(reservedText)"
        let resume: String
        if allowance == 0 {
            resume = ""
        } else if let count, prospectiveBotGames < allowance {
            let time = clearTimeText(count.clearsAt, now: now, calendar: calendar)
            resume = count.games >= allowance ? "; resumes about \(time)" : "; resumes about \(time), or sooner if enough of those don't start"
        } else if let count, count.games >= allowance {
            resume = "; resumes once those have resolved, not before about \(clearTimeText(count.clearsAt, now: now, calendar: calendar))"
        } else {
            resume = "; resumes as those resolve"
        }
        if allowance == 0 || counted >= allowance {
            return .allowanceUsed(reason: base + resume)
        }
        return .awaitingGamesThatMayStart(reason: base + resume)
    }

    /// "11:34 AM", or "11:34 AM tomorrow" when `date` is on the next day of
    /// `calendar` after `now` (a count clears at most a day ahead).
    static func clearTimeText(_ date: Date, now: Date, calendar: Calendar = .current) -> String {
        let time = date.formatted(Date.FormatStyle(date: .omitted, time: .shortened, timeZone: calendar.timeZone))
        return calendar.isDate(date, inSameDayAs: now) ? time : "\(time) tomorrow"
    }

    /// The Challenge sheet's one-line status.
    static func statusText(window: LichessBotBotGameWindow?, now: Date, calendar: Calendar = .current) -> String {
        guard let window else { return "DCM bot games: loading" }
        if let until = window.lichessReportedLimit(asOf: now) {
            return "DCM: \(LichessBotLimits.botGamesPerDay)/\(LichessBotLimits.botGamesPerDay) bot games (Lichess) · clears \(clearTimeText(until, now: now, calendar: calendar))"
        }
        guard let count = window.count(asOf: now) else { return "DCM: 0/\(LichessBotLimits.botGamesPerDay) bot games" }
        return "DCM: \(count.games)/\(LichessBotLimits.botGamesPerDay) bot games · clears \(clearTimeText(count.clearsAt, now: now, calendar: calendar))"
    }
}
