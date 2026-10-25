import Foundation

/// When DCM may challenge another bot, given Lichess' daily limit on a BOT
/// account's games against other bots (`LichessBotLimits.botGamesPerDay`).
///
/// Owner decision (2026-10-23, `LICHESS_BOT_OWN_BOT_LIMIT_PLAN.md`): the
/// limit is advisory on DCM's side. Lichess enforces it, and only Lichess's
/// refusal stops bot games: DCM never holds back because its own count says
/// it should, so it stays a little oversubscribed and plays as many bot
/// games as Lichess allows. Lichess's count isn't DCM's to track — it lives
/// in lila's memory, starts a day-long window at the first game it counts,
/// and is emptied by a lila restart (the plan's Task 2 finding) — so DCM
/// no longer replays it. DCM's own count is what it says: the bot games DCM
/// started today (`LichessBotAccountGameCounts.botGamesToday`), shown for
/// the operator, never used to block a send.
///
/// What remains here is Lichess's own word: once it refuses DCM for being
/// at its limit (`LichessBotOwnBotGameLimit`), DCM sends no bot challenge
/// until the time it gave. A bot's challenge to DCM is still accepted under
/// the ordinary rules (owner decision 2026-10-25): Lichess answers that
/// acceptance itself, and only its refusal stops the game.
enum LichessBotBotGameBudget {
    /// Lichess refused DCM for being at its daily bot-game limit: DCM sends
    /// no bot challenge until `until`, whoever asks it to.
    struct Block: Sendable, Equatable {
        let until: Date
        let reason: String
    }

    /// Lichess's own word that DCM is at its limit at `now`, for every
    /// sender (the operator's own challenges included: Lichess would refuse
    /// them). Nil when Lichess reported no limit in force at `now`.
    static func block(ownLimitUntil: Date?, now: Date, calendar: Calendar = .current) -> Block? {
        guard let until = ownLimitUntil, now < until else { return nil }
        return Block(until: until, reason: LichessBotOwnBotGameLimit.pausedText(until: until, now: now, calendar: calendar))
    }

    /// "11:34 AM", or "11:34 AM tomorrow" when `date` is on the next day of
    /// `calendar` after `now` (Lichess's limit ends at most a day ahead).
    static func clearTimeText(_ date: Date, now: Date, calendar: Calendar = .current) -> String {
        let time = date.formatted(Date.FormatStyle(date: .omitted, time: .shortened, timeZone: calendar.timeZone))
        return calendar.isDate(date, inSameDayAs: now) ? time : "\(time) tomorrow"
    }

    /// The Challenge sheet's one-line status: DCM's own count of today's
    /// bot games, and Lichess's pause when one is in force.
    static func statusText(counts: LichessBotAccountGameCounts.Reading, ownLimitUntil: Date?, now: Date, calendar: Calendar = .current) -> String {
        let today: String
        switch counts {
        case .loading: today = "DCM bot games today: loading"
        case .failed: today = "DCM bot games today: unavailable"
        case .counted(let counted): today = "DCM: \(counted.botGamesToday) bot games today"
        }
        guard let block = block(ownLimitUntil: ownLimitUntil, now: now, calendar: calendar) else { return today }
        return "\(today) · paused by Lichess until \(clearTimeText(block.until, now: now, calendar: calendar))"
    }
}
