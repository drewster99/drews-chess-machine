import Foundation

/// The tabs of the Lichess bot's Settings screen, in the order they are
/// shown. Each tab owns a fixed set of `LichessBotSettings` fields — the
/// fields its sections edit — and that ownership is what lets a validation
/// problem be shown on the tab where it can be fixed.
///
/// Why ownership is by field and not by message: `validationProblems()`
/// returns plain sentences, and matching their wording would break silently
/// the first time a message is reworded. Instead each tab's fields from the
/// draft are laid over the settings in force (which are always valid — the
/// controller only ever accepts valid settings) and validated on their own.
/// Every rule in `validationProblems()` reads fields of one tab only, so a
/// tab's overlay has a problem exactly when one of its own fields does. A
/// future rule spanning two tabs would show its problem in the header but
/// mark neither tab; `LichessBotSettingsTabTests` pins today's mapping.
///
/// The account id is a `connection` field, but it is edited on the Account
/// tab, so it belongs to Account and not to Connection.
enum LichessBotSettingsTab: String, CaseIterable, Identifiable, Sendable {
    /// Incoming challenges and outgoing matchmaking.
    case games
    /// How the bot plays, and which model plays.
    case play
    case chat
    case alerts
    /// Connection pacing, plus the live-games display settings.
    case connection
    /// The Lichess account, its token, and the BOT upgrade.
    case account

    var id: String { rawValue }

    var title: String {
        switch self {
        case .games: return "Games"
        case .play: return "Play"
        case .chat: return "Chat"
        case .alerts: return "Alerts"
        case .connection: return "Connection"
        case .account: return "Account"
        }
    }

    /// The tab Settings opens on: Account while no token is saved — the bot
    /// can't go online until one is — otherwise the operator's last pick.
    /// While the token check is still pending this is the last pick; the
    /// view moves to Account if the check then finds no token.
    static func opening(tokenState: LichessBotController.TokenState, remembered: LichessBotSettingsTab) -> LichessBotSettingsTab {
        if case .none = tokenState {
            return .account
        }
        return remembered
    }

    /// `baseline` with the fields this tab edits taken from `draft`.
    func overlaying(fieldsFrom draft: LichessBotSettings, onto baseline: LichessBotSettings) -> LichessBotSettings {
        var result = baseline
        switch self {
        case .games:
            result.challenge = draft.challenge
            result.matchmaking = draft.matchmaking
        case .play:
            result.play = draft.play
            result.model = draft.model
        case .chat:
            result.chat = draft.chat
        case .alerts:
            result.alerts = draft.alerts
        case .connection:
            result.connection = draft.connection
            result.connection.expectedAccountID = baseline.connection.expectedAccountID
            result.display = draft.display
        case .account:
            result.connection.expectedAccountID = draft.connection.expectedAccountID
        }
        return result
    }

    /// The tabs, in display order, whose fields in `draft` fail validation.
    /// `baseline` must be valid settings — the ones in force.
    static func tabsWithProblems(in draft: LichessBotSettings, comparedWith baseline: LichessBotSettings) -> [LichessBotSettingsTab] {
        allCases.filter { tab in
            !tab.overlaying(fieldsFrom: draft, onto: baseline).validationProblems().isEmpty
        }
    }
}
