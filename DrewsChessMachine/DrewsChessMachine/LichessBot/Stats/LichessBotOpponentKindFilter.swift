import Foundation

/// Which games count by who DCM played: every game, only games against BOT
/// accounts, or only games against humans. One filter for the Record card
/// (beside its Rated / Casual Games filter, so the two combine: rated games
/// against humans) and the All Games window. Raw values are stable
/// identifiers — the Record card remembers the selection in the defaults.
///
/// The kind is the one every record files (`LichessBotGameRecord.Player.kind`,
/// copied to `LichessBotGameSummary.opponentKind`, never optional): Lichess'
/// own AI, then the title "BOT", else human. A game against Lichess' AI is
/// neither a bot account nor a human, so it counts only under All — the same
/// split as the period table, whose vs-bots and vs-humans columns leave it
/// out too. Nothing is guessed: there is no row without a kind.
enum LichessBotOpponentKindFilter: String, CaseIterable, Sendable, Codable, Hashable {
    case all
    case bots
    case humans

    var label: String {
        switch self {
        case .all: return "All opponents"
        case .bots: return "Bots"
        case .humans: return "Humans"
        }
    }

    /// The pickers' help, in one place so both say the same.
    static let helpText = "Count only games against BOT accounts, or only games against humans. Games against Lichess' own AI count only under All opponents."

    /// For a log line computed under this filter: nothing for every
    /// opponent, else " opponents=<raw value>".
    var logSuffix: String {
        switch self {
        case .all: return ""
        case .bots, .humans: return " opponents=\(rawValue)"
        }
    }

    /// The one opponent-kind predicate: the Record card's statistics, its
    /// Model menu and the All Games window all filter through it.
    func includes(_ row: LichessBotGameSummary) -> Bool {
        switch self {
        case .all:
            return true
        case .bots:
            switch row.opponentKind {
            case .bot: return true
            case .human, .lichessAI: return false
            }
        case .humans:
            switch row.opponentKind {
            case .human: return true
            case .bot, .lichessAI: return false
            }
        }
    }
}
