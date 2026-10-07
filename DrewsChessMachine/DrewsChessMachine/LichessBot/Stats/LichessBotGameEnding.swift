import Foundation

/// How a game ended, for the Endings pane (`LICHESS_BOT_RECORD_STATS_PLAN.md`
/// §3.9): one classification from Lichess's status and winner, DCM's score,
/// and the draw rule DCM's own engine saw in the final position.
///
/// Lichess's `draw` status covers agreement and every claimed draw, so the
/// local draw condition is what tells a threefold repetition from a
/// fifty-move claim or an agreement. A status this build does not know is
/// never folded into a known row: it becomes `.other` with Lichess's own
/// spelling.
enum LichessBotGameEnding: Sendable, Hashable, Comparable {
    case checkmate
    case resignation
    case timeForfeit
    /// `timeout` with a winner: a player left and the other claimed the win.
    case leftTheGame
    case stalemate
    case threefoldRepetition
    case fiftyMoveRule
    case insufficientMaterial
    case insufficientMaterialClaim
    /// `outoftime` without a winner: the flag fell with the opponent unable
    /// to mate.
    case timeoutVersusInsufficientMaterial
    /// `draw` with no draw rule seen locally: agreement, or a claim DCM's
    /// engine did not see (OD-18).
    case agreedOrOtherDraw
    /// `draw` in an index row without per-game facts (only a hand-written
    /// fixture): the local draw condition is unknown, so the draw is not
    /// guessed to be an agreement.
    case drawRuleNotRecorded
    /// Any other scored status, with Lichess's spelling.
    case other(status: String)
    /// `ourScore` nil: aborted, never started, unfinished.
    case notCounted

    /// Row label.
    var label: String {
        switch self {
        case .checkmate: return "Checkmate"
        case .resignation: return "Resignation"
        case .timeForfeit: return "Time forfeit"
        case .leftTheGame: return "Left the game"
        case .stalemate: return "Stalemate"
        case .threefoldRepetition: return "Threefold repetition"
        case .fiftyMoveRule: return "Fifty-move rule"
        case .insufficientMaterial: return "Insufficient material"
        case .insufficientMaterialClaim: return "Insufficient-material claim"
        case .timeoutVersusInsufficientMaterial: return "Timeout vs insufficient material"
        case .agreedOrOtherDraw: return "Agreed / other draw"
        case .drawRuleNotRecorded: return "Draw (rule not recorded)"
        case .other(let status): return "Other: \(status)"
        case .notCounted: return "Not counted"
        }
    }

    /// Display order: the table's order, unknown statuses alphabetically
    /// after the known rows, "Not counted" last.
    private var order: Int {
        switch self {
        case .checkmate: return 0
        case .resignation: return 1
        case .timeForfeit: return 2
        case .leftTheGame: return 3
        case .stalemate: return 4
        case .threefoldRepetition: return 5
        case .fiftyMoveRule: return 6
        case .insufficientMaterial: return 7
        case .insufficientMaterialClaim: return 8
        case .timeoutVersusInsufficientMaterial: return 9
        case .agreedOrOtherDraw: return 10
        case .drawRuleNotRecorded: return 11
        case .other: return 12
        case .notCounted: return 13
        }
    }

    static func < (lhs: LichessBotGameEnding, rhs: LichessBotGameEnding) -> Bool {
        if lhs.order != rhs.order {
            return lhs.order < rhs.order
        }
        if case .other(let left) = lhs, case .other(let right) = rhs {
            return left < right
        }
        return false
    }

    /// Classify one game. `localDrawCondition` is nil both when DCM's
    /// engine saw no draw rule and when it is unknown; `drawConditionKnown`
    /// tells them apart (false only for an index row without facts).
    static func classify(status: String, winner: String?, ourScore: Double?, localDrawCondition: ChessDrawCondition?, drawConditionKnown: Bool) -> LichessBotGameEnding {
        guard ourScore != nil else {
            return .notCounted
        }
        switch (LichessBotGameStatusName(rawValue: status), winner != nil) {
        case (.mate, _):
            return .checkmate
        case (.resign, _):
            return .resignation
        case (.outOfTime, true):
            return .timeForfeit
        case (.outOfTime, false):
            return .timeoutVersusInsufficientMaterial
        case (.timeout, true):
            return .leftTheGame
        case (.stalemate, _):
            return .stalemate
        case (.insufficientMaterialClaim, _):
            return .insufficientMaterialClaim
        case (.draw, _):
            guard drawConditionKnown else {
                return .drawRuleNotRecorded
            }
            switch localDrawCondition {
            case .threefoldRepetition: return .threefoldRepetition
            case .fiftyMoveRule: return .fiftyMoveRule
            case .insufficientMaterial: return .insufficientMaterial
            case .none: return .agreedOrOtherDraw
            }
        default:
            return .other(status: status)
        }
    }
}
