import Foundation

/// Historical record of a completed tournament, appended to the
/// `tournamentHistory` array after each arena finishes. Displayed in
/// the training stats text panel so the user can see promotion
/// decisions across the session.
struct TournamentRecord: Sendable, Identifiable {
    let id = UUID()
    /// `trainingStats.steps` at the moment the tournament finished.
    let finishedAtStep: Int
    /// Wall-clock time the tournament finished. `nil` for legacy
    /// records loaded from session files written before the field
    /// existed. Live arenas always populate it; the only nil case
    /// in normal operation is "resumed from an old `.dcmsession`".
    var finishedAt: Date? = nil
    /// Candidate's `ModelID` as it played in the arena (the trainer's
    /// snapshot ID at arena start). `nil` on legacy records loaded
    /// from session files written before per-record IDs were stored.
    var candidateID: ModelID? = nil
    /// Champion's `ModelID` as it played in the arena (the live
    /// champion's ID at arena start, *before* any promotion copy).
    /// `nil` on legacy records.
    var championID: ModelID? = nil
    /// Number of arena games that actually completed before the
    /// tournament ended. May be less than `Self.tournamentGames` if
    /// the user clicked Abort or Promote mid-tournament, or if the
    /// session was stopped while the arena was in flight.
    let gamesPlayed: Int
    let candidateWins: Int
    let championWins: Int
    let draws: Int
    let score: Double
    let promoted: Bool
    /// Whether the promotion (when `promoted == true`) was triggered
    /// by the configured score threshold (`.automatic`) or by the
    /// user's Promote button (`.manual`). `nil` when `promoted` is
    /// false.
    let promotionKind: PromotionKind?
    /// ID of the candidate network when a promotion happened — the
    /// model the champion was replaced with. `nil` when `promoted`
    /// is false, so the arena history / logs can surface "which
    /// network just became the champion" alongside the kept/PROMOTED
    /// marker. Captured at the moment of promotion, before any
    /// subsequent trainer re-mint can change the candidate's ID.
    let promotedID: ModelID?
    /// Total wall-clock time the tournament took from the initial
    /// trainer → candidate sync through the last game. Promotion copy
    /// after the score threshold check is not included.
    let durationSec: Double

    // Per-side W/L/D from the candidate's perspective. Populated
    // from `TournamentStats` (playerA = candidate at the arena call
    // site). Defaulted to 0 on older load paths that predate this
    // split — with no data we still want the display to render
    // (showing "—" for the side breakdown) rather than crash.
    let candidateWinsAsWhite: Int
    let candidateWinsAsBlack: Int
    let candidateLossesAsWhite: Int
    let candidateLossesAsBlack: Int
    let candidateDrawsAsWhite: Int
    let candidateDrawsAsBlack: Int

    /// AlphaZero-style score for candidate's white games only.
    /// 0 if the tournament was aborted before a white game finished.
    var candidateScoreAsWhite: Double {
        let n = candidateWinsAsWhite + candidateLossesAsWhite + candidateDrawsAsWhite
        guard n > 0 else { return 0 }
        return (Double(candidateWinsAsWhite) + 0.5 * Double(candidateDrawsAsWhite)) / Double(n)
    }

    /// AlphaZero-style score for candidate's black games only.
    var candidateScoreAsBlack: Double {
        let n = candidateWinsAsBlack + candidateLossesAsBlack + candidateDrawsAsBlack
        guard n > 0 else { return 0 }
        return (Double(candidateWinsAsBlack) + 0.5 * Double(candidateDrawsAsBlack)) / Double(n)
    }

    /// 95% CI summary computed from this record's W/D/L. Cached per
    /// call — downstream formatters can call this freely without
    /// worrying about recomputation cost (all scalar math).
    var eloSummary: ArenaEloStats.Summary {
        ArenaEloStats.summary(wins: candidateWins, draws: draws, losses: championWins)
    }
}
