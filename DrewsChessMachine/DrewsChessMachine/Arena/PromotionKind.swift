import Foundation

/// How a tournament ended in terms of the promotion decision.
/// `nil` on a `TournamentRecord` means the champion was kept (no
/// promotion); otherwise this distinguishes an automatic promotion
/// (score met the configured threshold and a full tournament was
/// played) from a manual one (user clicked the Promote button,
/// which forces promotion regardless of score or completion).
enum PromotionKind: String, Sendable, Codable {
    case automatic
    case manual
}
