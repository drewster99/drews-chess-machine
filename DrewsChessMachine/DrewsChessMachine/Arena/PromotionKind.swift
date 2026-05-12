import Foundation

/// How a champion replacement happened, recorded on the
/// `TournamentRecord` it produced. `nil` on a record means the
/// champion was kept (no promotion). `.automatic` = an arena
/// tournament ran to completion and the candidate's score met
/// `arenaPromoteThreshold`. `.manual` = the user invoked
/// `Engine ▸ Promote Trainee Now` (`SessionController.promoteTrainerNow()`),
/// which promotes the current trainer weights into the champion with
/// no arena and no score gate; such records carry `gamesPlayed == 0`.
enum PromotionKind: String, Sendable, Codable {
    case automatic
    case manual
}
