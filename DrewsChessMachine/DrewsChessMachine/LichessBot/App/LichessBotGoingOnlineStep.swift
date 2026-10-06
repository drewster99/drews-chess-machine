import Foundation

/// What going online is doing while the connection reads Connecting
/// (follow-lineage plan §3.10). Going online verifies the token and account,
/// then builds the first model generation, then opens the bot's streams; the
/// model build can take seconds, so the operator sees which step is running.
enum LichessBotGoingOnlineStep: Equatable, Sendable {
    /// Reading the token and checking it, and the account, with Lichess.
    case verifyingAccount
    /// Building the first model generation for `source`; `detail` is the
    /// build's latest progress ("loading <file>", "building the network").
    case preparingModel(LichessBotModelSourceKind, detail: String)
    /// The model is built; the session (daily counts, the event stream, the
    /// poll loop) is starting.
    case startingSession

    var text: String {
        switch self {
        case .verifyingAccount:
            return "Verifying token"
        case .preparingModel(let source, let detail):
            return "Preparing model: \(source.displayName) — \(detail)"
        case .startingSession:
            return "Starting the session"
        }
    }
}

/// The last model refresh or source switch that failed while online, and
/// when the poll loop tries again. Cleared by the next success. The playing
/// generation keeps serving games meanwhile (follow-lineage plan §3.9,
/// §3.10).
struct LichessBotModelRefreshFailure: Equatable, Sendable {
    let text: String
    let retryAt: Date
}
