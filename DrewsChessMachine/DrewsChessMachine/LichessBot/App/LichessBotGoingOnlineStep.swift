import Foundation

/// What going online is doing while the connection reads Connecting
/// (follow-lineage plan §3.10). Going online loads the player notes, reads
/// and verifies the token and the account, builds the first model
/// generation, then opens the bot's streams; the model build can take
/// seconds, so the operator sees which step is running. A cancel names the
/// step it stopped after with the same value, so the status line and the
/// log speak one vocabulary.
enum LichessBotGoingOnlineStep: Equatable, Sendable {
    /// Loading the player notes and the challenge-outcome log the running
    /// bot consults.
    case loadingPlayerNotes
    /// Reading the stored token from the Keychain.
    case readingToken
    /// Taking the instance lock, so one bot runs per data folder.
    case takingInstanceLock
    /// Reopening the request gate if a breaker closed it.
    case checkingRequestGate
    /// Checking the token with Lichess.
    case verifyingToken
    /// Reading the bot account from Lichess.
    case readingAccount
    /// Building the first model generation for `source`; `detail` is the
    /// build's latest progress ("loading <file>", "building the network").
    case preparingModel(LichessBotModelSourceKind, detail: String)
    /// The model is built; the session's own state (one-game mode, a held
    /// rate limit) is being set.
    case startingSession
    /// Reading today's games from the records and the journals left from
    /// the last run, for the daily limits.
    case readingTodaysGames

    /// The operator-facing text, as the status chip and the Overview show it.
    var text: String {
        switch self {
        case .loadingPlayerNotes:
            return "Loading player notes"
        case .readingToken:
            return "Reading the token"
        case .takingInstanceLock:
            return "Taking the instance lock"
        case .checkingRequestGate:
            return "Checking the request gate"
        case .verifyingToken:
            return "Verifying the token"
        case .readingAccount:
            return "Reading the account"
        case .preparingModel(let source, let detail):
            return "Preparing model: \(source.displayName) — \(detail)"
        case .startingSession:
            return "Starting the session"
        case .readingTodaysGames:
            return "Reading today's games"
        }
    }

    /// The step as the log and a cancellation name it. The model step names
    /// its source but not the build's progress detail, which is the status
    /// line's.
    var logText: String {
        switch self {
        case .loadingPlayerNotes:
            return "loading player notes"
        case .readingToken:
            return "reading the token"
        case .takingInstanceLock:
            return "taking the instance lock"
        case .checkingRequestGate:
            return "checking the request gate"
        case .verifyingToken:
            return "verifying the token"
        case .readingAccount:
            return "reading the account"
        case .preparingModel(let source, _):
            return "preparing the model (\(source.displayName))"
        case .startingSession:
            return "starting the session"
        case .readingTodaysGames:
            return "reading today's games"
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
