import Foundation

/// When the bot's request connections must be replaced (owner request
/// 2026-10-26, `LICHESS_BOT_REQUEST_FAILURE_RECOVERY_PLAN.md`).
///
/// On 2026-10-26 every non-stream request failed for three minutes with
/// `NSURLErrorDomain -1001` while the event stream, on its own session, kept
/// its keep-alives. Each request timed out on its own and nothing looked at
/// the run of them, so the session was never replaced; a challenge accepted
/// in that window was never answered. A request that got *no answer at all*
/// (`isConnectionFailure`) says nothing about Lichess and everything about
/// the path to it, so two in a row replace the request session. Any HTTP
/// response — a 4xx or 5xx included — proves the path works and clears the
/// count.
///
/// A timeout can't tell a dead connection from an answer slower than the
/// request's idle limit (5 s for deferrable requests), so two slow answers
/// in a row also reset. That costs only new connections, which is why the
/// rule doesn't try to tell them apart.
///
/// Each request is tagged with the generation of the session it went out
/// on. A replaced session's late results don't count toward the new one: a
/// request still finishing on the old session after the reset says nothing
/// about the new connections.
///
/// At most one reset per `minimumResetSpacing`, so a network that is simply
/// down doesn't churn sessions; failures inside the spacing keep counting,
/// so the first failure after it resets.
struct LichessBotRequestConnectionHealth: Sendable, Equatable {
    /// Consecutive connection failures that replace the request session.
    static let failuresBeforeReset = 2
    static let minimumResetSpacing: Duration = .seconds(30)

    /// A replacement the transport must make.
    struct Reset: Sendable, Equatable {
        /// Consecutive connection failures behind it.
        let failures: Int
        /// The last of them, for the log.
        let lastFailure: String
        /// The generation the new session carries.
        let generation: Int
    }

    /// The current request session's generation. Starts at 0; each reset
    /// adds one.
    private(set) var generation = 0
    private(set) var consecutiveFailures = 0
    private var lastResetAt: Duration?

    /// A request on generation `generation` got an HTTP response.
    mutating func noteAnswered(generation: Int) {
        guard generation == self.generation else { return }
        consecutiveFailures = 0
    }

    /// A request on generation `generation` failed with `error` at `now`.
    /// Returns the reset to make, or nil.
    mutating func noteFailure(_ error: any Error, generation: Int, now: Duration) -> Reset? {
        guard generation == self.generation, Self.isConnectionFailure(error) else { return nil }
        consecutiveFailures += 1
        guard consecutiveFailures >= Self.failuresBeforeReset else { return nil }
        if let lastResetAt, now - lastResetAt < Self.minimumResetSpacing {
            return nil
        }
        let reset = Reset(failures: consecutiveFailures, lastFailure: error.localizedDescription, generation: self.generation + 1)
        self.generation = reset.generation
        consecutiveFailures = 0
        lastResetAt = now
        return reset
    }

    /// The `URLError` codes that mean no answer arrived: the connection, the
    /// route or the name lookup failed. Not `.cancelled` (DCM stopped it),
    /// and nothing that came with a response.
    static let connectionFailureCodes: Set<URLError.Code> = [
        .timedOut,
        .cannotConnectToHost,
        .networkConnectionLost,
        .cannotFindHost,
        .dnsLookupFailed,
        .notConnectedToInternet,
        .secureConnectionFailed,
    ]

    /// Whether `error` means the request got no answer from Lichess.
    static func isConnectionFailure(_ error: any Error) -> Bool {
        guard let urlError = error as? URLError else { return false }
        return connectionFailureCodes.contains(urlError.code)
    }
}
