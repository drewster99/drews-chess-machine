import Foundation

/// What the players DCM's pending challenges wait on are doing: Lichess's
/// online and playing flags, read in one `GET /api/users/status` for all of
/// them. A challenge sits unanswered while its bot plays another game, so
/// the Overview shows each pending row's player as playing, online or
/// offline.
struct LichessBotPendingOpponentStatuses: Sendable, Equatable {
    /// How long a read stands before the next. Lichess calls the endpoint
    /// cheap; every few seconds would still be noise next to the games'
    /// own requests, and a challenge waits tens of seconds at least.
    static let readInterval: TimeInterval = 15

    /// The last successful read, by lowercased id.
    private(set) var statuses: [String: LichessBotUserStatus] = [:]
    private(set) var readAt: Date?
    /// The last read attempt, successful or not: the interval runs from it,
    /// so a failing endpoint is tried once per interval.
    private(set) var attemptedAt: Date?
    /// Why the latest read failed; nil after a success.
    private(set) var failure: String?
    /// The ids the latest read asked for. Lichess leaves out a player it
    /// has no status for (a closed account), so the reply alone can't say
    /// which pending players were already asked about.
    private(set) var requestedIDs: Set<String> = []
    /// The ids the last successful read asked for: one among them that
    /// isn't in `statuses` is a player Lichess reported nothing for.
    private(set) var answeredIDs: Set<String> = []

    /// A read is due when any of `ids` has never been read (a new pending
    /// challenge), or the interval has passed since the last attempt. Never
    /// with nothing pending.
    func needsRead(of ids: Set<String>, now: Date) -> Bool {
        guard !ids.isEmpty else { return false }
        guard let attemptedAt else { return true }
        let elapsed = now.timeIntervalSince(attemptedAt)
        if elapsed >= Self.readInterval {
            return true
        }
        // A player not yet asked about waits only while a failure is being
        // backed off.
        return failure == nil && !ids.isSubset(of: requestedIDs.union(statuses.keys))
    }

    mutating func noteReadStarted(at date: Date) {
        attemptedAt = date
    }

    mutating func noteReadRequested(_ ids: Set<String>) {
        requestedIDs = ids
    }

    /// The read replaces the previous one: players no longer pending drop out.
    mutating func noteRead(_ newStatuses: [String: LichessBotUserStatus], at date: Date) {
        statuses = newStatuses
        answeredIDs = requestedIDs
        readAt = date
        failure = nil
    }

    mutating func noteReadFailed(_ text: String) {
        failure = text
    }

    /// What `userID`'s player was doing at the last read.
    func activity(of userID: String) -> Activity {
        if let status = statuses[userID.lowercased()], let readAt {
            if status.playing == true {
                return .playing(readAt: readAt)
            }
            return status.online == true ? .onlineNotPlaying(readAt: readAt) : .offline(readAt: readAt)
        }
        if let readAt, answeredIDs.contains(userID.lowercased()) {
            return .notReported(readAt: readAt)
        }
        if let failure {
            return .unknown(failure)
        }
        return .notReadYet
    }

    enum Activity: Sendable, Equatable {
        case playing(readAt: Date)
        case onlineNotPlaying(readAt: Date)
        case offline(readAt: Date)
        /// Lichess's reply left the player out (a closed or unknown account).
        case notReported(readAt: Date)
        case notReadYet
        /// The read failed, with why.
        case unknown(String)
    }
}
