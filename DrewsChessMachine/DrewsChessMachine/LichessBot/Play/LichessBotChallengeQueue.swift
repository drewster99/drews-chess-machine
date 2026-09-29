import Foundation

/// The operator's ordered list of players to challenge (plan §7.3 A). The
/// controller owns the one instance and sends its entries, one at a time,
/// whenever a slot is free; this type holds only the order and each entry's
/// state, and decides what to do next from what the controller tells it.
/// It never talks to Lichess, so every rule here is unit-tested directly.
///
/// **Entry lifecycle.** An entry waits until it is next and a slot is free,
/// is marked sending while its challenge goes out, and then either leaves
/// the queue (sent, or dropped because Lichess refused it — the reason goes
/// to the outcome line and the protocol log), stays listed as skipped with
/// its reason (per-opponent limit, offline, at its bot limit), or goes back
/// to waiting in its place (a 429 or the bot leaving Online stopped the
/// queue; it resumes from the same entry).
///
/// **One at a time.** While an entry is sending, nothing else is sent from
/// the queue, so its sends are staggered through the request gate and the
/// slot count is always re-read after the previous send settled.
struct LichessBotChallengeQueue: Sendable, Equatable {

    enum Status: Sendable, Equatable {
        case waiting
        case sending
        case skipped(reason: String)
    }

    struct Entry: Identifiable, Sendable, Equatable {
        let id: UUID
        let username: String
        /// Lowercased Lichess user id.
        let userID: String
        let request: LichessBotOutgoingChallenge
        var status: Status
    }

    /// A player to add: the name to challenge and the id for comparisons.
    struct Player: Sendable, Equatable {
        let username: String
        let userID: String

        init(username: String, userID: String) {
            self.username = username
            self.userID = userID.lowercased()
        }
    }

    /// What `add` did with each player, by username, in the order given.
    struct AddResult: Sendable, Equatable {
        var added: [String] = []
        /// Already waiting or being sent (or listed twice in one add).
        var alreadyQueued: [String] = []
        /// A challenge to them is already waiting for an answer.
        var alreadyPending: [String] = []
    }

    /// What the queue should do now.
    enum Step: Sendable, Equatable {
        /// Nothing waits.
        case idle
        /// Something waits, but can't be sent yet; the reason is shown.
        case wait(reason: String)
        /// Send this entry next.
        case send(Entry)
    }

    /// How one entry's send turned out.
    enum SendOutcome: Sendable, Equatable {
        /// The challenge was created; the entry leaves the queue.
        case sent
        /// Not sent, for a reason tied to this player; the entry stays
        /// listed with the reason and the queue moves on.
        case skipped(reason: String)
        /// Refused for another reason; the entry leaves the queue.
        case dropped(reason: String)
        /// Stopped by something that isn't about this player (a 429, the bot
        /// leaving Online, a slot taken meanwhile): the entry waits again in
        /// its place and the queue stops until the next pass.
        case stopped(reason: String)
    }

    static let waitingForSlotReason = "waiting for a free slot"
    static let sendInProgressReason = "a challenge is being sent"

    private(set) var entries: [Entry] = []

    var isEmpty: Bool {
        entries.isEmpty
    }

    /// Something still waits or is being sent. Skipped entries are done;
    /// they stay listed only so the operator sees why.
    var hasEntriesToSend: Bool {
        entries.contains { entry in
            switch entry.status {
            case .waiting, .sending: return true
            case .skipped: return false
            }
        }
    }

    /// Lowercased ids of players waiting or being sent.
    var activeUserIDs: Set<String> {
        Set(entries.filter { $0.status == .waiting || $0.status == .sending }.map(\.userID))
    }

    /// Append players in the order given. A player already waiting or being
    /// sent, or with a challenge pending, is not added again. A player
    /// listed as skipped is added afresh at the end and the skipped entry is
    /// removed: choosing them again is the operator asking to retry.
    mutating func add(_ players: [Player], request: LichessBotOutgoingChallenge, pendingUserIDs: Set<String>, makeID: () -> UUID = { UUID() }) -> AddResult {
        var result = AddResult()
        for player in players {
            if pendingUserIDs.contains(player.userID) {
                result.alreadyPending.append(player.username)
                continue
            }
            if activeUserIDs.contains(player.userID) {
                result.alreadyQueued.append(player.username)
                continue
            }
            entries.removeAll { $0.userID == player.userID }
            entries.append(Entry(id: makeID(), username: player.username, userID: player.userID, request: request, status: .waiting))
            result.added.append(player.username)
        }
        return result
    }

    /// The next step given whether sending is allowed at all (nil when it
    /// is, otherwise why not) and how many slots are free.
    func nextStep(sendingBlockedReason: String?, freeSlots: Int) -> Step {
        guard let next = entries.first(where: { $0.status == .waiting }) else { return .idle }
        if entries.contains(where: { $0.status == .sending }) {
            return .wait(reason: Self.sendInProgressReason)
        }
        if let sendingBlockedReason {
            return .wait(reason: sendingBlockedReason)
        }
        guard freeSlots > 0 else { return .wait(reason: Self.waitingForSlotReason) }
        return .send(next)
    }

    /// Mark the entry as going out. Returns false if it is gone (canceled)
    /// or no longer waiting.
    @discardableResult
    mutating func markSending(_ id: UUID) -> Bool {
        guard let index = entries.firstIndex(where: { $0.id == id }), entries[index].status == .waiting else { return false }
        entries[index].status = .sending
        return true
    }

    /// Mark a waiting entry skipped without sending it (a check that needs
    /// no request, such as a known bot limit).
    mutating func skip(_ id: UUID, reason: String) {
        guard let index = entries.firstIndex(where: { $0.id == id }) else { return }
        entries[index].status = .skipped(reason: reason)
    }

    /// Apply a send's outcome to its entry. Does nothing if the operator
    /// removed the entry while it was being sent.
    mutating func record(_ outcome: SendOutcome, for id: UUID) {
        guard let index = entries.firstIndex(where: { $0.id == id }) else { return }
        switch outcome {
        case .sent, .dropped:
            entries.remove(at: index)
        case .skipped(let reason):
            entries[index].status = .skipped(reason: reason)
        case .stopped:
            entries[index].status = .waiting
        }
    }

    /// Remove one entry (the operator's Cancel). Removing an entry being
    /// sent doesn't stop that send; its challenge then shows as pending.
    mutating func remove(_ id: UUID) {
        entries.removeAll { $0.id == id }
    }

    mutating func removeAll() {
        entries.removeAll()
    }
}
