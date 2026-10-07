import Foundation

/// Lichess's game sources (`gameStart.source`), from lila's `Source` enum
/// (`modules/core/src/main/game/misc.scala`, checked 2026-10-06), spelled as
/// lila names them: the case name lowercased. A value this build does not
/// know is kept verbatim by `LichessBotOpenValue`.
enum LichessBotGameSourceName: String, Sendable, Hashable, CaseIterable {
    case lobby
    case friend
    case ai
    case api
    case arena
    case position
    case `import`
    case importLive = "importlive"
    case simul
    case pool
    case swiss
}

/// How a game DCM played on Lichess began (challenge-log plan §3.5).
///
/// Named apart from `LichessBotSessionOrigin` (whether a game session is new
/// or resumed) and `LichessBotController.ChallengeOrigin` (who sent one of
/// DCM's challenges): this is the game's own beginning, recorded in its
/// journal and carried into its record and index row.
enum LichessBotGameOrigin: Sendable, Codable, Equatable, Hashable {
    /// DCM accepted a challenge someone sent it (or the operator accepted it
    /// by hand on lichess.org).
    case acceptedIncomingChallenge(challengeID: String, challengerID: String)
    /// A player accepted DCM's challenge; `sender` is who sent it.
    case outgoingChallengeAccepted(challengeID: String, sender: LichessBotChallengeSender)
    /// DCM's challenge (the echo proves the direction), sent by an earlier
    /// run or another client: who sent it was not recorded.
    case outgoingChallengeSenderNotRecorded(challengeID: String)
    /// Lichess paired DCM in a tournament.
    case tournament(source: LichessBotOpenValue<LichessBotGameSourceName>, tournamentID: String?)
    /// Not determined while the game was followed.
    case undetermined(source: LichessBotOpenValue<LichessBotGameSourceName>?, gap: LichessBotGameOriginGap)

    /// Whether the origin is known (anything but `undetermined`).
    var isDetermined: Bool {
        if case .undetermined = self { return false }
        return true
    }

    /// The challenge the game came from, when it came from one.
    var challengeID: String? {
        switch self {
        case .acceptedIncomingChallenge(let challengeID, _),
             .outgoingChallengeAccepted(let challengeID, _),
             .outgoingChallengeSenderNotRecorded(let challengeID):
            return challengeID
        case .tournament, .undetermined:
            return nil
        }
    }

    /// One fixed token per kind of origin: the PGN's `DCMOrigin` tag and the
    /// protocol log's `origin` field. The one place these spellings live.
    var token: String {
        switch self {
        case .acceptedIncomingChallenge:
            return "incoming"
        case .outgoingChallengeAccepted(_, let sender):
            switch sender {
            case .challengeSheet: return "sheet"
            case .casualResendOffer: return "casual-resend-offer"
            case .challengeQueue: return "queue"
            case .matchmaking(let trigger, _):
                switch trigger {
                case .automaticPass: return "matchmaking-auto"
                case .fillOpenSlots: return "matchmaking-fill"
                }
            case .matchmakingCasualResend: return "matchmaking-casual-resend"
            }
        case .outgoingChallengeSenderNotRecorded:
            return "outgoing-sender-not-recorded"
        case .tournament:
            return "tournament"
        case .undetermined:
            return "undetermined"
        }
    }

    /// The origin a game's journal holds, from every `gameOrigin` line in
    /// it, in order: the first determined one, else the last undetermined
    /// one (a determined origin always replaces an earlier undetermined
    /// one). `conflicting` lists later determined origins that differ from
    /// the one kept — a code bug, reported as an anomaly. The one rule shared
    /// by the record builder and the resumed journal.
    static func recorded(from origins: [LichessBotGameOrigin]) -> (origin: LichessBotGameOrigin?, conflicting: [LichessBotGameOrigin]) {
        let determined = origins.filter(\.isDetermined)
        if let first = determined.first {
            return (first, determined.dropFirst().filter { $0 != first })
        }
        return (origins.last, [])
    }
}

/// Why a game's origin was not determined.
enum LichessBotGameOriginGap: String, Sendable, Codable, Equatable, Hashable {
    /// No challenge with the game's id was in the challenge log by the time
    /// the game's session ended.
    case noChallengeRecord
    /// The challenge log did not load completely (a day file left out, or
    /// the read failed), so the challenge may be in the part that is missing.
    case challengeLogIncomplete
}

/// Decides each game's origin while the bot follows it (challenge-log plan
/// §3.5). Pure: the controller holds one, feeds it the game starts, session
/// starts and ends and the challenge log's facts, and writes what it
/// decides into the game's journal. Each game's origin is decided (and so
/// written) at most once per run.
struct LichessBotGameOriginResolver: Sendable {

    /// What a session start found.
    enum SessionStartDecision: Sendable, Equatable {
        /// The resumed journal already holds a determined origin: show it,
        /// write nothing.
        case alreadyRecorded(LichessBotGameOrigin)
        /// Decided now: write it.
        case decided(LichessBotGameOrigin)
        /// Not known yet (the POST race, an echo still unmatched, the log
        /// still loading): decided later, or `undetermined` at session end.
        case waiting
    }

    /// Each game's `gameStart`, for its source and tournament.
    private var gameStarts: [String: LichessBotGameEventInfo] = [:]
    /// Games waiting for an origin, with the origin their resumed journal
    /// already held (an undetermined one), if any.
    private var waiting: [String: LichessBotGameOrigin?] = [:]
    /// The origin each game's journal holds as far as this run knows: the
    /// one a resumed journal held, or the one this run wrote.
    private var known: [String: LichessBotGameOrigin] = [:]

    init() {}

    mutating func noteGameStart(_ info: LichessBotGameEventInfo) {
        gameStarts[info.gameId] = info
    }

    /// Whether `gameID` is waiting for its origin.
    func isWaiting(_ gameID: String) -> Bool {
        waiting[gameID] != nil
    }

    var waitingGameIDs: [String] {
        Array(waiting.keys)
    }

    mutating func sessionStarted(gameID: String, recordedOrigin: LichessBotGameOrigin?, ledger: LichessBotChallengeLedger?) -> SessionStartDecision {
        if let recordedOrigin, known[gameID] == nil {
            known[gameID] = recordedOrigin
        }
        if let existing = known[gameID], existing.isDetermined {
            waiting[gameID] = nil
            return .alreadyRecorded(existing)
        }
        if let origin = determine(gameID: gameID, ledger: ledger) {
            known[gameID] = origin
            waiting[gameID] = nil
            return .decided(origin)
        }
        waiting[gameID] = .some(known[gameID])
        return .waiting
    }

    /// A challenge fact was recorded (or the ledger loaded): decide the
    /// waiting game with that id, if the ledger now can.
    mutating func challengeKnown(gameID: String, ledger: LichessBotChallengeLedger?) -> LichessBotGameOrigin? {
        guard waiting[gameID] != nil, let origin = determine(gameID: gameID, ledger: ledger) else { return nil }
        waiting[gameID] = nil
        known[gameID] = origin
        return origin
    }

    /// The session ended with the origin still unknown: `undetermined`, with
    /// the gap the ledger's load status explains — unless the journal
    /// already ends in that same value. Nil when nothing is to be written.
    mutating func sessionEnded(gameID: String, ledger: LichessBotChallengeLedger?) -> LichessBotGameOrigin? {
        guard let journalHolds = waiting.removeValue(forKey: gameID) else { return nil }
        let gap: LichessBotGameOriginGap = ledger?.loadStatus == .complete ? .noChallengeRecord : .challengeLogIncomplete
        let origin = LichessBotGameOrigin.undetermined(source: source(of: gameID), gap: gap)
        guard origin != journalHolds else { return nil }
        known[gameID] = origin
        return origin
    }

    /// The origin the ledger and the game start determine, or nil when they
    /// don't (yet).
    private func determine(gameID: String, ledger: LichessBotChallengeLedger?) -> LichessBotGameOrigin? {
        if let row = ledger?.row(challengeID: gameID), let origin = LichessBotGameOrigin.fromChallengeLog(gameID: gameID, row: row) {
            return origin
        }
        if let info = gameStarts[gameID], let source = info.source, let known = source.known, known == .arena || known == .swiss {
            return .tournament(source: source, tournamentID: info.tournamentId)
        }
        return nil
    }

    private func source(of gameID: String) -> LichessBotOpenValue<LichessBotGameSourceName>? {
        gameStarts[gameID]?.source
    }
}

extension LichessBotGameOrigin {
    /// The origin the live challenge log's row for a game's id gives, or
    /// nil when it doesn't say: an incoming challenge (whatever DCM decided:
    /// a game started from it, so someone accepted it), DCM's challenge with
    /// its sender, or DCM's challenge seen only as its echo. The one mapping,
    /// shared by the resolver that writes origins and the display. An
    /// incoming row with no snapshot (only decisions, which this build never
    /// writes alone) doesn't say who challenged, so it gives nothing rather
    /// than an invented id.
    static func fromChallengeLog(gameID: String, row: LichessBotChallengeLedgerRow) -> LichessBotGameOrigin? {
        switch row.direction {
        case .incoming?:
            guard let challengerID = row.snapshot?.challenger.id else { return nil }
            return .acceptedIncomingChallenge(challengeID: gameID, challengerID: challengerID)
        case .outgoing?:
            if let sender = row.sender {
                return .outgoingChallengeAccepted(challengeID: gameID, sender: sender)
            }
            if !row.echoFacts.isEmpty {
                return .outgoingChallengeSenderNotRecorded(challengeID: gameID)
            }
            return nil
        case nil:
            return nil
        }
    }
}
