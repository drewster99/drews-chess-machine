import Foundation

/// The controller's half of the challenge log (challenge-log plan §3.4): the
/// in-memory ledger, the one point every challenge fact is recorded through,
/// and the bookkeeping that decides which facts are written — our own
/// challenges' echoes, game starts, replays.
///
/// **One funnel.** `record(_:)` is the only mutation of the ledger and the
/// only caller of the log's writer. In one synchronous step on the main
/// actor it folds the fact into the ledger (or holds it while the ledger is
/// loading) and enqueues the append, so the ledger and the files never
/// disagree about what was recorded in this run.
///
/// **Loading.** The ledger is nil until `load()` lands, and is loaded only
/// while nil, so a loaded ledger is never replaced by a re-read that would
/// miss appends still queued. The read is enqueued on the general file queue
/// at a known point on the main actor: every fact recorded before it is in
/// the files it reads (its append was enqueued first), and every fact after
/// it is held in `eventsAwaitingLedger` and folded on top when the read
/// lands. If the read fails, those held facts are the ledger's only content,
/// and its load status says so.
///
/// **Our own echoes.** Lichess echoes each of our challenges on the event
/// stream, usually before the POST that created it returns. Writing a line
/// per echo would duplicate nearly every send, so an echo is held, keyed by
/// id, until `outgoingCreated` for it is recorded. One the ledger already
/// knows (a reconnect replaying an earlier run's challenge) is not held. One
/// still unmatched after `echoMatchWindow` — no POST of this run can still
/// answer by then — is written as `outgoingSeenWithoutCreatedLine`,
/// attributed to the single unanswered send to that player in the window if
/// there is exactly one; teardown writes every echo still held as not
/// recorded.
///
/// **Game starts.** `gameStarted` is written when the id is known (in the
/// ledger, held, or among held echoes) and has none yet. A `gameStart`
/// seen before anything named its id (the POST race with a lost echo) is
/// remembered, and its `gameStarted` is written right after the fact that
/// first names the id.
@MainActor
@Observable
final class LichessBotChallengeLogRecorder {

    /// The fold of the challenge log; nil until loaded.
    private(set) var ledger: LichessBotChallengeLedger?

    /// Facts recorded while the ledger is nil, in order.
    @ObservationIgnored private var eventsAwaitingLedger: [LichessBotChallengeLogEntry] = []
    @ObservationIgnored private var loadInFlight = false
    /// Our own challenges seen on the event stream with no created line yet.
    @ObservationIgnored private var heldEchoes: [String: HeldEcho] = [:]
    /// This run's sends whose POST got no answer, for echo attribution.
    @ObservationIgnored private var unansweredSends: [UnansweredSend] = []
    /// Ids of this run's `gameStart`s.
    @ObservationIgnored private var gameStartsSeen: Set<String> = []

    private let log: LichessBotChallengeLog
    private let clock: () -> Date
    /// Where alarms go (the controller's alarm list); set by the owner.
    @ObservationIgnored var alarmSink: ((String) -> Void)?
    /// Called after each fact is recorded, and after the ledger loads: the
    /// game-origin resolver decides waiting games from them (§3.5).
    @ObservationIgnored var onRecorded: ((LichessBotChallengeLogEvent) -> Void)?
    @ObservationIgnored var onLoaded: (() -> Void)?

    /// How long an echo may wait for its created line: twice the longest a
    /// request can run, so no POST of this run can still answer after it.
    static let echoMatchWindow: TimeInterval = 2 * LichessBotRequestTimeouts.resource

    private struct HeldEcho {
        let challenge: LichessBotChallengeSnapshot
        let seenAt: Date
    }

    private struct UnansweredSend {
        let attemptID: UUID
        /// Lowercased.
        let opponentID: String
        let sender: LichessBotChallengeSender
        let at: Date
    }

    init(directory: LichessBotDataDirectory,
         fileQueue: LichessBotFileQueue,
         systemCalls: LichessBotJSONLines.AppendSystemCalls,
         clock: @escaping () -> Date) {
        // The writer reports failures from the file queue; they reach the
        // alarm list through this relay, set once `self` exists.
        let failureRelay = SyncBox<(@Sendable (String) -> Void)?>(nil)
        self.log = LichessBotChallengeLog(directory: directory, fileQueue: fileQueue, systemCalls: systemCalls) { error in
            let text = "Challenge log write failed: \(error.localizedDescription)"
            if let relay = failureRelay.value {
                relay(text)
            } else {
                SessionLogger.shared.log("[ALARM] LICHESS-BOT \(text)")
            }
        }
        self.clock = clock
        failureRelay.value = { [weak self] text in
            Task { @MainActor in
                if let self {
                    self.alarm(text)
                } else {
                    SessionLogger.shared.log("[ALARM] LICHESS-BOT \(text)")
                }
            }
        }
    }

    // MARK: - Loading

    /// Load the ledger from the day files, once. Logs the load; raises an
    /// alarm for lines written by a newer build and for files left out.
    func load() async {
        guard ledger == nil, !loadInFlight else { return }
        loadInFlight = true
        defer { loadInFlight = false }
        // Everything recorded so far was enqueued before this read, so it
        // is in the files; only what is recorded from here on is held.
        eventsAwaitingLedger = []
        let start = ContinuousClock.now
        var loaded: LichessBotChallengeLedger
        do {
            let contents = try await log.readAll()
            loaded = LichessBotChallengeLedger(contents: contents)
            let elapsed = ContinuousClock.now - start
            let milliseconds = Double(elapsed.components.seconds) * 1000 + Double(elapsed.components.attoseconds) / 1e15
            SessionLogger.shared.log("[LICHESS-BOT] challenge log loaded: files=\(contents.filesRead.count) lines=\(contents.lineCount) bytes=\(contents.byteCount) ms=\(String(format: "%.1f", milliseconds)) skipped_newer=\(contents.skippedNewerLines) files_left_out=\(contents.filesLeftOut.count)")
            if contents.skippedNewerLines > 0 {
                alarm("The challenge log has \(contents.skippedNewerLines) line(s) written by a newer build; they are skipped")
            }
            for file in contents.filesLeftOut {
                alarm("Challenge log file \(file.name) is left out: \(file.reason)")
            }
            for file in contents.filesRead where file.droppedTrailingByteCount > 0 {
                SessionLogger.shared.log("[LICHESS-BOT] challenge log \(file.name): dropped an unterminated final line of \(file.droppedTrailingByteCount) bytes (cut and recorded at the next append)")
            }
        } catch {
            loaded = LichessBotChallengeLedger(loadStatus: .failed(reason: error.localizedDescription))
            alarm("Loading the challenge log failed: \(error.localizedDescription); it holds only this run's challenges until it loads")
        }
        loaded.apply(contentsOf: eventsAwaitingLedger)
        eventsAwaitingLedger = []
        ledger = loaded
        onLoaded?()
    }

    // MARK: - The funnel

    /// Record one fact: fold it into the ledger (or hold it while loading)
    /// and append it. The only mutation of either.
    func record(_ event: LichessBotChallengeLogEvent) {
        let entry = LichessBotChallengeLogEntry(at: clock(), event: event)
        if ledger != nil {
            ledger?.apply(entry)
        } else {
            eventsAwaitingLedger.append(entry)
        }
        log.record(event, at: entry.at)
        afterRecording(event)
        onRecorded?(event)
    }

    /// The bookkeeping a recorded fact settles: its echo is matched, an
    /// unanswered send is remembered, and a game start seen before the id
    /// was known is written now.
    private func afterRecording(_ event: LichessBotChallengeLogEvent) {
        switch event {
        case .outgoingCreated(let challenge, _, _, _, _):
            heldEchoes[challenge.id] = nil
            writeGameStartedIfSeen(challenge.id)
        case .outgoingSeenWithoutCreatedLine(let challenge, _), .incomingReceived(let challenge):
            writeGameStartedIfSeen(challenge.id)
        case .outgoingNotCreated(let attemptID, let opponentID, let sender, _, _, let reason, _):
            if case .noAnswer = reason {
                unansweredSends.append(UnansweredSend(attemptID: attemptID, opponentID: opponentID.lowercased(), sender: sender, at: clock()))
            }
        case .withdrawalRequested, .withdrawalResult, .incomingDecided, .incomingResponseFailed,
             .declinedOnLichess, .canceledOnLichess, .gameStarted, .unterminatedLineCut:
            break
        }
    }

    private func writeGameStartedIfSeen(_ id: String) {
        guard gameStartsSeen.contains(id), !hasFact(for: id, where: { if case .gameStarted = $0 { return true }; return false }) else { return }
        record(.gameStarted(challengeID: id))
    }

    // MARK: - Stream facts

    /// One of our own challenges, echoed on the event stream: held until its
    /// created line, unless the log already has a line for it.
    func noteOwnEcho(_ challenge: LichessBotChallengeSnapshot) {
        let known = hasFact(for: challenge.id) { event in
            switch event {
            case .outgoingCreated, .outgoingSeenWithoutCreatedLine: return true
            default: return false
            }
        }
        guard !known, heldEchoes[challenge.id] == nil else { return }
        heldEchoes[challenge.id] = HeldEcho(challenge: challenge, seenAt: clock())
    }

    /// Someone else's challenge: recorded once per id (Lichess replays open
    /// challenges on every reconnect, and a replay writes nothing).
    func noteIncomingChallenge(_ challenge: LichessBotChallengeSnapshot) {
        let known = hasFact(for: challenge.id) { if case .incomingReceived = $0 { return true }; return false }
        guard !known else { return }
        record(.incomingReceived(challenge: challenge))
    }

    /// A `gameStart`: written as `gameStarted` when its id is a challenge
    /// the log knows (or an echo held) and has no game start yet;
    /// remembered either way, for an id named later.
    func noteGameStart(gameID: String) {
        gameStartsSeen.insert(gameID)
        let started = hasFact(for: gameID) { if case .gameStarted = $0 { return true }; return false }
        guard !started else { return }
        guard heldEchoes[gameID] != nil || hasFact(for: gameID, where: { _ in true }) else { return }
        record(.gameStarted(challengeID: gameID))
    }

    /// Write every echo held longer than `echoMatchWindow` (the poll loop's
    /// call).
    func writeExpiredEchoes() {
        let now = clock()
        let expired = heldEchoes.values
            .filter { now.timeIntervalSince($0.seenAt) >= Self.echoMatchWindow }
            .sorted { $0.seenAt < $1.seenAt }
        for echo in expired {
            heldEchoes[echo.challenge.id] = nil
            record(.outgoingSeenWithoutCreatedLine(challenge: echo.challenge, attribution: attribution(for: echo)))
        }
        unansweredSends.removeAll { now.timeIntervalSince($0.at) >= 2 * Self.echoMatchWindow }
    }

    /// Write every echo still held, unattributed (teardown: no send of this
    /// runtime can be matched any more).
    func writeAllHeldEchoes() {
        let held = heldEchoes.values.sorted { $0.seenAt < $1.seenAt }
        heldEchoes = [:]
        for echo in held {
            record(.outgoingSeenWithoutCreatedLine(challenge: echo.challenge, attribution: .notRecorded))
        }
        unansweredSends = []
    }

    /// The single unanswered send to the echo's player within the window
    /// around it, or not recorded when there are none or several.
    private func attribution(for echo: HeldEcho) -> LichessBotEchoAttribution {
        guard let opponentID = echo.challenge.destUser?.id.lowercased() else { return .notRecorded }
        let candidates = unansweredSends.filter {
            $0.opponentID == opponentID && abs($0.at.timeIntervalSince(echo.seenAt)) <= Self.echoMatchWindow
        }
        guard candidates.count == 1, let send = candidates.first else { return .notRecorded }
        return .unansweredSend(attemptID: send.attemptID, sender: send.sender)
    }

    /// Whether a fact about `id` matching `predicate` is in the ledger or
    /// among the facts held while it loads.
    private func hasFact(for id: String, where predicate: (LichessBotChallengeLogEvent) -> Bool) -> Bool {
        if let row = ledger?.row(challengeID: id), row.facts.contains(where: { predicate($0.event) }) {
            return true
        }
        return eventsAwaitingLedger.contains { entry in
            LichessBotChallengeLedgerRowKey(entry.event) == .challenge(id: id) && predicate(entry.event)
        }
    }

    // MARK: - Shutdown

    /// Wait for the appends recorded so far.
    func flush() async throws {
        try await log.flush()
    }

    func logSummary() {
        log.logSummary()
    }

    /// Raise `text` through the owner's alarm list, which also logs it; with
    /// no owner wired, log it here, so it is never lost.
    private func alarm(_ text: String) {
        if let alarmSink {
            alarmSink(text)
        } else {
            SessionLogger.shared.log("[ALARM] LICHESS-BOT \(text)")
        }
    }
}
