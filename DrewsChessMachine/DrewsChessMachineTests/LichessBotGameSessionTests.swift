import XCTest
@testable import DrewsChessMachine

/// A clock that only moves when the test advances it. Sleepers stay
/// suspended until then, so a stream watchdog never fires on its own and a
/// scheduled claim or reconnect runs exactly when the test says.
final class LichessBotManualTime: LichessBotTimeSource, @unchecked Sendable {
    private struct Sleeper: Sendable {
        let id: UInt64
        let deadline: Duration
        let continuation: CheckedContinuation<Void, Error>
    }

    private struct State: Sendable {
        var now: Duration = .zero
        var nextID: UInt64 = 0
        var sleepers: [Sleeper] = []
    }

    private let state = SyncBox(State())

    func now() -> Duration {
        state.value.now
    }

    /// How many sleepers are suspended waiting for the clock to reach their
    /// deadline. A test that must be sure an advance wakes a particular
    /// sleeper waits for it to be registered here first; an advance made
    /// before a sleeper registers only moves the clock, and the sleeper's
    /// deadline is then counted from the advanced time.
    var waitingSleeperCount: Int {
        state.value.sleepers.count
    }

    func sleep(for duration: Duration) async throws {
        let id = state.mutate { s -> UInt64 in
            s.nextID += 1
            return s.nextID
        }
        try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
                // Checked under the lock, so a cancellation either lands
                // before this check (and is seen by it) or finds the sleeper.
                let immediate: Result<Void, Error>? = state.mutate { s in
                    if Task.isCancelled {
                        return .failure(CancellationError())
                    }
                    if duration <= .zero {
                        return .success(())
                    }
                    s.sleepers.append(Sleeper(id: id, deadline: s.now + duration, continuation: continuation))
                    return nil
                }
                if let immediate {
                    continuation.resume(with: immediate)
                }
            }
        } onCancel: {
            let removed = state.mutate { s -> Sleeper? in
                guard let index = s.sleepers.firstIndex(where: { $0.id == id }) else { return nil }
                return s.sleepers.remove(at: index)
            }
            removed?.continuation.resume(throwing: CancellationError())
        }
    }

    func advance(by duration: Duration) {
        let due = state.mutate { s -> [Sleeper] in
            s.now += duration
            let now = s.now
            let due = s.sleepers.filter { $0.deadline <= now }
            s.sleepers.removeAll { $0.deadline <= now }
            return due
        }
        for sleeper in due {
            sleeper.continuation.resume()
        }
    }
}

/// A minimal Lichess for one game: it owns the true move list, answers the
/// session's calls, and streams `gameFull` / `gameState` lines the way the
/// real game stream does. The opponent plays the first legal move in UCI
/// order unless the script says otherwise.
actor LichessBotFakeGameServer: LichessBotGameAPI {
    enum MoveOutcome: Sendable {
        case accept
        case rateLimited
        case rejected(status: Int)
    }

    /// What the server does after accepting one of the bot's moves. Once
    /// the script runs out, the opponent replies.
    enum AfterOurMove: Sendable {
        case opponentReplies
        /// The opponent moves with `offeringDraw=true`: the move and the
        /// offer arrive in one `gameState`.
        case opponentRepliesOfferingDraw
        /// The opponent, to move, proposes taking back `plies` plies.
        case opponentProposesTakeback(plies: Int)
        case opponentGone(claimWinInSeconds: Int)
        /// The opponent is thinking; nothing is sent.
        case opponentThinks
        case finish(status: String, winner: String?)
    }

    struct Record: Sendable {
        let streamOpens: Int
        let moveAttempts: Int
        let acceptedPlies: [Int]
        let acceptedOfferingDraw: [Bool]
        let calls: [String]
        let chats: [String]
    }

    static let gameID = "game0001"
    static let botID = "drewschessmachine"
    private let botIsWhite: Bool
    private let variantKey: String
    private var engine = ChessGameEngine(adjudication: .serverAuthoritative)
    private var tokens: [String] = []
    private var status: String
    private var winner: String?
    private var drawOfferByOpponent = false
    private var takebackProposal: Int?
    private var continuation: LichessBotChunkStream.Continuation?

    private var streamOpens = 0
    private var moveAttempts = 0
    private var acceptedPlies: [Int] = []
    private var acceptedOfferingDraw: [Bool] = []
    private var calls: [String] = []
    private var chats: [String] = []

    private var moveOutcomes: [MoveOutcome] = []
    private var afterOurMoves: [AfterOurMove] = []
    private var victoryClaimSucceeds = true

    init(botIsWhite: Bool = true, variantKey: String = "standard", initialTokens: [String] = [], initialStatus: String = "started", initialWinner: String? = nil) throws {
        self.botIsWhite = botIsWhite
        self.variantKey = variantKey
        self.status = initialStatus
        self.winner = initialWinner
        for token in initialTokens {
            try Self.apply(token, to: engine)
            tokens.append(token)
        }
    }

    func setScript(moveOutcomes: [MoveOutcome] = [], afterOurMoves: [AfterOurMove]) {
        self.moveOutcomes = moveOutcomes
        self.afterOurMoves = afterOurMoves
    }

    func setVictoryClaimSucceeds(_ value: Bool) {
        victoryClaimSucceeds = value
    }

    func record() -> Record {
        Record(
            streamOpens: streamOpens,
            moveAttempts: moveAttempts,
            acceptedPlies: acceptedPlies,
            acceptedOfferingDraw: acceptedOfferingDraw,
            calls: calls,
            chats: chats
        )
    }

    // MARK: LichessBotGameAPI

    func openGameStream(gameID: String) async throws -> LichessBotChunkStream {
        streamOpens += 1
        continuation?.finish()
        let (stream, continuation) = LichessBotChunkStream.makeStream()
        self.continuation = continuation
        send(gameFullJSON())
        return stream
    }

    func makeMove(gameID: String, uci: String, offeringDraw: Bool) async throws {
        moveAttempts += 1
        let outcome = moveOutcomes.isEmpty ? .accept : moveOutcomes.removeFirst()
        switch outcome {
        case .accept:
            break
        case .rateLimited:
            throw LichessBotGateError.rateLimited(cooldown: .seconds(60))
        case .rejected(let status):
            throw LichessBotAPIError.http(status: status, message: "Not your turn, or game already over")
        }
        guard status == "started", isBotToMove else {
            throw LichessBotAPIError.http(status: 400, message: "Not your turn, or game already over")
        }
        acceptedPlies.append(tokens.count)
        acceptedOfferingDraw.append(offeringDraw)
        try Self.apply(uci, to: engine)
        tokens.append(uci)
        let opponentWasOffering = drawOfferByOpponent
        drawOfferByOpponent = false
        if offeringDraw && opponentWasOffering {
            status = "draw"
            sendState()
            return
        }
        sendState()

        let action = afterOurMoves.isEmpty ? .opponentReplies : afterOurMoves.removeFirst()
        switch action {
        case .opponentReplies:
            try opponentMoves()
        case .opponentRepliesOfferingDraw:
            drawOfferByOpponent = true
            try opponentMoves()
        case .opponentProposesTakeback(let plies):
            takebackProposal = plies
            sendState()
        case .opponentGone(let seconds):
            send(#"{"type":"opponentGone","gone":true,"claimWinInSeconds":\#(seconds)}"#)
        case .opponentThinks:
            break
        case .finish(let newStatus, let newWinner):
            status = newStatus
            winner = newWinner
            sendState()
        }
    }

    func respondToDraw(gameID: String, accept: Bool) async throws {
        calls.append("draw:\(accept)")
        if accept && drawOfferByOpponent {
            status = "draw"
            sendState()
        }
    }

    func respondToTakeback(gameID: String, accept: Bool) async throws {
        calls.append("takeback:\(accept)")
        guard accept, let plies = takebackProposal else { return }
        takebackProposal = nil
        let kept = Array(tokens.dropLast(plies))
        engine = ChessGameEngine(adjudication: .serverAuthoritative)
        tokens = []
        for token in kept {
            try Self.apply(token, to: engine)
            tokens.append(token)
        }
        sendState()
        if !isBotToMove {
            try opponentMoves()
        }
    }

    func resign(gameID: String) async throws {
        calls.append("resign")
        status = "resign"
        winner = botIsWhite ? "black" : "white"
        sendState()
    }

    func abort(gameID: String) async throws {
        calls.append("abort")
        status = "aborted"
        sendState()
    }

    func claimVictory(gameID: String) async throws {
        calls.append("claimVictory")
        guard victoryClaimSucceeds else {
            throw LichessBotAPIError.http(status: 400, message: "cannot claim victory")
        }
        status = "timeout"
        winner = botIsWhite ? "white" : "black"
        sendState()
    }

    func claimDraw(gameID: String) async throws {
        calls.append("claimDraw")
        status = "draw"
        sendState()
    }

    func chat(gameID: String, room: LichessBotChatRoom, text: String) async throws {
        chats.append(text)
    }

    // MARK: Server side

    /// End the stream without the game ending (a deploy, a proxy timeout):
    /// the session must reopen it.
    func dropStream() {
        continuation?.finish()
        continuation = nil
    }

    /// Play `moves` on the server as if other clients made them, and stream
    /// one new state carrying them all (plan §6.1 B).
    func injectMoves(_ moves: [String]) throws {
        for uci in moves {
            try Self.apply(uci, to: engine)
            tokens.append(uci)
        }
        sendState()
    }

    /// Stream a `gameState` whose move list is `tokens` without changing
    /// the server's game — a state the client cannot replay.
    func sendBogusState(_ tokens: [String]) {
        send(#"{"type":"gameState","moves":"\#(tokens.joined(separator: " "))","wtime":180000,"btime":180000,"winc":2000,"binc":2000,"status":"started"}"#)
    }

    /// Stream a spectator chat line carrying `marker`, after every line
    /// streamed so far. The session reads its stream one line at a time and
    /// finishes handling each line before reading the next, so once this
    /// line is observed every earlier line has been fully handled. Work a
    /// line hands to a separate task (a scheduled claim, a held move) is not
    /// covered by that guarantee.
    func sendSentinel(_ marker: String) {
        send(#"{"type":"chatLine","room":"spectator","username":"Watcher","text":"\#(marker)"}"#)
    }

    /// Reject every move POST from now on.
    func rejectAllMoves() {
        moveOutcomes = Array(repeating: .rejected(status: 400), count: 100)
    }

    /// End the game without sending a final state on the game stream, as
    /// Lichess may do on a resignation (plan E22).
    func endSilently(status newStatus: String, winner newWinner: String?) {
        status = newStatus
        winner = newWinner
    }

    private var isBotToMove: Bool {
        (engine.state.currentPlayer == .white) == botIsWhite
    }

    private func opponentMoves() throws {
        guard let uci = engine.currentLegalMoves.map(\.uci).sorted().first else { return }
        try Self.apply(uci, to: engine)
        tokens.append(uci)
        sendState()
    }

    private static func apply(_ token: String, to engine: ChessGameEngine) throws {
        guard let move = ChessMove.parseUCI(token, legal: engine.currentLegalMoves, state: engine.state) else {
            throw LichessBotPositionError.illegalMove(token: token, ply: engine.moveHistory.count, fen: FENParser.fen(from: engine.state))
        }
        try engine.applyMoveAndAdvance(move)
    }

    private func stateJSON() -> String {
        var fields = [
            #""type":"gameState""#,
            #""moves":"\#(tokens.joined(separator: " "))""#,
            #""wtime":180000,"btime":180000,"winc":2000,"binc":2000"#,
            #""status":"\#(status)""#,
        ]
        if let winner {
            fields.append(#""winner":"\#(winner)""#)
        }
        if drawOfferByOpponent {
            fields.append(botIsWhite ? #""bdraw":true"# : #""wdraw":true"#)
        }
        if takebackProposal != nil {
            fields.append(botIsWhite ? #""btakeback":true"# : #""wtakeback":true"#)
        }
        return "{" + fields.joined(separator: ",") + "}"
    }

    private func gameFullJSON() -> String {
        let bot = #"{"id":"\#(Self.botID)","name":"DrewsChessMachine","title":"BOT","rating":1500}"#
        let opponent = #"{"id":"alice","name":"Alice","rating":1500}"#
        let white = botIsWhite ? bot : opponent
        let black = botIsWhite ? opponent : bot
        return #"{"type":"gameFull","id":"\#(Self.gameID)","variant":{"key":"\#(variantKey)","name":"x","short":"x"},"clock":{"initial":180000,"increment":2000},"speed":"blitz","perf":{"name":"Blitz"},"rated":false,"createdAt":1700000000000,"white":\#(white),"black":\#(black),"initialFen":"startpos","state":\#(stateJSON())}"#
    }

    private func sendState() {
        send(stateJSON())
    }

    private func send(_ line: String) {
        continuation?.yield(Data((line + "\n").utf8))
    }
}

/// Plays the first legal move in UCI order, with a fixed value-head reading,
/// and records the ply of every decision.
final class LichessBotScriptedMoveSource: LichessBotMoveSource, @unchecked Sendable {
    let info: LichessBotGenerationInfo
    private let win: Float
    private let draw: Float
    private let loss: Float
    let decidedPlies = SyncBox<[Int]>([])

    init(win: Float = 0.3, draw: Float = 0.4, loss: Float = 0.3) {
        info = LichessBotGenerationInfo(
            generationID: 1,
            sourceKind: .champion,
            modelID: "20260928-1-TEST",
            trainingStep: nil,
            snapshotAt: Date(timeIntervalSince1970: 0),
            architectureSummary: "test",
            filePath: nil,
            fileSHA256: nil
        )
        self.win = win
        self.draw = draw
        self.loss = loss
    }

    func decide(_ request: LichessBotMoveRequest, schedule: SamplingSchedule) async throws -> LichessBotMoveDecision {
        decidedPlies.modify { $0.append(request.ply) }
        guard let uci = request.legalMoves.map(\.uci).sorted().first else {
            throw LichessBotMoveChooserError.noLegalMoves
        }
        return LichessBotMoveDecision(
            uci: uci, san: uci, chosenProbability: 1, topMoves: [],
            win: win, draw: draw, loss: loss,
            temperature: schedule.floorTau, legalMoveCount: request.legalMoves.count, randomish: false,
            encodeMilliseconds: 0, inferenceMilliseconds: 0, sampleMilliseconds: 0
        )
    }
}

/// Records every game event and turn status in order.
final class LichessBotRecordingGameObserver: LichessBotGameObserver, @unchecked Sendable {
    let events = SyncBox<[LichessBotGameEvent]>([])
    let turnStatuses = SyncBox<[LichessBotTurnStatus]>([])

    func gameEvent(gameID: String, _ event: LichessBotGameEvent) async {
        events.modify { $0.append(event) }
    }

    var actions: [String] {
        events.value.compactMap { event in
            if case .action(let text) = event { return text }
            return nil
        }
    }

    var anomalies: [String] {
        events.value.compactMap { event in
            if case .anomaly(let text) = event { return text }
            return nil
        }
    }

    var rejections: [String] {
        events.value.compactMap { event in
            if case .moveRejected(_, _, let error) = event { return error }
            return nil
        }
    }

    var streamEndCount: Int {
        events.value.filter { event in
            if case .streamEnded = event { return true }
            return false
        }.count
    }

    var receivedLines: [String] {
        events.value.compactMap { event in
            if case .streamLine(let data, _) = event { return String(decoding: data, as: UTF8.self) }
            return nil
        }
    }

    var finishedLocalDrawCondition: ChessDrawCondition? {
        events.value.compactMap { event -> ChessDrawCondition?? in
            if case .finished(_, _, let condition) = event { return .some(condition) }
            return nil
        }.last ?? nil
    }

    var finishedStatus: String? {
        events.value.compactMap { event in
            if case .finished(let status, _, _) = event { return status.raw }
            return nil
        }.last
    }
}

/// `LichessBotGameSession` against a fake Lichess (plan §6, §8, §12.4, §20).
final class LichessBotGameSessionTests: XCTestCase {

    private struct Harness {
        let server: LichessBotFakeGameServer
        let source: LichessBotScriptedMoveSource
        let observer: LichessBotRecordingGameObserver
        let time: LichessBotManualTime
        let session: LichessBotGameSession
    }

    private func makeHarness(
        server: LichessBotFakeGameServer,
        ourAccountID: String = LichessBotFakeGameServer.botID,
        source: LichessBotScriptedMoveSource = LichessBotScriptedMoveSource(),
        configure: (inout LichessBotSettings) -> Void = { _ in }
    ) -> Harness {
        var settings = LichessBotSettings.testBaseline()
        settings.chat.greetingEnabled = true
        settings.chat.goodbyeEnabled = false
        configure(&settings)
        let frozen = settings
        let observer = LichessBotRecordingGameObserver()
        let time = LichessBotManualTime()
        let session = LichessBotGameSession(
            gameID: LichessBotFakeGameServer.gameID,
            ourAccountID: ourAccountID,
            api: server,
            moveSource: source,
            latestMoveSource: { source },
            settingsProvider: { frozen },
            observer: observer,
            time: time,
            onTurnStatus: { _, status in observer.turnStatuses.modify { $0.append(status) } }
        )
        return Harness(server: server, source: source, observer: observer, time: time, session: session)
    }

    /// Run the session to completion, failing rather than hanging.
    private func runToEnd(_ harness: Harness, file: StaticString = #filePath, line: UInt = #line) async {
        let run = Task { await harness.session.run() }
        let finishedInTime = await withTaskGroup(of: Bool.self) { group in
            group.addTask {
                await run.value
                return true
            }
            group.addTask {
                do {
                    try await Task.sleep(for: .seconds(20))
                } catch {
                    // Cancelled because the session finished first.
                }
                return false
            }
            let first = await group.next()
            group.cancelAll()
            return first == true
        }
        if !finishedInTime {
            run.cancel()
            XCTFail("the session did not finish", file: file, line: line)
        }
    }

    /// Poll `condition`, optionally advancing the manual clock each time.
    private func waitUntil(_ description: String, advancing time: LichessBotManualTime? = nil, by step: Duration = .seconds(2), _ condition: () async -> Bool) async throws {
        for _ in 0..<2000 {
            if await condition() { return }
            time?.advance(by: step)
            try await Task.sleep(for: .milliseconds(5))
        }
        XCTFail("timed out waiting until \(description)")
        throw CancellationError()
    }

    /// Stream a sentinel line after everything streamed so far and wait
    /// until the session reads it; every earlier line has then been fully
    /// handled (see `LichessBotFakeGameServer.sendSentinel`).
    private func waitUntilEarlierLinesAreHandled(_ harness: Harness, marker: String) async throws {
        await harness.server.sendSentinel(marker)
        try await waitUntil("the sentinel \(marker) is read") { harness.observer.receivedLines.contains { $0.contains(marker) } }
    }

    // MARK: - Playing

    func testPlaysAGameToTheServersFinish() async throws {
        let server = try LichessBotFakeGameServer()
        await server.setScript(afterOurMoves: [.opponentReplies, .opponentReplies, .finish(status: "resign", winner: "white")])
        let h = makeHarness(server: server)
        await runToEnd(h)

        let record = await server.record()
        XCTAssertEqual(record.acceptedPlies, [0, 2, 4])
        XCTAssertEqual(record.acceptedOfferingDraw, [false, false, false])
        XCTAssertEqual(h.source.decidedPlies.value, [0, 2, 4])
        XCTAssertEqual(h.observer.finishedStatus, "resign")
        XCTAssertEqual(record.streamOpens, 1)
        XCTAssertEqual(record.chats.count, 1, "one greeting, no goodbye")
        XCTAssertEqual(h.observer.turnStatuses.value.last?.awaitingOurMove, false)
        XCTAssertEqual(h.observer.anomalies, [])
    }

    func testPlaysBlack() async throws {
        let server = try LichessBotFakeGameServer(botIsWhite: false, initialTokens: ["a2a3"])
        await server.setScript(afterOurMoves: [.opponentReplies, .finish(status: "mate", winner: "black")])
        let h = makeHarness(server: server)
        await runToEnd(h)
        let record = await server.record()
        XCTAssertEqual(record.acceptedPlies, [1, 3])
        XCTAssertEqual(h.observer.finishedStatus, "mate")
    }

    // MARK: - Connection and state disagreement

    /// A stream that closes mid-game is reopened after a backoff, and the
    /// new `gameFull` does not cause a second move for a ply already
    /// played (plan §6).
    func testReopensADroppedStreamWithoutReplaying() async throws {
        let server = try LichessBotFakeGameServer()
        await server.setScript(afterOurMoves: [.opponentThinks])
        let h = makeHarness(server: server)
        let run = Task { await h.session.run() }
        try await waitUntil("the first move is posted") { await server.record().acceptedPlies == [0] }
        await server.dropStream()
        try await waitUntil("the stream reopens", advancing: h.time) { await server.record().streamOpens == 2 }
        // The reopened stream's `gameFull` is streamed before the sentinel,
        // so any move it would trigger has been posted once the sentinel is
        // read.
        try await waitUntilEarlierLinesAreHandled(h, marker: "sentinel-after-reopen")
        let record = await server.record()
        XCTAssertEqual(record.moveAttempts, 1, "no move is re-sent on a reconnect")
        XCTAssertEqual(h.observer.streamEndCount, 1)
        run.cancel()
        await run.value
    }

    /// A 400 on a move is a state disagreement: the session reopens the
    /// stream for a fresh `gameFull` instead of resending (plan §5.5, E21).
    func testRejectedMoveResyncsThenPlays() async throws {
        let server = try LichessBotFakeGameServer()
        await server.setScript(moveOutcomes: [.rejected(status: 400)], afterOurMoves: [.finish(status: "resign", winner: "white")])
        let h = makeHarness(server: server)
        await runToEnd(h)
        let record = await server.record()
        XCTAssertEqual(record.streamOpens, 2)
        XCTAssertEqual(record.moveAttempts, 2)
        XCTAssertEqual(record.acceptedPlies, [0])
        XCTAssertEqual(h.observer.rejections.count, 1)
        XCTAssertEqual(record.chats.count, 1, "the greeting is not repeated on a resync")
    }

    /// A 429 on a move: post the same decision again once the gate lets
    /// requests through, since it is still our move at that ply (plan §5.4).
    func testRateLimitedMoveIsPostedAgain() async throws {
        let server = try LichessBotFakeGameServer()
        await server.setScript(moveOutcomes: [.rateLimited], afterOurMoves: [.finish(status: "resign", winner: "white")])
        let h = makeHarness(server: server)
        await runToEnd(h)
        let record = await server.record()
        XCTAssertEqual(record.streamOpens, 1)
        XCTAssertEqual(record.moveAttempts, 2)
        XCTAssertEqual(record.acceptedPlies, [0])
        XCTAssertEqual(h.source.decidedPlies.value, [0], "the decision is re-posted, not re-made")
        XCTAssertEqual(h.observer.rejections.count, 1)
        XCTAssertTrue(h.observer.rejections.first?.contains("rate limited") == true)
    }

    /// E22: when the game ends without a final `gameState`, a resync
    /// request (from the event stream's `gameFinish`) reopens the stream,
    /// and the new `gameFull` carries the finished status.
    func testResyncRequestPicksUpASilentFinish() async throws {
        let server = try LichessBotFakeGameServer()
        await server.setScript(afterOurMoves: [.opponentThinks])
        let h = makeHarness(server: server)
        let run = Task { await h.session.run() }
        try await waitUntil("the first move is posted") { await server.record().acceptedPlies == [0] }
        await server.endSilently(status: "resign", winner: "white")
        h.session.requestResync(reason: "gameFinish on the event stream")
        try await waitUntil("the game ends", advancing: h.time, by: .milliseconds(500)) { h.observer.finishedStatus != nil }
        await run.value
        let record = await server.record()
        XCTAssertEqual(record.streamOpens, 2)
        XCTAssertEqual(h.observer.finishedStatus, "resign")
    }

    // MARK: - Draw rules (plan E8, E10)

    /// The local engine sees a threefold but the server keeps the game
    /// live: DCM keeps playing and logs the disagreement once.
    func testLocalDrawRuleOnALiveGameIsLoggedAndPlayContinues() async throws {
        let shuffle = ["g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6", "f3g1", "f6g8"]
        let server = try LichessBotFakeGameServer(initialTokens: shuffle)
        await server.setScript(afterOurMoves: [.finish(status: "resign", winner: "white")])
        let h = makeHarness(server: server)
        await runToEnd(h)
        let record = await server.record()
        XCTAssertEqual(record.acceptedPlies, [shuffle.count])
        let disagreements = h.observer.anomalies.filter { $0.contains("threefoldRepetition") }
        XCTAssertEqual(disagreements.count, 1)
    }

    /// Lichess streams the move that completes a threefold with status
    /// `started`, then ends the game in the next `gameState` a moment later
    /// (observed live, game zZMhCXrX). The two agree, so that is not a
    /// disagreement: no anomaly may be logged.
    func testThreefoldEndedInTheNextStateIsNotAnAnomaly() async throws {
        // White's rook shuffles a1–a2 against a knight shuffle; the bot
        // plays the first legal move in UCI order, which is a1a2 here and
        // completes the third occurrence.
        let setup = ["a2a4", "g8f6", "a1a2", "f6g8", "a2a1", "g8f6", "a1a2", "f6g8", "a2a1", "g8f6"]
        let server = try LichessBotFakeGameServer(initialTokens: setup)
        await server.setScript(afterOurMoves: [.finish(status: "draw", winner: nil)])
        let h = makeHarness(server: server)
        await runToEnd(h)
        let record = await server.record()
        XCTAssertEqual(record.acceptedPlies, [setup.count])
        XCTAssertEqual(h.observer.finishedStatus, "draw")
        XCTAssertEqual(h.observer.finishedLocalDrawCondition, .threefoldRepetition)
        XCTAssertEqual(h.observer.anomalies.filter { $0.contains("threefoldRepetition") }, [])
    }

    /// The final position's local draw rule is recorded beside the
    /// server's status.
    func testFinishRecordsTheLocalDrawCondition() async throws {
        let shuffle = ["g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6", "f3g1", "f6g8"]
        let server = try LichessBotFakeGameServer(initialTokens: shuffle, initialStatus: "draw")
        let h = makeHarness(server: server)
        await runToEnd(h)
        XCTAssertEqual(h.observer.finishedStatus, "draw")
        XCTAssertEqual(h.observer.finishedLocalDrawCondition, .threefoldRepetition)
    }

    // MARK: - Takebacks

    /// E30: a takeback that removes a ply we had moved at must let us move
    /// at that ply again.
    func testTakebackOfOurMoveLetsUsMoveAgain() async throws {
        let server = try LichessBotFakeGameServer()
        await server.setScript(afterOurMoves: [.opponentReplies, .opponentProposesTakeback(plies: 2), .opponentReplies, .finish(status: "resign", winner: "white")])
        let h = makeHarness(server: server) { $0.play.maxTakebacksAcceptedPerGame = 1 }
        await runToEnd(h)
        let record = await server.record()
        XCTAssertEqual(record.calls, ["takeback:true"])
        XCTAssertEqual(record.acceptedPlies, [0, 2, 2, 4])
        XCTAssertEqual(h.observer.finishedStatus, "resign")
    }

    /// With no takebacks allowed, a proposal gets no response at all.
    func testTakebackIsNotAcceptedByDefault() async throws {
        let server = try LichessBotFakeGameServer()
        await server.setScript(afterOurMoves: [.opponentReplies, .opponentProposesTakeback(plies: 2)])
        let h = makeHarness(server: server)
        let run = Task { await h.session.run() }
        try await waitUntil("the proposal arrives") { h.observer.receivedLines.contains { $0.contains("btakeback") } }
        try await waitUntilEarlierLinesAreHandled(h, marker: "sentinel-after-takeback-proposal")
        let record = await server.record()
        XCTAssertEqual(record.calls, [])
        XCTAssertEqual(record.acceptedPlies, [0, 2])
        run.cancel()
        await run.value
    }

    // MARK: - Games DCM can't or shouldn't play

    /// E20: a game DCM can't play is aborted while that is allowed.
    func testUnplayableVariantIsAborted() async throws {
        let server = try LichessBotFakeGameServer(variantKey: "chess960")
        let h = makeHarness(server: server)
        await runToEnd(h)
        let record = await server.record()
        XCTAssertEqual(record.calls, ["abort"])
        XCTAssertEqual(record.moveAttempts, 0)
        XCTAssertEqual(h.observer.finishedStatus, "aborted")
        XCTAssertEqual(record.chats, [], "no greeting in a game DCM does not play")
    }

    /// E20: past the abort window, an unplayable game is resigned.
    func testUnplayableGamePastTheAbortWindowIsResigned() async throws {
        let server = try LichessBotFakeGameServer(variantKey: "chess960", initialTokens: ["a2a3", "a7a6"])
        let h = makeHarness(server: server)
        await runToEnd(h)
        let record = await server.record()
        XCTAssertEqual(record.calls, ["resign"])
        XCTAssertEqual(h.observer.finishedStatus, "resign")
    }

    /// A `gameFull` that already carries a finished status ends the session
    /// without a move (a reconnect after the game ended).
    func testFinishedStatusInGameFull() async throws {
        let server = try LichessBotFakeGameServer(initialTokens: ["f2f3", "e7e5", "g2g4", "d8h4"], initialStatus: "mate", initialWinner: "black")
        let h = makeHarness(server: server)
        await runToEnd(h)
        let record = await server.record()
        XCTAssertEqual(record.moveAttempts, 0)
        XCTAssertEqual(h.observer.finishedStatus, "mate")
        XCTAssertEqual(record.chats, [], "no greeting when joining a finished game")
    }

    func testAGameWithoutUsIsNotPlayed() async throws {
        let server = try LichessBotFakeGameServer()
        let h = makeHarness(server: server, ourAccountID: "someoneelse")
        await runToEnd(h)
        let record = await server.record()
        XCTAssertEqual(record.moveAttempts, 0)
        XCTAssertTrue(h.observer.anomalies.contains { $0.contains("not our game") })
    }

    // MARK: - Opponent gone

    func testClaimsVictoryWhenTheOpponentIsGone() async throws {
        let server = try LichessBotFakeGameServer()
        await server.setScript(afterOurMoves: [.opponentGone(claimWinInSeconds: 10)])
        let h = makeHarness(server: server)
        let run = Task { await h.session.run() }
        try await waitUntil("the claim is scheduled") { h.observer.actions.contains { $0.hasPrefix("opponent gone") } }
        let beforeCountdown = await server.record()
        XCTAssertEqual(beforeCountdown.calls, [], "nothing is claimed before the countdown")
        try await waitUntil("the game ends", advancing: h.time) { h.observer.finishedStatus != nil }
        await run.value
        let record = await server.record()
        XCTAssertEqual(record.calls, ["claimVictory"])
        XCTAssertEqual(h.observer.finishedStatus, "timeout")
    }

    func testFallsBackToAClaimedDrawWhenVictoryIsRefused() async throws {
        let server = try LichessBotFakeGameServer()
        await server.setScript(afterOurMoves: [.opponentGone(claimWinInSeconds: 10)])
        await server.setVictoryClaimSucceeds(false)
        let h = makeHarness(server: server)
        let run = Task { await h.session.run() }
        try await waitUntil("the claim is scheduled") { h.observer.actions.contains { $0.hasPrefix("opponent gone") } }
        try await waitUntil("the game ends", advancing: h.time) { h.observer.finishedStatus != nil }
        await run.value
        let record = await server.record()
        XCTAssertEqual(record.calls, ["claimVictory", "claimDraw"])
        XCTAssertEqual(h.observer.finishedStatus, "draw")
    }

    func testNoClaimWhenClaimingIsOff() async throws {
        let server = try LichessBotFakeGameServer()
        await server.setScript(afterOurMoves: [.opponentGone(claimWinInSeconds: 10)])
        let h = makeHarness(server: server) { $0.play.claimWhenOpponentGone = false }
        let run = Task { await h.session.run() }
        try await waitUntil("the opponent-gone line arrives") { h.observer.receivedLines.contains { $0.contains("opponentGone") } }
        try await waitUntilEarlierLinesAreHandled(h, marker: "sentinel-after-opponent-gone")
        // The session reports every claim it schedules while handling the
        // opponent-gone line, before the claim's own task exists, so this is
        // settled now that the line has been handled.
        XCTAssertFalse(h.observer.actions.contains { $0.hasPrefix("opponent gone") }, "no claim is scheduled")
        // Well past the claim countdown and its grace, in one step.
        h.time.advance(by: .seconds(20))
        try await waitUntilEarlierLinesAreHandled(h, marker: "sentinel-after-the-countdown")
        let record = await server.record()
        XCTAssertEqual(record.calls, [])
        run.cancel()
        await run.value
    }

    // MARK: - Value-head decisions

    /// A pending draw offer is accepted by riding it on our move when the
    /// value head's expected score is low enough (plan §12.4).
    func testAcceptsADrawOfferByOfferingOnTheMove() async throws {
        let server = try LichessBotFakeGameServer()
        await server.setScript(afterOurMoves: [.opponentRepliesOfferingDraw])
        let losing = LichessBotScriptedMoveSource(win: 0.05, draw: 0.3, loss: 0.65)
        let h = makeHarness(server: server, source: losing) {
            $0.play.acceptDrawEnabled = true
            $0.play.acceptDrawExpectedScore = 0.45
        }
        await runToEnd(h)
        let record = await server.record()
        XCTAssertEqual(record.acceptedOfferingDraw, [false, true])
        XCTAssertEqual(h.observer.finishedStatus, "draw")
    }

    func testResignsWhenTheValueHeadSaysLost() async throws {
        let server = try LichessBotFakeGameServer()
        let lost = LichessBotScriptedMoveSource(win: 0, draw: 0, loss: 1)
        let h = makeHarness(server: server, source: lost) {
            $0.play.resignEnabled = true
            $0.play.resignConsecutiveMoves = 2
            $0.play.resignMinimumPly = 0
        }
        await runToEnd(h)
        let record = await server.record()
        XCTAssertEqual(record.acceptedPlies, [0], "a single reading is not a streak")
        XCTAssertEqual(record.calls, ["resign"])
        XCTAssertEqual(h.observer.finishedStatus, "resign")
    }
}
