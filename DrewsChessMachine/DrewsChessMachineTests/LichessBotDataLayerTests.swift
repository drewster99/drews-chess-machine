import XCTest
@testable import DrewsChessMachine

/// The Lichess bot data layer (plan §10): journal, record building and
/// reconciliation, PGN, index, reconciler, instance lock, protocol log and
/// settings store. Every test works in its own temporary directory.
final class LichessBotDataLayerTests: XCTestCase {

    private var tempRoot: URL!
    private let botID = "drewschessmachine"

    override func setUpWithError() throws {
        tempRoot = FileManager.default.temporaryDirectory.appendingPathComponent("LichessBotDataLayerTests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: tempRoot, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try FileManager.default.removeItem(at: tempRoot)
    }

    private var directory: LichessBotDataDirectory {
        LichessBotDataDirectory(root: tempRoot)
    }

    private func makeStore() -> LichessBotRecordStore {
        LichessBotRecordStore(directory: directory, fileQueue: LichessBotFileQueue(), ourAccountID: botID)
    }

    // MARK: - Fixtures

    private static let generation = LichessBotGenerationInfo(
        generationID: 7,
        sourceKind: .champion,
        modelID: "20260928-1-TEST",
        trainingStep: nil,
        snapshotAt: Date(timeIntervalSince1970: 1_700_000_000),
        architectureSummary: "test",
        filePath: nil,
        fileSHA256: nil
    )

    private static func decision(_ uci: String) -> LichessBotMoveDecision {
        LichessBotMoveDecision(
            uci: uci, san: uci, chosenProbability: 0.5, topMoves: [],
            win: 0.3, draw: 0.4, loss: 0.3, temperature: 0.5, legalMoveCount: 20, randomish: false,
            encodeMilliseconds: 0.1, inferenceMilliseconds: 2, sampleMilliseconds: 0.1
        )
    }

    private static func stateJSON(_ tokens: [String], status: String = "started", winner: String? = nil, extra: String = "") -> String {
        let winnerField = winner.map { #","winner":"\#($0)""# } ?? ""
        return #"{"type":"gameState","moves":"\#(tokens.joined(separator: " "))","wtime":170000,"btime":160000,"winc":2000,"binc":2000,"status":"\#(status)"\#(winnerField)\#(extra)}"#
    }

    private func gameFullJSON(gameID: String, createdAt: Int64, tokens: [String] = [], variant: String = "standard") -> String {
        #"{"type":"gameFull","id":"\#(gameID)","variant":{"key":"\#(variant)"},"clock":{"initial":180000,"increment":2000},"speed":"blitz","perf":{"name":"Blitz"},"rated":false,"createdAt":\#(createdAt),"white":{"id":"\#(botID)","name":"DrewsChessMachine","title":"BOT","rating":1500},"black":{"id":"alice","name":"Alice","rating":1600},"initialFen":"startpos","state":\#(Self.stateJSON(tokens))}"#
    }

    /// A journal for a game where DCM is white and posted every white move,
    /// one `gameState` per move, optionally ending with a finished entry.
    private func syntheticJournal(gameID: String, createdAt: Int64, tokens: [String], finish: (status: String, winner: String?)?) -> [LichessBotJournalEntry] {
        var at = Date(timeIntervalSince1970: Double(createdAt) / 1000)
        func next() -> Date {
            at = at.addingTimeInterval(1)
            return at
        }
        var entries: [LichessBotJournalEntry] = [
            .init(at: next(), event: .header(schemaVersion: LichessBotJournal.schemaVersion, gameID: gameID, build: 1, gitHash: "test", resumed: false)),
            .init(at: next(), event: .streamOpened(attempt: 0)),
            .init(at: next(), event: .streamLine(raw: gameFullJSON(gameID: gameID, createdAt: createdAt))),
        ]
        for (ply, token) in tokens.enumerated() {
            if ply % 2 == 0 {
                entries.append(.init(at: next(), event: .moveDecided(ply: ply, decision: Self.decision(token), generation: Self.generation)))
                entries.append(.init(at: next(), event: .movePosted(ply: ply, uci: token, offeringDraw: false, milliseconds: 40)))
            }
            entries.append(.init(at: next(), event: .streamLine(raw: Self.stateJSON(Array(tokens.prefix(ply + 1))))))
        }
        if let finish {
            entries.append(.init(at: next(), event: .streamLine(raw: Self.stateJSON(tokens, status: finish.status, winner: finish.winner))))
            entries.append(.init(at: next(), event: .finished(status: finish.status, winner: finish.winner, localDrawCondition: nil)))
        }
        return entries
    }

    private func writeJournal(_ entries: [LichessBotJournalEntry], gameID: String, trailingGarbage: Data = Data()) throws {
        var data = Data()
        for entry in entries {
            data.append(try LichessBotJSONLines.encodeLine(entry))
        }
        data.append(trailingGarbage)
        try directory.createDirectories()
        try data.write(to: directory.inProgressJournalURL(gameID: gameID))
    }

    /// SAN for a UCI move list from the start, the way Lichess's export
    /// lists it.
    private static func sanMoves(_ tokens: [String]) throws -> [String] {
        let engine = ChessGameEngine(adjudication: .serverAuthoritative)
        var sans: [String] = []
        for token in tokens {
            let move = try XCTUnwrap(ChessMove.parseUCI(token, legal: engine.currentLegalMoves, state: engine.state), "illegal fixture move \(token)")
            sans.append(try SANFormatter.san(for: move, in: engine.state, legalMoves: engine.currentLegalMoves))
            try engine.applyMoveAndAdvance(move)
        }
        return sans
    }

    private func exportData(gameID: String, createdAt: Int64, tokens: [String], status: String, winner: String?, blackRatingDiff: Int? = nil) throws -> Data {
        let winnerField = winner.map { #""winner":"\#($0)","# } ?? ""
        let diff = blackRatingDiff.map { #","ratingDiff":\#($0)"# } ?? ""
        let json = #"{"id":"\#(gameID)","rated":false,"variant":"standard","speed":"blitz","perf":"blitz","createdAt":\#(createdAt),"lastMoveAt":\#(createdAt + 60000),"status":"\#(status)",\#(winnerField)"players":{"white":{"user":{"name":"DrewsChessMachine","title":"BOT","id":"\#(botID)"},"rating":1500},"black":{"user":{"name":"Alice","id":"alice"},"rating":1600\#(diff)}},"opening":{"eco":"C20","name":"King's Pawn Game","ply":2},"moves":"\#(try Self.sanMoves(tokens).joined(separator: " "))","clock":{"initial":180,"increment":2}}"#
        return Data(json.utf8)
    }

    private func export(gameID: String, createdAt: Int64, tokens: [String], status: String, winner: String?, blackRatingDiff: Int? = nil) throws -> LichessBotGameExport {
        try LichessBotGameExport.decode(exportData(gameID: gameID, createdAt: createdAt, tokens: tokens, status: status, winner: winner, blackRatingDiff: blackRatingDiff))
    }

    private let createdAt: Int64 = 1_759_000_000_000
    private let shortGame = ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "g8f6"]

    // MARK: - Journal

    func testJournalRoundTrips() throws {
        let entries = syntheticJournal(gameID: "g1", createdAt: createdAt, tokens: shortGame, finish: ("resign", "white"))
        try writeJournal(entries, gameID: "g1")
        let read = try LichessBotJournal.read(directory.inProgressJournalURL(gameID: "g1"))
        XCTAssertEqual(read.elements, entries)
        XCTAssertEqual(read.droppedTrailingByteCount, 0)
    }

    /// A crash mid-append leaves an unterminated last line: it is dropped
    /// and counted, and everything before it is kept.
    func testTruncatedFinalLineIsDroppedAndCounted() throws {
        let entries = syntheticJournal(gameID: "g1", createdAt: createdAt, tokens: shortGame, finish: nil)
        let garbage = Data(#"{"at":"2026-09-28T01:02:03.456Z","event":{"streamLi"#.utf8)
        try writeJournal(entries, gameID: "g1", trailingGarbage: garbage)
        let read = try LichessBotJournal.read(directory.inProgressJournalURL(gameID: "g1"))
        XCTAssertEqual(read.elements, entries)
        XCTAssertEqual(read.droppedTrailingByteCount, garbage.count)
    }

    func testCorruptCompleteLineIsAnErrorNotSkipped() throws {
        let entries = syntheticJournal(gameID: "g1", createdAt: createdAt, tokens: ["e2e4"], finish: nil)
        var data = Data()
        for entry in entries.prefix(2) {
            data.append(try LichessBotJSONLines.encodeLine(entry))
        }
        data.append(Data("not json\n".utf8))
        data.append(try LichessBotJSONLines.encodeLine(entries[2]))
        XCTAssertThrowsError(try LichessBotJSONLines.decode(LichessBotJournalEntry.self, from: data, fileName: "x")) { error in
            guard case .undecodableLine(_, let lineNumber, _)? = error as? LichessBotJSONLinesError else {
                return XCTFail("expected undecodableLine, got \(error)")
            }
            XCTAssertEqual(lineNumber, 3)
        }
    }

    // MARK: - Records and reconciliation

    func testRecordFromJournalMatchesExport() throws {
        let entries = syntheticJournal(gameID: "g1", createdAt: createdAt, tokens: shortGame, finish: ("resign", "white"))
        let journal = LichessBotJSONLines.Decoded(elements: entries, droppedTrailingByteCount: 0)
        let record = try LichessBotRecordBuilder.build(
            gameID: "g1", journal: journal,
            export: export(gameID: "g1", createdAt: createdAt, tokens: shortGame, status: "resign", winner: "white", blackRatingDiff: -3),
            exportUnavailableReason: nil, ourAccountID: botID, checkedAt: Date()
        )
        XCTAssertEqual(record.reconciliation.outcome, .matched, "\(record.reconciliation.mismatches)")
        XCTAssertEqual(record.ourColor, .white)
        XCTAssertEqual(record.outcome.pgnResult, "1-0")
        XCTAssertEqual(record.outcome.ourScore, 1)
        XCTAssertEqual(record.outcome.plies, shortGame.count)
        XCTAssertEqual(record.moves.map(\.san), try Self.sanMoves(shortGame))
        XCTAssertEqual(record.moves.map(\.uciAsGiven), shortGame)
        XCTAssertEqual(record.moves.filter(\.ours).count, 3)
        XCTAssertTrue(record.moves.filter(\.ours).allSatisfy { $0.decision != nil && $0.generationID == 7 && $0.postMilliseconds == 40 })
        XCTAssertTrue(record.moves.filter { !$0.ours }.allSatisfy { $0.decision == nil })
        XCTAssertEqual(record.opponent.kind, .human)
        XCTAssertEqual(record.opponent.ratingDiff, -3)
        XCTAssertEqual(record.us.kind, .bot)
        XCTAssertEqual(record.openingECO, "C20")
        XCTAssertEqual(record.generations, [Self.generation])
        XCTAssertEqual(record.anomalies, [])
        XCTAssertEqual(record.moves.last?.whiteClockMilliseconds, 170000)
    }

    /// E22: the journal never saw the finish (resignation-stream gap); the
    /// export supplies it and the mismatch is recorded.
    func testExportSuppliesAMissingFinish() throws {
        let entries = syntheticJournal(gameID: "g1", createdAt: createdAt, tokens: shortGame, finish: nil)
        let record = try LichessBotRecordBuilder.build(
            gameID: "g1", journal: .init(elements: entries, droppedTrailingByteCount: 0),
            export: export(gameID: "g1", createdAt: createdAt, tokens: shortGame, status: "resign", winner: "black"),
            exportUnavailableReason: nil, ourAccountID: botID, checkedAt: Date()
        )
        XCTAssertEqual(record.outcome.status, "resign")
        XCTAssertEqual(record.outcome.pgnResult, "0-1")
        XCTAssertEqual(record.reconciliation.outcome, .corrected)
        XCTAssertEqual(record.reconciliation.mismatches.count, 1)
    }

    /// The export wins on disagreement, and the disagreement is listed.
    func testExportWinsOnStatusDisagreement() throws {
        let entries = syntheticJournal(gameID: "g1", createdAt: createdAt, tokens: shortGame, finish: ("draw", nil))
        let record = try LichessBotRecordBuilder.build(
            gameID: "g1", journal: .init(elements: entries, droppedTrailingByteCount: 0),
            export: export(gameID: "g1", createdAt: createdAt, tokens: shortGame, status: "outoftime", winner: "white"),
            exportUnavailableReason: nil, ourAccountID: botID, checkedAt: Date()
        )
        XCTAssertEqual(record.outcome.status, "outoftime")
        XCTAssertEqual(record.outcome.winner, "white")
        XCTAssertEqual(record.reconciliation.outcome, .corrected)
        XCTAssertEqual(record.reconciliation.mismatches.count, 2)
    }

    /// E30: moves removed by a takeback are kept as retractions, and the
    /// replacement moves carry their own decisions.
    func testTakebackIsRecordedAsRetraction() throws {
        var entries = syntheticJournal(gameID: "g1", createdAt: createdAt, tokens: ["e2e4", "e7e5", "g1f3"], finish: nil)
        let at = Date(timeIntervalSince1970: Double(createdAt) / 1000 + 100)
        entries.append(.init(at: at, event: .streamLine(raw: Self.stateJSON(["e2e4"]))))
        entries.append(.init(at: at.addingTimeInterval(1), event: .streamLine(raw: Self.stateJSON(["e2e4", "c7c5"]))))
        entries.append(.init(at: at.addingTimeInterval(2), event: .moveDecided(ply: 2, decision: Self.decision("b1c3"), generation: Self.generation)))
        entries.append(.init(at: at.addingTimeInterval(3), event: .movePosted(ply: 2, uci: "b1c3", offeringDraw: false, milliseconds: 30)))
        entries.append(.init(at: at.addingTimeInterval(4), event: .streamLine(raw: Self.stateJSON(["e2e4", "c7c5", "b1c3"]))))
        let record = try LichessBotRecordBuilder.build(
            gameID: "g1", journal: .init(elements: entries, droppedTrailingByteCount: 0),
            export: nil, exportUnavailableReason: "test", ourAccountID: botID, checkedAt: Date()
        )
        XCTAssertEqual(record.moves.map(\.uciAsGiven), ["e2e4", "c7c5", "b1c3"])
        XCTAssertEqual(record.retractions.map(\.uciAsGiven), ["g1f3", "e7e5"])
        XCTAssertEqual(record.moves[2].decision?.uci, "b1c3")
        XCTAssertEqual(record.moves[2].postMilliseconds, 30)
        XCTAssertEqual(record.moves.map(\.san), ["e4", "c5", "Nc3"])
        XCTAssertFalse(record.anomalies.contains { $0.text.contains("not posted by this client") })
    }

    /// Plan §6.1 B: a move on our side that this client never posted means
    /// another client is playing on the account.
    func testOurSideMoveNotPostedHereIsAnAnomaly() throws {
        var entries = syntheticJournal(gameID: "g1", createdAt: createdAt, tokens: ["e2e4", "e7e5"], finish: nil)
        entries.append(.init(at: Date(timeIntervalSince1970: Double(createdAt) / 1000 + 100), event: .streamLine(raw: Self.stateJSON(["e2e4", "e7e5", "d2d4"]))))
        let record = try LichessBotRecordBuilder.build(
            gameID: "g1", journal: .init(elements: entries, droppedTrailingByteCount: 0),
            export: nil, exportUnavailableReason: "test", ourAccountID: botID, checkedAt: Date()
        )
        XCTAssertEqual(record.anomalies.filter { $0.text.contains("not posted by this client") }.count, 1)
        XCTAssertNil(record.moves[2].decision)
    }

    func testNotOurGameIsRefused() throws {
        let entries = syntheticJournal(gameID: "g1", createdAt: createdAt, tokens: [], finish: nil)
        XCTAssertThrowsError(try LichessBotRecordBuilder.build(
            gameID: "g1", journal: .init(elements: entries, droppedTrailingByteCount: 0),
            export: nil, exportUnavailableReason: nil, ourAccountID: "someoneelse", checkedAt: Date()
        )) { error in
            XCTAssertEqual(error as? LichessBotRecordError, .notOurGame(gameID: "g1", ourAccountID: "someoneelse"))
        }
    }

    func testResultMapping() {
        XCTAssertEqual(LichessBotRecordBuilder.pgnResult(status: "mate", winner: "black"), "0-1")
        XCTAssertEqual(LichessBotRecordBuilder.pgnResult(status: "stalemate", winner: nil), "1/2-1/2")
        XCTAssertEqual(LichessBotRecordBuilder.pgnResult(status: "outoftime", winner: nil), "1/2-1/2", "flag against insufficient material is a draw (E10)")
        XCTAssertEqual(LichessBotRecordBuilder.pgnResult(status: "aborted", winner: nil), "*")
        XCTAssertEqual(LichessBotRecordBuilder.pgnResult(status: "somethingNew", winner: nil), "*")
        XCTAssertNil(LichessBotRecordBuilder.ourScore(status: "aborted", winner: nil, ourColor: .white))
        XCTAssertEqual(LichessBotRecordBuilder.ourScore(status: "draw", winner: nil, ourColor: .white), 0.5)
        XCTAssertEqual(LichessBotRecordBuilder.ourScore(status: "resign", winner: "black", ourColor: .white), 0)
    }

    // MARK: - Finalize, files and index

    private func finalizeSyntheticGame(_ store: LichessBotRecordStore, gameID: String, createdAt: Int64, tokens: [String]) async throws -> LichessBotFinalizedGame {
        try writeJournal(syntheticJournal(gameID: gameID, createdAt: createdAt, tokens: tokens, finish: ("resign", "white")), gameID: gameID)
        return try await store.finalize(
            gameID: gameID,
            export: export(gameID: gameID, createdAt: createdAt, tokens: tokens, status: "resign", winner: "white"),
            exportUnavailableReason: nil
        )
    }

    /// Finalize writes the record, PGN and journal into Games/YYYY/MM/ and
    /// leaves nothing else behind: no temporary files, nothing in
    /// InProgress/.
    func testFinalizeFilesExactlyTheGamesArtifacts() async throws {
        let store = makeStore()
        let finalized = try await finalizeSyntheticGame(store, gameID: "g1", createdAt: createdAt, tokens: shortGame)
        let stem = LichessBotDataDirectory.fileStem(gameID: "g1", createdAt: Date(timeIntervalSince1970: Double(createdAt) / 1000))
        let folder = directory.gamesMonthDirectory(createdAt: Date(timeIntervalSince1970: Double(createdAt) / 1000))
        let files = try FileManager.default.contentsOfDirectory(atPath: folder.path).sorted()
        XCTAssertEqual(files, ["\(stem).journal.jsonl", "\(stem).json", "\(stem).pgn"].sorted())
        XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath: directory.inProgressDirectory.path), [])
        XCTAssertEqual(finalized.journalURL.lastPathComponent, "\(stem).journal.jsonl")
        let remaining = try await store.inProgressGameIDs()
        XCTAssertEqual(remaining, [])

        let reread = try LichessBotIndex.readRecord(at: finalized.recordURL)
        XCTAssertEqual(reread, finalized.record)
        let pgn = try String(contentsOf: finalized.pgnURL, encoding: .utf8)
        XCTAssertEqual(PGNImporter.sanTokens(from: pgn.components(separatedBy: "\n\n").dropFirst().joined(separator: "\n")), try Self.sanMoves(shortGame))
    }

    func testIncrementalIndexEqualsRebuild() async throws {
        let store = makeStore()
        _ = try await finalizeSyntheticGame(store, gameID: "g1", createdAt: createdAt, tokens: shortGame)
        _ = try await finalizeSyntheticGame(store, gameID: "g2", createdAt: createdAt + 3_600_000, tokens: ["d2d4", "d7d5"])
        _ = try await finalizeSyntheticGame(store, gameID: "g3", createdAt: createdAt + 7_200_000, tokens: ["c2c4"])
        let incremental = try await store.loadIndex()
        XCTAssertEqual(incremental.rows.map(\.gameID), ["g3", "g2", "g1"])
        let rebuilt = try await store.rebuildIndex()
        XCTAssertEqual(incremental, rebuilt)
    }

    func testMissingOrStaleIndexIsRebuilt() async throws {
        let store = makeStore()
        _ = try await finalizeSyntheticGame(store, gameID: "g1", createdAt: createdAt, tokens: shortGame)
        try FileManager.default.removeItem(at: directory.indexURL)
        let afterDelete = try await store.loadIndex()
        XCTAssertEqual(afterDelete.rows.map(\.gameID), ["g1"])

        try Data("{ not an index".utf8).write(to: directory.indexURL)
        let afterCorruption = try await store.loadIndex()
        XCTAssertEqual(afterCorruption.rows.map(\.gameID), ["g1"])
    }

    // MARK: - End to end through a game session

    /// A real game session writes the journal through the journal writer;
    /// finalize turns it into a record that agrees with the export.
    func testSessionJournalFinalizesIntoAMatchingRecord() async throws {
        let server = try LichessBotFakeGameServer()
        await server.setScript(afterOurMoves: [.opponentReplies, .opponentProposesTakeback(plies: 2), .opponentReplies, .finish(status: "resign", winner: "white")])
        let fileQueue = LichessBotFileQueue()
        let writeFailures = SyncBox<[String]>([])
        let finishedGames = SyncBox<[String]>([])
        let writer = LichessBotJournalWriter(
            directory: directory,
            fileQueue: fileQueue,
            onWriteFailure: { gameID, error in writeFailures.modify { $0.append("\(gameID): \(error)") } },
            onGameFinished: { gameID in finishedGames.modify { $0.append(gameID) } }
        )
        let recorder = LichessBotRecordingGameObserver()
        var settings = LichessBotSettings.testBaseline()
        settings.play.maxTakebacksAcceptedPerGame = 1
        let frozen = settings
        let source = LichessBotScriptedMoveSource()
        let session = LichessBotGameSession(
            gameID: LichessBotFakeGameServer.gameID,
            ourAccountID: LichessBotFakeGameServer.botID,
            api: server,
            moveSource: source,
            latestMoveSource: { source },
            settingsProvider: { frozen },
            observer: LichessBotGameObserverFanOut(observers: [writer, recorder]),
            time: LichessBotManualTime(),
            onTurnStatus: { _, _ in },
            carryover: .newGame
        )
        await session.run()
        XCTAssertEqual(writeFailures.value, [])
        XCTAssertEqual(finishedGames.value, [LichessBotFakeGameServer.gameID])

        let finalLine = try XCTUnwrap(recorder.receivedLines.last)
        let finalState = try XCTUnwrap({ () throws -> LichessBotGameState? in
            if case .gameState(let state) = try LichessBotGameStreamLine.decode(Data(finalLine.utf8)) { return state }
            return nil
        }())
        let tokens = finalState.moveTokens
        let store = makeStore()
        let finalized = try await store.finalize(
            gameID: LichessBotFakeGameServer.gameID,
            export: export(gameID: LichessBotFakeGameServer.gameID, createdAt: 1_700_000_000_000, tokens: tokens, status: "resign", winner: "white"),
            exportUnavailableReason: nil
        )
        let record = finalized.record
        XCTAssertEqual(record.reconciliation.outcome, .matched, "\(record.reconciliation.mismatches)")
        XCTAssertEqual(record.moves.map(\.uciAsGiven), tokens)
        XCTAssertEqual(record.retractions.count, 2)
        XCTAssertTrue(record.moves.filter(\.ours).allSatisfy { $0.decision != nil && $0.postMilliseconds != nil })
        XCTAssertEqual(record.anomalies.map(\.text), [])
        XCTAssertEqual(record.builds, [BuildInfo.buildNumber])
    }

    // MARK: - Reconciler

    private final class ScriptedExportAPI: LichessBotExportAPI, @unchecked Sendable {
        let responses: SyncBox<[Result<Data, Error>]>
        let calls = SyncBox<[Duration]>([])
        private let time: LichessBotManualTime

        init(time: LichessBotManualTime, responses: [Result<Data, Error>]) {
            self.time = time
            self.responses = SyncBox(responses)
        }

        func exportGame(gameID: String) async throws -> Data {
            calls.modify { $0.append(time.now()) }
            let next = responses.mutate { list -> Result<Data, Error>? in
                list.isEmpty ? nil : list.removeFirst()
            }
            guard let next else {
                throw LichessBotGateError.closed(reason: "script exhausted")
            }
            return try next.get()
        }
    }

    private func makeReconciler(api: ScriptedExportAPI, time: LichessBotManualTime, active: Set<String> = [], events: SyncBox<[LichessBotReconcilerEvent]>) -> LichessBotReconciler {
        LichessBotReconciler(
            api: api,
            store: makeStore(),
            time: time,
            settingsProvider: { LichessBotSettings.testBaseline() },
            hasLiveFilingOwner: { active.contains($0) },
            onEvent: { event in events.modify { $0.append(event) } }
        )
    }

    private func waitUntil(_ description: String, advancing time: LichessBotManualTime, by step: Duration = .seconds(1), _ condition: () async -> Bool) async throws {
        for _ in 0..<4000 {
            if await condition() { return }
            time.advance(by: step)
            try await Task.sleep(for: .milliseconds(2))
        }
        XCTFail("timed out waiting until \(description)")
        throw CancellationError()
    }

    private func finalizedCount(_ events: SyncBox<[LichessBotReconcilerEvent]>) -> Int {
        events.value.filter { event in
            if case .finalized = event { return true }
            return false
        }.count
    }

    /// E23: an export that still says `started` is retried until it shows
    /// the finish, at least the configured spacing apart.
    func testReconcilerRetriesALaggingExport() async throws {
        try writeJournal(syntheticJournal(gameID: "g1", createdAt: createdAt, tokens: shortGame, finish: ("resign", "white")), gameID: "g1")
        let time = LichessBotManualTime()
        let api = ScriptedExportAPI(time: time, responses: [
            .success(try exportData(gameID: "g1", createdAt: createdAt, tokens: shortGame, status: "started", winner: nil)),
            .success(try exportData(gameID: "g1", createdAt: createdAt, tokens: shortGame, status: "started", winner: nil)),
            .success(try exportData(gameID: "g1", createdAt: createdAt, tokens: shortGame, status: "resign", winner: "white")),
        ])
        let events = SyncBox<[LichessBotReconcilerEvent]>([])
        let reconciler = makeReconciler(api: api, time: time, events: events)
        await reconciler.enqueue(gameID: "g1")
        let run = Task { await reconciler.run() }
        try await waitUntil("the game is finalized", advancing: time) { finalizedCount(events) == 1 }
        run.cancel()
        await run.value
        let calls = api.calls.value
        XCTAssertEqual(calls.count, 3)
        let spacing = Duration.seconds(LichessBotSettings().connection.exportMinimumSpacingSeconds)
        for (earlier, later) in zip(calls, calls.dropFirst()) {
            XCTAssertGreaterThanOrEqual(later - earlier, spacing)
        }
        let queued = await reconciler.queuedGameIDs
        XCTAssertEqual(queued, [])
    }

    func testReconcilerGivesUpToUnreconciledAfterTheWindow() async throws {
        try writeJournal(syntheticJournal(gameID: "g1", createdAt: createdAt, tokens: shortGame, finish: ("resign", "white")), gameID: "g1")
        let time = LichessBotManualTime()
        let live = try exportData(gameID: "g1", createdAt: createdAt, tokens: shortGame, status: "started", winner: nil)
        let api = ScriptedExportAPI(time: time, responses: Array(repeating: .success(live), count: 200))
        let events = SyncBox<[LichessBotReconcilerEvent]>([])
        let reconciler = makeReconciler(api: api, time: time, events: events)
        await reconciler.enqueue(gameID: "g1")
        let run = Task { await reconciler.run() }
        try await waitUntil("the game is marked unreconciled", advancing: time, by: .seconds(5)) {
            await reconciler.unreconciledGameIDs == ["g1"]
        }
        run.cancel()
        await run.value
        let inProgress = try await makeStore().inProgressGameIDs()
        XCTAssertEqual(inProgress, ["g1"], "an unreconciled journal stays in InProgress")
    }

    /// Lichess keeps no export for some aborted games; the journal alone
    /// becomes the record.
    func testMissingExportForAnAbortedGameFinalizesFromTheJournal() async throws {
        try writeJournal(syntheticJournal(gameID: "g1", createdAt: createdAt, tokens: [], finish: ("aborted", nil)), gameID: "g1")
        let time = LichessBotManualTime()
        let api = ScriptedExportAPI(time: time, responses: [.failure(LichessBotAPIError.http(status: 404, message: "Not found"))])
        let events = SyncBox<[LichessBotReconcilerEvent]>([])
        let reconciler = makeReconciler(api: api, time: time, events: events)
        await reconciler.enqueue(gameID: "g1")
        let run = Task { await reconciler.run() }
        try await waitUntil("the game is finalized", advancing: time) { finalizedCount(events) == 1 }
        run.cancel()
        await run.value
        let record = try XCTUnwrap(events.value.compactMap { event -> LichessBotGameRecord? in
            if case .finalized(let finalized) = event { return finalized.record }
            return nil
        }.first)
        XCTAssertEqual(record.reconciliation.outcome, .exportUnavailable)
        XCTAssertEqual(record.outcome.pgnResult, "*")
    }

    func testActiveGamesAreLeftToTheirSession() async throws {
        try writeJournal(syntheticJournal(gameID: "g1", createdAt: createdAt, tokens: shortGame, finish: nil), gameID: "g1")
        let time = LichessBotManualTime()
        let api = ScriptedExportAPI(time: time, responses: [])
        let events = SyncBox<[LichessBotReconcilerEvent]>([])
        let reconciler = makeReconciler(api: api, time: time, active: ["g1"], events: events)
        await reconciler.enqueue(gameID: "g1")
        let run = Task { await reconciler.run() }
        try await waitUntil("the game leaves the queue", advancing: time) { await reconciler.queuedGameIDs.isEmpty }
        run.cancel()
        await run.value
        XCTAssertEqual(api.calls.value.count, 0)
    }

    private func quarantineEvents(_ events: SyncBox<[LichessBotReconcilerEvent]>) -> [(gameID: String, reason: String)] {
        events.value.compactMap { event in
            if case .quarantined(let gameID, let reason) = event { return (gameID, reason) }
            return nil
        }
    }

    /// A game already filed and handed to the reconciler again (launch
    /// recovery listing it just before it was filed, or the post-game chat
    /// wait ending after another path filed it) is recognized as filed: no
    /// second export, no quarantine alarm for a game that is safely on disk.
    func testEnqueueingAnAlreadyFiledGameIsNotQuarantined() async throws {
        try writeJournal(syntheticJournal(gameID: "g1", createdAt: createdAt, tokens: shortGame, finish: ("resign", "white")), gameID: "g1")
        let time = LichessBotManualTime()
        let terminal = try exportData(gameID: "g1", createdAt: createdAt, tokens: shortGame, status: "resign", winner: "white")
        let api = ScriptedExportAPI(time: time, responses: [.success(terminal), .success(terminal)])
        let events = SyncBox<[LichessBotReconcilerEvent]>([])
        let reconciler = makeReconciler(api: api, time: time, events: events)
        await reconciler.enqueue(gameID: "g1")
        let run = Task { await reconciler.run() }
        try await waitUntil("the game is finalized", advancing: time) { finalizedCount(events) == 1 }
        await reconciler.enqueue(gameID: "g1")
        try await waitUntil("the second enqueue is settled", advancing: time) {
            let queued = await reconciler.queuedGameIDs
            let quarantined = await reconciler.quarantinedGameIDs
            return queued.isEmpty || !quarantined.isEmpty
        }
        run.cancel()
        await run.value
        XCTAssertEqual(quarantineEvents(events).map(\.gameID), [])
        let quarantined = await reconciler.quarantinedGameIDs
        XCTAssertEqual(quarantined, [])
        XCTAssertEqual(finalizedCount(events), 1)
        XCTAssertEqual(api.calls.value.count, 1, "a filed game needs no second export")
        let inProgress = try await makeStore().inProgressGameIDs()
        XCTAssertEqual(inProgress, [])
    }

    /// A game with no journal in InProgress/ and no filed record has nothing
    /// to file: it is reported, never exported (an export alone can't make a
    /// record), and never alarmed as "won't be retried until the bot next
    /// goes online" — no later recovery would find it either.
    func testAGameWithNoJournalAndNoRecordIsReportedWithoutAnExport() async throws {
        try directory.createDirectories()
        let time = LichessBotManualTime()
        let api = ScriptedExportAPI(time: time, responses: [
            .success(try exportData(gameID: "g9", createdAt: createdAt, tokens: shortGame, status: "resign", winner: "white")),
        ])
        let events = SyncBox<[LichessBotReconcilerEvent]>([])
        let reconciler = makeReconciler(api: api, time: time, events: events)
        await reconciler.enqueue(gameID: "g9")
        let run = Task { await reconciler.run() }
        try await waitUntil("the game is no longer due", advancing: time) { await reconciler.dueGameIDs.isEmpty }
        run.cancel()
        await run.value
        XCTAssertEqual(api.calls.value.count, 0)
        XCTAssertEqual(finalizedCount(events), 0)
        XCTAssertEqual(quarantineEvents(events).map(\.gameID), [])
    }

    /// A record file for the game exists but doesn't decode: the game is
    /// quarantined with that reason, without an export, rather than treated
    /// as a game nobody ever filed.
    func testAGameWhoseFiledRecordDoesNotDecodeIsQuarantinedWithThatReason() async throws {
        try directory.createDirectories()
        let created = Date(timeIntervalSince1970: Double(createdAt) / 1000)
        let folder = directory.gamesMonthDirectory(createdAt: created)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        let stem = try LichessBotDataDirectory.validatedFileStem(gameID: "g1", createdAt: created)
        try Data("not a record".utf8).write(to: folder.appendingPathComponent("\(stem).json", isDirectory: false))
        let time = LichessBotManualTime()
        let api = ScriptedExportAPI(time: time, responses: [
            .success(try exportData(gameID: "g1", createdAt: createdAt, tokens: shortGame, status: "resign", winner: "white")),
        ])
        let events = SyncBox<[LichessBotReconcilerEvent]>([])
        let reconciler = makeReconciler(api: api, time: time, events: events)
        await reconciler.enqueue(gameID: "g1")
        let run = Task { await reconciler.run() }
        try await waitUntil("the game is no longer due", advancing: time) { await reconciler.dueGameIDs.isEmpty }
        run.cancel()
        await run.value
        XCTAssertEqual(api.calls.value.count, 0)
        let quarantined = quarantineEvents(events)
        XCTAssertEqual(quarantined.map(\.gameID), ["g1"])
        XCTAssertTrue(quarantined.first?.reason.contains("doesn't decode") == true, "\(quarantined)")
    }

    /// An export double that hands the game to a live owner while its export
    /// is in flight: a resumed game whose session is still starting, or a
    /// finished game the controller has begun collecting post-game chat for.
    private final class OwnerTakingExportAPI: LichessBotExportAPI, @unchecked Sendable {
        let owners: SyncBox<Set<String>>
        let calls = SyncBox<Int>(0)
        private let response: Data

        init(owners: SyncBox<Set<String>>, response: Data) {
            self.owners = owners
            self.response = response
        }

        func exportGame(gameID: String) async throws -> Data {
            calls.modify { $0 += 1 }
            owners.modify { $0.insert(gameID) }
            return response
        }
    }

    /// A reconciler whose live owners are whatever `owners` holds at each check.
    private func makeReconciler(api: any LichessBotExportAPI, time: LichessBotManualTime, owners: SyncBox<Set<String>>, events: SyncBox<[LichessBotReconcilerEvent]>) -> LichessBotReconciler {
        LichessBotReconciler(
            api: api,
            store: makeStore(),
            time: time,
            settingsProvider: { LichessBotSettings.testBaseline() },
            hasLiveFilingOwner: { owners.value.contains($0) },
            onEvent: { event in events.modify { $0.append(event) } }
        )
    }

    /// The owner check runs again after the export: a game that gained a live
    /// owner while its export was in flight is left to that owner (which
    /// enqueues it when done) instead of being filed under it.
    func testAGameTakenByAnOwnerDuringItsExportIsNotFiled() async throws {
        try writeJournal(syntheticJournal(gameID: "g1", createdAt: createdAt, tokens: shortGame, finish: ("resign", "white")), gameID: "g1")
        let time = LichessBotManualTime()
        let owners = SyncBox<Set<String>>([])
        let api = OwnerTakingExportAPI(owners: owners, response: try exportData(gameID: "g1", createdAt: createdAt, tokens: shortGame, status: "resign", winner: "white"))
        let events = SyncBox<[LichessBotReconcilerEvent]>([])
        let reconciler = makeReconciler(api: api, time: time, owners: owners, events: events)
        await reconciler.enqueue(gameID: "g1")
        let run = Task { await reconciler.run() }
        try await waitUntil("the attempt is settled", advancing: time) {
            let queued = await reconciler.queuedGameIDs
            return queued.isEmpty || finalizedCount(events) > 0
        }
        run.cancel()
        await run.value
        XCTAssertEqual(api.calls.value, 1)
        XCTAssertEqual(finalizedCount(events), 0)
        let inProgress = try await makeStore().inProgressGameIDs()
        XCTAssertEqual(inProgress, ["g1"], "the owner files it later; its journal stays in InProgress")
    }

    // MARK: - Instance lock (plan §6.1 A)

    func testInstanceLockIsExclusiveAndNamesTheHolder() throws {
        let holder = LichessBotInstanceLockHolder.current
        let first = try LichessBotInstanceLock.acquire(at: directory.lockURL, holder: holder)
        XCTAssertThrowsError(try LichessBotInstanceLock.acquire(at: directory.lockURL, holder: holder)) { error in
            guard case .heldByAnotherProcess(let description)? = error as? LichessBotInstanceLockError else {
                return XCTFail("expected heldByAnotherProcess, got \(error)")
            }
            XCTAssertTrue(description.contains("pid \(holder.pid)"), description)
        }
        first.release()
        let second = try LichessBotInstanceLock.acquire(at: directory.lockURL, holder: holder)
        second.release()
    }

    // MARK: - Protocol log and redaction (plan §10.3)

    func testRedaction() {
        XCTAssertEqual(LichessBotRedaction.redact("token lip_AbC123_x-y used"), "token [REDACTED] used")
        XCTAssertEqual(LichessBotRedaction.redact("Authorization: Bearer abc.DEF-123"), "Authorization: [REDACTED]")
        XCTAssertEqual(LichessBotRedaction.redact("nothing secret"), "nothing secret")
    }

    func testProtocolLogWritesRedactedEntriesPerDay() async throws {
        let failures = SyncBox<[String]>([])
        let log = LichessBotProtocolLog(directory: directory, fileQueue: LichessBotFileQueue()) { error in
            failures.modify { $0.append(String(describing: error)) }
        }
        let day = Date()
        log.record(.request, "POST move failed with Bearer lip_SECRET", gameID: "g1", fields: ["status": "400", "auth": "lip_SECRET"], at: day)
        log.record(.rateLimit, "429; cooldown 60s", at: day)
        try await log.flush()
        let read = try await log.entries(on: day)
        XCTAssertEqual(failures.value, [])
        XCTAssertEqual(read.elements.map(\.kind), [.request, .rateLimit])
        XCTAssertFalse(read.elements[0].message.contains("SECRET"))
        XCTAssertEqual(read.elements[0].fields["auth"], "[REDACTED]")
        XCTAssertEqual(read.elements[0].gameID, "g1")
    }

    // MARK: - Settings store (plan §12.1)

    private func makeDefaults() throws -> UserDefaults {
        let suite = "LichessBotDataLayerTests-\(UUID().uuidString)"
        let defaults = try XCTUnwrap(UserDefaults(suiteName: suite))
        addTeardownBlock { defaults.removePersistentDomain(forName: suite) }
        return defaults
    }

    func testSettingsStoreRoundTripsAndStartsFromDefaults() throws {
        let defaults = try makeDefaults()
        XCTAssertEqual(try LichessBotSettingsStore.load(from: defaults), LichessBotSettings())
        var settings = LichessBotSettings()
        settings.challenge.maxConcurrentGames = 3
        settings.play.acceptDrawEnabled = true
        try LichessBotSettingsStore.save(settings, to: defaults)
        XCTAssertEqual(try LichessBotSettingsStore.load(from: defaults), settings)
    }

    func testUnreadableSettingsAreAnErrorNotDefaults() throws {
        let defaults = try makeDefaults()
        defaults.set(Data("garbage".utf8), forKey: LichessBotSettingsStore.defaultsKey)
        XCTAssertThrowsError(try LichessBotSettingsStore.load(from: defaults)) { error in
            guard case .unreadable? = error as? LichessBotSettingsStoreError else {
                return XCTFail("expected unreadable, got \(error)")
            }
        }
        try LichessBotSettingsStore.reset(in: defaults)
        XCTAssertEqual(try LichessBotSettingsStore.load(from: defaults), LichessBotSettings())
    }

    func testInvalidSettingsAreNotSaved() throws {
        let defaults = try makeDefaults()
        var settings = LichessBotSettings()
        settings.challenge.allowedSpeeds = []
        XCTAssertThrowsError(try LichessBotSettingsStore.save(settings, to: defaults))
        XCTAssertNil(defaults.data(forKey: LichessBotSettingsStore.defaultsKey))
    }

    // MARK: - PGN

    func testPGNTagsAndClocks() throws {
        let entries = syntheticJournal(gameID: "g1", createdAt: createdAt, tokens: shortGame, finish: ("resign", "white"))
        let record = try LichessBotRecordBuilder.build(
            gameID: "g1", journal: .init(elements: entries, droppedTrailingByteCount: 0),
            export: export(gameID: "g1", createdAt: createdAt, tokens: shortGame, status: "resign", winner: "white"),
            exportUnavailableReason: nil, ourAccountID: botID, checkedAt: Date()
        )
        let pgn = LichessBotPGNWriter.pgn(for: record)
        XCTAssertTrue(pgn.contains(#"[Result "1-0"]"#))
        XCTAssertTrue(pgn.contains(#"[White "DrewsChessMachine"]"#))
        XCTAssertTrue(pgn.contains(#"[TimeControl "180+2"]"#))
        XCTAssertTrue(pgn.contains(#"[Termination "Normal"]"#))
        XCTAssertTrue(pgn.contains(#"[DCMModelIDs "20260928-1-TEST"]"#))
        XCTAssertTrue(pgn.hasSuffix("1-0\n"))
        XCTAssertEqual(LichessBotPGNWriter.clockString(milliseconds: 3_725_400), "1:02:05")
        XCTAssertTrue(pgn.split(separator: "\n").allSatisfy { $0.count <= 80 || $0.contains("{") })
    }
}
