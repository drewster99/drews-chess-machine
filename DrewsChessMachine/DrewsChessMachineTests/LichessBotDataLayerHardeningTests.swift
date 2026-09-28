import XCTest
@testable import DrewsChessMachine

/// The Lichess bot data layer's crash, ordering and filing edge cases: torn
/// file tails, index failures after filing, held-move ordering, games that
/// went on after the journal stopped, header races, repeated filing,
/// terminal filing failures, UTC naming, explicit outcomes and index
/// staleness. Every test works in its own temporary directory.
final class LichessBotDataLayerHardeningTests: XCTestCase {

    private var tempRoot: URL!
    private static let botID = "drewschessmachine"
    private static let gameCreatedAt: Int64 = 1_759_000_000_000
    private static let shortGame = ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "g8f6"]
    private static let waitSteps = 4000

    override func setUpWithError() throws {
        tempRoot = FileManager.default.temporaryDirectory.appendingPathComponent("LichessBotDataLayerHardeningTests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: tempRoot, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try FileManager.default.removeItem(at: tempRoot)
    }

    private var directory: LichessBotDataDirectory { LichessBotDataDirectory(root: tempRoot) }

    private func makeStore() -> LichessBotRecordStore {
        LichessBotRecordStore(directory: directory, fileQueue: LichessBotFileQueue(), ourAccountID: Self.botID)
    }

    private func makeWriter(failures: SyncBox<[String]>) -> LichessBotJournalWriter {
        LichessBotJournalWriter(
            directory: directory,
            fileQueue: LichessBotFileQueue(),
            onWriteFailure: { gameID, error in failures.modify { $0.append("\(gameID): \(error)") } },
            onGameFinished: { _ in }
        )
    }

    // MARK: - Fixtures

    private static let generation = LichessBotGenerationInfo(
        generationID: 7, sourceKind: .champion, modelID: "20260928-1-TEST", trainingStep: nil,
        snapshotAt: Date(timeIntervalSince1970: 1_700_000_000), architectureSummary: "test", filePath: nil, fileSHA256: nil
    )

    private static func decision(_ uci: String) -> LichessBotMoveDecision {
        LichessBotMoveDecision(
            uci: uci, san: uci, chosenProbability: 0.5, topMoves: [],
            win: 0.3, draw: 0.4, loss: 0.3, temperature: 0.5, legalMoveCount: 20, randomish: false,
            encodeMilliseconds: 0.1, inferenceMilliseconds: 2, sampleMilliseconds: 0.1
        )
    }

    private static func stateJSON(_ tokens: [String], status: String = "started", winner: String? = nil) -> String {
        let winnerField = winner.map { #","winner":"\#($0)""# } ?? ""
        return #"{"type":"gameState","moves":"\#(tokens.joined(separator: " "))","wtime":170000,"btime":160000,"winc":2000,"binc":2000,"status":"\#(status)"\#(winnerField)}"#
    }

    private static func gameFullJSON(gameID: String, createdAt: Int64, white: String, tokens: [String] = []) -> String {
        #"{"type":"gameFull","id":"\#(gameID)","variant":{"key":"standard"},"clock":{"initial":180000,"increment":2000},"speed":"blitz","perf":{"name":"Blitz"},"rated":false,"createdAt":\#(createdAt),"white":{"id":"\#(white)","name":"\#(white)","title":"BOT","rating":1500},"black":{"id":"alice","name":"Alice","rating":1600},"initialFen":"startpos","state":\#(stateJSON(tokens))}"#
    }

    private static func header(gameID: String, at: Date) -> LichessBotJournalEntry {
        .init(at: at, event: .header(schemaVersion: LichessBotJournal.schemaVersion, gameID: gameID, build: 1, gitHash: "test", resumed: false))
    }

    /// DCM (white by default) decides and posts each white move before its
    /// echo; optionally ends with a finish.
    private func journal(gameID: String = "g1", tokens: [String], finish: (status: String, winner: String?)?, createdAt: Int64 = LichessBotDataLayerHardeningTests.gameCreatedAt, white: String = LichessBotDataLayerHardeningTests.botID) -> [LichessBotJournalEntry] {
        var at = Date(timeIntervalSince1970: Double(createdAt) / 1000)
        func next() -> Date {
            at = at.addingTimeInterval(1)
            return at
        }
        var entries: [LichessBotJournalEntry] = [
            Self.header(gameID: gameID, at: next()),
            .init(at: next(), event: .streamOpened(attempt: 0)),
            .init(at: next(), event: .streamLine(raw: Self.gameFullJSON(gameID: gameID, createdAt: createdAt, white: white))),
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

    private func writeJournal(_ entries: [LichessBotJournalEntry], gameID: String = "g1", trailing: Data = Data()) throws {
        var data = Data()
        for entry in entries {
            data.append(try LichessBotJSONLines.encodeLine(entry))
        }
        data.append(trailing)
        try directory.createDirectories()
        try data.write(to: directory.inProgressJournalURL(gameID: gameID))
    }

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

    private func exportJSON(gameID: String = "g1", tokens: [String], status: String, winner: String?, createdAt: Int64 = LichessBotDataLayerHardeningTests.gameCreatedAt, white: String = LichessBotDataLayerHardeningTests.botID, clocks: [Int]? = nil, variant: String = "standard", extraFields: String = "", includeSpeed: Bool = true) throws -> String {
        let winnerField = winner.map { #""winner":"\#($0)","# } ?? ""
        let clocksField = clocks.map { #""clocks":[\#($0.map(String.init).joined(separator: ","))],"# } ?? ""
        let speedField = includeSpeed ? #""speed":"blitz","# : ""
        return #"{"id":"\#(gameID)","rated":false,"variant":"\#(variant)",\#(speedField)"perf":"blitz","createdAt":\#(createdAt),"status":"\#(status)",\#(winnerField)\#(clocksField)\#(extraFields)"players":{"white":{"user":{"name":"\#(white)","title":"BOT","id":"\#(white)"},"rating":1500},"black":{"user":{"name":"Alice","id":"alice"},"rating":1600}},"moves":"\#(try Self.sanMoves(tokens).joined(separator: " "))","clock":{"initial":180,"increment":2}}"#
    }

    private func export(gameID: String = "g1", tokens: [String], status: String, winner: String?, createdAt: Int64 = LichessBotDataLayerHardeningTests.gameCreatedAt, clocks: [Int]? = nil) throws -> LichessBotGameExport {
        try LichessBotGameExport.decode(Data(exportJSON(gameID: gameID, tokens: tokens, status: status, winner: winner, createdAt: createdAt, clocks: clocks).utf8))
    }

    private func fileGame(_ store: LichessBotRecordStore, gameID: String, createdAt: Int64) async throws -> LichessBotFinalizedGame {
        try writeJournal(journal(gameID: gameID, tokens: Self.shortGame, finish: ("resign", "white"), createdAt: createdAt), gameID: gameID)
        return try await store.finalize(gameID: gameID, export: export(gameID: gameID, tokens: Self.shortGame, status: "resign", winner: "white", createdAt: createdAt), exportUnavailableReason: nil)
    }

    // MARK: - D1 torn tails

    func testAppendAfterACrashTruncatedJournalLineCutsItAndRecordsIt() async throws {
        let original = Array(journal(tokens: ["e2e4"], finish: nil))
        let garbage = Data(#"{"at":"2026-09-28T01:02:03.456Z","event":{"streamLi"#.utf8)
        try writeJournal(original, trailing: garbage)
        let failures = SyncBox<[String]>([])
        let writer = makeWriter(failures: failures)
        await writer.gameEvent(gameID: "g1", .action("after the crash"))
        XCTAssertEqual(failures.value, [])
        let read = try LichessBotJournal.read(directory.inProgressJournalURL(gameID: "g1"))
        XCTAssertEqual(read.droppedTrailingByteCount, 0)
        XCTAssertEqual(Array(read.elements.prefix(original.count)), original)
        let added = read.elements.dropFirst(original.count).map(\.event)
        guard added.count == 3,
              case .header(_, _, _, _, let resumed) = added[0],
              case .anomaly(let note) = added[1],
              case .action("after the crash") = added[2] else {
            return XCTFail("unexpected appended entries: \(added)")
        }
        XCTAssertTrue(resumed)
        XCTAssertTrue(note.contains(garbage.base64EncodedString()), note)
    }

    func testCutUnterminatedFinalLineAcrossScanChunks() throws {
        let url = tempRoot.appendingPathComponent("tail.jsonl", isDirectory: false)
        let kept = Data(#"{"a":1}"#.utf8) + Data([UInt8(ascii: "\n")])
        let longTail = Data(repeating: UInt8(ascii: "x"), count: 200_000)
        try (kept + longTail).write(to: url)
        XCTAssertEqual(try LichessBotJSONLines.cutUnterminatedFinalLine(of: url), longTail)
        XCTAssertEqual(try Data(contentsOf: url), kept)
        XCTAssertEqual(try LichessBotJSONLines.cutUnterminatedFinalLine(of: url), Data(), "a clean file is left alone")
        try longTail.write(to: url)
        XCTAssertEqual(try LichessBotJSONLines.cutUnterminatedFinalLine(of: url), longTail, "no newline at all: the whole file is one fragment")
        XCTAssertEqual(try Data(contentsOf: url), Data())
    }

    func testProtocolLogAppendAfterATruncatedLineStaysReadable() async throws {
        let failures = SyncBox<[String]>([])
        let log = LichessBotProtocolLog(directory: directory, fileQueue: LichessBotFileQueue()) { error in
            failures.modify { $0.append(String(describing: error)) }
        }
        let day = Date()
        let url = log.fileURL(for: day)
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        var data = try LichessBotJSONLines.encodeLine(LichessBotProtocolEntry(at: day, kind: .game, gameID: nil, message: "before", fields: [:]))
        data.append(Data(#"{"at":"2026"#.utf8))
        try data.write(to: url)
        log.record(.game, "after", at: day)
        try await log.flush()
        let read = try await log.entries(on: day)
        XCTAssertEqual(failures.value, [])
        XCTAssertEqual(read.droppedTrailingByteCount, 0)
        XCTAssertEqual(read.elements.map(\.kind), [.game, .anomaly, .game])
        XCTAssertEqual(read.elements.last?.message, "after")
    }

    // MARK: - D2 index failures never undo filing

    func testUndecodableRecordIsLeftOutOfTheIndexAndReported() async throws {
        let bad = directory.gamesDirectory.appendingPathComponent("2020/01/20200101-000000-broken.json", isDirectory: false)
        try FileManager.default.createDirectory(at: bad.deletingLastPathComponent(), withIntermediateDirectories: true)
        try Data("{ not a record".utf8).write(to: bad)
        let store = makeStore()
        let filed = try await fileGame(store, gameID: "g1", createdAt: Self.gameCreatedAt)
        XCTAssertNil(filed.indexUpdateFailure)
        let index = try await store.loadIndex()
        XCTAssertEqual(index.rows.map(\.gameID), ["g1"])
        XCTAssertEqual(index.unreadableRecords.count, 1)
        XCTAssertTrue(index.unreadableRecords.allSatisfy { $0.path.hasSuffix("20200101-000000-broken.json") })
    }

    func testIndexWriteFailureAfterTheJournalMovedDoesNotFailFinalize() async throws {
        try directory.createDirectories()
        // index.json is a folder, so every index write fails.
        try FileManager.default.createDirectory(at: directory.indexURL, withIntermediateDirectories: true)
        let store = makeStore()
        let filed = try await fileGame(store, gameID: "g1", createdAt: Self.gameCreatedAt)
        XCTAssertNotNil(filed.indexUpdateFailure)
        XCTAssertTrue(FileManager.default.fileExists(atPath: filed.recordURL.path))
        XCTAssertTrue(FileManager.default.fileExists(atPath: filed.journalURL.path))
        let remaining = try await store.inProgressGameIDs()
        XCTAssertEqual(remaining, [])
    }

    // MARK: - D3 held move echoed before its POST

    func testHeldMoveEchoedBeforeItsPostKeepsItsDecision() throws {
        let start = Date(timeIntervalSince1970: Double(Self.gameCreatedAt) / 1000)
        func at(_ seconds: TimeInterval) -> Date { start.addingTimeInterval(seconds) }
        let entries: [LichessBotJournalEntry] = [
            Self.header(gameID: "g1", at: at(1)),
            .init(at: at(2), event: .streamOpened(attempt: 0)),
            .init(at: at(3), event: .streamLine(raw: Self.gameFullJSON(gameID: "g1", createdAt: Self.gameCreatedAt, white: Self.botID))),
            .init(at: at(4), event: .moveDecided(ply: 0, decision: Self.decision("e2e4"), generation: Self.generation)),
            .init(at: at(5), event: .streamLine(raw: Self.stateJSON(["e2e4"]))),
            .init(at: at(6), event: .movePosted(ply: 0, uci: "e2e4", offeringDraw: false, milliseconds: 40)),
            .init(at: at(7), event: .streamLine(raw: Self.stateJSON(["e2e4", "e7e5"]))),
            .init(at: at(8), event: .finished(status: "resign", winner: "white", localDrawCondition: nil)),
        ]
        let record = try LichessBotRecordBuilder.build(
            gameID: "g1", journal: .init(elements: entries, droppedTrailingByteCount: 0),
            export: nil, exportUnavailableReason: "test", ourAccountID: Self.botID, checkedAt: at(9)
        )
        XCTAssertEqual(record.moves[0].decision?.uci, "e2e4")
        XCTAssertEqual(record.moves[0].generationID, 7)
        XCTAssertEqual(record.moves[0].postMilliseconds, 40)
        XCTAssertFalse(record.anomalies.contains { $0.text.contains("not posted by this client") }, "\(record.anomalies)")
    }

    // MARK: - D4 the export extends a short journal

    func testExportExtendsAJournalThatStoppedEarly() throws {
        var entries = journal(tokens: Self.shortGame, finish: nil)
        // The app went down after posting its last move and before
        // journaling its echo; the game went on without it.
        entries.removeLast(2)
        let clocks = [17000, 16900, 16800, 16700, 16600, 16500]
        let record = try LichessBotRecordBuilder.build(
            gameID: "g1", journal: .init(elements: entries, droppedTrailingByteCount: 0),
            export: export(tokens: Self.shortGame, status: "resign", winner: "white", clocks: clocks),
            exportUnavailableReason: nil, ourAccountID: Self.botID, checkedAt: Date()
        )
        XCTAssertEqual(record.outcome.plies, Self.shortGame.count)
        XCTAssertEqual(record.moves.map(\.uciAsGiven), Self.shortGame)
        XCTAssertEqual(record.moves.map(\.san), try Self.sanMoves(Self.shortGame))
        XCTAssertNil(record.moves[4].receivedAt)
        XCTAssertTrue(record.moves[4].ours)
        XCTAssertEqual(record.moves[4].postMilliseconds, 40, "the POST journaled before the crash attaches to the export's move")
        XCTAssertEqual(record.moves[4].decision?.uci, "f1c4")
        XCTAssertEqual(record.moves[4].whiteClockMilliseconds, 166000)
        XCTAssertEqual(record.moves[5].blackClockMilliseconds, 165000)
        XCTAssertFalse(record.moves[5].ours)
        XCTAssertFalse(record.anomalies.contains { $0.text.contains("not posted by this client") }, "\(record.anomalies)")
        XCTAssertEqual(record.reconciliation.outcome, .corrected)
        XCTAssertTrue(record.reconciliation.mismatches.contains { $0.contains("journal ends at ply 4") }, "\(record.reconciliation.mismatches)")
        let pgn = LichessBotPGNWriter.pgn(for: record)
        XCTAssertEqual(PGNImporter.sanTokens(from: pgn.components(separatedBy: "\n\n").dropFirst().joined(separator: "\n")), try Self.sanMoves(Self.shortGame))
    }

    // MARK: - D6 header bookkeeping

    func testLateWriteAfterMarkFinalizedDoesNotRecreateTheJournal() async throws {
        let failures = SyncBox<[String]>([])
        let writer = makeWriter(failures: failures)
        await writer.gameEvent(gameID: "g1", .action("first"))
        let url = directory.inProgressJournalURL(gameID: "g1")
        try FileManager.default.removeItem(at: url)
        writer.markFinalized(gameID: "g1")
        // A write already past the finalized check when the game was filed.
        try await writer.append([LichessBotJournalEntry(at: Date(), event: .action("late"))], gameID: "g1", synchronize: false)
        XCTAssertFalse(FileManager.default.fileExists(atPath: url.path))
        XCTAssertEqual(failures.value, [])
    }

    func testConcurrentAppendsKeepEveryEntryUnderOneHeader() async throws {
        let failures = SyncBox<[String]>([])
        let writer = makeWriter(failures: failures)
        let count = 200
        await withTaskGroup(of: Void.self) { group in
            for index in 0..<count {
                group.addTask { await writer.gameEvent(gameID: "g1", .action("entry \(index)")) }
            }
        }
        XCTAssertEqual(failures.value, [])
        let read = try LichessBotJournal.read(directory.inProgressJournalURL(gameID: "g1"))
        let headers = read.elements.filter { if case .header = $0.event { return true }; return false }
        XCTAssertEqual(headers.count, 1)
        guard case .header = read.elements.first?.event else { return XCTFail("the first line is not the header") }
        let actions = Set(read.elements.compactMap { entry -> String? in
            if case .action(let text) = entry.event { return text }
            return nil
        })
        XCTAssertEqual(actions, Set((0..<count).map { "entry \($0)" }))
    }

    // MARK: - D7 filing the same game again

    func testRepeatedFragmentsGetDistinctNames() async throws {
        let store = makeStore()
        _ = try await fileGame(store, gameID: "g1", createdAt: Self.gameCreatedAt)
        var names: [String] = []
        for note in ["late one", "late two"] {
            try writeJournal([Self.header(gameID: "g1", at: Date(timeIntervalSince1970: 1_759_000_100)), .init(at: Date(timeIntervalSince1970: 1_759_000_101), event: .action(note))])
            let filed = try await store.finalize(gameID: "g1", export: export(tokens: Self.shortGame, status: "resign", winner: "white"), exportUnavailableReason: nil)
            names.append(filed.journalURL.lastPathComponent)
        }
        XCTAssertEqual(Set(names).count, 2)
        XCTAssertTrue(names.allSatisfy { $0.contains("fragment") })
        let remaining = try await store.inProgressGameIDs()
        XCTAssertEqual(remaining, [])
    }

    func testSecondJournalRebuildsFromEveryJournal() async throws {
        let store = makeStore()
        let first = try await fileGame(store, gameID: "g1", createdAt: Self.gameCreatedAt)
        let later = Date(timeIntervalSince1970: Double(Self.gameCreatedAt) / 1000 + 3600)
        try writeJournal([
            Self.header(gameID: "g1", at: later),
            .init(at: later.addingTimeInterval(1), event: .streamOpened(attempt: 0)),
            .init(at: later.addingTimeInterval(2), event: .streamLine(raw: Self.gameFullJSON(gameID: "g1", createdAt: Self.gameCreatedAt, white: Self.botID, tokens: Self.shortGame))),
            .init(at: later.addingTimeInterval(3), event: .finished(status: "resign", winner: "white", localDrawCondition: nil)),
        ])
        let second = try await store.finalize(gameID: "g1", export: export(tokens: Self.shortGame, status: "resign", winner: "white"), exportUnavailableReason: nil)
        XCTAssertEqual(second.recordURL, first.recordURL)
        XCTAssertEqual(second.record.moves.map(\.uciAsGiven), Self.shortGame)
        XCTAssertTrue(second.record.moves.filter(\.ours).allSatisfy { $0.decision != nil && $0.postMilliseconds != nil })
        XCTAssertNotEqual(second.journalURL, first.journalURL)
        XCTAssertTrue(FileManager.default.fileExists(atPath: first.journalURL.path))
        XCTAssertTrue(FileManager.default.fileExists(atPath: second.journalURL.path))
    }

    func testFragmentIsFiledBeforeAnyBuild() async throws {
        let store = makeStore()
        let first = try await fileGame(store, gameID: "g1", createdAt: Self.gameCreatedAt)
        let original = try Data(contentsOf: first.recordURL)
        try writeJournal([Self.header(gameID: "g1", at: Date(timeIntervalSince1970: 1_759_000_100)), .init(at: Date(timeIntervalSince1970: 1_759_000_101), event: .action("late"))])
        // No speed: this export alone can't describe the game.
        let sparse = try LichessBotGameExport.decode(Data(exportJSON(tokens: Self.shortGame, status: "resign", winner: "white", includeSpeed: false).utf8))
        let filed = try await store.finalize(gameID: "g1", export: sparse, exportUnavailableReason: nil)
        XCTAssertTrue(filed.journalURL.lastPathComponent.contains("fragment"))
        XCTAssertEqual(try Data(contentsOf: first.recordURL), original)
    }

    // MARK: - D8 terminal failures

    private final class CountingExportAPI: LichessBotExportAPI {
        let calls = SyncBox(0)
        private let data: Data
        init(data: Data) { self.data = data }
        func exportGame(gameID: String) async throws -> Data {
            calls.modify { $0 += 1 }
            return data
        }
    }

    func testNotOurGameIsQuarantinedOnceAndNeverRetried() async throws {
        try writeJournal(journal(tokens: Self.shortGame, finish: ("resign", "white"), white: "carol"))
        let api = CountingExportAPI(data: Data(try exportJSON(tokens: Self.shortGame, status: "resign", winner: "white", white: "carol").utf8))
        let time = LichessBotManualTime()
        let events = SyncBox<[LichessBotReconcilerEvent]>([])
        let reconciler = LichessBotReconciler(
            api: api, store: makeStore(), time: time, settingsProvider: { LichessBotSettings() },
            isGameActive: { _ in false }, onEvent: { event in events.modify { $0.append(event) } }
        )
        await reconciler.enqueue(gameID: "g1")
        let run = Task { await reconciler.run() }
        var quarantined: [String] = []
        for _ in 0..<Self.waitSteps where quarantined.isEmpty {
            quarantined = await reconciler.quarantinedGameIDs
            time.advance(by: .seconds(1))
            try await Task.sleep(for: .milliseconds(2))
        }
        XCTAssertEqual(quarantined, ["g1"])
        time.advance(by: LichessBotReconciler.unreconciledRetryInterval * 3)
        try await Task.sleep(for: .milliseconds(50))
        run.cancel()
        await run.value
        XCTAssertEqual(api.calls.value, 1)
        let kinds = events.value.map { event -> String in
            switch event {
            case .quarantined: return "quarantined"
            case .finalizeFailed: return "finalizeFailed"
            case .unreconciled: return "unreconciled"
            default: return "other"
            }
        }
        XCTAssertEqual(kinds.filter { $0 == "quarantined" }.count, 1)
        XCTAssertFalse(kinds.contains("finalizeFailed"))
        XCTAssertFalse(kinds.contains("unreconciled"))
    }

    // MARK: - D9 UTC

    /// Half an hour after midnight UTC: the previous day west of Greenwich.
    private static let justAfterUTCMidnight: Int64 = 1_758_933_000_000

    func testPGNDateIsUTC() throws {
        let entries = journal(tokens: Self.shortGame, finish: ("resign", "white"), createdAt: Self.justAfterUTCMidnight)
        let record = try LichessBotRecordBuilder.build(
            gameID: "g1", journal: .init(elements: entries, droppedTrailingByteCount: 0),
            export: nil, exportUnavailableReason: "test", ourAccountID: Self.botID, checkedAt: Date()
        )
        let pgn = LichessBotPGNWriter.pgn(for: record)
        XCTAssertTrue(pgn.contains(#"[Date "2025.09.27"]"#), pgn)
        XCTAssertTrue(pgn.contains(#"[UTCDate "2025.09.27"]"#), pgn)
    }

    func testProtocolLogFilesAreUTCDays() {
        let log = LichessBotProtocolLog(directory: directory, fileQueue: LichessBotFileQueue()) { _ in }
        let date = Date(timeIntervalSince1970: Double(Self.justAfterUTCMidnight) / 1000)
        XCTAssertEqual(log.fileURL(for: date).lastPathComponent, "events-20250927.jsonl")
    }

    // MARK: - D10 explicit outcomes and starting positions

    func testFinalizeRefusesAGameWithNoOutcome() async throws {
        try writeJournal(journal(tokens: Self.shortGame, finish: nil))
        let store = makeStore()
        do {
            _ = try await store.finalize(gameID: "g1", export: nil, exportUnavailableReason: "test")
            XCTFail("a game with no outcome was filed")
        } catch {
            XCTAssertEqual(error as? LichessBotRecordError, .noOutcome(gameID: "g1"))
        }
        let remaining = try await store.inProgressGameIDs()
        XCTAssertEqual(remaining, ["g1"])
    }

    func testExportInitialFenAndPGNSetUpTags() throws {
        let fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
        let exportText = try exportJSON(tokens: [], status: "resign", winner: "white", variant: "fromPosition", extraFields: #""initialFen":"\#(fen)","#)
        let record = try LichessBotRecordBuilder.build(
            gameID: "g1", journal: .init(elements: [Self.header(gameID: "g1", at: Date(timeIntervalSince1970: 1_759_000_001))], droppedTrailingByteCount: 0),
            export: try LichessBotGameExport.decode(Data(exportText.utf8)),
            exportUnavailableReason: nil, ourAccountID: Self.botID, checkedAt: Date()
        )
        XCTAssertEqual(record.setup.initialFen, fen)
        let pgn = LichessBotPGNWriter.pgn(for: record)
        XCTAssertTrue(pgn.contains(#"[SetUp "1"]"#), pgn)
        XCTAssertTrue(pgn.contains("[FEN \"\(fen)\"]"), pgn)
    }

    // MARK: - D13 index staleness

    func testIndexNoticesARecordRewrittenWithoutChangingCountOrNewestTime() async throws {
        let store = makeStore()
        let older = try await fileGame(store, gameID: "g1", createdAt: Self.gameCreatedAt)
        _ = try await fileGame(store, gameID: "g2", createdAt: Self.gameCreatedAt + 3_600_000)
        _ = try await store.loadIndex()
        let fm = FileManager.default
        let modified = try XCTUnwrap(try fm.attributesOfItem(atPath: older.recordURL.path)[.modificationDate] as? Date)
        let text = try String(contentsOf: older.recordURL, encoding: .utf8)
        try Data(text.replacingOccurrences(of: "\"Alice\"", with: "\"Alicia\"").utf8).write(to: older.recordURL)
        try fm.setAttributes([.modificationDate: modified.addingTimeInterval(-60)], ofItemAtPath: older.recordURL.path)
        let index = try await store.loadIndex()
        XCTAssertEqual(index.rows.first { $0.gameID == "g1" }?.opponentName, "Alicia")
    }
}
