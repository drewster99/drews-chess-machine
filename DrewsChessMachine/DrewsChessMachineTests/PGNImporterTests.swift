import XCTest
@testable import DrewsChessMachine

final class PGNImporterTests: XCTestCase {

    func testSANTokenizationStripsNumbersCommentsVariationsResult() {
        let movetext = "1. e4 e5 2. Nf3 {a comment} Nc6 (2... d6 3. d4) 3. Bb5 $1 a6 1-0"
        let tokens = PGNImporter.sanTokens(from: movetext)
        XCTAssertEqual(tokens, ["e4", "e5", "Nf3", "Nc6", "Bb5", "a6"])
    }

    func testMatchPawnAndKnightFromStart() throws {
        let engine = ChessGameEngine()
        let legal = MoveGenerator.legalMoves(for: engine.state)

        // e4: e2 (row 6,col 4) -> e4 (row 4,col 4) in the row-0-is-rank-8 frame.
        let e4 = try XCTUnwrap(PGNImporter.matchSAN("e4", state: engine.state, legal: legal))
        XCTAssertEqual([e4.fromRow, e4.fromCol, e4.toRow, e4.toCol], [6, 4, 4, 4])
        XCTAssertNil(e4.promotion)

        // Nf3: g1 (row 7,col 6) -> f3 (row 5,col 5).
        let nf3 = try XCTUnwrap(PGNImporter.matchSAN("Nf3", state: engine.state, legal: legal))
        XCTAssertEqual([nf3.fromRow, nf3.fromCol, nf3.toRow, nf3.toCol], [7, 6, 5, 5])
    }

    func testReplayKnownOpening(   ) throws {
        // Ruy Lopez incl. a capture-free line + kingside castling.
        let sans = ["e4", "e5", "Nf3", "Nc6", "Bb5", "a6", "Ba4", "Nf6", "O-O", "Be7"]
        let engine = ChessGameEngine()
        var moves: [ChessMove] = []
        for san in sans {
            let legal = MoveGenerator.legalMoves(for: engine.state)
            let move = try XCTUnwrap(PGNImporter.matchSAN(san, state: engine.state, legal: legal),
                                     "failed to match SAN \(san)")
            moves.append(move)
            do {
                try engine.applyMoveAndAdvance(move)
            } catch {
                XCTFail("engine rejected matched move for \(san): \(error)")
                return
            }
        }
        XCTAssertEqual(moves.count, sans.count)
        // The castling move (O-O) is the 9th ply: white king e1 (row7,col4) -> g1 (row7,col6).
        let castle = moves[8]
        XCTAssertEqual([castle.fromRow, castle.fromCol, castle.toRow, castle.toCol], [7, 4, 7, 6])
    }

    func testCaptureSANMatches() throws {
        // 1. e4 d5 2. exd5 — the capture should resolve to the e4 pawn taking d5.
        let engine = ChessGameEngine()
        for san in ["e4", "d5"] {
            let legal = MoveGenerator.legalMoves(for: engine.state)
            let m = try XCTUnwrap(PGNImporter.matchSAN(san, state: engine.state, legal: legal))
            try engine.applyMoveAndAdvance(m)
        }
        let legal = MoveGenerator.legalMoves(for: engine.state)
        let capture = try XCTUnwrap(PGNImporter.matchSAN("exd5", state: engine.state, legal: legal))
        // d5 = row 3, col 3; capturing pawn came from the e-file (col 4).
        XCTAssertEqual([capture.toRow, capture.toCol, capture.fromCol], [3, 3, 4])
    }

    func testTimeControlClassification() {
        XCTAssertEqual(PGNImporter.timeControlClass("60+0"), "bullet")
        XCTAssertEqual(PGNImporter.timeControlClass("180+0"), "blitz")
        XCTAssertEqual(PGNImporter.timeControlClass("600+0"), "rapid")
        XCTAssertEqual(PGNImporter.timeControlClass("1800+0"), "classical")
        XCTAssertEqual(PGNImporter.timeControlClass("300+3"), "blitz") // 300 + 40*3 = 420
    }

    // MARK: - Parallel import pipeline

    private func writeTempPGN(_ text: String) throws -> (pgn: URL, dir: URL) {
        let dir = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("pgnimp-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let pgn = dir.appendingPathComponent("games.pgn")
        try text.write(to: pgn, atomically: true, encoding: .utf8)
        return (pgn, dir)
    }

    private func importGames(pgn: URL,
                             into parent: URL,
                             threads: Int,
                             failOnError: Bool = true,
                             maxGames: Int? = nil) throws -> (summary: PGNImporter.Summary, games: [GameRecord]) {
        try FileManager.default.createDirectory(at: parent, withIntermediateDirectories: true)
        var cfg = PGNImportConfig(inputPath: pgn.path, corpusName: "t", minRating: nil,
                                  maxGames: maxGames, minPlies: 1, timeControlClasses: nil)
        cfg.importThreads = threads
        cfg.failOnError = failOnError
        cfg.outputParentDirectory = parent
        let summary = try PGNImporter.runImport(config: cfg)
        let corpusDir = try XCTUnwrap(
            try FileManager.default.contentsOfDirectory(at: parent, includingPropertiesForKeys: nil).first)
        let corpus = try GameCorpus.open(directory: corpusDir)
        return (summary, try corpus.allGames())
    }

    /// The parallel importer must preserve original file order and produce
    /// identical output regardless of worker count.
    func testParallelImportPreservesOrderAndIsDeterministic() throws {
        let line = ["e4", "e5", "Nf3", "Nc6", "Bb5", "a6", "Ba4", "Nf6", "O-O", "Be7"]
        var text = ""
        var expectedLengths: [Int] = []
        for k in 0..<80 {
            let len = 2 + (k % (line.count - 1))   // cycles 2...10 plies
            expectedLengths.append(len)
            let moves = line.prefix(len).joined(separator: " ")
            text += "[Event \"t\"]\n[Result \"1-0\"]\n\n\(moves) 1-0\n\n"
        }
        let (pgn, dir) = try writeTempPGN(text)
        defer {
            do { try FileManager.default.removeItem(at: dir) }
            catch { /* best-effort temp cleanup */ }
        }

        let serial = try importGames(pgn: pgn, into: dir.appendingPathComponent("s"), threads: 1)
        let parallel = try importGames(pgn: pgn, into: dir.appendingPathComponent("p"), threads: 8)

        XCTAssertEqual(serial.summary.imported, 80)
        XCTAssertEqual(parallel.summary.imported, 80)
        XCTAssertEqual(serial.games.map(\.moves.count), expectedLengths)   // order preserved
        XCTAssertEqual(parallel.games.map(\.moves.count), expectedLengths)
        XCTAssertEqual(serial.games, parallel.games)                       // thread-count independent
    }

    func testHardFailsOnUnparseableGameByDefault() throws {
        let text = "[Event \"t\"]\n[Result \"1-0\"]\n\ne4 e5 Zz9 1-0\n\n"
        let (pgn, dir) = try writeTempPGN(text)
        defer {
            do { try FileManager.default.removeItem(at: dir) }
            catch { /* best-effort temp cleanup */ }
        }
        XCTAssertThrowsError(try importGames(pgn: pgn, into: dir.appendingPathComponent("hf"), threads: 4)) { error in
            guard case PGNImportError.gameFailed = error else {
                return XCTFail("expected PGNImportError.gameFailed, got \(error)")
            }
        }
    }

    func testLenientCountsParseFailuresInsteadOfFailing() throws {
        let text = "[Event \"t\"]\n[Result \"1-0\"]\n\ne4 e5 Nf3 1-0\n\n"
                 + "[Event \"t\"]\n[Result \"1-0\"]\n\ne4 Zz9 1-0\n\n"
        let (pgn, dir) = try writeTempPGN(text)
        defer {
            do { try FileManager.default.removeItem(at: dir) }
            catch { /* best-effort temp cleanup */ }
        }
        let r = try importGames(pgn: pgn, into: dir.appendingPathComponent("len"),
                                threads: 4, failOnError: false)
        XCTAssertEqual(r.summary.imported, 1)
        XCTAssertEqual(r.summary.parseErrors, 1)
    }

    func testMaxGamesCapIsExactAndOrdered() throws {
        let line = ["e4", "e5", "Nf3", "Nc6", "Bb5", "a6"]
        var text = ""
        for k in 0..<50 {
            let len = 2 + (k % (line.count - 1))
            let moves = line.prefix(len).joined(separator: " ")
            text += "[Event \"t\"]\n[Result \"1-0\"]\n\n\(moves) 1-0\n\n"
        }
        let (pgn, dir) = try writeTempPGN(text)
        defer {
            do { try FileManager.default.removeItem(at: dir) }
            catch { /* best-effort temp cleanup */ }
        }
        let r = try importGames(pgn: pgn, into: dir.appendingPathComponent("cap"), threads: 8, maxGames: 10)
        XCTAssertEqual(r.summary.imported, 10)
        XCTAssertEqual(r.games.count, 10)
    }

    /// A game that legally plays on past a threefold repetition (claimable, not
    /// auto-terminating) must import. Regression for the import hard-failing
    /// with "illegal move … : The game has already ended" because the engine's
    /// self-play auto-draw vetoed a legal continuation.
    func testImportsGamePlayingThroughThreefoldRepetition() throws {
        // Knights out-and-back returns the start position three times as an
        // applied state, then play continues — a never-claimed threefold.
        let moves = "Nf3 Nf6 Ng1 Ng8 Nf3 Nf6 Ng1 Ng8 Nf3 Nf6 Ng1 Ng8 Nf3 Nf6"
        let text = "[Event \"t\"]\n[Result \"1/2-1/2\"]\n\n\(moves) 1/2-1/2\n\n"
        let (pgn, dir) = try writeTempPGN(text)
        defer {
            do { try FileManager.default.removeItem(at: dir) }
            catch { /* best-effort temp cleanup */ }
        }
        let r = try importGames(pgn: pgn, into: dir.appendingPathComponent("rep"), threads: 1)
        XCTAssertEqual(r.summary.imported, 1)
        XCTAssertEqual(r.games.first?.moves.count, 14)
    }

    // MARK: - SAN resolution via pseudo-legal + per-candidate legality (A)

    /// `resolveLegalSANMove` must pick the LEGAL match when a pinned piece also
    /// pseudo-matches the (undisambiguated) SAN token. Guards the
    /// pseudo-legal-then-verify resolution against returning an illegal move.
    func testResolveSANPicksLegalMoveOverPinnedPseudoMatch() throws {
        // White Ke1, Nc3 (pinned by Ba5 on the a5–e1 diagonal), Ng1; black Ke8.
        // For "Ne2", only Ng1–e2 is legal — Nc3–e2 would expose the king.
        let state = try FENParser.parse("4k3/8/8/b7/8/2N5/8/4K1N1 w - - 0 1")
        let move = try XCTUnwrap(PGNImporter.resolveLegalSANMove("Ne2", state: state))
        // g1 = (row 7, col 6) ; e2 = (row 6, col 4)
        XCTAssertEqual([move.fromRow, move.fromCol, move.toRow, move.toCol], [7, 6, 6, 4])
    }

    /// The pinned knight's Nc3–e2 is pseudo-legal but not legal — confirming the
    /// legality filter (not move generation) is what disambiguates above, and
    /// that the pseudoLegalMoves/legalMoves split holds.
    func testPinnedMoveIsPseudoLegalButNotLegal() throws {
        let state = try FENParser.parse("4k3/8/8/b7/8/2N5/8/4K1N1 w - - 0 1")
        func hasNc3e2(_ moves: [ChessMove]) -> Bool {
            moves.contains { $0.fromRow == 5 && $0.fromCol == 2 && $0.toRow == 6 && $0.toCol == 4 }
        }
        XCTAssertTrue(hasNc3e2(MoveGenerator.pseudoLegalMoves(for: state)), "Nc3–e2 should be pseudo-legal")
        XCTAssertFalse(hasNc3e2(MoveGenerator.legalMoves(for: state)), "Nc3–e2 should be illegal (pinned)")
    }

    /// A missing/unreadable input file must hard-fail (the reader subprocess
    /// exits nonzero), not silently produce an empty corpus.
    func testImportFailsOnMissingInputFile() throws {
        let dir = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("pgnimp-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer {
            do { try FileManager.default.removeItem(at: dir) }
            catch { /* best-effort temp cleanup */ }
        }
        let missing = dir.appendingPathComponent("does-not-exist.pgn")
        XCTAssertThrowsError(try importGames(pgn: missing, into: dir.appendingPathComponent("out"), threads: 2))
    }

    /// An undisambiguated SAN with more than one legal match must resolve to nil
    /// (the importer then fails loudly); the disambiguated form resolves uniquely.
    func testAmbiguousSANResolvesToNil() throws {
        // Both Nb1 and Nf3 can legally reach d2.
        let state = try FENParser.parse("4k3/8/8/8/8/5N2/8/1N2K3 w - - 0 1")
        XCTAssertNil(PGNImporter.resolveLegalSANMove("Nd2", state: state))
        let nbd2 = try XCTUnwrap(PGNImporter.resolveLegalSANMove("Nbd2", state: state))
        XCTAssertEqual([nbd2.fromRow, nbd2.fromCol], [7, 1])   // b1
    }
}
