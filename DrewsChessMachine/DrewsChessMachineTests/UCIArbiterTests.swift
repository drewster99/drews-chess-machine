//
//  UCIArbiterTests.swift
//  DrewsChessMachineTests
//
//  Covers the arbiter (GUI/controller) side of UCI — DCM driving an
//  external engine for train-vs-UCI game generation.
//
//  Two layers:
//    - Pure `UCIProtocol` grammar: position/go/setoption construction
//      and `bestmove` parsing (incl. (none)/0000/ponder/non-bestmove),
//      plus a LAN round-trip against `ChessMove` so the arbiter's move
//      tokens are guaranteed to resolve against the engine's own
//      legal-move list.
//    - `UCIArbiter` end-to-end against a *fake* engine subprocess (a
//      tiny autoflushing perl responder written to a temp file) so the
//      Process + line-reader + protocol state machine are exercised
//      without depending on a real Stockfish install: handshake, a
//      normal move, a null move, options delivery, a move timeout, an
//      engine that dies mid-move, and a launch failure.
//

import XCTest
@testable import DrewsChessMachine

final class UCIArbiterTests: XCTestCase {

    // MARK: - Pure protocol grammar

    func testPositionCommandFormatting() {
        XCTAssertEqual(
            UCIProtocol.positionCommand(startFEN: nil, moves: []),
            "position startpos"
        )
        XCTAssertEqual(
            UCIProtocol.positionCommand(startFEN: nil, moves: ["e2e4", "e7e5"]),
            "position startpos moves e2e4 e7e5"
        )
        let fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        XCTAssertEqual(
            UCIProtocol.positionCommand(startFEN: fen, moves: []),
            "position fen \(fen)"
        )
        XCTAssertEqual(
            UCIProtocol.positionCommand(startFEN: fen, moves: ["d2d4"]),
            "position fen \(fen) moves d2d4"
        )
    }

    func testGoAndSetOptionFormatting() {
        XCTAssertEqual(UCIProtocol.goCommand("nodes 1"), "go nodes 1")
        XCTAssertEqual(UCIProtocol.goCommand("  depth 6 "), "go depth 6")
        XCTAssertEqual(UCIProtocol.goCommand(""), "go")
        XCTAssertEqual(UCIProtocol.goCommand("   "), "go")
        XCTAssertEqual(
            UCIProtocol.setOptionCommand(name: "UCI_Elo", value: "1400"),
            "setoption name UCI_Elo value 1400"
        )
    }

    func testParseBestMoveVariants() {
        XCTAssertEqual(UCIProtocol.parseBestMove("bestmove e2e4"), .move("e2e4"))
        XCTAssertEqual(UCIProtocol.parseBestMove("bestmove e7e8q"), .move("e7e8q"))
        // A trailing ponder clause is ignored.
        XCTAssertEqual(UCIProtocol.parseBestMove("bestmove e2e4 ponder e7e5"), .move("e2e4"))
        // No-move sentinels.
        XCTAssertEqual(UCIProtocol.parseBestMove("bestmove (none)"), .null)
        XCTAssertEqual(UCIProtocol.parseBestMove("bestmove 0000"), .null)
        XCTAssertEqual(UCIProtocol.parseBestMove("bestmove"), .null)
        // Non-bestmove lines return nil so callers keep skipping.
        XCTAssertNil(UCIProtocol.parseBestMove("info depth 12 score cp 30 pv e2e4"))
        XCTAssertNil(UCIProtocol.parseBestMove("id name Stockfish"))
        XCTAssertNil(UCIProtocol.parseBestMove("uciok"))
        XCTAssertNil(UCIProtocol.parseBestMove(""))
    }

    /// The arbiter emits LAN tokens and consumes LAN tokens; both must
    /// interoperate with the engine's own `ChessMove` representation so
    /// a returned `bestmove` resolves against `MoveGenerator`'s legal
    /// list (the single source of truth for legality).
    func testBestMoveTokenResolvesAgainstLegalMoves() {
        let legal = MoveGenerator.legalMoves(for: .starting)
        guard let best = UCIProtocol.parseBestMove("bestmove e2e4"),
              case let .move(token) = best else {
            return XCTFail("expected a parsed move token")
        }
        let resolved = ChessMove.parseUCI(token, legal: legal)
        XCTAssertNotNil(resolved, "e2e4 should be legal from the start position")
        XCTAssertEqual(resolved?.uci, "e2e4")
    }

    // MARK: - Integration against a fake engine subprocess

    func testHandshakeAndBestMove() async throws {
        let engine = try writeFakeEngine(Self.normalEngine)
        let arbiter = UCIArbiter(configuration: .init(
            command: engine, goLimit: "nodes 1", label: "fake#normal"
        ))
        try await arbiter.launch()
        try await arbiter.handshake()
        try await arbiter.startNewGame()
        let best = try await arbiter.bestMove(startFEN: nil, moves: [])
        await arbiter.shutdown()
        // The fake appends a `ponder` clause; parsing must strip it.
        XCTAssertEqual(best, .move("e2e4"))
    }

    func testNullBestMove() async throws {
        let engine = try writeFakeEngine(Self.nullEngine)
        let arbiter = UCIArbiter(configuration: .init(
            command: engine, goLimit: "depth 1", label: "fake#null"
        ))
        try await arbiter.launch()
        try await arbiter.handshake()
        let best = try await arbiter.bestMove(startFEN: nil, moves: [])
        await arbiter.shutdown()
        XCTAssertEqual(best, .null)
    }

    /// Options must be sent after `uciok` and before the readiness
    /// barrier. The fake records every `setoption` line it receives to
    /// a sidecar file (passed as an argument); since the engine is
    /// single-threaded and prints `readyok` only after processing the
    /// preceding lines, the file is complete by the time `handshake()`
    /// returns.
    func testHandshakeSendsConfiguredOptions() async throws {
        let dir = try makeTempDir()
        let logURL = dir.appendingPathComponent("setoptions.log")
        let engine = try writeFakeEngine(Self.optionsEngine, in: dir)
        let arbiter = UCIArbiter(configuration: .init(
            command: engine,
            arguments: [logURL.path],
            options: [
                .init(name: "UCI_LimitStrength", value: "true"),
                .init(name: "UCI_Elo", value: "1400"),
            ],
            goLimit: "nodes 1",
            label: "fake#options"
        ))
        try await arbiter.launch()
        try await arbiter.handshake()
        await arbiter.shutdown()

        let recorded: String
        do {
            recorded = try String(contentsOf: logURL, encoding: .utf8)
        } catch {
            // Empty default gives a clearer assertion failure below than a
            // thrown file-not-found if no setoption was ever recorded.
            recorded = ""
        }
        XCTAssertTrue(
            recorded.contains("setoption name UCI_LimitStrength value true"),
            "missing UCI_LimitStrength; got:\n\(recorded)"
        )
        XCTAssertTrue(
            recorded.contains("setoption name UCI_Elo value 1400"),
            "missing UCI_Elo; got:\n\(recorded)"
        )
    }

    func testMoveTimeout() async throws {
        let engine = try writeFakeEngine(Self.silentGoEngine)
        let arbiter = UCIArbiter(configuration: .init(
            command: engine,
            goLimit: "nodes 1",
            label: "fake#silent",
            moveTimeout: .milliseconds(400)
        ))
        try await arbiter.launch()
        try await arbiter.handshake()
        do {
            _ = try await arbiter.bestMove(startFEN: nil, moves: [])
            XCTFail("expected a timeout from a silent engine")
        } catch let error as UCIArbiterError {
            XCTAssertEqual(error, .timeout)
        }
        await arbiter.shutdown()
    }

    func testEngineDeathDuringMove() async throws {
        let engine = try writeFakeEngine(Self.dyingEngine)
        let arbiter = UCIArbiter(configuration: .init(
            command: engine,
            goLimit: "nodes 1",
            label: "fake#dying",
            moveTimeout: .seconds(5)
        ))
        try await arbiter.launch()
        try await arbiter.handshake()
        do {
            _ = try await arbiter.bestMove(startFEN: nil, moves: [])
            XCTFail("expected engineTerminated when the engine exits mid-move")
        } catch let error as UCIArbiterError {
            XCTAssertEqual(error, .engineTerminated)
        }
        await arbiter.shutdown()
    }

    func testLaunchFailureOnMissingExecutable() async throws {
        let arbiter = UCIArbiter(configuration: .init(
            command: URL(fileURLWithPath: "/nonexistent/definitely-not-an-engine"),
            goLimit: "nodes 1",
            label: "fake#missing"
        ))
        do {
            try await arbiter.launch()
            XCTFail("expected launchFailed for a missing executable")
        } catch let error as UCIArbiterError {
            guard case .launchFailed = error else {
                return XCTFail("expected .launchFailed, got \(error)")
            }
        }
    }

    // MARK: - Fake engine fixtures

    /// Handshakes, then answers every `go` with `e2e4` plus a `ponder`
    /// clause (to prove ponder-stripping) after an `info` line.
    private static let normalEngine = """
    #!/usr/bin/perl
    $| = 1;
    while (my $line = <STDIN>) {
        chomp $line;
        if    ($line eq "uci")         { print "id name Fake\\nid author test\\nuciok\\n"; }
        elsif ($line =~ /^isready/)    { print "readyok\\n"; }
        elsif ($line =~ /^ucinewgame/) { }
        elsif ($line =~ /^position/)   { }
        elsif ($line =~ /^go/)         { print "info depth 1 score cp 12\\nbestmove e2e4 ponder e7e5\\n"; }
        elsif ($line =~ /^quit/)       { exit 0; }
    }
    """

    /// Answers every `go` with a null move.
    private static let nullEngine = """
    #!/usr/bin/perl
    $| = 1;
    while (my $line = <STDIN>) {
        chomp $line;
        if    ($line eq "uci")      { print "id name Fake\\nuciok\\n"; }
        elsif ($line =~ /^isready/) { print "readyok\\n"; }
        elsif ($line =~ /^go/)      { print "bestmove (none)\\n"; }
        elsif ($line =~ /^quit/)    { exit 0; }
    }
    """

    /// Records each `setoption` line to the file named in ARGV[0].
    private static let optionsEngine = """
    #!/usr/bin/perl
    $| = 1;
    my $log = $ARGV[0];
    while (my $line = <STDIN>) {
        chomp $line;
        if    ($line eq "uci")         { print "id name Fake\\nuciok\\n"; }
        elsif ($line =~ /^setoption/)  { open(my $fh, '>>', $log); print $fh "$line\\n"; close($fh); }
        elsif ($line =~ /^isready/)    { print "readyok\\n"; }
        elsif ($line =~ /^go/)         { print "bestmove e2e4\\n"; }
        elsif ($line =~ /^quit/)       { exit 0; }
    }
    """

    /// Handshakes normally but never answers `go` — used to exercise
    /// the move timeout.
    private static let silentGoEngine = """
    #!/usr/bin/perl
    $| = 1;
    while (my $line = <STDIN>) {
        chomp $line;
        if    ($line eq "uci")      { print "id name Fake\\nuciok\\n"; }
        elsif ($line =~ /^isready/) { print "readyok\\n"; }
        elsif ($line =~ /^quit/)    { exit 0; }
    }
    """

    /// Handshakes normally, then exits on the first `go` (closing its
    /// stdout) — used to exercise engineTerminated.
    private static let dyingEngine = """
    #!/usr/bin/perl
    $| = 1;
    while (my $line = <STDIN>) {
        chomp $line;
        if    ($line eq "uci")      { print "id name Fake\\nuciok\\n"; }
        elsif ($line =~ /^isready/) { print "readyok\\n"; }
        elsif ($line =~ /^go/)      { exit 1; }
        elsif ($line =~ /^quit/)    { exit 0; }
    }
    """

    // MARK: - Fixture helpers

    private func makeTempDir() throws -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("uciarbiter-tests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        addTeardownBlock {
            try? FileManager.default.removeItem(at: dir)
        }
        return dir
    }

    private func writeFakeEngine(_ body: String, in dir: URL? = nil) throws -> URL {
        let directory = try dir ?? makeTempDir()
        let url = directory.appendingPathComponent("fake-engine.pl")
        try body.write(to: url, atomically: true, encoding: .utf8)
        try FileManager.default.setAttributes(
            [.posixPermissions: 0o755],
            ofItemAtPath: url.path
        )
        return url
    }
}
