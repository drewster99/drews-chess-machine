import XCTest
@testable import DrewsChessMachine

/// Perft ("performance test"): the number of leaf nodes reachable in exactly
/// `depth` plies of legal play from a position. Matching the published
/// reference counts is the gold-standard correctness check for move generation
/// — a single count exercises castling, en passant, promotion, pins, and
/// double check together, and any bug in any of them perturbs the total.
///
/// `perft(_:depth:gen:)` is deliberately parameterized by the move generator,
/// so the same reference counts validate the canonical `legalMoves` today and
/// any alternative implementation added later (make/unmake, pin-based). When an
/// alternative disagrees, `firstPerftDivergence` reports the FEN of the first
/// position whose move set differs — the exact reproducer.
final class PerftTests: XCTestCase {

    /// Count legal-move leaf nodes at `depth`, enumerating moves with `gen`.
    private func perft(_ state: GameState, depth: Int, gen: (GameState) -> [ChessMove]) -> Int {
        if depth == 0 { return 1 }
        let moves = gen(state)
        if depth == 1 { return moves.count }
        var nodes = 0
        for move in moves {
            nodes += perft(MoveGenerator.applyMove(move, to: state), depth: depth - 1, gen: gen)
        }
        return nodes
    }

    /// (name, FEN, depth, nodes) — standard positions from the Chess
    /// Programming Wiki perft results.
    private static let cases: [(name: String, fen: String, depth: Int, nodes: Int)] = [
        ("start-d4",      "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 4, 197_281),
        ("kiwipete-d3",   "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 3, 97_862),
        ("endgame-d4",    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 4, 43_238),
        ("promotions-d3", "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", 3, 9_467),
        ("position5-d3",  "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8", 3, 62_379),
        ("position6-d3",  "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10", 3, 89_890),
    ]

    /// The canonical `legalMoves` must reproduce the reference perft counts.
    func testLegalMovesMatchesReferencePerft() throws {
        for c in Self.cases {
            let state = try FENParser.parse(c.fen)
            let nodes = perft(state, depth: c.depth, gen: MoveGenerator.legalMoves)
            XCTAssertEqual(nodes, c.nodes, "perft(\(c.depth)) mismatch for \(c.name)")
        }
    }

    /// Depth-1 legal-move counts for each reference position.
    func testDepthOneMoveCounts() throws {
        let expected: [(fen: String, count: Int)] = [
            ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 20),
            ("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 48),
            ("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 14),
            ("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", 6),
        ]
        for e in expected {
            let state = try FENParser.parse(e.fen)
            XCTAssertEqual(MoveGenerator.legalMoves(for: state).count, e.count, "depth-1 count for \(e.fen)")
        }
    }

    // MARK: - Differential oracle (used once make/unmake (B) and pin-based (C) land)

    /// Walk the perft tree to `depth` and return the FEN-ish description of the
    /// first position where `candidate`'s move set differs from `reference`'s,
    /// or nil if they agree everywhere. Order-independent (compares as sets).
    func firstPerftDivergence(_ state: GameState,
                              depth: Int,
                              reference: (GameState) -> [ChessMove],
                              candidate: (GameState) -> [ChessMove]) -> String? {
        let ref = Set(reference(state))
        let cand = Set(candidate(state))
        if ref != cand {
            let missing = ref.subtracting(cand)
            let extra = cand.subtracting(ref)
            return "divergence: missing=\(missing) extra=\(extra)"
        }
        if depth <= 1 { return nil }
        for move in ref {
            if let d = firstPerftDivergence(MoveGenerator.applyMove(move, to: state),
                                            depth: depth - 1,
                                            reference: reference,
                                            candidate: candidate) {
                return d
            }
        }
        return nil
    }

    // MARK: - B (make/unmake) — must match the reference exactly

    /// make/unmake reproduces the reference perft counts.
    func testMakeUnmakeMatchesReferencePerft() throws {
        for c in Self.cases {
            let state = try FENParser.parse(c.fen)
            let nodes = perft(state, depth: c.depth, gen: MoveGenerator.legalMovesMakeUnmake)
            XCTAssertEqual(nodes, c.nodes, "make/unmake perft(\(c.depth)) mismatch for \(c.name)")
        }
    }

    /// make/unmake produces the identical move set as `legalMoves` at every node
    /// of every reference tree — the strong differential check.
    func testMakeUnmakeAgreesWithLegalMovesEverywhere() throws {
        for c in Self.cases {
            let state = try FENParser.parse(c.fen)
            let divergence = firstPerftDivergence(state, depth: c.depth,
                                                  reference: MoveGenerator.legalMoves,
                                                  candidate: MoveGenerator.legalMovesMakeUnmake)
            XCTAssertNil(divergence, "make/unmake diverges from legalMoves in \(c.name): \(divergence ?? "")")
        }
    }

    // MARK: - C (pin-based) — must match the reference exactly

    /// pin-based reproduces the reference perft counts.
    func testPinBasedMatchesReferencePerft() throws {
        for c in Self.cases {
            let state = try FENParser.parse(c.fen)
            let nodes = perft(state, depth: c.depth, gen: MoveGenerator.legalMovesPinBased)
            XCTAssertEqual(nodes, c.nodes, "pin-based perft(\(c.depth)) mismatch for \(c.name)")
        }
    }

    /// pin-based produces the identical move set as `legalMoves` at every node
    /// — the strong differential check (catches a missed pin / check / EP that
    /// would let the fast path admit an illegal move).
    func testPinBasedAgreesWithLegalMovesEverywhere() throws {
        for c in Self.cases {
            let state = try FENParser.parse(c.fen)
            let divergence = firstPerftDivergence(state, depth: c.depth,
                                                  reference: MoveGenerator.legalMoves,
                                                  candidate: MoveGenerator.legalMovesPinBased)
            XCTAssertNil(divergence, "pin-based diverges from legalMoves in \(c.name): \(divergence ?? "")")
        }
    }

    // MARK: - FEN encoder (used to log cross-check divergences reproducibly)

    /// `FENParser.fen(from:)` must produce a parseable, stable FEN — encode →
    /// parse → encode is a fixed point — so a logged divergence is reproducible.
    func testFENEncoderRoundTrips() throws {
        for c in Self.cases {
            let encoded = FENParser.fen(from: try FENParser.parse(c.fen))
            let reEncoded = FENParser.fen(from: try FENParser.parse(encoded))
            XCTAssertEqual(reEncoded, encoded, "FEN round-trip unstable for \(c.name)")
        }
    }
}
